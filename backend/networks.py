"""
Neural Network Definitions for Neural Razz Trainer

Three network heads sharing similar architecture:
  Mode 1 (Strategy): 32 → 128 → 128 → 64 → 5 (softmax)
  Mode 2 (Regret):   32 → 256 → 256 → 128 → 5 (linear)
  Mode 3 (Value):    32 → 256 → 256 → 128 → 1 (linear)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from typing import List, Dict, Any

from features import FEATURE_DIM


class StrategyNetwork(nn.Module):
    """Mode 1: Predicts action probabilities directly.

    Input: 32-dim game features
    Output: 5 action probabilities (fold, check, call, bet, raise) via softmax
    """
    NUM_ACTIONS = 5

    def __init__(self, input_dim: int = FEATURE_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, self.NUM_ACTIONS)

        self._init_weights()

    def _init_weights(self):
        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=-1)
        return x

    def predict(self, features: List[float]) -> List[float]:
        """Single inference. Returns 5 action probabilities."""
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            probs = self.forward(x).squeeze(0).tolist()
        return probs


class RegretNetwork(nn.Module):
    """Mode 2 (Deep CFR): Predicts regret estimates per action.

    Input: 32-dim game features
    Output: 5 regret values (linear, can be negative)
    """
    NUM_ACTIONS = 5

    def __init__(self, input_dim: int = FEATURE_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, self.NUM_ACTIONS)

        self._init_weights()

    def _init_weights(self):
        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # Linear output — regrets can be negative
        return x

    def get_strategy(self, features: List[float]) -> List[float]:
        """Get strategy via regret matching on predicted regrets."""
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            regrets = self.forward(x).squeeze(0).tolist()

        # Regret matching: positive regrets normalized
        positive = [max(0, r) for r in regrets]
        total = sum(positive)
        if total > 0:
            return [p / total for p in positive]
        else:
            return [1.0 / len(regrets)] * len(regrets)


class ValueNetwork(nn.Module):
    """Mode 3: Predicts expected value of a game state.

    Input: 32-dim game features
    Output: 1 scalar (expected payoff for hero)
    """
    def __init__(self, input_dim: int = FEATURE_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

        self._init_weights()

    def _init_weights(self):
        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def predict(self, features: List[float]) -> float:
        """Single inference. Returns expected value."""
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            val = self.forward(x).item()
        return val


# ─── Model Export ───────────────────────────────────────────────────────────

def export_to_json(model: nn.Module, path: str, metadata: Dict[str, Any] = None):
    """Export a PyTorch model to JSON format compatible with Swift ValueNetwork.

    Format matches ValueNetwork.swift's NetworkData Codable structure.
    """
    layer_configs = []

    # Get all linear layers in order
    linear_layers = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Linear)]

    for i, (name, layer) in enumerate(linear_layers):
        is_last = (i == len(linear_layers) - 1)
        config = {
            'inputSize': layer.in_features,
            'outputSize': layer.out_features,
            'useReLU': not is_last,  # ReLU on all except output
            'weights': layer.weight.detach().cpu().tolist(),
            'biases': layer.bias.detach().cpu().tolist(),
        }
        layer_configs.append(config)

    data = {
        'inputSize': linear_layers[0][1].in_features if linear_layers else FEATURE_DIM,
        'trainStep': metadata.get('train_steps', 0) if metadata else 0,
        'learningRate': metadata.get('learning_rate', 0.001) if metadata else 0.001,
        'featureVersion': 4,  # v4 = 32-dim Neural Razz Trainer features
        'layerConfigs': layer_configs,
        'metadata': metadata or {},
    }

    with open(path, 'w') as f:
        json.dump(data, f)

    size_mb = os.path.getsize(path) / 1_000_000
    print(f"[Export] Saved to {path} ({size_mb:.1f} MB)")
    return path


import os  # needed for export_to_json


# ─── Quick validation ───────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== Network Tests ===\n")

    # Test Strategy Network
    snet = StrategyNetwork()
    dummy = [0.0] * FEATURE_DIM
    probs = snet.predict(dummy)
    print(f"StrategyNet output: {[f'{p:.3f}' for p in probs]}")
    assert len(probs) == 5
    assert abs(sum(probs) - 1.0) < 0.001, f"Probabilities should sum to 1, got {sum(probs)}"
    print("✅ StrategyNetwork: 5 probs summing to 1.0")

    # Test Regret Network
    rnet = RegretNetwork()
    strategy = rnet.get_strategy(dummy)
    print(f"RegretNet strategy: {[f'{s:.3f}' for s in strategy]}")
    assert len(strategy) == 5
    assert abs(sum(strategy) - 1.0) < 0.001
    print("✅ RegretNetwork: regret matching produces valid strategy")

    # Test Value Network
    vnet = ValueNetwork()
    val = vnet.predict(dummy)
    print(f"ValueNet prediction: {val:.4f}")
    assert isinstance(val, float)
    print("✅ ValueNetwork: single float output")

    # Test parameter counts
    for name, net in [("Strategy", snet), ("Regret", rnet), ("Value", vnet)]:
        params = sum(p.numel() for p in net.parameters())
        print(f"  {name}: {params:,} parameters")

    # Test export
    test_path = '/tmp/test_neural_razz_export.json'
    export_to_json(snet, test_path, {'train_steps': 100, 'mode': 'strategy'})
    with open(test_path) as f:
        data = json.load(f)
    assert 'layerConfigs' in data
    assert len(data['layerConfigs']) == 4  # 4 linear layers
    print(f"✅ Export: {len(data['layerConfigs'])} layers exported")

    print("\nAll network tests passed!")
