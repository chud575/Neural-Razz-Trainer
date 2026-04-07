#!/usr/bin/env python3
"""
Quick model tester — run from Neural Razz Trainer root directory.

Usage:
    python3 test_model.py                    # Test from checkpoint
    python3 test_model.py model.json         # Test specific exported model
    python3 test_model.py model.json 25000   # Test with 25K hands
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from arena_test import run_arena
from opponents import preload_all

preload_all()


def load_model(path=None):
    """Load model from checkpoint or exported JSON."""
    if path and os.path.exists(path):
        # Load from exported JSON
        import torch
        import json
        from networks import StrategyNetwork

        with open(path) as f:
            data = json.load(f)

        network = StrategyNetwork()
        layers = data.get('layerConfigs', [])
        sd = network.state_dict()
        for i, lc in enumerate(layers):
            sd[f'fc{i+1}.weight'] = torch.tensor(lc['weights'], dtype=torch.float32)
            sd[f'fc{i+1}.bias'] = torch.tensor(lc['biases'], dtype=torch.float32)
        network.load_state_dict(sd)
        network.eval()
        iters = data.get('metadata', {}).get('iterations', '?')
        print(f'Loaded from {os.path.basename(path)} ({iters} iterations)')
        return network

    # Load from checkpoint
    from checkpoint import load_checkpoint
    cp = load_checkpoint()
    if cp:
        print(f'Loaded from checkpoint (iteration {cp.get("base_iteration", "?")})')
        return cp['strategy_net']

    print('No model found')
    return None


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    num_hands = int(sys.argv[2]) if len(sys.argv) > 2 else 10000

    network = load_model(model_path)
    if not network:
        sys.exit(1)

    print(f'\n=== Python Arena Test ({num_hands:,} hands each) ===\n')

    opponents = ['pure_ev', 'tag', 'lag', 'calling_station', 'random']

    for opp in opponents:
        r = run_arena(network, num_hands=num_hands, opponent_type=opp)
        print(f'vs {opp:20s}  Win: {r["win_rate"]:5.1f}%  BB/100: {r["bb_per_100"]:+7.1f}  '
              f'Fold: {r["fold_rate"]:5.1f}%  SD Win: {r["sd_win_rate"]:5.1f}%')

    print()


if __name__ == '__main__':
    main()
