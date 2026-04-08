"""
Checkpoint Manager — Save/Load training state to disk.

Saves networks (PyTorch state dicts), reservoirs (pickled), and metadata
so training can resume after a server restart.
"""

import os
import json
import pickle
import time
import torch
from typing import Optional

from networks import StrategyNetwork, RegretNetwork, ValueNetwork
from reservoir import ReservoirBuffer

# Default checkpoint directories
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')
VALUE_CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints_value')


def save_checkpoint(state: dict, checkpoint_dir: str = CHECKPOINT_DIR) -> str:
    """Save Deep CFR training state to disk.

    Args:
        state: dict with keys:
            'strategy_net': StrategyNetwork
            'advantage_net': RegretNetwork
            'strategy_reservoir': ReservoirBuffer
            'advantage_reservoir': ReservoirBuffer
            'base_iteration': int

    Returns: path to checkpoint directory
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save network state dicts
    torch.save(state['strategy_net'].state_dict(),
               os.path.join(checkpoint_dir, 'strategy_net.pt'))
    torch.save(state['advantage_net'].state_dict(),
               os.path.join(checkpoint_dir, 'advantage_net.pt'))

    # Save reservoirs (pickled)
    with open(os.path.join(checkpoint_dir, 'strategy_reservoir.pkl'), 'wb') as f:
        pickle.dump(state['strategy_reservoir'], f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(checkpoint_dir, 'advantage_reservoir.pkl'), 'wb') as f:
        pickle.dump(state['advantage_reservoir'], f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save metadata
    metadata = {
        'base_iteration': state.get('base_iteration', 0),
        'timestamp': time.time(),
        'strategy_reservoir_size': len(state['strategy_reservoir']),
        'advantage_reservoir_size': len(state['advantage_reservoir']),
    }
    with open(os.path.join(checkpoint_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    total_size = sum(
        os.path.getsize(os.path.join(checkpoint_dir, f))
        for f in os.listdir(checkpoint_dir)
        if os.path.isfile(os.path.join(checkpoint_dir, f))
    )
    print(f"[Checkpoint] Saved to {checkpoint_dir} ({total_size / 1_000_000:.1f} MB)")
    print(f"  iteration: {metadata['base_iteration']}, "
          f"adv_reservoir: {metadata['advantage_reservoir_size']:,}, "
          f"strat_reservoir: {metadata['strategy_reservoir_size']:,}")

    return checkpoint_dir


def load_checkpoint(checkpoint_dir: str = CHECKPOINT_DIR) -> Optional[dict]:
    """Load Deep CFR training state from disk.

    Returns dict matching train_deep_cfr's resume_state format, or None if not found.
    """
    meta_path = os.path.join(checkpoint_dir, 'metadata.json')
    if not os.path.exists(meta_path):
        print(f"[Checkpoint] No checkpoint found at {checkpoint_dir}")
        return None

    try:
        # Load metadata
        with open(meta_path) as f:
            metadata = json.load(f)

        # Load networks
        strategy_net = StrategyNetwork()
        strategy_net.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, 'strategy_net.pt'), weights_only=True))
        strategy_net.eval()

        advantage_net = RegretNetwork()
        advantage_net.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, 'advantage_net.pt'), weights_only=True))
        advantage_net.eval()

        # Load reservoirs
        with open(os.path.join(checkpoint_dir, 'strategy_reservoir.pkl'), 'rb') as f:
            strategy_reservoir = pickle.load(f)
        with open(os.path.join(checkpoint_dir, 'advantage_reservoir.pkl'), 'rb') as f:
            advantage_reservoir = pickle.load(f)

        print(f"[Checkpoint] Loaded from {checkpoint_dir}")
        print(f"  iteration: {metadata['base_iteration']}, "
              f"adv_reservoir: {metadata.get('advantage_reservoir_size', '?')}, "
              f"strat_reservoir: {metadata.get('strategy_reservoir_size', '?')}")

        return {
            'strategy_net': strategy_net,
            'advantage_net': advantage_net,
            'strategy_reservoir': strategy_reservoir,
            'advantage_reservoir': advantage_reservoir,
            'base_iteration': metadata['base_iteration'],
        }

    except Exception as e:
        print(f"[Checkpoint] Failed to load: {e}")
        return None


def has_checkpoint(checkpoint_dir: str = CHECKPOINT_DIR) -> bool:
    """Check if a valid checkpoint exists."""
    return os.path.exists(os.path.join(checkpoint_dir, 'metadata.json'))


def delete_checkpoint(checkpoint_dir: str = CHECKPOINT_DIR):
    """Delete the checkpoint directory."""
    import shutil
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        print(f"[Checkpoint] Deleted {checkpoint_dir}")


# ─── Value Mode Checkpoints ──────────────────────────────────────────────


def save_value_checkpoint(state: dict, checkpoint_dir: str = VALUE_CHECKPOINT_DIR) -> str:
    """Save Value training state to disk.

    Args:
        state: dict with keys:
            'value_net': ValueNetwork
            'value_reservoir': ReservoirBuffer
            'base_iteration': int

    Returns: path to checkpoint directory
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.save(state['value_net'].state_dict(),
               os.path.join(checkpoint_dir, 'value_net.pt'))

    with open(os.path.join(checkpoint_dir, 'value_reservoir.pkl'), 'wb') as f:
        pickle.dump(state['value_reservoir'], f, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = {
        'base_iteration': state.get('base_iteration', 0),
        'timestamp': time.time(),
        'value_reservoir_size': len(state['value_reservoir']),
    }
    with open(os.path.join(checkpoint_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    total_size = sum(
        os.path.getsize(os.path.join(checkpoint_dir, f))
        for f in os.listdir(checkpoint_dir)
        if os.path.isfile(os.path.join(checkpoint_dir, f))
    )
    print(f"[Value Checkpoint] Saved to {checkpoint_dir} ({total_size / 1_000_000:.1f} MB)")
    print(f"  iteration: {metadata['base_iteration']}, "
          f"reservoir: {metadata['value_reservoir_size']:,}")

    return checkpoint_dir


def load_value_checkpoint(checkpoint_dir: str = VALUE_CHECKPOINT_DIR) -> Optional[dict]:
    """Load Value training state from disk.

    Returns dict matching train_value's resume_state format, or None if not found.
    """
    meta_path = os.path.join(checkpoint_dir, 'metadata.json')
    if not os.path.exists(meta_path):
        print(f"[Value Checkpoint] No checkpoint found at {checkpoint_dir}")
        return None

    try:
        with open(meta_path) as f:
            metadata = json.load(f)

        value_net = ValueNetwork()
        value_net.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, 'value_net.pt'), weights_only=True))
        value_net.eval()

        with open(os.path.join(checkpoint_dir, 'value_reservoir.pkl'), 'rb') as f:
            value_reservoir = pickle.load(f)

        print(f"[Value Checkpoint] Loaded from {checkpoint_dir}")
        print(f"  iteration: {metadata['base_iteration']}, "
              f"reservoir: {metadata.get('value_reservoir_size', '?')}")

        return {
            'value_net': value_net,
            'value_reservoir': value_reservoir,
            'base_iteration': metadata['base_iteration'],
        }

    except Exception as e:
        print(f"[Value Checkpoint] Failed to load: {e}")
        return None


def has_value_checkpoint(checkpoint_dir: str = VALUE_CHECKPOINT_DIR) -> bool:
    """Check if a valid value checkpoint exists."""
    return os.path.exists(os.path.join(checkpoint_dir, 'metadata.json'))


def delete_value_checkpoint(checkpoint_dir: str = VALUE_CHECKPOINT_DIR):
    """Delete the value checkpoint directory."""
    import shutil
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        print(f"[Value Checkpoint] Deleted {checkpoint_dir}")
