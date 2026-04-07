"""
Reservoir Sampling Buffer

Fixed-size buffer that maintains a uniform random sample of all items
ever added. Used by all training modes to store (features, target) pairs.
"""

import random
from typing import List, Tuple, Any


class ReservoirBuffer:
    """Fixed-size reservoir sampling buffer with iteration weighting.

    When the buffer is full, new items replace random existing items
    with decreasing probability, maintaining a uniform sample.

    Each entry stores (features, target, iteration) for Linear CFR weighting.
    """

    def __init__(self, max_size: int = 100_000):
        self.max_size = max_size
        self.buffer: List[Tuple[List[float], List[float], int]] = []
        self.total_added = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, features: List[float], target: List[float], iteration: int = 0):
        """Add a (features, target, iteration) triple to the reservoir."""
        self.total_added += 1

        if len(self.buffer) < self.max_size:
            self.buffer.append((features, target, iteration))
        else:
            # Reservoir sampling: replace random element with decreasing probability
            idx = random.randint(0, self.total_added - 1)
            if idx < self.max_size:
                self.buffer[idx] = (features, target, iteration)

    def sample(self, batch_size: int) -> Tuple[List[List[float]], List[List[float]], List[int]]:
        """Sample a batch. Returns (features_batch, targets_batch, iterations_batch)."""
        if not self.buffer:
            return [], [], []

        batch_size = min(batch_size, len(self.buffer))
        indices = random.sample(range(len(self.buffer)), batch_size)

        features = [self.buffer[i][0] for i in indices]
        targets = [self.buffer[i][1] for i in indices]
        iterations = [self.buffer[i][2] for i in indices]
        return features, targets, iterations

    @property
    def size(self) -> int:
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
        self.total_added = 0

    def resize(self, new_max_size: int):
        """Resize the reservoir. If growing, existing data is preserved.
        If shrinking, randomly samples down to the new size."""
        old_size = self.max_size
        if new_max_size >= len(self.buffer):
            # Growing or same — just update the cap, all data preserved
            self.max_size = new_max_size
            print(f"[Reservoir] Resized {old_size:,} → {new_max_size:,} ({len(self.buffer):,} samples preserved)")
        else:
            # Shrinking — randomly sample down
            self.buffer = random.sample(self.buffer, new_max_size)
            self.max_size = new_max_size
            print(f"[Reservoir] Resized {old_size:,} → {new_max_size:,} (sampled down from {len(self.buffer):,})")
