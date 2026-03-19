"""Simple replay buffer that stores episodes and samples fixed-length sequences."""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """Stores complete episodes and samples contiguous subsequences for training."""

    def __init__(self, capacity: int = 1_000_000, seq_len: int = 50):
        self.capacity = capacity
        self.seq_len = seq_len

        self._episodes: list[dict[str, np.ndarray]] = []
        self._total_steps = 0

    def add_episode(self, episode: dict[str, np.ndarray]):
        """Add a completed episode. Keys: obs, action, reward, cont (continue flag)."""
        ep_len = len(episode["reward"])
        if ep_len < self.seq_len:
            return  # too short

        self._episodes.append(episode)
        self._total_steps += ep_len

        # Evict old episodes if over capacity
        while self._total_steps > self.capacity and len(self._episodes) > 1:
            removed = self._episodes.pop(0)
            self._total_steps -= len(removed["reward"])

    @property
    def total_steps(self):
        return self._total_steps

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a batch of sequences of length seq_len."""
        obs_list, act_list, rew_list, cont_list = [], [], [], []

        for _ in range(batch_size):
            # Pick a random episode weighted by length
            ep_idx = np.random.randint(len(self._episodes))
            ep = self._episodes[ep_idx]
            ep_len = len(ep["reward"])

            # Pick a random start index
            max_start = ep_len - self.seq_len
            start = np.random.randint(0, max_start + 1)

            obs_list.append(ep["obs"][start:start + self.seq_len])
            act_list.append(ep["action"][start:start + self.seq_len])
            rew_list.append(ep["reward"][start:start + self.seq_len])
            cont_list.append(ep["cont"][start:start + self.seq_len])

        return {
            "obs": np.stack(obs_list),        # (B, T, 3, 64, 64)
            "action": np.stack(act_list),      # (B, T, action_dim)
            "reward": np.stack(rew_list),      # (B, T)
            "cont": np.stack(cont_list),       # (B, T)
        }
