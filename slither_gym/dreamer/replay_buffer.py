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
        self._ep_lengths: list[int] = []  # cached for fast sampling

    def add_episode(self, episode: dict[str, np.ndarray]):
        """Add a completed episode. Keys: obs, action, reward, cont (continue flag)."""
        ep_len = len(episode["reward"])
        if ep_len < self.seq_len:
            return  # too short

        self._episodes.append(episode)
        self._ep_lengths.append(ep_len)
        self._total_steps += ep_len

        # Evict old episodes if over capacity
        while self._total_steps > self.capacity and len(self._episodes) > 1:
            removed = self._episodes.pop(0)
            self._ep_lengths.pop(0)
            self._total_steps -= len(removed["reward"])

    @property
    def total_steps(self):
        return self._total_steps

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a batch of sequences of length seq_len (vectorized)."""
        n_eps = len(self._episodes)

        # Pick random episodes and start indices in bulk
        ep_indices = np.random.randint(0, n_eps, size=batch_size)
        # Compute max_start per selected episode, then sample starts
        max_starts = np.array([self._ep_lengths[i] - self.seq_len for i in ep_indices])
        starts = (np.random.random(batch_size) * (max_starts + 1)).astype(np.intp)

        # Pre-allocate output arrays
        sample_ep = self._episodes[ep_indices[0]]
        obs_shape = sample_ep["obs"].shape[1:]   # (3, 64, 64)
        act_dim = sample_ep["action"].shape[1]   # action_dim

        obs = np.empty((batch_size, self.seq_len, *obs_shape), dtype=np.float32)
        actions = np.empty((batch_size, self.seq_len, act_dim), dtype=np.float32)
        rewards = np.empty((batch_size, self.seq_len), dtype=np.float32)
        conts = np.empty((batch_size, self.seq_len), dtype=np.float32)

        for i in range(batch_size):
            ep = self._episodes[ep_indices[i]]
            s = starts[i]
            e = s + self.seq_len
            obs[i] = ep["obs"][s:e]
            actions[i] = ep["action"][s:e]
            rewards[i] = ep["reward"][s:e]
            conts[i] = ep["cont"][s:e]

        return {
            "obs": obs,           # (B, T, 3, 64, 64)
            "action": actions,    # (B, T, action_dim)
            "reward": rewards,    # (B, T)
            "cont": conts,        # (B, T)
        }
