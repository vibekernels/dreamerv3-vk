from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardConfig:
    food_eaten: float = 1.0
    kill_opponent: float = 5.0
    death: float = -1.0
    survival_per_step: float = 0.001
    boost_cost: float = -0.01


def compute_reward(events: dict, config: RewardConfig) -> float:
    reward = config.survival_per_step
    reward += events.get("food_eaten", 0.0) * config.food_eaten
    reward += events.get("killed_opponent", 0) * config.kill_opponent
    if events.get("died", False):
        reward += config.death
    if events.get("boosting", False):
        reward += config.boost_cost
    return reward
