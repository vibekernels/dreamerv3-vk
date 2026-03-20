#!/usr/bin/env python3
"""Evaluate a trained DreamerV3 agent on Slither-v0."""

import argparse
import numpy as np
import torch
import slither_gym
import gymnasium as gym
from slither_gym.dreamer.agent import DreamerV3Agent


def evaluate(checkpoint_path: str, num_episodes: int = 50, device: str = "cuda"):
    env = gym.make("Slither-v0")
    agent = DreamerV3Agent(action_dim=env.action_space.n, device=device)
    agent.load(checkpoint_path)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Running {num_episodes} evaluation episodes (greedy policy)...\n")

    returns, lengths, foods, kills, deaths = [], [], [], [], []

    for ep in range(num_episodes):
        obs, info = env.reset()
        agent.init_state(1)
        done = False
        ep_return = 0.0
        ep_len = 0
        ep_food = 0
        ep_kills = 0
        ep_death = False

        while not done:
            action = agent.act(obs, training=False)  # greedy
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += reward
            ep_len += 1

            # Parse rewards: +1 food, +5 kill, -1 death, +0.001 survival
            if reward >= 4.5:
                ep_kills += 1
            elif reward >= 0.9:
                ep_food += 1
            if terminated:
                ep_death = True

        returns.append(ep_return)
        lengths.append(ep_len)
        foods.append(ep_food)
        kills.append(ep_kills)
        deaths.append(1 if ep_death else 0)

        print(f"  Episode {ep+1:3d}: return={ep_return:7.2f}  length={ep_len:5d}  "
              f"food={ep_food:3d}  kills={ep_kills:2d}  {'DIED' if ep_death else 'SURVIVED'}")

    # Summary statistics
    returns = np.array(returns)
    lengths = np.array(lengths)
    foods = np.array(foods)
    kills = np.array(kills)
    deaths = np.array(deaths)

    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY ({num_episodes} episodes)")
    print(f"{'='*60}")
    print(f"Return:     mean={returns.mean():7.2f}  std={returns.std():7.2f}  "
          f"min={returns.min():7.2f}  max={returns.max():7.2f}")
    print(f"Length:     mean={lengths.mean():7.1f}  std={lengths.std():7.1f}  "
          f"min={lengths.min():5d}    max={lengths.max():5d}")
    print(f"Food/ep:    mean={foods.mean():5.1f}")
    print(f"Kills/ep:   mean={kills.mean():5.2f}")
    print(f"Death rate:  {deaths.mean()*100:.1f}%")
    print(f"Survival rate: {(1-deaths.mean())*100:.1f}%  "
          f"(survived to truncation at 4000 steps)")
    print(f"{'='*60}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/slither/checkpoint_final.pt")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.episodes, args.device)
