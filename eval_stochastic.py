#!/usr/bin/env python3
"""Evaluate with stochastic (training) policy for comparison."""
import numpy as np
import torch
import slither_gym
import gymnasium as gym
from slither_gym.dreamer.agent import DreamerV3Agent

env = gym.make("Slither-v0")
agent = DreamerV3Agent(action_dim=3, device="cuda")
agent.load("runs/slither/checkpoint_final.pt")
print("Evaluating with STOCHASTIC policy (50 episodes)...\n")

returns, lengths = [], []
for ep in range(50):
    obs, _ = env.reset()
    agent.init_state(1)
    done, ep_ret, ep_len = False, 0.0, 0
    while not done:
        action = agent.act(obs, training=True)  # stochastic
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ep_ret += reward
        ep_len += 1
    returns.append(ep_ret)
    lengths.append(ep_len)
    print(f"  Ep {ep+1:3d}: return={ep_ret:7.2f}  length={ep_len:5d}  "
          f"{'DIED' if terminated else 'SURVIVED'}")

returns = np.array(returns)
lengths = np.array(lengths)
print(f"\nSTOCHASTIC POLICY SUMMARY")
print(f"Return: mean={returns.mean():.2f}  std={returns.std():.2f}  max={returns.max():.2f}")
print(f"Length: mean={lengths.mean():.1f}  std={lengths.std():.1f}  max={lengths.max()}")
print(f"Survival rate: {(lengths >= 4000).mean()*100:.1f}%")
env.close()
