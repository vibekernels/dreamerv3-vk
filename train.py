#!/usr/bin/env python3
"""Train DreamerV3 on Slither-v0.

Usage:
    python train.py                          # CPU (slow, for testing)
    python train.py --device cuda            # GPU (recommended)
    python train.py --device cuda --steps 1000000 --logdir runs/slither

Logs to TensorBoard. View with: tensorboard --logdir runs/
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

import slither_gym  # noqa: F401
import gymnasium as gym

from slither_gym.dreamer.agent import DreamerV3Agent
from slither_gym.dreamer.replay_buffer import ReplayBuffer


def collect_episode(env: gym.Env, agent: DreamerV3Agent, action_dim: int) -> dict:
    """Collect one full episode, returning numpy arrays."""
    obs_list, act_list, rew_list, cont_list = [], [], [], []

    obs, info = env.reset()
    agent.init_state(1)
    done = False

    while not done:
        action = agent.act(obs, training=True)

        # One-hot encode action
        action_oh = np.zeros(action_dim, dtype=np.float32)
        action_oh[action] = 1.0

        # Store current step
        obs_t = np.transpose(obs, (2, 0, 1)).astype(np.float32)  # (3, 64, 64)
        obs_list.append(obs_t)
        act_list.append(action_oh)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        rew_list.append(np.float32(reward))
        cont_list.append(np.float32(0.0 if terminated else 1.0))

    return {
        "obs": np.stack(obs_list),
        "action": np.stack(act_list),
        "reward": np.array(rew_list, dtype=np.float32),
        "cont": np.array(cont_list, dtype=np.float32),
    }


def main():
    parser = argparse.ArgumentParser(description="Train DreamerV3 on Slither-v0")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--steps", type=int, default=500_000, help="Total env steps")
    parser.add_argument("--logdir", type=str, default="runs/slither", help="TensorBoard log dir")
    parser.add_argument("--save_every", type=int, default=50_000, help="Save checkpoint every N steps")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--seq_len", type=int, default=50, help="Sequence length for training")
    parser.add_argument("--train_ratio", type=int, default=512, help="Train steps per env step ratio (as in DreamerV3)")
    parser.add_argument("--prefill", type=int, default=5000, help="Random steps before training starts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Setup
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(logdir))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make("Slither-v0")
    action_dim = env.action_space.n

    agent = DreamerV3Agent(
        action_dim=action_dim,
        device=args.device,
    )

    if args.resume:
        agent.load(args.resume)
        print(f"Resumed from {args.resume}")

    buffer = ReplayBuffer(capacity=1_000_000, seq_len=args.seq_len)

    # --- Prefill with random actions ---
    print(f"Prefilling buffer with {args.prefill} steps of random actions...")
    prefill_steps = 0
    while prefill_steps < args.prefill:
        obs_list, act_list, rew_list, cont_list = [], [], [], []
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            action_oh = np.zeros(action_dim, dtype=np.float32)
            action_oh[action] = 1.0
            obs_t = np.transpose(obs, (2, 0, 1)).astype(np.float32)
            obs_list.append(obs_t)
            act_list.append(action_oh)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rew_list.append(np.float32(reward))
            cont_list.append(np.float32(0.0 if terminated else 1.0))
            prefill_steps += 1
        ep = {
            "obs": np.stack(obs_list),
            "action": np.stack(act_list),
            "reward": np.array(rew_list, dtype=np.float32),
            "cont": np.array(cont_list, dtype=np.float32),
        }
        buffer.add_episode(ep)
    print(f"Prefilled {buffer.total_steps} steps across {len(buffer._episodes)} episodes")

    # --- Main training loop ---
    total_env_steps = prefill_steps
    train_steps = 0
    episode_count = 0
    last_save = 0
    start_time = time.time()

    print(f"\nStarting training on {args.device} for {args.steps} env steps...")
    print(f"Train ratio: {args.train_ratio} (train {args.train_ratio} steps per episode)")
    print(f"Batch size: {args.batch_size}, Sequence length: {args.seq_len}")
    print(f"Logging to: {logdir}\n")

    while total_env_steps < args.steps:
        # Collect one episode
        episode = collect_episode(env, agent, action_dim)
        ep_len = len(episode["reward"])
        ep_return = episode["reward"].sum()
        episode_count += 1
        total_env_steps += ep_len
        buffer.add_episode(episode)

        # Log episode stats
        elapsed = time.time() - start_time
        sps = total_env_steps / elapsed if elapsed > 0 else 0
        print(f"Episode {episode_count:>5d} | "
              f"Steps: {total_env_steps:>8d}/{args.steps} | "
              f"Return: {ep_return:>7.2f} | "
              f"Length: {ep_len:>4d} | "
              f"SPS: {sps:.0f}")

        writer.add_scalar("episode/return", ep_return, total_env_steps)
        writer.add_scalar("episode/length", ep_len, total_env_steps)
        writer.add_scalar("performance/sps", sps, total_env_steps)

        # Train on collected data
        # DreamerV3 uses a high train ratio: many gradient steps per env step
        n_train = max(1, ep_len * args.train_ratio // (args.batch_size * args.seq_len))

        for i in range(n_train):
            batch = buffer.sample(args.batch_size)
            metrics = agent.train_step(batch)
            train_steps += 1

        # Log training metrics
        for k, v in metrics.items():
            writer.add_scalar(f"train/{k}", v, total_env_steps)
        writer.add_scalar("train/train_steps", train_steps, total_env_steps)

        # Save checkpoint
        if total_env_steps - last_save >= args.save_every:
            ckpt_path = logdir / f"checkpoint_{total_env_steps}.pt"
            agent.save(str(ckpt_path))
            print(f"  -> Saved checkpoint: {ckpt_path}")
            last_save = total_env_steps

    # Final save
    agent.save(str(logdir / "checkpoint_final.pt"))
    print(f"\nTraining complete! {total_env_steps} env steps, {train_steps} train steps")
    print(f"Final checkpoint: {logdir / 'checkpoint_final.pt'}")
    env.close()
    writer.close()


if __name__ == "__main__":
    main()
