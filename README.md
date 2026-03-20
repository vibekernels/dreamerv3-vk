# slither-gym

A slither.io clone as a Gymnasium environment with a built-in DreamerV3 implementation for RL training.

## Overview

- **Game**: Simplified slither.io — snakes eat food to grow, die on collision with other snakes or the arena boundary. 4 NPC bot snakes provide opponents.
- **Observation**: 64×64×3 uint8 RGB image (ego-centric view centered on the player's head)
- **Action**: `Discrete(3)` — straight, turn left, turn right
- **Reward**: +1 per food eaten, +5 per kill, -1 on death, +0.001 survival bonus per step

## Setup

```bash
git clone <repo-url> && cd slither-gym
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install torch tensorboard
```

For GPU training, install PyTorch with CUDA (see https://pytorch.org/get-started/locally/):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Training with DreamerV3

```bash
# GPU (recommended)
python train.py --device cuda --steps 500000 --logdir runs/slither

# CPU (very slow, only for smoke-testing)
python train.py --device cpu --steps 6000 --prefill 1000 --train_ratio 32 --batch_size 4 --seq_len 16
```

### Training flags

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `cpu` | `cpu` or `cuda` |
| `--steps` | 500,000 | Total environment steps |
| `--batch_size` | 16 | Sequences per training batch |
| `--seq_len` | 50 | Sequence length for world model training |
| `--train_ratio` | 512 | Gradient steps per env step (DreamerV3 default) |
| `--prefill` | 5,000 | Random exploration steps before training begins |
| `--save_every` | 50,000 | Checkpoint interval (env steps) |
| `--logdir` | `runs/slither` | TensorBoard log and checkpoint directory |
| `--resume` | None | Path to a checkpoint `.pt` file to resume from |
| `--seed` | 0 | Random seed |

### Monitoring

```bash
tensorboard --logdir runs/
```

Key metrics to watch:
- `episode/return` — total reward per episode (should trend upward)
- `episode/length` — survival time (longer = better)
- `train/recon_loss` — image reconstruction quality (should decrease)
- `train/entropy` — policy entropy (should decrease gradually)

### Checkpoints

Checkpoints are saved to `--logdir` as `checkpoint_<steps>.pt`. Resume training:

```bash
python train.py --device cuda --resume runs/slither/checkpoint_50000.pt
```

## Evaluation

Evaluate a trained checkpoint with greedy or stochastic policies:

```bash
# Greedy policy (deterministic action selection)
python eval.py --checkpoint runs/slither/checkpoint_final.pt --episodes 50

# Stochastic policy (samples from the learned distribution)
python eval_stochastic.py
```

### Results (500K steps, RTX 4090)

| Metric | Greedy Policy | Stochastic Policy |
|---|---|---|
| Mean Return | 18.49 | 19.24 |
| Max Return | 45.00 | 74.00 |
| Mean Episode Length | 2,007 | 2,039 |
| Survival Rate (4000 steps) | 24.0% | 30.0% |
| Food/episode | 15.2 | — |
| Kills/episode | 0.36 | — |

Training takes ~15 hours on an RTX 4090 (~9 env steps/sec). GPU memory usage is ~2.2 GB.

## Human play

You can play the game manually with keyboard controls:

```bash
pip install pygame
python examples/human_play.py
```

| Key | Action |
|-----|--------|
| Left Arrow / A | Turn left |
| Right Arrow / D | Turn right |
| (no key) | Go straight |
| R | Restart after death |
| ESC / Q | Quit |

## Using the environment directly

```python
import slither_gym
import gymnasium as gym

env = gym.make("Slither-v0")
obs, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Environment constructor options

```python
env = gym.make(
    "Slither-v0",
    render_mode="human",     # or "rgb_array" (default: None)
    num_npcs=4,              # number of bot opponents
    arena_radius=500.0,      # world size
    max_steps=4000,          # episode truncation limit
    obs_size=64,             # observation image size (NxN)
    viewport_radius=120.0,   # visible world radius around the player
)
```

Human rendering requires pygame: `pip install pygame`

## Project structure

```
slither_gym/
├── engine/          # Game simulation (NumPy-based)
│   ├── config.py    #   Game parameters
│   ├── snake.py     #   Snake with ring-buffer segments
│   ├── food.py      #   Food spawning and consumption
│   └── game.py      #   Main game loop, NPCs, collisions
├── env/             # Gymnasium interface
│   ├── slither_env.py  # Env subclass
│   └── rewards.py      # Configurable reward function
├── rendering/       # Observation rendering
│   └── numpy_renderer.py  # Ego-centric 64x64 RGB rasterizer
└── dreamer/         # DreamerV3 implementation (PyTorch)
    ├── networks.py     # RSSM, CNN encoder/decoder, actor, critic
    ├── agent.py        # Training logic (world model + actor-critic)
    └── replay_buffer.py
```
