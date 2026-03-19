"""DreamerV3 neural network components.

Implements: CNN encoder/decoder, RSSM world model, MLP actor & critic,
with symlog transforms and unimix categoricals per the DreamerV3 paper.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal, OneHotCategorical


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def twohot_encode(x: torch.Tensor, num_bins: int = 255, low: float = -20.0, high: float = 20.0) -> torch.Tensor:
    """Encode scalar values as two-hot vectors over a symlog-spaced grid."""
    bins = torch.linspace(low, high, num_bins, device=x.device)
    x = x.clamp(low, high)
    # Find bin indices
    below = (x.unsqueeze(-1) >= bins).sum(-1) - 1
    below = below.clamp(0, num_bins - 2)
    above = below + 1
    # Interpolation weights
    weight_above = (x - bins[below]) / (bins[above] - bins[below] + 1e-8)
    weight_above = weight_above.clamp(0, 1)
    weight_below = 1 - weight_above
    # Two-hot
    result = torch.zeros(*x.shape, num_bins, device=x.device)
    result.scatter_(-1, below.unsqueeze(-1), weight_below.unsqueeze(-1))
    result.scatter_(-1, above.unsqueeze(-1), weight_above.unsqueeze(-1))
    return result


def twohot_decode(logits: torch.Tensor, num_bins: int = 255, low: float = -20.0, high: float = 20.0) -> torch.Tensor:
    """Decode two-hot logits to scalar values."""
    bins = torch.linspace(low, high, num_bins, device=logits.device)
    probs = F.softmax(logits, dim=-1)
    return (probs * bins).sum(-1)


class LayerNormSiLU(nn.Module):
    """LayerNorm + SiLU activation (DreamerV3 default)."""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return F.silu(self.norm(x))


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    mods = []
    for i in range(layers):
        d_in = in_dim if i == 0 else hidden
        mods.append(nn.Linear(d_in, hidden))
        mods.append(LayerNormSiLU(hidden))
    mods.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*mods)


# ---------------------------------------------------------------------------
# CNN Encoder / Decoder (for 64x64 images)
# ---------------------------------------------------------------------------

class CNNEncoder(nn.Module):
    """Encodes 64x64x3 images to a flat embedding vector."""

    def __init__(self, depth: int = 32, out_dim: int = 512):
        super().__init__()
        # 64->32->16->8->4 with channels depth*[1,2,4,8]
        self.convs = nn.Sequential(
            nn.Conv2d(3, depth, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(depth, depth * 2, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(depth * 2, depth * 4, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(depth * 4, depth * 8, 4, 2, 1), nn.SiLU(),
        )
        self.flat_dim = depth * 8 * 4 * 4
        self.fc = nn.Linear(self.flat_dim, out_dim)
        self.norm = LayerNormSiLU(out_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, 3, 64, 64) float
        x = self.convs(obs)
        x = x.reshape(x.shape[0], -1)
        return self.norm(self.fc(x))


class CNNDecoder(nn.Module):
    """Decodes latent state to 64x64x3 image reconstruction."""

    def __init__(self, in_dim: int, depth: int = 32):
        super().__init__()
        self.depth = depth
        self.fc = nn.Linear(in_dim, depth * 8 * 4 * 4)
        self.norm = LayerNormSiLU(depth * 8 * 4 * 4)
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(depth * 8, depth * 4, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(depth * 4, depth * 2, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(depth * 2, depth, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(depth, 3, 4, 2, 1),  # -> 64x64x3, no activation (MSE loss)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.fc(features))
        x = x.reshape(-1, self.depth * 8, 4, 4)
        return self.deconvs(x)  # (B, 3, 64, 64)


# ---------------------------------------------------------------------------
# RSSM World Model
# ---------------------------------------------------------------------------

class RSSM(nn.Module):
    """Recurrent State-Space Model with discrete latents (DreamerV3 style).

    State = (deterministic h, stochastic z).
    z is represented as `num_classes` categorical variables each with `class_size` classes.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        action_dim: int = 3,
        deter_dim: int = 512,
        stoch_dim: int = 32,
        class_size: int = 32,
        hidden_dim: int = 512,
        unimix: float = 0.01,
    ):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.class_size = class_size
        self.unimix = unimix

        stoch_flat = stoch_dim * class_size

        # Prior: h -> z_hat (imagination)
        self.prior_net = mlp(deter_dim, hidden_dim, stoch_dim * class_size, layers=1)

        # Posterior: h, embed -> z (training)
        self.post_net = mlp(deter_dim + embed_dim, hidden_dim, stoch_dim * class_size, layers=1)

        # Sequence model: z_prev, a -> input for GRU
        self.pre_gru = nn.Sequential(
            nn.Linear(stoch_flat + action_dim, hidden_dim),
            LayerNormSiLU(hidden_dim),
        )
        self.gru = nn.GRUCell(hidden_dim, deter_dim)

    @property
    def state_dim(self) -> int:
        """Total feature dimension: deter + stoch_flat."""
        return self.deter_dim + self.stoch_dim * self.class_size

    def initial_state(self, batch_size: int, device: torch.device):
        return {
            "deter": torch.zeros(batch_size, self.deter_dim, device=device),
            "stoch": torch.zeros(batch_size, self.stoch_dim, self.class_size, device=device),
        }

    def get_features(self, state: dict) -> torch.Tensor:
        stoch_flat = state["stoch"].reshape(state["stoch"].shape[0], -1)
        return torch.cat([state["deter"], stoch_flat], dim=-1)

    def _categorical(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample from categorical with straight-through and unimix."""
        logits = logits.reshape(-1, self.stoch_dim, self.class_size)
        # Uniform mix for exploration
        if self.unimix > 0:
            probs = F.softmax(logits, dim=-1)
            probs = (1 - self.unimix) * probs + self.unimix / self.class_size
            logits = torch.log(probs + 1e-8)
        # Straight-through: sample but pass gradients through softmax
        dist = OneHotCategorical(logits=logits)
        sample = dist.sample()
        # Straight-through estimator
        probs = F.softmax(logits, dim=-1)
        return sample + probs - probs.detach()

    def observe_step(self, prev_state: dict, action: torch.Tensor, embed: torch.Tensor):
        """One step of posterior (training). Returns prior and posterior states."""
        prior_state = self.imagine_step(prev_state, action)
        h = prior_state["deter"]

        # Posterior
        post_logits = self.post_net(torch.cat([h, embed], dim=-1))
        post_stoch = self._categorical(post_logits)
        post_state = {"deter": h, "stoch": post_stoch}

        return post_state, prior_state, post_logits.reshape(-1, self.stoch_dim, self.class_size)

    def imagine_step(self, prev_state: dict, action: torch.Tensor):
        """One step of prior (imagination). Returns prior state."""
        stoch_flat = prev_state["stoch"].reshape(prev_state["stoch"].shape[0], -1)
        x = self.pre_gru(torch.cat([stoch_flat, action], dim=-1))
        h = self.gru(x, prev_state["deter"])

        prior_logits = self.prior_net(h)
        prior_stoch = self._categorical(prior_logits)
        return {"deter": h, "stoch": prior_stoch}

    def get_prior_logits(self, deter: torch.Tensor) -> torch.Tensor:
        return self.prior_net(deter).reshape(-1, self.stoch_dim, self.class_size)


# ---------------------------------------------------------------------------
# Actor & Critic
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """Discrete action policy head."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 512, layers: int = 3):
        super().__init__()
        self.net = mlp(state_dim, hidden, action_dim, layers=layers)

    def forward(self, features: torch.Tensor) -> torch.distributions.Distribution:
        logits = self.net(features)
        return OneHotCategorical(logits=logits)

    def log_prob(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        dist = self.forward(features)
        return dist.log_prob(actions)


class Critic(nn.Module):
    """Critic predicting twohot-encoded value (DreamerV3 style)."""

    def __init__(self, state_dim: int, hidden: int = 512, layers: int = 3, num_bins: int = 255):
        super().__init__()
        self.net = mlp(state_dim, hidden, num_bins, layers=layers)
        self.num_bins = num_bins

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns twohot logits."""
        return self.net(features)

    def value(self, features: torch.Tensor) -> torch.Tensor:
        """Returns scalar value estimate."""
        logits = self.forward(features)
        return twohot_decode(logits, self.num_bins)


# ---------------------------------------------------------------------------
# Full World Model
# ---------------------------------------------------------------------------

class WorldModel(nn.Module):
    def __init__(
        self,
        action_dim: int = 3,
        embed_dim: int = 512,
        deter_dim: int = 512,
        stoch_dim: int = 32,
        class_size: int = 32,
        hidden_dim: int = 512,
        cnn_depth: int = 32,
        reward_bins: int = 255,
    ):
        super().__init__()
        self.encoder = CNNEncoder(depth=cnn_depth, out_dim=embed_dim)
        self.rssm = RSSM(
            embed_dim=embed_dim,
            action_dim=action_dim,
            deter_dim=deter_dim,
            stoch_dim=stoch_dim,
            class_size=class_size,
            hidden_dim=hidden_dim,
        )
        state_dim = self.rssm.state_dim
        self.decoder = CNNDecoder(in_dim=state_dim, depth=cnn_depth)
        self.reward_head = mlp(state_dim, hidden_dim, reward_bins, layers=2)
        self.continue_head = mlp(state_dim, hidden_dim, 1, layers=2)

        self.reward_bins = reward_bins

    @property
    def state_dim(self):
        return self.rssm.state_dim
