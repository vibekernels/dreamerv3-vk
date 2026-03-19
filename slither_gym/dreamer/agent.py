"""DreamerV3 agent: world model training, imagination, actor-critic."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .networks import (
    WorldModel, Actor, Critic,
    symlog, symexp, twohot_encode, twohot_decode,
)


class DreamerV3Agent:
    """Full DreamerV3 agent with world model, actor, and critic."""

    def __init__(
        self,
        action_dim: int = 3,
        device: str = "cpu",
        # Architecture
        embed_dim: int = 512,
        deter_dim: int = 512,
        stoch_dim: int = 32,
        class_size: int = 32,
        hidden_dim: int = 512,
        cnn_depth: int = 32,
        # Training
        lr_model: float = 1e-4,
        lr_actor: float = 3e-5,
        lr_critic: float = 3e-5,
        imagine_horizon: int = 15,
        gamma: float = 0.997,
        lambda_: float = 0.95,
        entropy_scale: float = 3e-4,
        reward_bins: int = 255,
        free_nats: float = 1.0,
        kl_scale: float = 0.5,
    ):
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.imagine_horizon = imagine_horizon
        self.gamma = gamma
        self.lambda_ = lambda_
        self.entropy_scale = entropy_scale
        self.reward_bins = reward_bins
        self.free_nats = free_nats
        self.kl_scale = kl_scale

        # Networks
        self.world_model = WorldModel(
            action_dim=action_dim,
            embed_dim=embed_dim,
            deter_dim=deter_dim,
            stoch_dim=stoch_dim,
            class_size=class_size,
            hidden_dim=hidden_dim,
            cnn_depth=cnn_depth,
            reward_bins=reward_bins,
        ).to(self.device)

        state_dim = self.world_model.state_dim
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, hidden_dim, num_bins=reward_bins).to(self.device)
        self.slow_critic = Critic(state_dim, hidden_dim, num_bins=reward_bins).to(self.device)
        self.slow_critic.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.model_opt = torch.optim.Adam(self.world_model.parameters(), lr=lr_model, eps=1e-8)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr_actor, eps=1e-8)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-8)

        # Running state for acting
        self._prev_state = None
        self._prev_action = None

    def init_state(self, batch_size: int = 1):
        """Reset agent state for acting."""
        self._prev_state = self.world_model.rssm.initial_state(batch_size, self.device)
        self._prev_action = torch.zeros(batch_size, self.action_dim, device=self.device)

    @torch.no_grad()
    def act(self, obs: np.ndarray, training: bool = True) -> int:
        """Select action given a single observation."""
        if self._prev_state is None:
            self.init_state(1)

        # Preprocess obs: (H, W, 3) uint8 -> (1, 3, H, W) float
        obs_t = torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0

        # Encode
        embed = self.world_model.encoder(obs_t)

        # RSSM step
        post_state, _, _ = self.world_model.rssm.observe_step(
            self._prev_state, self._prev_action, embed
        )

        # Get features and sample action
        features = self.world_model.rssm.get_features(post_state)
        dist = self.actor(features)

        if training:
            action = dist.sample()
        else:
            # Greedy
            action = F.one_hot(dist.logits.argmax(-1), self.action_dim).float()

        self._prev_state = post_state
        self._prev_action = action

        return action.argmax(-1).item()

    def train_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """One training step on a batch of sequences. Returns loss metrics."""
        # Convert to tensors
        obs = torch.from_numpy(batch["obs"]).float().to(self.device) / 255.0  # (B, T, 3, 64, 64)
        actions = torch.from_numpy(batch["action"]).float().to(self.device)     # (B, T, A)
        rewards = torch.from_numpy(batch["reward"]).float().to(self.device)     # (B, T)
        conts = torch.from_numpy(batch["cont"]).float().to(self.device)         # (B, T)

        B, T = obs.shape[:2]

        # --- World Model Training ---
        model_metrics = self._train_world_model(obs, actions, rewards, conts, B, T)

        # --- Actor-Critic Training (imagination) ---
        ac_metrics = self._train_actor_critic(obs, actions, B, T)

        # Slow critic EMA update
        with torch.no_grad():
            for sp, tp in zip(self.slow_critic.parameters(), self.critic.parameters()):
                sp.data.lerp_(tp.data, 0.02)

        return {**model_metrics, **ac_metrics}

    def _train_world_model(self, obs, actions, rewards, conts, B, T):
        # Encode all observations
        obs_flat = obs.reshape(B * T, *obs.shape[2:])
        embed_flat = self.world_model.encoder(obs_flat)
        embed = embed_flat.reshape(B, T, -1)

        # Run RSSM forward
        prev_state = self.world_model.rssm.initial_state(B, self.device)
        posts, priors_logits = [], []
        features_list = []

        for t in range(T):
            post_state, prior_state, prior_logits = self.world_model.rssm.observe_step(
                prev_state, actions[:, t], embed[:, t]
            )
            features = self.world_model.rssm.get_features(post_state)
            posts.append(post_state)
            priors_logits.append(prior_logits)
            features_list.append(features)
            prev_state = post_state

        features = torch.stack(features_list, dim=1)  # (B, T, D)

        # Decode
        feat_flat = features.reshape(B * T, -1)
        recon = self.world_model.decoder(feat_flat)
        obs_target = obs.reshape(B * T, *obs.shape[2:])

        # Image reconstruction loss (MSE on symlog-scaled pixels)
        recon_loss = F.mse_loss(recon, obs_target)

        # Reward prediction loss (twohot cross-entropy)
        reward_logits = self.world_model.reward_head(feat_flat).reshape(B, T, -1)
        reward_target = twohot_encode(
            symlog(rewards), self.reward_bins
        )
        reward_loss = -torch.sum(reward_target * F.log_softmax(reward_logits, dim=-1), dim=-1).mean()

        # Continue prediction loss (binary cross-entropy)
        cont_logits = self.world_model.continue_head(feat_flat).reshape(B, T)
        cont_loss = F.binary_cross_entropy_with_logits(cont_logits, conts)

        # KL loss (free nats, balanced)
        kl_loss = self._kl_loss(posts, priors_logits)

        # Total
        loss = recon_loss + reward_loss + cont_loss + self.kl_scale * kl_loss

        self.model_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.world_model.parameters(), 100.0)
        self.model_opt.step()

        return {
            "recon_loss": recon_loss.item(),
            "reward_loss": reward_loss.item(),
            "cont_loss": cont_loss.item(),
            "kl_loss": kl_loss.item(),
            "model_loss": loss.item(),
        }

    def _kl_loss(self, posts, priors_logits):
        """Compute KL divergence with free nats."""
        total_kl = 0.0
        for post_state, prior_logits in zip(posts, priors_logits):
            post_logits = self.world_model.rssm.post_net(
                torch.cat([
                    post_state["deter"],
                    self.world_model.encoder(torch.zeros(1, device=self.device).expand(post_state["deter"].shape[0], -1)) if False else
                    torch.zeros(1)  # placeholder
                ], dim=-1)
            ) if False else None

            # Use stored stochastic samples to compute KL
            post_stoch = post_state["stoch"]  # (B, stoch_dim, class_size)
            post_probs = post_stoch  # already one-hot-ish from straight-through
            prior_probs = F.softmax(prior_logits, dim=-1)

            # KL(post || prior) ≈ sum post * (log post - log prior)
            post_probs_safe = post_probs.clamp(1e-8, 1.0)
            prior_probs_safe = prior_probs.clamp(1e-8, 1.0)

            kl = (post_probs_safe * (post_probs_safe.log() - prior_probs_safe.log())).sum(dim=-1).mean()
            kl = torch.clamp(kl, min=self.free_nats)
            total_kl = total_kl + kl

        return total_kl / len(posts)

    def _train_actor_critic(self, obs, actions, B, T):
        """Train actor and critic through imagination in the world model."""
        # Get a starting state from the world model (detach from model graph)
        with torch.no_grad():
            obs_flat = obs.reshape(B * T, *obs.shape[2:])
            embed_flat = self.world_model.encoder(obs_flat)
            embed = embed_flat.reshape(B, T, -1)

            state = self.world_model.rssm.initial_state(B, self.device)
            for t in range(T):
                state, _, _ = self.world_model.rssm.observe_step(state, actions[:, t], embed[:, t])

        # Flatten batch for imagination start states
        start_features = self.world_model.rssm.get_features(state).detach()

        # Imagine forward
        imagined_features = [start_features]
        imagined_actions = []
        curr_state = {k: v.detach() for k, v in state.items()}

        for _ in range(self.imagine_horizon):
            features = self.world_model.rssm.get_features(curr_state)
            action_dist = self.actor(features)
            action = action_dist.sample()
            imagined_actions.append(action)

            curr_state = self.world_model.rssm.imagine_step(curr_state, action)
            imagined_features.append(self.world_model.rssm.get_features(curr_state))

        features_stack = torch.stack(imagined_features, dim=1)  # (B, H+1, D)
        actions_stack = torch.stack(imagined_actions, dim=1)     # (B, H, A)

        # Predict rewards and continues in imagination
        feat_flat = features_stack[:, 1:].reshape(-1, features_stack.shape[-1])
        reward_logits = self.world_model.reward_head(feat_flat)
        imagined_rewards = symexp(twohot_decode(reward_logits, self.reward_bins))
        imagined_rewards = imagined_rewards.reshape(B, self.imagine_horizon)

        cont_logits = self.world_model.continue_head(feat_flat).reshape(B, self.imagine_horizon)
        imagined_conts = torch.sigmoid(cont_logits)

        # Critic values
        values = self.critic.value(features_stack.reshape(-1, features_stack.shape[-1]))
        values = values.reshape(B, self.imagine_horizon + 1)

        slow_values = self.slow_critic.value(features_stack.reshape(-1, features_stack.shape[-1]).detach())
        slow_values = slow_values.reshape(B, self.imagine_horizon + 1)

        # Compute lambda-returns
        lambda_returns = self._compute_lambda_returns(
            imagined_rewards, imagined_conts, slow_values
        )

        # --- Critic loss ---
        critic_features = features_stack[:, :-1].reshape(-1, features_stack.shape[-1]).detach()
        critic_logits = self.critic(critic_features)
        target = symlog(lambda_returns.detach()).reshape(-1)
        target_twohot = twohot_encode(target, self.reward_bins)
        critic_loss = -torch.sum(target_twohot * F.log_softmax(critic_logits, dim=-1), dim=-1).mean()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
        self.critic_opt.step()

        # --- Actor loss ---
        # Advantages (with return normalization per DreamerV3)
        with torch.no_grad():
            advantages = lambda_returns - values[:, :-1]
            # Per-batch percentile normalization
            hi = torch.quantile(lambda_returns, 0.95)
            lo = torch.quantile(lambda_returns, 0.05)
            scale = max(hi - lo, 1.0)
            advantages = advantages / scale

        # Policy gradient with entropy bonus
        actor_features = features_stack[:, :-1].reshape(-1, features_stack.shape[-1])
        action_dist = self.actor(actor_features)
        log_probs = action_dist.log_prob(actions_stack.reshape(-1, self.action_dim))
        entropy = action_dist.entropy()

        actor_loss = -(log_probs * advantages.reshape(-1).detach()).mean()
        actor_loss -= self.entropy_scale * entropy.mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.actor_opt.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "imagined_reward_mean": imagined_rewards.mean().item(),
            "entropy": entropy.mean().item(),
        }

    def _compute_lambda_returns(self, rewards, conts, values):
        """Compute GAE-lambda returns."""
        H = rewards.shape[1]
        returns = torch.zeros_like(rewards)
        last = values[:, -1]

        for t in reversed(range(H)):
            delta = rewards[:, t] + self.gamma * conts[:, t] * last - values[:, t]
            last = values[:, t] + delta
            returns[:, t] = (1 - self.lambda_) * values[:, t] + self.lambda_ * (
                rewards[:, t] + self.gamma * conts[:, t] * last
            )
            last = returns[:, t]

        return returns

    def save(self, path: str):
        torch.save({
            "world_model": self.world_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "slow_critic": self.slow_critic.state_dict(),
            "model_opt": self.model_opt.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.world_model.load_state_dict(ckpt["world_model"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.slow_critic.load_state_dict(ckpt["slow_critic"])
        self.model_opt.load_state_dict(ckpt["model_opt"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
