"""Shared native TorchRL policy-gradient training.

REINFORCE: exact Monte Carlo reward-to-go, TorchRL ReinforceLoss + critic baseline.
PPO: TorchRL ClipPPOLoss + GAE, multiple minibatch update epochs.
A2C: TorchRL A2CLoss + GAE, one update per on-policy batch.

Each rollout is a full 24-month episode. All 2024 signals are excluded.
"""
from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import A2CLoss, ClipPPOLoss, ReinforceLoss
from torchrl.objectives.value import GAE

from environment.torchrl_adapter import HORIZON, IndustrialTorchRLEnv
from training.networks import make_critic, make_policy


def monte_carlo_targets(trajectory: TensorDict, gamma: float) -> torch.Tensor:
    """Per-episode Monte Carlo return-to-go, without action-dependent normalization.

    Input: [episodes, time] TensorDict, each trajectory exactly HORIZON long.
    Output: [episodes, time, 1] values, no bootstrap beyond terminal.
    """
    rewards = trajectory["next", "reward"].float()
    if rewards.ndim != 3 or rewards.shape[1:] != (HORIZON, 1):
        raise ValueError(f"Unexpected episodic reward shape: {tuple(rewards.shape)}")
    returns = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[:, 0])
    for step in range(HORIZON - 1, -1, -1):
        carry = rewards[:, step] + gamma * carry
        returns[:, step] = carry
    return returns


def make_onpolicy_loss(cfg, actor, critic):
    if cfg.algorithm == "reinforce":
        return ReinforceLoss(
            actor_network=actor,
            critic_network=critic,
            loss_critic_type="l2",
        )
    if cfg.algorithm == "ppo":
        return ClipPPOLoss(
            actor_network=actor,
            critic_network=critic,
            clip_epsilon=cfg.clip_epsilon,
            entropy_bonus=True,
            entropy_coeff=cfg.entropy_coeff,
            critic_coeff=cfg.value_coeff,
            loss_critic_type="l2",
        )
    if cfg.algorithm == "a2c":
        return A2CLoss(
            actor_network=actor,
            critic_network=critic,
            entropy_bonus=True,
            entropy_coeff=cfg.entropy_coeff,
            critic_coeff=cfg.value_coeff,
            loss_critic_type="l2",
        )
    raise ValueError(f"Not an on-policy algorithm: {cfg.algorithm}")


def native_objective(components: TensorDict) -> torch.Tensor:
    keys = components.keys()
    losses = [components[key] for key in keys if str(key).startswith("loss_")]
    if not losses:
        raise RuntimeError(f"TorchRL objective did not produce loss components: {keys}")
    return torch.stack([x.float().reshape(()) for x in losses]).sum()


def train_policy_gradient(cfg):
    env = IndustrialTorchRLEnv(split="historical")
    actor = make_policy(cfg.hidden, env.action_spec)
    critic = make_critic(cfg.hidden)
    objective = make_onpolicy_loss(cfg, actor, critic)
    # Only one source of optimizer parameters, even when TorchRL functionalizes
    # the actor and critic networks for native loss computation.
    optimizer = torch.optim.Adam(objective.parameters(), lr=cfg.lr)
    gae = None
    if cfg.algorithm in ("ppo", "a2c"):
        gae = GAE(
            gamma=cfg.gamma,
            lmbda=cfg.gae_lambda,
            value_network=critic,
            average_gae=False,
        )
    records = []
    episode = 0
    try:
        while episode < cfg.episodes:
            trajectories = []
            n = min(cfg.rollout_episodes, cfg.episodes - episode)
            for local in range(n):
                env.set_seed(cfg.seed + episode + local)
                with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
                    rollout = env.rollout(max_steps=HORIZON, policy=actor)
                if rollout.numel() != HORIZON:
                    raise RuntimeError("Truncated rollout: expected 24 investment steps")
                trajectories.append(rollout.detach())
            batch = torch.stack(trajectories, 0).contiguous()
            if cfg.algorithm == "reinforce":
                returns = monte_carlo_targets(batch, cfg.gamma)
                with torch.no_grad():
                    predicted = critic(batch.clone())["state_value"]
                batch.set("advantage", (returns - predicted).detach())
                batch.set("value_target", returns.detach())
            else:
                with torch.no_grad():
                    gae(batch)
                if cfg.normalize_advantage:
                    advantage = batch["advantage"]
                    batch.set("advantage", (
                        (advantage - advantage.mean())
                        / advantage.std(unbiased=False).clamp_min(1e-6)
                    ))
            flat = batch.reshape(-1).detach()
            losses, actor_losses, critic_losses, entropies = [], [], [], []
            if cfg.algorithm == "ppo":
                epochs = cfg.update_epochs
                mini_size = min(cfg.minibatch_size, len(flat))
            else:
                epochs = 1
                mini_size = len(flat)
            for _ in range(epochs):
                for indices in torch.randperm(len(flat)).split(mini_size):
                    minibatch = flat[indices]
                    values = objective(minibatch)
                    loss = native_objective(values)
                    if cfg.algorithm == "reinforce" and cfg.entropy_coeff:
                        # ReinforceLoss does not provide an entropy term itself.
                        entropy = actor.get_dist(minibatch).entropy().mean()
                        loss = loss - cfg.entropy_coeff * entropy
                    else:
                        with torch.no_grad():
                            entropy = actor.get_dist(minibatch).entropy().mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"{cfg.algorithm}: nonfinite native TorchRL loss")
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(objective.parameters(), 1.0)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                    actor_loss = values.get("loss_actor", values.get("loss_objective"))
                    if actor_loss is not None:
                        actor_losses.append(float(actor_loss.detach()))
                    critic_loss = values.get("loss_value", values.get("loss_critic"))
                    if critic_loss is not None:
                        critic_losses.append(float(critic_loss.detach()))
                    entropies.append(float(entropy.detach()))
            loss_mean = float(np.mean(losses))
            entropy_mean = float(np.mean(entropies))
            actor_mean = float(np.mean(actor_losses)) if actor_losses else float("nan")
            critic_mean = float(np.mean(critic_losses)) if critic_losses else float("nan")
            for local, trajectory in enumerate(trajectories):
                records.append({
                    "episode": episode + local + 1,
                    "reward": float(trajectory["next", "reward"].sum()),
                    "loss": loss_mean,
                    "actor_loss": actor_mean,
                    "critic_loss": critic_mean,
                    "entropy": entropy_mean,
                    "steps": HORIZON,
                    "env_steps": (episode + local + 1) * HORIZON,
                })
            episode += n
            if episode % max(1, cfg.episodes // 5) == 0 or episode == cfg.episodes:
                print(
                    f"{cfg.algorithm.upper()} {episode}/{cfg.episodes}: "
                    f"reward={float(trajectories[-1]['next', 'reward'].sum()):+.4f} "
                    f"loss={loss_mean:.4f} entropy={entropy_mean:.3f}",
                    flush=True,
                )
    finally:
        env.close()
    return actor, critic, records
