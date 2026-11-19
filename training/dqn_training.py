"""DQN trained with TorchRL DQNLoss, masked QValueActor and native replay."""
from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.objectives import DQNLoss, HardUpdate, ValueEstimators

from environment.torchrl_adapter import HORIZON, IndustrialTorchRLEnv
from training.networks import make_q_actor


class EpsilonGreedyPolicy:
    """Collect TorchRL rollouts with reproducible legal-action exploration."""

    def __init__(self, actor, seed: int, start: float, end: float, decay_steps: int):
        self.actor = actor
        self.rng = np.random.default_rng(seed + 100_000)
        self.start = float(start)
        self.end = float(end)
        self.decay_steps = max(1, int(decay_steps))
        self.steps = 0

    @property
    def epsilon(self) -> float:
        frac = min(self.steps / self.decay_steps, 1.0)
        return self.start + frac * (self.end - self.start)

    def __call__(self, td: TensorDict) -> TensorDict:
        eps = self.epsilon
        td = self.actor(td)
        if self.rng.random() < eps:
            mask = td["action_mask"].to(torch.bool).cpu().numpy()
            legal = np.flatnonzero(mask)
            td.set("action", torch.tensor(int(self.rng.choice(legal)), dtype=torch.int64))
        self.steps += 1
        return td


def train_dqn(cfg):
    env = IndustrialTorchRLEnv(split="historical")
    actor = make_q_actor(cfg.hidden, env.action_spec)
    loss_module = DQNLoss(
        value_network=actor, action_space="categorical",
        delay_value=True, double_dqn=False, loss_function="smooth_l1",
    )
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=cfg.gamma)
    updater = HardUpdate(loss_module, value_network_update_interval=cfg.target_update)
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg.lr)
    replay = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.replay_size), batch_size=cfg.batch_size
    )
    explorer = EpsilonGreedyPolicy(
        actor, cfg.seed, cfg.eps_start, cfg.eps_end, cfg.eps_decay_steps
    )
    rows = []
    try:
        for episode in range(cfg.episodes):
            env.set_seed(cfg.seed + episode)
            trajectory = env.rollout(max_steps=HORIZON, policy=explorer)
            if trajectory.numel() != HORIZON:
                raise RuntimeError("A DQN rollout did not contain 24 investment decisions")
            replay.extend(trajectory.reshape(-1).detach())
            losses = []
            # One update per environment interaction, subject to replay warmup.
            for step_in_episode in range(HORIZON):
                absolute_step = episode * HORIZON + step_in_episode + 1
                if absolute_step < cfg.warmup or len(replay) < cfg.batch_size:
                    continue
                sample = replay.sample()
                result = loss_module(sample)
                loss = result["loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite native TorchRL DQNLoss")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1.0)
                optimizer.step()
                updater.step()
                losses.append(float(loss.detach()))
            reward = float(trajectory["next", "reward"].sum())
            rows.append({
                "episode": episode + 1,
                "reward": reward,
                "loss": float(np.mean(losses)) if losses else float("nan"),
                "entropy": float("nan"),
                "epsilon": float(explorer.epsilon),
                "steps": HORIZON,
                "env_steps": (episode + 1) * HORIZON,
            })
            if (episode + 1) % max(1, cfg.episodes // 5) == 0:
                print(f"DQN {episode + 1}/{cfg.episodes} reward={reward:+.4f} epsilon={explorer.epsilon:.3f}", flush=True)
    finally:
        env.close()
    return actor, rows
