"""Train/evaluate one model with the native TorchRL RL objective classes.

Run: uv run python -m training.experiment --algorithm ppo --episodes 8 --run-id check --rollout-episodes 4
Production hyperparameter trials are launched by training.run_sweep.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

from environment.torchrl_adapter import HORIZON, IndustrialTorchRLEnv
from training.networks import make_policy, make_q_actor
from training.dqn_training import train_dqn
from training.pg_training import train_policy_gradient

ALGORITHMS = ("dqn", "reinforce", "ppo", "a2c")


@dataclass
class Config:
    algorithm: str = "dqn"
    episodes: int = 8
    seed: int = 42
    run_id: str = "check"
    lr: float = 5e-4
    gamma: float = 0.98
    hidden: int = 64
    batch_size: int = 32
    replay_size: int = 5000
    warmup: int = 250
    target_update: int = 100
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 16000
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    update_epochs: int = 4
    minibatch_size: int = 48
    rollout_episodes: int = 8
    normalize_advantage: bool = True
    eval_episodes: int = 10
    output_dir: str = "logs/training"
    models_dir: str = "models"


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def evaluate_model(model, algorithm: str, episodes: int, seed: int,
                   split: str = "historical", stochastic: bool = False,
                   include_actions: bool = False) -> dict:
    env = IndustrialTorchRLEnv(split=split)
    scores = []
    action_counts = np.zeros(17, dtype=np.int64)
    ending_wealth = []
    try:
        model.eval()
        for index in range(episodes):
            env.set_seed(seed + index)
            # Reset Torch RNG for reproducible sampled actions; evaluation never updates weights.
            torch.manual_seed(seed + index + 500_000)
            mode = (ExplorationType.RANDOM if stochastic and algorithm != "dqn"
                    else ExplorationType.MODE)
            with torch.no_grad(), set_exploration_type(mode):
                rollout = env.rollout(max_steps=HORIZON, policy=model)
            scores.append(float(rollout["next", "reward"].sum()))
            actions = rollout["action"].cpu().numpy().reshape(-1).astype(int)
            action_counts += np.bincount(actions, minlength=17)
            ending_wealth.append(float(env.backend.public_state()["net_worth"]))
    finally:
        env.close()
    result = {
        "mean_reward": float(np.mean(scores)),
        "std_reward": float(np.std(scores)),
        "mean_end_net_worth": float(np.mean(ending_wealth)),
    }
    if include_actions:
        result["action_counts"] = action_counts.tolist()
    return result


def train(cfg: Config) -> dict:
    if cfg.algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {cfg.algorithm}")
    if min(cfg.episodes, cfg.eval_episodes, cfg.rollout_episodes) < 1:
        raise ValueError("Episodes, evaluation count, and rollout batch must be positive")
    if cfg.algorithm == "dqn" and (cfg.batch_size < 1 or cfg.replay_size < cfg.batch_size or cfg.warmup < 1):
        raise ValueError("Invalid DQN replay/warmup parameters")
    if cfg.algorithm == "ppo" and (cfg.update_epochs < 1 or cfg.minibatch_size < 1):
        raise ValueError("Invalid PPO update parameters")
    fix_seed(cfg.seed)
    if cfg.algorithm == "dqn":
        model, history = train_dqn(cfg)
        critic = None
    else:
        model, critic, history = train_policy_gradient(cfg)
    validation = evaluate_model(model, cfg.algorithm, cfg.eval_episodes, seed=50_000)
    sampled = (
        evaluate_model(model, cfg.algorithm, cfg.eval_episodes, seed=50_000, stochastic=True)
        if cfg.algorithm != "dqn" else None
    )
    output = Path(cfg.output_dir) / cfg.algorithm / cfg.run_id
    output.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in history for key in row))
    with (output / "episodes.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)
    checkpoint = Path(cfg.models_dir) / cfg.algorithm / (cfg.run_id + ".pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "algorithm": cfg.algorithm,
        "actor_state": model.state_dict(),
        "critic_state": critic.state_dict() if critic is not None else None,
        "config": asdict(cfg),
        "obs_dim": 53,
        "n_actions": 17,
        "framework": "native TorchRL objectives",
    }, checkpoint)
    summary = {
        "run_id": cfg.run_id,
        "algorithm": cfg.algorithm,
        "train_episodes": cfg.episodes,
        "environment_steps": cfg.episodes * HORIZON,
        "last_100_train_mean": float(np.mean([row["reward"] for row in history[-100:]])),
        "validation_mean_reward": validation["mean_reward"],
        "validation_std_reward": validation["std_reward"],
        "validation_mean_end_net_worth": validation["mean_end_net_worth"],
        "validation_stochastic_mean_reward": sampled["mean_reward"] if sampled else None,
        "validation_stochastic_std_reward": sampled["std_reward"] if sampled else None,
        "validation_seeds": [50000, 50000 + cfg.eval_episodes - 1],
        "evaluation_split": "historical 2016-2023; 2024 not used for tuning",
        "model": str(checkpoint).replace("\\", "/"),
        "config": asdict(cfg),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {checkpoint} and {output / 'summary.json'}", flush=True)
    print(f"Validation greedy reward {validation['mean_reward']:+.4f} ± {validation['std_reward']:.4f}")
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-id", default="check")
    p.add_argument("--lr", type=float, default=0.0005)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--replay-size", type=int, default=5000)
    p.add_argument("--warmup", type=int, default=250)
    p.add_argument("--target-update", type=int, default=100)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-steps", type=int, default=16000)
    p.add_argument("--entropy-coeff", type=float, default=0.01)
    p.add_argument("--value-coeff", type=float, default=0.5)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-epsilon", type=float, default=0.2)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=48)
    p.add_argument("--rollout-episodes", type=int, default=8)
    p.add_argument("--normalize-advantage", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--output-dir", default="logs/training")
    p.add_argument("--models-dir", default="models")
    return p


def main():
    cfg = Config(**vars(parser().parse_args()))
    train(cfg)


if __name__ == "__main__":
    main()
