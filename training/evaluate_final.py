"""Evaluate the four best historical-validation TorchRL policies on 2024 once.

Call only after the fixed 40-run sweep is completed. 2024 signals initialize
synthetic simulations; outcomes are NOT observed 2024 investment returns.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from environment.trade_env import TradeSignalEnv
from environment.torchrl_adapter import IndustrialTorchRLEnv
from evaluation.baselines import choose_action
from training.experiment import evaluate_model, fix_seed
from training.networks import make_policy, make_q_actor


ROOT = Path("logs/training")


def baseline(name: str, episodes: int = 100, seed: int = 60_000) -> dict:
    env = TradeSignalEnv(split="holdout")
    values = []
    net_worth = []
    try:
        for i in range(episodes):
            obs, info = env.reset(seed=seed+i)
            rng = np.random.default_rng(seed+i+100000)
            total = 0.0
            for _ in range(24):
                choice = choose_action(name, obs, info["action_mask"], rng)
                obs, reward, terminated, truncated, info = env.step(choice)
                total += float(reward)
                if terminated or truncated:
                    break
            values.append(total)
            net_worth.append(float(env.public_state()["net_worth"]))
    finally:
        env.close()
    return {
        "mean_reward": float(np.mean(values)),
        "std_reward": float(np.std(values)),
        "mean_end_net_worth": float(np.mean(net_worth)),
    }


def main():
    rows = []
    for algorithm in ("dqn", "reinforce", "ppo", "a2c"):
        runs = []
        for path in sorted((ROOT / algorithm).glob("*/summary.json")):
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get("train_episodes") == 1000:
                runs.append(summary)
        if len(runs) != 10:
            raise RuntimeError(f"{algorithm}: expected 10 completed configurations, found {len(runs)}")
        # Ties broken deterministically by the lexicographically first run ID.
        best = sorted(runs, key=lambda s: (-s["validation_mean_reward"], s["run_id"]))[0]
        cfg = best["config"]
        fix_seed(int(cfg["seed"]))
        env = IndustrialTorchRLEnv(split="holdout")
        try:
            actor = make_q_actor(int(cfg["hidden"]), env.action_spec) if algorithm == "dqn" else make_policy(int(cfg["hidden"]), env.action_spec)
        finally:
            env.close()
        checkpoint = torch.load(best["model"], map_location="cpu", weights_only=True)
        if checkpoint.get("framework") != "native TorchRL objectives" or checkpoint.get("algorithm") != algorithm:
            raise RuntimeError("Model checkpoint uses an incompatible training implementation")
        actor.load_state_dict(checkpoint["actor_state"])
        greedy = evaluate_model(actor, algorithm, episodes=100, seed=60_000, split="holdout", include_actions=True)
        sampled = (evaluate_model(actor, algorithm, episodes=100, seed=60_000, split="holdout", stochastic=True)
                   if algorithm != "dqn" else None)
        rows.append({
            "algorithm": algorithm, "best_run_id": best["run_id"],
            "historical_validation_mean": best["validation_mean_reward"],
            "heldout_2024_mean_reward": greedy["mean_reward"],
            "heldout_2024_std_reward": greedy["std_reward"],
            "heldout_2024_mean_net_worth": greedy["mean_end_net_worth"],
            "heldout_2024_sampled_mean": sampled["mean_reward"] if sampled else "",
            "holdout_action_counts": json.dumps(greedy["action_counts"]),
            "checkpoint": best["model"],
        })
        print(f"{algorithm.upper():10s} {best['run_id']:16s} historical {best['validation_mean_reward']:+.4f} 2024 {greedy['mean_reward']:+.4f}")
    for strategy in ("always_hold", "random_valid", "rule_based"):
        b = baseline(strategy)
        rows.append({
            "algorithm": strategy,
            "best_run_id": "baseline",
            "historical_validation_mean": "",
            "heldout_2024_mean_reward": b["mean_reward"],
            "heldout_2024_std_reward": b["std_reward"],
            "heldout_2024_mean_net_worth": b["mean_end_net_worth"],
            "heldout_2024_sampled_mean": "",
            "holdout_action_counts": "",
            "checkpoint": "",
        })
    path = ROOT / "final_2024_evaluation.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:", path)
    print("2024 is a held-out trade-signal initialization, not observed financial outcomes.")


if __name__ == "__main__":
    main()
