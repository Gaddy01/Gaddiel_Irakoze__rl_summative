"""Pre-registered 40-run native TorchRL experiment sweep.

One fixed 1000-episode (24000-transition) budget per configuration. Training
uses only 2016-2023 historical signals, with 2024 reserved for the final test.

Usage:
    uv run python -m training.run_sweep --dry-run
    uv run python -m training.run_sweep --algorithm dqn
    uv run python -m training.run_sweep
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

EPISODES = 1000
EVAL_EPISODES = 100
SEED = 42

# Deliberately meaningful, pre-registered variations. No tuning against 2024.
CONFIGS = {
    "dqn": [
        {"lr": .0005, "gamma": .98, "hidden": 64, "batch_size": 32, "replay_size": 5000, "target_update": 100, "eps_decay_steps": 16000, "warmup": 250},
        {"lr": .0002, "gamma": .98, "hidden": 64, "batch_size": 32, "replay_size": 5000, "target_update": 100, "eps_decay_steps": 16000, "warmup": 250},
        {"lr": .0010, "gamma": .98, "hidden": 64, "batch_size": 32, "replay_size": 5000, "target_update": 100, "eps_decay_steps": 16000, "warmup": 250},
        {"lr": .0005, "gamma": .95, "hidden": 64, "batch_size": 32, "replay_size": 5000, "target_update": 100, "eps_decay_steps": 16000, "warmup": 250},
        {"lr": .0005, "gamma": .995, "hidden": 64, "batch_size": 32, "replay_size": 5000, "target_update": 100, "eps_decay_steps": 16000, "warmup": 250},
        {"lr": .0005, "gamma": .98, "hidden": 128, "batch_size": 32, "replay_size": 5000, "target_update": 100, "eps_decay_steps": 16000, "warmup": 250},
        {"lr": .0005, "gamma": .98, "hidden": 64, "batch_size": 64, "replay_size": 10000, "target_update": 100, "eps_decay_steps": 16000, "warmup": 250},
        {"lr": .0005, "gamma": .98, "hidden": 64, "batch_size": 32, "replay_size": 5000, "target_update": 250, "eps_decay_steps": 16000, "warmup": 250},
        {"lr": .0005, "gamma": .98, "hidden": 64, "batch_size": 32, "replay_size": 5000, "target_update": 100, "eps_decay_steps": 8000, "warmup": 250},
        {"lr": .0005, "gamma": .98, "hidden": 64, "batch_size": 32, "replay_size": 5000, "target_update": 100, "eps_decay_steps": 20000, "eps_end": .10, "warmup": 250},
    ],
    "reinforce": [
        {"lr": .0005, "gamma": .98, "hidden": 64, "rollout_episodes": 8, "entropy_coeff": .01},
        {"lr": .0001, "gamma": .98, "hidden": 64, "rollout_episodes": 8, "entropy_coeff": .01},
        {"lr": .001, "gamma": .98, "hidden": 64, "rollout_episodes": 8, "entropy_coeff": .01},
        {"lr": .0005, "gamma": .95, "hidden": 64, "rollout_episodes": 8, "entropy_coeff": .01},
        {"lr": .0005, "gamma": .995, "hidden": 64, "rollout_episodes": 8, "entropy_coeff": .01},
        {"lr": .0005, "gamma": .98, "hidden": 128, "rollout_episodes": 8, "entropy_coeff": .01},
        {"lr": .0005, "gamma": .98, "hidden": 32, "rollout_episodes": 8, "entropy_coeff": .01},
        {"lr": .0005, "gamma": .98, "hidden": 64, "rollout_episodes": 4, "entropy_coeff": .01},
        {"lr": .0005, "gamma": .98, "hidden": 64, "rollout_episodes": 16, "entropy_coeff": .05},
        {"lr": .0002, "gamma": .95, "hidden": 128, "rollout_episodes": 8, "entropy_coeff": 0.0},
    ],
    "ppo": [
        {"lr": .0005, "gamma": .98, "hidden": 64, "clip_epsilon": .2, "update_epochs": 4, "gae_lambda": .95, "entropy_coeff": .01, "rollout_episodes": 8},
        {"lr": .0002, "gamma": .98, "hidden": 64, "clip_epsilon": .2, "update_epochs": 4, "gae_lambda": .95, "entropy_coeff": .01, "rollout_episodes": 8},
        {"lr": .0010, "gamma": .98, "hidden": 64, "clip_epsilon": .2, "update_epochs": 4, "gae_lambda": .95, "entropy_coeff": .01, "rollout_episodes": 8},
        {"lr": .0005, "gamma": .98, "hidden": 64, "clip_epsilon": .1, "update_epochs": 4, "gae_lambda": .95, "entropy_coeff": .01, "rollout_episodes": 8},
        {"lr": .0005, "gamma": .98, "hidden": 64, "clip_epsilon": .3, "update_epochs": 4, "gae_lambda": .95, "entropy_coeff": .01, "rollout_episodes": 8},
        {"lr": .0005, "gamma": .98, "hidden": 64, "clip_epsilon": .2, "update_epochs": 2, "gae_lambda": .95, "entropy_coeff": .01, "rollout_episodes": 8},
        {"lr": .0005, "gamma": .98, "hidden": 64, "clip_epsilon": .2, "update_epochs": 8, "gae_lambda": .95, "entropy_coeff": .01, "rollout_episodes": 8},
        {"lr": .0005, "gamma": .98, "hidden": 64, "clip_epsilon": .2, "update_epochs": 4, "gae_lambda": .90, "entropy_coeff": .01, "rollout_episodes": 8},
        {"lr": .0005, "gamma": .98, "hidden": 64, "clip_epsilon": .2, "update_epochs": 4, "gae_lambda": .95, "entropy_coeff": .05, "rollout_episodes": 4},
        {"lr": .0005, "gamma": .98, "hidden": 128, "clip_epsilon": .2, "update_epochs": 4, "gae_lambda": .95, "entropy_coeff": 0.0, "rollout_episodes": 16},
    ],
    "a2c": [
        {"lr": .0005, "gamma": .98, "hidden": 64, "gae_lambda": .95, "entropy_coeff": .01, "value_coeff": .5, "rollout_episodes": 8},
        {"lr": .0002, "gamma": .98, "hidden": 64, "gae_lambda": .95, "entropy_coeff": .01, "value_coeff": .5, "rollout_episodes": 8},
        {"lr": .0010, "gamma": .98, "hidden": 64, "gae_lambda": .95, "entropy_coeff": .01, "value_coeff": .5, "rollout_episodes": 8},
        {"lr": .0005, "gamma": .95, "hidden": 64, "gae_lambda": .95, "entropy_coeff": .01, "value_coeff": .5, "rollout_episodes": 8},
        {"lr": .0005, "gamma": .995, "hidden": 64, "gae_lambda": .95, "entropy_coeff": .01, "value_coeff": .5, "rollout_episodes": 8},
        {"lr": .0005, "gamma": .98, "hidden": 64, "gae_lambda": .90, "entropy_coeff": .01, "value_coeff": .5, "rollout_episodes": 8},
        {"lr": .0005, "gamma": .98, "hidden": 64, "gae_lambda": 1.0, "entropy_coeff": .01, "value_coeff": .5, "rollout_episodes": 8},
        {"lr": .0005, "gamma": .98, "hidden": 64, "gae_lambda": .95, "entropy_coeff": .05, "value_coeff": .5, "rollout_episodes": 4},
        {"lr": .0005, "gamma": .98, "hidden": 64, "gae_lambda": .95, "entropy_coeff": .01, "value_coeff": 1.0, "rollout_episodes": 8},
        {"lr": .0002, "gamma": .98, "hidden": 128, "gae_lambda": .95, "entropy_coeff": 0.0, "value_coeff": .25, "rollout_episodes": 16},
    ],
}


def planned_runs() -> list[dict]:
    records = []
    for algorithm, choices in CONFIGS.items():
        assert len(choices) == 10, algorithm
        assert len({json.dumps(x, sort_keys=True) for x in choices}) == 10, algorithm
        for i, settings in enumerate(choices, 1):
            records.append({"algorithm": algorithm, "run_id": f"{algorithm}_{i:02d}", **settings})
    assert len(records) == 40
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=tuple(CONFIGS), help="Only run these 10 configurations")
    parser.add_argument("--dry-run", action="store_true", help="Print configurations without training")
    args = parser.parse_args()
    root = Path("logs/training")
    root.mkdir(parents=True, exist_ok=True)
    runs = planned_runs()
    manifest = {
        "implementation": "native TorchRL 0.15 algorithms",
        "train_episodes_per_run": EPISODES,
        "validation_episodes_per_run": EVAL_EPISODES,
        "train_seed": SEED,
        "validation_seeds": [50000, 50099],
        "historical_signal_years": [2016, 2023],
        "heldout_year": 2024,
        "train_env_steps_per_run": EPISODES * 24,
        "runs": runs,
    }
    plan_path = root / "experiment_plan.json"
    if plan_path.exists():
        previous = json.loads(plan_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise RuntimeError("The pre-registered experimental plan has changed: do not silently overwrite it.")
    else:
        plan_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    selected = [r for r in runs if args.algorithm is None or r["algorithm"] == args.algorithm]
    for n, r in enumerate(selected, 1):
        result_path = root / r["algorithm"] / r["run_id"] / "summary.json"
        if result_path.exists():
            previous = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                previous.get("train_episodes") == EPISODES
                and previous.get("config", {}).get("algorithm") == r["algorithm"]
                and all(previous.get("config", {}).get(k) == v for k, v in r.items())
            ):
                print(f"[{n}/{len(selected)}] Already complete: {r['run_id']}", flush=True)
                continue
            raise RuntimeError(f"Conflicting existing result: {result_path}")
        command = [
            sys.executable, "-m", "training.experiment",
            "--algorithm", r["algorithm"], "--run-id", r["run_id"],
            "--episodes", str(EPISODES), "--eval-episodes", str(EVAL_EPISODES),
            "--seed", str(SEED),
        ]
        for key, value in r.items():
            if key in ("algorithm", "run_id"):
                continue
            command.extend(("--" + key.replace("_", "-"), str(value)))
        print(f"[{n}/{len(selected)}] {r['run_id']}: {r}", flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
    if args.dry_run:
        return
    completed = []
    for r in runs:
        path = root / r["algorithm"] / r["run_id"] / "summary.json"
        if path.is_file():
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get("train_episodes") == EPISODES:
                completed.append({
                    "algorithm": r["algorithm"],
                    "run_id": r["run_id"],
                    "validation_mean_reward": summary["validation_mean_reward"],
                    "validation_std_reward": summary["validation_std_reward"],
                    "validation_stochastic_mean_reward": summary["validation_stochastic_mean_reward"],
                    "last_100_train_mean": summary["last_100_train_mean"],
                    **{k: v for k, v in r.items() if k not in ("algorithm", "run_id")},
                })
    if completed:
        fieldnames = list(dict.fromkeys(key for row in completed for key in row))
        with (root / "sweep_summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(completed)
        print(f"Completed {len(completed)}/40 native TorchRL configurations", flush=True)


if __name__ == "__main__":
    main()
