"""Compare Hold, valid-random, and rule-based investment policies.

Run from repository root:
    uv run python -m scripts.evaluate_baselines --episodes 100 --seed 1000
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from environment.actions import ACTIONS
from environment.custom_env import IndustrialInvestmentEnv
from evaluation.baselines import POLICY_NAMES, choose_action


EPISODE_FIELDS = (
    "policy", "seed", "episode_reward", "final_net_worth",
    "employment_index", "import_substitution_index", "risk_exposure_index",
    "invalid_actions", "hold_actions", "study_actions", "pilot_actions",
    "expand_actions", "exit_actions", "terminated", "steps",
)
SUMMARY_FIELDS = (
    "policy", "episodes", "mean_reward", "std_reward", "median_reward",
    "positive_reward_rate", "mean_net_worth", "mean_employment",
    "mean_import_substitution", "mean_invalid_actions",
    "mean_studies", "mean_pilots", "mean_expansions", "mean_exits",
)


def run_episode(policy: str, seed: int) -> dict:
    """Evaluate one policy using only the observation and valid-action mask."""
    env = IndustrialInvestmentEnv()
    action_rng = np.random.default_rng(seed + 100_000)
    counts = Counter()
    total_reward = 0.0
    steps = 0
    terminated = False

    try:
        observation, info = env.reset(seed=seed)
        for _ in range(24):
            action = choose_action(
                policy, observation, info["action_mask"], action_rng
            )
            counts[ACTIONS[action].action_type.value] += 1
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            if not info["action_valid"]:
                counts["invalid"] += 1
            if terminated or truncated:
                break
        final = info["portfolio"]
        return {
            "policy": policy,
            "seed": seed,
            "episode_reward": total_reward,
            "final_net_worth": final["net_worth"],
            "employment_index": final["employment_index"],
            "import_substitution_index": final["import_substitution_index"],
            "risk_exposure_index": final["risk_exposure_index"],
            "invalid_actions": counts["invalid"],
            "hold_actions": counts["hold"],
            "study_actions": counts["study"],
            "pilot_actions": counts["pilot"],
            "expand_actions": counts["expand"],
            "exit_actions": counts["exit"],
            "terminated": terminated,
            "steps": steps,
        }
    finally:
        env.close()


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    for policy in POLICY_NAMES:
        group = [r for r in rows if r["policy"] == policy]
        rewards = np.asarray([r["episode_reward"] for r in group], dtype=float)
        get_mean = lambda field: float(np.mean([r[field] for r in group]))
        summaries.append({
            "policy": policy,
            "episodes": len(group),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
            "median_reward": float(np.median(rewards)),
            "positive_reward_rate": float(np.mean(rewards > 0)),
            "mean_net_worth": get_mean("final_net_worth"),
            "mean_employment": get_mean("employment_index"),
            "mean_import_substitution": get_mean("import_substitution_index"),
            "mean_invalid_actions": get_mean("invalid_actions"),
            "mean_studies": get_mean("study_actions"),
            "mean_pilots": get_mean("pilot_actions"),
            "mean_expansions": get_mean("expand_actions"),
            "mean_exits": get_mean("exit_actions"),
        })
    return summaries


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Investment baseline sanity checks")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("logs/baselines"))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    seeds = range(args.seed, args.seed + args.episodes)
    rows = [
        run_episode(policy, seed)
        for policy in POLICY_NAMES
        for seed in seeds
    ]
    summaries = summarize(rows)
    write_csv(args.output_dir / "episodes.csv", EPISODE_FIELDS, rows)
    write_csv(args.output_dir / "summary.csv", SUMMARY_FIELDS, summaries)

    print("\nINDUSTRIAL INVESTMENT RL — BASELINE EVALUATION")
    print(f"Episodes per policy: {args.episodes}")
    print(f"Seeds: {args.seed} ... {args.seed + args.episodes - 1}")
    print("\nPolicy          Mean reward   Std reward   Net worth   Employment  Import subst.  Invalid")
    print("-" * 98)
    for s in summaries:
        print(
            f"{s['policy']:<16} {s['mean_reward']:>+10.4f} "
            f"{s['std_reward']:>11.4f} {s['mean_net_worth']:>11.2f} "
            f"{s['mean_employment']:>12.3f} "
            f"{s['mean_import_substitution']:>14.3f} "
            f"{s['mean_invalid_actions']:>8.2f}"
        )
    print("\nAverage action counts per episode:")
    for s in summaries:
        print(
            f"  {s['policy']:<15} studies={s['mean_studies']:.2f}, "
            f"pilots={s['mean_pilots']:.2f}, "
            f"expansions={s['mean_expansions']:.2f}, "
            f"exits={s['mean_exits']:.2f}"
        )
    print(f"\nSaved: {args.output_dir / 'episodes.csv'}")
    print(f"Saved: {args.output_dir / 'summary.csv'}")
    print("\nNOTE: Equal seeds give comparable starting scenarios, but action-dependent RNG")
    print("consumption can produce different later market-shock sequences across policies.")


if __name__ == "__main__":
    main()
