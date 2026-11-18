"""Activity 8: compare baseline behavior, exit strategies, and a synthetic downturn.

Run from project root:
    uv run python -m scripts.evaluate_stress --episodes 10 --seed 1000
    uv run python -m scripts.evaluate_stress --episodes 100 --seed 1000

Results are saved separately from Activity 7 in logs/activity8/.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from environment.actions import ACTIONS
from evaluation.scenarios import SCENARIOS, ScenarioInvestmentEnv
from evaluation.stress_policies import POLICIES, select_action

EPISODE_COLUMNS = (
    "scenario", "policy", "seed", "reward", "net_worth", "employment",
    "import_substitution", "invalid_actions", "exits", "steps",
    "reward_financial", "reward_employment", "reward_import_substitution",
    "reward_risk", "reward_invalid", "reward_financial_heavy",
    "reward_development_heavy", "reward_risk_averse",
)
SUMMARY_COLUMNS = (
    "scenario", "policy", "episodes", "mean_reward", "std_reward",
    "positive_reward_rate", "mean_net_worth", "profitable_rate",
    "mean_employment", "mean_import_substitution", "mean_exits", "mean_invalid",
)
SENSITIVITY_COLUMNS = (
    "scenario", "policy", "weighting", "mean_reweighted_reward",
)


def weighted_reward(components: dict[str, float], weighting: str) -> float:
    """Re-score *the same trajectory*; no policy retraining or changed actions."""
    multipliers = {
        "default": (1.0, 1.0, 1.0, 1.0),
        "financial_heavy": (1.25, 0.5, 0.5, 2.0),
        "development_heavy": (1.0, 2.0, 2.0, 1.0),
        "risk_averse": (1.0, 1.0, 1.0, 4.0),
    }
    if weighting not in multipliers:
        raise ValueError(f"Unknown weighting: {weighting}")
    f, e, s, r = multipliers[weighting]
    return (
        f * components["financial"]
        + e * components["employment"]
        + s * components["import_substitution"]
        + r * components["risk_penalty"]
        + components["invalid_action_penalty"]
    )


def evaluate_episode(scenario: str, policy: str, seed: int) -> dict:
    """Run an entire episode while respecting the policy information boundary."""
    env = ScenarioInvestmentEnv(scenario=scenario)
    rng = np.random.default_rng(seed + 100_000)
    total_reward = 0.0
    components = {
        "financial": 0.0,
        "employment": 0.0,
        "import_substitution": 0.0,
        "risk_penalty": 0.0,
        "invalid_action_penalty": 0.0,
    }
    counts = Counter()
    steps = 0
    try:
        obs, info = env.reset(seed=seed)
        for _ in range(24):
            action = select_action(policy, obs, info["action_mask"], rng)
            counts[ACTIONS[action].action_type.value] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            for key in components:
                components[key] += info["reward_breakdown"][key]
            if not info["action_valid"]:
                counts["invalid"] += 1
            steps += 1
            if terminated or truncated:
                break
        if steps != 24 or not terminated:
            raise RuntimeError(f"Unexpected incomplete episode: {steps}")
        if not np.isclose(total_reward, sum(components.values()), atol=1e-8):
            raise AssertionError("Reward breakdown does not sum to episode reward")
        final = info["portfolio"]
        row = {
            "scenario": scenario,
            "policy": policy,
            "seed": seed,
            "reward": total_reward,
            "net_worth": final["net_worth"],
            "employment": final["employment_index"],
            "import_substitution": final["import_substitution_index"],
            "invalid_actions": counts["invalid"],
            "exits": counts["exit"],
            "steps": steps,
            "reward_financial": components["financial"],
            "reward_employment": components["employment"],
            "reward_import_substitution": components["import_substitution"],
            "reward_risk": components["risk_penalty"],
            "reward_invalid": components["invalid_action_penalty"],
            "reward_financial_heavy": weighted_reward(components, "financial_heavy"),
            "reward_development_heavy": weighted_reward(components, "development_heavy"),
            "reward_risk_averse": weighted_reward(components, "risk_averse"),
        }
        return row
    finally:
        env.close()


def summarize(rows: list[dict]) -> list[dict]:
    result = []
    for scenario in SCENARIOS:
        for policy in POLICIES:
            group = [r for r in rows if r["scenario"] == scenario and r["policy"] == policy]
            if not group:
                continue
            rewards = np.asarray([r["reward"] for r in group], dtype=float)
            avg = lambda field: float(np.mean([r[field] for r in group]))
            result.append({
                "scenario": scenario, "policy": policy, "episodes": len(group),
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
                "positive_reward_rate": float(np.mean(rewards > 0)),
                "mean_net_worth": avg("net_worth"),
                "profitable_rate": float(np.mean([r["net_worth"] >= 100.0 for r in group])),
                "mean_employment": avg("employment"),
                "mean_import_substitution": avg("import_substitution"),
                "mean_exits": avg("exits"),
                "mean_invalid": avg("invalid_actions"),
            })
    return result


def sensitivity_summary(rows: list[dict]) -> list[dict]:
    result = []
    weightings = {
        "default": "reward",
        "financial_heavy": "reward_financial_heavy",
        "development_heavy": "reward_development_heavy",
        "risk_averse": "reward_risk_averse",
    }
    for scenario in SCENARIOS:
        for policy in POLICIES:
            group = [r for r in rows if r["scenario"] == scenario and r["policy"] == policy]
            if not group:
                continue
            for weighting, field in weightings.items():
                result.append({
                    "scenario": scenario, "policy": policy, "weighting": weighting,
                    "mean_reweighted_reward": float(np.mean([r[field] for r in group])),
                })
    return result


def save_csv(path: Path, fieldnames: tuple, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Activity 8: synthetic downturn stress test")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("logs/activity8"))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")
    rows = [
        evaluate_episode(scenario, policy, seed)
        for scenario in SCENARIOS
        for policy in POLICIES
        for seed in range(args.seed, args.seed + args.episodes)
    ]
    summaries = summarize(rows)
    save_csv(args.output_dir / "episodes.csv", EPISODE_COLUMNS, rows)
    save_csv(args.output_dir / "summary.csv", SUMMARY_COLUMNS, summaries)
    save_csv(args.output_dir / "reward_sensitivity.csv", SENSITIVITY_COLUMNS,
             sensitivity_summary(rows))

    print("\nACTIVITY 8 — ECONOMIC SIMULATION REFINEMENT")
    print(f"Episodes per scenario/policy: {args.episodes}")
    print(f"Seed range: {args.seed}–{args.seed + args.episodes - 1}")
    print(f"Total episodes: {len(rows)}")
    print("\nScenario       Policy          Reward    Net worth   Employment  Substitution  Exits")
    print("-" * 88)
    for s in summaries:
        print(f"{s['scenario']:<14} {s['policy']:<15} "
              f"{s['mean_reward']:+.4f}    {s['mean_net_worth']:8.2f}     "
              f"{s['mean_employment']:.3f}       {s['mean_import_substitution']:.3f}     "
              f"{s['mean_exits']:.2f}")
    print("\nSaved: " + str(args.output_dir / "episodes.csv"))
    print("Saved: " + str(args.output_dir / "summary.csv"))
    print("Saved: " + str(args.output_dir / "reward_sensitivity.csv"))
    print("\nNOTE: Downturn is an illustrative counterfactual; not estimated Rwanda data.")
    print("NOTE: Reward sensitivity re-scores FIXED trajectories, without re-optimizing policies.")
    print("NOTE: Same seeds do not ensure identical future random shocks across different actions.")


if __name__ == "__main__":
    main()
