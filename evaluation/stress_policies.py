"""Additional diagnostic policies that use observations and valid-action mask only."""

import numpy as np

from environment.actions import SECTORS
from evaluation.baselines import choose_action, FEATURE_INDEX

POLICIES = ("always_hold", "random_valid", "rule_based", "quick_exit", "risk_aware")
SECTOR_IDS = tuple(SECTORS)


def select_action(policy: str, observation: np.ndarray,
                  action_mask: np.ndarray, rng: np.random.Generator) -> int:
    """Choose a valid action without accessing the environment's hidden state."""
    if policy in ("always_hold", "random_valid", "rule_based"):
        return choose_action(policy, observation, action_mask, rng)
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")

    mask = np.asarray(action_mask, dtype=bool)
    if mask.shape != (17,) or not mask[0]:
        raise ValueError("Invalid action mask")

    for i, sector_id in enumerate(SECTOR_IDS):
        operating = observation[FEATURE_INDEX[f"{sector_id}.stage_operating"]] > 0.5
        exit_action = 13 + i
        if not operating or not mask[exit_action]:
            continue
        if policy == "quick_exit":
            return exit_action  # Sell next time an operating asset is observed.
        # A deliberately simple, observation-based risk trigger: sell an
        # operating company after month 8 if its demand signal weakens.
        month_fraction = float(observation[FEATURE_INDEX["month_fraction"]])
        demand = float(observation[FEATURE_INDEX[f"{sector_id}.demand_score"]])
        if policy == "risk_aware" and month_fraction >= 8 / 24 and demand < 0.70:
            return exit_action

    return choose_action("rule_based", observation, mask, rng)
