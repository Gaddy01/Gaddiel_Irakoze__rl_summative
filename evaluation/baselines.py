"""Transparent baseline policies for the industrial-investment environment.

Policies may access ONLY the encoded observation and current valid-action mask.
They deliberately do not access env.sectors, true_feasibility, or future shocks.
"""

from typing import Literal

import numpy as np

from environment.actions import SECTORS
from environment.observations import OBSERVATION_NAMES

PolicyName = Literal["always_hold", "random_valid", "rule_based"]
POLICY_NAMES: tuple[PolicyName, ...] = (
    "always_hold", "random_valid", "rule_based"
)
SECTOR_IDS = tuple(SECTORS)
FEATURE_INDEX = {name: i for i, name in enumerate(OBSERVATION_NAMES)}


def _sector_value(observation: np.ndarray, sector_id: str, feature: str) -> float:
    return float(observation[FEATURE_INDEX[f"{sector_id}.{feature}"]])


def _sector_score(observation: np.ndarray, sector_id: str) -> float:
    """Simple, openly specified screening heuristic (NOT a learned model)."""
    return (
        0.55 * _sector_value(observation, sector_id, "demand_score")
        + 0.25 * _sector_value(observation, sector_id, "growth_signal")
        + 0.20 * _sector_value(observation, sector_id, "supplier_concentration")
    )


def choose_action(
    policy: PolicyName,
    observation: np.ndarray,
    action_mask: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """Return a valid action ID, using no hidden simulator state.

    The rule-based policy prioritizes completing worthwhile pilots, then funding
    researched projects, then studying a promising yet unstudied product group.
    """
    mask = np.asarray(action_mask, dtype=bool)
    if mask.shape != (17,) or not mask.any():
        raise ValueError("Expected a nonempty boolean mask of 17 actions.")

    if policy == "always_hold":
        if not mask[0]:
            raise ValueError("Hold action unexpectedly unavailable.")
        return 0

    if policy == "random_valid":
        return int(rng.choice(np.flatnonzero(mask)))

    if policy != "rule_based":
        raise ValueError(f"Unknown baseline policy: {policy}")

    # Action IDs: hold=0; study=1..4; pilot=5..8;
    # expand=9..12; exit=13..16.
    order = sorted(SECTOR_IDS, key=lambda s: _sector_score(observation, s), reverse=True)
    active_count = sum(
        _sector_value(observation, sid, "stage_pilot") > 0.5
        or _sector_value(observation, sid, "stage_operating") > 0.5
        for sid in SECTOR_IDS
    )

    # First complete matured pilots when research estimate was promising.
    for sid in order:
        i = SECTOR_IDS.index(sid)
        is_mature_pilot = (
            _sector_value(observation, sid, "stage_pilot") > 0.5
            and _sector_value(observation, sid, "project_progress") >= 0.999
        )
        estimate = _sector_value(observation, sid, "feasibility_estimate")
        if is_mature_pilot and estimate >= 0.58 and mask[9 + i]:
            return 9 + i  # Expand
        if is_mature_pilot and estimate < 0.58 and mask[13 + i]:
            return 13 + i  # Sell an unpromising pilot

    # Limit active projects to two, so the heuristic retains a cash reserve.
    if active_count < 2:
        for sid in order:
            i = SECTOR_IDS.index(sid)
            estimate = _sector_value(observation, sid, "feasibility_estimate")
            if estimate >= 0.58 and mask[5 + i]:
                return 5 + i  # Fund a studied sector's pilot

        for sid in order:
            i = SECTOR_IDS.index(sid)
            if _sector_score(observation, sid) >= 0.55 and mask[1 + i]:
                return 1 + i  # Research a promising unstudied sector

    return 0  # Preserve cash if no preferred action is available.
