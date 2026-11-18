
from dataclasses import dataclass
from math import isfinite

from environment.observations import INITIAL_CAPITAL


@dataclass(frozen=True)
class RewardWeights:
    """Initial reward coefficients."""

    financial: float = 1.0
    employment: float = 0.10
    import_substitution: float = 0.10
    risk: float = 0.005
    invalid_action: float = 0.02


@dataclass(frozen=True)
class RewardSnapshot:
    """
    Outcome information used to calculate a reward.

    These variables are supplied by the simulator.
    They are not necessarily all visible to the agent.
    """

    net_worth: float
    employment_index: float
    import_substitution_index: float
    risk_exposure_index: float


@dataclass(frozen=True)
class RewardBreakdown:
    """Individual contributions to the final reward."""

    financial: float
    employment: float
    import_substitution: float
    risk_penalty: float
    invalid_action_penalty: float
    total: float


def validate_reward_snapshot(
    snapshot: RewardSnapshot,
) -> None:
    """Verify that outcome values are valid."""

    fields = {
        "net_worth": snapshot.net_worth,
        "employment_index": snapshot.employment_index,
        "import_substitution_index":
            snapshot.import_substitution_index,
        "risk_exposure_index": snapshot.risk_exposure_index,
    }

    for name, value in fields.items():
        if not isfinite(value):
            raise ValueError(f"{name} must be finite.")

    if snapshot.net_worth < 0:
        raise ValueError("Net worth cannot be negative.")

    for name in (
        "employment_index",
        "import_substitution_index",
        "risk_exposure_index",
    ):
        value = fields[name]

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} must be between 0 and 1."
            )


def calculate_reward(
    before: RewardSnapshot,
    after: RewardSnapshot,
    action_valid: bool = True,
    weights: RewardWeights | None = None,
) -> RewardBreakdown:
    """
    Calculate the reward for one monthly transition.

    Financial performance is based on the change in
    total portfolio net worth, not cash expenditure alone.
    """

    validate_reward_snapshot(before)
    validate_reward_snapshot(after)

    if weights is None:
        weights = RewardWeights()

    financial_reward = (
        weights.financial
        * (after.net_worth - before.net_worth)
        / INITIAL_CAPITAL
    )

    employment_reward = (
        weights.employment
        * (
            after.employment_index
            - before.employment_index
        )
    )

    substitution_reward = (
        weights.import_substitution
        * (
            after.import_substitution_index
            - before.import_substitution_index
        )
    )

    risk_penalty = (
        -weights.risk * after.risk_exposure_index
    )

    invalid_penalty = (
        -weights.invalid_action
        if not action_valid
        else 0.0
    )

    total = (
        financial_reward
        + employment_reward
        + substitution_reward
        + risk_penalty
        + invalid_penalty
    )

    return RewardBreakdown(
        financial=float(financial_reward),
        employment=float(employment_reward),
        import_substitution=float(substitution_reward),
        risk_penalty=float(risk_penalty),
        invalid_action_penalty=float(invalid_penalty),
        total=float(total),
    )
