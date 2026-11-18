
from dataclasses import dataclass
from enum import Enum

from gymnasium import spaces


# Provisional sector labels.
# We will map them to specific HS4 products later.
SECTORS = {
    "agro_processing": "Agro-processing",
    "textiles": "Textiles",
    "construction_materials": "Construction Materials",
    "pharmaceuticals": "Pharmaceuticals",
}


class ActionType(str, Enum):
    HOLD = "hold"
    STUDY = "study"
    PILOT = "pilot"
    EXPAND = "expand"
    EXIT = "exit"


@dataclass(frozen=True)
class InvestmentAction:
    action_id: int
    action_type: ActionType
    sector_id: str | None
    description: str


def build_action_catalog() -> tuple[InvestmentAction, ...]:
    actions = [
        InvestmentAction(
            action_id=0,
            action_type=ActionType.HOLD,
            sector_id=None,
            description="Hold capital",
        )
    ]

    operations = [
        (ActionType.STUDY, "Conduct feasibility study"),
        (ActionType.PILOT, "Fund pilot"),
        (ActionType.EXPAND, "Expand investment"),
        (ActionType.EXIT, "Exit investment"),
    ]

    for action_type, label in operations:
        for sector_id, sector_name in SECTORS.items():
            actions.append(
                InvestmentAction(
                    action_id=len(actions),
                    action_type=action_type,
                    sector_id=sector_id,
                    description=f"{label}: {sector_name}",
                )
            )

    return tuple(actions)


ACTIONS = build_action_catalog()

# Fixed, discrete action space shared by all RL models.
ACTION_SPACE = spaces.Discrete(len(ACTIONS))


def decode_action(action_id: int) -> InvestmentAction:
    """Convert a numeric action into its investment meaning."""

    if not ACTION_SPACE.contains(action_id):
        raise ValueError(f"Invalid action ID: {action_id}")

    return ACTIONS[int(action_id)]
