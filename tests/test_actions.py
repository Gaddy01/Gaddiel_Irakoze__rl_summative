
import pytest

from environment.actions import (
    ACTIONS,
    ACTION_SPACE,
    ActionType,
    decode_action,
)


def test_action_space_size():
    assert ACTION_SPACE.n == 17
    assert len(ACTIONS) == 17


def test_action_ids_are_sequential():
    ids = [action.action_id for action in ACTIONS]
    assert ids == list(range(17))


def test_hold_action():
    action = decode_action(0)

    assert action.action_type == ActionType.HOLD
    assert action.sector_id is None


def test_pilot_action():
    action = decode_action(6)

    assert action.action_type == ActionType.PILOT
    assert action.sector_id == "textiles"


def test_each_sector_has_four_specific_actions():
    for action_type in (
        ActionType.STUDY,
        ActionType.PILOT,
        ActionType.EXPAND,
        ActionType.EXIT,
    ):
        matching = [
            action for action in ACTIONS
            if action.action_type == action_type
        ]

        assert len(matching) == 4


def test_invalid_action():
    with pytest.raises(ValueError):
        decode_action(17)
