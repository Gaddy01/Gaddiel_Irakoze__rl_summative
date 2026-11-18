
from dataclasses import replace

import pytest

from environment.rewards import (
    RewardSnapshot,
    calculate_reward,
)


@pytest.fixture
def initial_state():
    return RewardSnapshot(
        net_worth=100.0,
        employment_index=0.0,
        import_substitution_index=0.0,
        risk_exposure_index=0.0,
    )


def test_no_change_no_reward(initial_state):
    result = calculate_reward(
        initial_state,
        initial_state,
    )

    assert result.total == pytest.approx(0.0)


def test_research_expense_reduces_reward(initial_state):
    after = replace(
        initial_state,
        net_worth=97.0,
    )

    result = calculate_reward(initial_state, after)

    assert result.total == pytest.approx(-0.03)


def test_capital_reallocation_is_not_profit(initial_state):
    # The investor transfers cash into an equally valued
    # investment asset. Total wealth remains unchanged.
    after = replace(
        initial_state,
        net_worth=100.0,
    )

    result = calculate_reward(initial_state, after)

    assert result.financial == pytest.approx(0.0)
    assert result.total == pytest.approx(0.0)


def test_financial_gain(initial_state):
    after = replace(
        initial_state,
        net_worth=108.0,
    )

    result = calculate_reward(initial_state, after)

    assert result.financial == pytest.approx(0.08)
    assert result.total == pytest.approx(0.08)


def test_employment_increase(initial_state):
    after = replace(
        initial_state,
        employment_index=0.5,
    )

    result = calculate_reward(initial_state, after)

    assert result.employment == pytest.approx(0.05)


def test_employment_decrease_is_penalized():
    before = RewardSnapshot(
        net_worth=100.0,
        employment_index=0.5,
        import_substitution_index=0.0,
        risk_exposure_index=0.0,
    )

    after = replace(
        before,
        employment_index=0.2,
    )

    result = calculate_reward(before, after)

    assert result.employment == pytest.approx(-0.03)


def test_import_substitution_increase(initial_state):
    after = replace(
        initial_state,
        import_substitution_index=0.3,
    )

    result = calculate_reward(initial_state, after)

    assert result.import_substitution == pytest.approx(0.03)


def test_risk_penalty(initial_state):
    after = replace(
        initial_state,
        risk_exposure_index=0.4,
    )

    result = calculate_reward(initial_state, after)

    assert result.risk_penalty == pytest.approx(-0.002)
    assert result.total == pytest.approx(-0.002)


def test_invalid_action_penalty(initial_state):
    result = calculate_reward(
        initial_state,
        initial_state,
        action_valid=False,
    )

    assert result.invalid_action_penalty == pytest.approx(
        -0.02
    )


def test_combined_reward_breakdown(initial_state):
    after = RewardSnapshot(
        net_worth=108.0,
        employment_index=0.4,
        import_substitution_index=0.2,
        risk_exposure_index=0.2,
    )

    result = calculate_reward(initial_state, after)

    component_sum = (
        result.financial
        + result.employment
        + result.import_substitution
        + result.risk_penalty
        + result.invalid_action_penalty
    )

    assert result.total == pytest.approx(component_sum)
    assert result.total == pytest.approx(0.139)


def test_out_of_range_indicator_rejected(initial_state):
    invalid_state = replace(
        initial_state,
        employment_index=1.5,
    )

    with pytest.raises(ValueError):
        calculate_reward(initial_state, invalid_state)


def test_nonfinite_value_rejected(initial_state):
    invalid_state = replace(
        initial_state,
        net_worth=float("nan"),
    )

    with pytest.raises(ValueError):
        calculate_reward(initial_state, invalid_state)
