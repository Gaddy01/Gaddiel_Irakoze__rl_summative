"""Activity 8 regression tests, using the existing project environment."""

import numpy as np
import pytest

from environment.observations import OBSERVATION_NAMES, ProjectStage
from evaluation.scenarios import ScenarioInvestmentEnv
from evaluation.stress_policies import select_action
from scripts.evaluate_stress import evaluate_episode, weighted_reward


def test_baseline_starts_with_53_observations():
    env = ScenarioInvestmentEnv("baseline")
    obs, info = env.reset(seed=12)
    assert env.observation_space.contains(obs)
    assert obs.shape == (53,)
    assert info["action_mask"].shape == (17,)
    env.close()


def test_downturn_is_not_revealed_as_hidden_feasibility():
    base = ScenarioInvestmentEnv("baseline")
    stress = ScenarioInvestmentEnv("downturn")
    base_obs, _ = base.reset(seed=19)
    stress_obs, _ = stress.reset(seed=19)
    # Observations identical before new research; true feasibility stays hidden.
    np.testing.assert_array_equal(base_obs, stress_obs)
    assert stress.sectors["agro_processing"].true_feasibility < base.sectors["agro_processing"].true_feasibility
    base.close()
    stress.close()


def test_downturn_reduces_market_demand():
    base = ScenarioInvestmentEnv("baseline")
    stress = ScenarioInvestmentEnv("downturn")
    base.reset(seed=19)
    stress.reset(seed=19)
    for _ in range(12):
        base.step(0)
        stress.step(0)
    assert stress.sectors["agro_processing"].demand_score < base.sectors["agro_processing"].demand_score
    base.close()
    stress.close()


def test_exit_strategies_choose_valid_exit_for_operating_company():
    env = ScenarioInvestmentEnv("baseline")
    obs, info = env.reset(seed=5)
    for act in [1, 5, 0, 9]:
        obs, _, _, _, info = env.step(act)
    assert env.sectors["agro_processing"].stage == ProjectStage.OPERATING
    for policy in ("quick_exit", "risk_aware"):
        a = select_action(policy, obs, info["action_mask"], np.random.default_rng(1))
        assert info["action_mask"][a]
    assert select_action("quick_exit", obs, info["action_mask"], np.random.default_rng(1)) == 13
    env.close()


def test_rule_baseline_policies_respect_mask():
    for policy in ("always_hold", "random_valid", "rule_based", "quick_exit", "risk_aware"):
        env = ScenarioInvestmentEnv("downturn")
        obs, info = env.reset(seed=17)
        rng = np.random.default_rng(112)
        for _ in range(24):
            action = select_action(policy, obs, info["action_mask"], rng)
            assert info["action_mask"][action]
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        env.close()


def test_complete_episode_returns_24_steps_and_consistent_rewards():
    result = evaluate_episode("downturn", "risk_aware", 101)
    assert result["steps"] == 24
    assert result["invalid_actions"] == 0
    total_parts = sum(result[k] for k in (
        "reward_financial", "reward_employment", "reward_import_substitution",
        "reward_risk", "reward_invalid",
    ))
    assert result["reward"] == pytest.approx(total_parts)


def test_hold_remains_neutral_in_both_scenarios():
    for scenario in ("baseline", "downturn"):
        result = evaluate_episode(scenario, "always_hold", 1000)
        assert result["reward"] == pytest.approx(0)
        assert result["net_worth"] == pytest.approx(100)


def test_deterministic_episode_with_seed():
    first = evaluate_episode("downturn", "quick_exit", 121)
    second = evaluate_episode("downturn", "quick_exit", 121)
    assert first == second


def test_reward_weight_sensitivity():
    c = {
        "financial": 0.08,
        "employment": 0.04,
        "import_substitution": 0.02,
        "risk_penalty": -0.001,
        "invalid_action_penalty": 0.0,
    }
    assert weighted_reward(c, "default") == pytest.approx(0.139)
    assert weighted_reward(c, "development_heavy") == pytest.approx(0.199)
    with pytest.raises(ValueError):
        weighted_reward(c, "unknown")


def test_observation_still_has_no_hidden_feasibility_field():
    assert len(OBSERVATION_NAMES) == 53
    assert not any("true_feasibility" in name for name in OBSERVATION_NAMES)
