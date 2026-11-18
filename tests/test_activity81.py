"""Activity 8.1: observation and company-operation regression tests."""

import numpy as np
import pytest

from environment.observations import OBSERVATION_NAMES, ProjectStage
from environment.custom_env import IndustrialInvestmentEnv
from evaluation.scenarios import ScenarioInvestmentEnv


def _operating_env(seed=5, env_type=IndustrialInvestmentEnv):
    env = env_type()
    observation, info = env.reset(seed=seed)
    for action_id in [1, 5, 0, 9]:
        observation, _, _, _, info = env.step(action_id)
        assert info["action_valid"]
    return env, observation, info


def test_53_observations_and_named_globals():
    env = IndustrialInvestmentEnv()
    observation, info = env.reset(seed=10)
    assert observation.shape == (53,)
    assert len(OBSERVATION_NAMES) == 53
    assert env.observation_space.contains(observation)
    assert OBSERVATION_NAMES[3:5] == (
        "employment_index", "import_substitution_index"
    )
    assert observation[3] == pytest.approx(0)
    assert observation[4] == pytest.approx(0)
    env.close()


def test_global_economic_observations_match_public_outcomes():
    env, observation, info = _operating_env()
    assert observation[3] == pytest.approx(info["portfolio"]["employment_index"], abs=0.001)
    assert observation[4] == pytest.approx(info["portfolio"]["import_substitution_index"], abs=0.001)
    assert observation[3] > 0
    assert observation[4] > 0
    env.close()


def test_selling_ownership_does_not_close_company():
    env, _, _ = _operating_env()
    sector = env.sectors["agro_processing"]
    before = env._reward_snapshot()
    assert sector.stage == ProjectStage.OPERATING
    env._apply_action(13)  # Precisely at the transaction point, before market evolution.
    after = env._reward_snapshot()
    assert sector.stage == ProjectStage.EXITED
    assert sector.invested_capital == 0
    assert sector.asset_value == 0
    assert sector.post_exit_capacity > 0
    assert before.employment_index == pytest.approx(after.employment_index)
    assert before.import_substitution_index == pytest.approx(after.import_substitution_index)
    assert after.risk_exposure_index == 0
    env.close()


def test_sold_company_changes_with_market_demand():
    env, _, _ = _operating_env()
    env.step(13)
    sector = env.sectors["agro_processing"]
    assert sector.stage == ProjectStage.EXITED
    original = env._reward_snapshot()
    sector.demand_score *= 0.60
    stressed = env._reward_snapshot()
    assert stressed.employment_index < original.employment_index
    assert stressed.import_substitution_index < original.import_substitution_index
    assert stressed.risk_exposure_index == 0
    assert stressed.net_worth == pytest.approx(original.net_worth)
    env.close()


def test_owned_company_also_changes_with_market_demand():
    env, _, _ = _operating_env()
    original = env._reward_snapshot()
    env.sectors["agro_processing"].demand_score *= 0.60
    stressed = env._reward_snapshot()
    assert stressed.employment_index < original.employment_index
    assert stressed.import_substitution_index < original.import_substitution_index
    env.close()


def test_downturn_affects_sold_company_productive_outcomes():
    env = ScenarioInvestmentEnv("downturn")
    env.reset(seed=15)
    for action in [1, 5, 0, 9, 13]:
        env.step(action)
    after_sale = env._reward_snapshot()
    assert after_sale.employment_index > 0
    assert after_sale.import_substitution_index > 0
    for _ in range(12):
        _, _, _, _, info = env.step(0)
    later = env._reward_snapshot()
    assert later.employment_index < after_sale.employment_index
    assert later.import_substitution_index < after_sale.import_substitution_index
    assert later.risk_exposure_index == 0
    assert later.net_worth == pytest.approx(after_sale.net_worth)
    assert info["portfolio"]["month"] == 17
    env.close()


def test_pilot_exit_does_not_create_operating_company():
    env = IndustrialInvestmentEnv()
    env.reset(seed=6)
    for action in [1, 5, 13]:
        _, _, _, _, info = env.step(action)
        assert info["action_valid"]
    s = env.sectors["agro_processing"]
    assert s.stage == ProjectStage.EXITED
    assert s.post_exit_capacity == pytest.approx(0)
    assert info["portfolio"]["employment_index"] == pytest.approx(0)
    assert info["portfolio"]["import_substitution_index"] == pytest.approx(0)
    env.close()


def test_reset_discards_prior_company_and_economic_outcomes():
    env, _, _ = _operating_env()
    env.step(13)
    obs, info = env.reset(seed=22)
    assert obs.shape == (53,)
    assert obs[3] == pytest.approx(0)
    assert obs[4] == pytest.approx(0)
    assert info["portfolio"]["employment_index"] == 0
    assert env.sectors["agro_processing"].post_exit_capacity == 0
    env.close()


def test_no_hidden_feasibility_field_in_observation():
    assert len(OBSERVATION_NAMES) == 53
    assert not any("true_feasibility" in name for name in OBSERVATION_NAMES)
    env = IndustrialInvestmentEnv()
    obs, info = env.reset(seed=21)
    assert "true_feasibility" not in str(info)
    assert obs.dtype == np.float32
    env.close()


def test_53_feature_rollout_remains_valid_and_terminates():
    env = ScenarioInvestmentEnv("downturn")
    obs, info = env.reset(seed=1001)
    for month in range(24):
        # Fixed valid action, independent of the simulator's hidden state.
        obs, reward, terminated, truncated, info = env.step(0)
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        assert terminated == (month == 23)
        assert not truncated
    env.close()
