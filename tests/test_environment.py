from dataclasses import replace

import numpy as np
import pytest

from environment.custom_env import IndustrialInvestmentEnv
from environment.observations import ProjectStage


def test_reset_observation_and_initial_mask():
    env = IndustrialInvestmentEnv()
    observation, info = env.reset(seed=42)
    assert observation.shape == (53,)
    assert env.observation_space.contains(observation)
    assert info["portfolio"]["cash"] == 100.0
    assert info["action_mask"].shape == (17,)
    assert info["action_mask"][0]
    assert info["action_mask"][1]
    assert not info["action_mask"][5]


def test_seed_reproducibility():
    env_a = IndustrialInvestmentEnv()
    env_b = IndustrialInvestmentEnv()
    a, _ = env_a.reset(seed=123)
    b, _ = env_b.reset(seed=123)
    np.testing.assert_array_equal(a, b)
    for action_id in [1, 5, 0, 9, 0, 13]:
        obs_a, reward_a, _, _, info_a = env_a.step(action_id)
        obs_b, reward_b, _, _, info_b = env_b.step(action_id)
        np.testing.assert_array_equal(obs_a, obs_b)
        assert reward_a == reward_b
        assert info_a["portfolio"] == info_b["portfolio"]


def test_study_cost_and_uncertainty_reduction():
    env = IndustrialInvestmentEnv()
    env.reset(seed=5)
    _, _, _, _, info = env.step(1)
    assert info["action_valid"]
    assert env.cash == pytest.approx(97.0)
    assert env.sectors["agro_processing"].stage == ProjectStage.STUDIED
    assert env.sectors["agro_processing"].uncertainty == pytest.approx(0.25)
    assert info["reward_breakdown"]["financial"] == pytest.approx(-0.03)
    assert "true_feasibility" not in str(info)


def test_study_cannot_be_repeated():
    env = IndustrialInvestmentEnv()
    env.reset(seed=5)
    env.step(1)
    _, _, _, _, info = env.step(1)
    assert not info["action_valid"]
    assert env.cash == pytest.approx(97.0)
    assert info["reward_breakdown"]["invalid_action_penalty"] == pytest.approx(-0.02)


def test_pilot_requires_study():
    env = IndustrialInvestmentEnv()
    env.reset(seed=5)
    _, _, _, _, info = env.step(5)
    assert not info["action_valid"]
    assert env.sectors["agro_processing"].stage == ProjectStage.UNSTUDIED
    assert env.cash == pytest.approx(100.0)


def test_pilot_takes_two_months():
    env = IndustrialInvestmentEnv()
    env.reset(seed=5)
    env.step(1)
    _, _, _, _, info = env.step(5)
    assert info["action_valid"]
    assert env.cash == pytest.approx(82.0)
    assert env.sectors["agro_processing"].stage == ProjectStage.PILOT
    assert env.sectors["agro_processing"].project_progress == pytest.approx(0.5)
    assert not info["action_mask"][9]
    _, _, _, _, info = env.step(0)
    assert env.sectors["agro_processing"].project_progress == pytest.approx(1.0)
    assert info["action_mask"][9]


def test_expand_only_after_completed_pilot():
    env = IndustrialInvestmentEnv()
    env.reset(seed=5)
    env.step(1)
    env.step(5)
    _, _, _, _, info = env.step(9)  # Not mature at the start of the step.
    assert not info["action_valid"]
    assert env.sectors["agro_processing"].stage == ProjectStage.PILOT
    _, _, _, _, info = env.step(9)  # Now mature.
    assert info["action_valid"]
    assert env.sectors["agro_processing"].stage == ProjectStage.OPERATING
    assert env.sectors["agro_processing"].invested_capital == pytest.approx(35.0)


def test_operating_project_generates_development_indicators():
    env = IndustrialInvestmentEnv()
    env.reset(seed=5)
    for action_id in [1, 5, 0, 9]:
        _, _, _, _, info = env.step(action_id)
    state = info["portfolio"]
    assert state["employment_index"] > 0
    assert state["import_substitution_index"] > 0
    assert state["net_worth"] > 0


def test_exit_preserves_development_impact():
    env = IndustrialInvestmentEnv()
    env.reset(seed=5)

    for action_id in [1, 5, 0, 9]:
        env.step(action_id)

    before = env.public_state()

    _, _, _, _, info = env.step(13)
    after = info["portfolio"]

    assert info["action_valid"]
    assert env.sectors["agro_processing"].stage == (ProjectStage.EXITED)

    # Investor ownership ends.
    assert env.sectors["agro_processing"].asset_value == pytest.approx(0.0)

    # The underlying company's productive: economic contribution is preserved.
    assert after["employment_index"] > 0
    assert after["import_substitution_index"] > 0

    assert after["employment_index"] == pytest.approx(before["employment_index"], abs=0.02)
    assert after["import_substitution_index"] == pytest.approx(before["import_substitution_index"], abs=0.02)

    # Investor no longer bears exposure to this asset.
    assert after["risk_exposure_index"] == pytest.approx(0.0)

    env.close()


def test_invalid_action_consumes_a_month():
    env = IndustrialInvestmentEnv()
    env.reset(seed=5)
    _, _, _, _, info = env.step(9)
    assert not info["action_valid"]
    assert env.month == 1
    assert info["reward_breakdown"]["invalid_action_penalty"] == pytest.approx(-0.02)


def test_episode_ends_after_24_months():
    env = IndustrialInvestmentEnv()
    env.reset(seed=5)
    for month in range(24):
        obs, reward, terminated, truncated, info = env.step(0)
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert not truncated
        assert terminated == (month == 23)
    with pytest.raises(RuntimeError):
        env.step(0)


def test_reset_allows_new_episode():
    env = IndustrialInvestmentEnv()
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset(seed=5)
    for _ in range(24):
        env.step(0)
    obs, info = env.reset(seed=6)
    assert env.month == 0
    assert env.cash == pytest.approx(100.0)
    assert env.observation_space.contains(obs)
    assert not env._done
