"""Tests for baseline policies and short environment episodes."""

import numpy as np

from environment.custom_env import IndustrialInvestmentEnv
from environment.observations import OBSERVATION_NAMES
from evaluation.baselines import POLICY_NAMES, choose_action
from scripts.evaluate_baselines import run_episode


def test_always_hold_action():
    env = IndustrialInvestmentEnv()
    obs, info = env.reset(seed=7)
    assert choose_action("always_hold", obs, info["action_mask"], np.random.default_rng(1)) == 0
    env.close()


def test_random_policy_never_selects_invalid_action():
    env = IndustrialInvestmentEnv()
    obs, info = env.reset(seed=7)
    rng = np.random.default_rng(9)
    for _ in range(100):
        action = choose_action("random_valid", obs, info["action_mask"], rng)
        assert info["action_mask"][action]
    env.close()


def test_rule_based_starts_by_studying_strongest_sector():
    env = IndustrialInvestmentEnv()
    obs, info = env.reset(seed=7)
    action = choose_action("rule_based", obs, info["action_mask"], np.random.default_rng(9))
    assert action == 3  # Construction materials has highest initial screen score.
    env.close()


def test_rule_based_uses_only_observations_and_mask():
    env = IndustrialInvestmentEnv()
    obs, info = env.reset(seed=7)
    action = choose_action("rule_based", obs.copy(), info["action_mask"].copy(), np.random.default_rng(9))
    assert info["action_mask"][action]
    assert all("true_feasibility" not in name for name in OBSERVATION_NAMES)
    env.close()


def test_policy_rollouts_complete_without_invalid_actions():
    for policy in POLICY_NAMES:
        result = run_episode(policy, seed=123)
        assert result["steps"] == 24
        assert result["terminated"]
        assert result["invalid_actions"] == 0
        assert np.isfinite(result["episode_reward"])


def test_episode_result_reproducible_by_seed():
    for policy in POLICY_NAMES:
        first = run_episode(policy, seed=321)
        second = run_episode(policy, seed=321)
        assert first == second
