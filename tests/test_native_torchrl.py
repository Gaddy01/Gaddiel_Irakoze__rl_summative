"""Native TorchRL integration tests; no 1000-episode training required."""
from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import DQNLoss, HardUpdate, ValueEstimators
from torchrl.objectives.value import GAE

from environment.torchrl_adapter import HORIZON, IndustrialTorchRLEnv
from training.dqn_training import EpsilonGreedyPolicy
from training.experiment import Config
from training.networks import make_policy, make_q_actor, make_critic
from training.pg_training import make_onpolicy_loss, monte_carlo_targets, native_objective


def collect(algorithm, count=2):
    env = IndustrialTorchRLEnv(split="historical")
    actor = make_q_actor(32, env.action_spec) if algorithm == "dqn" else make_policy(32, env.action_spec)
    history = []
    try:
        for i in range(count):
            env.set_seed(100+i)
            with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
                episode = env.rollout(max_steps=HORIZON, policy=actor)
            assert episode.numel() == HORIZON
            history.append(episode)
    finally:
        env.close()
    return actor, torch.stack(history, 0).contiguous()


def test_53_observations_17_actions_and_terminal():
    env = IndustrialTorchRLEnv(split="historical")
    try:
        env.set_seed(42)
        td = env.reset()
        assert td["observation"].shape == (53,)
        assert td["action_mask"].shape == (17,)
        assert td["action_mask"][0].item()
        assert td["observation"].dtype == torch.float32
        actor = make_policy(32, env.action_spec)
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            trajectory = env.rollout(max_steps=HORIZON, policy=actor)
        assert bool(trajectory["next", "done"][-1])
        assert bool(trajectory["next", "terminated"][-1])
        assert not bool(trajectory["next", "truncated"][-1])
        assert int(trajectory.numel()) == HORIZON
    finally:
        env.close()


def test_masked_probabilistic_actor_only_selects_legal_actions():
    actor, batch = collect("reinforce")
    mask = batch["action_mask"].reshape(-1, 17)
    actions = batch["action"].reshape(-1).long()
    assert bool(mask.gather(-1, actions.unsqueeze(-1)).all())
    assert torch.isfinite(batch["action_log_prob"]).all()


def test_masked_q_actor_and_epsilon_explorer():
    env = IndustrialTorchRLEnv()
    try:
        actor = make_q_actor(32, env.action_spec)
        explorer = EpsilonGreedyPolicy(actor, seed=17, start=1, end=1, decay_steps=100)
        env.set_seed(12)
        episode = env.rollout(max_steps=HORIZON, policy=explorer)
        assert bool(episode["action_mask"].gather(-1, episode["action"].long().reshape(-1, 1)).all())
    finally:
        env.close()


def test_monte_carlo_returns_do_not_cross_episode_boundaries():
    reward = torch.ones(2, HORIZON, 1)
    batch = TensorDict({"next": {"reward": reward}}, batch_size=[2,HORIZON])
    got = monte_carlo_targets(batch, 0.9)
    assert got.shape == (2,HORIZON,1)
    assert float(got[0,-1,0]) == pytest.approx(1.0)
    assert float(got[1,0,0]) == pytest.approx(sum(.9**i for i in range(HORIZON)))
    assert torch.allclose(got[0],got[1])


def test_dqn_native_loss_and_gradients():
    actor, batch = collect("dqn")
    loss_module = DQNLoss(actor, action_space="categorical", delay_value=True,
                          double_dqn=False, loss_function="smooth_l1")
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=0.98)
    updater = HardUpdate(loss_module, value_network_update_interval=10)
    result = loss_module(batch.reshape(-1))
    loss = result["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in loss_module.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    updater.step()


@pytest.mark.parametrize("algorithm", ["reinforce", "ppo", "a2c"])
def test_native_policy_gradient_loss_and_gradients(algorithm):
    actor, batch = collect(algorithm)
    critic = make_critic(32)
    cfg = Config(algorithm=algorithm, hidden=32, rollout_episodes=2)
    objective = make_onpolicy_loss(cfg, actor, critic)
    if algorithm == "reinforce":
        targets = monte_carlo_targets(batch, cfg.gamma)
        with torch.no_grad():
            values = critic(batch.clone())["state_value"]
        batch.set("advantage", targets-values)
        batch.set("value_target", targets)
    else:
        gae = GAE(gamma=cfg.gamma, lmbda=cfg.gae_lambda, value_network=critic)
        with torch.no_grad():
            gae(batch)
    output = objective(batch.reshape(-1))
    loss = native_objective(output)
    assert torch.isfinite(loss), algorithm
    if algorithm == "ppo":
        assert "loss_objective" in output.keys()
    loss.backward()
    grads = [p.grad for p in objective.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads), algorithm


def test_invalid_q_values_are_masked_before_target_backup():
    env = IndustrialTorchRLEnv(split="historical")
    try:
        td = env.reset()
        actor = make_q_actor(32, env.action_spec)
        with torch.no_grad():
            result = actor(td)
        values = result["action_value"]
        mask = result["action_mask"]
        assert (~mask).any(), "The initial state must contain illegal actions"
        assert torch.all(values[~mask] < -1e8)
        assert torch.isfinite(values[mask]).all()
    finally:
        env.close()
