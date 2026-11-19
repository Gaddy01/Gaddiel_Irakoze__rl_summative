"""Mandatory native TorchRL preflight before any 1000-episode experiment.

Exercises all four actual training loss modules, evaluates policies, and writes
short-lived checkpoints in separate smoke directories. NEVER counts as a trial.

Run: uv run python -m training.smoke_torchrl
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torchrl
from torchrl.envs.utils import ExplorationType, set_exploration_type

from environment.torchrl_adapter import HORIZON, IndustrialTorchRLEnv
from training.experiment import Config, train, evaluate_model
from training.networks import make_policy, make_q_actor


def main():
    print(f"PyTorch={torch.__version__} TorchRL={torchrl.__version__}")
    env = IndustrialTorchRLEnv(split="historical")
    try:
        obs = env.reset()
        assert tuple(obs["observation"].shape) == (53,)
        assert tuple(obs["action_mask"].shape) == (17,)
        assert obs["action_mask"].dtype == torch.bool
        for algorithm in ("dqn", "reinforce"):
            actor = (
                make_q_actor(32, env.action_spec)
                if algorithm == "dqn"
                else make_policy(32, env.action_spec)
            )
            env.set_seed(1234)
            with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
                batch = env.rollout(max_steps=HORIZON, policy=actor)
            assert batch.numel() == HORIZON
            assert bool(batch["action_mask"].gather(-1, batch["action"].long().reshape(-1, 1)).all())
            assert torch.isfinite(batch["next", "reward"]).all()
        print("TorchRL environment and action-mask tests PASSED")
    finally:
        env.close()

    for algorithm in ("dqn", "reinforce", "ppo", "a2c"):
        config = Config(
            algorithm=algorithm,
            run_id=algorithm + "_preflight",
            episodes=8,
            seed=101,
            hidden=32,
            batch_size=8,
            replay_size=128,
            warmup=8,
            target_update=8,
            eps_decay_steps=96,
            rollout_episodes=4,
            minibatch_size=24,
            update_epochs=2,
            eval_episodes=4,
            output_dir="logs/smoke",
            models_dir="models/smoke",
        )
        result = train(config)
        if not math.isfinite(result["validation_mean_reward"]):
            raise FloatingPointError(f"{algorithm}: nonfinite validation reward")
        if not Path(result["model"]).is_file():
            raise FileNotFoundError(result["model"])
        checkpoint = torch.load(result["model"], map_location="cpu", weights_only=True)
        reload_env = IndustrialTorchRLEnv(split="historical")
        try:
            reloaded = (
                make_q_actor(32, reload_env.action_spec)
                if algorithm == "dqn"
                else make_policy(32, reload_env.action_spec)
            )
        finally:
            reload_env.close()
        reloaded.load_state_dict(checkpoint["actor_state"])
        recheck = evaluate_model(reloaded, algorithm, episodes=4, seed=50_000)
        if abs(recheck["mean_reward"] - result["validation_mean_reward"]) > 1e-6:
            raise AssertionError(f"{algorithm}: reloaded checkpoint changed greedy validation")
        print(f"{algorithm.upper()}: native loss -> optimizer -> save -> reload -> evaluate PASSED")
    print("ALL NATIVE TORCHRL PREFLIGHT CHECKS PASSED")
    print("Only now start the fixed 1000-episode sweeps.")


if __name__ == "__main__":
    main()
