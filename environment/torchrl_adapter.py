"""Native TorchRL view of the frozen industrial investment Gymnasium simulator.

The underlying transitions, observation dimension, 17 actions, reward weights,
and historical/2024 data separation are unchanged. Unlike the old smoke test,
the action mask is now part of the actual training observation TensorDict.
"""
from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import Binary, Bounded, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from environment.trade_env import TradeSignalEnv

OBS_DIM = 53
ACTION_DIM = 17
HORIZON = 24


class IndustrialTorchRLEnv(EnvBase):
    """A single, unbatched TorchRL EnvBase with a 24-decision horizon."""

    def __init__(self, split: str = "historical", device: str = "cpu"):
        super().__init__(device=torch.device(device), batch_size=torch.Size([]))
        self.backend = TradeSignalEnv(split=split)
        self._episode_seed = None
        self.observation_spec = Composite(
            observation=Bounded(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=torch.float32),
            action_mask=Binary(n=ACTION_DIM, dtype=torch.bool),
            shape=torch.Size([]),
        )
        self.action_spec = Categorical(n=ACTION_DIM, dtype=torch.int64)
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
        # EnvBase supplies standard done and terminated specs automatically.
        # Keep truncated as a separate field for transparent Gymnasium semantics.

    def _set_seed(self, seed: int | None) -> int | None:
        self._episode_seed = None if seed is None else int(seed)
        return seed

    @staticmethod
    def _state(obs: np.ndarray, info: dict) -> dict:
        mask = np.asarray(info["action_mask"], dtype=np.bool_)
        if mask.shape != (ACTION_DIM,) or not bool(mask[0]):
            raise ValueError("Invalid 17-action mask from investment simulator")
        state = np.asarray(obs, dtype=np.float32)
        if state.shape != (OBS_DIM,) or not np.isfinite(state).all():
            raise ValueError("Invalid 53-feature investment observation")
        return {
            "observation": torch.from_numpy(state.copy()),
            "action_mask": torch.from_numpy(mask.copy()),
        }

    def _reset(self, tensordict=None, **kwargs) -> TensorDict:
        obs, info = self.backend.reset(seed=self._episode_seed)
        values = self._state(obs, info)
        values.update({
            "done": torch.tensor([False], dtype=torch.bool),
            "terminated": torch.tensor([False], dtype=torch.bool),
            "truncated": torch.tensor([False], dtype=torch.bool),
        })
        return TensorDict(values, batch_size=[])

    def _step(self, tensordict: TensorDict) -> TensorDict:
        action = int(tensordict["action"].item())
        if not bool(self.backend.valid_action_mask()[action]):
            raise AssertionError("An RL policy attempted an unavailable investment action")
        obs, reward, terminated, truncated, info = self.backend.step(action)
        values = self._state(obs, info)
        done = bool(terminated or truncated)
        values.update({
            "reward": torch.tensor([float(reward)], dtype=torch.float32),
            "done": torch.tensor([done], dtype=torch.bool),
            # A 24-month finite horizon is a true terminal state for this task.
            "terminated": torch.tensor([done], dtype=torch.bool),
            "truncated": torch.tensor([bool(truncated)], dtype=torch.bool),
        })
        return TensorDict(values, batch_size=[])

    def _close(self) -> None:
        self.backend.close()
