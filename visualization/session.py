"""Deterministic, step-driven replay of the frozen PPO policy and a heuristic.

No reward formula, action logic, observation encoding or environment state is
reimplemented here. Both strategies invoke TradeSignalEnv.step directly.
"""
from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Callable, Any
import logging
import math

import numpy as np

from environment.actions import ACTIONS, SECTORS
from evaluation.baselines import choose_action

LOG = logging.getLogger("industrial_visualization")
DEFAULT_CHECKPOINT = Path("models/ppo/ppo_03.pt")
HORIZON = 24


def load_ppo_policy(checkpoint: Path) -> Callable[[np.ndarray, np.ndarray], int]:
    """Load the *already trained* native-TorchRL policy, not an imitation."""
    import torch
    from tensordict import TensorDict
    from torchrl.envs.utils import ExplorationType, set_exploration_type
    from environment.torchrl_adapter import IndustrialTorchRLEnv
    from training.networks import make_policy

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing selected model: {checkpoint}. Run from the repository root "
            "containing models/ppo/ppo_03.pt. The visualization does not train a replacement."
        )
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if data.get("algorithm") != "ppo" or data.get("framework") != "native TorchRL objectives":
        raise ValueError("Checkpoint is not the selected native-TorchRL PPO model")
    model_id = str(data.get("config", {}).get("run_id", ""))
    if model_id != "ppo_03":
        raise ValueError(f"Expected ppo_03, checkpoint identifies {model_id!r}")
    hidden = int(data["config"]["hidden"])
    wrapper = IndustrialTorchRLEnv(split="historical")
    try:
        actor = make_policy(hidden, wrapper.action_spec)
    finally:
        wrapper.close()
    actor.load_state_dict(data["actor_state"], strict=True)
    actor.eval()

    def choose(observation: np.ndarray, mask: np.ndarray) -> int:
        td = TensorDict({
            "observation": torch.as_tensor(observation, dtype=torch.float32),
            "action_mask": torch.as_tensor(mask, dtype=torch.bool),
        }, batch_size=[])
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            result = actor(td)
        action = int(result["action"].item())
        if not 0 <= action < len(ACTIONS) or not bool(mask[action]):
            raise RuntimeError(f"PPO chose invalid action {action}")
        return action

    return choose


class SimulationController:
    """Thread-safe single-room simulation with synchronous paired rollouts.

    Dependency injections are solely for isolated unit tests. Production always
    uses the native-TorchRL PPO checkpoint and existing TradeSignalEnv.
    """

    def __init__(self, *, checkpoint: Path = DEFAULT_CHECKPOINT,
                 env_factory: Callable[..., Any] | None = None,
                 policy: Callable[[np.ndarray, np.ndarray], int] | None = None,
                 seed: int = 60000, split: str = "holdout"):
        if env_factory is None:
            from environment.trade_env import TradeSignalEnv
            env_factory = TradeSignalEnv
        self.env_factory = env_factory
        self.ppo_policy = policy if policy is not None else load_ppo_policy(Path(checkpoint))
        self.checkpoint = str(checkpoint).replace("\\", "/")
        self.lock = RLock()
        self.envs: dict[str, Any] = {}
        self.states: dict[str, dict] = {}
        self.events: list[dict] = []
        self.seed = seed
        self.split = split
        self.month = 0
        self.year = None
        self.reset(seed=seed, split=split)

    def reset(self, *, seed: int, split: str) -> dict:
        if split not in ("historical", "holdout"):
            raise ValueError("Choose historical (2016–2023) or holdout (2024)")
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2_000_000_000:
            raise ValueError("Seed must be an integer from 0 to 2,000,000,000")
        with self.lock:
            for env in self.envs.values():
                env.close()
            self.envs = {}
            self.states = {}
            self.events = []
            self.seed = seed
            self.split = split
            self.month = 0
            self.rule_rng = np.random.default_rng(seed + 100_000)
            for name in ("ppo", "rule_based"):
                env = self.env_factory(split=split)
                self.envs[name] = env
                obs, info = env.reset(seed=seed)
                self.states[name] = {
                    "observation": obs,
                    "info": info,
                    "total_reward": 0.0,
                    "last_action": None,
                    "last_reward": 0.0,
                    "last_breakdown": None,
                    "history": [self._point(env, 0.0, 0.0)],
                }
            self.year = getattr(self.envs["ppo"], "signal_year", None)
            LOG.info("RESET split=%s year=%s seed=%d checkpoint=%s", split, self.year, seed, self.checkpoint)
            return self.snapshot()

    @staticmethod
    def _point(env, reward: float, total: float) -> dict:
        state = env.public_state()
        return {
            "month": int(state["month"]),
            "net_worth": float(state["net_worth"]),
            "cash": float(state["cash"]),
            "employment_index": float(state["employment_index"]),
            "import_substitution_index": float(state["import_substitution_index"]),
            "risk_exposure_index": float(state["risk_exposure_index"]),
            "reward": round(float(reward), 7),
            "cumulative_reward": round(float(total), 7),
        }

    def step(self) -> dict:
        with self.lock:
            if self.month >= HORIZON:
                return self.snapshot()
            next_month = self.month + 1
            for name in ("ppo", "rule_based"):
                state = self.states[name]
                env = self.envs[name]
                obs, info = state["observation"], state["info"]
                mask = np.asarray(info["action_mask"], dtype=bool)
                if name == "ppo":
                    action_id = int(self.ppo_policy(obs, mask))
                else:
                    action_id = int(choose_action("rule_based", obs, mask, self.rule_rng))
                if not 0 <= action_id < len(ACTIONS) or not bool(mask[action_id]):
                    raise RuntimeError(f"{name} selected illegal action {action_id}")
                next_obs, reward, terminated, truncated, next_info = env.step(action_id)
                if not math.isfinite(float(reward)):
                    raise FloatingPointError("Environment returned non-finite reward")
                action = ACTIONS[action_id]
                action_event = {
                    "month": next_month,
                    "policy": name,
                    "action_id": action_id,
                    "type": action.action_type.value,
                    "sector": action.sector_id,
                    "description": action.description,
                    "reward": round(float(reward), 6),
                    "valid": bool(next_info.get("action_valid", True)),
                }
                state["total_reward"] += float(reward)
                state["observation"] = next_obs
                state["info"] = next_info
                state["last_action"] = action_event
                state["last_reward"] = float(reward)
                state["last_breakdown"] = next_info.get("reward_breakdown")
                state["history"].append(self._point(env, float(reward), state["total_reward"]))
                self.events.append(action_event)
                portfolio = env.public_state()
                LOG.info(
                    "M%02d %-10s action=%02d %-45s reward=%+.5f cumulative=%+.5f "
                    "cash=%.2f wealth=%.2f jobs=%.3f substitution=%.3f risk=%.3f",
                    next_month, name.upper(), action_id, action.description,
                    reward, state["total_reward"], portfolio["cash"],
                    portfolio["net_worth"], portfolio["employment_index"],
                    portfolio["import_substitution_index"], portfolio["risk_exposure_index"],
                )
                if next_month == HORIZON and not (terminated or truncated):
                    raise RuntimeError("Expected termination at 24 months")
            self.month = next_month
            if self.month >= HORIZON:
                LOG.info("EPISODE COMPLETED seed=%d year=%s PPO=%+.5f RULE=%+.5f",
                         self.seed, self.year, self.states["ppo"]["total_reward"],
                         self.states["rule_based"]["total_reward"])
            return self.snapshot()

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "meta": {
                    "month": self.month,
                    "horizon": HORIZON,
                    "done": self.month == HORIZON,
                    "seed": self.seed,
                    "split": self.split,
                    "year": self.year,
                    "checkpoint": self.checkpoint,
                    "model": "PPO 03 (greedy native TorchRL policy)",
                    "simulation_notice": "Import proxies are observed; all investment returns, jobs and risks are simulated.",
                },
                "sector_labels": dict(SECTORS),
                "action_catalog": [
                    {"id": action.action_id, "type": action.action_type.value,
                     "sector": action.sector_id, "description": action.description}
                    for action in ACTIONS
                ],
                "policies": {
                    name: {
                        "portfolio": env.public_state(),
                        "total_reward": round(self.states[name]["total_reward"], 6),
                        "last_action": self.states[name]["last_action"],
                        "last_reward": round(self.states[name]["last_reward"], 6),
                        "last_breakdown": self.states[name]["last_breakdown"],
                        "history": self.states[name]["history"],
                    }
                    for name, env in self.envs.items()
                },
                "events": self.events[-48:],
            }

    def close(self):
        with self.lock:
            for env in self.envs.values():
                env.close()
            self.envs.clear()
