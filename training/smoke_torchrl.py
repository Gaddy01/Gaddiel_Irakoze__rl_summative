"""Check TorchRL <-> Gymnasium compatibility before RL optimization."""
import torch
import torchrl
import gymnasium as gym
from torchrl.envs import GymWrapper
from environment.custom_env import IndustrialInvestmentEnv


class StripNonTensorInfo(gym.Wrapper):
    """Keep this wrapper deliberately minimal for a TorchRL API smoke test.

    Actual training reads the full mask directly from IndustrialInvestmentEnv.
    """
    def reset(self, *, seed=None, options=None):
        obs, _ = self.env.reset(seed=seed, options=options)
        return obs, {}

    def step(self, action):
        obs, reward, terminated, truncated, _ = self.env.step(action)
        return obs, reward, terminated, truncated, {}


def main():
    print(f"torch={torch.__version__} torchrl={torchrl.__version__}")
    wrapped = GymWrapper(StripNonTensorInfo(IndustrialInvestmentEnv()), categorical_action_encoding=True)
    try:
        td = wrapped.reset()
        print("TensorDict keys:", td.keys())
        obs = td["observation"]
        print("Observation shape:", tuple(obs.shape))
        assert obs.shape[-1] == 53, f"Expected 53 features, got {obs.shape}"
        td.set("action", torch.tensor(0, dtype=torch.int64))
        nxt = wrapped.step(td)
        print("Reward:", nxt["next", "reward"])
        assert torch.isfinite(nxt["next", "reward"]).all()
        print("TorchRL GymWrapper smoke test PASSED")
    finally:
        wrapped.close()


if __name__ == "__main__":
    main()
