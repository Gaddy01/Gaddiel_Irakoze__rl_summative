"""TorchRL actors for the fixed 53-observation / 17-action environment."""
from __future__ import annotations

from torch import nn
from tensordict.nn import TensorDictModule
from torchrl.modules import MaskedCategorical, ProbabilisticActor, QValueActor, ValueOperator

from environment.torchrl_adapter import ACTION_DIM, OBS_DIM


def mlp(out_features: int, hidden: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(OBS_DIM, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_features),
    )


class MaskedQNetwork(nn.Module):
    """Mask invalid Q-values before both action selection AND Bellman targets.

    Native DQNLoss can therefore never bootstrap through an invalid action.
    """
    def __init__(self, hidden: int):
        super().__init__()
        self.model = mlp(ACTION_DIM, hidden)

    def forward(self, observation, action_mask):
        q = self.model(observation)
        return q.masked_fill(~action_mask.bool(), -1e9)


def make_q_actor(hidden: int, action_spec) -> QValueActor:
    q = TensorDictModule(
        MaskedQNetwork(hidden),
        in_keys=["observation", "action_mask"],
        out_keys=["action_value"],
    )
    return QValueActor(
        q, spec=action_spec, action_mask_key="action_mask"
    )


def make_policy(hidden: int, action_spec) -> ProbabilisticActor:
    logits = TensorDictModule(
        mlp(ACTION_DIM, hidden),
        in_keys=["observation"],
        out_keys=["logits"],
    )
    return ProbabilisticActor(
        module=logits,
        spec=action_spec,
        in_keys={"logits": "logits", "mask": "action_mask"},
        distribution_class=MaskedCategorical,
        return_log_prob=True,
    )


def make_critic(hidden: int) -> ValueOperator:
    return ValueOperator(mlp(1, hidden), in_keys=["observation"])
