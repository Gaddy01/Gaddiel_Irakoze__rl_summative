# Intelligent Industrial Investment Decision Simulator

**African Leadership University — Machine Learning Techniques II Summative**

A mission-based reinforcement learning simulator investigating industrial
investment decisions in a *synthetic Rwanda-inspired economy*. The agent
may hold capital, conduct feasibility studies, fund pilots, expand, and
sell investments in four product groups. Rewards combine hypothetical
net-worth changes, modeled employment, import substitution, and risk.

## Framework and current state

The active training code uses **native TorchRL**:
DQNLoss, ReinforceLoss, ClipPPOLoss, A2CLoss, GAE, action-masked
ProbabilisticActor/QValueActor, TensorDict rollouts, and TorchRL replay.

- Observation: 53 normalized float32 features.
- Action: 17 discrete options, including Hold.
- Episode: exactly 24 monthly decisions.
- Source data: UN Comtrade HS4 Rwanda import proxies (2015 baseline;
  2016–2023 development; 2024 holdout).
- Investment outcomes and dynamic market conditions are synthetic.
- Initial 40-run hybrid implementation, logs and checkpoints are
  archived at the [rl-sweep-v1-40runs tag](https://github.com/Gaddy01/Gaddiel_Irakoze__rl_summative/tree/rl-sweep-v1-40runs).
  Those earlier models are incompatible with the active trainer.

**The native TorchRL implementation must pass local integration tests before
the 40 new configurations are executed.** A successful static review or
Git commit is not sufficient to establish that the complete training
pipeline works on your machine.

## Set up and verify

Install [uv](https://docs.astral.sh/uv/). From the repository root:

```powershell
uv sync
uv run python -m pytest -v
uv run python -m training.smoke_torchrl
uv run main.py
```

The smoke check performs brief **actual TorchRL training and checkpoint
saving for every algorithm**. If it fails, stop and fix the error before
committing to the large sweep.

## Fixed, resumable experiment runs

Inspect the 40 unique hyperparameter combinations:

```powershell
uv run python -m training.run_sweep --dry-run
```

Train each algorithm separately (ten configurations each):

```powershell
uv run python -m training.run_sweep --algorithm dqn
uv run python -m training.run_sweep --algorithm reinforce
uv run python -m training.run_sweep --algorithm ppo
uv run python -m training.run_sweep --algorithm a2c
```

Or execute all configurations, skipping completed ones:

```powershell
uv run python -m training.run_sweep
```

The protocol is frozen at 1,000 episodes (24,000 interactions) and
100 historical validation episodes per configuration. Outputs go to
models/<algorithm>/ and logs/training/<algorithm>/. The best model is
chosen by historical greedy validation only.

After all 40 runs have finished, use **2024 once**:

```powershell
uv run python -m training.evaluate_final
```

This script evaluates four chosen policies and matched baselines with
2024 trade signals initializing NEW synthetic 24-month episodes. These
are not actual 2024 firm-level investment results.

## Directory guide

- environment/actions.py, custom_env.py, observations.py, rewards.py:
  unchanged Gymnasium simulator.
- environment/trade_env.py: historical/holdout economic signals.
- environment/torchrl_adapter.py: TorchRL EnvBase, full mask and specs.
- training/networks.py: TorchRL Q actor, masked stochastic actor, critic.
- training/dqn_training.py: native DQNLoss and replay.
- training/pg_training.py: native ReinforceLoss / ClipPPOLoss / A2CLoss.
- training/experiment.py: one-model CLI, evaluation, checkpoint I/O.
- training/run_sweep.py: 40 pre-registered fixed-budget configurations.
- training/evaluate_final.py: one-time held-out generalization test.
- training/smoke_torchrl.py and tests/: correctness gates.
- docs/environment_design.md: environment and reward definitions.
- docs/training_protocol.md: detailed scientific training methodology.

## Economic-data integrity

The UN Comtrade preview values are **not proof of directly reported
Rwanda records**: their source flags currently say is_reported=False
and the HS classification editions vary by year. HS4 imports are demand
proxies, not measurements of actual domestic project viability, returns,
employment, or industrial transformation.

The summative also requires an advanced visualization, 7–10-page report
and recorded demo. Those are separate deliverables still to complete.


## Interactive 3D PPO mission demonstration

Run the industrial visualization, including the already trained PPO 03
checkpoint and a synchronized Rule-Based benchmark:

    uv sync --frozen
    uv run python play.py --open

Open http://127.0.0.1:8765, then use STEP or PLAY to navigate all 24
months. The browser renders four Three.js industrial districts, dynamic
project stages, cumulative rewards, investment metrics and a comparison
against the heuristic. The Python terminal prints each policy's actual
monthly decisions and reward contributions.

No frontend build is necessary. Internet access is required for the
Three.js CDN on the first browser visit. The original checkpoint is
loaded from models/ppo/ppo_03.pt and is never retrained.

See visualization/README.md for controls, validation, and the five-minute
video filming checklist. Real-world import proxies supply economic
starting conditions, while all simulated investment outcomes are synthetic.
