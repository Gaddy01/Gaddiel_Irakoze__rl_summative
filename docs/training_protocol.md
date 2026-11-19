# Native TorchRL training protocol

This document is the specification for the **single active implementation**.
The original 40-run PyTorch/TensorDict hybrid is preserved under the Git
tag **rl-sweep-v1-40runs**, not duplicated in the active source tree.

## Fixed mission/environment

The investor allocates 100 hypothetical capital units for 24 monthly steps.
There are 17 actions (Hold and Study/Pilot/Expand/Exit for four industrial
product groups) and 53 normalized numerical observations. All four algorithms
use the identical, unchanged Gymnasium IndustrialInvestmentEnv as wrapped by
TradeSignalEnv and the new TorchRL EnvBase adapter.

All transitions, feasibility, financial consequences, employment, import
substitution, and shocks remain **simulated**. Real annual HS4 Rwanda import
values from UN Comtrade's public preview API initialize only the four sector
demand and growth signals: 2015 provides a YoY baseline, 2016–2023 is
training and validation, and 2024 is held out for the very last evaluation.
The public API rows in this dataset all have is_reported=False; these
must not be described as independently confirmed direct reporter values.

The TorchRL adapter outputs both observation (53 float32 values) and
action_mask (17 booleans). Invalid actions retain the original Gymnasium
penalty semantics. Policies respect the mask. The horizon is terminal,
so no value function bootstraps into a subsequent episode.

## Actual native TorchRL components in training

- DQN: torchrl.modules.QValueActor with action_mask_key,
  torchrl.data.TensorDictReplayBuffer and LazyTensorStorage,
  torchrl.objectives.DQNLoss with categorical actions and smooth-L1 loss,
  plus HardUpdate of target Q-values. Epsilon-greedy exploration samples
  only legal actions. Training collects a full episode and then updates
  24 times (subject to replay warm-up). This update cadence differs from V1.
- REINFORCE: TorchRL ProbabilisticActor + MaskedCategorical, native
  torchrl.objectives.ReinforceLoss with a learned critic baseline, and
  complete-episode Monte Carlo reward-to-go. The actor has a true Monte
  Carlo policy-gradient objective (no TD bootstrapping). A separately
  calculated entropy bonus is optional.
- PPO: TorchRL ProbabilisticActor, ValueOperator, GAE, ClipPPOLoss,
  multiple optimization epochs, and minibatches; old policy log-probs
  are saved at action collection time.
- A2C: the same TorchRL policy/value operators and GAE, native A2CLoss,
  one optimization pass per batch; no PPO clipping.

The policy-gradient collector gathers up to eight episodes per batch by
default (192 transitions). Some configurations vary this number.

## Frozen experimental protocol

1. Install dependencies using: `uv sync`
2. Run checks using: `uv run python -m pytest -v`
3. **Mandatory** native training preflight:
   `uv run python -m training.smoke_torchrl`
4. Inspect the predeclared sweep:
   `uv run python -m training.run_sweep --dry-run`
5. Run four ten-configuration sweeps:
   `uv run python -m training.run_sweep --algorithm dqn`, then
   the same for reinforce, ppo, and a2c.
6. Each of the 40 runs receives **1,000 complete episodes and 24,000
   environment steps**. Training seed 42, historical validation
   seeds 50000–50099 (100 episodes). No 2024 access.
7. Only after 40/40 finish, run:
   `uv run python -m training.evaluate_final`
   to select by historical greedy validation and evaluate the chosen
   checkpoint for each algorithm on the 2024 holdout using seeds
   60000–60099. Hold/Random Valid/Rule-Based use matching conditions.

The manifest is stored at logs/training/experiment_plan.json. Changing
the registered settings causes a hard error; do not tune on the 2024 data.
Tied validation scores are broken by lexicographic run ID.

Every run saves actor and critic weights (where applicable) to
models/<algorithm>/<run_id>.pt, and episodes.csv + summary.json in
logs/training/<algorithm>/<run_id>/. Brief smoke tests write to separate
models/smoke and logs/smoke paths. These do not count toward the 40 runs.

## Interpretation and limits

Rewards combine simulated economic and societal objectives; they are not
investment returns or real employment statistics. DQN's Huber TD objective,
policy-gradient losses, and entropy should not be compared as if they
were identical scales. Record their trajectories and reward distributions,
discuss premature policy concentration, and do not claim convergence solely
from a low loss. Using one fixed training seed per configuration limits
claims of statistical superiority. The 2024 experiment tests generalization
across economic starting signals in a synthetic system, not real investment
performance.

The original first 40 experiments are frozen and available under the
rl-sweep-v1-40runs tag. They are **not** TorchRL-native V2 measurements
and must never be mixed silently into V2 results.

References:
- https://docs.pytorch.org/rl/stable/reference/generated/torchrl.objectives.DQNLoss.html
- https://docs.pytorch.org/rl/stable/reference/objectives_policy.html
- https://docs.pytorch.org/rl/stable/tutorials/coding_ppo.html
