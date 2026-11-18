# Intelligent Industrial Investment Decision Simulator
## Environment Design — Version 2 (November 2026)

**Status:** Implemented and evaluated as a synthetic Gymnasium environment. **Not yet trained with TorchRL; no real trade data or graphical simulator integrated.** This document is the authoritative current specification. Historical Version 1 results are identified separately rather than mixed into the active rules.

## 1. Mission, decision problem, and scope

The environment models a hypothetical industrial investment manager allocating capital across four provisional opportunities in a simulated Rwandan economy. The objective is to learn sequential decisions about research, piloting, expanding, holding, and exiting investments while balancing investor wealth with modeled employment and import-substitution outcomes.

The four opportunities are **agro-processing, textiles, construction materials, and pharmaceuticals**. These are illustrative sector labels, **not verified HS4 product mappings**. The present economic indicators and investment economics are synthetic; they must not be presented as measured import demand, actual jobs, firm valuations, or predicted investment returns for Rwanda.

One episode represents **24 monthly decisions** by one agent. The environment is partly observable: the agent receives bounded current signals and research estimates, not latent true feasibility or future randomness.

## 2. Software interfaces

| Item | Current implementation |
|---|---|
| Core class | IndustrialInvestmentEnv in environment/custom_env.py |
| Base API | Gymnasium reset(seed=...) and step(action) |
| Action space | Discrete(17) |
| Observation space | Box(0.0, 1.0, shape=(53,), dtype=float32) |
| Horizon | 24 steps, one per simulated month |
| End condition | terminated=True after month 24; truncated=False |
| Render | Optional ANSI text; advanced graphical rendering pending |
| Extra step information | info includes action_mask, portfolio, and (after a step) action_description, action_valid, reward_breakdown |
| Reproducibility | Identical seed and identical action sequence reproduce the same episode |

The action mask is an **additional convention in the info dictionary**, not a built-in part of Gymnasium's Discrete space. It is used by the baseline policies. A future TorchRL training implementation must define consistently whether/how the same mask is applied to **all** four algorithms.

## 3. Action catalog and constraints

Action 0 is **Hold**. For each of four sectors in this fixed order — agro-processing, textiles, construction materials, pharmaceuticals — the remaining actions are:

| Action IDs | Operation | Preconditions | State/financial change |
|---|---|---|---|
| 1–4 | Study | UNSTUDIED; cash at least 3 | Pay 3 units; change to STUDIED; receive noisy feasibility estimate and uncertainty 0.25 |
| 5–8 | Pilot | STUDIED; cash at least 15 | Transfer 15 units from cash to project asset; change to PILOT |
| 9–12 | Expand | PILOT; progress at least 1.0; cash at least 20 | Transfer 20 more units into asset; change to OPERATING |
| 13–16 | Exit | PILOT or OPERATING | Sell asset, credit cash at 99% of current asset value; change ownership stage to EXITED |

A sector may only progress through the permitted stages **UNSTUDIED → STUDIED → PILOT → OPERATING → EXITED**, with exits also permitted from PILOT. There is no re-entry after EXITED in Version 2.

An invalid action causes no requested investment transaction, but **one month still advances** and an additional penalty of 0.02 is charged to the reward. Hold is always valid before the episode ends.

## 4. State, observations, and information boundaries

The **53 agent-visible float32 features** are encoded in the fixed order specified by environment/observations.py. Each value lies in [0, 1].

**Five global features:**

| Feature | Meaning / source | Update |
|---|---|---|
| month_fraction | Current simulation month / 24 | Every step |
| available_capital_scaled | Cash / (cash + 100) | After financial changes |
| committed_capital_scaled | Sum of invested principals / (sum + 100) | After investments/exits |
| employment_index | Current aggregate normalized synthetic employment capacity | After productive/market changes |
| import_substitution_index | Current aggregate normalized synthetic domestic-production contribution | After productive/market changes |

**Twelve features per sector**, repeated for the four sectors in their fixed catalog order:

| Feature | Meaning / source | Update |
|---|---|---|
| demand_score | Synthetic initial signal + simulated demand dynamics | Every step |
| growth_signal | Synthetic initial signal + stochastic mean reversion | Every step |
| supplier_concentration | Synthetic initial signal (constant in Version 2) | At reset |
| feasibility_estimate | Initially 0.50; noisy research result after Study | On Study |
| uncertainty | 1.00 initially; 0.25 after Study | On Study |
| investment_scaled | Sector committed principal / (principal + 100) | After pilot, expansion, exit |
| project_progress | 0, then 0.5 and 1.0 over the pilot's two months; 1.0 during operation | During pilot/stage changes |
| stage_unstudied | One-hot project/ownership stage | On action |
| stage_studied | One-hot project/ownership stage | On action |
| stage_pilot | One-hot project/ownership stage | On action |
| stage_operating | One-hot project/ownership stage | On action |
| stage_exited | One-hot project/ownership stage | On action |

**Total = 5 + 4 × (7 numerical + 5 one-hot) = 53.**

The original observation specification was 51 features; Version 2 **supersedes** it by adding the two aggregate development indicators. Do not train models with an older observation dimension.

**Hidden internal variables:** true_feasibility, asset_value, operating_reference_demand, and internal post-sale production capacity are not separately included as observation features. Future market shocks are also unknown. The agent may infer some hidden factors indirectly from outcomes and rewards; the environment is therefore **partially observable**, not guaranteed Markov-complete from its observation vector alone. The dashboard's public_state() exposes only selected aggregate/sector outcomes, not true_feasibility.

The initial synthetic signals (demand, growth, supplier concentration) are created by make_demo_snapshot() in environment/observations.py. Their numerical values are **illustrative, not downloaded or calibrated trade statistics**.

## 5. Financial and economic transitions

**Initial conditions:** month=0, cash=100 normalized monetary units; no active assets; all sectors UNSTUDIED; initial feasibility estimate 0.50 and uncertainty 1.00. Each sector receives a hidden feasibility value sampled uniformly from [0.35, 0.90] at reset.

**Study:** reduces cash by 3 and yields a clipped estimate of latent feasibility with Gaussian measurement noise (standard deviation 0.08). Studying provides information, not immediate productive output.

**Pilot:** moves 15 units of cash to an equally valued pilot asset. Each monthly advance adds 0.5 project progress, including the funding month. At maturity, asset value is multiplied once by (0.85 + 0.30 × true_feasibility).

**Expansion:** transfers another 20 into the existing asset, enters OPERATING, and records current demand as the operating reference (minimum reference 0.05). The project then participates in monthly stochastic operating cash flow and valuation changes. The operating cash-flow margin is:

    0.010 + 0.050 × (true_feasibility − 0.5)
          + 0.012 × (demand_score − 0.5) + Normal(0, 0.007)

Cash flow equals committed principal times that margin. If a negative cash flow exceeds cash on hand, the shortfall reduces asset value (floored at zero). Separate valuation changes also affect the asset. Project performance is **simulated**, not estimated from actual financial statements.

**Market movement:** Each month, growth signals mean-revert toward 0.5 with Gaussian noise; demand changes according to the growth signal and Gaussian shocks, bounded to [0.05, 0.95]. The scenario stress test adds additional shocks described in Section 9.

**Net worth:** current cash plus the sum of current **owned** investment asset values. Transferring cash to an equal-value asset is not itself profit. Study expenses, operating losses, valuation changes, cash flows, and exit fees affect wealth.

## 6. Investor exit is not business closure

Version 2 **separates ownership from production**. Exiting an OPERATING investment sells the owner's asset and removes ongoing financial returns, risk exposure, and principal from the original investor's portfolio. The company nevertheless continues operating at its established capacity, even though its ownership stage is EXITED.

Both owned and sold operating businesses use the same synthetic development formula. For an owned company, capacity is invested_capital / 35; after a sale, the established capacity is retained as post_exit_capacity. Define:

    utilization = clip(demand_score / operating_reference_demand, 0, 1.25)
    employment_contribution = 0.25 × capacity × (0.3 + 0.7 × true_feasibility) × utilization
    substitution_contribution = 0.25 × capacity × demand_score × (0.5 + 0.5 × supplier_concentration)

Aggregate development indices are the sums across relevant companies, clipped to [0, 1]. Demand deterioration can reduce the economic contributions of **both owned and sold** companies. Exiting at PILOT stage does not establish a continuing operating company.

Limitations: no external buyer's balance sheet, buyer management policy, company closure, or post-sale financing is simulated. The original investor is not liable for a sold firm's future costs. The stage label EXITED represents the investor's ownership status, **not necessarily business closure**.

## 7. Reward design

The per-step scalar reward is:

    r_t = (W[t+1] − W[t]) / 100
        + 0.10 × (E[t+1] − E[t])
        + 0.10 × (S[t+1] − S[t])
        − 0.005 × R[t+1]
        − 0.02 × I[t]

Here W is investor net worth; E and S are current normalized aggregate employment and import-substitution indicators; R is clipped risk exposure; I equals 1 for an invalid action, 0 otherwise. Current reward coefficients are fixed defaults, **not empirically calibrated weights**.

Risk exposure aggregates the owned PILOT or OPERATING asset value relative to 100, multiplied by (1 − hidden feasibility), with factor 0.4 for pilot and 1.0 for operating investments; the total is clipped to [0, 1]. Sold investments no longer create investor risk exposure. The agent receives a scalar reward; the reward breakdown is recorded in info for diagnostic use.

There is no separate terminal bonus. Using changes in E and S avoids awarding the same development impact repeatedly when those indices are unchanged. Positive returns remain possible from prior investment actions even if the current action is Hold.

## 8. Baseline policies and evaluation method

Evaluation code lives in evaluation/ and scripts/. Baselines make decisions using **observations and the current action mask**, never by accessing true_feasibility directly.

| Policy | Rule |
|---|---|
| always_hold | Always choose Hold |
| random_valid | Randomly select among permitted actions |
| rule_based | Rank sectors by 0.55 × demand + 0.25 × growth + 0.20 × supplier concentration; research candidates, fund/expand those with observed feasibility estimate ≥ 0.58; limit concurrent active projects to two |
| quick_exit | Rule-based, but sell an operating investment at the next opportunity |
| risk_aware | Rule-based, but exit an operating investment after month 8 when its observed demand falls below 0.70 |

Reported experiments use **100 episodes per policy/scenario**, seeds 1000–1099, and 24 steps per episode. This yields 300 episodes for the three-policy baseline evaluation or 1,000 episodes for five policies × two scenarios.

Matching seeds yield comparable initial latent feasibility but **not necessarily matching market shocks across different action sequences**, because randomness is consumed conditionally.

**These are diagnostic policy evaluations, not TorchRL training runs or part of the 40 required RL hyperparameter experiments.**

## 9. Scenario stress test and recorded results

evaluation/scenarios.py defines two diagnostic conditions:

- **Baseline:** identical dynamics to IndustrialInvestmentEnv.
- **Downturn:** subtract 0.10 from each hidden true feasibility at reset (minimum 0.10). From month 9, reduce sector demand an additional 0.035 per month. Each currently owned OPERATING project also incurs 0.55 monthly expense and 1.3% monthly asset write-down. Previously sold firms respond to deteriorating demand in their development indicators, but the selling investor does not pay their later costs.

The downturn is a **synthetic counterfactual, not observed Rwanda macroeconomic data**.

**Version 2 results** (100 seeds per combination; logs/activity81/full/summary.csv):

| Scenario | Strategy | Mean reward | Mean ending net worth | Employment index | Import-substitution index |
|---|---|---:|---:|---:|---:|
| Baseline | Always Hold | 0.0000 | 100.00 | 0.000 | 0.000 |
| Baseline | Random Valid | −0.0282 | 92.54 | 0.290 | 0.222 |
| Baseline | Rule-Based | **+0.2971** | **125.13** | 0.369 | 0.254 |
| Baseline | Quick Exit | −0.0069 | 91.71 | 0.463 | 0.311 |
| Baseline | Risk-Aware | +0.2072 | 114.88 | 0.417 | 0.283 |
| Downturn | Always Hold | **0.0000** | **100.00** | 0.000 | 0.000 |
| Downturn | Random Valid | −0.1406 | 85.28 | 0.073 | 0.051 |
| Downturn | Rule-Based | −0.1347 | 86.89 | 0.061 | 0.045 |
| Downturn | Quick Exit | −0.0930 | 89.69 | 0.066 | 0.047 |
| Downturn | Risk-Aware | −0.0672 | 92.59 | 0.064 | 0.047 |

Interpretation: the rule-based strategy leads in the baseline; a risk-aware exit policy reduces losses among investing policies in the downturn, but holding cash performs best overall during the modeled downturn. These findings describe **the simulator's assumed dynamics**, not actual investment performance.

**Reward sensitivity:** scripts/evaluate_stress.py also re-scores fixed trajectories under financial-heavy, development-heavy, and risk-averse weightings. No policy is reoptimized in this analysis. Results are in logs/activity81/full/reward_sensitivity.csv.

**Historical archives:** logs/baselines/ and logs/activity8/full/ reflect earlier Version 1 assumptions, including the old post-exit treatment. Do not mix their reward curves with Version 2 training results without explicit labeling.

## 10. Validation and reproducibility

The reported local development checks were **65 passing pytest tests**, a successful Gymnasium check_env call, a complete 24-step preview, and complete multi-seed baseline/stress evaluations. This repository currently does **not** include automated GitHub Actions CI; those local results are not independently verified by a hosted CI run.

From the repository root:

    uv sync
    uv run pytest -v
    uv run main.py
    uv run python -m scripts.preview_observations
    uv run python -m scripts.evaluate_baselines --episodes 100 --seed 1000 --output-dir logs/reproduced_baselines
    uv run python -m scripts.evaluate_stress --episodes 100 --seed 1000 --output-dir logs/reproduced_stress

Use separate output folders to preserve committed historical results. The main.py entry point runs the 24-month text demonstration; the advanced renderer and learned-policy replay remain pending.

## 11. Current limitations and remaining work

1. Economic data: initial features, feasibility, returns, employment, and substitution remain synthetic; specific HS4 product selection, public trade-data ingestion, provenance, and calibrated transformations have **not** been implemented.
2. Partial observability: the 53 features exclude some valuation and latent economic state. This is an intentional simplification, not a claim of a fully observable MDP.
3. Economic realism: no financing, debt, firm-level accounting validation, buyer control, real employment counts, or real import-volume displacement.
4. Training: **DQN, REINFORCE, PPO, and A2C are not yet implemented in TorchRL**. At least 10 meaningful hyperparameter configurations per method (40 total) are required.
5. Visualization: only ANSI/terminal rendering exists. A mission-faithful advanced 2D/3D environment and visual replay still need to be built.
6. Evaluation/reporting: learned reward curves, DQN loss, policy-gradient entropy, convergence, unseen-initial-state generalization, a 7–10-page report, and the demonstration video remain outstanding.

All four learned algorithms must use the **same frozen Version 2 training task, observations, action interpretation, reward weights, and evaluation protocol**. Stress scenarios may be reported separately as robustness evaluations, not substituted for apples-to-apples comparisons.
