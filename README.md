# Intelligent Industrial Investment Decision Simulator

A **mission-based reinforcement learning environment** exploring sequential industrial investment decisions in a *synthetic* Rwandan economy. The investor can conduct studies, fund pilot projects, expand, hold capital, and sell investments. Rewards combine financial net-worth changes, modeled employment and import-substitution outcomes, and risk exposure.

**Current stage (November 2026):** Gymnasium environment **Version 2**, five baseline/stress evaluation policies, and 65 local passing tests. The four required TorchRL algorithms (DQN, REINFORCE, PPO, A2C), real HS4 data integration, advanced graphical simulation, final report, and video **are not yet implemented**. All current economic input values are synthetic, *not Rwanda investment statistics*.

## Quick start

Install [uv](https://docs.astral.sh/uv/), then run these commands in the repository root:

    uv sync
    uv run main.py
    uv run pytest -v

The default main.py launches a deterministic **24-month scripted demonstration**, not a trained RL agent or graphical UI. To view the 17 investment actions instead:

    uv run main.py --catalog

Other previews:

    uv run python -m scripts.preview_observations
    uv run python -m scripts.preview_rewards
    uv run python -m scripts.preview_environment

The original development run reported 65 passing automated tests and a successful Gymnasium checker. Re-run the tests in your own checkout; hosted CI is not yet configured.

## Environment at a glance

| Setting | Version 2 |
|---|---|
| Sector labels | Agro-processing, textiles, construction materials, pharmaceuticals (provisional) |
| Action space | 17 discrete actions |
| Observation space | 53 normalized float32 features |
| Initial capital | 100 hypothetical monetary units |
| Episode horizon | 24 monthly decisions |
| Objective | Financial performance + modeled productive development − risk/invalid-action penalties |
| Exit semantics | Investor sells the asset; an established business can continue producing under new ownership |

The **authoritative** and fully consolidated specification is [docs/environment_design.md](docs/environment_design.md). It describes observations, transitions, reward weights, post-exit economic continuity, experimental limitations, and references to the Version 2 logs.

## Reproduce evaluation

Three baseline strategies (Hold, Random Valid, Rule-Based), 100 episodes each:

    uv run python -m scripts.evaluate_baselines --episodes 100 --seed 1000 --output-dir logs/reproduced_baselines

Five strategies under baseline and synthetic downturn conditions, 100 episodes per combination:

    uv run python -m scripts.evaluate_stress --episodes 100 --seed 1000 --output-dir logs/reproduced_stress

Use new output folders to avoid overwriting historical results. The stress script produces episodes.csv, summary.csv, and reward_sensitivity.csv. Reweighted scores are **fixed-trajectory sensitivity checks**, not retrained policies.

**Reference Version 2** results: [logs/activity81/full/summary.csv](logs/activity81/full/summary.csv). In the baseline scenario, Rule-Based averaged +0.2971 cumulative reward and 125.13 end net worth. In the synthetic downturn, Risk-Aware averaged −0.0672 reward and 92.59 end net worth; Always Hold preserved 100. These are outcomes of the *assumed simulator*, not financial forecasts.

## Repository structure

| Directory or file | Purpose |
|---|---|
| environment/ | Gymnasium environment, actions, 53-feature observations, reward function |
| evaluation/ | Benchmark policies and synthetic stress scenarios |
| scripts/ | Runnable previews and evaluation scripts |
| tests/ | 65 automated tests covering actions, observations, rewards, transitions, policies and stress behavior |
| logs/baselines/ | Original historical baseline |
| logs/activity8/full/ | Historical Version 1 stress results |
| logs/activity81/full/ | Corrected Version 2 results |
| docs/environment_design.md | Current technical specification |
| pyproject.toml, uv.lock | Dependency specification and pinned lockfile |

## Scientific scope and next steps

The four sector categories are provisional; **specific HS4 products and actual public trade indicators are not yet included**. Hidden feasibility and investment returns are simulated. Same-seed policies begin with comparable conditions, but action-dependent random draws may change subsequent market shocks. The 53-feature observation is partly observable, not a claim of perfect information.

Still required: public economic-data provenance and integration; four TorchRL agents; **at least 10 meaningful hyperparameter runs per algorithm**; reward/loss/entropy and generalization analysis; advanced visual replay; a 7–10-page report; and demonstration video. All four trained algorithms must use an identical frozen environment and evaluation protocol.

This is an academic simulation, not investment advice.
