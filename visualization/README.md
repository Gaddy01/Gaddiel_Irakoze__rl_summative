# Interactive industrial investment mission control

The Python backend loads the already-trained native TorchRL PPO 03 model from
models/ppo/ppo_03.pt. At each month it calls the actual TradeSignalEnv.step
with the model's greedy masked action; a Rule-Based policy receives the same
seed and independently navigates the identical environment.

The interface uses Three.js WebGL to represent each sector as an animated
industrial district and provides live month-by-month rewards, portfolio
metrics, capital, employment, import substitution, and risk indicators.
No training or alternative economic transition calculations are performed.

## Launch from the repository root

Install Python dependencies and run the local application:

    uv sync --frozen
    uv run python -m pytest tests/test_visualization.py -v
    uv run python play.py --open

If the browser does not open, visit http://127.0.0.1:8765 yourself.
You can also specify a different port:

    uv run python play.py --open --port 8877

There is no npm installation, no Node server and no additional Python
dependencies. The browser does need internet access for Three.js from the
jsDelivr CDN, but playback metrics and the Python terminal remain usable
even when the 3D scene cannot load.

Use STEP +1, PLAY / PAUSE, speed selector, orbit / zoom, sector inspection,
seed reset and historical/2024 input switch. The default is the previously
evaluated 2024 scenario using seed 60000. The same 24-month episode is run
in two *separate* simulator instances for fair paired comparison.

## Filming checklist

1. Capture the entire screen and webcam with the VS Code terminal visible.
2. Briefly explain the 17 investment actions, 53 encoded observations, four
   industrial sectors and 24-month delayed rewards.
3. Run the above play.py command visibly in the terminal.
4. Play all 24 months while showing both the Three.js scene and verbose
   Python PPO/Rule-Based decision and reward lines.
5. Compare cumulative reward and simulated wealth, employment, import
   substitution and risk at the conclusion.
6. Explain that real import proxies initialize synthetic economics, and
   that this playback is **not** another independent experiment.
7. Keep within the assignment's five-minute video requirement.

The 2024 holdout has already been evaluated and used in the scientific
analysis. Use this interface for illustration, not for additional
hyperparameter selection or a new claim of out-of-sample generalization.

Automated tests include a production PPO checkpoint replay for seed 60000:
the resulting return and final wealth must match the existing scientific
analysis table. CI checks Python and JavaScript syntax, plus the HTTP API.

This server intentionally binds to localhost only. It is a single-user
demonstration application, not a production authentication system.
