"""Small, explicitly synthetic stress scenarios for environment validation.

Does not change the original IndustrialInvestmentEnv or its reward function.
The 'baseline' setting is identical to the original dynamics.
The 'downturn' setting reduces hidden viability at reset and introduces a
simulated demand decline plus operating/asset headwinds from month 9 onward.
These are *illustrative* stress assumptions, not forecasts for Rwanda.
"""

import numpy as np

from environment.custom_env import IndustrialInvestmentEnv
from environment.observations import ProjectStage

SCENARIOS = ("baseline", "downturn")


class ScenarioInvestmentEnv(IndustrialInvestmentEnv):
    def __init__(self, scenario: str = "baseline", render_mode=None):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        super().__init__(render_mode=render_mode)
        self.scenario = scenario

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        if self.scenario == "downturn":
            # Hidden viability becomes weaker. This is not directly observed;
            # agents still need to pay for information via feasibility studies.
            for sector in self.sectors.values():
                sector.true_feasibility = max(0.10, sector.true_feasibility - 0.10)
            observation, info = self._observation(), self._info()
        return observation, info

    def _advance_month(self):
        super()._advance_month()
        if self.scenario != "downturn" or self.month < 9:
            return

        for sector in self.sectors.values():
            # Adverse demand change after the usual monthly market update.
            sector.demand_score = float(np.clip(
                sector.demand_score - 0.035, 0.05, 0.95
            ))
            if sector.stage == ProjectStage.OPERATING:
                # An operating-cost shock and a modest monthly asset write-down.
                monthly_cost = 0.55
                if self.cash >= monthly_cost:
                    self.cash -= monthly_cost
                else:
                    shortfall = monthly_cost - self.cash
                    self.cash = 0.0
                    sector.asset_value = max(0.0, sector.asset_value - shortfall)
                sector.asset_value = max(0.0, sector.asset_value * 0.987)
