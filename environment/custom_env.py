"""Synthetic, mission-based industrial-investment environment."""

from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from environment.actions import ACTIONS, ACTION_SPACE, SECTORS, ActionType, decode_action
from environment.observations import (
    EPISODE_MONTHS,
    INITIAL_CAPITAL,
    OBSERVATION_SPACE,
    PortfolioSnapshot,
    ProjectStage,
    SectorSnapshot,
    encode_observation,
    make_demo_snapshot,
)
from environment.rewards import RewardSnapshot, calculate_reward

STUDY_COST = 3.0
PILOT_COST = 15.0
EXPANSION_COST = 20.0
PILOT_MONTHS = 2
EXIT_FEE = 0.01


@dataclass
class SectorState:
    demand_score: float
    growth_signal: float
    supplier_concentration: float
    true_feasibility: float  # Hidden from observations.
    feasibility_estimate: float = 0.50
    uncertainty: float = 1.00
    invested_capital: float = 0.0
    asset_value: float = 0.0
    project_progress: float = 0.0
    stage: ProjectStage = ProjectStage.UNSTUDIED
    post_exit_employment: float = 0.0
    post_exit_substitution: float = 0.0
    # Economic capacity survives the ownership sale.
    post_exit_capacity: float = 0.0
    operating_reference_demand: float = 1.0


class IndustrialInvestmentEnv(gym.Env):
    """One action per month over a finite, 24-month investment horizon."""

    metadata = {"render_modes": ["ansi"], "render_fps": 2}

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")
        self.render_mode = render_mode
        self.action_space = ACTION_SPACE
        self.observation_space = OBSERVATION_SPACE
        self.month = 0
        self.cash = INITIAL_CAPITAL
        self.sectors: dict[str, SectorState] = {}
        self._ready = False
        self._done = False

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        initial = make_demo_snapshot()
        self.month = 0
        self.cash = INITIAL_CAPITAL
        self._done = False
        self._ready = True
        self.sectors = {}
        for sector_id, demo_sector in initial.sectors.items():
            self.sectors[sector_id] = SectorState(
                demand_score=demo_sector.demand_score,
                growth_signal=demo_sector.growth_signal,
                supplier_concentration=demo_sector.supplier_concentration,
                true_feasibility=float(self.np_random.uniform(0.35, 0.90)),
            )
        return self._observation(), self._info()

    def valid_action_mask(self) -> np.ndarray:
        """Boolean mask aligned with the fixed 17-action catalog."""
        mask = np.zeros(len(ACTIONS), dtype=np.bool_)
        mask[0] = True
        for action in ACTIONS[1:]:
            sector = self.sectors[action.sector_id]
            if action.action_type == ActionType.STUDY:
                valid = sector.stage == ProjectStage.UNSTUDIED and self.cash >= STUDY_COST
            elif action.action_type == ActionType.PILOT:
                valid = sector.stage == ProjectStage.STUDIED and self.cash >= PILOT_COST
            elif action.action_type == ActionType.EXPAND:
                valid = (
                    sector.stage == ProjectStage.PILOT
                    and sector.project_progress >= 1.0
                    and self.cash >= EXPANSION_COST
                )
            else:  # EXIT
                valid = sector.stage in (ProjectStage.PILOT, ProjectStage.OPERATING)
            mask[action.action_id] = valid
        return mask

    def _apply_action(self, action_id: int) -> None:
        action = decode_action(action_id)
        if action.action_type == ActionType.HOLD:
            return
        sector = self.sectors[action.sector_id]
        if action.action_type == ActionType.STUDY:
            self.cash -= STUDY_COST
            sector.stage = ProjectStage.STUDIED
            sector.feasibility_estimate = float(
                np.clip(sector.true_feasibility + self.np_random.normal(0, 0.08), 0, 1)
            )
            sector.uncertainty = 0.25
        elif action.action_type == ActionType.PILOT:
            self.cash -= PILOT_COST
            sector.invested_capital = PILOT_COST
            sector.asset_value = PILOT_COST
            sector.project_progress = 0.0
            sector.stage = ProjectStage.PILOT
        elif action.action_type == ActionType.EXPAND:
            self.cash -= EXPANSION_COST
            sector.invested_capital += EXPANSION_COST
            sector.asset_value += EXPANSION_COST
            sector.stage = ProjectStage.OPERATING
            sector.project_progress = 1.0
            # Baseline demand at the time the company enters production.
            sector.operating_reference_demand = max(sector.demand_score, 0.05)
        elif action.action_type == ActionType.EXIT:
            if sector.stage == ProjectStage.OPERATING:
                # Sell the investor's asset, NOT the factory's productive capacity.
                sector.post_exit_capacity = sector.invested_capital / (
                    PILOT_COST + EXPANSION_COST
                )
                sector.post_exit_employment, sector.post_exit_substitution = (
                    self._sector_development(sector)
                )
            self.cash += sector.asset_value * (1 - EXIT_FEE)
            sector.asset_value = 0.0
            sector.invested_capital = 0.0
            sector.project_progress = 0.0
            sector.stage = ProjectStage.EXITED

    def _advance_month(self) -> None:
        """Apply market shocks, progress pilots, and generate operating cash flows."""
        for sector in self.sectors.values():
            sector.growth_signal = float(np.clip(
                0.9 * sector.growth_signal + 0.1 * 0.5
                + self.np_random.normal(0, 0.012), 0.05, 0.95
            ))
            sector.demand_score = float(np.clip(
                sector.demand_score + 0.012 * (sector.growth_signal - 0.5)
                + self.np_random.normal(0, 0.01), 0.05, 0.95
            ))

            if sector.stage == ProjectStage.PILOT and sector.project_progress < 1.0:
                sector.project_progress = min(
                    1.0, sector.project_progress + 1.0 / PILOT_MONTHS
                )
                if sector.project_progress >= 1.0:
                    # One-time simulated pilot outcome, not guaranteed profit.
                    sector.asset_value *= 0.85 + 0.30 * sector.true_feasibility

            elif sector.stage == ProjectStage.OPERATING:
                margin = (
                    0.010 + 0.050 * (sector.true_feasibility - 0.5)
                    + 0.012 * (sector.demand_score - 0.5)
                    + self.np_random.normal(0, 0.007)
                )
                cash_flow = sector.invested_capital * float(margin)
                new_cash = self.cash + cash_flow
                if new_cash < 0:
                    # Losses exceeding cash must reduce the project's asset value.
                    sector.asset_value = max(0.0, sector.asset_value + new_cash)
                    self.cash = 0.0
                else:
                    self.cash = new_cash
                valuation_change = (
                    0.002 * (sector.true_feasibility - 0.5)
                    + 0.008 * (sector.demand_score - 0.5)
                    + self.np_random.normal(0, 0.003)
                )
                sector.asset_value = max(
                    0.0, sector.asset_value * (1 + valuation_change)
                )

        self.month += 1

    @staticmethod
    def _sector_development(sector: SectorState) -> tuple[float, float]:
        """Productive outcomes depend on production capacity and market demand.

        Ownership does not determine operations: owned and sold operating
        companies use the SAME formula. The sold company has no continuing
        claim on the original investor's cash or asset values.
        """
        if sector.stage == ProjectStage.OPERATING:
            capacity = sector.invested_capital / (PILOT_COST + EXPANSION_COST)
        elif sector.stage == ProjectStage.EXITED:
            capacity = sector.post_exit_capacity
        else:
            return 0.0, 0.0
        if capacity <= 0:
            return 0.0, 0.0
        utilisation = float(np.clip(
            sector.demand_score / sector.operating_reference_demand,
            0.0, 1.25,
        ))
        employment = (
            0.25 * capacity * (0.3 + 0.7 * sector.true_feasibility)
            * utilisation
        )
        substitution = (
            0.25 * capacity * sector.demand_score
            * (0.5 + 0.5 * sector.supplier_concentration)
        )
        return float(employment), float(substitution)

    def _reward_snapshot(self) -> RewardSnapshot:
        employment = 0.0
        substitution = 0.0
        risk = 0.0
        for sector in self.sectors.values():
            jobs, imports = self._sector_development(sector)
            employment += jobs
            substitution += imports
            if sector.stage in (ProjectStage.PILOT, ProjectStage.OPERATING):
                exposure_factor = 0.4 if sector.stage == ProjectStage.PILOT else 1.0
                risk += (
                    sector.asset_value / INITIAL_CAPITAL
                    * (1 - sector.true_feasibility)
                    * exposure_factor
                )
        return RewardSnapshot(
            net_worth=self.cash + sum(s.asset_value for s in self.sectors.values()),
            employment_index=float(np.clip(employment, 0, 1)),
            import_substitution_index=float(np.clip(substitution, 0, 1)),
            risk_exposure_index=float(np.clip(risk, 0, 1)),
        )

    def _observation(self) -> np.ndarray:
        outcome = self._reward_snapshot()
        portfolio = PortfolioSnapshot(
            month=self.month,
            employment_index=outcome.employment_index,
            import_substitution_index=outcome.import_substitution_index,
            available_capital=self.cash,
            committed_capital=sum(s.invested_capital for s in self.sectors.values()),
            sectors={
                sector_id: SectorSnapshot(
                    demand_score=s.demand_score,
                    growth_signal=s.growth_signal,
                    supplier_concentration=s.supplier_concentration,
                    feasibility_estimate=s.feasibility_estimate,
                    uncertainty=s.uncertainty,
                    invested_capital=s.invested_capital,
                    project_progress=s.project_progress,
                    stage=s.stage,
                )
                for sector_id, s in self.sectors.items()
            },
        )
        return encode_observation(portfolio)

    def public_state(self) -> dict[str, Any]:
        """Dashboard-safe snapshot: never exposes hidden true feasibility."""
        outcome = self._reward_snapshot()
        return {
            "month": self.month,
            "cash": round(self.cash, 3),
            "net_worth": round(outcome.net_worth, 3),
            "employment_index": round(outcome.employment_index, 3),
            "import_substitution_index": round(outcome.import_substitution_index, 3),
            "risk_exposure_index": round(outcome.risk_exposure_index, 3),
            "sectors": {
                sector_id: {
                    "stage": s.stage.value,
                    "invested_capital": round(s.invested_capital, 3),
                    "asset_value": round(s.asset_value, 3),
                    "progress": round(s.project_progress, 3),
                    "demand_score": round(s.demand_score, 3),
                    "feasibility_estimate": round(s.feasibility_estimate, 3),
                    "uncertainty": round(s.uncertainty, 3),
                }
                for sector_id, s in self.sectors.items()
            },
        }

    def _info(self, *, action_id: int | None = None,
              action_valid: bool | None = None,
              reward_breakdown: dict | None = None) -> dict[str, Any]:
        info = {
            "action_mask": self.valid_action_mask(),
            "portfolio": self.public_state(),
        }
        if action_id is not None:
            info["action_description"] = decode_action(action_id).description
            info["action_valid"] = action_valid
            info["reward_breakdown"] = reward_breakdown
        return info

    def step(self, action: int):
        if not self._ready:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode finished. Call reset() before the next step().")
        if not self.action_space.contains(action):
            raise ValueError(f"Action must be an integer in [0, {len(ACTIONS) - 1}].")
        action_id = int(action)
        before = self._reward_snapshot()
        valid = bool(self.valid_action_mask()[action_id])
        if valid:
            self._apply_action(action_id)
        self._advance_month()
        after = self._reward_snapshot()
        breakdown = calculate_reward(before, after, action_valid=valid)
        self._done = self.month >= EPISODE_MONTHS
        return (
            self._observation(), float(breakdown.total), bool(self._done), False,
            self._info(
                action_id=action_id,
                action_valid=valid,
                reward_breakdown=asdict(breakdown),
            ),
        )

    def render(self) -> str | None:
        if self.render_mode == "ansi":
            state = self.public_state()
            return (
                f"Month {state['month']:02d}/{EPISODE_MONTHS} | "
                f"Cash: {state['cash']:.2f} | "
                f"Net worth: {state['net_worth']:.2f}"
            )
        return None

    def close(self) -> None:
        pass
