"""Bounded, agent-visible features for the investment simulator (v2, 53-D).

The first three indicators remain at positions 0-2. Two observable economic
indices follow; twelve features for each sector complete the vector.
All public measurements are deterministic functions of the current state;
true feasibility and future shocks are never included.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np
from gymnasium import spaces

from environment.actions import SECTORS

INITIAL_CAPITAL = 100.0
EPISODE_MONTHS = 24


class ProjectStage(str, Enum):
    UNSTUDIED = "unstudied"
    STUDIED = "studied"
    PILOT = "pilot"
    OPERATING = "operating"
    EXITED = "exited"


GLOBAL_FEATURES = (
    "month_fraction",
    "available_capital_scaled",
    "committed_capital_scaled",
    "employment_index",
    "import_substitution_index",
)
SECTOR_FEATURES = (
    "demand_score",
    "growth_signal",
    "supplier_concentration",
    "feasibility_estimate",
    "uncertainty",
    "investment_scaled",
    "project_progress",
)
STAGE_FEATURES = tuple(f"stage_{stage.value}" for stage in ProjectStage)
OBSERVATION_NAMES = GLOBAL_FEATURES + tuple(
    f"{sector_id}.{feature}"
    for sector_id in SECTORS
    for feature in (*SECTOR_FEATURES, *STAGE_FEATURES)
)
OBSERVATION_SPACE = spaces.Box(
    low=0.0, high=1.0,
    shape=(len(OBSERVATION_NAMES),),
    dtype=np.float32,
)


@dataclass(frozen=True)
class SectorSnapshot:
    demand_score: float
    growth_signal: float
    supplier_concentration: float
    feasibility_estimate: float
    uncertainty: float
    invested_capital: float
    project_progress: float
    stage: ProjectStage


@dataclass(frozen=True)
class PortfolioSnapshot:
    month: int
    available_capital: float
    committed_capital: float
    sectors: Mapping[str, SectorSnapshot]
    employment_index: float = 0.0
    import_substitution_index: float = 0.0


def scale_capital(amount: float) -> float:
    """Map nonnegative capital into [0,1)."""
    if not np.isfinite(amount) or amount < 0:
        raise ValueError("Capital must be finite and nonnegative.")
    return float(amount / (amount + INITIAL_CAPITAL))


def validate_score(name: str, value: float) -> None:
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite value between 0 and 1.")


def encode_observation(snapshot: PortfolioSnapshot) -> np.ndarray:
    """Encode the information available at a decision time as 53 floats."""
    if not isinstance(snapshot.month, int) or not 0 <= snapshot.month <= EPISODE_MONTHS:
        raise ValueError("Month must be an integer within the episode horizon.")
    if set(snapshot.sectors) != set(SECTORS):
        raise ValueError("Snapshot must contain exactly the four configured sectors.")

    validate_score("employment_index", snapshot.employment_index)
    validate_score("import_substitution_index", snapshot.import_substitution_index)
    values = [
        snapshot.month / EPISODE_MONTHS,
        scale_capital(snapshot.available_capital),
        scale_capital(snapshot.committed_capital),
        snapshot.employment_index,
        snapshot.import_substitution_index,
    ]
    for sector_id in SECTORS:
        sector = snapshot.sectors[sector_id]
        numbers = (
            ("demand_score", sector.demand_score),
            ("growth_signal", sector.growth_signal),
            ("supplier_concentration", sector.supplier_concentration),
            ("feasibility_estimate", sector.feasibility_estimate),
            ("uncertainty", sector.uncertainty),
            ("project_progress", sector.project_progress),
        )
        for name, value in numbers:
            validate_score(f"{sector_id}.{name}", value)
        if not isinstance(sector.stage, ProjectStage):
            raise ValueError(f"Invalid stage for {sector_id}")
        values.extend([
            sector.demand_score,
            sector.growth_signal,
            sector.supplier_concentration,
            sector.feasibility_estimate,
            sector.uncertainty,
            scale_capital(sector.invested_capital),
            sector.project_progress,
        ])
        values.extend(1.0 if sector.stage == stage else 0.0 for stage in ProjectStage)

    observation = np.asarray(values, dtype=np.float32)
    if not OBSERVATION_SPACE.contains(observation):
        raise ValueError("Generated observation is outside its declared space")
    return observation


def make_demo_snapshot() -> PortfolioSnapshot:
    """Illustrative signals only; NOT measured data from Rwanda."""
    demo_signals = {
        "agro_processing": (0.75, 0.61, 0.42),
        "textiles": (0.63, 0.57, 0.55),
        "construction_materials": (0.82, 0.67, 0.61),
        "pharmaceuticals": (0.59, 0.55, 0.72),
    }
    sectors = {
        sid: SectorSnapshot(
            demand_score=demo_signals[sid][0],
            growth_signal=demo_signals[sid][1],
            supplier_concentration=demo_signals[sid][2],
            feasibility_estimate=0.50,
            uncertainty=1.00,
            invested_capital=0.0,
            project_progress=0.0,
            stage=ProjectStage.UNSTUDIED,
        )
        for sid in SECTORS
    }
    return PortfolioSnapshot(
        month=0, available_capital=INITIAL_CAPITAL,
        committed_capital=0.0, sectors=sectors,
        employment_index=0.0, import_substitution_index=0.0,
    )
