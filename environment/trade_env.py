"""Observed annual HS4 import indicators as initial conditions for the synthetic RL economy.

One 24-step episode starts with signals from ONE observed calendar year.
The within-episode dynamics, feasibility and finance remain synthetic.
No rows from 2024 enter training or validation. No 2024 parameter tuning.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

from environment.actions import SECTORS
from environment.custom_env import IndustrialInvestmentEnv

DATA_DIR = Path('data/processed')
DATA_FILES = {
    'historical': 'sector_signals_historical.csv',
    'holdout': 'sector_signals_2024_holdout.csv',
}


def read_signals(split: str, data_dir: Path = DATA_DIR) -> dict[int, dict[str, dict[str, float]]]:
    if split not in DATA_FILES:
        raise ValueError(f'Unknown split: {split!r}')
    path = data_dir / DATA_FILES[split]
    if not path.is_file():
        raise FileNotFoundError(f'Missing economic indicators: {path}. Run scripts.prepare_trade_data.')
    observations: dict[int, dict[str, dict[str, float]]] = {}
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            year = int(row['signal_year'])
            sector = row['sector']
            if sector not in SECTORS:
                raise ValueError(f'Unrecognized sector: {sector}')
            if (split == 'historical' and not 2016 <= year <= 2023) or (split == 'holdout' and year != 2024):
                raise ValueError(f'Unexpected year {year} in {split} input')
            if sector in observations.setdefault(year, {}):
                raise ValueError(f'Duplicate data record: {year}/{sector}')
            demand = float(row['demand_score'])
            growth = float(row['growth_signal'])
            if any(not math.isfinite(v) or v < 0 or v > 1 for v in (demand, growth)):
                raise ValueError(f'Unbounded or invalid signal: {year}/{sector}')
            observations[year][sector] = {'demand_score': demand, 'growth_signal': growth}

    years = list(range(2016, 2024)) if split == 'historical' else [2024]
    if sorted(observations) != years:
        raise ValueError(f'Expected all years {years}, found {sorted(observations)}')
    for year in years:
        if set(observations[year]) != set(SECTORS):
            raise ValueError(f'Incomplete sectors in year {year}')
    return observations


class TradeSignalEnv(IndustrialInvestmentEnv):
    """Same 53-D 17-action environment, with real year-specific initial signals.

    Synthetic supplier concentration, true feasibility, project returns, shocks,
    jobs and substitution formulas are inherited unchanged. All four algorithms
    must use this exact environment configuration for comparable training.
    """

    def __init__(self, split: str = 'historical', data_dir: Path = DATA_DIR, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.split = split
        self.signals = read_signals(split, Path(data_dir))
        self.years = sorted(self.signals)
        self.signal_year: int | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed, options=options)
        # Uniform, deterministic year cycling across all algorithms for equal seeds.
        index = int(seed) % len(self.years) if seed is not None else int(self.np_random.integers(len(self.years)))
        year = self.years[index]
        for sid, state in self.sectors.items():
            s = self.signals[year][sid]
            state.demand_score = s['demand_score']
            state.growth_signal = s['growth_signal']
        self.signal_year = year
        obs, info = self._observation(), self._info()
        info['signal_year'] = year
        info['signal_source'] = 'UN Comtrade HS4 processed indicators; remaining economics synthetic'
        return obs, info
