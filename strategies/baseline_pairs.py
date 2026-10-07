"""
Baseline cointegration pairs strategy (few parameters, fixed a priori).

Per walk-forward window:
1. Select pairs on the TRAINING window only (Engle-Granger p-value, half-life, beta>0).
2. Trade the test window: log spread with the training beta, causal 24h z-score,
   enter on a fresh cross of +-entry_z, exit at +-exit_z, stop at +-stop_z,
   time stop after ``time_stop_half_lives`` half-lives (capped).
All fills happen on the next bar's open (core.backtest timing contract).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import polars as pl

from core.backtest import Execution, ExitRules, simulate_pair
from core.pairs import PairSpec
from strategies.common import PreparedPair, Window, crossing_entries, prepare_pair


@dataclass(frozen=True)
class BaselineParams:
    bar_minutes: int = 5
    z_window_minutes: int = 1440
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    time_stop_half_lives: float = 3.0
    max_hold_minutes: float = 3 * 1440
    latency_bars: int = 0
    exec_mode: str = "taker"           # "maker": passive fill model (variant 10)
    maker_timeout_bars: int = 3
    maker_through_bp: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


class BaselinePairsStrategy:
    name = "baseline"

    def __init__(self, params: BaselineParams | None = None):
        self.params = params or BaselineParams()

    def n_trials(self) -> int:
        return 1  # parameters are fixed a priori, nothing is tuned

    def exit_rules(self, spec: PairSpec) -> ExitRules:
        p = self.params
        hold_min = min(p.time_stop_half_lives * spec.half_life_minutes, p.max_hold_minutes)
        return ExitRules(exit_z=p.exit_z, stop_z=p.stop_z, stop_mode="absolute",
                         max_hold_bars=max(1, int(hold_min // p.bar_minutes)))

    def prepare(self, bars_1m: dict[str, pl.DataFrame], spec: PairSpec, window: Window) -> PreparedPair | None:
        p = self.params
        return prepare_pair(bars_1m, spec, window.test_start, window.test_end, p.bar_minutes,
                            p.z_window_minutes, warmup_minutes=p.z_window_minutes + 2 * p.bar_minutes)

    def candidates_for_pair(self, prep: PreparedPair) -> list[dict]:
        p = self.params
        entries = crossing_entries(prep.z, p.entry_z)
        return simulate_pair(prep.arrays, prep.z, entries, self.exit_rules(prep.spec), prep.spec.beta,
                             pair=prep.spec.name, latency_bars=p.latency_bars,
                             trade_window=prep.trade_window, execution=self.execution())

    def execution(self) -> Execution:
        p = self.params
        return Execution(p.exec_mode, p.maker_timeout_bars, p.maker_through_bp)

    def generate(self, bars_1m: dict[str, pl.DataFrame], specs: list[PairSpec], window: Window) -> list[dict]:
        out = []
        for spec in specs:
            prep = self.prepare(bars_1m, spec, window)
            if prep is None:
                continue
            for t in self.candidates_for_pair(prep):
                t["window"] = window.index
                t["half_life_minutes"] = spec.half_life_minutes
                t["coint_pvalue"] = spec.coint_pvalue
                out.append(t)
        return out
