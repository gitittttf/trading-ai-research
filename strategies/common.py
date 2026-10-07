"""
Shared plumbing for spread strategies: windows, per-pair bar preparation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from core.backtest import PairArrays
from core.data import MINUTE_MS, align_pair, resample_klines, time_slice
from core.pairs import PairSpec, rolling_zscore, spread_series

DAY_MS = 86_400_000


@dataclass(frozen=True)
class Window:
    index: int
    train_start: int
    train_end: int     # exclusive; == test_start
    test_start: int
    test_end: int      # exclusive

    def label(self) -> str:
        import datetime as dt
        f = lambda ms: dt.datetime.fromtimestamp(ms / 1000, dt.UTC).strftime("%Y-%m-%d")
        return f"W{self.index:02d} train {f(self.train_start)}..{f(self.train_end)} test {f(self.test_start)}..{f(self.test_end)}"


def make_windows(data_start: int, data_end: int, train_days: int, test_days: int,
                 holdout_days: int = 0) -> list[Window]:
    """Rolling walk-forward windows; the last ``holdout_days`` are never touched."""
    end = data_end - holdout_days * DAY_MS
    windows = []
    t0 = data_start
    i = 0
    while True:
        tr_end = t0 + train_days * DAY_MS
        te_end = min(tr_end + test_days * DAY_MS, end)
        if te_end - tr_end < DAY_MS:
            break
        windows.append(Window(i, t0, tr_end, tr_end, te_end))
        i += 1
        if te_end >= end:
            break
        t0 += test_days * DAY_MS
    return windows


def pair_arrays(df: pl.DataFrame) -> PairArrays:
    """PairArrays from an aligned bar frame (``align_pair`` output)."""
    return PairArrays(ts=df["timestamp"].to_numpy(), open1=df["open1"].to_numpy(), close1=df["close1"].to_numpy(),
                      open2=df["open2"].to_numpy(), close2=df["close2"].to_numpy(),
                      vol1_usd=(df["volume1"] * df["close1"]).to_numpy(),
                      vol2_usd=(df["volume2"] * df["close2"]).to_numpy(),
                      high1=df["high1"].to_numpy(), low1=df["low1"].to_numpy(),
                      high2=df["high2"].to_numpy(), low2=df["low2"].to_numpy())


@dataclass
class PreparedPair:
    spec: PairSpec
    bars: pl.DataFrame          # aligned resampled bars (warm-up + test)
    arrays: PairArrays
    spread: np.ndarray
    z: np.ndarray
    trade_window: tuple[int, int]   # decision indices inside the test window


def prepare_pair(bars_1m: dict[str, pl.DataFrame], spec: PairSpec, start_ms: int, end_ms: int,
                 bar_minutes: int, z_window_minutes: int, warmup_minutes: int) -> PreparedPair | None:
    """Resample, align and z-score one pair from (start - warmup) to end.

    Decisions are only allowed for bars with start_ms <= ts < end_ms; bars before
    start_ms only feed the rolling statistics (they are in the past, so this is
    not look-ahead).
    """
    lo_ms = start_ms - warmup_minutes * MINUTE_MS
    b1 = resample_klines(time_slice(bars_1m[spec.asset1], lo_ms, end_ms), bar_minutes)
    b2 = resample_klines(time_slice(bars_1m[spec.asset2], lo_ms, end_ms), bar_minutes)
    df = align_pair(b1, b2)
    if df.height < 10:
        return None
    ts = df["timestamp"].to_numpy()
    c1, c2 = df["close1"].to_numpy(), df["close2"].to_numpy()
    arrays = pair_arrays(df)
    spread = spread_series(c1, c2, spec)
    win = max(2, int(round(z_window_minutes / bar_minutes)))
    z = rolling_zscore(spread, win)
    lo = int(np.searchsorted(ts, start_ms, side="left"))
    hi = int(np.searchsorted(ts, end_ms, side="left"))
    if hi - lo < 2:
        return None
    return PreparedPair(spec=spec, bars=df, arrays=arrays, spread=spread, z=z, trade_window=(lo, hi))


def crossing_entries(z: np.ndarray, entry_z: float) -> np.ndarray:
    """+1 when z crosses below -entry_z, -1 when it crosses above +entry_z (fresh cross only).

    Requiring a fresh cross means a stopped-out trade is not re-entered while the
    spread is still stretched.
    """
    sig = np.zeros(len(z))
    prev = np.concatenate([[np.nan], z[:-1]])
    ok = np.isfinite(z) & np.isfinite(prev)
    sig[ok & (z < -entry_z) & (prev >= -entry_z)] = 1
    sig[ok & (z > entry_z) & (prev <= entry_z)] = -1
    return sig
