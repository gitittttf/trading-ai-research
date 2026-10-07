"""
Point-in-time pair selection and spread statistics.

Everything here only ever sees the TRAINING window it is given. The walk-forward
runner calls ``select_pairs`` once per window with data strictly before the test
window starts, so the tested universe never depends on future prices.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, asdict

import numpy as np
import polars as pl
from statsmodels.tsa.stattools import coint

from core.data import MINUTE_MS


@dataclass(frozen=True)
class PairSpec:
    asset1: str
    asset2: str
    alpha: float               # spread = log p1 - beta * log p2 - alpha
    beta: float
    coint_pvalue: float        # Engle-Granger p-value (MacKinnon), training window only
    half_life_minutes: float
    return_corr: float
    train_start_ms: int
    train_end_ms: int

    @property
    def name(self) -> str:
        return f"{self.asset1.replace('/', '_')}__{self.asset2.replace('/', '_')}"

    def to_dict(self) -> dict:
        return asdict(self)


def fit_hedge(logp1: np.ndarray, logp2: np.ndarray) -> tuple[float, float]:
    """OLS of log p1 on log p2 with intercept -> (alpha, beta)."""
    x = np.column_stack([np.ones_like(logp2), logp2])
    coef, *_ = np.linalg.lstsq(x, logp1, rcond=None)
    return float(coef[0]), float(coef[1])


def half_life_bars(resid: np.ndarray) -> float:
    """Half-life of mean reversion from an AR(1) fit dS_t = lam * S_{t-1} + c."""
    s_lag = resid[:-1]
    ds = np.diff(resid)
    x = np.column_stack([np.ones_like(s_lag), s_lag])
    coef, *_ = np.linalg.lstsq(x, ds, rcond=None)
    lam = coef[1]
    if lam >= 0:
        return math.inf
    return float(-math.log(2) / lam)


def _close_matrix(closes: dict[str, pl.DataFrame], start_ms: int, end_ms: int,
                  bar_minutes: int) -> tuple[list[str], np.ndarray]:
    """Wide matrix of log closes on a common bar grid (last close per bar)."""
    step = bar_minutes * MINUTE_MS
    frames = []
    for sym, df in closes.items():
        sl = (df.filter((pl.col("timestamp") >= start_ms) & (pl.col("timestamp") < end_ms))
                .with_columns(((pl.col("timestamp") // step) * step).alias("timestamp"))
                .group_by("timestamp").agg(pl.col("close").last().alias(sym)))
        frames.append(sl)
    if not frames:
        return [], np.empty((0, 0))
    wide = frames[0]
    for f in frames[1:]:
        wide = wide.join(f, on="timestamp", how="full", coalesce=True)
    wide = wide.sort("timestamp")
    syms = [c for c in wide.columns if c != "timestamp"]
    mat = wide.select(syms).to_numpy().astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return syms, np.log(mat)


def evaluate_pair(lp1: np.ndarray, lp2: np.ndarray, bar_minutes: int) -> dict | None:
    """Statistics of one candidate pair on aligned log prices (NaN rows removed)."""
    mask = np.isfinite(lp1) & np.isfinite(lp2)
    lp1, lp2 = lp1[mask], lp2[mask]
    if len(lp1) < 200:
        return None
    alpha, beta = fit_hedge(lp1, lp2)
    resid = lp1 - beta * lp2 - alpha
    try:
        pval = float(coint(lp1, lp2, trend="c", autolag="aic")[1])
    except Exception:
        return None
    hl = half_life_bars(resid) * bar_minutes
    r1, r2 = np.diff(lp1), np.diff(lp2)
    corr = float(np.corrcoef(r1, r2)[0, 1]) if np.std(r1) > 0 and np.std(r2) > 0 else 0.0
    return {"alpha": alpha, "beta": beta, "coint_pvalue": pval,
            "half_life_minutes": hl, "return_corr": corr, "n": int(len(lp1))}


def select_pairs(closes: dict[str, pl.DataFrame], train_start_ms: int, train_end_ms: int,
                 selection_bar_minutes: int = 60, max_pvalue: float = 0.05,
                 min_half_life_minutes: float = 60.0, max_half_life_minutes: float = 3 * 1440.0,
                 min_return_corr: float = 0.3, top_k: int = 10, max_pairs_per_asset: int = 2,
                 exclude: set[str] | None = None) -> list[PairSpec]:
    """
    Rank all pairs by Engle-Granger p-value on the training window and keep the top_k
    that pass the filters. ``closes`` maps symbol -> 1m (or any) bars with
    timestamp/close; only rows with train_start_ms <= ts < train_end_ms are used.
    """
    exclude = exclude or set()
    syms, mat = _close_matrix({s: d for s, d in closes.items() if s not in exclude},
                              train_start_ms, train_end_ms, selection_bar_minutes)
    candidates = []
    for i, j in itertools.combinations(range(len(syms)), 2):
        stats = evaluate_pair(mat[:, i], mat[:, j], selection_bar_minutes)
        if stats is None:
            continue
        if not (stats["beta"] > 0 and stats["coint_pvalue"] <= max_pvalue
                and min_half_life_minutes <= stats["half_life_minutes"] <= max_half_life_minutes
                and stats["return_corr"] >= min_return_corr):
            continue
        candidates.append((stats["coint_pvalue"], syms[i], syms[j], stats))
    candidates.sort(key=lambda c: (c[0], c[1], c[2]))  # deterministic tie-break

    chosen: list[PairSpec] = []
    per_asset: dict[str, int] = {}
    for pval, a1, a2, st in candidates:
        if per_asset.get(a1, 0) >= max_pairs_per_asset or per_asset.get(a2, 0) >= max_pairs_per_asset:
            continue
        chosen.append(PairSpec(asset1=a1, asset2=a2, alpha=st["alpha"], beta=st["beta"],
                               coint_pvalue=pval, half_life_minutes=st["half_life_minutes"],
                               return_corr=st["return_corr"], train_start_ms=train_start_ms,
                               train_end_ms=train_end_ms))
        per_asset[a1] = per_asset.get(a1, 0) + 1
        per_asset[a2] = per_asset.get(a2, 0) + 1
        if len(chosen) >= top_k:
            break
    return chosen


def spread_series(close1: np.ndarray, close2: np.ndarray, spec: PairSpec) -> np.ndarray:
    return np.log(close1) - spec.beta * np.log(close2) - spec.alpha


def rolling_zscore(x: np.ndarray, window: int, min_periods: int | None = None) -> np.ndarray:
    """Causal z-score: value at t uses x[t-window+1 .. t] only. NaN during warm-up."""
    min_periods = min_periods or window
    s = pl.Series(x)
    mean = s.rolling_mean(window_size=window, min_samples=min_periods).to_numpy()
    std = s.rolling_std(window_size=window, min_samples=min_periods).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (x - mean) / std
    z[~np.isfinite(z)] = np.nan
    return z
