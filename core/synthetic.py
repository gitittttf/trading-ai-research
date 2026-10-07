"""
Synthetic 1-minute markets with KNOWN properties, used to test the pipeline.

* ``random_walk_market``: assets share a market factor but no pair mean-reverts.
  Any strategy must earn ~0 gross here; a profit means a bug (look-ahead etc.).
* ``cointegrated_market``: adds pairs whose log spread follows an OU process with a
  chosen half-life. A working mean-reversion strategy must earn gross here.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from core.data import MINUTE_MS

START_MS = 1_704_067_200_000  # 2024-01-01 00:00 UTC


def _bars_from_logprice(logp: np.ndarray, rng: np.random.Generator, ts: np.ndarray,
                        vol_scale: float) -> pl.DataFrame:
    close = np.exp(logp)
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1]                  # 24/7 market: next open = previous close
    wiggle = np.abs(rng.normal(0, vol_scale, size=(2, len(close))))
    high = np.maximum(open_, close) * (1 + wiggle[0])
    low = np.minimum(open_, close) * (1 - wiggle[1])
    volume = rng.lognormal(mean=8.0, sigma=0.5, size=len(close))
    taker = volume * rng.uniform(0.4, 0.6, size=len(close))
    return pl.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low,
                         "close": close, "volume": volume, "taker_buy_volume": taker})


def random_walk_market(n_assets: int = 6, n_minutes: int = 60 * 24 * 60, seed: int = 0,
                       sigma: float = 0.0008, market_beta: float = 0.8) -> dict[str, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    ts = START_MS + np.arange(n_minutes, dtype=np.int64) * MINUTE_MS
    market = rng.normal(0, sigma, n_minutes)
    out = {}
    for k in range(n_assets):
        idio = rng.normal(0, sigma, n_minutes)
        logp = np.log(10.0 + k) + np.cumsum(market_beta * market + idio)
        out[f"A{k}/USDT"] = _bars_from_logprice(logp, rng, ts, sigma / 2)
    return out


def cointegrated_market(n_pairs: int = 3, n_minutes: int = 60 * 24 * 60, seed: int = 0,
                        sigma: float = 0.0008, half_life_minutes: float = 240.0,
                        spread_sigma: float = 0.0006, beta: float = 1.0) -> dict[str, pl.DataFrame]:
    """Pairs (P{k}A, P{k}B) with log pA = beta * log pB + OU spread."""
    rng = np.random.default_rng(seed)
    ts = START_MS + np.arange(n_minutes, dtype=np.int64) * MINUTE_MS
    theta = np.log(2) / half_life_minutes
    out = {}
    market = np.cumsum(rng.normal(0, sigma, n_minutes))
    for k in range(n_pairs):
        base = np.log(20.0 + k) + 0.8 * market + np.cumsum(rng.normal(0, sigma, n_minutes))
        eps = rng.normal(0, spread_sigma, n_minutes)
        s = np.empty(n_minutes)
        s[0] = 0.0
        for i in range(1, n_minutes):
            s[i] = s[i - 1] * (1 - theta) + eps[i]
        logp_a = beta * base + 0.3 + s
        out[f"P{k}A/USDT"] = _bars_from_logprice(logp_a, rng, ts, sigma / 2)
        out[f"P{k}B/USDT"] = _bars_from_logprice(base, rng, ts, sigma / 2)
    return out


def zero_funding(symbols: list[str], start_ms: int = START_MS, days: int = 60) -> dict[str, pl.DataFrame]:
    ts = start_ms + np.arange(days * 3, dtype=np.int64) * 8 * 3_600_000
    return {s: pl.DataFrame({"timestamp": ts, "funding_rate": np.zeros(len(ts))}) for s in symbols}
