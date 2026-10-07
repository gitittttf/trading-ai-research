"""
Reconstruct Binance USD-M funding rates from the 1-minute premium index.

Why: funding history is only published as MONTHLY archives on data.binance.vision, but the
premium index klines are published DAILY. The forward paper trader needs funding the day
after it happened, so it rebuilds it with Binance's published formula and replaces it with
the official numbers once the monthly archive appears.

Formula (Binance USD-M docs): for each funding interval of h hours ending at t,
    P = time-weighted average premium index over (t - h, t], later samples weigh more
        (weights 1, 2, ..., n)
    F = P + clamp(I - P, -0.05%, +0.05%),   I = 0.01% * h / 8 (interest per interval)
Checked against the official August 2026 rates for BTC, SOL and DOGE: mean absolute error
about 2e-6 per payment, sum over the month within 1.2% (see docs/EVALUATION_PROTOCOL.md).
Funding caps (rarely hit) are not applied.
"""
from __future__ import annotations

import numpy as np

MINUTE_MS = 60_000
HOUR_MS = 3_600_000
DAY_MS = 86_400_000


def reconstruct_day(open_time: np.ndarray, premium_close: np.ndarray, day_start_ms: int,
                    interval_hours: int) -> tuple[np.ndarray, np.ndarray]:
    """Funding events (timestamps, rates) settling inside (day_start, day_start + 1 day].

    open_time / premium_close: 1-minute premium index klines (any order, may span more days).
    An interval needs at least 90% of its minutes; otherwise its rate is NaN (unknown)."""
    if 24 % interval_hours:
        raise ValueError(f"funding interval {interval_hours}h does not divide a day")
    order = np.argsort(open_time)
    t, x = np.asarray(open_time)[order], np.asarray(premium_close, float)[order]
    h_ms = interval_hours * HOUR_MS
    ts, rates = [], []
    for k in range(1, 24 // interval_hours + 1):
        end = day_start_ms + k * h_ms
        m = (t >= end - h_ms) & (t < end)
        n = int(m.sum())
        ts.append(end)
        if n < 0.9 * interval_hours * 60:
            rates.append(np.nan)
            continue
        w = np.arange(1, n + 1, dtype=float)
        p = float(np.sum(w * x[m]) / w.sum())
        i_rate = 0.0001 * interval_hours / 8.0
        rates.append(p + float(np.clip(i_rate - p, -0.0005, 0.0005)))
    return np.array(ts, dtype=np.int64), np.array(rates)


def interval_hours_from_history(funding_ts: np.ndarray) -> int | None:
    """Most recent funding interval (hours) of a symbol from its official funding timestamps."""
    if funding_ts is None or len(funding_ts) < 4:
        return None
    d = np.diff(np.sort(funding_ts)[-10:]) / HOUR_MS
    h = int(round(float(np.median(d))))
    return h if h in (1, 2, 4, 8) else None
