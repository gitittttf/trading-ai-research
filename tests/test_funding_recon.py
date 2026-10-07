import numpy as np
import pytest

from core.funding_recon import DAY_MS, interval_hours_from_history, reconstruct_day

D0 = 1_790_000_000_000 // DAY_MS * DAY_MS


def _minutes(day, values):
    return day + np.arange(len(values), dtype=np.int64) * 60_000, np.asarray(values, float)


def test_flat_premium_gives_interest_rate():
    t, x = _minutes(D0, np.zeros(1440))
    ts, r = reconstruct_day(t, x, D0, 8)
    assert list(ts) == [D0 + 8 * 3_600_000, D0 + 16 * 3_600_000, D0 + DAY_MS]
    assert np.allclose(r, 0.0001)            # P = 0 -> F = clamp(I, +-0.05%) = 0.01% per 8h
    _, r4 = reconstruct_day(t, x, D0, 4)
    assert np.allclose(r4, 0.00005) and len(r4) == 6


def test_large_premium_passes_through_and_weights_late_minutes_more():
    vals = np.full(1440, 0.002)
    t, x = _minutes(D0, vals)
    _, r = reconstruct_day(t, x, D0, 8)
    assert np.allclose(r, 0.002 - 0.0005)    # I - P = -0.19% clamped to -0.05%
    ramp = np.concatenate([np.zeros(240), np.full(240, 0.001)] * 3)   # second half of each 8h = 0.1%
    t, x = _minutes(D0, ramp)
    _, r = reconstruct_day(t, x, D0, 8)
    w = np.arange(1, 481)
    p = np.sum(w[240:] * 0.001) / w.sum()     # linearly increasing weights favour the late half
    assert r[0] == pytest.approx(p + max(-0.0005, min(0.0005, 0.0001 - p)))
    assert p > 0.0005


def test_missing_minutes_make_the_interval_unknown():
    t, x = _minutes(D0, np.zeros(1440))
    keep = (t < D0 + 3 * 3_600_000) | (t >= D0 + 8 * 3_600_000)   # 5 of the first 8 hours missing
    _, r = reconstruct_day(t[keep], x[keep], D0, 8)
    assert np.isnan(r[0]) and np.allclose(r[1:], 0.0001)


def test_interval_detection():
    h8 = D0 + np.arange(20) * 8 * 3_600_000
    assert interval_hours_from_history(h8) == 8
    h4 = np.concatenate([h8[:10], h8[9] + np.arange(1, 12) * 4 * 3_600_000])
    assert interval_hours_from_history(h4) == 4
    assert interval_hours_from_history(h8[:2]) is None
