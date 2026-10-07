"""Phase 7 (protocol variant 16): sizing, re-hedge, interest on idle capital and the margin check."""
import os
import sys

import numpy as np
import pytest

from core.costs import CostModel
from core.daily import DAY_MS, DailyPanel, harvest_sized, margin_breaches, run_daily

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

START = 1_704_067_200_000  # 2024-01-01


def _panel(close, high=None):
    T, N = close.shape
    return DailyPanel(np.arange(T, dtype=np.int64) * DAY_MS + START, [f"S{i}" for i in range(N)], close,
                      np.zeros((T, N)), np.ones((T, N)), high)


def _w12(held_by_day, n_perp=3):
    """Variant-12 style weights: perp j short and its spot column n_perp + j long."""
    w = np.zeros((len(held_by_day), 2 * n_perp))
    for d, held in enumerate(held_by_day):
        for j in held:
            w[d, j], w[d, n_perp + j] = -1 / 15, 1 / 15
    reb = np.r_[True, np.any(w[1:] != w[:-1], axis=1)]
    return w, reb


def test_sizing_uses_capital_with_few_positions_and_caps_per_coin():
    held = [[], [0], [0, 1], [0, 1, 2], [0, 1, 2], []]
    w12, reb12 = _w12(held)
    w, reb, tied = harvest_sized(_panel(np.full((6, 6), 10.0)), w12, reb12, n_perp=3)
    assert w[1, 0] == pytest.approx(-0.25) and w[1, 3] == pytest.approx(0.25) and tied[1] == pytest.approx(0.5)
    assert np.allclose(w[2, [0, 1]], -0.25) and tied[2] == pytest.approx(1.0)
    assert np.allclose(w[3, [0, 1, 2]], -1 / 6) and np.allclose(w[3, 3:], 1 / 6) and tied[3] == pytest.approx(1.0)
    assert np.allclose(w.sum(axis=1), 0.0)                                  # delta neutral
    assert tied[0] == 0 and tied[5] == 0
    assert np.array_equal(reb, reb12)                                       # flat prices: no re-hedge


def test_rehedge_when_perp_is_50_percent_above_last_rehedge():
    held = [[0]] * 6
    w12, reb12 = _w12(held)
    close = np.full((6, 6), 10.0)
    close[:, 0] = [10, 12, 14.9, 15.0, 20, 22.6]                           # +50% at day 3, +50% again at day 5
    close[:, 3] = close[:, 0]
    _, reb, _ = harvest_sized(_panel(close), w12, reb12, n_perp=3)
    assert reb.tolist() == [True, False, False, True, False, True]


def test_idle_capital_earns_cash_rate():
    close = np.full((3, 2), 10.0)
    w = np.array([[-0.25, 0.25], [-0.25, 0.25], [-0.25, 0.25]])
    zero = CostModel(taker_fee=0, half_spread=0, slippage=0)
    cash = np.array([0.0, 0.001, 0.002])
    res = run_daily(_panel(close), w, zero, np.array([True, False, False]), cash_rate=cash,
                    tied=np.array([0.5, 0.5, 0.5]))
    assert res.interest.tolist() == pytest.approx([0.0005, 0.001])
    assert res.net.tolist() == pytest.approx([0.0005, 0.001])


def test_margin_breach_only_when_daily_high_eats_the_margin():
    close = np.full((4, 2), 10.0)
    high = np.full((4, 2), 10.0)
    high[1, 0] = 15.0                     # +50%: 1x margin survives, 0.5x margin does not
    high[3, 0] = 20.0                     # +100%: 1x margin is gone too
    w = np.array([[-0.25, 0.25]] * 4)
    reb = np.array([True, False, False, False])
    p = _panel(close, high)
    assert margin_breaches(p, w, reb, n_perp=1, margin=1.0) == [3]
    assert margin_breaches(p, w, reb, n_perp=1, margin=0.5) == [1, 3]


def test_tbill_rate_of_a_day_is_known_before_it_starts(tmp_path):
    import daily_backtest
    f = tmp_path / "DTB3.csv"
    f.write_text("observation_date,DTB3\n2024-01-01,3.60\n2024-01-02,.\n2024-01-03,7.20\n")
    days = START + np.arange(5, dtype=np.int64) * DAY_MS                    # 2024-01-01 .. 01-05
    r = daily_backtest.tbill_daily(str(f), days)
    assert r[0] == 0.0                                                      # nothing published before 01-01
    assert r[1] == pytest.approx(0.036 / 360)                              # 01-02: missing value skipped
    assert r[2] == pytest.approx(0.036 / 360)                              # 01-03 must not use 01-03's rate
    assert r[3] == pytest.approx(0.072 / 360) and r[4] == pytest.approx(0.072 / 360)
