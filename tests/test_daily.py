import numpy as np
import polars as pl
import pytest

from core.costs import CostModel
from core.daily import DAY_MS, DailyPanel, build_panel, carry_weights, run_daily, trend_weights

START = 1_704_067_200_000  # 2024-01-01 00:00 UTC (Monday)


def _panel(close, funding=None):
    T, N = close.shape
    return DailyPanel(np.arange(T, dtype=np.int64) * DAY_MS + START, [f"S{i}" for i in range(N)],
                      close, np.zeros((T, N)) if funding is None else funding)


def test_run_daily_accounting_by_hand():
    close = np.array([[100.0, 50.0], [110.0, 50.0], [110.0, 55.0]])
    fund = np.array([[0, 0], [0.001, 0.0], [0.0, 0.0]])
    w = np.array([[0.5, -0.5], [0.5, -0.5], [0, 0]])
    res = run_daily(_panel(close, fund), w, CostModel(taker_fee=0.001, half_spread=0, slippage=0))
    # day 1: +0.5*10% - (-0.5)*0% = +5% gross; long pays 0.5*0.1% funding; turnover 1.0 -> 0.1% cost
    assert res.gross[0] == pytest.approx(0.05)
    assert res.funding[0] == pytest.approx(0.0005)
    assert res.cost[0] == pytest.approx(0.001)
    assert res.net[0] == pytest.approx(0.05 - 0.0005 - 0.001)
    # day 2: drifted weights (0.55/1.05, -0.5/1.05) re-targeted to (0.5, -0.5) -> small turnover
    drift = np.array([0.55, -0.5]) / 1.05
    assert res.turnover[1] == pytest.approx(np.abs(np.array([0.5, -0.5]) - drift).sum())
    assert res.gross[1] == pytest.approx(-0.5 * 0.10)


def test_no_lookahead_on_tomorrows_availability():
    close = np.array([[100.0], [np.nan], [np.nan]])           # delisted after day 0
    res = run_daily(_panel(close), np.array([[1.0], [1.0], [1.0]]), CostModel())
    assert res.gross[0] == 0.0          # unknown price -> flat, not "avoided"
    assert res.turnover[0] == pytest.approx(1.0)              # we did buy at day 0's close
    assert res.turnover[1] == pytest.approx(1.0)              # and could only sell it later


def test_strategy_weights_are_causal():
    rng = np.random.default_rng(0)
    T, N = 400, 6
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, (T, N)), axis=0))
    fund = rng.normal(0.0001, 0.0002, (T, N))
    p = _panel(close, fund)
    w_trend = trend_weights(p)
    w_carry, _ = carry_weights(p)
    cut = 300
    close2, fund2 = close.copy(), fund.copy()
    close2[cut + 1:] *= np.exp(rng.normal(0, 0.2, (T - cut - 1, N)))
    fund2[cut + 1:] = rng.normal(0, 0.01, (T - cut - 1, N))
    p2 = _panel(close2, fund2)
    np.testing.assert_array_equal(trend_weights(p2)[:cut + 1], w_trend[:cut + 1])
    np.testing.assert_array_equal(carry_weights(p2)[0][:cut + 1], w_carry[:cut + 1])
    assert np.abs(w_trend[200:]).sum() > 0 and np.abs(w_carry).sum() > 0


def test_carry_is_dollar_neutral_and_rebalances_weekly():
    rng = np.random.default_rng(1)
    T, N = 60, 9
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, (T, N)), axis=0))
    w, reb = carry_weights(_panel(close, rng.normal(0.0001, 0.0003, (T, N))))
    assert reb.sum() >= 7
    for d in np.where(reb)[0]:
        if np.abs(w[d]).sum() > 0:
            assert w[d].sum() == pytest.approx(0.0)
            assert np.abs(w[d]).sum() == pytest.approx(1.0)


def test_build_panel_day_close_and_funding_mapping():
    ts = START + np.arange(3 * 1440, dtype=np.int64) * 60_000
    bars = {"A": pl.DataFrame({"timestamp": ts, "close": np.arange(len(ts), dtype=float)})}
    # funding at 08:00 of day 0 and at 00:00:00.005 of day 1 (belongs to day 1's interval)
    funding = {"A": (np.array([START + 8 * 3_600_000, START + DAY_MS + 5]), np.array([0.001, 0.002]))}
    p = build_panel(bars, funding)
    assert p.close[0, 0] == 1439.0 and p.close[1, 0] == 2879.0
    assert p.funding[0, 0] == pytest.approx(0.001) and p.funding[1, 0] == pytest.approx(0.002)


def test_carry_broad_is_point_in_time_and_inverse_vol():
    from core.daily import carry_broad_weights
    rng = np.random.default_rng(3)
    T, N = 120, 20
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, (T, N)), axis=0))
    close[:50, 0] = np.nan                       # listed later
    qv = rng.uniform(1e6, 1e8, (T, N))
    fund = rng.normal(0.0001, 0.0003, (T, N))
    p = DailyPanel(np.arange(T, dtype=np.int64) * DAY_MS + START, [f"S{i}" for i in range(N)], close, fund, qv)
    w, reb, elig = carry_broad_weights(p, top_n=10)
    assert not elig[:79, 0].any()                # needs 30 days of history after listing (day 50)
    assert elig.sum(axis=1).max() == 10
    for d in np.where(reb)[0]:
        if np.abs(w[d]).sum() > 0:
            assert w[d].sum() == pytest.approx(0.0) and np.abs(w[d]).sum() == pytest.approx(1.0)
            assert set(np.where(w[d] != 0)[0]) <= set(np.where(elig[d])[0])
    # causality: changing data after day 90 leaves earlier decisions untouched
    c2, q2, f2 = close.copy(), qv.copy(), fund.copy()
    c2[91:] *= 2.0
    q2[91:] = rng.uniform(1, 2, (T - 91, N))
    f2[91:] = 0.05
    w2, _, e2 = carry_broad_weights(DailyPanel(p.days, p.symbols, c2, f2, q2), top_n=10)
    np.testing.assert_array_equal(w2[:91], w[:91])
    np.testing.assert_array_equal(e2[:91], elig[:91])
