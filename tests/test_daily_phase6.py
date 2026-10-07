"""Phase 6 (protocol variants 12-14): hand-checked rules and causality of the daily strategies."""
import numpy as np
import polars as pl
import pytest

from core.costs import CostModel
from core.daily import (DAY_MS, DailyPanel, add_spot_columns, harvest_sized, harvest_weights, listing_short_weights,
                        run_daily,
                        xs_momentum_weights)

START = 1_704_067_200_000  # 2024-01-01 (Monday)


def _panel(close, funding=None, qv=None, symbols=None):
    T, N = close.shape
    return DailyPanel(np.arange(T, dtype=np.int64) * DAY_MS + START, symbols or [f"S{i}USDT" for i in range(N)],
                      close, np.zeros((T, N)) if funding is None else funding,
                      np.ones((T, N)) * 1e6 if qv is None else qv)


def test_run_daily_per_column_cost_rates():
    close = np.array([[100.0, 100.0], [100.0, 100.0], [100.0, 100.0]])
    w = np.array([[-0.5, 0.5], [-0.5, 0.5], [0.0, 0.0]])
    res = run_daily(_panel(close), w, CostModel(multiplier=2.0), col_rate=np.array([0.0005, 0.001]))
    assert res.cost[0] == pytest.approx(2.0 * (0.5 * 0.0005 + 0.5 * 0.001))
    assert res.cost[1] == pytest.approx(0.0)


def test_add_spot_columns_maps_same_named_pairs():
    close = np.full((3, 2), 10.0)
    p = _panel(close, symbols=["AUSDT", "BUSDT"])
    spot = {"BUSDT": pl.DataFrame({"timestamp": p.days[1:], "close": [9.0, 9.5], "quote_volume": [1.0, 0.0]})}
    wide, m = add_spot_columns(p, spot)
    assert wide.symbols == ["AUSDT", "BUSDT", "SPOT:BUSDT"] and m == {1: 2}
    assert np.isnan(wide.close[0, 2]) and wide.close[1, 2] == 9.0
    assert np.isnan(wide.close[2, 2])            # zero volume = not traded
    assert np.all(wide.funding[:, 2] == 0)


def _harvest_market(T=120, N=4, seed=0):
    rng = np.random.default_rng(seed)
    perp = 50 * np.exp(np.cumsum(rng.normal(0, 0.03, (T, N)), axis=0))
    fund = np.full((T, N), 0.0001)                # quiet: 0.07% per week
    fund[40:70, 1] = 0.001                        # coin 1: 0.7% per week -> enter
    fund[50:90, 2] = 0.0006                       # coin 2: 0.42% per week -> enter
    p = _panel(perp, fund)
    spot = {s: pl.DataFrame({"timestamp": p.days, "close": perp[:, j], "quote_volume": np.ones(T)})
            for j, s in enumerate(p.symbols) if j != 3}          # coin 3 has no spot market
    return add_spot_columns(p, spot)


def test_harvest_rules_by_hand():
    wide, m = _harvest_market()
    w, reb, eps = harvest_weights(wide, m, n_perp=4, top_n=60, max_pos=10)
    n = 1 / 15
    f7 = lambda d, j: wide.funding[d - 6:d + 1, j].sum()
    first1 = next(d for d in range(30, 120) if f7(d, 1) >= 0.0035)
    assert w[first1 - 1, 1] == 0 and w[first1, 1] == pytest.approx(-n) and w[first1, m[1]] == pytest.approx(n)
    exit1 = next(d for d in range(first1, 120) if f7(d, 1) < 0.0015)
    assert w[exit1 - 1, 1] == pytest.approx(-n) and w[exit1, 1] == 0 and w[exit1, m[1]] == 0
    assert np.all(w[:, 0] == 0) and np.all(w[:, 3] == 0)        # quiet coin / no spot market
    assert np.allclose(w.sum(axis=1), 0.0)                      # delta neutral every day
    assert reb[first1] and reb[exit1] and not reb[first1 + 1]
    assert (1, first1, exit1) in eps


def test_harvest_pnl_is_funding_minus_costs_when_spot_equals_perp():
    wide, m = _harvest_market()
    w, reb, _ = harvest_weights(wide, m, n_perp=4)
    rates = np.where(np.array([s.startswith("SPOT:") for s in wide.symbols]), 0.00115, 0.00065)
    res = run_daily(wide, w, CostModel(), reb, col_rate=rates)
    assert np.allclose(res.gross, 0.0, atol=1e-12)              # identical paths: price risk cancels
    assert res.funding.sum() < 0                                # the short perp RECEIVES funding
    assert res.net.sum() == pytest.approx(-res.funding.sum() - res.cost.sum())


def test_harvest_max_positions_and_benchmark_keeps_timing():
    wide, m = _harvest_market()
    w, reb, eps = harvest_weights(wide, m, n_perp=4, max_pos=1)
    assert np.max(np.sum(w[:, :4] != 0, axis=1)) == 1
    wb, _, _ = harvest_weights(wide, m, n_perp=4, max_pos=1, rng=np.random.default_rng(0), episodes=eps)
    held_real = np.sum(w[:, :4] != 0, axis=1)
    held_rand = np.sum(wb[:, :4] != 0, axis=1)
    assert np.array_equal(held_real > 0, held_rand > 0)
    assert np.allclose(wb.sum(axis=1), 0.0)


def test_listing_short_events_hedge_and_cap():
    T, N = 80, 6
    close = np.full((T, N), 10.0)
    close[:20, 2] = np.nan            # listed on day 20
    close[:25, 3] = np.nan            # listed on day 25
    close[:25, 4] = np.nan            # listed on day 25 too, but the cap is 1 -> skipped
    close[40:, 3] = np.nan            # delisted on day 40
    p = _panel(close)
    w, reb, ev = listing_short_weights(p, {2: 20, 3: 25, 4: 25}, btc_col=0, hold_days=30, size=0.1, max_events=2)
    assert ev == [(2, 21, 51), (3, 26, 40)]
    assert w[20, 2] == 0 and w[21, 2] == pytest.approx(-0.1) and w[50, 2] == pytest.approx(-0.1) and w[51, 2] == 0
    assert w[39, 3] == pytest.approx(-0.1) and w[40, 3] == 0
    assert np.allclose(w[:, 0], -w[:, 1:].sum(axis=1))           # BTC long = sum of shorts
    assert np.all(w[:, 4] == 0)


def test_momentum_legs_and_neutrality():
    rng = np.random.default_rng(2)
    T, N = 150, 30
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, (T, N)), axis=0))
    p = _panel(close, qv=rng.uniform(1e6, 1e8, (T, N)))
    w, reb, elig = xs_momentum_weights(p, top_n=20)
    d = int(np.where(reb & (np.abs(w).sum(axis=1) > 0))[0][-1])
    assert w[d].sum() == pytest.approx(0.0) and np.abs(w[d]).sum() == pytest.approx(1.0)
    mom = np.log(close[d] / close[d - 28])
    longs, shorts = np.where(w[d] > 0)[0], np.where(w[d] < 0)[0]
    assert mom[longs].min() > mom[shorts].max()
    assert set(longs) | set(shorts) <= set(np.where(elig[d])[0])


@pytest.mark.parametrize("which", ["harvest", "harvest_sized", "listing", "momentum"])
def test_phase6_weights_are_causal(which):
    rng = np.random.default_rng(5)
    T, N = 160, 12
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.03, (T, N)), axis=0))
    fund = rng.normal(0.0003, 0.0004, (T, N))
    qv = rng.uniform(1e6, 1e8, (T, N))
    close[:70, 5] = np.nan
    cut = 110

    def build(c, f, q):
        p = _panel(c, f, q)
        if which.startswith("harvest"):
            spot = {s: pl.DataFrame({"timestamp": p.days, "close": c[:, j] * 1.001, "quote_volume": q[:, j]})
                    for j, s in enumerate(p.symbols) if j % 2 == 0 and j != 5}
            wide, m = add_spot_columns(p, spot)
            w, reb, _ = harvest_weights(wide, m, n_perp=N, top_n=8)
            return w if which == "harvest" else np.column_stack([*harvest_sized(wide, w, reb, N)[:2]])
        if which == "listing":
            return listing_short_weights(p, {5: 70}, btc_col=0)[0]
        return xs_momentum_weights(p, top_n=10)[0]

    w = build(close, fund, qv)
    c2, f2, q2 = close.copy(), fund.copy(), qv.copy()
    c2[cut + 1:] *= np.exp(rng.normal(0, 0.3, (T - cut - 1, N)))
    f2[cut + 1:] = rng.normal(0, 0.01, (T - cut - 1, N))
    q2[cut + 1:] = rng.uniform(1, 2, (T - cut - 1, N))
    w2 = build(c2, f2, q2)
    np.testing.assert_array_equal(w2[:cut + 1], w[:cut + 1])
    assert np.abs(w[:cut + 1]).sum() > 0


def test_ruin_is_minus_100_percent_and_final():
    # short 50% of equity in a coin that goes up 3x: loss 100% -> liquidated, nothing after
    close = np.array([[1.0, 1.0], [3.0, 1.0], [0.5, 1.0], [0.5, 1.0]])
    w = np.array([[-0.5, 0.0], [-0.5, 0.0], [-0.5, 0.0], [0.0, 0.0]])
    res = run_daily(_panel(close), w, CostModel(), np.array([True, False, False, True]))
    assert res.net[0] == -1.0 and np.all(res.net[1:] == 0.0)
    assert np.prod(1 + res.net) == 0.0


def test_liquidation_when_drift_makes_leverage_extreme():
    # equity survives the day (-95%) but the short is now ~29x equity -> liquidated
    close = np.array([[1.0, 1.0], [2.9, 1.0], [2.9, 1.0]])
    w = np.array([[-0.5, 0.0], [-0.5, 0.0], [0.0, 0.0]])
    res = run_daily(_panel(close), w, CostModel(taker_fee=0, half_spread=0, slippage=0),
                    np.array([True, False, True]))
    assert res.net[0] == -1.0 and res.net[1] == 0.0
