import numpy as np
import pytest

from core.backtest import (ExitRules, PairArrays, PortfolioConfig, leg_weights, run_portfolio,
                           simulate_pair, _unit_return)
from core.costs import CostModel


def make_arrays(n=50, p1=None, p2=None):
    ts = np.arange(n, dtype=np.int64) * 300_000
    p1 = np.full(n, 10.0) if p1 is None else np.asarray(p1, float)
    p2 = np.full(n, 20.0) if p2 is None else np.asarray(p2, float)
    # distinct opens so we can see which price was used
    return PairArrays(ts=ts, open1=p1 * 1.001, close1=p1, open2=p2 * 1.001, close2=p2)


def test_unit_return_sign_and_beta_weights():
    w1, w2 = leg_weights(1.0)
    assert (w1, w2) == (0.5, 0.5)
    # long spread: long leg1 (+1%), short leg2 (flat) -> +0.5%
    assert _unit_return(1, w1, w2, 10, 20, 10.1, 20) == pytest.approx(0.005)
    assert _unit_return(-1, w1, w2, 10, 20, 10.1, 20) == pytest.approx(-0.005)
    w1, w2 = leg_weights(2.0)
    assert w1 == pytest.approx(1 / 3) and w2 == pytest.approx(2 / 3)


def test_entry_fills_on_next_bar_open_not_on_signal_close():
    arr = make_arrays()
    z = np.zeros(50)
    z[10:20] = -3.0
    sig = np.zeros(50)
    sig[10] = 1
    trades = simulate_pair(arr, z, sig, ExitRules(exit_z=0.5, stop_z=None), beta=1.0)
    assert len(trades) == 1
    t = trades[0]
    assert t["signal_idx"] == 10 and t["entry_idx"] == 11
    assert t["entry_p1"] == pytest.approx(arr.open1[11])
    # z back to 0 at bar 20 -> exit decided on close 20, filled at open 21
    assert t["exit_idx"] == 21 and t["exit_reason"] == "take_profit"
    assert t["exit_p1"] == pytest.approx(arr.open1[21])


def test_latency_delays_fills():
    arr = make_arrays()
    z = np.zeros(50)
    z[10:20] = -3.0
    sig = np.zeros(50)
    sig[10] = 1
    t = simulate_pair(arr, z, sig, ExitRules(exit_z=0.5, stop_z=None), beta=1.0, latency_bars=2)[0]
    assert t["entry_idx"] == 13 and t["exit_idx"] == 23


def test_stop_is_path_dependent_first_event_wins():
    arr = make_arrays()
    z = np.zeros(50)
    z[10:15] = -3.0
    z[15] = -4.5      # stop touched first
    z[16:30] = -0.1   # would be a take profit later
    sig = np.zeros(50)
    sig[10] = 1
    t = simulate_pair(arr, z, sig, ExitRules(exit_z=0.5, stop_z=4.0), beta=1.0)[0]
    assert t["exit_reason"] == "stop_z" and t["exit_idx"] == 16


def test_stop_loss_on_unrealized_return_uses_closes():
    n = 50
    p1 = np.full(n, 10.0)
    p1[12:] = 9.0       # leg1 falls 10% -> long spread loses ~5% per unit gross
    arr = make_arrays(n, p1=p1)
    z = np.full(n, -3.0)
    sig = np.zeros(n)
    sig[10] = 1
    t = simulate_pair(arr, z, sig, ExitRules(exit_z=0.5, stop_z=None, stop_loss=0.03), beta=1.0)[0]
    assert t["exit_reason"] == "stop_loss" and t["exit_idx"] == 13


def test_decisions_restricted_to_trade_window_and_closed_at_window_end():
    arr = make_arrays()
    z = np.full(50, -3.0)
    sig = np.zeros(50)
    sig[5] = 1      # before window: ignored
    sig[30] = 1
    trades = simulate_pair(arr, z, sig, ExitRules(exit_z=0.5, stop_z=None), beta=1.0, trade_window=(20, 40))
    assert len(trades) == 1
    assert trades[0]["signal_idx"] == 30 and trades[0]["exit_idx"] == 39
    assert trades[0]["exit_reason"] == "end_of_window"
    # priced at the close of bar 39 -> booked at the END of bar 39, not at its open
    assert trades[0]["exit_ts"] == int(arr.ts[39]) + 300_000


def test_portfolio_books_pnl_at_exit_and_sizes_from_realized_equity():
    base = {"w1": 0.5, "w2": 0.5, "side": 1, "beta": 1.0, "entry_p1": 10.0, "entry_p2": 20.0,
            "exit_p1": 10.0, "exit_p2": 20.0, "hold_minutes": 10.0, "exit_reason": "x"}
    a = dict(base, pair="A", entry_ts=0, exit_ts=100, gross_ret=0.10)
    b = dict(base, pair="B", entry_ts=50, exit_ts=200, gross_ret=0.0)
    c = dict(base, pair="C", entry_ts=150, exit_ts=300, gross_ret=0.0)
    res = run_portfolio([a, b, c], CostModel(taker_fee=0, half_spread=0, slippage=0), PortfolioConfig(
        start_equity=1000, alloc_per_trade=0.5, max_positions=5, max_gross_leverage=10),
        pair_assets={"A": ("a1", "a2"), "B": ("b1", "b2"), "C": ("c1", "c2")})
    tr = res.trades.set_index("pair")
    assert tr.loc["A", "gross_notional"] == pytest.approx(500)
    assert tr.loc["B", "gross_notional"] == pytest.approx(500)   # A's +50 not yet realized at t=50
    assert tr.loc["C", "gross_notional"] == pytest.approx(525)   # realized after A exited at t=100
    assert res.equity.iloc[-1] == pytest.approx(1050)


def test_portfolio_limits_positions():
    base = {"w1": 0.5, "w2": 0.5, "side": 1, "beta": 1.0, "entry_p1": 10.0, "entry_p2": 20.0,
            "exit_p1": 10.0, "exit_p2": 20.0, "hold_minutes": 10.0, "exit_reason": "x", "gross_ret": 0.0}
    cands = [dict(base, pair=p, entry_ts=i, exit_ts=1000) for i, p in enumerate("ABC")]
    res = run_portfolio(cands, CostModel(), PortfolioConfig(max_positions=2, alloc_per_trade=0.1),
                        pair_assets={p: ("x", "y") for p in "ABC"})
    assert len(res.trades) == 2 and res.rejected == 1
