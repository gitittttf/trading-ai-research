"""
Passive (maker) fill model of protocol variants 10/11.

Hand-built bars check every rule; synthetic markets check that the model creates no
edge on random walks and does not look ahead.
"""
import numpy as np
import pytest

from core.backtest import (Execution, ExitRules, PairArrays, PortfolioConfig, maker_fill_bar, run_portfolio,
                           simulate_pair, trade_costs)
from core.costs import CostModel
from core.metrics import bootstrap_mean_ci
from core.synthetic import START_MS, cointegrated_market, random_walk_market
from strategies.baseline_pairs import BaselineParams, BaselinePairsStrategy
from strategies.common import make_windows

BAR = 300_000
DAY = 86_400_000
EX = Execution("maker", timeout_bars=2, through_bp=1.0)
RULES = ExitRules(exit_z=0.5, stop_z=4.0)


def _arr(n, p1=100.0, p2=50.0):
    """Flat bars (high = low = close = open); tests then set single highs/lows."""
    c1, c2 = np.full(n, p1), np.full(n, p2)
    return PairArrays(ts=START_MS + np.arange(n, dtype=np.int64) * BAR, open1=c1.copy(), close1=c1.copy(),
                      open2=c2.copy(), close2=c2.copy(), vol1_usd=np.full(n, 1e9), vol2_usd=np.full(n, 1e9),
                      high1=c1.copy(), low1=c1.copy(), high2=c2.copy(), low2=c2.copy())


def _true_spec():
    """The synthetic pair's true hedge (log pA = log pB + 0.3 + OU), no estimation needed."""
    from core.pairs import PairSpec
    return PairSpec("P0A/USDT", "P0B/USDT", 0.3, 1.0, 0.0, 240.0, 0.9, START_MS, START_MS + 3 * DAY)


def _signal(n, t, side):
    s = np.zeros(n)
    s[t] = side
    return s


def test_fill_needs_a_trade_through_not_a_touch():
    a = _arr(6)
    a.low1[1] = 100 * (1 - 0.00005)      # 0.5 bp through: not enough
    a.low1[2] = 100 * (1 - 0.0001)       # exactly 1 bp: not strictly through
    a.low1[3] = 100 * (1 - 0.00011)      # 1.1 bp through: filled
    assert maker_fill_bar(a, 1, True, 100.0, 1, 5, 1.0) == 3
    assert maker_fill_bar(a, 1, True, 100.0, 1, 2, 1.0) == -1          # outside the timeout
    a.high2[4] = 50 * (1 + 0.0002)
    assert maker_fill_bar(a, 2, False, 50.0, 1, 5, 1.0) == 4            # sell fills on the high
    assert maker_fill_bar(a, 2, True, 50.0, 1, 5, 1.0) == -1            # a buy never fills on a high


def test_entry_both_legs_maker_then_passive_take_profit():
    n = 12
    a = _arr(n)
    z = np.full(n, -2.5)
    a.low1[3] = 99.9            # long spread: buy leg1 ...
    a.high2[4] = 50.1           # ... and sell leg2, both within the 2-bar timeout after t=2
    z[6:] = 0.0                 # back inside +-0.5 at the close of bar 6 -> take profit
    a.close1[6], a.close2[6] = 101.0, 50.0
    a.high1[7] = 101.2          # exit: sell leg1, buy leg2 back, both fill in bar 7
    a.low2[7] = 49.9
    tr = simulate_pair(a, z, _signal(n, 2, 1), RULES, beta=1.0, execution=EX)
    assert len(tr) == 1
    t = tr[0]
    assert (t["entry_idx"], t["exit_decision_idx"], t["exit_idx"]) == (4, 6, 7)
    assert (t["entry_p1"], t["entry_p2"]) == (100.0, 50.0)          # filled at the limits (bar 2 close)
    assert (t["exit_p1"], t["exit_p2"]) == (101.0, 50.0)            # filled at the bar 6 close
    assert t["entry_maker1"] and t["entry_maker2"] and t["exit_maker1"] and t["exit_maker2"]
    assert t["exit_ts"] == a.ts[7] + BAR - 1                       # inside bar 7, not before it
    assert t["gross_ret"] == pytest.approx(0.5 * 0.01)
    assert t["exit_reason"] == "take_profit"


def test_one_leg_filled_other_completed_as_taker():
    n = 12
    a = _arr(n)
    z = np.full(n, 2.5)          # short spread: sell leg1, buy leg2
    a.high1[3] = 100.2           # leg1 fills; leg2 never trades through
    a.open2[5] = 50.3            # taker completion at the open of bar t + timeout + 1 = 5
    tr = simulate_pair(a, z, _signal(n, 2, -1), RULES, beta=1.0, execution=EX)
    t = tr[0]
    assert t["entry_idx"] == 5
    assert (t["entry_p1"], t["entry_p2"]) == (100.0, 50.3)
    assert t["entry_maker1"] and not t["entry_maker2"]
    assert t["exit_reason"] == "end_of_window" and not t["exit_maker1"] and not t["exit_maker2"]


def test_unfilled_entry_expires_and_blocks_decisions_while_resting():
    n = 12
    a = _arr(n)                  # flat market: nothing ever trades through
    z = np.full(n, -2.5)
    sig = np.zeros(n)
    sig[[2, 3, 4]] = 1           # 3 is inside the resting period of the order placed at 2
    tr = simulate_pair(a, z, sig, RULES, beta=1.0, execution=EX)
    assert tr == []
    a.low1[5], a.high2[6] = 99.0, 51.0       # the decision at 4 (= 2 + timeout) can fill
    tr = simulate_pair(a, z, sig, RULES, beta=1.0, execution=EX)
    assert [t["signal_idx"] for t in tr] == [4] and tr[0]["entry_idx"] == 6


def test_stop_exit_is_always_taker_at_next_open():
    n = 12
    a = _arr(n)
    z = np.full(n, -2.5)
    a.low1[3], a.high2[3] = 99.0, 51.0
    z[5:] = -4.5                 # stop at the close of bar 5
    a.open1[6], a.open2[6] = 97.0, 51.0
    t = simulate_pair(a, z, _signal(n, 2, 1), RULES, beta=1.0, execution=EX)[0]
    assert t["exit_reason"] == "stop_z" and t["exit_idx"] == 6
    assert (t["exit_p1"], t["exit_p2"]) == (97.0, 51.0)
    assert not t["exit_maker1"] and not t["exit_maker2"]


def test_take_profit_waiting_at_window_end_falls_back_to_last_close():
    n = 8
    a = _arr(n)
    z = np.full(n, -2.5)
    a.low1[3], a.high2[3] = 99.0, 51.0
    z[6:] = 0.0                  # take profit decided at 6, timeout would need bars 7..8, 9 for taker
    a.close1[7], a.close2[7] = 100.5, 50.0
    t = simulate_pair(a, z, _signal(n, 2, 1), RULES, beta=1.0, execution=EX)[0]
    assert t["exit_reason"] == "take_profit_at_window_end"
    assert t["exit_idx"] == 7 and (t["exit_p1"], t["exit_p2"]) == (100.5, 50.0)
    assert t["exit_ts"] == a.ts[7] + BAR


def test_costs_per_fill_follow_the_maker_flags():
    trade = {"w1": 0.5, "w2": 0.5, "side": 1, "entry_p1": 10.0, "entry_p2": 20.0, "exit_p1": 10.0,
             "exit_p2": 20.0, "entry_ts": 0, "exit_ts": 1, "entry_vol1_usd": 1e4, "entry_vol2_usd": 1e4,
             "exit_vol1_usd": 1e4, "exit_vol2_usd": 1e4,
             "entry_maker1": True, "entry_maker2": False, "exit_maker1": True, "exit_maker2": True}
    c = CostModel()
    execution, _ = trade_costs(trade, 1000.0, c, None, "A", "B")
    taker_leg = 500 * 0.00065 + 500 * 0.001 * np.sqrt(500 / 1e4)       # fee+spread+slippage + impact
    assert execution == pytest.approx(3 * 500 * 0.0002 + taker_leg)
    # stress multiplier applies to maker fills as well
    ex15, _ = trade_costs(trade, 1000.0, CostModel(multiplier=1.5), None, "A", "B")
    assert ex15 == pytest.approx(1.5 * execution)


def test_taker_path_is_unchanged_by_the_execution_argument():
    bars = cointegrated_market(n_pairs=1, n_minutes=6 * 1440, seed=3)
    from strategies.common import prepare_pair
    spec = _true_spec()
    prep = prepare_pair(bars, spec, START_MS + 3 * DAY, START_MS + 6 * DAY, 5, 1440, 1450)
    from strategies.common import crossing_entries
    sig = crossing_entries(prep.z, 2.0)
    a = simulate_pair(prep.arrays, prep.z, sig, RULES, spec.beta, trade_window=prep.trade_window)
    b = simulate_pair(prep.arrays, prep.z, sig, RULES, spec.beta, trade_window=prep.trade_window,
                      execution=Execution("taker"))
    assert a == b and len(a) > 0


def test_maker_fills_do_not_look_ahead():
    bars = cointegrated_market(n_pairs=1, n_minutes=8 * 1440, seed=5)
    from strategies.common import crossing_entries, prepare_pair
    spec = _true_spec()
    prep = prepare_pair(bars, spec, START_MS + 3 * DAY, START_MS + 8 * DAY, 5, 1440, 1450)
    sig = crossing_entries(prep.z, 1.5)
    ex = Execution("maker", timeout_bars=3)
    base = simulate_pair(prep.arrays, prep.z, sig, RULES, spec.beta, trade_window=prep.trade_window, execution=ex)
    cut = prep.trade_window[0] + (prep.trade_window[1] - prep.trade_window[0]) // 2
    arr = PairArrays(**{k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in vars(prep.arrays).items()})
    rng = np.random.default_rng(0)
    shock = np.exp(rng.normal(0, 0.05, len(arr) - cut - 1))
    for name in ["open1", "close1", "high1", "low1", "open2", "close2", "high2", "low2"]:
        getattr(arr, name)[cut + 1:] *= shock
    z2 = prep.z.copy()
    z2[cut + 1:] = rng.normal(0, 3, len(z2) - cut - 1)
    sig2 = sig.copy()
    sig2[cut + 1:] = rng.choice([-1, 0, 1], len(sig2) - cut - 1)
    pert = simulate_pair(arr, z2, sig2, RULES, spec.beta, trade_window=prep.trade_window, execution=ex)
    done = [t for t in base if t["exit_idx"] <= cut]
    assert len(done) >= 3
    assert [t for t in pert if t["exit_idx"] <= cut] == done


def _maker_run(bars, days, spec_fn):
    strat = BaselinePairsStrategy(BaselineParams(exec_mode="maker", maker_timeout_bars=3))
    cands, pair_assets = [], {}
    for w in make_windows(START_MS, START_MS + days * DAY, train_days=20, test_days=10):
        specs = spec_fn(bars, w.train_start, w.train_end)
        for s in specs:
            pair_assets[s.name] = (s.asset1, s.asset2)
        cands += strat.generate(bars, specs, w)
    return cands, pair_assets


def test_null_random_walk_maker_has_no_gross_edge():
    from test_null_and_positive_control import _forced_specs
    bars = random_walk_market(n_assets=6, n_minutes=120 * 1440, seed=11)
    cands, pair_assets = _maker_run(bars, 120, _forced_specs)
    gross = np.array([c["gross_ret"] for c in cands])
    assert len(gross) > 150
    mean, lo, hi = bootstrap_mean_ci(gross, n_boot=4000)
    assert lo <= 0.0, f"maker fill model creates an edge on random walks: mean {mean*1e4:.2f}bp"
    res = run_portfolio(cands, CostModel(), PortfolioConfig(), pair_assets)
    assert res.trades["net_pnl"].sum() < 0


def test_positive_control_ou_spreads_still_profitable_with_maker_fills():
    from core.pairs import select_pairs
    bars = cointegrated_market(n_pairs=3, n_minutes=80 * 1440, seed=12, half_life_minutes=240)
    cands, pair_assets = _maker_run(bars, 80, lambda b, s, e: select_pairs(b, s, e, top_k=5))
    gross = np.array([c["gross_ret"] for c in cands])
    assert len(gross) > 50
    mean, lo, hi = bootstrap_mean_ci(gross, n_boot=4000)
    assert lo > 0.0
