"""
Live/backtest parity: feeding bars one at a time into core.streaming must produce
the same entry and exit decisions as the vectorized backtest on the same bars.
"""
import numpy as np

from core.features import rolling_adf_pvalue
from core.pairs import select_pairs
from core.streaming import StreamingConfig, StreamingPair
from core.synthetic import START_MS, cointegrated_market
from strategies.baseline_pairs import BaselinePairsStrategy
from strategies.common import Window
from strategies.xgb_spread import XGBParams, XGBSpreadStrategy

DAY = 86_400_000


def _bar(df, i):
    return {k: df[k][i] for k in ["timestamp", "open1", "high1", "low1", "close1", "volume1", "taker_buy_volume1",
                                  "open2", "high2", "low2", "close2", "volume2", "taker_buy_volume2"]}


def _decisions_from_trades(trades, ts):
    out = []
    for t in trades:
        out.append(("enter", int(ts[t["signal_idx"]]), t["side"]))
        if not t["exit_reason"].endswith("window") and not t["exit_reason"].endswith("window_end"):
            out.append(("exit", int(ts[t["exit_idx"] - 1]), t["exit_reason"]))
    return out


def _stream(sp: StreamingPair, df, lo, hi, last_exit_ok=True):
    out = []
    for i in range(hi):
        d = sp.on_bar(_bar(df, i), warmup_only=i < lo)
        if d is None:
            continue
        if d["action"] == "enter":
            if i >= hi - 1:   # backtest cannot fill an entry decided on the last bar
                sp.force_flat()
                continue
            out.append(("enter", d["ts"], d["side"]))
        else:
            out.append(("exit", d["ts"], d["reason"]))
    return out


def test_baseline_streaming_matches_backtest():
    bars = cointegrated_market(n_pairs=2, n_minutes=45 * 1440, seed=8)
    tr_end = START_MS + 30 * DAY
    spec = select_pairs(bars, START_MS, tr_end, top_k=1)[0]
    w = Window(0, START_MS, tr_end, tr_end, START_MS + 45 * DAY)
    strat = BaselinePairsStrategy()
    prep = strat.prepare(bars, spec, w)
    trades = strat.candidates_for_pair(prep)
    assert len(trades) >= 10
    expected = _decisions_from_trades(trades, prep.arrays.ts)
    p = strat.params
    sp = StreamingPair(spec, StreamingConfig(strategy="baseline", bar_minutes=p.bar_minutes,
                                             z_window_minutes=p.z_window_minutes, entry_z=p.entry_z,
                                             rules=strat.exit_rules(spec)))
    got = _stream(sp, prep.bars, *prep.trade_window)
    # the backtest force-closes at the window end; compare everything before that
    assert got[:len(expected)] == expected


import pytest


@pytest.mark.parametrize("adf", [False, True])
def test_xgb_streaming_matches_backtest(adf):
    bars = cointegrated_market(n_pairs=2, n_minutes=50 * 1440, seed=9)
    tr_end = START_MS + 35 * DAY
    specs = select_pairs(bars, START_MS + 3 * DAY, tr_end, top_k=2)
    w = Window(0, START_MS + 3 * DAY, tr_end, tr_end, START_MS + 50 * DAY)
    strat = XGBSpreadStrategy(XGBParams(n_models=2, adf_feature=adf, sit_out_if_negative=False,
                                        xgb_params=dict(n_estimators=60, max_depth=3, learning_rate=0.1,
                                                        tree_method="hist", n_jobs=1)))
    cands = strat.generate(bars, specs, w, btc_symbol=None)
    spec = specs[0]
    trades = [c for c in cands if c["pair"] == spec.name]
    assert len(trades) >= 5
    pd_ = strat._pair_data(bars, spec, w.test_start, w.test_end, None)   # what the strategy trades on
    import polars as pl
    df = pl.DataFrame({"timestamp": pd_.arrays.ts, "open1": pd_.arrays.open1, "close1": pd_.arrays.close1,
                       "open2": pd_.arrays.open2, "close2": pd_.arrays.close2})
    # rebuild the full aligned bar frame the strategy used (volumes etc. are needed for features)
    from core.data import align_pair, resample_klines, time_slice
    lo_ms = w.test_start - (2 * 1440 + 120) * 60_000
    full = align_pair(resample_klines(time_slice(bars[spec.asset1], lo_ms, w.test_end), 5),
                      resample_klines(time_slice(bars[spec.asset2], lo_ms, w.test_end), 5))
    assert full["timestamp"].to_list() == df["timestamp"].to_list()
    ts = pd_.arrays.ts
    lo, hi = int(np.searchsorted(ts, w.test_start)), int(np.searchsorted(ts, w.test_end))
    rt = strat.costs.round_trip_rate()
    cfg = StreamingConfig(strategy="xgb", bar_minutes=5, entry_z=strat.chosen["entry_z"],
                          rules=strat._exit_rules(spec), need_edge=strat.chosen["cost_mult"] * rt,
                          penalty=strat.params.penalty, use_btc=False, adf=adf, history_bars=1500)
    sp = StreamingPair(spec, cfg, models=strat.models)
    got = _stream(sp, full, lo, hi)
    expected = _decisions_from_trades(trades, ts)
    assert got[:len(expected)] == expected


def test_clock_aligned_adf_is_identical_on_a_truncated_buffer():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=3000)) * 0.01
    ts = START_MS + np.arange(3000, dtype=np.int64) * 300_000
    full = rolling_adf_pvalue(x, 288, ts, 3_600_000)
    part = rolling_adf_pvalue(x[1000:], 288, ts[1000:], 3_600_000)
    # after the buffer's own warm-up plus one clock hour, values must match exactly
    k = 1000 + 288 + 12
    np.testing.assert_array_equal(part[k - 1000:], full[k:])


def test_xgb_test_features_do_not_depend_on_history_length():
    """window_data computes test features from test_start (+ warm-up) instead of the
    training start; every feature value inside the test window must be identical."""
    bars = cointegrated_market(n_pairs=1, n_minutes=20 * 1440, seed=10)
    spec = select_pairs(bars, START_MS, START_MS + 10 * DAY, top_k=1)[0]
    strat = XGBSpreadStrategy(XGBParams(adf_feature=True))
    test_start, test_end = START_MS + 14 * DAY, START_MS + 20 * DAY
    long_ = strat._pair_data(bars, spec, START_MS + 3 * DAY, test_end, None)
    short = strat._pair_data(bars, spec, test_start, test_end, None)
    a = long_.feats.filter(long_.feats["timestamp"] >= test_start)
    b = short.feats.filter(short.feats["timestamp"] >= test_start)
    assert a["timestamp"].to_list() == b["timestamp"].to_list()
    for c in a.columns:
        np.testing.assert_allclose(a[c].to_numpy().astype(float), b[c].to_numpy().astype(float),
                                   rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=c)
