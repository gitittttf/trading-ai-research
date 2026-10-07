"""
Look-ahead tests: results at time t must not change when data AFTER t changes.
"""
import numpy as np
import polars as pl

from core.data import resample_klines
from core.pairs import rolling_zscore, select_pairs
from core.synthetic import START_MS, cointegrated_market
from strategies.baseline_pairs import BaselinePairsStrategy
from strategies.common import Window

DAY = 86_400_000


def _perturb_after(df: pl.DataFrame, cut_ms: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    mask = (df["timestamp"] >= cut_ms).to_numpy()
    factor = np.where(mask, np.exp(np.cumsum(rng.normal(0, 0.01, len(mask)) * mask)), 1.0)
    return df.with_columns([(pl.col(c) * factor).alias(c) for c in ["open", "high", "low", "close"]])


def test_rolling_zscore_is_causal():
    x = np.cumsum(np.random.default_rng(1).normal(size=2000))
    full = rolling_zscore(x, 100)
    for k in [150, 700, 1999]:
        part = rolling_zscore(x[:k], 100)
        np.testing.assert_allclose(part, full[:k], equal_nan=True)


def test_resample_bar_contains_only_its_own_minutes():
    ts = START_MS + np.arange(10, dtype=np.int64) * 60_000
    df = pl.DataFrame({"timestamp": ts, "open": np.arange(10.0) + 1, "high": np.arange(10.0) + 2,
                       "low": np.arange(10.0), "close": np.arange(10.0) + 1.5,
                       "volume": np.ones(10), "taker_buy_volume": np.ones(10) / 2})
    r = resample_klines(df, 5)
    assert r["timestamp"].to_list() == [START_MS, START_MS + 300_000]
    assert r["close"].to_list() == [5.5, 10.5]       # close of minute 4 and minute 9
    assert r["open"].to_list() == [1.0, 6.0]
    assert r["volume"].to_list() == [5.0, 5.0]


def test_pair_selection_ignores_data_after_train_end():
    bars = cointegrated_market(n_pairs=2, n_minutes=40 * 1440, seed=3)
    train_end = START_MS + 30 * DAY
    sel_a = select_pairs(bars, START_MS, train_end, top_k=5)
    perturbed = {s: _perturb_after(d, train_end, i) for i, (s, d) in enumerate(bars.items())}
    sel_b = select_pairs(perturbed, START_MS, train_end, top_k=5)
    assert [p.to_dict() for p in sel_a] == [p.to_dict() for p in sel_b]
    assert len(sel_a) >= 1


def test_baseline_trades_before_t_do_not_depend_on_data_after_t():
    bars = cointegrated_market(n_pairs=2, n_minutes=50 * 1440, seed=4)
    train_end = START_MS + 30 * DAY
    specs = select_pairs(bars, START_MS, train_end, top_k=5)
    w = Window(0, START_MS, train_end, train_end, START_MS + 50 * DAY)
    strat = BaselinePairsStrategy()
    trades_a = strat.generate(bars, specs, w)
    cut = START_MS + 40 * DAY
    perturbed = {s: _perturb_after(d, cut, i + 10) for i, (s, d) in enumerate(bars.items())}
    trades_b = strat.generate(perturbed, specs, w)
    # every trade that was fully closed before the cut must be identical
    done_a = [t for t in trades_a if t["exit_ts"] < cut]
    done_b = [t for t in trades_b if t["exit_ts"] < cut]
    assert len(done_a) > 0
    assert done_a == done_b


def test_untraded_zero_volume_minutes_are_dropped(tmp_path):
    from core.data import load_klines
    ts = START_MS + np.arange(6, dtype=np.int64) * 60_000
    pl.DataFrame({"timestamp": ts, "open": [1.0] * 6, "high": [1.0] * 6, "low": [1.0] * 6, "close": [1.0] * 6,
                  "volume": [5.0, 0.0, 3.0, 0.0, 0.0, 0.0], "taker_buy_volume": [2.0, 0, 1.0, 0, 0, 0]}
                 ).write_parquet(tmp_path / "X_USDT_klines.parquet")
    df = load_klines("X/USDT", str(tmp_path))
    assert df["timestamp"].to_list() == [int(ts[0]), int(ts[2])]
    assert load_klines("X/USDT", str(tmp_path), drop_untraded=False).height == 6
