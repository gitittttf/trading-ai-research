"""
Look-ahead and null tests for the ML spread strategy.
"""
import numpy as np
import polars as pl

from core.data import align_pair, resample_klines
from core.features import FEATURE_COLS, build_pair_features, forward_unit_return
from core.pairs import PairSpec, select_pairs
from core.synthetic import START_MS, cointegrated_market, random_walk_market
from strategies.common import Window
from strategies.xgb_spread import XGBParams, XGBSpreadStrategy

DAY = 86_400_000


def _same(a: list[dict], b: list[dict]) -> bool:
    """Trade lists equal, treating NaN == NaN (dict == would say NaN != NaN)."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x.keys() != y.keys():
            return False
        for k in x:
            u, v = x[k], y[k]
            if isinstance(u, float) and isinstance(v, float) and np.isnan(u) and np.isnan(v):
                continue
            if u != v:
                return False
    return True


def _perturb_after(df, cut_ms, seed):
    rng = np.random.default_rng(seed)
    mask = (df["timestamp"] >= cut_ms).to_numpy()
    factor = np.where(mask, np.exp(np.cumsum(rng.normal(0, 0.01, len(mask)) * mask)), 1.0)
    return df.with_columns([(pl.col(c) * factor).alias(c) for c in ["open", "high", "low", "close"]] +
                           [(pl.col("taker_buy_volume") * np.where(mask, 0.3, 1.0)).alias("taker_buy_volume")])


def test_features_at_t_ignore_data_after_t():
    bars = cointegrated_market(n_pairs=1, n_minutes=6 * 1440, seed=5)
    a, b = "P0A/USDT", "P0B/USDT"
    spec = PairSpec(a, b, 0.3, 1.0, 0.01, 240.0, 0.9, START_MS, START_MS + DAY)
    cut = START_MS + 4 * DAY
    def feats(bs):
        df = align_pair(resample_klines(bs[a], 5), resample_klines(bs[b], 5))
        return build_pair_features(df, spec, 5, btc_close=df["close2"].to_numpy())[0]
    fa = feats(bars)
    fb = feats({a: _perturb_after(bars[a], cut, 1), b: _perturb_after(bars[b], cut, 2)})
    before = (fa["timestamp"] < cut).to_numpy()
    for c in FEATURE_COLS:
        np.testing.assert_allclose(fa[c].to_numpy()[before].astype(float), fb[c].to_numpy()[before].astype(float),
                                   equal_nan=True, err_msg=c)
    # and the perturbation really did change something after the cut
    assert not np.allclose(fa["z_24h"].to_numpy()[~before], fb["z_24h"].to_numpy()[~before], equal_nan=True)


def test_forward_target_is_only_future_and_nan_at_end():
    s = np.arange(10.0)
    y = forward_unit_return(s, 1.0, 3)
    assert np.all(np.isnan(y[-3:]))
    np.testing.assert_allclose(y[:-3], 3.0 / 2.0)


def test_xgb_trades_before_cut_do_not_depend_on_later_data():
    bars = cointegrated_market(n_pairs=2, n_minutes=55 * 1440, seed=6)
    tr_end = START_MS + 35 * DAY
    specs = select_pairs(bars, START_MS + 3 * DAY, tr_end, top_k=3)
    w = Window(0, START_MS + 3 * DAY, tr_end, tr_end, START_MS + 55 * DAY)
    strat = XGBSpreadStrategy(XGBParams(n_models=2, adf_feature=False,
                                        xgb_params=dict(n_estimators=60, max_depth=3, learning_rate=0.1,
                                                        tree_method="hist", n_jobs=2)))
    cut = START_MS + 45 * DAY
    ta = strat.generate(bars, specs, w, btc_symbol=None)
    tb = strat.generate({s: _perturb_after(d, cut, i) for i, (s, d) in enumerate(bars.items())}, specs, w,
                        btc_symbol=None)
    done_a = [t for t in ta if t["exit_ts"] < cut]
    done_b = [t for t in tb if t["exit_ts"] < cut]
    assert len(done_a) > 0
    assert _same(done_a, done_b)


def _null_run(bars, sit_out: bool):
    strat = XGBSpreadStrategy(XGBParams(n_models=2, adf_feature=False, cost_mult_grid=(0.0,),
                                        entry_z_grid=(1.5,), sit_out_if_negative=sit_out, min_val_trades=1,
                                        validation_selection=False,
                                        xgb_params=dict(n_estimators=60, max_depth=3, learning_rate=0.1,
                                                        tree_method="hist", n_jobs=2)))
    gross = []
    syms = sorted(bars)
    for k in range(4):
        tr_s = START_MS + k * 20 * DAY
        w = Window(k, tr_s, tr_s + 40 * DAY, tr_s + 40 * DAY, tr_s + 60 * DAY)
        specs = [PairSpec(syms[i], syms[j], 0.0, 1.0, 0.5, 240.0, 0.5, w.train_start, w.train_end)
                 for i in range(len(syms)) for j in range(i + 1, len(syms))]
        gross += [c["gross_ret"] for c in strat.generate(bars, specs, w, btc_symbol=None)]
    return np.array(gross)


def test_xgb_null_random_walk_no_gross_edge():
    bars = random_walk_market(n_assets=5, n_minutes=100 * 1440, seed=21)
    # forced to trade: the ML pipeline must not find a (fake) edge on random walks
    gross = _null_run(bars, sit_out=False)
    assert len(gross) >= 100
    se = gross.std(ddof=1) / np.sqrt(len(gross))
    assert abs(gross.mean()) < 3 * se, f"gross edge on random walks: {gross.mean()*1e4:.2f}bp (se {se*1e4:.2f})"
    # with the default validation gate it should simply refuse to trade most windows
    assert len(_null_run(bars, sit_out=True)) < len(gross)
