"""
Causal feature engineering for the ML spread strategy (any bar size).

Design rules:
* every window is given in MINUTES and converted to bars, so 1m/5m/15m all work;
* features are computed on the beta-hedged LOG spread of the selected pair, not
  on the raw price ratio p1/p2;
* correlation is computed on RETURNS (correlating price levels is meaningless);
* only features backed by real data are built: there are no mocked OI/funding
  inputs, and funding enters only when real funding data is provided;
* nothing is zero-filled silently: warm-up rows stay NaN and are reported via
  ``warmup_bars`` so training and live code can skip them.
Every rolling statistic at bar t uses bars <= t only (tested in tests/test_features_and_xgb.py).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from core.pairs import PairSpec


def _bars(minutes: float, bar_minutes: int) -> int:
    return max(1, int(round(minutes / bar_minutes)))


def _adf_p(adfuller, seg: np.ndarray) -> float:
    try:  # statsmodels >= 0.15 warns unless the return type is chosen explicitly
        return float(adfuller(seg, maxlag=1, regression="c", autolag=None, result_object=False)[1])
    except TypeError:
        return float(adfuller(seg, maxlag=1, regression="c", autolag=None)[1])


def rolling_adf_pvalue(x: np.ndarray, window: int, ts: np.ndarray, step_ms: int) -> np.ndarray:
    """ADF p-value of x[t-window+1..t], re-evaluated on the first bar of every
    ``step_ms`` clock interval (e.g. every full hour) and forward filled.

    Evaluation times are tied to the CLOCK, not to the array start, so a live
    buffer that starts at a different bar produces the same values as the full
    backtest array (needed for live/backtest parity)."""
    from statsmodels.tsa.stattools import adfuller
    n = len(x)
    out = np.full(n, np.nan)
    last = np.nan
    bucket = ts // step_ms
    for i in range(n):
        if i >= window - 1 and (i == 0 or bucket[i] != bucket[i - 1]):
            seg = x[i - window + 1:i + 1]
            if np.all(np.isfinite(seg)) and np.std(seg) > 0:
                try:
                    last = _adf_p(adfuller, seg)
                except Exception:
                    last = np.nan
            else:
                last = np.nan
        out[i] = last
    return out


FEATURE_COLS = [
    "z_4h", "z_12h", "z_24h", "z_mom_1", "z_mom_3",
    "spread_vol_1h", "spread_vol_ratio",
    "ret_corr_5h", "trend1_24h", "trend2_24h",
    "dollar_vol_ratio_log", "of_imb_spread_1h", "of_imb_spread_norm",
    "btc_trend_24h", "btc_vol_24h",
    "adf_pvalue_24h",
]
FUNDING_FEATURE_COLS = ["funding_diff"]
BTC_FEATURE_COLS = ["btc_trend_24h", "btc_vol_24h"]


def active_feature_cols(with_btc: bool, with_adf: bool, with_funding: bool = False) -> list[str]:
    """Feature columns that are actually computed for this configuration (no all-NaN columns)."""
    cols = [c for c in FEATURE_COLS
            if (with_btc or c not in BTC_FEATURE_COLS) and (with_adf or c != "adf_pvalue_24h")]
    return cols + (FUNDING_FEATURE_COLS if with_funding else [])


def build_pair_features(df: pl.DataFrame, spec: PairSpec, bar_minutes: int,
                        btc_close: np.ndarray | None = None,
                        funding_diff: np.ndarray | None = None,
                        adf: bool = True) -> tuple[pl.DataFrame, int]:
    """
    df: aligned bars with open1/close1/volume1/taker_buy_volume1 and the same for leg 2.
    btc_close: BTC closes aligned to df rows (optional).
    funding_diff: last known funding rate of leg1 minus leg2 (as-of, optional).
    Returns (features DataFrame incl. timestamp, warmup_bars).
    """
    b = lambda m: _bars(m, bar_minutes)
    c1, c2 = df["close1"].to_numpy(), df["close2"].to_numpy()
    spread = np.log(c1) - spec.beta * np.log(c2) - spec.alpha
    unit = np.concatenate([[np.nan], np.diff(spread)]) / (1.0 + abs(spec.beta))
    out = df.select("timestamp").with_columns(
        pl.Series("spread", spread), pl.Series("unit_ret", unit),
        pl.Series("r1", np.concatenate([[np.nan], np.diff(np.log(c1))])),
        pl.Series("r2", np.concatenate([[np.nan], np.diff(np.log(c2))])),
        pl.Series("c1", c1), pl.Series("c2", c2),
        (pl.Series("dv1", (df["volume1"] * df["close1"]).to_numpy())),
        (pl.Series("dv2", (df["volume2"] * df["close2"]).to_numpy())),
        pl.Series("imb1", ((2 * df["taker_buy_volume1"] - df["volume1"]) / (df["volume1"] + 1e-12)).to_numpy()),
        pl.Series("imb2", ((2 * df["taker_buy_volume2"] - df["volume2"]) / (df["volume2"] + 1e-12)).to_numpy()),
    )

    def z(col: str, minutes: float) -> pl.Expr:
        w = b(minutes)
        return (pl.col(col) - pl.col(col).rolling_mean(w)) / pl.col(col).rolling_std(w)

    out = out.with_columns(
        z("spread", 240).alias("z_4h"), z("spread", 720).alias("z_12h"), z("spread", 1440).alias("z_24h"),
        pl.col("unit_ret").rolling_std(b(60)).alias("spread_vol_1h"),
        pl.col("unit_ret").rolling_std(b(1440)).alias("spread_vol_24h"),
        pl.rolling_corr(pl.col("r1"), pl.col("r2"), window_size=b(300)).alias("ret_corr_5h"),
        (pl.col("c1") / pl.col("c1").rolling_mean(b(1440)) - 1).alias("trend1_24h"),
        (pl.col("c2") / pl.col("c2").rolling_mean(b(1440)) - 1).alias("trend2_24h"),
        (pl.col("dv1").rolling_sum(b(60)) / (pl.col("dv2").rolling_sum(b(60)) + 1e-12)).log().alias("dollar_vol_ratio_log"),
        (pl.col("imb1") - pl.col("imb2")).rolling_mean(b(60)).alias("of_imb_spread_1h"),
    )
    out = out.with_columns(
        (pl.col("z_24h") - pl.col("z_24h").shift(1)).alias("z_mom_1"),
        (pl.col("z_24h") - pl.col("z_24h").shift(3)).alias("z_mom_3"),
        (pl.col("spread_vol_1h") / pl.col("spread_vol_24h")).alias("spread_vol_ratio"),
        z("of_imb_spread_1h", 1440).alias("of_imb_spread_norm"),
    )
    if btc_close is not None:
        bc = pl.Series(btc_close)
        out = out.with_columns(
            (bc / bc.rolling_mean(b(1440)) - 1).alias("btc_trend_24h"),
            (bc.log().diff().rolling_std(b(1440))).alias("btc_vol_24h"),
        )
    else:
        out = out.with_columns(pl.lit(np.nan).alias("btc_trend_24h"), pl.lit(np.nan).alias("btc_vol_24h"))
    if adf:
        ts = df["timestamp"].to_numpy()
        adf_vals = rolling_adf_pvalue(spread, b(1440), ts, 60 * 60_000)
        # a value only becomes valid after the first full-hour evaluation following warm-up
        out = out.with_columns(pl.Series("adf_pvalue_24h", adf_vals))
    else:
        out = out.with_columns(pl.lit(np.nan).alias("adf_pvalue_24h"))
    if funding_diff is not None:
        out = out.with_columns(pl.Series("funding_diff", funding_diff))

    # longest dependency chain: 24h rolling of a 1h rolling (of_imb_spread_norm) + 1 diff,
    # plus up to one hour until the first clock-aligned ADF evaluation
    warmup = b(1440) + 2 * b(60) + 1
    out = out.with_columns(pl.col(pl.Float64).fill_nan(None))
    return out, warmup


def forward_unit_return(spread: np.ndarray, beta: float, horizon_bars: int) -> np.ndarray:
    """Return of a long-spread position (per unit gross notional) from close t to close t+h.
    NaN for the last ``horizon_bars`` rows (no future there)."""
    y = np.full(len(spread), np.nan)
    y[:-horizon_bars] = (spread[horizon_bars:] - spread[:-horizon_bars]) / (1.0 + abs(beta))
    return y
