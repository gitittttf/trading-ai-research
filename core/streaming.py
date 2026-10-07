"""
Bar-by-bar ("streaming") version of the spread strategies, used by live paper
trading. It reuses the backtest's exit rule (core.backtest.exit_reason), entry rule
and feature code, and tests/test_live_parity.py checks that feeding historical bars
one at a time produces exactly the decisions of the vectorized backtest.

Protocol: call ``on_bar(bar)`` once per CLOSED bar. It returns a decision taken on
that bar's close ({"action": "enter"|"exit", ...}) or None. The caller executes it
immediately (= next bar's open in backtest terms) and reports entry fill prices via
``set_fill`` (only needed for an unrealized stop-loss).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from core.backtest import ExitRules, _unit_return, exit_reason, leg_weights
from core.features import _adf_p, active_feature_cols, build_pair_features
from core.pairs import PairSpec, rolling_zscore, spread_series

BAR_KEYS = ["timestamp", "open1", "high1", "low1", "close1", "volume1", "taker_buy_volume1",
            "open2", "high2", "low2", "close2", "volume2", "taker_buy_volume2"]


@dataclass
class StreamingConfig:
    strategy: str                       # "baseline" | "xgb"
    bar_minutes: int = 5
    z_window_minutes: int = 1440
    entry_z: float = 2.0
    rules: ExitRules = field(default_factory=ExitRules)
    history_bars: int = 2000
    # xgb only
    horizon_minutes: int = 60
    need_edge: float = 0.0              # cost_mult * round_trip_rate chosen in validation
    penalty: float = 1.0
    use_btc: bool = True
    adf: bool = True


class StreamingPair:
    def __init__(self, spec: PairSpec, cfg: StreamingConfig, models: list | None = None):
        self.spec = spec
        self.cfg = cfg
        self.models = models or []
        self.rows: list[dict] = []
        self.last_btc: float | None = None
        self._adf_key: int | None = None
        self._adf_val = np.nan
        self.w1, self.w2 = leg_weights(spec.beta)
        self.side = 0
        self.entry_z = 0.0
        self.held = 0
        self.fill: tuple[float, float] | None = None
        self.last_ts: int | None = None
        self.last_z = np.nan
        self.last_info: dict = {}

    # ------------------------------------------------------------------
    def restore_position(self, side: int, entry_z: float, bars_held: int, fill: tuple[float, float] | None):
        self.side, self.entry_z, self.held, self.fill = side, entry_z, bars_held, fill

    def set_fill(self, e1: float, e2: float):
        self.fill = (e1, e2)

    def force_flat(self):
        self.side, self.held, self.fill = 0, 0, None

    # ------------------------------------------------------------------
    def _z_and_info(self) -> tuple[float, float, dict]:
        """z at the last two bars and strategy info at the last bar."""
        c1 = np.array([r["close1"] for r in self.rows])
        c2 = np.array([r["close2"] for r in self.rows])
        if self.cfg.strategy == "baseline":
            z = rolling_zscore(spread_series(c1, c2, self.spec),
                               max(2, int(round(self.cfg.z_window_minutes / self.cfg.bar_minutes))))
            return z[-1], (z[-2] if len(z) > 1 else np.nan), {}
        df = pl.DataFrame({k: [r[k] for r in self.rows] for k in BAR_KEYS})
        btc = np.array([r["btc_close"] for r in self.rows]) if self.cfg.use_btc else None
        # ADF is cached per clock hour (identical values to core.features.rolling_adf_pvalue)
        feats, warmup = build_pair_features(df, self.spec, self.cfg.bar_minutes, btc_close=btc, adf=False)
        z = feats["z_24h"].fill_null(np.nan).to_numpy()
        info = {"pred_ret": np.nan, "unc": np.nan}
        if len(self.rows) > warmup and self.models:
            cols = active_feature_cols(with_btc=btc is not None, with_adf=self.cfg.adf)
            x = feats.select(cols).tail(1).to_numpy().astype(float)
            if self.cfg.adf:
                x[0, cols.index("adf_pvalue_24h")] = self._adf_last(feats["spread"].to_numpy())
            if np.all(np.isfinite(x)):
                preds = np.array([m.predict(x)[0] for m in self.models])
                h = max(1, self.cfg.horizon_minutes // self.cfg.bar_minutes)
                scale = float(feats["spread_vol_24h"].fill_null(np.nan).to_numpy()[-1]) * np.sqrt(h)
                info = {"pred_ret": float(preds.mean() * scale), "unc": float(preds.std() * scale)}
        return z[-1], (z[-2] if len(z) > 1 else np.nan), info

    def _adf_last(self, spread: np.ndarray) -> float:
        """ADF p-value as of the newest bar: evaluated on the first bar of the latest clock
        hour (same rule as core.features.rolling_adf_pvalue), cached per hour."""
        from statsmodels.tsa.stattools import adfuller
        window = max(1, int(round(1440 / self.cfg.bar_minutes)))
        ts = np.array([r["timestamp"] for r in self.rows], dtype=np.int64)
        bucket = ts // 3_600_000
        j = len(ts) - 1
        while j > 0 and bucket[j] == bucket[j - 1]:
            j -= 1
        if j < window - 1:
            return np.nan
        if self._adf_key == int(ts[j]):
            return self._adf_val
        seg = spread[j - window + 1:j + 1]
        val = np.nan
        if np.all(np.isfinite(seg)) and np.std(seg) > 0:
            try:
                val = _adf_p(adfuller, seg)
            except Exception:
                val = np.nan
        self._adf_key, self._adf_val = int(ts[j]), val
        return val

    def on_bar(self, bar: dict, btc_close: float | None = None, warmup_only: bool = False) -> dict | None:
        ts = int(bar["timestamp"])
        if self.last_ts is not None and ts <= self.last_ts:
            return None                      # duplicate or out of order: processed exactly once
        self.last_ts = ts
        row = {k: float(bar[k]) if k != "timestamp" else ts for k in BAR_KEYS}
        if btc_close is not None and np.isfinite(btc_close):
            self.last_btc = float(btc_close)
        # BTC is forward filled exactly like the backtest's as-of join
        row["btc_close"] = self.last_btc if self.last_btc is not None else np.nan
        self.rows.append(row)
        if len(self.rows) > self.cfg.history_bars:
            self.rows = self.rows[-self.cfg.history_bars:]
        if warmup_only:
            return None                      # history only; z/features are computed from the buffer later
        zt, zprev, info = self._z_and_info()
        self.last_z, self.last_info = zt, info

        if self.side == 0:
            if not (np.isfinite(zt) and np.isfinite(zprev)):
                return None
            e = self.cfg.entry_z
            side = 1 if (zt < -e and zprev >= -e) else (-1 if (zt > e and zprev <= e) else 0)
            if side == 0:
                return None
            if self.cfg.strategy == "xgb":
                if not np.isfinite(info.get("pred_ret", np.nan)):
                    return None
                edge = side * info["pred_ret"] - self.cfg.penalty * info["unc"]
                if not edge > self.cfg.need_edge:
                    return None
                info = dict(info, edge=edge)
            self.side, self.entry_z, self.held, self.fill = side, float(zt), 0, None
            return {"action": "enter", "side": side, "z": float(zt), "ts": ts, "info": info}

        self.held += 1
        unit = 0.0
        if self.cfg.rules.stop_loss is not None and self.fill is not None:
            unit = _unit_return(self.side, self.w1, self.w2, self.fill[0], self.fill[1],
                                self.rows[-1]["close1"], self.rows[-1]["close2"])
        reason = exit_reason(self.side, self.entry_z, zt, self.cfg.rules, unit, self.held)
        if reason is None:
            return None
        side, held = self.side, self.held
        self.force_flat()
        return {"action": "exit", "side": side, "reason": reason, "z": float(zt), "ts": ts, "held": held}
