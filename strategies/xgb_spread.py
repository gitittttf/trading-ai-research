"""
ML spread strategy: XGBoost predictions of the spread return on the look-ahead-free engine.

Design rules:
* pairs come from point-in-time selection, not from a list chosen with future
  data; no "OOS coint gate" that peeks into the test window;
* target = forward return of a long-spread position per unit gross notional over
  ``horizon`` (beta-hedged log spread, no BTC subtraction); the model
  predicts it in RETURN units, so the cost filter compares like with like;
* standard squared-error loss (an asymmetric loss would shrink predictions
  toward zero and make the confidence threshold meaningless);
* thresholds are chosen on a purged VALIDATION slice at the end of the training
  window using realized net returns after costs, never on in-sample model output;
  if no configuration makes money in validation, the window is not traded;
* fills on the next bar, path-dependent exits, costs from core.costs;
* everything is deterministic for a given seed (n_jobs fixed, no parallel
  optimizer);
* the optional meta-filter is trained on ALL validation candidates labelled with
  their own realized outcome; rejected trades are never relabelled as losers.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core.backtest import Execution, ExitRules, PairArrays, simulate_pair
from core.costs import CostModel
from core.data import MINUTE_MS, align_pair, resample_klines, time_slice
from core.features import active_feature_cols, build_pair_features, forward_unit_return
from core.pairs import PairSpec, select_pairs
from strategies.common import Window, pair_arrays

META_FEATURES = ["abs_z", "edge", "unc", "spread_vol_ratio", "adf_pvalue_24h", "half_life_hours"]


@dataclass(frozen=True)
class XGBParams:
    bar_minutes: int = 5
    horizon_minutes: int = 60
    entry_z_grid: tuple = (1.5, 2.0, 2.5)
    cost_mult_grid: tuple = (0.5, 1.0, 2.0)
    exit_z: float = 0.5
    stop_z: float = 4.0
    time_stop_half_lives: float = 3.0
    max_hold_minutes: float = 3 * 1440
    penalty: float = 1.0            # edge = side*pred - penalty*ensemble_std
    n_models: int = 3
    val_fraction: float = 0.3
    min_val_trades: int = 10
    sit_out_if_negative: bool = True   # do not trade a window whose best validation score is <= 0
    validation_selection: bool = True  # select fit/validation pairs on the fit part only
    select_top_k: int = 10
    select_max_pvalue: float = 0.05
    use_meta: bool = False
    meta_threshold: float = 0.5
    latency_bars: int = 0
    seed: int = 0
    adf_feature: bool = True
    exec_mode: str = "taker"           # "maker": passive fill model in the TEST window (variant 11)
    maker_timeout_bars: int = 3
    maker_through_bp: float = 1.0
    xgb_params: dict = field(default_factory=lambda: dict(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=50, reg_lambda=5.0, tree_method="hist", n_jobs=4))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PairData:
    spec: PairSpec
    arrays: PairArrays
    feats: pl.DataFrame
    X: np.ndarray
    y_norm: np.ndarray       # target / scale (for training)
    scale: np.ndarray        # per-bar return scale (causal)
    z: np.ndarray
    warmup: int


class XGBSpreadStrategy:
    name = "xgb"

    def __init__(self, params: XGBParams | None = None, costs: CostModel | None = None):
        self.params = params or XGBParams()
        self.costs = costs or CostModel()
        self.last_fit: dict = {}

    def n_trials(self) -> int:
        return len(self.params.entry_z_grid) * len(self.params.cost_mult_grid)

    # ------------------------------------------------------------------ data
    def _pair_data(self, bars_1m: dict[str, pl.DataFrame], spec: PairSpec, start_ms: int,
                   end_ms: int, btc_symbol: str | None) -> PairData | None:
        p = self.params
        warm_min = 2 * 1440 + 120
        lo_ms = start_ms - warm_min * MINUTE_MS
        b1 = resample_klines(time_slice(bars_1m[spec.asset1], lo_ms, end_ms), p.bar_minutes)
        b2 = resample_klines(time_slice(bars_1m[spec.asset2], lo_ms, end_ms), p.bar_minutes)
        df = align_pair(b1, b2)
        if df.height < 500:
            return None
        btc = None
        if btc_symbol and btc_symbol in bars_1m:
            bb = resample_klines(time_slice(bars_1m[btc_symbol], lo_ms, end_ms), p.bar_minutes)
            btc = (df.select("timestamp").join(bb.select("timestamp", "close"), on="timestamp", how="left")
                     .with_columns(pl.col("close").forward_fill())["close"].to_numpy())
        feats, warmup = build_pair_features(df, spec, p.bar_minutes, btc_close=btc, adf=p.adf_feature)
        h = max(1, p.horizon_minutes // p.bar_minutes)
        spread = feats["spread"].to_numpy()
        y = forward_unit_return(spread, spec.beta, h)
        scale = feats["spread_vol_24h"].to_numpy() * np.sqrt(h)
        with np.errstate(invalid="ignore", divide="ignore"):
            y_norm = y / scale
        cols = active_feature_cols(with_btc=btc is not None, with_adf=p.adf_feature)
        X = feats.select(cols).to_numpy().astype(float)
        arrays = pair_arrays(df)
        return PairData(spec, arrays, feats, X, y_norm, scale, feats["z_24h"].to_numpy(), warmup)

    # ---------------------------------------------------------------- model
    def _fit_models(self, X: np.ndarray, y: np.ndarray, seed: int | None = None) -> list:
        seed = self.params.seed if seed is None else seed
        models = []
        for i in range(self.params.n_models):
            m = xgb.XGBRegressor(objective="reg:squarederror", random_state=seed * 100 + i,
                                 **self.params.xgb_params)
            m.fit(X, y)
            models.append(m)
        return models

    @staticmethod
    def _predict(models: list, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        preds = np.array([m.predict(X) for m in models])
        return preds.mean(axis=0), preds.std(axis=0)

    # -------------------------------------------------------------- signals
    def _entries(self, pd_: PairData, pred_ret: np.ndarray, unc_ret: np.ndarray,
                 entry_z: float, cost_mult: float) -> tuple[np.ndarray, np.ndarray]:
        z = pd_.z
        prev = np.concatenate([[np.nan], z[:-1]])
        ok = np.isfinite(z) & np.isfinite(prev) & np.isfinite(pred_ret)
        side = np.zeros(len(z))
        side[ok & (z < -entry_z) & (prev >= -entry_z)] = 1
        side[ok & (z > entry_z) & (prev <= entry_z)] = -1
        edge = side * pred_ret - self.params.penalty * unc_ret
        need = cost_mult * self.costs.round_trip_rate()
        sig = np.where((side != 0) & (edge > need), side, 0.0)
        return sig, edge

    def _exit_rules(self, spec: PairSpec) -> ExitRules:
        p = self.params
        hold_min = min(p.time_stop_half_lives * spec.half_life_minutes, p.max_hold_minutes)
        return ExitRules(exit_z=p.exit_z, stop_z=p.stop_z, max_hold_bars=max(1, int(hold_min // p.bar_minutes)))

    def execution(self) -> Execution:
        p = self.params
        return Execution(p.exec_mode, p.maker_timeout_bars, p.maker_through_bp)

    def _candidates(self, pd_: PairData, pred_ret, unc_ret, entry_z, cost_mult, lo, hi,
                    execution: Execution | None = None) -> list[dict]:
        sig, edge = self._entries(pd_, pred_ret, unc_ret, entry_z, cost_mult)
        info = {"pred_ret": pred_ret, "unc": unc_ret, "edge": edge, "abs_z": np.abs(pd_.z),
                "spread_vol_ratio": pd_.feats["spread_vol_ratio"].fill_null(np.nan).to_numpy(),
                "adf_pvalue_24h": pd_.feats["adf_pvalue_24h"].fill_null(np.nan).to_numpy(),
                "half_life_hours": np.full(len(sig), pd_.spec.half_life_minutes / 60.0)}
        return simulate_pair(pd_.arrays, pd_.z, sig, self._exit_rules(pd_.spec), pd_.spec.beta,
                             pair=pd_.spec.name, latency_bars=self.params.latency_bars,
                             trade_window=(lo, hi), info=info, execution=execution)

    # -------------------------------------------------------------- window
    def _validation_specs(self, bars_1m: dict[str, pl.DataFrame], window: Window, fit_end: int,
                          specs: list[PairSpec]) -> list[PairSpec]:
        """Pairs for model fitting + threshold validation, selected on the FIT part only.

        Selecting them on the whole training window would make validation in-sample:
        a pair picked because it mean-reverted during the validation slice scores well
        there by construction (review finding)."""
        p = self.params
        if not p.validation_selection:
            return specs
        closes = {s: d.select("timestamp", "close") for s, d in bars_1m.items()
                  if int(d["timestamp"][0]) <= window.train_start + 7 * 86_400_000
                  and int(d["timestamp"][-1]) >= fit_end}
        return select_pairs(closes, window.train_start, fit_end, top_k=p.select_top_k,
                            max_pvalue=p.select_max_pvalue)

    def _load(self, bars_1m, specs, start, end, btc_symbol) -> list[PairData]:
        out = []
        for spec in specs:
            d = self._pair_data(bars_1m, spec, start, end, btc_symbol)
            if d is not None:
                out.append(d)
        return out

    def _predictions(self, models, data: list[PairData]) -> list[tuple[np.ndarray, np.ndarray]]:
        preds = []
        for d in data:
            mu, sd = self._predict(models, np.nan_to_num(d.X, nan=0.0))
            bad = ~np.all(np.isfinite(d.X), axis=1) | (np.arange(len(mu)) < d.warmup)
            pr, un = mu * d.scale, sd * d.scale
            pr[bad] = np.nan
            un[bad] = np.nan
            preds.append((pr, un))
        return preds

    def window_data(self, bars_1m: dict[str, pl.DataFrame], specs: list[PairSpec], window: Window,
                    btc_symbol: str | None) -> dict | None:
        """Everything of a window that does NOT depend on the model seed (computed once)."""
        p = self.params
        h_ms = p.horizon_minutes * MINUTE_MS
        train_len = window.train_end - window.train_start
        fit_end = window.train_start + int((1 - p.val_fraction) * train_len)
        val_specs = self._validation_specs(bars_1m, window, fit_end, specs)
        val_data = self._load(bars_1m, val_specs, window.train_start, window.train_end, btc_symbol)
        if not val_data:
            return None
        Xs, ys = [], []
        for d in val_data:
            ts = d.arrays.ts
            m = (ts >= window.train_start) & (ts < fit_end - h_ms)
            m &= np.arange(len(ts)) >= d.warmup
            m &= np.all(np.isfinite(d.X), axis=1) & np.isfinite(d.y_norm)
            Xs.append(d.X[m])
            ys.append(np.clip(d.y_norm[m], -10, 10))
        X_fit, y_fit = np.vstack(Xs), np.concatenate(ys)
        if len(y_fit) < 2000:
            return None
        # test features only need their own warm-up (all windows are <= 2 days), so they are
        # computed from the test start; values are identical to a longer history
        test_data = (self._load(bars_1m, specs, window.test_start, window.test_end, btc_symbol)
                     if window.test_end > window.test_start else [])
        return {"fit_end": fit_end, "val_specs": val_specs, "val_data": val_data,
                "test_data": test_data, "X_fit": X_fit, "y_fit": y_fit}

    def run_seed(self, wd: dict, window: Window, seed: int, meta_modes=(False,)) -> dict:
        """Fit the models with ``seed``, choose thresholds on validation, trade the test window.
        Returns {use_meta: candidates} for each requested meta mode (same models, same thresholds)."""
        p = self.params
        fit_end = wd["fit_end"]
        models = self._fit_models(wd["X_fit"], wd["y_fit"], seed)
        val_preds = self._predictions(models, wd["val_data"])
        rt = self.costs.round_trip_rate()
        best = None
        grid_scores = []
        val_cands_by_cfg = {}
        for ez in p.entry_z_grid:
            for cm in p.cost_mult_grid:
                cands = []
                for d, (pr, un) in zip(wd["val_data"], val_preds):
                    lo = int(np.searchsorted(d.arrays.ts, fit_end))
                    hi = int(np.searchsorted(d.arrays.ts, window.train_end))
                    cands += self._candidates(d, pr, un, ez, cm, lo, hi)
                net = np.array([c["gross_ret"] - rt for c in cands])
                score = float(net.sum()) if len(net) >= p.min_val_trades else -np.inf
                grid_scores.append({"entry_z": ez, "cost_mult": cm, "n": len(net), "score": score})
                val_cands_by_cfg[(ez, cm)] = cands
                if best is None or score > best[0]:
                    best = (score, ez, cm)
        self.last_fit = {"window": window.index, "seed": seed, "grid": grid_scores, "chosen": best,
                         "validation_pairs": [s.name for s in wd["val_specs"]],
                         "n_fit_rows": int(len(wd["y_fit"])), "traded": bool(best and best[0] > 0)}
        empty = {m: [] for m in meta_modes}
        if best is None or not np.isfinite(best[0]):
            return empty
        if p.sit_out_if_negative and best[0] <= 0:
            return empty       # nothing worked in validation -> sit this window out
        _, ez, cm = best
        self.models = models
        self.chosen = {"entry_z": ez, "cost_mult": cm}
        meta = self._fit_meta(val_cands_by_cfg[(ez, cm)], rt) if any(meta_modes) else None
        self.meta = meta
        self.last_fit["meta_active"] = meta is not None

        base = []
        for d, (pr, un) in zip(wd["test_data"], self._predictions(models, wd["test_data"])):
            lo = int(np.searchsorted(d.arrays.ts, window.test_start))
            hi = int(np.searchsorted(d.arrays.ts, window.test_end))
            for c in self._candidates(d, pr, un, ez, cm, lo, hi, execution=self.execution()):
                c["window"] = window.index
                c["entry_z_param"] = ez
                c["cost_mult_param"] = cm
                base.append(c)
        out = {}
        for use_meta in meta_modes:
            if not use_meta:
                out[use_meta] = base
                continue
            kept = []
            for c in base:
                if meta is None:
                    kept.append(c)          # too few validation trades for a meta model: no filter
                    continue
                row = [c[k] for k in META_FEATURES]
                if not np.all(np.isfinite(row)):
                    continue                # unknown inputs -> do not trade
                c2 = dict(c, meta_prob=float(meta.predict_proba([row])[0, 1]))
                if c2["meta_prob"] >= p.meta_threshold:
                    kept.append(c2)
            out[use_meta] = kept
        return out

    def generate(self, bars_1m: dict[str, pl.DataFrame], specs: list[PairSpec], window: Window,
                 btc_symbol: str | None = "BTC/USDT") -> list[dict]:
        """
        fit part   [train_start, fit_end - horizon) : train the models (purged)
        validation [fit_end, train_end)             : choose thresholds on realized net returns
        test       [test_start, test_end)           : trade ``specs`` (selected on the full
                                                      training window by the runner)
        Pairs for fit + validation are selected on the fit part only.
        """
        self.last_fit = {"window": window.index, "traded": False}
        wd = self.window_data(bars_1m, specs, window, btc_symbol)
        if wd is None:
            return []
        return self.run_seed(wd, window, self.params.seed, (self.params.use_meta,))[self.params.use_meta]

    def _fit_meta(self, cands: list[dict], rt: float):
        if len(cands) < 30:
            return None
        X = np.array([[c[k] for k in META_FEATURES] for c in cands], dtype=float)
        y = np.array([1 if c["gross_ret"] - rt > 0 else 0 for c in cands])
        if len(set(y)) < 2 or not np.all(np.isfinite(X)):
            return None
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
        m.fit(X, y)
        return m
