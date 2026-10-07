"""
Export the live paper-trading bundle: pairs, parameters and (for xgb) models,
fitted exactly like one walk-forward window that ends NOW.

  uv run python scripts/export_live_bundle.py --strategy baseline
  uv run python scripts/export_live_bundle.py --strategy xgb --seed 0

Writes models/live_bundle/ with manifest.json (git commit, data end, validity,
SHA-256 of every model file). The live core refuses to start without a valid
manifest, so dummy or hand-edited artifacts can no longer be traded.
Re-export every --valid-days (default 30 = the walk-forward test length).
"""
from __future__ import annotations

import os as _os

# Single-threaded BLAS: the research code runs thousands of tiny regressions (coint/ADF);
# multi-threaded OpenBLAS on tiny matrices oversubscribes the CPU and was measured to
# slow a window from ~22s to ~180s when two jobs ran at once.
for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

import argparse
import json
import os
import shutil
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from core.backtest import PortfolioConfig                      # noqa: E402
from core.constants import BTC_SYMBOL, DATA_DIR, UNIVERSE      # noqa: E402
from core.costs import CostModel                               # noqa: E402
from core.data import load_klines                              # noqa: E402
from core.manifest import git_state, sha256_file               # noqa: E402
from core.pairs import select_pairs                            # noqa: E402
from strategies.baseline_pairs import BaselineParams, BaselinePairsStrategy  # noqa: E402
from strategies.common import DAY_MS, Window                   # noqa: E402
from strategies.xgb_spread import XGBParams, XGBSpreadStrategy  # noqa: E402

BUNDLE_DIR = os.path.join(ROOT_DIR, "models", "live_bundle")


def build_bundle(bars: dict, strategy: str, train_days: int, valid_days: int, seed: int,
                 out_dir: str, top_k: int = 10, costs: CostModel | None = None,
                 portfolio: PortfolioConfig | None = None, bar_minutes: int = 5) -> dict:
    costs = costs or CostModel()
    portfolio = portfolio or PortfolioConfig()
    end = max(int(d["timestamp"][-1]) for d in bars.values()) + 60_000
    end = (end // (bar_minutes * 60_000)) * (bar_minutes * 60_000)
    start = end - train_days * DAY_MS
    closes = {s: d.select("timestamp", "close") for s, d in bars.items()
              if int(d["timestamp"][0]) <= start + 7 * DAY_MS}
    specs = select_pairs(closes, start, end, top_k=top_k)
    window = Window(0, start, end, end, end)
    tmp = out_dir + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    model_files = {}
    if strategy == "baseline":
        strat = BaselinePairsStrategy(BaselineParams(bar_minutes=bar_minutes))
        p = strat.params
        cfg = {"strategy": "baseline", "bar_minutes": p.bar_minutes, "z_window_minutes": p.z_window_minutes,
               "entry_z": p.entry_z, "exit_z": p.exit_z, "stop_z": p.stop_z, "stop_mode": "absolute"}
        pairs = [dict(s.to_dict(), max_hold_bars=strat.exit_rules(s).max_hold_bars) for s in specs]
        trade = len(specs) > 0
    else:
        strat = XGBSpreadStrategy(XGBParams(bar_minutes=bar_minutes, seed=seed), costs)
        strat.generate(bars, specs, window, btc_symbol=BTC_SYMBOL if BTC_SYMBOL in bars else None)
        fit = strat.last_fit
        trade = bool(fit.get("traded"))
        p = strat.params
        cfg = {"strategy": "xgb", "bar_minutes": p.bar_minutes, "z_window_minutes": 1440,
               "exit_z": p.exit_z, "stop_z": p.stop_z, "stop_mode": "absolute", "penalty": p.penalty,
               "horizon_minutes": p.horizon_minutes, "use_btc": BTC_SYMBOL in bars, "adf": p.adf_feature,
               "validation": fit}
        if trade:
            cfg["entry_z"] = strat.chosen["entry_z"]
            cfg["need_edge"] = strat.chosen["cost_mult"] * costs.round_trip_rate()
            for i, m in enumerate(strat.models):
                name = f"xgb_{i}.json"
                m.save_model(os.path.join(tmp, name))
                model_files[name] = sha256_file(os.path.join(tmp, name))
        pairs = [dict(s.to_dict(), max_hold_bars=strat._exit_rules(s).max_hold_bars) for s in specs]
    manifest = {
        "bundle_version": 2,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": git_state(),
        "strategy": strategy,
        "trade": trade,
        "data_end_ms": end,
        "valid_until_ms": end + valid_days * DAY_MS,
        "train_days": train_days,
        "streaming_config": cfg,
        "pairs": pairs,
        "model_files": model_files,
        "costs": costs.to_dict(),
        "portfolio": portfolio.__dict__,
    }
    with open(os.path.join(tmp, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.replace(tmp, out_dir)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", choices=["baseline", "xgb"], default="baseline")
    ap.add_argument("--train-days", type=int, default=90)
    ap.add_argument("--valid-days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out", default=BUNDLE_DIR)
    args = ap.parse_args()
    bars = {s: d for s in UNIVERSE if (d := load_klines(s, args.data_dir)) is not None}
    if not bars:
        raise SystemExit("no data; run scripts/fetch_binance_vision.py first")
    m = build_bundle(bars, args.strategy, args.train_days, args.valid_days, args.seed, args.out, args.top_k)
    print(json.dumps({k: m[k] for k in ["strategy", "trade", "created_utc", "valid_until_ms"]}, indent=1))
    for p in m["pairs"]:
        print(f"  {p['asset1']:<14} {p['asset2']:<14} beta={p['beta']:.3f} p={p['coint_pvalue']:.4f} "
              f"half-life={p['half_life_minutes'] / 60:.1f}h")


if __name__ == "__main__":
    main()
