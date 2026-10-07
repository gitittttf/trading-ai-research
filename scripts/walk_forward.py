"""
Walk-forward research runner (honest version).

Examples
--------
  # end-to-end smoke test on synthetic data (no downloads needed)
  uv run python scripts/walk_forward.py --strategy baseline --synthetic

  # real data (after scripts/fetch_binance_vision.py), holdout untouched
  uv run python scripts/walk_forward.py --strategy baseline --bars 5
  uv run python scripts/walk_forward.py --strategy xgb --bars 5 --seeds 0 1 2 3 4

Every run writes results/<run_id>/ with manifest.json, trades.csv, windows.json,
summary.json and equity.png. Numbers in reports must come from these folders.
The last ``--holdout-days`` are excluded unless --use-holdout is passed; the
evaluation protocol allows touching the holdout exactly once per strategy.
"""
from __future__ import annotations

import os as _os

# Single-threaded BLAS: the research code runs thousands of tiny regressions (coint/ADF);
# multi-threaded OpenBLAS on tiny matrices oversubscribes the CPU and was measured to
# slow a window from ~22s to ~180s when two jobs ran at once.
for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from core.backtest import (Execution, PortfolioConfig, _unit_return, maker_entry, maker_exit,   # noqa: E402
                           run_portfolio)
from core.constants import BTC_SYMBOL, DATA_DIR, UNIVERSE                 # noqa: E402
from core.costs import CostModel                                           # noqa: E402
from core.data import funding_path, klines_path, load_funding, load_klines, validate_klines  # noqa: E402
from core.manifest import new_run_dir, write_manifest                     # noqa: E402
from core.metrics import summarize                                         # noqa: E402
from core.pairs import select_pairs                                        # noqa: E402
from strategies.baseline_pairs import BaselineParams, BaselinePairsStrategy  # noqa: E402
from strategies.common import DAY_MS, make_windows, prepare_pair           # noqa: E402
from strategies.xgb_spread import XGBParams, XGBSpreadStrategy             # noqa: E402


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_date(s: str) -> int:
    return int(dt.datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=dt.UTC).timestamp() * 1000)


def load_universe(args) -> tuple[dict, dict, list[str]]:
    if args.synthetic:
        from core.synthetic import cointegrated_market, random_walk_market
        bars = cointegrated_market(n_pairs=3, n_minutes=args.synthetic_days * 1440, seed=args.synthetic_seed,
                                   half_life_minutes=240)
        rw = random_walk_market(n_assets=4, n_minutes=args.synthetic_days * 1440, seed=args.synthetic_seed + 1)
        bars.update(rw)
        bars[BTC_SYMBOL] = rw.pop("A0/USDT")
        bars.pop("A0/USDT", None)
        return bars, {}, []
    bars, funding, files = {}, {}, []
    symbols = args.symbols or UNIVERSE
    for s in symbols:
        df = load_klines(s, args.data_dir)
        if df is None:
            log(f"  missing klines for {s} - skipped")
            continue
        rep = validate_klines(df)
        if not rep.ok:
            raise SystemExit(f"data validation failed for {s}: {rep}")
        bars[s] = df
        files.append(klines_path(s, args.data_dir))
        f = load_funding(s, args.data_dir)
        if f is not None:
            funding[s] = (f["timestamp"].to_numpy(), f["funding_rate"].to_numpy())
            files.append(funding_path(s, args.data_dir))
    if not bars:
        raise SystemExit(f"no data in {args.data_dir}; run scripts/fetch_binance_vision.py first")
    return bars, funding, files


def build_strategy(args, seed: int):
    costs = CostModel(use_maker=args.maker, multiplier=args.cost_mult)
    ex = dict(exec_mode=args.exec, maker_timeout_bars=args.maker_timeout_bars, maker_through_bp=args.maker_through_bp)
    if args.strategy == "baseline":
        return BaselinePairsStrategy(BaselineParams(bar_minutes=args.bars, latency_bars=args.latency,
                                                    entry_z=args.entry_z, z_window_minutes=args.z_window,
                                                    max_hold_minutes=args.max_hold_days * 1440, **ex)), costs
    return XGBSpreadStrategy(XGBParams(bar_minutes=args.bars, latency_bars=args.latency, seed=seed,
                                       use_meta=args.meta, adf_feature=not args.no_adf, **ex), costs), costs


def _random_maker_trade(t: dict, arr, lo: int, hi: int, rng, ex: Execution) -> dict | None:
    """One random trade through the SAME passive fill model as the strategy: random decision
    bar and side, the real trade's holding time (bars from entry fill to exit decision) and
    the real trade's exit type (take-profit exits rest in the book, all others are taker)."""
    last = hi - 1
    bar_ms = int(arr.ts[1] - arr.ts[0]) if len(arr.ts) > 1 else 60_000
    hold = max(1, int(t["exit_decision_idx"]) - int(t["entry_idx"]))
    maker_exit_type = str(t["exit_reason"]).startswith("take_profit")
    for _ in range(20):                       # an unfilled entry is redrawn (the strategy's count is kept)
        top = last - hold - 2 * ex.timeout_bars - 2
        if top <= lo + 1:
            return None
        d = int(rng.integers(lo, top))
        side = int(rng.choice([-1, 1]))
        ent = maker_entry(arr, side, d, ex, last)
        if ent is None:
            continue
        e_idx, e1, e2, em1, em2 = ent
        xd = e_idx + hold
        if maker_exit_type:
            x_idx, x1, x2, xm1, xm2, exit_ts, _ = maker_exit(arr, side, xd, ex, last, bar_ms)
        else:
            x_idx, x1, x2, xm1, xm2 = xd + 1, float(arr.open1[xd + 1]), float(arr.open2[xd + 1]), False, False
            exit_ts = int(arr.ts[x_idx])
        r = dict(t)
        r.update(side=side, signal_idx=d, entry_idx=e_idx, exit_decision_idx=xd, exit_idx=x_idx,
                 entry_ts=int(arr.ts[e_idx]), exit_ts=exit_ts, entry_p1=e1, entry_p2=e2, exit_p1=x1, exit_p2=x2,
                 entry_maker1=em1, entry_maker2=em2, exit_maker1=xm1, exit_maker2=xm2,
                 entry_vol1_usd=float(arr.vol1_usd[e_idx]), entry_vol2_usd=float(arr.vol2_usd[e_idx]),
                 exit_vol1_usd=float(arr.vol1_usd[x_idx]), exit_vol2_usd=float(arr.vol2_usd[x_idx]),
                 gross_ret=_unit_return(side, t["w1"], t["w2"], e1, e2, x1, x2))
        return r
    return None


def random_entry_benchmark(trades: pd.DataFrame, arrays_by_key: dict, costs: CostModel,
                           cfg: PortfolioConfig, pair_assets: dict, funding: dict,
                           n_sims: int, seed: int, execution: Execution | None = None) -> dict:
    """Same pairs, windows, trade count and holding times, but random entry bars and
    random sides. Identical costs, sizing, limits and execution model as the strategy."""
    rng = np.random.default_rng(seed)
    real = float(trades["net_pnl"].sum()) if len(trades) else 0.0
    sims = []
    base = trades.to_dict("records")
    maker = execution is not None and execution.maker
    for _ in range(n_sims):
        cands = []
        for t in base:
            key = (t["window"], t["pair"])
            if key not in arrays_by_key:
                continue
            arr, lo, hi = arrays_by_key[key]
            if maker:
                r = _random_maker_trade(t, arr, lo, hi, rng, execution)
                if r is not None:
                    cands.append(r)
                continue
            bar_ms = int(arr.ts[1] - arr.ts[0]) if len(arr.ts) > 1 else 60_000
            hold = max(1, int(round((t["exit_ts"] - t["entry_ts"]) / bar_ms)))   # same holding TIME
            if hi - 1 - hold <= lo + 1:
                continue
            e = int(rng.integers(lo + 1, hi - 1 - hold))
            x = e + hold
            side = int(rng.choice([-1, 1]))
            r = dict(t)
            r.update(side=side, entry_idx=e, exit_idx=x, entry_ts=int(arr.ts[e]), exit_ts=int(arr.ts[x]),
                     entry_p1=float(arr.open1[e]), entry_p2=float(arr.open2[e]),
                     exit_p1=float(arr.open1[x]), exit_p2=float(arr.open2[x]),
                     gross_ret=_unit_return(side, t["w1"], t["w2"], float(arr.open1[e]), float(arr.open2[e]),
                                            float(arr.open1[x]), float(arr.open2[x])))
            cands.append(r)
        res = run_portfolio(cands, costs, cfg, pair_assets, funding)
        sims.append(float(res.trades["net_pnl"].sum()) if len(res.trades) else 0.0)
    sims = np.array(sims)
    return {"n_sims": n_sims, "real_net_pnl": real, "random_mean": float(sims.mean()),
            "random_std": float(sims.std()), "p_value": float((np.sum(sims >= real) + 1) / (len(sims) + 1))}


def cached_select(args, eligible: dict, w) -> list:
    """Pair selection is deterministic given data + window + params; cache it on disk so
    the protocol's variants (which all share the selection) do not recompute it."""
    from core.pairs import PairSpec
    key_src = json.dumps({"data": args.data_digest, "syms": sorted(eligible), "w": [w.train_start, w.train_end],
                          "k": args.top_k, "p": args.max_pvalue, "v": 1}, sort_keys=True)
    key = hashlib.sha256(key_src.encode()).hexdigest()[:24]
    cache_dir = os.path.join(ROOT_DIR, "results", "_cache_pairs")
    path = os.path.join(cache_dir, f"{key}.json")
    if args.data_digest and os.path.exists(path):
        with open(path) as f:
            return [PairSpec(**d) for d in json.load(f)]
    specs = select_pairs(eligible, w.train_start, w.train_end, top_k=args.top_k, max_pvalue=args.max_pvalue)
    if args.data_digest:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump([s.to_dict() for s in specs], f)
    return specs


def evaluation_windows(args, bars, funding) -> list:
    first = max(min(int(d["timestamp"][0]) for d in bars.values()), parse_date(args.start)) if args.start \
        else min(int(d["timestamp"][0]) for d in bars.values())
    first = (first // DAY_MS + 1) * DAY_MS            # align to a full UTC day
    last = max(int(d["timestamp"][-1]) for d in bars.values())
    if funding:
        # funding comes from monthly archives only; do not evaluate days where it is unknown.
        # Symbols whose trading stopped (delisted) together with their funding are ignored here.
        ends = [int(ts[-1]) for s, (ts, _) in funding.items()
                if len(ts) and s in bars and int(bars[s]["timestamp"][-1]) > int(ts[-1]) + DAY_MS]
        fund_end = (min(ends) if ends else last) + 8 * 3_600_000
        if fund_end < last:
            log(f"capping evaluation at funding coverage end "
                f"{dt.datetime.fromtimestamp(fund_end / 1000, dt.UTC):%Y-%m-%d %H:%M}")
            last = fund_end
    last = (last // DAY_MS) * DAY_MS
    if args.use_holdout:
        ho_start = last - args.holdout_days * DAY_MS
        from strategies.common import Window
        log("!!! HOLDOUT EVALUATION - this must be done once per strategy and then recorded")
        return [Window(0, ho_start - args.train_days * DAY_MS, ho_start, ho_start, last)]
    return make_windows(first, last, args.train_days, args.test_days, holdout_days=args.holdout_days)


def finalize(args, label: str, seed: int, cands: list, win_info: list, windows: list, costs, cfg,
             pair_assets: dict, funding: dict, arrays_by_key: dict, n_trials: int, run_dir: str) -> dict:
    res = run_portfolio(cands, costs, cfg, pair_assets, funding)
    start_ms, end_ms = windows[0].test_start, windows[-1].test_end
    daily = res.daily_returns(start_ms, end_ms)
    summary = summarize(res.trades, daily, cfg.start_equity, n_trials=n_trials)
    summary["rejected_by_portfolio"] = res.rejected
    summary["n_candidates"] = len(cands)
    summary["windows_traded"] = int(sum(1 for w in win_info if w.get("n_candidates", 0) > 0))
    summary["test_period"] = [dt.datetime.fromtimestamp(start_ms / 1000, dt.UTC).isoformat(),
                              dt.datetime.fromtimestamp(end_ms / 1000, dt.UTC).isoformat()]
    if args.exec == "maker" and len(res.trades):
        tr = res.trades
        legs = tr[["entry_maker1", "entry_maker2", "exit_maker1", "exit_maker2"]].to_numpy(dtype=bool)
        summary["maker_fills"] = {"share_of_fills": float(legs.mean()),
                                  "entries_both_legs_maker": float(legs[:, :2].all(axis=1).mean()),
                                  "exits_both_legs_maker": float(legs[:, 2:].all(axis=1).mean())}
    if args.benchmark and len(res.trades):
        execution = Execution(args.exec, args.maker_timeout_bars, args.maker_through_bp)
        summary["random_entry_benchmark"] = random_entry_benchmark(
            res.trades, arrays_by_key, costs, cfg, pair_assets, funding, args.benchmark, seed, execution)
    sub = os.path.join(run_dir, label)
    os.makedirs(sub, exist_ok=True)
    res.trades.to_csv(os.path.join(sub, "trades.csv"), index=False)
    daily.to_csv(os.path.join(sub, "daily_returns.csv"), header=["ret"])
    with open(os.path.join(sub, "windows.json"), "w") as f:
        json.dump(win_info, f, indent=1, default=str)
    with open(os.path.join(sub, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if len(res.equity):
            plt.figure(figsize=(11, 5))
            plt.plot(res.equity.index, res.equity.values)
            plt.axhline(cfg.start_equity, color="grey", ls="--", lw=0.8)
            plt.title(f"{args.strategy} {label}: realized equity (net of all costs)")
            plt.tight_layout()
            plt.savefig(os.path.join(sub, "equity.png"), dpi=120)
            plt.close()
    except Exception as e:  # plotting must never break a run
        log(f"plot failed: {e}")
    return summary


def run_all(args, bars, funding, run_dir: str) -> dict:
    """One pass over the windows; for xgb every requested seed and meta mode is evaluated on
    the same seed-independent window data (pairs, features), so variants stay comparable."""
    costs = CostModel(use_maker=args.maker, multiplier=args.cost_mult)
    cfg = PortfolioConfig(start_equity=args.equity, alloc_per_trade=args.alloc,
                          max_positions=args.max_positions, max_gross_leverage=args.max_leverage)
    windows = evaluation_windows(args, bars, funding)
    if args.strategy == "baseline":
        strat, _ = build_strategy(args, 0)
        labels = {"seed0": (0, False)}
    else:
        strat, _ = build_strategy(args, args.seeds[0])
        metas = [False, True] if args.meta_both else [bool(args.meta)]
        labels = {f"seed{s}{'_meta' if m else ''}": (s, m) for s in args.seeds for m in metas}
    log(f"{len(windows)} windows, strategy={args.strategy}, bars={args.bars}m, variants={list(labels)}")
    btc = BTC_SYMBOL if BTC_SYMBOL in bars else None

    closes = {s: d.select("timestamp", "close") for s, d in bars.items()}
    cands = {lab: [] for lab in labels}
    win_info = {lab: [] for lab in labels}
    pair_assets, arrays_by_key = {}, {}
    for w in windows:
        t0 = time.time()
        # only symbols with data covering most of the training window take part
        eligible = {s: c for s, c in closes.items()
                    if int(c["timestamp"][0]) <= w.train_start + 7 * DAY_MS and int(c["timestamp"][-1]) >= w.test_start}
        specs = cached_select(args, eligible, w)
        for sp in specs:
            pair_assets[sp.name] = (sp.asset1, sp.asset2)
        base_info = {"window": w.label(), "n_eligible": len(eligible), "pairs": [sp.to_dict() for sp in specs]}
        if args.strategy == "baseline":
            c = strat.generate(bars, specs, w)
            cands["seed0"] += c
            win_info["seed0"].append(dict(base_info, n_candidates=len(c)))
        else:
            wd = strat.window_data(bars, specs, w, btc)
            for seed in args.seeds:
                if wd is None:
                    res, fit = {m: [] for m in metas}, {"window": w.index, "traded": False, "reason": "no data"}
                else:
                    res, fit = strat.run_seed(wd, w, seed, tuple(metas)), strat.last_fit
                for m in metas:
                    lab = f"seed{seed}{'_meta' if m else ''}"
                    cands[lab] += res[m]
                    win_info[lab].append(dict(base_info, n_candidates=len(res[m]), fit=fit))
        if args.benchmark:
            for sp in specs:
                prep = prepare_pair(bars, sp, w.test_start, w.test_end, args.bars, 1440, 1440 + 2 * args.bars)
                if prep is not None:
                    arrays_by_key[(w.index, sp.name)] = (prep.arrays, *prep.trade_window)
        n = {lab: sum(1 for c in cands[lab] if c["window"] == w.index) for lab in labels}
        log(f"  {w.label()}: {len(specs)} pairs, candidates {n} ({time.time() - t0:.1f}s)")

    n_trials = args.n_trials or strat.n_trials()
    return {lab: finalize(args, lab, seed, cands[lab], win_info[lab], windows, costs, cfg, pair_assets,
                          funding, arrays_by_key, n_trials, run_dir)
            for lab, (seed, _) in labels.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", choices=["baseline", "xgb"], default="baseline")
    ap.add_argument("--bars", type=int, default=5, help="bar size in minutes")
    ap.add_argument("--train-days", type=int, default=90)
    ap.add_argument("--test-days", type=int, default=30)
    ap.add_argument("--holdout-days", type=int, default=90)
    ap.add_argument("--use-holdout", action="store_true")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--meta", action="store_true", help="xgb: enable meta-label filter")
    ap.add_argument("--meta-both", action="store_true", help="xgb: evaluate without AND with meta filter")
    ap.add_argument("--no-adf", action="store_true", help="xgb: skip rolling ADF feature (faster)")
    ap.add_argument("--entry-z", type=float, default=2.0, help="baseline entry threshold")
    ap.add_argument("--z-window", type=int, default=1440, help="baseline z-score window in minutes")
    ap.add_argument("--max-hold-days", type=float, default=3.0, help="baseline time-stop cap in days")
    ap.add_argument("--latency", type=int, default=0, help="extra bars between decision and fill")
    ap.add_argument("--cost-mult", type=float, default=1.0)
    ap.add_argument("--maker", action="store_true", help="assume maker fills (optimistic)")
    ap.add_argument("--exec", choices=["taker", "maker"], default="taker",
                    help="maker = passive fill model of protocol variants 10/11")
    ap.add_argument("--maker-timeout-bars", type=int, default=3)
    ap.add_argument("--maker-through-bp", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-pvalue", type=float, default=0.05)
    ap.add_argument("--equity", type=float, default=1000.0)
    ap.add_argument("--alloc", type=float, default=0.25)
    ap.add_argument("--max-positions", type=int, default=4)
    ap.add_argument("--max-leverage", type=float, default=1.0)
    ap.add_argument("--n-trials", type=int, default=0, help="trials for the Deflated Sharpe (0 = strategy grid)")
    ap.add_argument("--benchmark", type=int, default=0, help="number of random-entry simulations")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--synthetic-days", type=int, default=150)
    ap.add_argument("--synthetic-seed", type=int, default=7)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.synthetic:
        args.start, args.holdout_days, args.train_days, args.test_days = None, 0, min(args.train_days, 30), min(args.test_days, 15)
    if args.exec == "maker" and (args.latency or args.maker):
        raise SystemExit("--exec maker models fills itself: do not combine it with --latency or --maker")

    run_id, run_dir = new_run_dir(f"{args.strategy}{'_' + args.tag if args.tag else ''}")
    log(f"run {run_id}")
    bars, funding, files = load_universe(args)
    log(f"loaded {len(bars)} symbols, funding for {len(funding)}")
    mpath = write_manifest(run_dir, run_id, vars(args), files)
    with open(mpath) as f:
        hashes = json.load(f)["data_files"]
    args.data_digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest() if hashes else ""

    summaries = run_all(args, bars, funding, run_dir)
    for lab, s in summaries.items():
        log(f"{lab}: trades {s['n_trades']}, net {s.get('net_pnl', 0):.2f} USDT, "
            f"sharpe {s.get('sharpe', 0):.2f}, mean net/trade {s.get('mean_net_ret_bp', 0):.1f}bp "
            f"CI {s.get('mean_net_ret_ci95_bp')}, DSR {s.get('dsr', 0):.3f}")
    agg = {"run_id": run_id, "seeds": summaries}
    for suffix in ["", "_meta"]:
        group = {k: v for k, v in summaries.items() if k.endswith(suffix) and (suffix or not k.endswith("_meta"))}
        if len(group) > 1:
            nets = [v.get("net_pnl", 0.0) for v in group.values()]
            shs = [v.get("sharpe", 0.0) for v in group.values()]
            agg["across_seeds" + suffix] = {
                "labels": list(group), "net_pnl_mean": float(np.mean(nets)), "net_pnl_min": float(np.min(nets)),
                "net_pnl_max": float(np.max(nets)), "sharpe_mean": float(np.mean(shs)),
                "sharpe_min": float(np.min(shs)), "sharpe_max": float(np.max(shs)),
                "share_positive": float(np.mean(np.array(nets) > 0))}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(agg, f, indent=2, default=str)
    log(f"done -> {run_dir}")


if __name__ == "__main__":
    main()
