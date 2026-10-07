"""
Daily portfolio strategies (protocol variants 7 and 8) on Binance USD-M perpetuals.

  uv run python scripts/daily_backtest.py --strategy trend --n-trials 10 --benchmark 500
  uv run python scripts/daily_backtest.py --strategy carry --n-trials 10 --benchmark 500

Parameters are fixed a priori (literature values, see docs/EVALUATION_PROTOCOL.md);
nothing is fitted, so the whole period before the holdout is out-of-sample except for
the choice of the strategy family itself (accounted for by the DSR trial count).
"""
from __future__ import annotations

import os as _os

for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

import argparse      # noqa: E402
import datetime as dt  # noqa: E402
import json          # noqa: E402
import os            # noqa: E402
import sys           # noqa: E402
import time          # noqa: E402

import numpy as np   # noqa: E402

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from core.constants import BTC_SYMBOL, DATA_DIR, UNIVERSE                  # noqa: E402
from core.costs import CostModel                                            # noqa: E402
from core.daily import (DAY_MS, DailyResult, add_spot_columns, block_bootstrap_mean_ci, build_panel,  # noqa: E402
                        build_panel_from_daily, carry_broad_weights, carry_weights, harvest_sized,
                        harvest_weights, listing_short_weights, margin_breaches, run_daily, trend_weights, xs_momentum_weights)
from core.data import funding_path, klines_path, load_funding, load_klines  # noqa: E402
from core.manifest import new_run_dir, write_manifest                      # noqa: E402
from core.metrics import deflated_sharpe, max_drawdown, sharpe, sortino   # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stats(res, n_trials: int) -> dict:
    net = res.net
    eq = np.cumprod(1 + net)
    m, lo, hi = block_bootstrap_mean_ci(net)
    years = len(net) / 365.0
    return {
        "n_days": int(len(net)),
        "total_return": float(eq[-1] - 1) if len(eq) else 0.0,
        "cagr": float(eq[-1] ** (1 / years) - 1) if len(eq) and years > 0 and eq[-1] > 0 else -1.0,
        "sharpe": sharpe(net), "sortino": sortino(net),
        "max_drawdown": max_drawdown(np.concatenate([[1.0], eq])),
        "mean_daily_bp": m * 1e4, "mean_daily_ci95_bp": [lo * 1e4, hi * 1e4],
        "dsr": deflated_sharpe(net, n_trials), "n_trials_for_dsr": n_trials,
        "gross_return_sum": float(res.gross.sum()), "cost_sum": float(res.cost.sum()),
        "funding_sum": float(res.funding.sum()),
        "turnover_per_year": float(res.turnover.sum() / max(years, 1e-9)),
        "avg_gross_exposure": float(res.exposure.mean()) if len(res.exposure) else 0.0,
    }


def permuted(weights: np.ndarray, panel, rng, rebalance: np.ndarray | None,
             eligible: np.ndarray | None = None) -> np.ndarray:
    """Shuffle each decision's weight vector among the coins that are tradable (or eligible)
    then: same number of positions and same exposure, random coins."""
    w = np.zeros_like(weights)
    for d in range(len(weights)):
        if rebalance is not None and not rebalance[d]:
            continue
        mask = eligible[d] if eligible is not None else np.isfinite(panel.close[d])
        live = np.where(mask)[0]
        if len(live) == 0:
            continue
        vals = weights[d, live]
        w[d, live] = vals[rng.permutation(len(live))]
    return w


def finish(args, run_dir, panel, weights, reb, lo, hi, costs, btc_symbol, eligible=None, col_rate=None,
           bench_fn=None, extra=None, cash=None, hurdle=None):
    """bench_fn(rng) -> (weights, rebalance) of one random-signal simulation (default: shuffle the
    weights between eligible coins). col_rate: per-column cost rates (run_daily).
    cash: optional (cash_rate, tied) per panel day, interest on idle capital (run_daily).
    hurdle: optional per-day T-bill return; adds the comparison with holding only T-bills."""
    sub = panel.slice(lo, hi)
    run_kw = {} if cash is None else {"cash_rate": cash[0][lo:hi], "tied": cash[1][lo:hi]}
    res = run_daily(sub, weights[lo:hi], costs, None if reb is None else reb[lo:hi], col_rate=col_rate, **run_kw)
    summary = stats(res, args.n_trials)
    summary["period"] = [dt.datetime.fromtimestamp(sub.days[0] / 1000, dt.UTC).date().isoformat(),
                         dt.datetime.fromtimestamp(sub.days[-1] / 1000, dt.UTC).date().isoformat()]
    if res.interest is not None:
        summary["interest_sum"] = float(res.interest.sum())
    if hurdle is not None:
        tb = hurdle[lo:hi][1:len(res.net) + 1]          # T-bill return over the same days as res.net
        summary["tbill_total_return"] = float(np.prod(1 + tb) - 1)
        summary["excess_over_tbill_total"] = float(np.prod(1 + res.net) / np.prod(1 + tb) - 1)
    j_btc = panel.symbols.index(btc_symbol)
    w_btc = np.zeros_like(weights)
    w_btc[:, j_btc] = 1.0
    b_btc = run_daily(sub, w_btc[lo:hi], costs)
    base = eligible if eligible is not None else np.isfinite(panel.close)
    w_ew = np.where(base, 1.0, 0.0)
    w_ew = w_ew / np.maximum(w_ew.sum(axis=1, keepdims=True), 1)
    b_ew = run_daily(sub, w_ew[lo:hi], costs)
    summary["benchmark_btc_long"] = stats(b_btc, 1)
    summary["benchmark_equal_weight_long"] = stats(b_ew, 1)
    if args.benchmark:
        rng = np.random.default_rng(0)
        sims = []
        for _ in range(args.benchmark):
            if bench_fn is None:
                wr, rr = permuted(weights, panel, rng, reb, eligible), reb
            else:
                wr, rr = bench_fn(rng)
            sims.append(sharpe(run_daily(sub, wr[lo:hi], costs, None if rr is None else rr[lo:hi],
                                         col_rate=col_rate, **run_kw).net))
        sims = np.array(sims)
        summary["random_signal_benchmark"] = {
            "n_sims": len(sims), "real_sharpe": summary["sharpe"], "random_sharpe_mean": float(sims.mean()),
            "random_sharpe_std": float(sims.std()),
            "p_value": float((np.sum(sims >= summary["sharpe"]) + 1) / (len(sims) + 1))}
    if extra:
        summary.update(extra)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    np.savetxt(os.path.join(run_dir, "daily_net_returns.csv"), np.column_stack([res.days, res.net]),
               delimiter=",", header="day_ms,net_return", comments="", fmt=["%d", "%.10f"])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = [dt.datetime.fromtimestamp(d / 1000, dt.UTC) for d in res.days]
        plt.figure(figsize=(11, 5))
        plt.plot(x, np.cumprod(1 + res.net), label=f"{args.strategy} (net)")
        plt.plot(x, np.cumprod(1 + b_btc.net), label="BTC long (net)", alpha=0.6)
        plt.plot(x, np.cumprod(1 + b_ew.net), label="equal-weight long (net)", alpha=0.6)
        plt.legend()
        plt.title(f"{args.strategy}: growth of 1 (all costs and funding)")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "equity.png"), dpi=120)
        plt.close()
    except Exception as e:
        log(f"plot failed: {e}")
    s = summary
    log(f"{args.strategy}: {s['period']} days {s['n_days']}, total {s['total_return'] * 100:.1f}%, "
        f"sharpe {s['sharpe']:.2f}, maxDD {s['max_drawdown'] * 100:.1f}%, mean {s['mean_daily_bp']:.2f}bp/day "
        f"CI {s['mean_daily_ci95_bp']}, DSR {s['dsr']:.3f}, bench p {s.get('random_signal_benchmark', {}).get('p_value')}")
    log(f"done -> {run_dir}")
    return res, summary


def load_broad(args):
    import glob
    import polars as pl
    daily, funding, files = {}, {}, []
    for path in sorted(glob.glob(os.path.join(args.daily_dir, "*_1d.parquet"))):
        sym = os.path.basename(path)[:-len("_1d.parquet")]
        daily[sym] = pl.read_parquet(path)
        files.append(path)
        fp = os.path.join(args.daily_dir, f"{sym}_funding.parquet")
        if os.path.exists(fp):
            f = pl.read_parquet(fp)
            funding[sym] = (f["timestamp"].to_numpy(), f["funding_rate"].to_numpy())
            files.append(fp)
    spot = {}
    if args.strategy in ("harvest", "harvest_sized", "combo"):
        for path in sorted(glob.glob(os.path.join(args.daily_dir, "spot", "*_1d.parquet"))):
            spot[os.path.basename(path)[:-len("_1d.parquet")]] = pl.read_parquet(path)
            files.append(path)
    return daily, funding, spot, files


SPOT_RATE = 0.0010 + 0.0001 + 0.00005      # spot VIP0 fee + half spread + slippage per fill
PHASE6_WARMUP_DAYS = 90                    # history loaded before the evaluation start
MAINTENANCE_MARGIN = 0.05                  # of perp notional, for the phase-7 margin check


def tbill_daily(path: str, days: np.ndarray) -> np.ndarray:
    """Return of 3-month T-bills over each panel day (FRED DTB3, percent, act/360): the last rate
    published before the day starts, so nothing from the day itself is used."""
    import polars as pl
    df = pl.read_csv(path, infer_schema=False).rename(lambda c: c.lower())
    df = df.filter(pl.col("dtb3") != ".").select(
        pl.col("observation_date").str.to_date().cast(pl.Datetime("ms")).cast(pl.Int64).alias("ts"),
        pl.col("dtb3").cast(pl.Float64))
    idx = np.searchsorted(df["ts"].to_numpy(), days, side="left") - 1
    rate = np.where(idx >= 0, df["dtb3"].to_numpy()[np.maximum(idx, 0)], np.nan)
    return np.nan_to_num(rate / 100.0 / 360.0)


def phase6_component(name, args, panel, spot, lo, hi, costs, first_trade_ms=None, tbill=None):
    """(weights, rebalance, col_rate, bench_fn, eligible, extra, panel, cash) of one phase-6/7 strategy.
    first_trade_ms: symbol -> first day with volume in the WHOLE archive (listing date).
    tbill: per-day T-bill return (variant 16 earns it on idle capital); cash = (rate, tied) or None."""
    import datetime as _dt
    T, N = panel.close.shape
    base = CostModel().per_fill_rate()
    perp_live = np.isfinite(panel.close)
    if name in ("harvest", "harvest_sized"):
        wide, m = add_spot_columns(panel, spot)
        w, reb, eps = harvest_weights(wide, m, n_perp=N)
        rate = np.where(np.arange(len(wide.symbols)) >= N, args.spot_rate, base)
        elig = np.zeros(wide.close.shape, dtype=bool)
        elig[:, :N] = perp_live
        n_in = sum(1 for _, e, _ in eps if lo <= e < hi)
        extra = {"episodes_started": n_in, "spot_pairs": len(m)}
        margin, cash = 0.5, None
        if name == "harvest":
            bench = lambda rng: harvest_weights(wide, m, n_perp=N, rng=rng, episodes=eps)[:2]
        else:
            w, reb, tied = harvest_sized(wide, w, reb, N)
            cash = (tbill if tbill is not None else np.zeros(T), tied)
            margin = 1.0
            bench = lambda rng: harvest_sized(wide, *harvest_weights(wide, m, n_perp=N, rng=rng, episodes=eps)[:2],
                                              N)[:2]
            extra["rehedges"] = int(np.sum(reb[lo:hi])) - n_in
            extra["avg_capital_tied"] = float(tied[lo:hi].mean())
        if wide.high is not None:
            br = margin_breaches(wide.slice(lo, hi), w[lo:hi], reb[lo:hi], N, margin, MAINTENANCE_MARGIN)
            held = w[lo:hi, :N] < 0
            extra["margin_check"] = {
                "margin_per_notional": margin, "maintenance": MAINTENANCE_MARGIN, "breach_days": len(br),
                "breach_dates": [_dt.datetime.fromtimestamp(wide.days[lo + b] / 1000, _dt.UTC).date().isoformat()
                                 for b in br][:20],
                "held_days_with_high": float(np.isfinite(wide.high[lo:hi, :N][held]).mean()) if held.any() else None}
        return w, reb, rate, bench, elig, extra, wide, cash
    if name == "listing_short":
        cutoff = _dt.datetime(2021, 2, 1, tzinfo=_dt.UTC).timestamp() * 1000
        # listing day = first day with volume in the whole archive (starts 2021-01), so coins that
        # existed before the archive never count; the entry day L + 1 must lie in [lo, hi)
        lst = {}
        for j, sym in enumerate(panel.symbols):
            ts = (first_trade_ms or {}).get(sym)
            if ts is None or ts < cutoff:
                continue
            L = int(np.searchsorted(panel.days, ts // DAY_MS * DAY_MS))
            if L < T and panel.days[L] == ts // DAY_MS * DAY_MS and lo <= L + 1 < hi:
                lst[j] = L
        btc = panel.symbols.index("BTCUSDT")
        w, reb, ev = listing_short_weights(panel, lst, btc)
        rate = np.where(np.arange(N) == btc, base, 2 * base)
        bench = lambda rng: listing_short_weights(panel, lst, btc, rng=rng)[:2]
        return w, reb, rate, bench, perp_live, {"events": len(ev), "listings_in_period": len(lst)}, panel, None
    if name == "momentum":
        w, reb, elig = xs_momentum_weights(panel)
        return w, reb, None, None, elig, {}, panel, None
    raise ValueError(name)


def run_broad(args, run_id, run_dir):
    daily, funding, spot, files = load_broad(args)
    write_manifest(run_dir, run_id, vars(args), files)
    log(f"run {run_id}: {len(daily)} symbols (broad universe), {len(spot)} spot pairs")
    start_ms = int(dt.datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=dt.UTC).timestamp() * 1000)
    # quote currency of the venue: Binance USDT perps (BTCUSDT), Kraken USD perps (BTCUSD)
    quote = "USDT" if "BTCUSDT" in daily else "USD"
    btc_sym = "BTC" + quote
    fund_end = min(int(funding[s][0][-1]) for s in [btc_sym, "ETH" + quote]) + 8 * 3_600_000
    end_ms = fund_end // DAY_MS * DAY_MS
    if args.end:
        end_ms = min(end_ms, int(dt.datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=dt.UTC).timestamp() * 1000))
        if args.holdout_days:
            log("explicit --end: no holdout is carved out of this period")
            args.holdout_days = 0
    if args.strategy == "carry_broad":           # variant 9, unchanged
        panel = build_panel_from_daily(daily, funding, start_ms, end_ms)
        T = len(panel.days)
        weights, reb, elig = carry_broad_weights(panel)
        holdout_start = T - args.holdout_days
        if args.use_holdout:
            log("!!! HOLDOUT EVALUATION - once per strategy, then recorded")
            lo, hi = holdout_start - 1, T
        else:
            lo, hi = 30, holdout_start
        finish(args, run_dir, panel, weights, reb, lo, hi, CostModel(multiplier=args.cost_mult), btc_sym, elig)
        return

    # phase 6: load a warm-up before the evaluation start so every rule has its history
    panel = build_panel_from_daily(daily, funding, start_ms - PHASE6_WARMUP_DAYS * DAY_MS, end_ms)
    T = len(panel.days)
    first_eval = int(np.searchsorted(panel.days, start_ms))
    data_start = int(np.argmax(np.isfinite(panel.close).any(axis=1)))
    lo = max(first_eval, data_start + 60)        # 60 days = the longest history any rule needs
    import polars as pl
    first_trade = {sym: int(df.filter(pl.col("quote_volume") > 0)["timestamp"].min())
                   for sym, df in daily.items() if df.filter(pl.col("quote_volume") > 0).height}
    hi = T - args.holdout_days
    if args.use_holdout:
        log("!!! HOLDOUT EVALUATION (phase 6: informative only, see protocol)")
        lo, hi = hi - 1, T
    costs = CostModel(multiplier=args.cost_mult)
    tbill = tbill_daily(args.rates_file, panel.days) if os.path.exists(args.rates_file) else None
    if tbill is None and args.strategy == "harvest_sized":
        raise SystemExit(f"{args.rates_file} missing: run scripts/fetch_rates.py first")
    if args.strategy != "combo":
        w, reb, rate, bench, elig, extra, pan, cash = phase6_component(args.strategy, args, panel, spot, lo, hi, costs,
                                                                       first_trade, tbill)
        finish(args, run_dir, pan, w, reb, lo, hi, costs, btc_sym, elig, rate, bench, extra, cash,
               tbill if args.strategy.startswith("harvest") else None)
        return
    # variant 15: fixed thirds of the capital in variants 12, 13, 14 (return-level combination)
    parts, comp = [], {}
    for name in ("harvest", "listing_short", "momentum"):
        w, reb, rate, bench, elig, extra, pan, _cash = phase6_component(name, args, panel, spot, lo, hi, costs, first_trade)
        sub_dir = os.path.join(run_dir, name)
        os.makedirs(sub_dir, exist_ok=True)
        args_c = argparse.Namespace(**{**vars(args), "strategy": name})
        res, summ = finish(args_c, sub_dir, pan, w, reb, lo, hi, costs, btc_sym, elig, rate, bench, extra)
        parts.append(res)
        comp[name] = {k: summ[k] for k in ("total_return", "sharpe", "max_drawdown", "mean_daily_bp")}
    fields = ["days", "net", "gross", "cost", "funding", "turnover", "exposure"]
    combo = DailyResult(**{k: (parts[0].days if k == "days" else np.mean([getattr(r, k) for r in parts], axis=0))
                           for k in fields})
    summary = stats(combo, args.n_trials)
    summary["period"] = [dt.datetime.fromtimestamp(combo.days[0] / 1000 - 86400, dt.UTC).date().isoformat(),
                         dt.datetime.fromtimestamp(combo.days[-1] / 1000, dt.UTC).date().isoformat()]
    summary["components"] = comp
    summary["component_correlation"] = np.corrcoef([r.net for r in parts]).round(3).tolist()
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    np.savetxt(os.path.join(run_dir, "daily_net_returns.csv"), np.column_stack([combo.days, combo.net]),
               delimiter=",", header="day_ms,net_return", comments="", fmt=["%d", "%.10f"])
    s = summary
    log(f"combo: {s['period']} days {s['n_days']}, total {s['total_return'] * 100:.1f}%, sharpe {s['sharpe']:.2f}, "
        f"maxDD {s['max_drawdown'] * 100:.1f}%, mean {s['mean_daily_bp']:.2f}bp/day CI {s['mean_daily_ci95_bp']}, "
        f"DSR {s['dsr']:.3f}, corr {s['component_correlation']}")
    log(f"done -> {run_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", choices=["trend", "carry", "carry_broad", "harvest", "harvest_sized", "listing_short",
                                           "momentum",
                                           "combo"], required=True)
    ap.add_argument("--universe", choices=["fixed", "broad"], default="fixed",
                    help="fixed = core.constants.UNIVERSE from 1m data; broad = all USDT perps from data_daily/")
    ap.add_argument("--daily-dir", default=os.path.join(ROOT_DIR, "data_daily"))
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default=None, help="exclusive end date YYYY-MM-DD (out-of-period tests)")
    ap.add_argument("--holdout-days", type=int, default=90)
    ap.add_argument("--use-holdout", action="store_true")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--benchmark", type=int, default=500)
    ap.add_argument("--cost-mult", type=float, default=1.0)
    ap.add_argument("--spot-rate", type=float, default=SPOT_RATE,
                    help="spot cost per fill before --cost-mult (phase 6/7 harvest)")
    ap.add_argument("--rates-file", default=os.path.join(ROOT_DIR, "data_daily", "rates", "DTB3.csv"),
                    help="FRED DTB3 csv (scripts/fetch_rates.py): T-bill hurdle and interest on idle capital")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    run_id, run_dir = new_run_dir(f"daily_{args.strategy}{'_' + args.tag if args.tag else ''}")
    if args.strategy not in ("trend", "carry") and args.universe != "broad":
        raise SystemExit(f"{args.strategy} needs --universe broad")
    bars, funding, files = {}, {}, []
    if args.universe == "broad":
        run_broad(args, run_id, run_dir)
        return
    for s in UNIVERSE:
        df = load_klines(s, args.data_dir)
        if df is None:
            continue
        bars[s] = df.select("timestamp", "close")
        files.append(klines_path(s, args.data_dir))
        f = load_funding(s, args.data_dir)
        if f is not None:
            funding[s] = (f["timestamp"].to_numpy(), f["funding_rate"].to_numpy())
            files.append(funding_path(s, args.data_dir))
    write_manifest(run_dir, run_id, vars(args), files)
    log(f"run {run_id}: {len(bars)} symbols")

    # evaluation never goes past the funding coverage (monthly archives only)
    ends = [int(ts[-1]) for s, (ts, _) in funding.items()
            if int(bars[s]["timestamp"][-1]) > int(ts[-1]) + DAY_MS]
    data_end = (min(ends) // DAY_MS + 1) * DAY_MS if ends else None
    start_ms = int(dt.datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=dt.UTC).timestamp() * 1000)
    if args.end:
        end_ms = int(dt.datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=dt.UTC).timestamp() * 1000)
        data_end = min(data_end, end_ms) if data_end else end_ms
        if args.holdout_days:
            log("explicit --end: no holdout is carved out of this period")
            args.holdout_days = 0
    panel = build_panel(bars, funding, start_ms=start_ms, end_ms=data_end)
    T = len(panel.days)
    holdout_start = T - args.holdout_days
    costs = CostModel(multiplier=args.cost_mult)

    if args.strategy == "trend":
        weights, reb, warm = trend_weights(panel), None, 160
    else:
        weights, reb = carry_weights(panel)
        warm = 7
    if args.use_holdout:
        log("!!! HOLDOUT EVALUATION - once per strategy, then recorded")
        lo, hi = holdout_start - 1, T
    else:
        lo, hi = warm, holdout_start
    # restrict to the evaluation range [lo, hi): decisions from lo, returns until hi-1
    sub = type(panel)(panel.days[lo:hi], panel.symbols, panel.close[lo:hi], panel.funding[lo:hi])
    res = run_daily(sub, weights[lo:hi], costs, None if reb is None else reb[lo:hi])
    summary = stats(res, args.n_trials)
    summary["period"] = [dt.datetime.fromtimestamp(sub.days[0] / 1000, dt.UTC).date().isoformat(),
                         dt.datetime.fromtimestamp(sub.days[-1] / 1000, dt.UTC).date().isoformat()]

    # benchmarks with identical costs and funding
    j_btc = panel.symbols.index(BTC_SYMBOL)
    w_btc = np.zeros_like(weights)
    w_btc[:, j_btc] = 1.0
    b_btc = run_daily(sub, w_btc[lo:hi], costs)
    live = np.isfinite(panel.close)
    w_ew = np.where(live, 1.0, 0.0)
    w_ew = w_ew / np.maximum(w_ew.sum(axis=1, keepdims=True), 1)
    b_ew = run_daily(sub, w_ew[lo:hi], costs)
    summary["benchmark_btc_long"] = stats(b_btc, 1)
    summary["benchmark_equal_weight_long"] = stats(b_ew, 1)

    if args.benchmark:
        rng = np.random.default_rng(0)
        sims = []
        for _ in range(args.benchmark):
            wr = permuted(weights, panel, rng, reb)
            sims.append(sharpe(run_daily(sub, wr[lo:hi], costs, None if reb is None else reb[lo:hi]).net))
        sims = np.array(sims)
        summary["random_signal_benchmark"] = {
            "n_sims": len(sims), "real_sharpe": summary["sharpe"], "random_sharpe_mean": float(sims.mean()),
            "random_sharpe_std": float(sims.std()),
            "p_value": float((np.sum(sims >= summary["sharpe"]) + 1) / (len(sims) + 1))}

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    np.savetxt(os.path.join(run_dir, "daily_net_returns.csv"), np.column_stack([res.days, res.net]),
               delimiter=",", header="day_ms,net_return", comments="", fmt=["%d", "%.10f"])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = [dt.datetime.fromtimestamp(d / 1000, dt.UTC) for d in res.days]
        plt.figure(figsize=(11, 5))
        plt.plot(x, np.cumprod(1 + res.net), label=f"{args.strategy} (net)")
        plt.plot(x, np.cumprod(1 + b_btc.net), label="BTC long (net)", alpha=0.6)
        plt.plot(x, np.cumprod(1 + b_ew.net), label="equal-weight long (net)", alpha=0.6)
        plt.legend()
        plt.title(f"{args.strategy}: growth of 1 (all costs and funding)")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "equity.png"), dpi=120)
        plt.close()
    except Exception as e:
        log(f"plot failed: {e}")
    s = summary
    log(f"{args.strategy}: days {s['n_days']}, total {s['total_return'] * 100:.1f}%, sharpe {s['sharpe']:.2f}, "
        f"maxDD {s['max_drawdown'] * 100:.1f}%, mean {s['mean_daily_bp']:.2f}bp/day CI {s['mean_daily_ci95_bp']}, "
        f"DSR {s['dsr']:.3f}, bench p {s.get('random_signal_benchmark', {}).get('p_value')}")
    log(f"done -> {run_dir}")


if __name__ == "__main__":
    main()
