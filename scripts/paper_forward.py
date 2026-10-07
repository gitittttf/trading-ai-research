"""
Forward paper trading of the registered daily strategies on data published AFTER the
research data ended (last research day 2026-09-30). No money, no exchange account.

Every run rebuilds the whole forward period from the public archives with exactly the same
strategy code as the backtest (core/daily.py), so live and backtest cannot drift apart:
  * perp and spot daily klines: monthly archives, daily archives for the current month;
  * funding: official monthly archives where published, otherwise reconstructed from the
    1-minute premium index (core/funding_recon.py) and replaced once the archive appears;
  * decisions at each day's close use only data up to that close (the backtest's tests).

  uv run python scripts/paper_forward.py

Writes paper/forward_<strategy>.csv, paper/summary.json and appends paper/runs.jsonl.
Only variant 12 is the protocol's paper candidate; 9 and 14 are tracked as observations.
"""
from __future__ import annotations

import os as _os

for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

import argparse      # noqa: E402
import datetime as dt  # noqa: E402
import io            # noqa: E402
import json          # noqa: E402
import os            # noqa: E402
import sys           # noqa: E402
import time          # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

import numpy as np   # noqa: E402
import polars as pl  # noqa: E402

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "scripts"))

import fetch_binance_vision as fbv                                       # noqa: E402
import fetch_daily_universe as fdu                                       # noqa: E402
from core.costs import CostModel                                         # noqa: E402
from core.daily import (DAY_MS, add_spot_columns, build_panel_from_daily, carry_broad_weights,  # noqa: E402
                        harvest_weights, liquid_universe, run_daily, xs_momentum_weights)
from core.funding_recon import interval_hours_from_history, reconstruct_day  # noqa: E402
from core.manifest import git_state                                      # noqa: E402
from core.metrics import max_drawdown, sharpe                            # noqa: E402

FORWARD_START = dt.date(2026, 10, 1)      # first day whose return counts (decision at the close before)
WARMUP_MONTHS = 6                         # history the rules need (30d universe, 60d momentum, 7d funding)
SPOT_RATE = 0.0010 + 0.0001 + 0.00005     # same as scripts/daily_backtest.py
PAPER_DIR = os.path.join(ROOT_DIR, "paper")
BASE = "https://data.binance.vision/"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ms(d: dt.date) -> int:
    return int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.UTC).timestamp() * 1000)


def get(key: str) -> bytes | None:
    """Verified, cached archive file (csv bytes) or None if it does not exist."""
    blob = fdu.get_file(key)
    return None if blob is None else fbv._read_zip_csv(blob)


def months(first: dt.date, last: dt.date) -> list[dt.date]:
    out, d = [], first.replace(day=1)
    while d <= last:
        out.append(d)
        d = (d.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    return out


def days_of_month(m: dt.date, until: dt.date) -> list[dt.date]:
    d, out = m, []
    while d.month == m.month and d <= until:
        out.append(d)
        d += dt.timedelta(days=1)
    return out


def kline_keys(market: str, sym: str, m: dt.date, monthly: bool, day: dt.date | None = None) -> str:
    root = "data/futures/um" if market == "perp" else "data/spot"
    if monthly:
        return f"{root}/monthly/klines/{sym}/1d/{sym}-1d-{m:%Y-%m}.zip"
    return f"{root}/daily/klines/{sym}/1d/{sym}-1d-{day:%Y-%m-%d}.zip"


def load_klines(market: str, sym: str, mlist: list[dt.date], until: dt.date) -> pl.DataFrame | None:
    frames = []
    for m in mlist:
        raw = get(kline_keys(market, sym, m, True))
        if raw is not None:
            frames.append(fdu.parse_daily_kline(raw))
            continue
        for d in days_of_month(m, until):          # month not archived yet: daily files
            raw = get(kline_keys(market, sym, m, False, d))
            if raw is not None:
                frames.append(fdu.parse_daily_kline(raw))
    if not frames:
        return None
    return pl.concat(frames).unique(subset=["timestamp"], keep="last").sort("timestamp")


def load_funding(sym: str, mlist: list[dt.date]) -> tuple[np.ndarray, np.ndarray, set]:
    """Official funding (timestamps, rates) and the set of months it covers."""
    ts, rs, covered = [], [], set()
    for m in mlist:
        raw = get(f"data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{m:%Y-%m}.zip")
        if raw is None:
            continue
        f = fbv.parse_funding_csv(raw)
        ts.append(f["timestamp"].to_numpy())
        rs.append(f["funding_rate"].to_numpy())
        covered.add(m)
    if not ts:
        return np.array([], dtype=np.int64), np.array([]), covered
    t, r = np.concatenate(ts), np.concatenate(rs)
    o = np.argsort(t)
    return t[o], r[o], covered


def reconstruct(sym: str, days: list[dt.date], interval: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Funding from the 1m premium index for the given days. Returns (ts, rates, unknown events)."""
    ts, rs, unknown = [], [], 0
    for d in days:
        raw = get(f"data/futures/um/daily/premiumIndexKlines/{sym}/1m/{sym}-1m-{d:%Y-%m-%d}.zip")
        if raw is None:
            unknown += 24 // interval
            continue
        first = raw.split(b"\n", 1)[0]
        df = pl.read_csv(io.BytesIO(raw), has_header=first[:1].isalpha(), infer_schema=False)
        t = df[df.columns[0]].cast(pl.Int64).to_numpy()
        t = np.where(t > 10**14, t // 1000, t)
        x = df[df.columns[4]].cast(pl.Float64).to_numpy()
        e, r = reconstruct_day(t, x, ms(d), interval)
        unknown += int(np.isnan(r).sum())
        ok = np.isfinite(r)
        ts.append(e[ok])
        rs.append(r[ok])
    if not ts:
        return np.array([], dtype=np.int64), np.array([]), unknown
    return np.concatenate(ts), np.concatenate(rs), unknown


def perp_symbols() -> list[str]:
    syms = set()
    for prefix in ("data/futures/um/monthly/klines/", "data/futures/um/daily/klines/"):
        _, pre = fdu.s3_list(prefix, "/")
        syms |= {p.rstrip("/").rsplit("/", 1)[1] for p in pre}
    return sorted(s for s in syms if s.endswith("USDT") and "_" not in s and s.isascii())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--until", default=None, help="last day to use (default: yesterday UTC)")
    args = ap.parse_args()
    t0 = time.time()
    today = dt.datetime.now(dt.UTC).date()
    until = dt.date.fromisoformat(args.until) if args.until else today - dt.timedelta(days=1)
    first_month = (FORWARD_START.replace(day=1) - dt.timedelta(days=1)).replace(day=1)
    for _ in range(WARMUP_MONTHS - 1):
        first_month = (first_month - dt.timedelta(days=1)).replace(day=1)
    mlist = months(first_month, until)
    last_research_month = (FORWARD_START - dt.timedelta(days=1)).replace(day=1)

    syms = perp_symbols()
    log(f"{len(syms)} USDT perp symbols listed; months {mlist[0]:%Y-%m}..{mlist[-1]:%Y-%m}, until {until}")
    # active = traded in the last research month or listed later (delisted coins keep their history)
    def active(s):
        if get(kline_keys("perp", s, last_research_month, True)) is not None:
            return True
        return get(kline_keys("perp", s, mlist[-1], False, until)) is not None
    with ThreadPoolExecutor(args.workers) as ex:
        act = [s for s, ok in zip(syms, ex.map(active, syms)) if ok]
    with ThreadPoolExecutor(args.workers) as ex:
        perp = {s: df for s, df in zip(act, ex.map(lambda s: load_klines("perp", s, mlist, until), act))
                if df is not None and df.height}
    btc_last = dt.datetime.fromtimestamp(int(perp["BTCUSDT"]["timestamp"].max()) / 1000, dt.UTC).date()
    until = min(until, btc_last)
    log(f"{len(perp)} active perps loaded, data until {until} ({time.time() - t0:.0f}s)")

    # official funding where archived
    with ThreadPoolExecutor(args.workers) as ex:
        fund = dict(zip(perp, ex.map(lambda s: load_funding(s, mlist), perp)))
    # rank by volume to know which coins the rules can touch: reconstruct funding only for those
    start_ms, end_ms = ms(mlist[0]), ms(until) + DAY_MS
    panel0 = build_panel_from_daily(perp, {s: f[:2] for s, f in fund.items()}, start_ms, end_ms)
    fwd0 = int(np.searchsorted(panel0.days, ms(FORWARD_START))) - 1
    relevant = set()
    for d in range(max(fwd0 - 7, 30), len(panel0.days)):
        relevant |= {panel0.symbols[j] for j in liquid_universe(panel0, d, 100, 30)}
    recon_info = {}

    def fill(s):
        t, r, covered = fund[s]
        traded = {dt.datetime.fromtimestamp(int(x) / 1000, dt.UTC).date()
                  for x in perp[s].filter(pl.col("quote_volume") > 0)["timestamp"].to_list()}
        missing = [d for m in mlist if m not in covered for d in days_of_month(m, until) if d in traded]
        if not missing or s not in relevant:
            return s, (t, r), {"reconstructed_days": 0, "unknown_events": 0}
        h = interval_hours_from_history(t) or 8
        rt, rr, unk = reconstruct(s, missing, h)
        tt, rrr = np.concatenate([t, rt]), np.concatenate([r, rr])
        o = np.argsort(tt)
        return s, (tt[o], rrr[o]), {"reconstructed_days": len(missing), "unknown_events": unk, "interval_h": h}
    with ThreadPoolExecutor(args.workers) as ex:
        funding = {}
        for s, f, info in ex.map(fill, perp):
            funding[s] = f
            if info["reconstructed_days"]:
                recon_info[s] = info
    log(f"funding: {len(recon_info)} symbols with reconstructed days ({time.time() - t0:.0f}s)")

    # spot for the coins variant 12 can trade
    spot_syms = sorted(s for s in relevant if not s.startswith("1000"))
    with ThreadPoolExecutor(args.workers) as ex:
        spot = {s: df for s, df in zip(spot_syms, ex.map(lambda s: load_klines("spot", s, mlist, until), spot_syms))
                if df is not None and df.height}
    log(f"{len(spot)} spot pairs loaded ({time.time() - t0:.0f}s)")

    panel = build_panel_from_daily(perp, funding, start_ms, end_ms)
    T, N = panel.close.shape
    lo = int(np.searchsorted(panel.days, ms(FORWARD_START))) - 1      # decision at the last research close
    costs = CostModel()
    results = {}

    def start_at(w, reb, every_day: bool):
        """The paper account starts at the forward start holding what the strategy targets then:
        its current weights (rules that set weights every day) or those of its last rebalance."""
        w2, reb2 = w.copy(), reb.copy()
        if not every_day and reb[:lo + 1].any():
            w2[lo] = w[int(np.where(reb[:lo + 1])[0][-1])]
        reb2[lo] = True
        return w2, reb2

    wide, m = add_spot_columns(panel, spot)
    w12, reb12, eps = harvest_weights(wide, m, n_perp=N)
    w12, reb12 = start_at(w12, reb12, every_day=True)
    rate12 = np.where(np.arange(len(wide.symbols)) >= N, SPOT_RATE, costs.per_fill_rate())
    results["v12_harvest"] = (wide, run_daily(wide.slice(lo, T), w12[lo:T], costs, reb12[lo:T], col_rate=rate12),
                              w12, "paper candidate")
    w9, reb9, _ = carry_broad_weights(panel)
    w9, reb9 = start_at(w9, reb9, every_day=False)
    results["v9_carry"] = (panel, run_daily(panel.slice(lo, T), w9[lo:T], costs, reb9[lo:T]), w9, "observation only")
    w14, reb14, _ = xs_momentum_weights(panel)
    w14, reb14 = start_at(w14, reb14, every_day=False)
    results["v14_momentum"] = (panel, run_daily(panel.slice(lo, T), w14[lo:T], costs, reb14[lo:T]), w14,
                               "observation only")

    os.makedirs(PAPER_DIR, exist_ok=True)
    summary = {"forward_start": FORWARD_START.isoformat(), "data_until": until.isoformat(),
               "run_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "git": git_state(),
               "funding_reconstructed_symbols": len(recon_info),
               "funding_unknown_events": int(sum(v["unknown_events"] for v in recon_info.values())),
               "strategies": {}}
    for name, (pan, res, w, role) in results.items():
        days = [dt.datetime.fromtimestamp(d / 1000, dt.UTC).date().isoformat() for d in res.days]
        held = [";".join(f"{pan.symbols[j]}:{w[lo + i, j]:+.4f}" for j in np.where(w[lo + i] != 0)[0])
                for i in range(len(res.days))]
        pl.DataFrame({"day": days, "net": res.net, "gross": res.gross, "cost": res.cost, "funding": res.funding,
                      "turnover": res.turnover, "positions_decided_prev_close": held}).write_csv(
            os.path.join(PAPER_DIR, f"forward_{name}.csv"))
        eq = np.cumprod(1 + res.net) if len(res.net) else np.array([1.0])
        summary["strategies"][name] = {
            "role": role, "days": int(len(res.net)), "total_return": float(eq[-1] - 1),
            "max_drawdown": max_drawdown(np.concatenate([[1.0], eq])),
            "sharpe": sharpe(res.net) if len(res.net) >= 20 else None,
            "positions_now": int(np.sum(w[T - 1, :N] != 0))}
        log(f"{name} ({role}): {len(res.net)} days, total {(eq[-1] - 1) * 100:+.2f}%")
    with open(os.path.join(PAPER_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(PAPER_DIR, "runs.jsonl"), "a") as f:
        f.write(json.dumps({"run_utc": summary["run_utc"], "data_until": summary["data_until"],
                            "commit": summary["git"].get("commit"),
                            "total_return": {k: v["total_return"] for k, v in summary["strategies"].items()},
                            "funding_reconstructed_symbols": summary["funding_reconstructed_symbols"]}) + "\n")
    log(f"done ({time.time() - t0:.0f}s) -> {PAPER_DIR}")


if __name__ == "__main__":
    main()
