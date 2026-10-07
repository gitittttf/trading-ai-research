"""
Reconstruct Kraken funding before the 12 months the API returns, and validate the reconstruction
exactly as the protocol fixed it (phase 7, variant 17) before it is used.

Per hour t: rate = clip((P / I - 1) / k, -c, +c), settled at t + 1h, with P, I the mean of open and
close of the hourly perp candle ('mark' or 'trade') and of the index candle ('spot'); k and c are the
instrument's fundingRateCoefficient and maxRelativeFundingRate.
Validation on the days with real funding (until 2026-09-30): pooled correlation of daily sums
>= 0.9 and Jaccard >= 0.8 of the coin-days with a 7-day sum >= 0.35 %. The proxy with the higher
correlation is used. Writes data_kraken/funding_validation.json and, per coin,
data_kraken/<NAME>_funding.parquet = real funding where it exists, reconstructed before that
(only if the validation passes; otherwise real funding only).

  uv run python scripts/kraken_funding_recon.py
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import sys

import numpy as np
import polars as pl

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from core.daily import DAY_MS   # noqa: E402

KDIR = os.path.join(ROOT_DIR, "data_kraken")
HOUR_MS = 3_600_000
VALID_END = int(dt.datetime(2026, 10, 1, tzinfo=dt.UTC).timestamp() * 1000)
MIN_CORR, MIN_JACCARD, ENTER, MIN_EVENTS = 0.9, 0.8, 0.0035, 20


def reconstruct(name: str, proxy: str, k: float, cap: float) -> pl.DataFrame | None:
    paths = [os.path.join(KDIR, "hourly", f"{name}_{t}_1h.parquet") for t in (proxy, "spot")]
    if not all(os.path.exists(p) for p in paths):
        return None
    mid = lambda p: pl.read_parquet(p).select("timestamp", ((pl.col("open") + pl.col("close")) / 2).alias("m"))
    j = mid(paths[0]).join(mid(paths[1]), on="timestamp", suffix="_i").filter(pl.col("m_i") > 0)
    return j.select((pl.col("timestamp") + HOUR_MS).alias("timestamp"),
                    ((pl.col("m") / pl.col("m_i") - 1) / k).clip(-cap, cap).alias("funding_rate"))


def daily(f: pl.DataFrame) -> pl.DataFrame:
    # event at f belongs to day d if end(d-1) < f <= end(d), as in core.daily._map_funding
    return (f.with_columns(((pl.col("timestamp") + DAY_MS - 1) // DAY_MS * DAY_MS - DAY_MS).alias("day"))
             .group_by("day").agg(pl.col("funding_rate").sum().alias("s"), pl.len().alias("n")))


def main():
    meta = {f"{i['base']}USD": i for i in
            json.load(open(os.path.join(KDIR, "instruments.json"), encoding="utf-8"))["instruments"]}
    names = sorted(os.path.basename(p)[:-len("_funding.parquet")]
                   for p in glob.glob(os.path.join(KDIR, "actual", "*_funding.parquet")))
    report = {}
    recon_by = {}
    for proxy in ("mark", "trade"):
        xs, ys, sig_a, sig_r = [], [], set(), set()
        recon_by[proxy] = {}
        for name in names:
            m = meta.get(name)
            if m is None:
                continue
            r = reconstruct(name, proxy, float(m["fundingRateCoefficient"]), float(m["maxRelativeFundingRate"]))
            if r is None or r.height == 0:
                continue
            recon_by[proxy][name] = r
            a = pl.read_parquet(os.path.join(KDIR, "actual", f"{name}_funding.parquet"))
            a = a.filter(pl.col("timestamp") < VALID_END)
            both = (daily(a).join(daily(r.filter(pl.col("timestamp") < VALID_END)), on="day", suffix="_r")
                    .filter((pl.col("n") >= MIN_EVENTS) & (pl.col("n_r") >= MIN_EVENTS)).sort("day"))
            if both.height == 0:
                continue
            xs.append(both["s"].to_numpy())
            ys.append(both["s_r"].to_numpy())
            days = both["day"].to_numpy()
            for col, sig in (("s", sig_a), ("s_r", sig_r)):
                v = both[col].to_numpy()
                for i in range(6, len(v)):
                    if days[i] - days[i - 6] == 6 * DAY_MS and v[i - 6:i + 1].sum() >= ENTER:
                        sig.add((name, int(days[i])))
        x, y = np.concatenate(xs), np.concatenate(ys)
        union = sig_a | sig_r
        report[proxy] = {"coins": len(xs), "coin_days": int(len(x)), "corr_daily": float(np.corrcoef(x, y)[0, 1]),
                         "jaccard_entry_signal": len(sig_a & sig_r) / len(union) if union else float("nan"),
                         "entry_coin_days_actual": len(sig_a), "entry_coin_days_recon": len(sig_r),
                         "mean_daily_actual": float(x.mean()), "mean_daily_recon": float(y.mean())}
    best = max(report, key=lambda p: report[p]["corr_daily"])
    ok = report[best]["corr_daily"] >= MIN_CORR and report[best]["jaccard_entry_signal"] >= MIN_JACCARD
    out = {"proxies": report, "chosen": best, "passed": bool(ok),
           "rule": {"min_corr": MIN_CORR, "min_jaccard": MIN_JACCARD, "enter_7d": ENTER, "valid_end": "2026-09-30"}}
    n = 0
    for name in names:
        a = pl.read_parquet(os.path.join(KDIR, "actual", f"{name}_funding.parquet"))
        f = a
        if ok and name in recon_by[best] and a.height:
            f = pl.concat([recon_by[best][name].filter(pl.col("timestamp") < a["timestamp"].min()), a])
        f.sort("timestamp").write_parquet(os.path.join(KDIR, f"{name}_funding.parquet"))
        n += 1
    out["funding_files"] = n
    with open(os.path.join(KDIR, "funding_validation.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
