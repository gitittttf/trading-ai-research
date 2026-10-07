"""
Kraken data for protocol variant 17 (funding harvest on a venue EU residents may use).

Public Kraken Futures API, perps PF_<BASE>USD that the EU platform ("europa") permits:
  data_kraken/<BASE>USD_1d.parquet          perp daily 'trade' candles: timestamp, close, high, quote_volume
  data_kraken/spot/<BASE>USD_1d.parquet     daily 'spot' (index) candles for coins with a Kraken spot
                                            pair <BASE>/USD (quote_volume = 1: the index has no volume)
  data_kraken/actual/<BASE>USD_funding.parquet   real hourly relativeFundingRate (last 12 months only)
  data_kraken/hourly/<BASE>USD_<tick>_1h.parquet hourly mark/trade/spot candles for the reconstruction
  data_kraken/instruments.json              instrument metadata snapshot (coefficients, caps)
Raw API responses are cached in data/raw/kraken/. The API knows only today's instruments, so
delisted perps are missing (survivorship, see protocol).

  uv run python scripts/fetch_kraken.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import polars as pl

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "scripts"))

import fetch_binance_vision as fbv   # noqa: E402

FUT = "https://futures.kraken.com"
SPOT_PAIRS = "https://api.kraken.com/0/public/AssetPairs"
OUT = os.path.join(ROOT_DIR, "data_kraken")
RAW = os.path.join(ROOT_DIR, "data", "raw", "kraken")
SPOT_ALIAS = {"XBT": "BTC", "XDG": "DOGE"}      # Kraken spot names -> futures base names
HOUR_S = 3600
PAGE = 2000                                      # candles per chart request


def get_json(url: str, cache: bool = True) -> dict:
    """GET JSON; immutable responses (closed time ranges) are cached on disk."""
    path = os.path.join(RAW, hashlib.sha256(url.encode()).hexdigest()[:24] + ".json")
    if cache and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    blob = fbv.http_get(url)
    if blob is None:
        raise RuntimeError(f"404 {url}")
    data = json.loads(blob)
    if cache:
        os.makedirs(RAW, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    return data


def eu_perps() -> list[dict]:
    ins = get_json(f"{FUT}/derivatives/api/v3/instruments", cache=False)["instruments"]
    return sorted((i for i in ins if i["symbol"].startswith("PF_") and i.get("tradeable")
                   and "europa" in (i.get("platformsPermitted") or []) and not i.get("tradfi")),
                  key=lambda i: i["symbol"])


def spot_bases() -> set[str]:
    res = get_json(SPOT_PAIRS, cache=False)["result"]
    out = set()
    for v in res.values():
        ws = v.get("wsname", "")
        if ws.endswith("/USD"):
            b = ws[:-4]
            out.add(SPOT_ALIAS.get(b, b))
    return out


def candles(tick: str, symbol: str, res: str, start_s: int, end_s: int) -> list[dict]:
    """All candles in [start_s, end_s); pages are cached once they lie completely in the past."""
    step = {"1h": HOUR_S, "1d": 86_400}[res]
    out, frm = [], start_s
    while frm < end_s:
        to = min(frm + PAGE * step, end_s)
        url = f"{FUT}/api/charts/v1/{tick}/{symbol}/{res}?from={frm}&to={to}"
        page = get_json(url, cache=to < time.time() - 2 * step).get("candles", [])
        out += [c for c in page if frm * 1000 <= c["time"] < to * 1000]
        frm = to
    seen, uniq = set(), []
    for c in out:
        if c["time"] not in seen:
            seen.add(c["time"])
            uniq.append(c)
    return sorted(uniq, key=lambda c: c["time"])


def frame(cs: list[dict], cols: tuple[str, ...]) -> pl.DataFrame:
    return pl.DataFrame({"timestamp": [int(c["time"]) for c in cs],
                         **{k: [float(c[k]) for c in cs] for k in cols}},
                        schema={"timestamp": pl.Int64, **{k: pl.Float64 for k in cols}})


def fetch_symbol(inst: dict, has_spot: bool, start_s: int, end_s: int) -> dict:
    sym = inst["symbol"]
    name = f"{inst['base']}USD"
    rep = {"symbol": sym, "name": name, "spot_pair": has_spot}
    day = frame(candles("trade", sym, "1d", start_s, end_s), ("close", "high", "volume"))
    if day.height == 0:
        return rep
    day = day.with_columns((pl.col("volume") * pl.col("close")).alias("quote_volume")).drop("volume")
    day.write_parquet(os.path.join(OUT, f"{name}_1d.parquet"))
    rep["days"] = day.height
    if has_spot:
        idx = frame(candles("spot", sym, "1d", start_s, end_s), ("close", "high"))
        idx.with_columns(pl.lit(1.0).alias("quote_volume")).write_parquet(os.path.join(OUT, "spot", f"{name}_1d.parquet"))
    f = get_json(f"{FUT}/derivatives/api/v4/historicalfundingrates?symbol={sym}", cache=False).get("rates", [])
    if f:
        ts = [int(dt.datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")).timestamp() * 1000) for r in f]
        pl.DataFrame({"timestamp": ts, "funding_rate": [float(r["relativeFundingRate"]) for r in f]},
                     schema={"timestamp": pl.Int64, "funding_rate": pl.Float64}).sort("timestamp").write_parquet(
            os.path.join(OUT, "actual", f"{name}_funding.parquet"))
        rep["funding_rows"] = len(f)
    for tick in ("mark", "trade", "spot"):
        h = frame(candles(tick, sym, "1h", start_s, end_s), ("open", "close"))
        h.write_parquet(os.path.join(OUT, "hourly", f"{name}_{tick}_1h.parquet"))
        rep[f"hours_{tick}"] = h.height
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2022-03-01")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    for sub in ("spot", "actual", "hourly"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    start_s = int(dt.datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=dt.UTC).timestamp())
    end_s = int(time.time()) // 86_400 * 86_400            # up to the last full UTC day
    perps, spots = eu_perps(), spot_bases()
    with open(os.path.join(OUT, "instruments.json"), "w", encoding="utf-8") as f:
        json.dump({"fetched_utc": dt.datetime.now(dt.UTC).isoformat(), "instruments": perps}, f, indent=1)
    print(f"{len(perps)} EU perps, {sum(p['base'] in spots for p in perps)} with a Kraken spot pair", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        reps = list(ex.map(lambda p: fetch_symbol(p, p["base"] in spots, start_s, end_s), perps))
    with open(os.path.join(OUT, "fetch_report.json"), "w", encoding="utf-8") as f:
        json.dump(reps, f, indent=1)
    print(f"done: {sum('days' in r for r in reps)} perps with daily candles ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
