"""
Download DAILY klines (with quote volume) and funding for ALL Binance USD-M USDT
perpetuals ever listed in the public archive, including delisted ones.

Used by protocol variant 9 (broad point-in-time carry): the tradable universe is
chosen at each rebalance from liquidity known at that time, so it does not depend
on which coins survived until today. Every file is verified against its SHA-256.

  uv run python scripts/fetch_daily_universe.py --start 2021-01
Writes data_daily/<SYMBOL>_1d.parquet (timestamp, close, quote_volume, high) and
data_daily/<SYMBOL>_funding.parquet (timestamp, funding_rate).
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor

import polars as pl

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "scripts"))

import fetch_binance_vision as fbv   # noqa: E402

LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
OUT_DIR = os.path.join(ROOT_DIR, "data_daily")
RAW = os.path.join(ROOT_DIR, "data", "raw", "binance_vision_daily")


def s3_list(prefix: str, delimiter: str | None = None) -> tuple[list[str], list[str]]:
    """(keys, common_prefixes) of the public archive bucket under ``prefix`` (paginated)."""
    keys, prefixes, marker = [], [], ""
    while True:
        url = (f"{LIST_URL}?prefix={quote(prefix)}&marker={quote(marker)}"
               + (f"&delimiter={quote(delimiter)}" if delimiter else ""))
        root = ET.fromstring(fbv.http_get(url))
        keys += [c.find(f"{NS}Key").text for c in root.findall(f"{NS}Contents")]
        prefixes += [p.find(f"{NS}Prefix").text for p in root.findall(f"{NS}CommonPrefixes")]
        if root.find(f"{NS}IsTruncated").text != "true":
            break
        nxt = root.find(f"{NS}NextMarker")
        marker = nxt.text if nxt is not None else (keys[-1] if keys else prefixes[-1])
    return keys, prefixes


def usdt_perp_symbols() -> list[str]:
    _, prefixes = s3_list("data/futures/um/monthly/klines/", "/")
    syms = [p.rstrip("/").rsplit("/", 1)[1] for p in prefixes]
    return sorted(s for s in syms if s.endswith("USDT") and "_" not in s)


def parse_daily_kline(raw: bytes) -> pl.DataFrame:
    first = raw.split(b"\n", 1)[0]
    has_header = first[:1].isalpha()
    df = pl.read_csv(io.BytesIO(raw), has_header=has_header, infer_schema=False,
                     new_columns=None if has_header else fbv.KLINE_CSV_COLS)
    if not has_header:
        df = df.rename(dict(zip(df.columns, fbv.KLINE_CSV_COLS)))
    ts = df["open_time"].cast(pl.Int64)
    ts = pl.when(ts > 10**14).then(ts // 1000).otherwise(ts)
    return df.select(ts.alias("timestamp"), pl.col("close").cast(pl.Float64),
                     pl.col("quote_volume").cast(pl.Float64), pl.col("high").cast(pl.Float64))


def months_ok(key: str, start: str) -> bool:
    ym = key.rsplit("-", 2)
    try:
        return f"{ym[-2]}-{ym[-1][:2]}" >= start
    except Exception:
        return True


def symbol_files(sym: str, start: str) -> tuple[list[str], list[str]]:
    kkeys, _ = s3_list(f"data/futures/um/monthly/klines/{sym}/1d/")
    fkeys, _ = s3_list(f"data/futures/um/monthly/fundingRate/{sym}/")
    pick = lambda keys: [k for k in keys if k.endswith(".zip") and months_ok(k[:-4], start)]
    return pick(kkeys), pick(fkeys)


def get_file(key: str) -> bytes | None:
    # non-ASCII symbol names exist in the archive: the URL path must be percent-encoded
    return fbv.fetch_verified(f"https://data.binance.vision/{quote(key)}", os.path.join(RAW, key))


def build_symbol(sym: str, kzips: list[str], fzips: list[str], blobs: dict) -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {"symbol": sym}
    frames = [parse_daily_kline(fbv._read_zip_csv(blobs[k])) for k in kzips if blobs.get(k) is not None]
    if frames:
        df = pl.concat(frames).unique(subset=["timestamp"], keep="last").sort("timestamp")
        df.write_parquet(os.path.join(OUT_DIR, f"{sym}_1d.parquet"))
        out["days"] = df.height
    ff = [fbv.parse_funding_csv(fbv._read_zip_csv(blobs[k])) for k in fzips if blobs.get(k) is not None]
    if ff:
        f = pl.concat(ff).unique(subset=["timestamp"], keep="last").sort("timestamp")
        f.write_parquet(os.path.join(OUT_DIR, f"{sym}_funding.parquet"))
        out["funding_rows"] = f.height
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2021-01")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    syms = usdt_perp_symbols()
    print(f"{len(syms)} USDT perpetual symbols in the archive", flush=True)
    t0 = time.time()
    # file-level parallelism: list everything first, then download all files concurrently
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        listing = dict(zip(syms, ex.map(lambda s: symbol_files(s, args.start), syms)))
    keys = [k for kz, fz in listing.values() for k in kz + fz]
    print(f"listed {len(keys)} files in {time.time() - t0:.0f}s", flush=True)
    blobs = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (k, b) in enumerate(zip(keys, ex.map(get_file, keys)), 1):
            blobs[k] = b
            if i % 2000 == 0:
                print(f"  {i}/{len(keys)} files ({time.time() - t0:.0f}s)", flush=True)
    n = 0
    for sym, (kz, fz) in listing.items():
        rep = build_symbol(sym, kz, fz, blobs)
        n += "days" in rep
    print(f"done: {n} symbols with daily klines ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
