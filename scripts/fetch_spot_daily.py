"""
Download DAILY spot klines (close, quote volume) from data.binance.vision for every
USDT perpetual in data_daily/ that also trades as a spot pair with the same name.

Used by protocol variant 12 (delta-neutral funding harvest: short perp, long spot).
Every file is verified against its published SHA-256.

  uv run python scripts/fetch_spot_daily.py --start 2021-01
Writes data_daily/spot/<SYMBOL>_1d.parquet (timestamp, close, quote_volume, high).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import polars as pl

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "scripts"))

import fetch_binance_vision as fbv            # noqa: E402
import fetch_daily_universe as fdu           # noqa: E402

PERP_DIR = os.path.join(ROOT_DIR, "data_daily")
OUT_DIR = os.path.join(PERP_DIR, "spot")


def perp_symbols() -> list[str]:
    syms = [os.path.basename(p)[:-len("_1d.parquet")] for p in glob.glob(os.path.join(PERP_DIR, "*_1d.parquet"))]
    # 1000-multiplier contracts have no same-named spot pair (and a different price scale)
    return sorted(s for s in syms if not s.startswith("1000") and s.isascii())


def spot_files(sym: str, start: str) -> list[str]:
    keys, _ = fdu.s3_list(f"data/spot/monthly/klines/{sym}/1d/")
    return [k for k in keys if k.endswith(".zip") and fdu.months_ok(k[:-4], start)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2021-01")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    syms = perp_symbols()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        listing = dict(zip(syms, ex.map(lambda s: spot_files(s, args.start), syms)))
    keys = [k for v in listing.values() for k in v]
    print(f"{len(syms)} perp symbols, {sum(1 for v in listing.values() if v)} with spot klines, "
          f"{len(keys)} files ({time.time() - t0:.0f}s)", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        blobs = dict(zip(keys, ex.map(fdu.get_file, keys)))
    n = 0
    for sym, ks in listing.items():
        frames = [fdu.parse_daily_kline(fbv._read_zip_csv(blobs[k])) for k in ks if blobs.get(k) is not None]
        if not frames:
            continue
        df = pl.concat(frames).unique(subset=["timestamp"], keep="last").sort("timestamp")
        df.write_parquet(os.path.join(OUT_DIR, f"{sym}_1d.parquet"))
        n += 1
    print(f"done: {n} spot symbols ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
