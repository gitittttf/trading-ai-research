"""
Download Binance USD-M futures history from data.binance.vision with checksums.

Why this source: it is Binance's own public archive (monthly + daily ZIP files),
every file ships a .CHECKSUM with its SHA-256, and bulk history downloads in
minutes instead of hours of paginated API calls.

Writes (same schema the research code reads):
  data/<BASE>_USDT_klines.parquet   timestamp, open, high, low, close, volume, taker_buy_volume
  data/<BASE>_USDT_funding.parquet  timestamp, funding_rate
Raw ZIPs are cached under data/raw/binance_vision/ (git-ignored).

Usage:
  uv run python scripts/fetch_binance_vision.py                   # whole UNIVERSE from 2024-01
  uv run python scripts/fetch_binance_vision.py --symbols BTC/USDT DOT/USDT --start 2025-01
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import http.client
import io
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import polars as pl

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from core.constants import DATA_DIR, UNIVERSE                 # noqa: E402
from core.data import KLINE_COLS, symbol_to_file_stem, validate_klines  # noqa: E402

BASE_URL = "https://data.binance.vision/data/futures/um"
RAW_DIR = os.path.join(DATA_DIR, "raw", "binance_vision")
KLINE_CSV_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                  "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


class ChecksumError(RuntimeError):
    pass


def exchange_symbol(symbol: str) -> str:
    """'1000PEPE/USDT' -> '1000PEPEUSDT'."""
    return symbol.replace("/", "").replace("_", "").replace(":USDT", "")


def http_get(url: str, timeout: float = 60.0, retries: int = 4) -> bytes | None:
    """GET with retries. Returns None on 404 (file does not exist, e.g. before listing)."""
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "trading-ai-research/0.2"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, OSError):
            # includes IncompleteRead: a transfer cut off mid-way is retried (the checksum
            # check would reject a truncated file anyway)
            if attempt == retries:
                raise
        time.sleep(delay)
        delay *= 2
    return None


def verify_checksum(blob: bytes, checksum_text: str, filename: str) -> None:
    expected = checksum_text.strip().split()[0].lower()
    actual = hashlib.sha256(blob).hexdigest()
    if expected != actual:
        raise ChecksumError(f"{filename}: sha256 {actual} != expected {expected}")


def fetch_verified(url: str, cache_path: str, fetch=http_get) -> bytes | None:
    """Download url (+ .CHECKSUM), verify SHA-256, cache to disk. None if missing."""
    if os.path.exists(cache_path) and os.path.exists(cache_path + ".CHECKSUM"):
        blob = open(cache_path, "rb").read()
        verify_checksum(blob, open(cache_path + ".CHECKSUM", encoding="utf-8").read(), os.path.basename(cache_path))
        return blob
    blob = fetch(url)
    if blob is None:
        return None
    chk = fetch(url + ".CHECKSUM")
    if chk is None:
        raise ChecksumError(f"no checksum published for {url}")
    chk_text = chk.decode()
    verify_checksum(blob, chk_text, os.path.basename(url))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        f.write(blob)
    # the checksum text contains the file name, which is non-ASCII for some symbols
    with open(cache_path + ".CHECKSUM", "w", encoding="utf-8") as f:
        f.write(chk_text)
    return blob


def _read_zip_csv(blob: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected one csv in zip, got {names}")
        return z.read(names[0])


def parse_klines_csv(raw: bytes) -> pl.DataFrame:
    """Binance kline CSV (with or without header) -> our 1m schema."""
    first = raw.split(b"\n", 1)[0]
    has_header = first[:1].isalpha() or first.startswith(b"open_time")
    # read everything as text and cast explicitly: type inference fails when the first
    # rows of a file happen to contain integer-looking volumes (e.g. "6701" then "6701.80")
    df = pl.read_csv(io.BytesIO(raw), has_header=has_header, infer_schema=False,
                     new_columns=None if has_header else KLINE_CSV_COLS)
    if not has_header:
        df = df.rename(dict(zip(df.columns, KLINE_CSV_COLS)))
    ts = df["open_time"].cast(pl.Int64)
    # some archives use microseconds; normalize to milliseconds
    ts = pl.when(ts > 10**14).then(ts // 1000).otherwise(ts)
    return df.select(ts.alias("timestamp"),
                     *[pl.col(c).cast(pl.Float64) for c in ["open", "high", "low", "close", "volume",
                                                           "taker_buy_volume"]]).select(KLINE_COLS)


def parse_funding_csv(raw: bytes) -> pl.DataFrame:
    """fundingRate CSV: calc_time, funding_interval_hours, last_funding_rate."""
    df = pl.read_csv(io.BytesIO(raw), has_header=True, infer_schema=False)
    tcol = "calc_time" if "calc_time" in df.columns else df.columns[0]
    rcol = "last_funding_rate" if "last_funding_rate" in df.columns else df.columns[-1]
    ts = df[tcol].cast(pl.Int64)
    return pl.DataFrame({"timestamp": ts, "funding_rate": df[rcol].cast(pl.Float64)})


def month_range(start: str, end_month: dt.date) -> list[tuple[int, int]]:
    y, m = map(int, start.split("-")[:2])
    out = []
    while (y, m) <= (end_month.year, end_month.month):
        out.append((y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def daily_kline_urls(sym: str, first_day: dt.date, end_excl: dt.date) -> list[tuple[str, str]]:
    out = []
    d = first_day
    while d < end_excl:
        name = f"{sym}-1m-{d.isoformat()}.zip"
        out.append((f"{BASE_URL}/daily/klines/{sym}/1m/{name}", os.path.join(RAW_DIR, "klines", sym, name)))
        d += dt.timedelta(days=1)
    return out


def monthly_kline_urls(sym: str, start: str, today: dt.date) -> list[tuple[tuple[int, int], str, str]]:
    last_month = today.replace(day=1) - dt.timedelta(days=1)
    out = []
    for y, m in month_range(start, last_month):
        name = f"{sym}-1m-{y:04d}-{m:02d}.zip"
        out.append(((y, m), f"{BASE_URL}/monthly/klines/{sym}/1m/{name}", os.path.join(RAW_DIR, "klines", sym, name)))
    return out


def klines_urls(sym: str, start: str, today: dt.date) -> list[tuple[str, str]]:
    """(url, cache_path) for monthly files up to last month + daily files of this month."""
    out = [(u, p) for _, u, p in monthly_kline_urls(sym, start, today)]
    return out + daily_kline_urls(sym, today.replace(day=1), today)


def funding_urls(sym: str, start: str, today: dt.date) -> list[tuple[str, str]]:
    last_month = today.replace(day=1) - dt.timedelta(days=1)
    out = []
    for y, m in month_range(start, last_month):
        name = f"{sym}-fundingRate-{y:04d}-{m:02d}.zip"
        out.append((f"{BASE_URL}/monthly/fundingRate/{sym}/{name}", os.path.join(RAW_DIR, "funding", sym, name)))
    return out


def build_symbol(symbol: str, start: str, today: dt.date, data_dir: str, workers: int, fetch=http_get,
                 max_gap_days: float = 3.0) -> dict:
    sym = exchange_symbol(symbol)
    stem = symbol_to_file_stem(symbol)
    report = {"symbol": symbol}

    def get(item):
        url, path = item
        return fetch_verified(url, path, fetch)

    monthly = monthly_kline_urls(sym, start, today)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        mblobs = list(ex.map(get, [(u, p) for _, u, p in monthly]))
        # a month that is missing although the coin already traded before it (e.g. the monthly
        # archive is not published yet early in the month) is filled from daily files, so no
        # month-long hole slips into the data
        have = [i for i, b in enumerate(mblobs) if b is not None]
        fallback = []
        if have:
            for i in range(have[0], len(monthly)):
                if mblobs[i] is None:
                    (y, m), _, _ = monthly[i]
                    first = dt.date(y, m, 1)
                    nxt = dt.date(y + (m == 12), m % 12 + 1, 1)
                    fallback += daily_kline_urls(sym, first, nxt)
        fallback += daily_kline_urls(sym, today.replace(day=1), today)
        dblobs = list(ex.map(get, fallback))
    blobs = mblobs + dblobs
    frames = [parse_klines_csv(_read_zip_csv(b)) for b in blobs if b is not None]
    report["kline_files"] = len(frames)
    report["daily_fallback_files"] = sum(b is not None for b in dblobs)
    if frames:
        df = pl.concat(frames).unique(subset=["timestamp"], keep="last").sort("timestamp")
        rep = validate_klines(df)
        report["kline_rows"] = df.height
        report["gaps"] = rep.gaps
        report["largest_gap_minutes"] = rep.largest_gap_minutes
        if not rep.ok:
            raise SystemExit(f"{symbol}: validation failed {rep}")
        if rep.largest_gap_minutes > max_gap_days * 1440:
            raise SystemExit(f"{symbol}: data gap of {rep.largest_gap_minutes / 1440:.1f} days "
                             f"(> {max_gap_days}); refusing to write a series with a hole")
        df.write_parquet(os.path.join(data_dir, f"{stem}_klines.parquet"))
        report["first"] = dt.datetime.fromtimestamp(df["timestamp"][0] / 1000, dt.UTC).isoformat()
        report["last"] = dt.datetime.fromtimestamp(df["timestamp"][-1] / 1000, dt.UTC).isoformat()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        fblobs = list(ex.map(get, funding_urls(sym, start, today)))
    ff = [parse_funding_csv(_read_zip_csv(b)) for b in fblobs if b is not None]
    report["funding_files"] = len(ff)
    if ff:
        fdf = pl.concat(ff).unique(subset=["timestamp"], keep="last").sort("timestamp")
        fdf.write_parquet(os.path.join(data_dir, f"{stem}_funding.parquet"))
        report["funding_rows"] = fdf.height
        # the archive only has MONTHLY funding files: the current month is not covered.
        # walk_forward.py caps the evaluated period at the funding coverage end.
        report["funding_last"] = dt.datetime.fromtimestamp(fdf["timestamp"][-1] / 1000, dt.UTC).isoformat()
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="*", default=UNIVERSE)
    ap.add_argument("--start", default="2024-01", help="first month YYYY-MM")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-gap-days", type=float, default=3.0)
    args = ap.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)
    today = dt.datetime.now(dt.UTC).date()
    for s in args.symbols:
        t0 = time.time()
        try:
            rep = build_symbol(s, args.start, today, args.data_dir, args.workers, max_gap_days=args.max_gap_days)
        except ChecksumError as e:
            print(f"CHECKSUM FAILURE {s}: {e}", flush=True)
            raise
        print(f"{s:14s} {rep} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
