"""
Download the 3-month US T-bill rate (FRED series DTB3, percent, discount basis) used by phase 7:
the hurdle every harvest variant is compared against, and the interest variant 16/17 earns on
idle capital. FRED publishes no checksum; the file's SHA-256 ends up in each run's manifest.

  uv run python scripts/fetch_rates.py
Writes data_daily/rates/DTB3.csv (observation_date, DTB3).
"""
from __future__ import annotations

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "scripts"))

import fetch_binance_vision as fbv   # noqa: E402

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"
OUT = os.path.join(ROOT_DIR, "data_daily", "rates", "DTB3.csv")


def main():
    blob = fbv.http_get(URL)
    if not blob or not blob.startswith(b"observation_date,DTB3"):
        raise SystemExit(f"unexpected response from {URL}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        f.write(blob)
    rows = blob.decode().strip().splitlines()
    print(f"{len(rows) - 1} rows, last {rows[-1]} -> {OUT}")


if __name__ == "__main__":
    main()
