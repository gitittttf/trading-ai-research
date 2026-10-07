"""
Market data loading, validation and resampling.

On-disk schema (written by scripts/fetch_binance_vision.py):
  data/<BASE>_USDT_klines.parquet  : timestamp(ms, bar open), open, high, low, close,
                                     volume, taker_buy_volume  (1-minute bars)
  data/<BASE>_USDT_funding.parquet : timestamp(ms), funding_rate
All timestamps are UTC epoch milliseconds of the bar OPEN.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import polars as pl

KLINE_COLS = ["timestamp", "open", "high", "low", "close", "volume", "taker_buy_volume"]
MINUTE_MS = 60_000


def symbol_to_file_stem(symbol: str) -> str:
    """'DOT/USDT' | 'DOTUSDT' | 'DOT_USDT' -> 'DOT_USDT'."""
    s = symbol.replace("/", "_").replace(":USDT", "")
    if "_" not in s and s.endswith("USDT"):
        s = s[:-4] + "_USDT"
    return s


def klines_path(symbol: str, data_dir: str) -> str:
    return os.path.join(data_dir, f"{symbol_to_file_stem(symbol)}_klines.parquet")


def funding_path(symbol: str, data_dir: str) -> str:
    return os.path.join(data_dir, f"{symbol_to_file_stem(symbol)}_funding.parquet")


def load_klines(symbol: str, data_dir: str, drop_untraded: bool = True) -> pl.DataFrame | None:
    """Load 1m bars. ``drop_untraded`` removes minutes with zero volume: nothing could be
    traded there, and after a delisting the archive keeps emitting flat zero-volume
    "zombie" candles (FTM: constant 0.7702 from 2025-01-06 on) that would otherwise look
    like a perfectly calm market."""
    path = klines_path(symbol, data_dir)
    if not os.path.exists(path):
        return None
    df = pl.read_parquet(path)
    missing = [c for c in KLINE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    df = (df.select(KLINE_COLS)
            .with_columns(pl.col("timestamp").cast(pl.Int64),
                          *[pl.col(c).cast(pl.Float64) for c in KLINE_COLS[1:]])
            .unique(subset=["timestamp"], keep="last")
            .sort("timestamp"))
    if drop_untraded:
        df = df.filter(pl.col("volume") > 0)
    return df if df.height else None


def load_funding(symbol: str, data_dir: str) -> pl.DataFrame | None:
    path = funding_path(symbol, data_dir)
    if not os.path.exists(path):
        return None
    df = pl.read_parquet(path)
    return (df.select(["timestamp", "funding_rate"])
              .with_columns(pl.col("timestamp").cast(pl.Int64), pl.col("funding_rate").cast(pl.Float64))
              .unique(subset=["timestamp"], keep="last")
              .sort("timestamp"))


@dataclass
class ValidationReport:
    rows: int
    duplicates: int
    unsorted: bool
    bad_ohlc: int
    nonpositive_price: int
    gaps: int
    largest_gap_minutes: float

    @property
    def ok(self) -> bool:
        return self.duplicates == 0 and not self.unsorted and self.bad_ohlc == 0 and self.nonpositive_price == 0


def validate_klines(df: pl.DataFrame, bar_minutes: int = 1) -> ValidationReport:
    ts = df["timestamp"].to_numpy()
    diffs = np.diff(ts) if len(ts) > 1 else np.array([], dtype=np.int64)
    step = bar_minutes * MINUTE_MS
    bad_ohlc = df.filter(
        (pl.col("high") < pl.max_horizontal("open", "close")) |
        (pl.col("low") > pl.min_horizontal("open", "close"))
    ).height
    nonpos = df.filter((pl.col("open") <= 0) | (pl.col("high") <= 0) |
                       (pl.col("low") <= 0) | (pl.col("close") <= 0)).height
    gaps = int(np.sum(diffs > step)) if len(diffs) else 0
    largest = float(diffs.max() / MINUTE_MS) if len(diffs) else 0.0
    return ValidationReport(
        rows=df.height,
        duplicates=df.height - df["timestamp"].n_unique(),
        unsorted=bool(np.any(diffs < 0)) if len(diffs) else False,
        bad_ohlc=bad_ohlc,
        nonpositive_price=nonpos,
        gaps=gaps,
        largest_gap_minutes=largest,
    )


def resample_klines(df: pl.DataFrame, bar_minutes: int) -> pl.DataFrame:
    """Aggregate 1-minute bars to ``bar_minutes`` bars (left-labelled, left-closed).

    A bar is labelled with its OPEN time and only contains minutes inside it, so a
    decision taken on a bar's close never sees a later minute.
    """
    if bar_minutes == 1:
        return df.with_columns(pl.lit(1, dtype=pl.UInt32).alias("n_minutes"))
    step = bar_minutes * MINUTE_MS
    return (df.with_columns(((pl.col("timestamp") // step) * step).alias("bucket"))
              .group_by("bucket", maintain_order=True)
              .agg(pl.col("open").first(), pl.col("high").max(), pl.col("low").min(),
                   pl.col("close").last(), pl.col("volume").sum(),
                   pl.col("taker_buy_volume").sum(), pl.len().alias("n_minutes"))
              .rename({"bucket": "timestamp"})
              .sort("timestamp"))


def align_pair(df1: pl.DataFrame, df2: pl.DataFrame) -> pl.DataFrame:
    """Inner-join two bar frames on timestamp with suffixes 1/2."""
    a = df1.rename({c: f"{c}1" for c in df1.columns if c != "timestamp"})
    b = df2.rename({c: f"{c}2" for c in df2.columns if c != "timestamp"})
    return a.join(b, on="timestamp", how="inner").sort("timestamp")


def time_slice(df: pl.DataFrame, start_ms: int, end_ms: int) -> pl.DataFrame:
    """Rows with start_ms <= timestamp < end_ms (half-open, so windows never overlap)."""
    return df.filter((pl.col("timestamp") >= start_ms) & (pl.col("timestamp") < end_ms))
