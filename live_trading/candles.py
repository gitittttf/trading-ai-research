"""
Pure candle bookkeeping for live trading (no I/O, fully unit-tested).

* ``ClosedCandleTracker``: websocket kline streams push every update of the
  CURRENT (still open) candle. A candle is final only once a candle with a later
  open time appears. The tracker turns the update stream into exactly one closed
  candle per minute (appending every partial update as a new row would make a
  "24h" window cover only a few hours).
* ``BarAggregator``: collects closed 1m candles per symbol and emits completed
  N-minute bars, once, when the last minute of the bucket has arrived.
"""
from __future__ import annotations

MINUTE_MS = 60_000
CANDLE_KEYS = ["timestamp", "open", "high", "low", "close", "volume", "taker_buy_volume"]


class ClosedCandleTracker:
    def __init__(self):
        self.current: dict[str, dict] = {}       # symbol -> latest update of the open candle
        self.last_emitted: dict[str, int] = {}   # symbol -> open time of last closed candle

    def on_update(self, symbol: str, candle: dict) -> list[dict]:
        """Feed one websocket update; returns the candles that became final (0 or 1)."""
        ts = int(candle["timestamp"])
        cur = self.current.get(symbol)
        out = []
        if cur is not None and ts > int(cur["timestamp"]):
            if int(cur["timestamp"]) > self.last_emitted.get(symbol, -1):
                out.append(cur)
                self.last_emitted[symbol] = int(cur["timestamp"])
        if cur is None or ts >= int(cur["timestamp"]):
            self.current[symbol] = dict(candle)
        return out


class BarAggregator:
    def __init__(self, bar_minutes: int):
        self.bar_ms = bar_minutes * MINUTE_MS
        self.buckets: dict[str, dict[int, dict[int, dict]]] = {}   # symbol -> bucket -> minute -> candle
        self.emitted: dict[str, int] = {}

    def add(self, symbol: str, candle: dict) -> dict | None:
        """Add a CLOSED 1m candle; return the completed bar if this minute closes a bucket."""
        ts = int(candle["timestamp"])
        bucket = (ts // self.bar_ms) * self.bar_ms
        if bucket <= self.emitted.get(symbol, -1):
            return None                                   # late duplicate of an emitted bar
        self.buckets.setdefault(symbol, {}).setdefault(bucket, {})[ts] = candle
        if ts + MINUTE_MS < bucket + self.bar_ms:
            return None                                   # not the last minute of the bucket yet
        minutes = [self.buckets[symbol][bucket][k] for k in sorted(self.buckets[symbol][bucket])]
        del self.buckets[symbol][bucket]
        for old in [b for b in self.buckets[symbol] if b < bucket]:
            del self.buckets[symbol][old]                 # incomplete older buckets are dropped
        self.emitted[symbol] = bucket
        tb = [m.get("taker_buy_volume") for m in minutes]
        return {
            "timestamp": bucket,
            "open": float(minutes[0]["open"]),
            "high": max(float(m["high"]) for m in minutes),
            "low": min(float(m["low"]) for m in minutes),
            "close": float(minutes[-1]["close"]),
            "volume": sum(float(m["volume"]) for m in minutes),
            "taker_buy_volume": (sum(float(x) for x in tb) if all(x is not None for x in tb) else float("nan")),
            "n_minutes": len(minutes),
        }
