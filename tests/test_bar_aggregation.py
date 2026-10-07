import math

from live_trading.candles import BarAggregator, ClosedCandleTracker

M = 60_000


def c(ts, close, vol=1.0, tb=0.5):
    return {"timestamp": ts, "open": close, "high": close + 1, "low": close - 1, "close": close,
            "volume": vol, "taker_buy_volume": tb}


def test_partial_updates_yield_exactly_one_closed_candle_per_minute():
    tr = ClosedCandleTracker()
    out = []
    # 3 updates of minute 0, 2 of minute 1, then minute 2 starts
    for ts, px in [(0, 10), (0, 11), (0, 12), (M, 13), (M, 14), (2 * M, 15)]:
        out += tr.on_update("X", c(ts, px))
    assert [o["timestamp"] for o in out] == [0, M]
    assert [o["close"] for o in out] == [12, 14]        # the LAST update of each minute
    # a stale update for an already-closed minute changes nothing
    assert tr.on_update("X", c(M, 99)) == []
    assert tr.on_update("X", c(3 * M, 16))[0]["close"] == 15


def test_symbols_are_independent():
    tr = ClosedCandleTracker()
    tr.on_update("A", c(0, 1))
    assert tr.on_update("B", c(M, 1)) == []
    assert tr.on_update("A", c(M, 2))[0]["timestamp"] == 0


def test_bar_aggregator_emits_once_when_bucket_complete():
    agg = BarAggregator(5)
    bars = []
    for i in range(10):
        b = agg.add("X", c(i * M, 10 + i, vol=2.0, tb=1.0))
        if b:
            bars.append(b)
    assert [b["timestamp"] for b in bars] == [0, 5 * M]
    assert bars[0]["open"] == 10 and bars[0]["close"] == 14 and bars[0]["volume"] == 10.0
    assert bars[0]["high"] == 15 and bars[0]["low"] == 9 and bars[0]["n_minutes"] == 5
    # duplicate of the closing minute does not emit again
    assert agg.add("X", c(9 * M, 99)) is None


def test_bar_aggregator_gap_and_missing_taker_volume():
    agg = BarAggregator(5)
    agg.add("X", c(0, 1))
    agg.add("X", {**c(M, 2), "taker_buy_volume": None})
    b = agg.add("X", c(4 * M, 3))            # minutes 2,3 missing
    assert b["n_minutes"] == 3 and b["close"] == 3
    assert math.isnan(b["taker_buy_volume"])
