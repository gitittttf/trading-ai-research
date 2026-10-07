"""
Wiring test for the live core without network: closed 1m candles go through the
BarAggregator into the streaming engines and must yield the same decisions as
feeding the streaming engine directly with the resampled bars.
"""
import asyncio

import polars as pl
import pytest

from core.data import resample_klines
from core.synthetic import cointegrated_market
from live_trading.bundle import load_bundle


def test_core_routes_candles_to_pairs_exactly_once(tmp_path, monkeypatch):
    from scripts_helper import build_baseline_bundle
    import live_trading.shared.state as state_mod
    import live_trading.trading_core as tc
    out = str(tmp_path / "bundle")
    build_baseline_bundle(out)
    monkeypatch.setattr(state_mod, "DB_PATH", str(tmp_path / "state.db"))
    monkeypatch.setattr(state_mod.StateManager.__init__, "__defaults__", (str(tmp_path / "state.db"),))
    core = tc.TradingCore(load_bundle(out))
    ref = load_bundle(out)              # independent engines for the reference run
    name = next(iter(core.bundle.pairs))
    a1, a2 = core.bundle.pair_assets[name]
    bars = cointegrated_market(n_pairs=2, n_minutes=40 * 1440, seed=3)
    seen = []

    async def fake_execute(n, d):
        seen.append((n, d["action"], d["ts"]))
    monkeypatch.setattr(core, "execute", fake_execute)

    minutes = 3 * 1440
    async def feed():
        d1 = bars[a1].tail(minutes).to_dicts()
        d2 = bars[a2].tail(minutes).to_dicts()
        for c1, c2 in zip(d1, d2):
            await core.on_candle(a1, c1)
            await core.on_candle(a2, c2)
            await core.on_candle(a2, c2)      # duplicate delivery must not create a second bar
    asyncio.run(feed())

    r1 = resample_klines(bars[a1].tail(minutes), 5)
    r2 = resample_klines(bars[a2].tail(minutes), 5)
    sp = ref.pairs[name]
    expected = []
    for b1, b2 in zip(r1.iter_rows(named=True), r2.iter_rows(named=True)):
        bar = {"timestamp": b1["timestamp"]}
        for k in ["open", "high", "low", "close", "volume", "taker_buy_volume"]:
            bar[k + "1"], bar[k + "2"] = b1[k], b2[k]
        d = sp.on_bar(bar)
        if d:
            expected.append((name, d["action"], d["ts"]))
    got = [x for x in seen if x[0] == name]
    assert len(expected) >= 2, "test needs at least one entry and one exit"
    assert got == expected
    assert sp.rows and len(core.bundle.pairs[name].rows) == len(sp.rows)
    core.state.close()
