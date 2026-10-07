"""
Live paper-trading core.

Data flow:  ingestion daemon --(closed 1m futures candles, ZMQ)--> BarAggregator
            --(completed N-minute bars, once per pair)--> core.streaming.StreamingPair
            --(decision on bar close)--> PaperPortfolio (order-book fills) --> SQLite

The strategy code is the SAME as in the backtest (core.streaming re-uses the
backtest's entry/exit rules and features; tests/test_live_parity.py proves the
decisions match). Safeguards: only closed futures candles are used, exactly once;
params come only from a checksum-verified bundle, never edited by hand; order
flow uses the real taker volume; the history is warmed up via REST
(``history_bars``); if the order book cannot be read, the close is retried and
alerted and the position stays open (no silent close at the entry price); fees
come from core.costs and funding from the exchange history; all paths are
absolute.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time

import ccxt.pro as ccp
import numpy as np
import polars as pl
import uvicorn
import zmq
import zmq.asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.backtest import PortfolioConfig                           # noqa: E402
from core.costs import CostModel, funding_payment                   # noqa: E402
from core.data import resample_klines                               # noqa: E402
from live_trading.bundle import BundleError, LiveBundle, load_bundle  # noqa: E402
from live_trading.candles import BarAggregator                      # noqa: E402
from live_trading.paper_broker import PaperPortfolio, walk_book     # noqa: E402
from live_trading.shared.config import (API_HOST, API_PORT, BTC_SYMBOL, BUNDLE_DIR,  # noqa: E402
                                        ZMQ_OB_PORT, ZMQ_TICK_PORT)
from live_trading.shared.state import StateManager                  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("TradingCore")
MINUTE_MS = 60_000


class TradingCore:
    def __init__(self, bundle: LiveBundle):
        self.bundle = bundle
        m = bundle.manifest
        self.bar_minutes = m["streaming_config"]["bar_minutes"]
        self.use_btc = bool(m["streaming_config"].get("use_btc"))
        self.costs = CostModel(**m["costs"])
        self.state = StateManager()
        st = self.state._load_state()
        self.portfolio = PaperPortfolio(PortfolioConfig(**m["portfolio"]), self.costs,
                                        float(st.get("current_equity", m["portfolio"]["start_equity"])))
        self.start_equity = float(m["portfolio"]["start_equity"])
        self.symbols = sorted({s for a in bundle.pair_assets.values() for s in a} | ({BTC_SYMBOL} if self.use_btc else set()))
        self.agg = BarAggregator(self.bar_minutes)
        self.bars: dict[str, dict[int, dict]] = {s: {} for s in self.symbols}
        self.db_ids: dict[str, int] = {}
        self.alerts: list[str] = []
        self.running = False
        self.context = zmq.asyncio.Context.instance()
        self.sub = self.context.socket(zmq.SUB)
        self.sub.connect(f"tcp://127.0.0.1:{ZMQ_TICK_PORT}")
        self.sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.req = self.context.socket(zmq.REQ)
        self.req.setsockopt(zmq.RCVTIMEO, 8000)
        self.req.setsockopt(zmq.REQ_RELAXED, 1)
        self.req.setsockopt(zmq.REQ_CORRELATE, 1)
        self.req.connect(f"tcp://127.0.0.1:{ZMQ_OB_PORT}")
        self.ob_lock = asyncio.Lock()
        self.exchange = ccp.binanceusdm({"enableRateLimit": True})
        self.app = FastAPI(title="LeadQuant Paper Trader v2")
        self.app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
        self._routes()

    # ------------------------------------------------------------------ API
    def snapshot(self) -> dict:
        positions = []
        for p in self.portfolio.positions.values():
            positions.append({"pair": p.pair, "trade_dir": p.side, "entry_ts": p.entry_ts, "entry_z": p.entry_z,
                              "entry_p1": p.entry_p1, "entry_p2": p.entry_p2, "allocated_margin": p.gross_notional,
                              "size_p1": p.units1, "size_p2": p.units2, "entry_std": 0.0,
                              "delta_z_stop": self.bundle.manifest["streaming_config"]["stop_z"],
                              "adjusted_alpha": p.extra.get("edge", 0.0), "uncertainty": p.extra.get("unc", 0.0),
                              "entry_adf": 0.0, "entry_volatility": 0.0, "z_momentum": 0.0, "meta_prob": 0.0})
        return {"equity": round(self.portfolio.equity, 2), "starting_equity": self.start_equity,
                "pnl": round(self.portfolio.equity - self.start_equity, 2),
                "breaker_penalty": 0.0, "msi_breakdown_rate": 0.0, "msi_stressed": False,
                "active_trades": len(positions), "open_positions": positions,
                "recent_trades": self.state.get_recent_trades(limit=10),
                "bundle": {"strategy": self.bundle.manifest["strategy"], "expired": self.bundle.expired,
                           "allows_entries": self.bundle.allows_entries,
                           "pairs": list(self.bundle.pairs)},
                "pair_z": {k: (None if not np.isfinite(sp.last_z) else round(float(sp.last_z), 3))
                           for k, sp in self.bundle.pairs.items()},
                "alerts": self.alerts[-20:], "timestamp": time.time()}

    def _routes(self):
        @self.app.get("/state")
        async def get_state():
            return self.snapshot()

        @self.app.websocket("/ws")
        async def ws(sock: WebSocket):
            await sock.accept()
            try:
                while self.running:
                    await sock.send_json(self.snapshot())
                    await asyncio.sleep(1.0)
            except (WebSocketDisconnect, Exception):
                pass

    def alert(self, msg: str):
        logger.error(msg)
        self.alerts.append(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}")

    # ------------------------------------------------------------- warm-up
    async def fetch_history(self, symbol: str, minutes: int) -> pl.DataFrame:
        market = self.exchange.market(symbol)
        end = int(time.time() * 1000) // MINUTE_MS * MINUTE_MS
        start = end - minutes * MINUTE_MS
        rows, t = [], start
        while t < end:
            k = await self.exchange.fapiPublicGetKlines({"symbol": market["id"], "interval": "1m",
                                                         "startTime": t, "limit": 1500})
            if not k:
                break
            rows += [{"timestamp": int(r[0]), "open": float(r[1]), "high": float(r[2]), "low": float(r[3]),
                      "close": float(r[4]), "volume": float(r[5]), "taker_buy_volume": float(r[9])} for r in k]
            t = int(k[-1][0]) + MINUTE_MS
        df = pl.DataFrame(rows).unique(subset=["timestamp"], keep="last").sort("timestamp")
        return df.filter(pl.col("timestamp") < end)   # the minute starting at `end` is still open

    async def warmup(self):
        await self.exchange.load_markets()
        hist_bars = max(sp.cfg.history_bars for sp in self.bundle.pairs.values())
        minutes = hist_bars * self.bar_minutes + 60
        bar_ms = self.bar_minutes * MINUTE_MS
        for s in self.symbols:
            m1 = await self.fetch_history(s, minutes)
            if m1.height == 0:
                self.alert(f"warm-up: no history for {s}")
                continue
            last_min = int(m1["timestamp"][-1])
            # buckets whose last minute is already closed are complete; the rest is handed to the
            # aggregator so the live candles finish that bar instead of being dropped
            open_bucket = ((last_min + MINUTE_MS) // bar_ms) * bar_ms if (last_min + MINUTE_MS) % bar_ms else None
            done = m1 if open_bucket is None else m1.filter(pl.col("timestamp") < open_bucket)
            df = resample_klines(done, self.bar_minutes)
            for r in df.iter_rows(named=True):
                self.bars[s][int(r["timestamp"])] = r
            self.agg.emitted[s] = int(df["timestamp"][-1]) if df.height else -1
            if open_bucket is not None:
                for c in m1.filter(pl.col("timestamp") >= open_bucket).iter_rows(named=True):
                    self.agg.add(s, c)
            logger.info(f"warm-up {s}: {df.height} complete bars")
        for name, sp in self.bundle.pairs.items():
            a1, a2 = self.bundle.pair_assets[name]
            for ts in sorted(set(self.bars[a1]) & set(self.bars[a2])):
                sp.on_bar(self._pair_bar(a1, a2, ts), btc_close=self._btc(ts), warmup_only=True)
        self._restore_positions()

    def _restore_positions(self):
        for pos in self.state.get_open_positions():
            extra = json.loads(pos.get("extra_json") or "{}")
            name = pos["pair"]
            if name not in self.bundle.pairs or "units1" not in extra:
                self.alert(f"open position {name} is not managed by the current bundle - close it manually")
                continue
            p = self.portfolio.open(name, int(pos["trade_dir"]), extra["beta"], int(pos["entry_ts"]),
                                    float(pos["entry_z"]), float(pos["allocated_margin"]),
                                    float(pos["entry_p1"]), float(pos["entry_p2"]), extra)
            p.units1, p.units2, p.entry_fees = extra["units1"], extra["units2"], extra["entry_fees"]
            sp = self.bundle.pairs[name]
            bar_ms = self.bar_minutes * MINUTE_MS
            # entry_ts = end of the decision bar; held = bars processed after the decision bar
            held = 0 if sp.last_ts is None else max(0, int((sp.last_ts - p.entry_ts) // bar_ms) + 1)
            sp.restore_position(p.side, p.entry_z, held, (p.entry_p1, p.entry_p2))
            self.db_ids[name] = pos["id"]
            logger.info(f"restored {name} side={p.side} held={held} bars")

    # ------------------------------------------------------------- bars
    def _pair_bar(self, a1: str, a2: str, ts: int) -> dict:
        b1, b2 = self.bars[a1][ts], self.bars[a2][ts]
        out = {"timestamp": ts}
        for k in ["open", "high", "low", "close", "volume", "taker_buy_volume"]:
            out[k + "1"], out[k + "2"] = b1[k], b2[k]
        return out

    def _btc(self, ts: int):
        if not self.use_btc:
            return None
        b = self.bars.get(BTC_SYMBOL, {}).get(ts)
        return None if b is None else b["close"]

    async def complete_taker_volume(self, symbol: str, candle: dict) -> dict:
        if candle.get("taker_buy_volume") is not None:
            return candle
        for attempt in range(3):
            try:
                market = self.exchange.market(symbol)
                k = await self.exchange.fapiPublicGetKlines({"symbol": market["id"], "interval": "1m",
                                                             "startTime": int(candle["timestamp"]), "limit": 1})
                if k and int(k[0][0]) == int(candle["timestamp"]):
                    return dict(candle, volume=float(k[0][5]), taker_buy_volume=float(k[0][9]))
            except Exception:
                await asyncio.sleep(1 + attempt)
        self.alert(f"taker volume unknown for {symbol} {candle['timestamp']}: neutral estimate used")
        return dict(candle, taker_buy_volume=0.5 * float(candle["volume"]))

    async def on_candle(self, symbol: str, candle: dict):
        if symbol not in self.bars:
            return
        candle = await self.complete_taker_volume(symbol, candle)
        bar = self.agg.add(symbol, candle)
        if bar is None:
            return
        ts = bar["timestamp"]
        self.bars[symbol][ts] = bar
        for s in self.bars:          # keep memory bounded
            if len(self.bars[s]) > 5000:
                for old in sorted(self.bars[s])[:-4000]:
                    del self.bars[s][old]
        for name, (a1, a2) in self.bundle.pair_assets.items():
            if symbol not in (a1, a2, BTC_SYMBOL):
                continue
            sp = self.bundle.pairs[name]
            pending = sorted(t for t in self.bars[a1] if t in self.bars[a2] and (sp.last_ts is None or t > sp.last_ts))
            btc_bars = self.bars.get(BTC_SYMBOL, {})
            for i, t in enumerate(pending):
                if self.use_btc and t not in btc_bars:
                    newer_btc = bool(btc_bars) and max(btc_bars) > t
                    if not newer_btc and i == len(pending) - 1:
                        break          # wait (briefly) for BTC's bar; it is forward filled otherwise
                decision = sp.on_bar(self._pair_bar(a1, a2, t), btc_close=self._btc(t))
                if decision:
                    await self.execute(name, decision)

    # ------------------------------------------------------------- execution
    async def order_book_price(self, symbol: str, side: str, notional: float) -> float | None:
        for attempt in range(3):
            async with self.ob_lock:
                try:
                    await self.req.send_string(symbol)
                    ob = await self.req.recv_json()
                except Exception as e:
                    ob = {"error": str(e)}
            if "error" not in ob:
                px = walk_book(ob["asks"] if side == "buy" else ob["bids"], notional)
                if px is not None:
                    return px
                self.alert(f"order book too thin for {symbol} {side} {notional:.2f} USD")
                return None
            await asyncio.sleep(1.0 + attempt)
        self.alert(f"order book unavailable for {symbol} after 3 attempts")
        return None

    async def funding_cost(self, pos, a1: str, a2: str, exit_ts: int) -> float:
        total = 0.0
        for sym, units, px in [(a1, pos.units1, pos.entry_p1), (a2, pos.units2, pos.entry_p2)]:
            try:
                market = self.exchange.market(sym)
                rows = await self.exchange.fapiPublicGetFundingRate(
                    {"symbol": market["id"], "startTime": pos.entry_ts, "endTime": exit_ts, "limit": 1000})
                fts = np.array([int(r["fundingTime"]) for r in rows], dtype=np.int64)
                frs = np.array([float(r["fundingRate"]) for r in rows])
                total += funding_payment(int(np.sign(units)), abs(units) * px, fts, frs, pos.entry_ts, exit_ts)
            except Exception as e:
                self.alert(f"funding history unavailable for {sym}: {e} (funding not charged)")
        return total

    async def execute(self, name: str, d: dict):
        sp = self.bundle.pairs[name]
        a1, a2 = self.bundle.pair_assets[name]
        if d["action"] == "enter":
            gross = self.portfolio.size_for_new(name) if self.bundle.allows_entries else 0.0
            if gross <= 0:
                logger.info(f"{name}: entry signal skipped (limits/bundle), pair stays blocked like in the backtest")
                return
            n1, n2 = self.portfolio.leg_notionals(sp.spec.beta, gross)
            side = d["side"]
            p1 = await self.order_book_price(a1, "buy" if side == 1 else "sell", n1)
            p2 = await self.order_book_price(a2, "sell" if side == 1 else "buy", n2)
            if p1 is None or p2 is None:
                self.alert(f"{name}: entry NOT executed (no reliable fill price)")
                return
            fill_ts = d["ts"] + self.bar_minutes * MINUTE_MS
            pos = self.portfolio.open(name, side, sp.spec.beta, fill_ts, d["z"], gross, p1, p2,
                                      extra={**d.get("info", {}), "beta": sp.spec.beta})
            sp.set_fill(p1, p2)
            pos.extra.update(units1=pos.units1, units2=pos.units2, entry_fees=pos.entry_fees)
            self.state.add_open_position({"pair": name, "trade_dir": side, "entry_ts": fill_ts, "entry_p1": p1,
                                          "entry_p2": p2, "entry_z": d["z"], "allocated_margin": gross,
                                          "size_p1": pos.units1, "size_p2": pos.units2,
                                          "adjusted_alpha": d.get("info", {}).get("edge", 0.0),
                                          "uncertainty": d.get("info", {}).get("unc", 0.0), "extra": pos.extra})
            self.db_ids[name] = self.state.get_open_positions()[-1]["id"]
            logger.info(f">>> ENTRY {name} side={side} z={d['z']:.2f} gross={gross:.2f} @ {p1:.6g}/{p2:.6g}")
        else:
            pos = self.portfolio.positions.get(name)
            if pos is None:
                return                   # the entry was skipped: nothing to close
            p1 = await self.order_book_price(a1, "sell" if pos.units1 > 0 else "buy", abs(pos.units1) * pos.entry_p1)
            p2 = await self.order_book_price(a2, "sell" if pos.units2 > 0 else "buy", abs(pos.units2) * pos.entry_p2)
            if p1 is None or p2 is None:
                # never close silently at a made-up price: keep the position and retry on the next bar
                self.alert(f"{name}: EXIT NOT executed ({d['reason']}), position kept open, retry next bar")
                sp.restore_position(pos.side, pos.entry_z, d["held"], (pos.entry_p1, pos.entry_p2))
                return
            now = d["ts"] + self.bar_minutes * MINUTE_MS     # end of the decision bar = fill time
            fund = await self.funding_cost(pos, a1, a2, now)
            r = self.portfolio.close(name, p1, p2, fund)
            self.state.close_position(self.db_ids.pop(name), {
                "exit_ts": now, "exit_p1": p1, "exit_p2": p2, "net_pnl": r["net_pnl"],
                "trade_cost": r["fees"] + r["funding_cost"], "exit_reason": d["reason"],
                "balance_after": r["equity_after"], "extra": r})
            logger.info(f"### EXIT {name} {d['reason']} net={r['net_pnl']:.2f} equity={r['equity_after']:.2f}")

    # ------------------------------------------------------------- main loop
    async def run(self):
        self.running = True
        if self.bundle.expired:
            self.alert("live bundle expired: no new entries until scripts/export_live_bundle.py is re-run")
        await self.warmup()
        server = uvicorn.Server(uvicorn.Config(self.app, host=API_HOST, port=API_PORT, log_level="warning"))
        self.api_task = asyncio.create_task(server.serve())
        logger.info(f"trading {list(self.bundle.pairs)} on {self.bar_minutes}m bars")
        while self.running:
            try:
                topic, payload = await self.sub.recv_multipart()
                await self.on_candle(topic.decode(), json.loads(payload.decode()))
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.alert(f"main loop error: {e}")
                await asyncio.sleep(1)

    async def shutdown(self):
        if not self.running:
            return
        self.running = False
        self.state.update_equity(self.portfolio.equity)
        self.state.close()
        self.sub.close()
        self.req.close()
        self.context.term()
        await self.exchange.close()


if __name__ == "__main__":
    try:
        core = TradingCore(load_bundle(BUNDLE_DIR))
    except BundleError as e:
        raise SystemExit(f"refusing to start: {e}")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for s in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(s, lambda: asyncio.create_task(core.shutdown()))
    try:
        loop.run_until_complete(core.run())
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(core.shutdown())
        loop.close()
