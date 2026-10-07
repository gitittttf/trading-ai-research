"""
Ingestion daemon: Binance USD-M FUTURES 1m candles -> ZMQ, plus order-book snapshots.

What it does:
* uses ``ccxt.pro.binanceusdm`` (futures, the market the models were trained on),
  not the spot client ``ccxt.pro.binance``;
* publishes only CLOSED candles, exactly once per minute and symbol
  (live_trading/candles.ClosedCandleTracker);
* enriches each closed candle with the exchange's final values incl.
  taker-buy volume from the REST kline endpoint (websocket OHLCV has none);
* subscribes to the symbols of the live bundle, absolute paths.
"""
import asyncio
import json
import logging
import os
import signal
import sys

import ccxt.pro as ccp
import polars as pl
import zmq
import zmq.asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from live_trading.candles import ClosedCandleTracker                                        # noqa: E402
from live_trading.shared.config import DATA_DIR, ZMQ_OB_PORT, ZMQ_TICK_PORT, bundle_symbols  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ingestion")


class IngestionDaemon:
    def __init__(self, symbols: list[str]):
        if not symbols:
            raise SystemExit("no symbols: export a live bundle first (scripts/export_live_bundle.py)")
        self.context = zmq.asyncio.Context.instance()
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.setsockopt(zmq.LINGER, 0)
        self.pub_socket.bind(f"tcp://127.0.0.1:{ZMQ_TICK_PORT}")
        self.rep_socket = self.context.socket(zmq.REP)
        self.rep_socket.setsockopt(zmq.LINGER, 0)
        self.rep_socket.bind(f"tcp://127.0.0.1:{ZMQ_OB_PORT}")
        self.exchange = ccp.binanceusdm({'enableRateLimit': True})
        self.symbols = symbols
        self.tracker = ClosedCandleTracker()
        self.live_data = {s: [] for s in symbols}
        os.makedirs(DATA_DIR, exist_ok=True)
        self.running = False
        self.tasks = []

    async def finalize(self, symbol: str, candle: dict) -> dict:
        """Replace websocket values of a closed candle with the exchange's final kline."""
        try:
            market = self.exchange.market(symbol)
            rows = await self.exchange.fapiPublicGetKlines(
                {"symbol": market["id"], "interval": "1m", "startTime": int(candle["timestamp"]), "limit": 1})
            if rows and int(rows[0][0]) == int(candle["timestamp"]):
                k = rows[0]
                return {"timestamp": int(k[0]), "open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
                        "close": float(k[4]), "volume": float(k[5]), "taker_buy_volume": float(k[9]),
                        "source": "rest"}
        except Exception as e:
            logger.warning(f"REST finalize failed for {symbol} {candle['timestamp']}: {e}")
        return dict(candle, taker_buy_volume=None, source="ws")

    async def watch_ohlcv(self, symbol: str):
        logger.info(f"Starting OHLCV watcher for {symbol}")
        while self.running:
            try:
                candles = await self.exchange.watch_ohlcv(symbol, '1m')
                for raw in candles[-2:]:   # the newest update (and the one before, in case we skipped)
                    update = {"timestamp": int(raw[0]), "open": float(raw[1]), "high": float(raw[2]),
                              "low": float(raw[3]), "close": float(raw[4]), "volume": float(raw[5])}
                    for closed in self.tracker.on_update(symbol, update):
                        final = await self.finalize(symbol, closed)
                        await self.pub_socket.send_multipart([symbol.encode(), json.dumps(final).encode()])
                        self.live_data[symbol].append(final)
            except ccp.BadSymbol as e:
                logger.error(f"Symbol {symbol} not supported: {e}. Stopping watcher.")
                break
            except (ccp.NetworkError, ccp.ExchangeNotAvailable) as e:
                logger.warning(f"Network error on {symbol}: {e}. Retrying in 5s...")
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Unexpected error in watch_ohlcv for {symbol}: {e}")
                await asyncio.sleep(5)

    async def handle_order_book_requests(self):
        while self.running:
            try:
                symbol = (await self.rep_socket.recv()).decode()
                try:
                    ob = await asyncio.wait_for(self.exchange.fetch_order_book(symbol, 50), timeout=5.0)
                    await self.rep_socket.send_json({"bids": ob["bids"], "asks": ob["asks"],
                                                     "timestamp": ob.get("timestamp")})
                except Exception as e:
                    logger.error(f"order book {symbol}: {e}")
                    await self.rep_socket.send_json({"error": str(e)})
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Unexpected error in order book handler: {e}")
                await asyncio.sleep(1)

    async def flush_to_parquet(self):
        while self.running:
            try:
                await asyncio.sleep(60)
                await self._do_flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in flush_to_parquet: {e}")

    async def _do_flush(self):
        schema = {"timestamp": pl.Int64, "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
                  "close": pl.Float64, "volume": pl.Float64, "taker_buy_volume": pl.Float64}
        for symbol in self.symbols:
            data, self.live_data[symbol] = self.live_data[symbol], []
            if not data:
                continue
            try:
                df_new = pl.DataFrame([{k: d.get(k) for k in schema} for d in data], schema=schema)
                path = os.path.join(DATA_DIR, f"{symbol.replace('/', '_')}.parquet")
                if os.path.exists(path):
                    df_new = pl.concat([pl.read_parquet(path), df_new])
                df_new.unique(subset=["timestamp"], keep="last").sort("timestamp").write_parquet(path)
            except Exception as e:
                logger.error(f"Failed to write Parquet for {symbol}: {e}")
                self.live_data[symbol] = data + self.live_data[symbol]

    async def run(self):
        self.running = True
        await self.exchange.load_markets()
        logger.info(f"Ingestion for {self.symbols}")
        for s in self.symbols:
            self.tasks.append(asyncio.create_task(self.watch_ohlcv(s)))
        self.tasks.append(asyncio.create_task(self.handle_order_book_requests()))
        self.tasks.append(asyncio.create_task(self.flush_to_parquet()))
        try:
            await asyncio.gather(*self.tasks)
        except asyncio.CancelledError:
            pass

    async def shutdown(self):
        self.running = False
        for t in self.tasks:
            t.cancel()
        await self._do_flush()
        self.pub_socket.close(linger=0)
        self.rep_socket.close(linger=0)
        self.context.term()
        await self.exchange.close()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    daemon = IngestionDaemon(bundle_symbols())
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(daemon.shutdown()))
        except NotImplementedError:
            pass
    try:
        loop.run_until_complete(daemon.run())
    except KeyboardInterrupt:
        pass
    finally:
        try:
            loop.run_until_complete(daemon.shutdown())
        except Exception:
            pass
        loop.close()
