import json
import os

# All paths are absolute, so nothing depends on the working directory.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Database
DB_PATH = os.path.join(ROOT_DIR, "data", "trading_state.db")

# Closed 1m candles recorded by the ingestion daemon
DATA_DIR = os.path.join(ROOT_DIR, "data", "ohlcv")

# Live bundle written by scripts/export_live_bundle.py (models + pairs + params + manifest)
BUNDLE_DIR = os.path.join(ROOT_DIR, "models", "live_bundle")

# ZMQ Ports
ZMQ_TICK_PORT = 5555        # PUB/SUB for CLOSED 1m candles
ZMQ_OB_PORT = 5556          # REQ/REP for order book snapshots

# HTTP API (dashboard)
API_HOST = "127.0.0.1"
API_PORT = 8000

BTC_SYMBOL = "BTC/USDT"


def bundle_symbols(bundle_dir: str = BUNDLE_DIR) -> list[str]:
    """Symbols the live bundle trades (both legs of every pair + BTC)."""
    path = os.path.join(bundle_dir, "manifest.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        manifest = json.load(f)
    syms = {BTC_SYMBOL}
    for p in manifest.get("pairs", []):
        syms.update([p["asset1"], p["asset2"]])
    return sorted(syms)
