"""
Shared paths and the research universe.
"""
import os

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DIRECTORY PATHS (derived from this file's location in core/)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT_DIR, "data")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RESEARCH UNIVERSE (Binance USD-M perpetuals)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixed list of liquid USD-M perpetuals. Pairs are selected point-in-time inside
# every walk-forward window from whatever has data in that window. Note: picking
# this list today still carries some survivorship bias; FTM is kept on purpose
# although it was delisted (data until delisting is used).
UNIVERSE = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT",
    "AVAX/USDT", "DOT/USDT", "LINK/USDT", "NEAR/USDT",
    "ATOM/USDT", "LTC/USDT", "BCH/USDT", "XRP/USDT", "UNI/USDT",
    "APT/USDT", "OP/USDT", "ARB/USDT", "POL/USDT", "1000PEPE/USDT",
    "RENDER/USDT", "FET/USDT", "SUI/USDT", "STX/USDT", "TAO/USDT",
    "INJ/USDT", "TIA/USDT", "SEI/USDT", "FTM/USDT",
]
BTC_SYMBOL = "BTC/USDT"
