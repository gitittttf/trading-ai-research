"""
Single source of truth for trading costs (backtest, live paper trading, benchmarks).

Conventions
-----------
* Every leg is opened with one fill and closed with one fill, so a pair round trip
  is 4 fills. Each fill pays ``fee + half_spread + slippage (+ impact)`` on that
  fill's notional.
* Funding on Binance USD-M perpetuals: at each funding timestamp a LONG position
  pays ``notional * rate`` and a SHORT position receives it (negative rates flip
  this). A position pays/receives funding for timestamps ``f`` with
  ``entry_ts < f <= exit_ts``.

Default rates: Binance USD-M VIP0 is 0.02% maker / 0.05% taker (checked 10/2026,
see docs). Half-spread and slippage are deliberately conservative for mid-cap alts.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class CostModel:
    taker_fee: float = 0.0005        # 5 bp per fill (VIP0 taker)
    maker_fee: float = 0.0002        # 2 bp per fill (VIP0 maker)
    half_spread: float = 0.0001      # 1 bp: crossing half the bid/ask spread per fill
    slippage: float = 0.00005        # 0.5 bp extra adverse fill per fill
    impact_coef: float = 0.001       # sqrt impact: 10 bp at 100% participation of bar volume
    use_maker: bool = False          # maker fills pay no spread/slippage but maker fee
    multiplier: float = 1.0          # stress factor (e.g. 1.5 for robustness checks)

    def per_fill_rate(self, maker: bool | None = None) -> float:
        """Cost per unit notional of one fill, excluding size-dependent impact.
        ``maker`` overrides ``use_maker`` for a single fill (used by the maker fill model)."""
        if self.use_maker if maker is None else maker:
            base = self.maker_fee
        else:
            base = self.taker_fee + self.half_spread + self.slippage
        return base * self.multiplier

    def impact_rate(self, notional: float, bar_volume_usd: float) -> float:
        """Square-root market impact per unit notional for a fill of ``notional``."""
        if bar_volume_usd is None or bar_volume_usd <= 0 or notional <= 0:
            return 0.0
        return self.impact_coef * np.sqrt(notional / bar_volume_usd) * self.multiplier

    def fill_cost(self, notional: float, bar_volume_usd: float | None = None, maker: bool | None = None) -> float:
        """USD cost of a single fill. A fill flagged ``maker=True`` by the fill model rested in the
        book: it pays the maker fee only (no spread, slippage or impact)."""
        notional = abs(notional)
        if maker:
            return notional * self.per_fill_rate(maker=True)
        return notional * (self.per_fill_rate(maker) + self.impact_rate(notional, bar_volume_usd or 0.0))

    def round_trip_rate(self) -> float:
        """Cost per unit of *gross* pair notional for open+close of both legs (no impact)."""
        # gross notional G = n1 + n2 is traded twice (open and close)
        return 2.0 * self.per_fill_rate()

    def to_dict(self) -> dict:
        return asdict(self)


def funding_payment(position_sign: int, notional: float, funding_ts: np.ndarray,
                    funding_rates: np.ndarray, entry_ts: int, exit_ts: int) -> float:
    """
    USD funding COST of one leg (positive = we pay, negative = we receive).

    position_sign: +1 long, -1 short.
    funding_ts / funding_rates: sorted arrays of funding event timestamps (ms) and rates.
    """
    if funding_ts is None or len(funding_ts) == 0 or position_sign == 0:
        return 0.0
    lo = np.searchsorted(funding_ts, entry_ts, side="right")   # f > entry_ts
    hi = np.searchsorted(funding_ts, exit_ts, side="right")    # f <= exit_ts
    if hi <= lo:
        return 0.0
    rate_sum = float(np.sum(funding_rates[lo:hi]))
    return position_sign * abs(notional) * rate_sum


DEFAULT_COSTS = CostModel()
