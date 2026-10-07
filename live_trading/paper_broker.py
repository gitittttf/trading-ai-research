"""
Paper execution and portfolio accounting for live trading (pure, unit-tested).

Mirrors core.backtest.run_portfolio: sizes from REALIZED equity, max positions,
gross leverage cap, PnL booked at exit. Fill prices come from walking the real
order book, so spread and slippage are already in the prices; on top we charge the
exchange fee per fill (CostModel fee rate) and funding per leg.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.backtest import PortfolioConfig, leg_weights
from core.costs import CostModel


def walk_book(levels: list, notional_usd: float) -> float | None:
    """Volume-weighted fill price for spending ``notional_usd`` against ``levels``
    ([[price, qty], ...] best first). None if the book is too thin (never guess)."""
    remaining = notional_usd
    qty = cost = 0.0
    for price, size in levels:
        price, size = float(price), float(size)
        lvl = price * size
        take = min(lvl, remaining)
        qty += take / price
        cost += take
        remaining -= take
        if remaining <= 1e-9:
            return cost / qty
    return None


@dataclass
class OpenPosition:
    pair: str
    side: int
    beta: float
    entry_ts: int
    entry_z: float
    gross_notional: float
    units1: float          # signed units of leg 1 (+ long)
    units2: float          # signed units of leg 2
    entry_p1: float
    entry_p2: float
    entry_fees: float
    extra: dict = field(default_factory=dict)


@dataclass
class PaperPortfolio:
    cfg: PortfolioConfig
    costs: CostModel
    equity: float
    positions: dict[str, OpenPosition] = field(default_factory=dict)

    def fee_rate(self) -> float:
        return (self.costs.maker_fee if self.costs.use_maker else self.costs.taker_fee) * self.costs.multiplier

    def size_for_new(self, pair: str) -> float:
        """Gross notional for a new position, 0 if limits forbid it (same rules as the backtest)."""
        if pair in self.positions or len(self.positions) >= self.cfg.max_positions or self.equity <= 0:
            return 0.0
        g = self.cfg.alloc_per_trade * self.equity
        open_gross = sum(p.gross_notional for p in self.positions.values())
        g = min(g, self.cfg.max_gross_leverage * self.equity - open_gross)
        return max(g, 0.0)

    def leg_notionals(self, beta: float, gross: float) -> tuple[float, float]:
        w1, w2 = leg_weights(beta)
        return gross * w1, gross * w2

    def open(self, pair: str, side: int, beta: float, ts: int, entry_z: float, gross: float,
             p1: float, p2: float, extra: dict | None = None) -> OpenPosition:
        n1, n2 = self.leg_notionals(beta, gross)
        pos = OpenPosition(pair=pair, side=side, beta=beta, entry_ts=ts, entry_z=entry_z, gross_notional=gross,
                           units1=side * n1 / p1, units2=-side * n2 / p2, entry_p1=p1, entry_p2=p2,
                           entry_fees=(n1 + n2) * self.fee_rate(), extra=extra or {})
        self.positions[pair] = pos
        return pos

    def close(self, pair: str, p1: float, p2: float, funding_cost: float) -> dict:
        pos = self.positions.pop(pair)
        gross_pnl = pos.units1 * (p1 - pos.entry_p1) + pos.units2 * (p2 - pos.entry_p2)
        exit_fees = (abs(pos.units1) * p1 + abs(pos.units2) * p2) * self.fee_rate()
        net = gross_pnl - pos.entry_fees - exit_fees - funding_cost
        self.equity += net
        return {"gross_pnl": gross_pnl, "fees": pos.entry_fees + exit_fees, "funding_cost": funding_cost,
                "net_pnl": net, "equity_after": self.equity}
