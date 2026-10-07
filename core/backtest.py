"""
Honest backtest engine for two-leg spread trades.

Timing contract (the most important rule in this repo)
------------------------------------------------------
* A decision is taken at the CLOSE of bar t using only information up to and
  including bar t.
* It is filled at the OPEN of bar t + 1 + latency_bars.
* Exits are checked on bar closes while the position is open (path dependent) and
  are filled at the next bar's open as well. Nothing is clipped after the fact.

Accounting contract
-------------------
* ``simulate_pair`` produces candidate trades with returns per unit of GROSS pair
  notional (|leg1| + |leg2| = 1). It knows nothing about money.
* ``run_portfolio`` walks the candidates in time order, sizes them from REALIZED
  equity only, enforces position/leverage limits, charges costs from
  ``core.costs`` (fees, spread, slippage, impact, funding) and books PnL at the
  EXIT timestamp.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.costs import CostModel, funding_payment

DAY_MS = 86_400_000


@dataclass
class PairArrays:
    """Aligned bar arrays of one pair (index i is the same bar for both legs)."""
    ts: np.ndarray
    open1: np.ndarray
    close1: np.ndarray
    open2: np.ndarray
    close2: np.ndarray
    vol1_usd: np.ndarray | None = None   # bar quote volume, used for impact
    vol2_usd: np.ndarray | None = None
    high1: np.ndarray | None = None      # bar high/low, only needed by the maker fill model
    low1: np.ndarray | None = None
    high2: np.ndarray | None = None
    low2: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.ts)


@dataclass(frozen=True)
class Execution:
    """How orders are executed.

    ``taker`` (default): market orders, filled at the next bar's open (timing contract above).
    ``maker``: limit orders at the decision bar's close (protocol variants 10/11). A leg only
    fills if the market trades at least ``through_bp`` THROUGH the limit within the next
    ``timeout_bars`` bars, and it fills at the limit (no price improvement). Missing legs are
    completed as taker at the next bar's open. Only entries and take-profit exits rest in the
    book; stops, time stops and window ends are always taker.
    """
    mode: str = "taker"
    timeout_bars: int = 3
    through_bp: float = 1.0

    @property
    def maker(self) -> bool:
        return self.mode == "maker"


def maker_fill_bar(arr: PairArrays, leg: int, buy: bool, limit: float, first: int, last: int,
                   through_bp: float) -> int:
    """First bar index in [first, last] in which a resting limit order of ``leg`` fills, or -1.

    Filled only if the market trades through the limit by ``through_bp`` (a touch is not
    enough: our order may sit behind the queue at that price)."""
    if first > last:
        return -1
    if leg == 1:
        lo, hi = arr.low1, arr.high1
    else:
        lo, hi = arr.low2, arr.high2
    if lo is None or hi is None:
        raise ValueError("the maker fill model needs bar high/low arrays")
    k = through_bp / 10_000.0
    seg = lo[first:last + 1] < limit * (1.0 - k) if buy else hi[first:last + 1] > limit * (1.0 + k)
    hit = np.flatnonzero(seg)
    return first + int(hit[0]) if len(hit) else -1


def maker_entry(arr: PairArrays, side: int, t: int, ex: Execution, last: int):
    """Passive entry decided at the close of bar t. Returns None if no leg filled within the
    timeout, else (entry_idx, p1, p2, maker1, maker2). Needs t + timeout + 1 <= last."""
    end = t + ex.timeout_bars
    l1, l2 = float(arr.close1[t]), float(arr.close2[t])
    f1 = maker_fill_bar(arr, 1, side == 1, l1, t + 1, end, ex.through_bp)
    f2 = maker_fill_bar(arr, 2, side == -1, l2, t + 1, end, ex.through_bp)
    if f1 < 0 and f2 < 0:
        return None
    if f1 >= 0 and f2 >= 0:
        return max(f1, f2), l1, l2, True, True
    c = end + 1                       # the missing leg is completed as taker at the next open
    if f1 >= 0:
        return c, l1, float(arr.open2[c]), True, False
    return c, float(arr.open1[c]), l2, False, True


def maker_exit(arr: PairArrays, side: int, t: int, ex: Execution, last: int, bar_ms: int):
    """Passive take-profit exit decided at the close of bar t.
    Returns (exit_idx, p1, p2, maker1, maker2, exit_ts, at_window_end)."""
    end = min(t + ex.timeout_bars, last)
    l1, l2 = float(arr.close1[t]), float(arr.close2[t])
    g1 = maker_fill_bar(arr, 1, side == -1, l1, t + 1, end, ex.through_bp)
    g2 = maker_fill_bar(arr, 2, side == 1, l2, t + 1, end, ex.through_bp)
    if g1 >= 0 and g2 >= 0:
        g = max(g1, g2)
        # filled somewhere inside bar g: still held at its open, gone before the next bar
        return g, l1, l2, True, True, int(arr.ts[g]) + bar_ms - 1, False
    c = t + ex.timeout_bars + 1
    if c <= last:
        x1, x2, idx, ts, at_end = float(arr.open1[c]), float(arr.open2[c]), c, int(arr.ts[c]), False
    else:                             # window ends while waiting: taker at the last close
        x1, x2, idx, ts, at_end = (float(arr.close1[last]), float(arr.close2[last]), last,
                                   int(arr.ts[last]) + bar_ms, True)
    return (idx, l1 if g1 >= 0 else x1, l2 if g2 >= 0 else x2, g1 >= 0, g2 >= 0, ts, at_end)


@dataclass
class ExitRules:
    exit_z: float = 0.5              # take profit when the spread is back inside +-exit_z
    stop_z: float | None = 4.0       # stop when |z| moves beyond stop_z against us
    stop_mode: str = "absolute"      # "absolute": |z| >= stop_z ; "relative": z beyond entry_z +- stop_z
    max_hold_bars: int | None = None
    stop_loss: float | None = None   # unrealized loss per unit gross notional at a bar close


def leg_weights(beta: float) -> tuple[float, float]:
    """Weights of leg1/leg2 per unit gross notional for spread log p1 - beta log p2."""
    b = abs(beta)
    return 1.0 / (1.0 + b), b / (1.0 + b)


def _unit_return(side: int, w1: float, w2: float, e1: float, e2: float, p1: float, p2: float) -> float:
    """Return per unit gross notional: long spread = long leg1, short leg2."""
    return side * w1 * (p1 / e1 - 1.0) - side * w2 * (p2 / e2 - 1.0)


def exit_reason(side: int, entry_z: float, zt: float, rules: ExitRules, unit_ret: float,
                bars_held: int, extra_flags: dict[str, bool] | None = None) -> str | None:
    """Exit decision on a bar close. Shared by the backtest and the live engine so
    both apply EXACTLY the same rules (checked in tests/test_live_parity.py)."""
    reason = None
    if np.isfinite(zt):
        if side == 1:   # bought the spread at a low z, profit when z rises back
            if zt >= -rules.exit_z:
                reason = "take_profit"
            elif rules.stop_z is not None:
                lim = -rules.stop_z if rules.stop_mode == "absolute" else entry_z - rules.stop_z
                if zt <= lim:
                    reason = "stop_z"
        else:
            if zt <= rules.exit_z:
                reason = "take_profit"
            elif rules.stop_z is not None:
                lim = rules.stop_z if rules.stop_mode == "absolute" else entry_z + rules.stop_z
                if zt >= lim:
                    reason = "stop_z"
    if reason is None and rules.stop_loss is not None and unit_ret <= -rules.stop_loss:
        reason = "stop_loss"
    if reason is None and rules.max_hold_bars is not None and bars_held >= rules.max_hold_bars:
        reason = "time_stop"
    if reason is None and extra_flags:
        for name, flag in extra_flags.items():
            if flag:
                reason = name
                break
    return reason


def simulate_pair(arr: PairArrays, z: np.ndarray, entry_signal: np.ndarray, rules: ExitRules,
                  beta: float, pair: str = "", latency_bars: int = 0,
                  trade_window: tuple[int, int] | None = None,
                  extra_exits: dict[str, np.ndarray] | None = None,
                  info: dict[str, np.ndarray] | None = None,
                  execution: Execution | None = None) -> list[dict]:
    """
    Turn per-bar decisions into candidate trades.

    entry_signal[t] in {-1, 0, +1}: +1 = long spread (long leg1 / short leg2),
    decided on bar t's close. ``trade_window=(lo, hi)`` restricts DECISIONS to
    lo <= t < hi (warm-up bars before lo are context only); a position still open
    at hi - 1 is closed at the close of the last available bar <= hi - 1.
    extra_exits: name -> bool array; True at t forces an exit decision at close t.
    info: name -> array of per-bar values copied into the trade at the decision bar.
    execution: ``Execution(mode="maker", ...)`` switches to the passive fill model.
    """
    if execution is not None and execution.maker:
        if latency_bars:
            raise ValueError("the maker fill model assumes latency_bars=0")
        return _simulate_pair_maker(arr, z, entry_signal, rules, beta, pair, trade_window,
                                    extra_exits, info, execution)
    n = len(arr)
    lo, hi = trade_window if trade_window is not None else (0, n)
    hi = min(hi, n)
    w1, w2 = leg_weights(beta)
    trades: list[dict] = []
    extra_exits = extra_exits or {}
    info = info or {}

    side = 0
    fill_idx = -1
    sig_idx = -1
    e1 = e2 = 0.0
    entry_z = 0.0

    bar_ms = int(np.median(np.diff(arr.ts))) if n > 1 else 60_000

    def close_trade(exit_idx: int, x1: float, x2: float, reason: str, at_close: bool = False):
        # an exit priced at a bar's CLOSE happens at the end of that bar, not at its open time
        exit_ts = int(arr.ts[exit_idx]) + (bar_ms if at_close else 0)
        t = {
            "pair": pair, "side": side, "beta": beta, "w1": w1, "w2": w2,
            "signal_idx": sig_idx, "entry_idx": fill_idx, "exit_idx": exit_idx,
            "entry_ts": int(arr.ts[fill_idx]), "exit_ts": exit_ts,
            "entry_p1": e1, "entry_p2": e2, "exit_p1": x1, "exit_p2": x2,
            "entry_z": entry_z,
            "gross_ret": _unit_return(side, w1, w2, e1, e2, x1, x2),
            "hold_minutes": (exit_ts - int(arr.ts[fill_idx])) / 60_000.0,
            "exit_reason": reason,
            "entry_vol1_usd": float(arr.vol1_usd[fill_idx]) if arr.vol1_usd is not None else 0.0,
            "entry_vol2_usd": float(arr.vol2_usd[fill_idx]) if arr.vol2_usd is not None else 0.0,
            "exit_vol1_usd": float(arr.vol1_usd[exit_idx]) if arr.vol1_usd is not None else 0.0,
            "exit_vol2_usd": float(arr.vol2_usd[exit_idx]) if arr.vol2_usd is not None else 0.0,
        }
        for k, v in info.items():
            t[k] = float(v[sig_idx])
        trades.append(t)

    last = hi - 1
    for t in range(lo, hi):
        if side == 0:
            s = int(entry_signal[t]) if np.isfinite(entry_signal[t]) else 0
            if s == 0 or not np.isfinite(z[t]):
                continue
            f = t + 1 + latency_bars
            if f > last:
                continue          # cannot be filled inside the window
            side, sig_idx, fill_idx = s, t, f
            e1, e2 = float(arr.open1[f]), float(arr.open2[f])
            entry_z = float(z[t])
            continue

        if t < fill_idx:
            continue              # order not filled yet (latency)

        unit_now = (_unit_return(side, w1, w2, e1, e2, float(arr.close1[t]), float(arr.close2[t]))
                    if rules.stop_loss is not None else 0.0)
        reason = exit_reason(side, entry_z, z[t], rules, unit_now, t - fill_idx + 1,
                             {name: bool(flags[t]) for name, flags in extra_exits.items()})

        if reason is not None:
            f = t + 1 + latency_bars
            if f <= last:
                close_trade(f, float(arr.open1[f]), float(arr.open2[f]), reason)
            else:
                close_trade(last, float(arr.close1[last]), float(arr.close2[last]), reason + "_at_window_end",
                            at_close=True)
            side = 0
            continue

        if t == last:
            close_trade(last, float(arr.close1[last]), float(arr.close2[last]), "end_of_window", at_close=True)
            side = 0

    return trades


def _simulate_pair_maker(arr: PairArrays, z: np.ndarray, entry_signal: np.ndarray, rules: ExitRules,
                         beta: float, pair: str, trade_window: tuple[int, int] | None,
                         extra_exits: dict[str, np.ndarray] | None, info: dict[str, np.ndarray] | None,
                         ex: Execution) -> list[dict]:
    """simulate_pair with the passive fill model (see ``Execution``). Same decisions and exit
    rules as the taker path; only how (and whether) orders fill differs. While an order of the
    pair rests in the book, the pair takes no new decision."""
    n = len(arr)
    lo, hi = trade_window if trade_window is not None else (0, n)
    hi = min(hi, n)
    last = hi - 1
    w1, w2 = leg_weights(beta)
    extra_exits = extra_exits or {}
    info = info or {}
    bar_ms = int(np.median(np.diff(arr.ts))) if n > 1 else 60_000
    trades: list[dict] = []

    def vol(a, i):
        return float(a[i]) if a is not None else 0.0

    t = lo
    while t <= last:
        s = int(entry_signal[t]) if np.isfinite(entry_signal[t]) else 0
        if s == 0 or not np.isfinite(z[t]) or t + ex.timeout_bars + 1 > last:
            t += 1
            continue
        ent = maker_entry(arr, s, t, ex, last)
        if ent is None:
            t += ex.timeout_bars        # order expired unfilled at the close of t + timeout
            continue
        sig_idx, entry_z = t, float(z[t])
        fill_idx, e1, e2, em1, em2 = ent
        # path-dependent exit checks from the close of the bar that completed the entry
        u = fill_idx
        reason, exit_dec = None, None
        while u <= last:
            unit_now = (_unit_return(s, w1, w2, e1, e2, float(arr.close1[u]), float(arr.close2[u]))
                        if rules.stop_loss is not None else 0.0)
            reason = exit_reason(s, entry_z, z[u], rules, unit_now, u - fill_idx + 1,
                                 {name: bool(flags[u]) for name, flags in extra_exits.items()})
            if reason is not None:
                exit_dec = u
                break
            u += 1
        if reason is None:              # still open at the end of the window: taker at the last close
            reason, exit_dec = "end_of_window", last
            x_idx, x1, x2, xm1, xm2 = last, float(arr.close1[last]), float(arr.close2[last]), False, False
            exit_ts = int(arr.ts[last]) + bar_ms
        elif reason == "take_profit":
            x_idx, x1, x2, xm1, xm2, exit_ts, at_end = maker_exit(arr, s, exit_dec, ex, last, bar_ms)
            if at_end:
                reason += "_at_window_end"
        else:                           # stops, time stop, forced exits: taker at the next open
            f = exit_dec + 1
            if f <= last:
                x_idx, x1, x2 = f, float(arr.open1[f]), float(arr.open2[f])
                exit_ts = int(arr.ts[f])
            else:
                x_idx, x1, x2 = last, float(arr.close1[last]), float(arr.close2[last])
                exit_ts = int(arr.ts[last]) + bar_ms
                reason += "_at_window_end"
            xm1 = xm2 = False
        tr = {
            "pair": pair, "side": s, "beta": beta, "w1": w1, "w2": w2,
            "signal_idx": sig_idx, "entry_idx": fill_idx, "exit_decision_idx": exit_dec, "exit_idx": x_idx,
            "entry_ts": int(arr.ts[fill_idx]), "exit_ts": exit_ts,
            "entry_p1": e1, "entry_p2": e2, "exit_p1": x1, "exit_p2": x2,
            "entry_z": entry_z,
            "gross_ret": _unit_return(s, w1, w2, e1, e2, x1, x2),
            "hold_minutes": (exit_ts - int(arr.ts[fill_idx])) / 60_000.0,
            "exit_reason": reason,
            "entry_vol1_usd": vol(arr.vol1_usd, fill_idx), "entry_vol2_usd": vol(arr.vol2_usd, fill_idx),
            "exit_vol1_usd": vol(arr.vol1_usd, x_idx), "exit_vol2_usd": vol(arr.vol2_usd, x_idx),
            "entry_maker1": em1, "entry_maker2": em2, "exit_maker1": xm1, "exit_maker2": xm2,
        }
        for k, v in info.items():
            tr[k] = float(v[sig_idx])
        trades.append(tr)
        # a new decision is possible at the close of the bar in which the exit completed
        t = max(x_idx, exit_dec + 1)
    return trades


# ----------------------------------------------------------------------------
# Portfolio
# ----------------------------------------------------------------------------

@dataclass
class PortfolioConfig:
    start_equity: float = 1000.0
    alloc_per_trade: float = 0.25      # gross notional per trade as a fraction of realized equity
    max_positions: int = 4
    max_gross_leverage: float = 1.0    # sum(open gross notional) / realized equity


@dataclass
class PortfolioResult:
    trades: pd.DataFrame
    equity: pd.Series                  # realized equity indexed by exit time (UTC)
    rejected: int
    start_equity: float
    meta: dict = field(default_factory=dict)

    def daily_returns(self, start_ms: int | None = None, end_ms: int | None = None) -> pd.Series:
        """Daily returns of realized equity, including flat days (0 return)."""
        if self.equity.empty:
            return pd.Series(dtype=float)
        eq = self.equity.copy()
        start = pd.to_datetime(start_ms, unit="ms", utc=True) if start_ms is not None else eq.index.min()
        end = pd.to_datetime(end_ms, unit="ms", utc=True) if end_ms is not None else eq.index.max()
        idx = pd.date_range(start.floor("D"), end.floor("D"), freq="D", tz="UTC")
        daily = eq.groupby(eq.index.floor("D")).last().reindex(idx).ffill()
        daily = daily.fillna(self.start_equity)
        prev = daily.shift(1).fillna(self.start_equity)
        return daily / prev - 1.0


def trade_costs(trade: dict, gross_notional: float, costs: CostModel,
                funding: dict[str, tuple[np.ndarray, np.ndarray]] | None,
                asset1: str, asset2: str) -> tuple[float, float]:
    """(execution_cost_usd, funding_cost_usd) for one trade of the given gross notional."""
    n1 = gross_notional * trade["w1"]
    n2 = gross_notional * trade["w2"]
    x1n = n1 * trade["exit_p1"] / trade["entry_p1"]
    x2n = n2 * trade["exit_p2"] / trade["entry_p2"]
    # maker flags come from the passive fill model; absent (taker path) -> costs.use_maker
    execution = (costs.fill_cost(n1, trade.get("entry_vol1_usd"), trade.get("entry_maker1")) +
                 costs.fill_cost(n2, trade.get("entry_vol2_usd"), trade.get("entry_maker2")) +
                 costs.fill_cost(x1n, trade.get("exit_vol1_usd"), trade.get("exit_maker1")) +
                 costs.fill_cost(x2n, trade.get("exit_vol2_usd"), trade.get("exit_maker2")))
    fund = 0.0
    if funding:
        side = trade["side"]
        if asset1 in funding:
            fts, frs = funding[asset1]
            fund += funding_payment(side, n1, fts, frs, trade["entry_ts"], trade["exit_ts"])
        if asset2 in funding:
            fts, frs = funding[asset2]
            fund += funding_payment(-side, n2, fts, frs, trade["entry_ts"], trade["exit_ts"])
    return execution, fund


def run_portfolio(candidates: list[dict], costs: CostModel, cfg: PortfolioConfig,
                  pair_assets: dict[str, tuple[str, str]],
                  funding: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
                  accept: callable = None) -> PortfolioResult:
    """
    Event-driven portfolio simulation over candidate trades from several pairs.

    accept(trade) -> bool: optional extra filter evaluated at entry time (e.g. a
    meta-model); it must only use fields known at the decision bar.
    """
    cands = sorted(candidates, key=lambda t: (t["entry_ts"], t["pair"]))
    equity = cfg.start_equity
    open_pos: list[tuple[int, str, dict]] = []      # (exit_ts, pair, booked trade)
    booked: list[dict] = []
    eq_points: list[tuple[int, float]] = []
    rejected = 0

    def realize_until(ts: int):
        nonlocal equity, open_pos
        open_pos.sort(key=lambda x: (x[0], x[1]))
        still = []
        for item in open_pos:
            if item[0] <= ts:
                tr = item[2]
                equity += tr["net_pnl"]
                tr["equity_after"] = equity
                eq_points.append((item[0], equity))
            else:
                still.append(item)
        open_pos = still

    for c in cands:
        realize_until(c["entry_ts"])
        if accept is not None and not accept(c):
            rejected += 1
            continue
        if any(p == c["pair"] for _, p, _ in open_pos) or len(open_pos) >= cfg.max_positions:
            rejected += 1
            continue
        g = cfg.alloc_per_trade * equity
        open_gross = sum(tr["gross_notional"] for _, _, tr in open_pos)
        g = min(g, cfg.max_gross_leverage * equity - open_gross)
        if g <= 1e-9 or equity <= 0:
            rejected += 1
            continue
        a1, a2 = pair_assets[c["pair"]]
        execution, fund = trade_costs(c, g, costs, funding, a1, a2)
        tr = dict(c)
        tr["gross_notional"] = g
        tr["gross_pnl"] = g * c["gross_ret"]
        tr["execution_cost"] = execution
        tr["funding_cost"] = fund
        tr["net_pnl"] = tr["gross_pnl"] - execution - fund
        tr["net_ret"] = tr["net_pnl"] / g
        tr["equity_at_entry"] = equity
        booked.append(tr)
        open_pos.append((c["exit_ts"], c["pair"], tr))
    realize_until(np.iinfo(np.int64).max)

    df = pd.DataFrame(booked)
    if eq_points:
        eq = pd.Series([e for _, e in eq_points],
                       index=pd.to_datetime([t for t, _ in eq_points], unit="ms", utc=True))
    else:
        eq = pd.Series(dtype=float)
    return PortfolioResult(trades=df, equity=eq, rejected=rejected, start_equity=cfg.start_equity)
