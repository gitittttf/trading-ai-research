"""
Daily portfolio backtester for slow, low-turnover strategies (trend, funding carry).

Timing: weights w[d] are decided at the CLOSE of day d (00:00 UTC of d+1) from data
up to and including day d, and earn the return from close d to close d+1. Costs are
charged on the turnover needed to move from the drifted weights to the new target;
funding is paid by long and received by short positions for every funding event in
(close d, close d+1]. Nothing here uses data after the decision time (tested).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import polars as pl

from core.costs import CostModel

DAY_MS = 86_400_000


@dataclass
class DailyPanel:
    days: np.ndarray            # day start (ms, UTC 00:00); close of day d is at days[d] + DAY_MS
    symbols: list[str]
    close: np.ndarray           # [T, N] close at the END of each day (NaN = no trading that day)
    funding: np.ndarray         # [T, N] sum of funding rates with events in (end of d-1, end of d]
    quote_volume: np.ndarray | None = None   # [T, N] USDT volume of the day (broad universe only)
    high: np.ndarray | None = None           # [T, N] highest trade of the day (margin checks only)

    def slice(self, lo: int, hi: int) -> "DailyPanel":
        return DailyPanel(self.days[lo:hi], self.symbols, self.close[lo:hi], self.funding[lo:hi],
                          None if self.quote_volume is None else self.quote_volume[lo:hi],
                          None if self.high is None else self.high[lo:hi])


def _map_funding(days: np.ndarray, fund: np.ndarray, j: int, fts: np.ndarray, frs: np.ndarray):
    # event at time f belongs to day d if end(d-1) < f <= end(d)  <=>  d = ceil(f / DAY) - 1
    fd = (np.ceil(fts / DAY_MS).astype(np.int64) - 1) * DAY_MS
    k = np.searchsorted(days, fd)
    okf = (k < len(days)) & (days[np.minimum(k, len(days) - 1)] == fd)
    np.add.at(fund[:, j], k[okf], frs[okf])


def build_panel_from_daily(daily: dict[str, pl.DataFrame], funding: dict[str, tuple[np.ndarray, np.ndarray]],
                           start_ms: int, end_ms: int) -> DailyPanel:
    """Panel from exchange DAILY klines (timestamp = day open, close = end-of-day close).
    Days with zero quote volume (e.g. delisted 'zombie' candles) count as not traded.
    The daily high is filled where the files have it (NaN otherwise)."""
    days = np.arange(start_ms // DAY_MS * DAY_MS, end_ms // DAY_MS * DAY_MS, DAY_MS, dtype=np.int64)
    symbols = sorted(daily)
    close = np.full((len(days), len(symbols)), np.nan)
    high = np.full((len(days), len(symbols)), np.nan)
    qv = np.zeros((len(days), len(symbols)))
    fund = np.zeros((len(days), len(symbols)))
    for j, s in enumerate(symbols):
        d = daily[s].filter(pl.col("quote_volume") > 0)
        idx = np.searchsorted(days, d["timestamp"].to_numpy())
        ok = (idx < len(days)) & (days[np.minimum(idx, len(days) - 1)] == d["timestamp"].to_numpy())
        close[idx[ok], j] = d["close"].to_numpy()[ok]
        qv[idx[ok], j] = d["quote_volume"].to_numpy()[ok]
        if "high" in d.columns:
            high[idx[ok], j] = d["high"].to_numpy()[ok]
        if s in funding:
            _map_funding(days, fund, j, *funding[s])
    return DailyPanel(days, symbols, close, fund, qv, high)


def build_panel(bars: dict[str, pl.DataFrame], funding: dict[str, tuple[np.ndarray, np.ndarray]],
                start_ms: int | None = None, end_ms: int | None = None) -> DailyPanel:
    symbols = sorted(bars)
    first = min(int(d["timestamp"][0]) for d in bars.values()) // DAY_MS * DAY_MS
    last = max(int(d["timestamp"][-1]) for d in bars.values()) // DAY_MS * DAY_MS
    if start_ms is not None:
        first = max(first, start_ms // DAY_MS * DAY_MS)
    if end_ms is not None:
        last = min(last, (end_ms // DAY_MS - 1) * DAY_MS)
    days = np.arange(first, last + DAY_MS, DAY_MS, dtype=np.int64)
    close = np.full((len(days), len(symbols)), np.nan)
    fund = np.zeros((len(days), len(symbols)))
    for j, s in enumerate(symbols):
        d = (bars[s].select("timestamp", "close")
             .with_columns((pl.col("timestamp") // DAY_MS * DAY_MS).alias("day"))
             .group_by("day").agg(pl.col("close").last(), pl.col("timestamp").max().alias("last_ts"))
             .sort("day"))
        # a day only counts as closed if trading reached its last hour (no partial last day)
        d = d.filter(pl.col("last_ts") >= pl.col("day") + DAY_MS - 3_600_000)
        idx = np.searchsorted(days, d["day"].to_numpy())
        ok = (idx < len(days)) & (days[np.minimum(idx, len(days) - 1)] == d["day"].to_numpy())
        close[idx[ok], j] = d["close"].to_numpy()[ok]
        if s in funding:
            _map_funding(days, fund, j, *funding[s])
    return DailyPanel(days, symbols, close, fund)


@dataclass
class DailyResult:
    days: np.ndarray            # day index of each return (return from close d-1 to close d)
    net: np.ndarray
    gross: np.ndarray
    cost: np.ndarray
    funding: np.ndarray
    turnover: np.ndarray
    exposure: np.ndarray        # gross exposure held during the day
    interest: np.ndarray | None = None   # interest earned on idle capital (included in net)


LIQUIDATION_LEVERAGE = 20.0     # gross exposure / equity at which a cross-margin account is liquidated


def run_daily(panel: DailyPanel, weights: np.ndarray, costs: CostModel, rebalance: np.ndarray | None = None,
              start: int = 0, col_rate: np.ndarray | None = None, cash_rate: np.ndarray | None = None,
              tied: np.ndarray | None = None) -> DailyResult:
    """
    weights[d]: target weights decided at the close of day d (NaN treated as 0).
    rebalance[d]: if False, no trading at d (positions drift). Default: every day.
    col_rate: optional per-column cost per unit traded (before ``costs.multiplier``), e.g.
    spot columns (higher fee) or a wider spread on newly listed coins.
    cash_rate[d]: optional interest per unit of idle capital earned over day d (close d-1 to close d);
    tied[d]: fraction of equity tied up by the positions held from close d (default 0), so
    1 - tied[d] earns cash_rate[d + 1].
    Ruin: if a day's loss takes the equity to <= 0, or the drifted positions exceed
    LIQUIDATION_LEVERAGE x equity, the account is liquidated at that close: that day's return is
    -100% and every later day is 0 (nothing left to trade).
    Returns per-day results for days start+1 .. T-1.
    """
    T, N = panel.close.shape
    rate = costs.per_fill_rate() if col_rate is None else np.asarray(col_rate, float) * costs.multiplier
    rebalance = np.ones(T, dtype=bool) if rebalance is None else rebalance
    cash_rate = np.zeros(T) if cash_rate is None else np.nan_to_num(np.asarray(cash_rate, float))
    tied = np.zeros(T) if tied is None else np.asarray(tied, float)
    w_prev = np.zeros(N)            # drifted weights carried into day d's close
    out = {k: [] for k in ["days", "net", "gross", "cost", "funding", "turnover", "exposure", "interest"]}
    for d in range(start, T - 1):
        target = np.nan_to_num(weights[d]) if rebalance[d] else w_prev
        # only what has a price NOW can be traded (not listed yet / delisted -> sold at the last
        # known price). Tomorrow's availability is NOT used: that would be look-ahead.
        now = np.isfinite(panel.close[d])
        target = np.where(now, target, 0.0)
        trade = np.abs(target - w_prev)
        turnover = float(np.sum(trade))
        cost = float(np.sum(trade * rate))
        nxt = now & np.isfinite(panel.close[d + 1])
        # no price tomorrow (trading halt / delisting): assume flat for that day
        r = np.where(nxt, panel.close[d + 1] / np.where(nxt, panel.close[d], 1.0) - 1.0, 0.0)
        gross = float(np.sum(target * r))
        fund = float(np.sum(target * panel.funding[d + 1]))   # long pays positive funding
        interest = max(1.0 - float(tied[d]), 0.0) * float(cash_rate[d + 1])
        net = gross - fund - cost + interest
        grown = target * (1.0 + r)
        w_prev = grown / (1.0 + gross) if (1.0 + gross) > 0 else np.zeros(N)
        ruined = (1.0 + net) <= 0 or (1.0 + gross) <= 0 or float(np.sum(np.abs(w_prev))) > LIQUIDATION_LEVERAGE
        if ruined:
            net = -1.0
        for k, v in zip(out, [panel.days[d + 1], net, gross, cost, fund, turnover, float(np.sum(np.abs(target))),
                              interest]):
            out[k].append(v)
        if ruined:
            for dd in range(d + 1, T - 1):
                for k, v in zip(out, [panel.days[dd + 1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
                    out[k].append(v)
            break
    return DailyResult(**{k: np.array(v) for k, v in out.items()})


def block_bootstrap_mean_ci(x: np.ndarray, block: int = 10, n_boot: int = 5000, seed: int = 0,
                            alpha: float = 0.05) -> tuple[float, float, float]:
    """Moving-block bootstrap CI of the mean (keeps short-range autocorrelation)."""
    x = np.asarray(x, float)
    n = len(x)
    if n < block * 2:
        return float(np.mean(x)) if n else 0.0, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


# ---------------------------------------------------------------------------- strategies

def trend_weights(panel: DailyPanel, lookbacks=(10, 20, 40, 80, 160), vol_window: int = 30,
                  vol_target: float = 0.5, max_gross: float = 1.0) -> np.ndarray:
    """Long/flat trend ensemble with inverse-volatility sizing (decided at each close)."""
    c = panel.close
    T, N = c.shape
    w = np.zeros((T, N))
    logc = np.log(c)
    ret = np.vstack([np.full((1, N), np.nan), np.diff(logc, axis=0)])
    L = max(lookbacks)
    for d in range(L, T):
        sig = np.zeros(N)
        cnt = np.zeros(N)
        for lb in lookbacks:
            past = c[d - lb]
            ok = np.isfinite(past) & np.isfinite(c[d])
            sig[ok] += (c[d][ok] > past[ok]).astype(float)
            cnt[ok] += 1
        valid = (cnt == len(lookbacks)) & np.isfinite(c[d])
        window = ret[d - vol_window + 1:d + 1]
        enough = np.sum(np.isfinite(window), axis=0) >= vol_window * 0.8
        vol = np.nanstd(window, axis=0, ddof=1) * np.sqrt(365.0)
        valid &= enough & (vol > 0)
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        s = np.where(valid, sig / np.maximum(cnt, 1), 0.0)
        raw = np.where(valid, s * vol_target / np.where(valid, vol, 1.0), 0.0) / n_valid
        g = np.sum(np.abs(raw))
        w[d] = raw * (max_gross / g if g > max_gross else 1.0)
    return w


def carry_weights(panel: DailyPanel, lookback_days: int = 7, frac: float = 1 / 3,
                  rebalance_weekday: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Dollar-neutral funding carry: long the lowest, short the highest trailing funding.
    Returns (weights, rebalance mask); rebalances on the given weekday (0 = Monday)."""
    T, N = panel.close.shape
    w = np.zeros((T, N))
    reb = np.zeros(T, dtype=bool)
    # day d is the day whose close is at days[d] + 1 day; weekday of the decision time
    weekday = ((panel.days + DAY_MS) // DAY_MS + 3) % 7      # 1970-01-01 was a Thursday (3)
    for d in range(lookback_days, T):
        if weekday[d] != rebalance_weekday:
            continue
        reb[d] = True
        f = panel.funding[d - lookback_days + 1:d + 1].sum(axis=0)
        live = np.all(np.isfinite(panel.close[d - lookback_days + 1:d + 1]), axis=0)
        idx = np.where(live)[0]
        if len(idx) < 6:
            continue
        order = idx[np.argsort(f[idx], kind="stable")]
        k = max(1, int(len(order) * frac))
        longs, shorts = order[:k], order[-k:]
        w[d, longs] = 0.5 / k
        w[d, shorts] = -0.5 / k
    return w, reb


def carry_broad_weights(panel: DailyPanel, top_n: int = 60, lookback_days: int = 7, vol_window: int = 30,
                        min_history: int = 30, frac: float = 1 / 3,
                        rebalance_weekday: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Variant 9: carry on a point-in-time liquid universe with inverse-vol weights per leg.
    Returns (weights, rebalance mask, eligible mask) - eligibility uses data up to day d only."""
    T, N = panel.close.shape
    w = np.zeros((T, N))
    reb = np.zeros(T, dtype=bool)
    elig = np.zeros((T, N), dtype=bool)
    weekday = ((panel.days + DAY_MS) // DAY_MS + 3) % 7
    logc = np.log(panel.close)
    ret = np.vstack([np.full((1, N), np.nan), np.diff(logc, axis=0)])
    for d in range(max(min_history, vol_window, lookback_days), T):
        hist = panel.close[d - min_history + 1:d + 1]
        live = np.all(np.isfinite(hist), axis=0)
        if live.sum() < 6:
            continue
        adv = panel.quote_volume[d - min_history + 1:d + 1].mean(axis=0)
        cand = np.where(live)[0]
        top = cand[np.argsort(-adv[cand], kind="stable")[:top_n]]
        elig[d, top] = True
        if weekday[d] != rebalance_weekday:
            continue
        reb[d] = True
        f = panel.funding[d - lookback_days + 1:d + 1].sum(axis=0)
        order = top[np.argsort(f[top], kind="stable")]
        k = max(1, int(len(order) * frac))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN columns of unlisted coins
            vol = np.nanstd(ret[d - vol_window + 1:d + 1], axis=0, ddof=1)
        for leg, sign in ((order[:k], 1.0), (order[-k:], -1.0)):
            iv = 1.0 / np.where(vol[leg] > 0, vol[leg], np.nan)
            iv = np.nan_to_num(iv, nan=0.0)
            if iv.sum() > 0:
                w[d, leg] = sign * 0.5 * iv / iv.sum()
    return w, reb, elig


# ---------------------------------------------------------------------------- phase 6 (variants 12-15)

SPOT_PREFIX = "SPOT:"


def add_spot_columns(panel: DailyPanel, spot: dict[str, pl.DataFrame]) -> tuple[DailyPanel, dict[int, int]]:
    """Append spot closes as extra columns 'SPOT:<symbol>' (no funding). Returns the wider
    panel and {perp column -> spot column} for perps that have a same-named spot pair."""
    T, N = panel.close.shape
    names = [s for s in sorted(spot) if s in panel.symbols]
    close = np.full((T, len(names)), np.nan)
    qv = np.zeros((T, len(names)))
    for k, s in enumerate(names):
        d = spot[s].filter(pl.col("quote_volume") > 0)
        idx = np.searchsorted(panel.days, d["timestamp"].to_numpy())
        ok = (idx < T) & (panel.days[np.minimum(idx, T - 1)] == d["timestamp"].to_numpy())
        close[idx[ok], k] = d["close"].to_numpy()[ok]
        qv[idx[ok], k] = d["quote_volume"].to_numpy()[ok]
    pqv = panel.quote_volume if panel.quote_volume is not None else np.zeros((T, N))
    high = None if panel.high is None else np.hstack([panel.high, np.full((T, len(names)), np.nan)])
    wide = DailyPanel(panel.days, panel.symbols + [SPOT_PREFIX + s for s in names],
                      np.hstack([panel.close, close]), np.hstack([panel.funding, np.zeros((T, len(names)))]),
                      np.hstack([pqv, qv]), high)
    return wide, {panel.symbols.index(s): N + k for k, s in enumerate(names)}


def liquid_universe(panel: DailyPanel, d: int, top_n: int, min_history: int, cols: np.ndarray | None = None) -> np.ndarray:
    """Column indices of the ``top_n`` coins by average quote volume over the last ``min_history``
    days among those with a close on each of these days (data up to day d only)."""
    hist = panel.close[d - min_history + 1:d + 1]
    live = np.all(np.isfinite(hist), axis=0)
    if cols is not None:
        keep = np.zeros_like(live)
        keep[cols] = True
        live &= keep
    cand = np.where(live)[0]
    if len(cand) == 0:
        return cand
    adv = panel.quote_volume[d - min_history + 1:d + 1].mean(axis=0)
    return cand[np.argsort(-adv[cand], kind="stable")[:top_n]]


def harvest_weights(panel: DailyPanel, spot_col: dict[int, int], n_perp: int, top_n: int = 60,
                    lookback_days: int = 7, enter: float = 0.0035, exit_: float = 0.0015,
                    max_pos: int = 10, notional: float = 1 / 15, min_history: int = 30,
                    rng: np.random.Generator | None = None, episodes: list | None = None):
    """Variant 12: short perp + long spot (same coin, same notional) while trailing funding is high.

    Returns (weights, rebalance mask, episodes). ``episodes`` = [(perp col, entry day, exit day)].
    With ``rng`` and ``episodes`` given, the benchmark is built instead: every episode keeps its
    timing but holds a random coin that was a valid candidate on its entry day."""
    T, N = panel.close.shape
    w = np.zeros((T, N))
    reb = np.zeros(T, dtype=bool)
    perp_cols = np.arange(n_perp)

    def candidates(d):
        uni = liquid_universe(panel, d, top_n, min_history, perp_cols)
        ok = [j for j in uni if j in spot_col
              and np.all(np.isfinite(panel.close[d - min_history + 1:d + 1, spot_col[j]]))]
        return np.array(ok, dtype=int)

    if rng is not None:
        busy: dict[int, int] = {}          # random coin -> day its episode ends (no double holding)
        for _, e, x in sorted(episodes, key=lambda t: (t[1], t[0])):
            cand = [c for c in candidates(e) if busy.get(int(c), -1) <= e]
            if not cand:
                continue
            c = int(rng.choice(cand))
            busy[c] = x
            for d in range(e, x):
                if not (np.isfinite(panel.close[d, c]) and np.isfinite(panel.close[d, spot_col[c]])):
                    break
                w[d, c] -= notional
                w[d, spot_col[c]] += notional
        reb[1:] = np.any(w[1:] != w[:-1], axis=1)
        reb[0] = True
        return w, reb, episodes

    hold: dict[int, int] = {}            # perp col -> entry day
    out_eps = []
    start = max(min_history, lookback_days)
    for d in range(start, T):
        cand = candidates(d)
        cset = set(cand.tolist())
        f7 = panel.funding[d - lookback_days + 1:d + 1].sum(axis=0)
        changed = False
        for j in list(hold):
            if j not in cset or f7[j] < exit_:
                out_eps.append((j, hold.pop(j), d))
                changed = True
        free = max_pos - len(hold)
        if free > 0:
            new = [j for j in cand if j not in hold and f7[j] >= enter]
            new.sort(key=lambda j: (-f7[j], j))
            for j in new[:free]:
                hold[j] = d
                changed = True
        for j in hold:
            w[d, j] = -notional
            w[d, spot_col[j]] = notional
        reb[d] = changed
    for j, e in hold.items():
        out_eps.append((j, e, T))
    return w, reb, out_eps


def harvest_sized(panel: DailyPanel, w12: np.ndarray, reb12: np.ndarray, n_perp: int, max_notional: float = 0.25,
                  rehedge_up: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Variant 16: the positions of variant 12 (``w12``: short perp and long spot columns) sized
    q = min(max_notional, 1 / (2 n)) per leg for n positions, each tying up spot q plus perp margin q.
    All positions are re-hedged at a close where a perp is >= ``rehedge_up`` above its close at the
    last re-hedge (and whenever variant 12 changes positions). Returns (weights, rebalance mask,
    tied capital fraction 2 q n)."""
    T = w12.shape[0]
    w = np.zeros_like(w12)
    reb = np.zeros(T, dtype=bool)
    tied = np.zeros(T)
    ref: dict[int, float] = {}
    for d in range(T):
        held = np.where(w12[d, :n_perp] < 0)[0]
        if len(held) == 0:
            ref = {}
            reb[d] = bool(reb12[d])
            continue
        q = min(max_notional, 1.0 / (2 * len(held)))
        w[d, held] = -q
        w[d, n_perp + np.where(w12[d, n_perp:] > 0)[0]] = q
        tied[d] = 2 * q * len(held)
        c = panel.close[d, held]
        up = any(int(j) in ref and np.isfinite(c[k]) and c[k] >= (1 + rehedge_up) * ref[int(j)]
                 for k, j in enumerate(held))
        reb[d] = bool(reb12[d]) or up
        if reb[d] or set(ref) != set(held.tolist()):
            ref = {int(j): float(c[k]) for k, j in enumerate(held)}
    return w, reb, tied


def margin_breaches(panel: DailyPanel, weights: np.ndarray, rebalance: np.ndarray, n_perp: int, margin: float,
                    maintenance: float = 0.05) -> list[int]:
    """Day indices on which a cross-margin futures account holding the short perps of ``weights``
    would fall below maintenance if every held perp traded at its daily HIGH at the same moment.
    Each position holds ``margin`` x its notional; its loss since the last re-hedge is
    notional x (high / ref - 1). Funding received is ignored (conservative). Needs ``panel.high``."""
    T = len(panel.days)
    out, ref, q, held_prev = [], None, None, None
    for d in range(T - 1):
        held = np.where(weights[d, :n_perp] < 0)[0]
        if len(held) == 0:
            ref = None
            continue
        if ref is None or rebalance[d] or not np.array_equal(held, held_prev):
            ref, q, held_prev = panel.close[d, held], -weights[d, held], held
        h = panel.high[d + 1, held]
        h = np.where(np.isfinite(h), h, np.where(np.isfinite(panel.close[d + 1, held]), panel.close[d + 1, held], ref))
        ratio = h / ref
        wallet = float(np.sum(q * (margin - (ratio - 1.0))))
        if wallet < maintenance * float(np.sum(q * ratio)):
            out.append(d + 1)
    return out


def listing_short_weights(panel: DailyPanel, listing_day: dict[int, int], btc_col: int, hold_days: int = 30,
                          size: float = 0.1, max_events: int = 10,
                          rng: np.random.Generator | None = None, top_n: int = 100, min_history: int = 60):
    """Variant 13: short each newly listed perp from the close of the day after its listing day
    for ``hold_days`` days, hedged with an equal long in BTC. ``listing_day``: column -> panel day
    of the first trading day (only listings inside the registered period). Returns
    (weights, rebalance mask, events). With ``rng``: the benchmark shorts a random established
    coin (top ``top_n`` by volume with ``min_history`` days) instead, same days and hedge."""
    T, N = panel.close.shape
    w = np.zeros((T, N))
    events = []
    active_until: list[int] = []
    for j, L in sorted(listing_day.items(), key=lambda kv: (kv[1], panel.symbols[kv[0]])):
        e = L + 1
        if e >= T - 1:
            continue
        active_until = [x for x in active_until if x > e]
        if len(active_until) >= max_events:
            continue
        c = j
        if rng is not None:
            if e < min_history:
                continue
            uni = [k for k in liquid_universe(panel, e, top_n, min_history) if k != btc_col]
            if not uni:
                continue
            c = int(rng.choice(uni))
        if not np.isfinite(panel.close[e, c]):
            continue
        end = min(e + hold_days, T)
        for d in range(e, end):
            if not np.isfinite(panel.close[d, c]):
                end = d                     # delisted: position is gone from this day on
                break
            w[d, c] -= size
            w[d, btc_col] += size
        active_until.append(end)
        events.append((c, e, end))
    reb = np.zeros(T, dtype=bool)
    reb[1:] = np.any(w[1:] != w[:-1], axis=1)
    reb[0] = True
    return w, reb, events


def xs_momentum_weights(panel: DailyPanel, top_n: int = 100, lookback_days: int = 28, vol_window: int = 30,
                        min_history: int = 60, frac: float = 0.2,
                        rebalance_weekday: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Variant 14: weekly long top / short bottom quintile by trailing return, inverse vol per leg."""
    T, N = panel.close.shape
    w = np.zeros((T, N))
    reb = np.zeros(T, dtype=bool)
    elig = np.zeros((T, N), dtype=bool)
    weekday = ((panel.days + DAY_MS) // DAY_MS + 3) % 7
    logc = np.log(panel.close)
    ret = np.vstack([np.full((1, N), np.nan), np.diff(logc, axis=0)])
    for d in range(max(min_history, lookback_days, vol_window), T):
        top = liquid_universe(panel, d, top_n, min_history)
        if len(top) < 10:
            continue
        elig[d, top] = True
        if weekday[d] != rebalance_weekday:
            continue
        reb[d] = True
        mom = logc[d, top] - logc[d - lookback_days, top]
        order = top[np.argsort(mom, kind="stable")]
        k = max(1, int(len(order) * frac))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN columns of unlisted coins
            vol = np.nanstd(ret[d - vol_window + 1:d + 1], axis=0, ddof=1)
        for leg, sign in ((order[-k:], 1.0), (order[:k], -1.0)):
            iv = np.nan_to_num(1.0 / np.where(vol[leg] > 0, vol[leg], np.nan), nan=0.0)
            if iv.sum() > 0:
                w[d, leg] = sign * 0.5 * iv / iv.sum()
    return w, reb, elig
