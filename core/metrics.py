"""
Performance statistics with an emphasis on NOT fooling ourselves.

* Sharpe/Sortino on DAILY returns of realized equity (crypto trades 365 days).
* Bootstrap confidence interval of the mean net trade return.
* Probabilistic and Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014): the
  DSR asks how likely the observed Sharpe beats the best Sharpe one would expect
  from ``n_trials`` strategies with no skill.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def sharpe(returns: np.ndarray, periods_per_year: float = 365.0) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or np.std(r, ddof=1) == 0:
        return 0.0
    return float(np.mean(r) / np.std(r, ddof=1) * math.sqrt(periods_per_year))


def sortino(returns: np.ndarray, periods_per_year: float = 365.0) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    if downside == 0:
        return 0.0
    return float(np.mean(r) / downside * math.sqrt(periods_per_year))


def max_drawdown(equity: np.ndarray) -> float:
    eq = np.asarray(equity, dtype=float)
    if len(eq) == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    return float(np.max((peak - eq) / peak))


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05,
                      seed: int = 0) -> tuple[float, float, float]:
    """(mean, lower, upper) percentile bootstrap CI of the mean."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def probabilistic_sharpe(sr: float, n_obs: int, skew: float, kurt: float, sr_benchmark: float = 0.0) -> float:
    """PSR for a NON-annualized Sharpe ``sr`` estimated from ``n_obs`` returns.
    kurt is the (non-excess) kurtosis, i.e. 3 for a normal distribution."""
    if n_obs < 3:
        return 0.0
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if denom <= 0:
        return 0.0
    return float(stats.norm.cdf((sr - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(denom)))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum (non-annualized) Sharpe of n_trials skill-less strategies."""
    if n_trials <= 1:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(sr_variance) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe(returns: np.ndarray, n_trials: int, trial_sr_variance: float | None = None) -> float:
    """
    DSR of a daily return series given how many configurations were tried.
    If the variance of Sharpe ratios across trials is unknown we use the variance
    of the Sharpe estimator itself, 1/(T-1), which is the null-hypothesis value.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    t = len(r)
    if t < 3 or np.std(r, ddof=1) == 0:
        return 0.0
    sr = float(np.mean(r) / np.std(r, ddof=1))
    var = trial_sr_variance if trial_sr_variance is not None else 1.0 / (t - 1)
    sr0 = expected_max_sharpe(n_trials, var)
    return probabilistic_sharpe(sr, t, float(stats.skew(r)), float(stats.kurtosis(r, fisher=False)), sr0)


def summarize(trades: pd.DataFrame, daily: pd.Series, start_equity: float,
              n_trials: int = 1) -> dict:
    """Headline numbers of one run. Every number is computed, never typed."""
    out: dict = {"n_trades": int(len(trades))}
    if len(trades) == 0:
        out.update({"net_pnl": 0.0, "total_return": 0.0, "sharpe": 0.0, "sortino": 0.0,
                    "max_drawdown": 0.0, "dsr": 0.0, "psr": 0.0})
        return out
    net = trades["net_pnl"].to_numpy()
    nr = trades["net_ret"].to_numpy()
    gr = trades["gross_ret"].to_numpy()
    eq = start_equity + np.cumsum(trades.sort_values("exit_ts")["net_pnl"].to_numpy())
    m, lo, hi = bootstrap_mean_ci(nr)
    d = daily.to_numpy()
    out.update({
        "net_pnl": float(net.sum()),
        "total_return": float(net.sum() / start_equity),
        "gross_pnl": float(trades["gross_pnl"].sum()),
        "execution_cost": float(trades["execution_cost"].sum()),
        "funding_cost": float(trades["funding_cost"].sum()),
        "win_rate": float(np.mean(net > 0)),
        "mean_gross_ret_bp": float(np.mean(gr) * 1e4),
        "mean_net_ret_bp": float(m * 1e4),
        "mean_net_ret_ci95_bp": [float(lo * 1e4), float(hi * 1e4)],
        "median_hold_minutes": float(trades["hold_minutes"].median()),
        "n_days": int(len(d)),
        "sharpe": sharpe(d),
        "sortino": sortino(d),
        "max_drawdown": max_drawdown(np.concatenate([[start_equity], eq])),
        "psr": probabilistic_sharpe(
            float(np.mean(d) / np.std(d, ddof=1)) if len(d) > 2 and np.std(d, ddof=1) > 0 else 0.0,
            len(d), float(stats.skew(d)) if len(d) > 2 else 0.0,
            float(stats.kurtosis(d, fisher=False)) if len(d) > 3 else 3.0),
        "dsr": deflated_sharpe(d, n_trials),
        "n_trials_for_dsr": int(n_trials),
        "exit_reasons": trades["exit_reason"].value_counts().to_dict(),
    })
    return out
