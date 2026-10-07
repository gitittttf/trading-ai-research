"""
Null test and positive control for the full baseline pipeline.

* Random walks (no mean reversion): the expected gross return of ANY causal rule is
  zero (optional stopping on a martingale). A significant gross profit here would
  mean the backtester leaks the future.
* OU spreads (true mean reversion): the same pipeline must make money gross.
"""
import itertools

import numpy as np

from core.backtest import PortfolioConfig, run_portfolio
from core.costs import CostModel
from core.metrics import bootstrap_mean_ci
from core.pairs import PairSpec, evaluate_pair, select_pairs
from core.synthetic import START_MS, cointegrated_market, random_walk_market
from strategies.baseline_pairs import BaselinePairsStrategy
from strategies.common import make_windows

DAY = 86_400_000


def _forced_specs(bars, train_start, train_end):
    """All pairs with OLS beta from the training window, no statistical filter."""
    specs = []
    for a, b in itertools.combinations(sorted(bars), 2):
        d1 = bars[a].filter((bars[a]["timestamp"] >= train_start) & (bars[a]["timestamp"] < train_end))
        d2 = bars[b].filter((bars[b]["timestamp"] >= train_start) & (bars[b]["timestamp"] < train_end))
        st = evaluate_pair(np.log(d1["close"].to_numpy()[::60]), np.log(d2["close"].to_numpy()[::60]), 60)
        specs.append(PairSpec(a, b, st["alpha"], st["beta"], st["coint_pvalue"], 240.0,
                              st["return_corr"], train_start, train_end))
    return specs


def _run(bars, spec_fn, days):
    strat = BaselinePairsStrategy()
    cands = []
    windows = make_windows(START_MS, START_MS + days * DAY, train_days=20, test_days=10)
    pair_assets = {}
    for w in windows:
        specs = spec_fn(bars, w.train_start, w.train_end)
        for s in specs:
            pair_assets[s.name] = (s.asset1, s.asset2)
        cands += strat.generate(bars, specs, w)
    return cands, pair_assets


def test_null_random_walk_has_no_gross_edge_and_loses_net():
    bars = random_walk_market(n_assets=6, n_minutes=120 * 1440, seed=11)
    cands, pair_assets = _run(bars, _forced_specs, 120)
    gross = np.array([c["gross_ret"] for c in cands])
    assert len(gross) > 150
    mean, lo, hi = bootstrap_mean_ci(gross, n_boot=4000)
    assert lo <= 0.0 <= hi, f"gross edge on random walks: mean {mean*1e4:.2f}bp CI [{lo*1e4:.2f},{hi*1e4:.2f}]"
    res = run_portfolio(cands, CostModel(), PortfolioConfig(), pair_assets)
    assert res.trades["net_pnl"].sum() < 0


def test_positive_control_ou_spreads_make_money_gross():
    bars = cointegrated_market(n_pairs=3, n_minutes=80 * 1440, seed=12, half_life_minutes=240)
    cands, pair_assets = _run(bars, lambda b, s, e: select_pairs(b, s, e, top_k=5), 80)
    gross = np.array([c["gross_ret"] for c in cands])
    assert len(gross) > 50
    mean, lo, hi = bootstrap_mean_ci(gross, n_boot=4000)
    assert lo > 0.0, f"no gross edge on OU spreads: mean {mean*1e4:.2f}bp"
    res = run_portfolio(cands, CostModel(), PortfolioConfig(), pair_assets)
    assert res.trades["net_pnl"].sum() > 0
