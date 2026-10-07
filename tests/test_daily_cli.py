"""The daily CLI must run end to end for the fixed-universe strategies (variants 7 and 8)."""
import json
import os
import sys

import pytest

from core import manifest
from core.constants import BTC_SYMBOL, UNIVERSE
from core.data import funding_path, klines_path
from core.synthetic import random_walk_market, zero_funding

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


@pytest.mark.parametrize("strategy", ["trend", "carry"])
def test_fixed_universe_cli_writes_summary(strategy, tmp_path, monkeypatch):
    import daily_backtest

    data_dir, results = tmp_path / "data", tmp_path / "results"
    data_dir.mkdir()
    symbols = [BTC_SYMBOL] + [s for s in UNIVERSE if s != BTC_SYMBOL][:3]
    days = 170 if strategy == "trend" else 30          # trend needs its 160-day lookback
    bars = random_walk_market(n_assets=len(symbols), n_minutes=days * 1440, seed=1)
    for sym, df in zip(symbols, bars.values()):
        df.write_parquet(klines_path(sym, str(data_dir)))
    for sym, df in zero_funding(symbols, days=days).items():
        df.write_parquet(funding_path(sym, str(data_dir)))
    monkeypatch.setattr(manifest, "RESULTS_DIR", str(results))
    monkeypatch.setattr(sys, "argv", ["daily_backtest.py", "--strategy", strategy, "--data-dir", str(data_dir),
                                      "--holdout-days", "0", "--benchmark", "2"])
    daily_backtest.main()
    (run_dir,) = results.iterdir()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["n_days"] > 0
    assert "random_signal_benchmark" in summary
