import json
import os

import pytest

from core.backtest import PortfolioConfig
from core.costs import CostModel
from core.synthetic import cointegrated_market
from live_trading.bundle import BundleError, load_bundle
from live_trading.paper_broker import PaperPortfolio, walk_book


def test_walk_book_vwap_and_thin_book():
    asks = [[10.0, 1.0], [10.5, 2.0]]
    assert walk_book(asks, 5.0) == pytest.approx(10.0)
    # 10 USD at 10.0 (1 unit) + 10.5 USD at 10.5 (1 unit) -> VWAP 10.25
    assert walk_book(asks, 20.5) == pytest.approx(20.5 / 2.0)
    assert walk_book(asks, 1000.0) is None


def test_portfolio_limits_and_pnl_match_backtest_conventions():
    pf = PaperPortfolio(PortfolioConfig(start_equity=1000, alloc_per_trade=0.25, max_positions=2,
                                        max_gross_leverage=1.0), CostModel(taker_fee=0.0005), 1000.0)
    g = pf.size_for_new("A")
    assert g == pytest.approx(250)
    pf.open("A", side=1, beta=1.0, ts=0, entry_z=-2.1, gross=g, p1=10.0, p2=20.0)
    assert pf.size_for_new("A") == 0.0           # one position per pair
    pf.open("B", side=-1, beta=1.0, ts=0, entry_z=2.1, gross=pf.size_for_new("B"), p1=5.0, p2=5.0)
    assert pf.size_for_new("C") == 0.0           # max positions
    # long spread A: leg1 +1%, leg2 flat -> +0.5% of 250 = 1.25 gross
    r = pf.close("A", 10.1, 20.0, funding_cost=0.0)
    assert r["gross_pnl"] == pytest.approx(1.25)
    assert r["fees"] == pytest.approx((250 + 125 * 1.01 + 125) * 0.0005)
    assert pf.equity == pytest.approx(1000 + r["net_pnl"])


def test_bundle_refuses_missing_and_tampered(tmp_path):
    with pytest.raises(BundleError):
        load_bundle(str(tmp_path))
    from scripts_helper import build_baseline_bundle
    out = str(tmp_path / "b")
    build_baseline_bundle(out)
    b = load_bundle(out)
    assert len(b.pairs) >= 1 and b.manifest["strategy"] == "baseline"
    # tamper with a model checksum entry -> refused
    m = json.load(open(os.path.join(out, "manifest.json")))
    m["model_files"] = {"xgb_0.json": "0" * 64}
    open(os.path.join(out, "xgb_0.json"), "w").write("{}")
    json.dump(m, open(os.path.join(out, "manifest.json"), "w"))
    with pytest.raises(BundleError):
        load_bundle(out)
