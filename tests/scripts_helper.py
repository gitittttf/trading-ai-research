"""Test helper: build a small baseline live bundle from synthetic data."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def build_baseline_bundle(out_dir: str):
    import export_live_bundle as elb
    from core.synthetic import cointegrated_market
    bars = cointegrated_market(n_pairs=2, n_minutes=40 * 1440, seed=3)
    return elb.build_bundle(bars, "baseline", train_days=30, valid_days=30, seed=0, out_dir=out_dir)
