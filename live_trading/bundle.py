"""
Load and verify the live bundle written by scripts/export_live_bundle.py.

The live core refuses to run on anything it cannot verify:
missing manifest, unknown bundle version, model files whose SHA-256 differs from
the manifest (e.g. hand-edited or replaced by dummies), or no pairs.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import xgboost as xgb

from core.backtest import ExitRules
from core.manifest import sha256_file
from core.pairs import PairSpec
from core.streaming import StreamingConfig, StreamingPair


class BundleError(RuntimeError):
    pass


@dataclass
class LiveBundle:
    manifest: dict
    pairs: dict[str, StreamingPair]          # pair name -> streaming engine
    pair_assets: dict[str, tuple[str, str]]

    @property
    def expired(self) -> bool:
        return time.time() * 1000 > self.manifest["valid_until_ms"]

    @property
    def allows_entries(self) -> bool:
        return bool(self.manifest.get("trade")) and not self.expired


def load_bundle(bundle_dir: str) -> LiveBundle:
    path = os.path.join(bundle_dir, "manifest.json")
    if not os.path.exists(path):
        raise BundleError(f"no live bundle at {bundle_dir}: run scripts/export_live_bundle.py")
    with open(path) as f:
        m = json.load(f)
    if m.get("bundle_version") != 2:
        raise BundleError(f"unsupported bundle version {m.get('bundle_version')}")
    if not m.get("pairs"):
        raise BundleError("bundle has no pairs")
    for name, digest in m.get("model_files", {}).items():
        fp = os.path.join(bundle_dir, name)
        if not os.path.exists(fp):
            raise BundleError(f"model file missing: {name}")
        if sha256_file(fp) != digest:
            raise BundleError(f"model file {name} does not match the manifest checksum (edited or replaced?)")
    cfg = m["streaming_config"]
    if cfg["strategy"] == "xgb" and m.get("trade") and not m.get("model_files"):
        raise BundleError("xgb bundle marked tradable but contains no models")
    models = []
    for name in sorted(m.get("model_files", {})):
        mdl = xgb.XGBRegressor()
        mdl.load_model(os.path.join(bundle_dir, name))
        models.append(mdl)

    pairs, assets = {}, {}
    for p in m["pairs"]:
        spec = PairSpec(**{k: p[k] for k in PairSpec.__dataclass_fields__})
        rules = ExitRules(exit_z=cfg["exit_z"], stop_z=cfg["stop_z"], stop_mode=cfg.get("stop_mode", "absolute"),
                          max_hold_bars=p.get("max_hold_bars"))
        sc = StreamingConfig(strategy=cfg["strategy"], bar_minutes=cfg["bar_minutes"],
                             z_window_minutes=cfg.get("z_window_minutes", 1440),
                             entry_z=cfg.get("entry_z", float("inf")), rules=rules,
                             horizon_minutes=cfg.get("horizon_minutes", 60), need_edge=cfg.get("need_edge", 0.0),
                             penalty=cfg.get("penalty", 1.0), use_btc=cfg.get("use_btc", False),
                             adf=cfg.get("adf", False))
        pairs[spec.name] = StreamingPair(spec, sc, models=models)
        assets[spec.name] = (spec.asset1, spec.asset2)
    return LiveBundle(manifest=m, pairs=pairs, pair_assets=assets)
