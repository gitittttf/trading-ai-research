"""
Run manifests: every research number must be traceable to code + data + params.

results/<run_id>/manifest.json records the git commit (and whether the tree was
dirty), the exact command, parameters, seeds, SHA-256 of every data file read and
the package versions. Reports are generated from these folders only.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from importlib import metadata

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

_PACKAGES = ["numpy", "pandas", "polars", "scipy", "scikit-learn", "statsmodels", "xgboost"]


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT_DIR, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return ""
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def package_versions() -> dict:
    out = {}
    for p in _PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def _portable_arg(arg: str) -> str:
    """Absolute paths in argv are stored relative to the repo (no local user paths in manifests)."""
    if not os.path.isabs(arg):
        return arg
    try:
        return os.path.relpath(arg, ROOT_DIR)
    except ValueError:              # different drive on Windows
        return os.path.basename(arg)


def new_run_dir(tag: str) -> tuple[str, str]:
    run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + f"_{tag}"
    path = os.path.join(RESULTS_DIR, run_id)
    os.makedirs(path, exist_ok=False)
    return run_id, path


def write_manifest(run_dir: str, run_id: str, params: dict, data_files: list[str],
                   extra: dict | None = None) -> str:
    manifest = {
        "run_id": run_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": [_portable_arg(a) for a in sys.argv],
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.python_implementation()}",
        "git": git_state(),
        "packages": package_versions(),
        "params": params,
        "data_files": {os.path.relpath(p, ROOT_DIR): sha256_file(p) for p in sorted(set(data_files))
                       if os.path.exists(p)},
    }
    if extra:
        manifest.update(extra)
    path = os.path.join(run_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path
