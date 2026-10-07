"""
Summarize walk-forward runs from their result folders (never type numbers by hand).

  uv run python scripts/report.py results/<run_id> [results/<run_id> ...]

Prints one markdown row per run/seed with the protocol's Go criteria evaluated.
"""
from __future__ import annotations

import json
import os
import sys

CRITERIA = {
    "trades>=200": lambda s: s.get("n_trades", 0) >= 200,
    "CI>0": lambda s: (s.get("mean_net_ret_ci95_bp") or [0, 0])[0] > 0,
    "Sharpe>=1": lambda s: s.get("sharpe", 0) >= 1.0,
    "DSR>=0.95": lambda s: s.get("dsr", 0) >= 0.95,
    "benchmark p<0.05": lambda s: (s.get("random_entry_benchmark") or {}).get("p_value", 1.0) < 0.05,
}


def load(run_dir: str) -> tuple[dict, dict]:
    with open(os.path.join(run_dir, "manifest.json")) as f:
        manifest = json.load(f)
    with open(os.path.join(run_dir, "summary.json")) as f:
        summary = json.load(f)
    return manifest, summary


def row(run_id: str, seed: str, s: dict) -> str:
    ci = s.get("mean_net_ret_ci95_bp") or [float("nan"), float("nan")]
    bench = s.get("random_entry_benchmark") or {}
    checks = " ".join(("✓" if f(s) else "✗") for f in CRITERIA.values())
    return (f"| {run_id} | {seed} | {s.get('n_trades', 0)} | {s.get('net_pnl', 0):.2f} | "
            f"{s.get('mean_gross_ret_bp', 0):.1f} | {s.get('mean_net_ret_bp', 0):.1f} [{ci[0]:.1f}, {ci[1]:.1f}] | "
            f"{s.get('sharpe', 0):.2f} | {s.get('max_drawdown', 0) * 100:.1f}% | {s.get('dsr', 0):.3f} | "
            f"{bench.get('p_value', float('nan')):.3f} | {checks} |")


def main():
    runs = sys.argv[1:]
    print("| run | seed | trades | net PnL USDT | gross bp/trade | net bp/trade [95% CI] | Sharpe | MaxDD | DSR | "
          "bench p | " + " ".join(CRITERIA) + " |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    daily_rows = []
    for r in runs:
        manifest, summary = load(r)
        if "n_days" in summary and "seeds" not in summary:
            daily_rows.append((manifest["run_id"], summary))
            continue
        for seed, s in summary.get("seeds", {}).items():
            print(row(manifest["run_id"], seed, s))
        for key in ["across_seeds", "across_seeds_meta"]:
            if key in summary:
                a = summary[key]
                print(f"| {manifest['run_id']} | {key} | | mean {a['net_pnl_mean']:.2f} (min {a['net_pnl_min']:.2f}, "
                      f"max {a['net_pnl_max']:.2f}) | | | mean {a['sharpe_mean']:.2f} | | | | "
                      f"positive seeds {a['share_positive'] * 100:.0f}% |")
    if daily_rows:
        print()
        print("| run | days | total return | CAGR | Sharpe | MaxDD | mean bp/day [95% block CI] | DSR | random p | "
              "BTC long Sharpe | EW long Sharpe |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for run_id, s in daily_rows:
            ci = s["mean_daily_ci95_bp"]
            p = (s.get("random_signal_benchmark") or {}).get("p_value", float("nan"))
            print(f"| {run_id} | {s['n_days']} | {s['total_return'] * 100:.1f}% | {s['cagr'] * 100:.1f}% | "
                  f"{s['sharpe']:.2f} | {s['max_drawdown'] * 100:.1f}% | {s['mean_daily_bp']:.2f} "
                  f"[{ci[0]:.2f}, {ci[1]:.2f}] | {s['dsr']:.3f} | {p:.3f} | "
                  f"{s['benchmark_btc_long']['sharpe']:.2f} | {s['benchmark_equal_weight_long']['sharpe']:.2f} |")


if __name__ == "__main__":
    main()
