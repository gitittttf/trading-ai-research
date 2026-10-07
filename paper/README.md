# Forward Paper Trading (no real money)

This directory holds results on data published **after** the end of the research data (2026-09-30).
No analysis saw these days beforehand, which makes this the only clean test.

- `summary.json`: status, total return and drawdown per strategy, data cutoff, commit.
  The 2026-10-05 run was produced before this public snapshot, so its `git`/`commit`
  fields were removed from `summary.json` and `runs.jsonl`; runs from the public
  repository include them again.
- `forward_<strategy>.csv`: day-by-day net, gross, costs, funding and the positions held.
- `runs.jsonl`: one entry per run. If an earlier value changes there, the official
  funding archive has become available and replaced the reconstructed values.

Recompute: `uv run python scripts/paper_forward.py` (downloads the Binance archives, roughly 10–20 minutes).

| Strategy | Role |
|---|---|
| `v12_harvest` | Paper candidate per the protocol: short perp and long spot when funding is high |
| `v9_carry` | observation only (failed the holdout) |
| `v14_momentum` | observation only (negative in 2021–2023) |

The evaluation criteria are defined in `docs/EVALUATION_PROTOCOL.md` (forward paper trading section).
