# trading_ai: pre-registered crypto strategy research with a look-ahead-free backtester

[![tests](https://github.com/gitittttf/trading-ai-research/actions/workflows/tests.yml/badge.svg)](https://github.com/gitittttf/trading-ai-research/actions/workflows/tests.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This is a research and paper-trading stack for systematic crypto strategies. It began as stat-arb on Binance USDⓈ-M perpetuals and later expanded to Kraken. **17 strategy variants were pre-registered and tested against fixed go criteria. None passed, so no real money was used.** The best candidate is a delta-neutral funding harvest. On Binance data it does no better than 3-month T-bills in normal years. On the venue an EU resident may legally use (Kraken Derivatives EU), it loses money after costs. Every result, failures included, is recorded in [`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md).

> Research only. Not investment advice. Paper trading only: nothing here routes orders to an exchange.

## Why this is interesting

The point of this project is the process, not a winning strategy. A backtest can be tuned until it looks good, and the rules here are designed to stop that:

- **Pre-registration.** The protocol was committed before any real market data was loaded. Any change made after seeing results counts as a new trial. It goes into a change log that records the date, the change, the reason, and what had already been seen. This public repository is a clean snapshot of a private research repository, so the order of pre-registration shows in the change-log dates, not in the public git history.
- **Locked holdout.** The last 90 days are excluded automatically. A strategy that passes all other criteria is evaluated on them exactly once, and the result is recorded even when it is bad.
- **Multiple-testing correction.** The Deflated Sharpe Ratio (DSR) uses a trial count that grows with each registered variant: 10, then 12, 16 and 18.
- **Null benchmarks.** Each strategy is compared with randomized versions of itself that have the same exposure.
- **Negative results are reported.** This includes one documented deviation from the protocol and a data reconstruction that failed its own pre-set validation.

## Method

| Component | What it does |
|---|---|
| **No look-ahead** | Decisions are made at the close of bar t and filled at the open of bar t+1. Pairs, hedge ratios and thresholds come from the training window only. Tests change future data and check that past decisions do not change. |
| **Walk-forward** | Training 90 days, test 30 days, step 30 days, 5-minute bars. Pairs are selected in each window: Engle-Granger p ≤ 0.05, β > 0, half-life 1 h to 3 d, return correlation ≥ 0.3, top 10 pairs, at most 2 per coin. |
| **Costs** | A single cost function (`core/costs.py`). Each perp fill pays a 5 bp taker fee, 1 bp half-spread and 0.5 bp slippage, plus square-root impact. Funding comes with its sign from real data. A spot fill pays 10 bp on Binance and 38 bp taker on Kraken, plus spread and slippage. Stress runs use ×1.5 and ×2 costs. |
| **Portfolio** | 1,000 USDT starting capital, 25 % gross notional per trade, at most 4 positions, leverage ≤ 1, sized from realized capital. |
| **Statistics** | Sharpe (daily, annualized with √365), Sortino, drawdown, bootstrap CIs (10-day block bootstrap for the daily strategies), PSR and DSR. |
| **Random benchmarks** | Random entries with the same holding time and exit type. Signals shuffled across coins with the same exposure. Random eligible coins at the same timestamps. A Go requires p < 0.05. |
| **Second period** | 2021–2023 is used as an independent out-of-period test. The broad universe is point-in-time: the 60 most liquid USDT perps by 30-day quote volume, including coins that were delisted later. |
| **Reproducibility** | Every run writes `results/<run_id>/manifest.json` with the commit, the SHA-256 of the data and the parameters. `scripts/report.py` generates the walk-forward and daily result tables from the run folders; the remaining figures are taken from each run's `summary.json`. Binance downloads are checked against the published SHA-256; for other sources the file hash is recorded in the run manifest. The run folders themselves are not published (`results/` is gitignored); they can be regenerated with the commands under Quickstart. |

**Go criteria for trade-based strategies.** All of these must hold:

- at least 200 trades;
- the bootstrap 95 % CI of the mean net return per trade lies above 0;
- net Sharpe ≥ 1.0 and DSR ≥ 0.95;
- net PnL stays positive at 1.5× costs;
- random benchmark p < 0.05;
- for XGBoost, at least 4 of 5 seeds are net positive;
- after all of that, the holdout, evaluated once.

The daily strategies have equivalent criteria. They must pass in both periods.

Variants 1–5 were fixed when the protocol was created, and variants 6–9 were added in later change-log entries. Later variants were registered in groups called phases: phase 5 = variants 10–11 (maker execution), phase 6 = variants 12–15 (new strategy families), phase 7 = variants 16–17 (capital-efficient harvest, Kraken).

Phase 6 adds a lower tier, the **paper candidate**. It allows forward paper trading only, with no money. Phase 7 adds two more requirements: the strategy must beat 100 % T-bills in both periods and must have zero margin-breach days.

## Results

Unless stated otherwise, each period ends at the start of the holdout, 2026-07-03. The protocol's results section has the exact figures, confidence intervals and run IDs.

| # | Strategy | Family | Key number | Verdict |
|---|---|---|---|---|
| 1 | Cointegration pairs, 5m | Spread | 8,056 trades, −879.98 USDT on 1,000; net −10.2 bp/trade | No-Go |
| 2 | Cointegration pairs, 15m | Spread | −785.32 USDT, Sharpe −3.28 | No-Go |
| 3 | Pairs 5m, 1-bar latency | Spread | −892.15 USDT, Sharpe −4.85 | No-Go |
| 4 | XGBoost spread, seeds 0–4 | Spread / ML | mean −169.89 USDT, 0/5 seeds positive | No-Go |
| 5 | XGBoost + meta-label filter | Spread / ML | mean −128.90 USDT, 0/5 seeds positive | No-Go |
| 6 | Slow pairs, 60m bars, hold up to 7 days | Spread | 2,104 trades, −348.96 USDT | No-Go |
| 7 | Daily trend ensemble, long/flat | Daily | 2024–26 −17.5 %; 2021–23 +41.4 % | No-Go |
| 8 | Funding carry, dollar-neutral | Daily | Sharpe 0.88, DSR 0.427 | No-Go |
| 9 | Broad point-in-time carry (top 60) | Daily | Sharpe 1.31 / 1.27 in the two periods; **holdout −12.6 %** | No-Go |
| 10 | Variant 6 with limit orders | Maker execution | −290.89 USDT; gross edge from +5.8 to −1.2 bp | No-Go |
| 11 | Variant 5 with limit orders | Maker execution | mean −29.75 USDT, 2 of 5 seeds positive | No-Go |
| 12 | Delta-neutral funding harvest (short perp, long spot) | Carry | 2021–23 +24.9 %, 2024–26 +6.2 %; 5.5 % p.a. pooled | **Paper candidate** |
| 13 | Short new listings, BTC-hedged | Event | 2021–23 +184.4 %, 2024–26 −100 % (ruin) | No-Go |
| 14 | Cross-sectional momentum | Daily | 2024–26 +61.0 %, 2021–23 −8.1 % | No-Go |
| 15 | Fixed thirds of 12/13/14 | Combination | 2021–23 +62.7 %, 2024–26 −4.3 % | No-Go |
| 16 | Harvest with full-capital sizing and interest on cash (Binance) | Carry | 2024–26 +11.0 % vs T-bills +11.6 %; 3 margin-breach days | No-Go |
| 17 | Variant 16 on Kraken (EU-permitted) | Carry | Period B 2025-10 to 2026-09: −12.6 % vs T-bills +3.7 % | No-Go (informative only) |

**Variant 9 holdout.** This is the one documented deviation from the protocol: the holdout was opened even though variant 9 missed the formal criteria (DSR 0.68 / 0.72 vs 0.95 required; the 2024–26 CI narrowly included 0). The strategy returned −12.6 % in 90 days, with a Sharpe of −2.54. Funding contributed +5.5 % as expected, but the price side lost −17.5 % while BTC rose 33.6 %. Given the earlier Sharpe, one such quarter is roughly a 2-sigma event, because the standard error of a 90-day annualized Sharpe is about 2.0. That does not disprove carry, but it removes any case for real money.

## Key lessons

**1. The cost wall.** The spread strategies have a small positive gross edge of 2.5–10.4 bp per trade. A taker round trip costs roughly 13 bp. For two XGBoost seeds, entry selection beats random entries (p 0.02–0.035), but the strategy still loses money after costs.

**2. Maker fills suffer adverse selection.** About 85–88 % of exits were take-profits, so passive execution looked promising. Limit orders cut execution costs from roughly 13 bp to roughly 5 bp per trade. But a resting order is mostly filled when the market moves against it. Variant 10's gross edge went from +5.8 bp to −1.2 bp. Variant 11's went from 8.2 bp to 4.2 bp, which is −0.8 bp net. The fill model was fixed in advance and requires the market to trade 1 bp through the limit.

**3. Cash is a real hurdle.** A delta-neutral harvest with a Sharpe above 1 can still lose to T-bills.
- Over 2024-01 to 2026-07, variant 12 returned +6.2 % against +11.6 % for 3-month T-bills, with only 21 % of capital deployed on average.
- Variant 16 was the capital-efficient redesign and reached +11.0 %. Interest on idle cash made up 7.2 percentage points of that. Funding plus price added 7.2 %, and costs took 3.5 %.

**4. "Market-neutral" trades still carry margin risk.** In 16 holding periods the short perp rose ≥ 47 % since the last re-hedge, which would have used up a 50 % margin. A daily worst-case check counts margin-breach days. The new sizing rule q = min(1/4, 1/(2n)), with a re-hedge at +50 %, cut them from 12 to 0 in 2021–23 and from 7 to 3 in 2024–26.

**5. Venue and regulation decide what is tradable.** Binance has not offered derivatives to customers in Germany since 2021, and has not accepted new orders from EU customers since 2026-07-01. The Binance results therefore only test the idea; for real money, only the Kraken variant counts. On Kraken:
- funding brought in 7.4 % of capital, and costs took 21.7 %;
- there were 139 entries in one year, against 54 in 2.5 years on Binance;
- each spot fill cost 39.5 bp;
- the result stayed negative with maker orders and with smaller sizing.

**6. Tax is part of the hurdle.** In Germany, the perp and funding fall under § 20 EStG and the spot leg under § 23 EStG. Losses in one bucket do not offset gains in the other. This is a caveat for the go decision, not tax advice, and tax is not modeled.

**7. Validate a reconstruction before trusting it.** Kraken's API only returns 12 months of real funding. Older funding was reconstructed from hourly candles as clip((P/I − 1)/k, ±c), with per-instrument coefficients fixed before any data was loaded. The pass thresholds were also set in advance: correlation ≥ 0.9, and a Jaccard index ≥ 0.8 on the entry signals.
- Mark candles reached 0.13 / 0.46, and trade candles −0.07 / 0.22.
- A timing bug was ruled out: the hourly correlation peaks at lag 0 for BTC, ETH and SOL (0.78–0.82).
- The reconstruction was rejected. Without it, variant 17 has no second period, so a Go is impossible.

## Forward paper test

Variant 12, the paper candidate, was registered for a forward test before any forward data was evaluated. `scripts/paper_forward.py` computes it day by day from 2026-10-01, using only data published after the research data ends and the same code as the backtest. Variants 9 and 14 run alongside for observation only. Results are committed to [`paper/`](paper/).

The test will be evaluated after at least 90 forward days, with positions held on at least 30 of them. It passes if the net return is > 0 and the maximum drawdown is ≤ 5 %. It is still far too early to draw conclusions. Variant 12 runs on Binance data, which the owner cannot trade from the EU, so even a passed forward test only confirms the idea and is not a route to real money.

## Repository structure

```
core/            backtest engine (t+1 fills, path-dependent exits, maker fill model), costs, data loading,
                 point-in-time pair selection, causal features, bar-by-bar (live) version of the rules,
                 metrics (Sharpe, PSR, DSR, bootstrap), daily portfolio backtester with margin checks,
                 funding reconstruction, run manifests, synthetic data
strategies/      cointegration baseline and XGBoost spread strategy
scripts/         data fetchers (Binance Vision, daily universe, spot, FRED T-bills, Kraken),
                 Kraken funding reconstruction, walk_forward.py, daily_backtest.py, report.py,
                 paper_forward.py, export_live_bundle.py
live_trading/    paper-trading stack: closed-candle ingestion, trading core, paper broker, TUI dashboard
tests/           offline tests on synthetic data (look-ahead, fill timing, null and positive controls,
                 cost model, live/backtest parity, maker execution, daily strategies)
paper/           forward paper-trading results (version-controlled)
docs/            EVALUATION_PROTOCOL.md: pre-registration, change log, all results
```

Research and paper trading share the same strategy code. `core/streaming.py` applies the rules bar by bar, and `tests/test_live_parity.py` checks that it matches the backtest. Data and output folders (`data*/`, `results/`, `models/`) are gitignored.

## Quickstart

Requirements: [uv](https://docs.astral.sh/uv/) and the Python version given in `.python-version`. Installing (first run) and fetching data need network access. The tests and the smoke test run offline.

```bash
uv sync --extra shap        # the extra adds SHAP feature selection for the XGBoost strategy
uv run pytest               # offline, synthetic data

# End-to-end smoke test without any market data
uv run python scripts/walk_forward.py --strategy baseline --synthetic
```

**Real data.** All sources are public. Binance files are checked against their published SHA-256.

```bash
uv run python scripts/fetch_binance_vision.py      # 1m klines + funding for the fixed universe
uv run python scripts/fetch_daily_universe.py      # daily candles + funding for all USDT perps, incl. delisted
uv run python scripts/fetch_spot_daily.py          # Binance spot daily candles (harvest variants)
uv run python scripts/fetch_rates.py               # FRED DTB3 T-bill rate
uv run python scripts/fetch_kraken.py              # Kraken Futures data
uv run python scripts/kraken_funding_recon.py      # reconstruct and validate Kraken funding
```

**Research runs.** The holdout is excluded automatically unless you pass `--use-holdout`.

```bash
uv run python scripts/walk_forward.py --strategy baseline --bars 5 --n-trials 10 --benchmark 200
uv run python scripts/walk_forward.py --strategy xgb --bars 5 --seeds 0 1 2 3 4 --n-trials 10
uv run python scripts/daily_backtest.py --strategy carry_broad --universe broad --n-trials 10
uv run python scripts/daily_backtest.py --strategy harvest_sized --universe broad --n-trials 18
uv run python scripts/report.py results/<run_id> [results/<run_id> ...]
```

`daily_backtest.py` also accepts `trend`, `carry`, `harvest`, `listing_short`, `momentum` and `combo`. All of these except `trend` and `carry` need `--universe broad`. Use `--start` and `--end` for out-of-period runs and `--cost-mult` for cost stress. For Kraken runs, point `--daily-dir` at the Kraken data and set the spot cost with `--spot-rate`.

**Paper stack.** Export a bundle, then run each of the last three commands in its own terminal.

```bash
uv run python scripts/export_live_bundle.py --strategy baseline
uv run python live_trading/data_ingestion.py
uv run python live_trading/trading_core.py
uv run python live_trading/dashboard_tui.py
```

## Limitations

- **Survivorship.** The fixed 28-coin universe for the spread strategies was chosen in 2026. The Kraken API lists only perps that exist today, so coins Kraken has delisted are missing.
- **Holdout use.** After the variant 9 deviation, the holdout counts as used up for carry strategies. For variants 12–15 it is not clean, because the market moves in the holdout quarter were already known.
- **Model assumptions.** The maker fill model, the impact model and the margin check are simplifications. The margin check uses the last price instead of the mark price, which is conservative. Variant 17 prices the spot leg from the Kraken index instead of the spot order book.
- **Data gaps.** Kraken funding before the 12 months the API returns could not be reconstructed reliably. In 2021–2023, variant 9 has a small number of position-days with no funding data; these are counted as zero.
- **No live execution.** Order routing, position reconciliation with the exchange and a kill switch are left out on purpose.

## Disclaimer

This repository is for research and education only. It is not investment advice and does not recommend trading any instrument. All trading here is simulated, in backtests and paper trading, and the code places no orders on any exchange. Crypto derivatives are high-risk, and whether you can use them, and how they are taxed, depends on where you live.

Built with AI coding assistance (Claude Code).

## License

MIT, see [LICENSE](LICENSE).
