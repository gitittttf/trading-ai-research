# Evaluation Protocol (fixed in advance)

This protocol is **committed before any real market data is loaded**. Anything changed after looking at results counts as a new trial and must be recorded in the change log below. Goal: the result must not come about by tweaking until it looks good.

## Data
- Source: `data.binance.vision`, USDⓈ-M futures, 1-minute klines and `fundingRate`. Every file is verified against its published SHA-256 (`scripts/fetch_binance_vision.py`).
- Period: from 2024-01 up to the last complete day.
- Universe: `core/constants.UNIVERSE` (28 coins, including the delisted FTM). In each window, only coins that largely cover the training window take part.

## Holdout
- The **last 90 days** are locked. `scripts/walk_forward.py` excludes them automatically unless `--use-holdout` is given.
- The holdout is evaluated **exactly once per strategy**, and only after the final variant has been fixed. The result is recorded here, even if it is bad.

## Walk-Forward
- Training 90 days, test 30 days, step 30 days, 5-minute bars.
- Pair selection per window on training data only:
  - Engle-Granger p ≤ 0.05, β > 0;
  - half-life 1 h to 3 d;
  - return correlation ≥ 0.3;
  - top 10, at most 2 pairs per coin.
- Decisions are made at the bar close, fills occur at the next bar open. Stops are path-dependent.

## Costs and Portfolio
- **Per fill:** 5 bp taker fee, 1 bp half-spread and 0.5 bp slippage. On top of that comes square-root impact, and funding enters with its sign from the real data.
- **Portfolio:** 1,000 USDT starting capital, 25 % gross notional per trade. At most 4 positions and leverage at most 1. Sizing is based on realized capital.

## Variants (count as trials for the Deflated Sharpe Ratio)
| # | Variant | Command |
|---|---|---|
| 1 | Baseline 5m | `--strategy baseline --bars 5` |
| 2 | Baseline 15m | `--strategy baseline --bars 15` |
| 3 | Baseline 5m, latency 1 bar | `--strategy baseline --latency 1` |
| 4 | XGB 5m, seeds 0–4 | `--strategy xgb --seeds 0 1 2 3 4` |
| 5 | XGB 5m with meta-filter | `--strategy xgb --meta --seeds 0 1 2 3 4` |
| 6 | Slow baseline: 60m bars, z-window 72 h, time stop up to 7 days | `--strategy baseline --bars 60 --z-window 4320 --max-hold-days 7` |
| 7 | Daily trend long/flat, volatility-weighted (ensemble 10/20/40/80/160 days) | `scripts/daily_backtest.py --strategy trend` |
| 8 | Funding carry cross-sectional, dollar-neutral, weekly (ranked by 7-day funding) | `scripts/daily_backtest.py --strategy carry` |
| 9 | Broad funding carry: as 8, but with a point-in-time universe = the 60 most liquid USDT perpetuals by 30-day quote volume (all listed ones, including those delisted later) and inverse volatility within each leg | `scripts/daily_backtest.py --strategy carry_broad --universe broad` |

**Go criteria for the daily strategies 7 and 8** (instead of trade-based):
- at least 365 daily observations;
- the block-bootstrap 95 % CI (block length 10 days) of the mean daily net return lies above 0;
- net Sharpe ≥ 1.0;
- DSR ≥ 0.95 at n_trials = 10;
- at 1.5× costs the net return stays > 0;
- a random benchmark with signals shuffled across coins (same exposure) yields p < 0.05;
- after that, the holdout, once.

Variant 9 must meet these criteria **in both periods**, 2021-01 to 2023-12 and 2024-01 to the start of the holdout. Because of the less liquid universe, an additional rule applies: at double costs (`--cost-mult 2`) the net return stays > 0.

For the DSR calculation, **n_trials = 10** is used conservatively (`--n-trials 10`). The following runs are **robustness checks and not selection candidates**:
- costs ×1.5 (`--cost-mult 1.5`);
- maker assumption (`--maker`, shows only the potential);
- random-entry benchmark (`--benchmark 200`).

## Go Criteria (all must be met on walk-forward OOS)
1. at least 200 trades;
2. the bootstrap 95 % confidence interval of the mean net return per trade lies entirely above 0;
3. net Sharpe (daily, annualized with √365) ≥ 1.0;
4. DSR ≥ 0.95 at n_trials = 10;
5. at 1.5× costs the net PnL stays > 0;
6. random-entry benchmark: p < 0.05;
7. XGB only: at least 4 of 5 seeds have a net PnL > 0;
8. after that, the holdout, once: net PnL > 0 and Sharpe > 0.

**No-Go means: no real money.** In that case, phase 5 of the plan follows: slower bars, maker execution with a fill model, and other strategies. No further filters are stacked onto the existing strategy.

## Change Log
| Date | Change | Reason | Seen before the change |
|---|---|---|---|
| 2026-10-05 | Protocol created | – | no real data |
| 2026-10-05 | Variant 6 (slow baseline) added, n_trials stays 10 | Research: the cost share falls with a longer horizon (arXiv 2608.21888) | Data loaded, but not a single backtest run on real data yet |
| 2026-10-05 | Variants 7 and 8 (new strategy families with fixed parameters from the literature) added, n_trials stays 10 | Research: trend ensembles with vol sizing (Zarattini/Pagani/Barbon 2025), funding carry (several sources, mixed for 2024–2025) | Variants 1, 2, 3, 6 seen: all No-Go (gross +3 to +6 bp per trade against approx. 13 bp costs). 7 and 8 are families independent of these; nothing about them was tuned to these results. |
| 2026-10-05 | Additional out-of-period test 2021-01 to 2023-12 for the **unchanged** variants 7 and 8 (`--data-dir data_ext --start 2021-01-01 --end 2024-01-01`). It does not replace the holdout, which stays locked. | Variant 8 failed narrowly (Sharpe 0.88, CI includes 0). More independent data without any adjustment is the most honest way to settle this. | Variants 1–3 and 6–8 seen on 2024–2026. The 2021–2023 data had not been looked at by any analysis before. Survivorship caveat: the universe was chosen in 2026, and several coins did not yet exist in 2021–2022. |
| 2026-10-05 | Variant 9 (broad point-in-time universe, inverse volatility) registered before the data download, n_trials stays 10 (9 variants) | Cross-sectional strategies benefit from breadth. The liquidity-based PIT universe also removes the survivorship bias of the coin list chosen in 2026. | Variants 1–3 and 6–8 seen (2024–2026), 7 and 8 also 2021–2023: carry positive in both periods (Sharpe 0.88 and 0.43), but not significant. |
| 2026-10-05 | **Deviation:** the holdout is evaluated once for the unchanged variant 9, even though it formally misses the Go criteria (DSR 0.68/0.72 < 0.95; the 2024–2026 CI narrowly includes 0). The result is recorded here, however it turns out. For later carry variants the holdout is then used up; after that, only forward paper trading counts. | Variant 9 is similarly good in both independent periods (Sharpe 1.31/1.27, MaxDD 14 %/13 %, random p 0.016/0.010, Sharpe 1.13/1.09 at double costs). The third, untouched period is the most honest additional information for the owner's decision. | All results above. The holdout (2026-07-03 to 2026-10-01) had not been looked at by any analysis. |
| 2026-10-05 | Variants 10 and 11 registered: the same signals as variant 6 and 5 respectively, but **executed passively** (maker fill model below, fixed before the first run). n_trials for both = 12 (then 11 variants). This is phase 5 of this protocol ("maker execution with a fill model"). | All spread variants have a small gross edge (2.5–10.4 bp per trade) but fail on roughly 13 bp of taker costs. 85–88 % of their exits are take-profits and can therefore be executed passively. | Variants 1–9 seen, plus the holdout of variant 9 (visible there: BTC +33.6 % in the holdout quarter). No analysis has looked at spread results in the holdout. |
| 2026-10-05 | Phase 6 registered: variants 12–15 (three new strategy families and their fixed combination), n_trials = 16 for all four. Also new: the "paper candidate" stage (no money, forward paper trading only). Both are fixed before spot data is loaded or any of these variants is computed. | The owner commissioned an autonomous search for strategies with a genuine edge. Instead of further tweaking the failed families, the search tests families with their own economic rationale. The number of trials is capped in advance so that the search does not run until something happens to look good by chance. | Variants 1–11 and the holdout of variant 9. Known from these: in the holdout quarter BTC rose by 33.6 %, the 60 most liquid alts on average by only 7.8 %, and alt funding was negative on average. The holdout is therefore **no longer clean** for variants 12–15 and counts as informative only. |
| 2026-10-07 | Phase 7 registered: variants 16 and 17 (funding harvest with a new sizing rule, re-hedging and interest-bearing cash; 17 on Kraken), n_trials = 18. New in the data: the daily candles additionally store the daily high (for the margin check), plus the T-bill rate DTB3 from FRED. Everything is fixed before Kraken prices or funding are loaded or variant 16 is computed. | Reality check of variant 12 (2024-01 to 2026-07): +6.2 % against +11.6 % for 3-month T-bills. On average only 21 % of capital was deployed. In 16 holding periods the perp rose by ≥ 47 % since the last re-hedge; the assumed 50 % margin would have been used up. Binance has not offered derivatives to customers in Germany since 2021 and has not accepted new orders from EU customers since 2026-07-01. | All results of variants 1–15 and the reality check on Binance data (T-bill comparison, EUR view, delay of 1 and 3 days, price rises since re-hedge, tax buckets). From Kraken only the API structure (instrument list, field names, time ranges) and the fee schedule, **no Kraken prices and no Kraken funding**. |
| 2026-10-07 | **Correction before the data:** the reconstruction of Kraken funding divides by the instrument's `fundingRateCoefficient` and caps at its `maxRelativeFundingRate`, instead of a fixed division by 24 and a cap of ±0.25 %. Kraken spot pairs are mapped via the tickers XBT = BTC and XDG = DOGE. | The instrument list shows coefficient 8 and a cap of 0.5 % for PF_XBTUSD. The fixed 24 came from a help page and does not match today's instruments. The spot list names Bitcoin XBT. | Only the metadata of the instrument list and the spot pair list. Still no Kraken prices and no Kraken funding. |

## Phase 6: New Strategy Families (Variants 12–15, fixed in advance)
All four are daily strategies on the point-in-time universe from `data_daily` (all USDT perpetuals ever listed, including delisted ones). Decision at the daily close, trading at the same close with taker costs, funding from the real data. The parameters are fixed and are not optimized.

| # | Variant | Idea and rules |
|---|---|---|
| 12 | Delta-neutral funding harvest | Short perp and long spot of the same coin at the same notional, i.e. no price risk except the basis. Only coins in the top-60 liquidity universe (as in variant 9) with a spot market `<SYMBOL>` on Binance and 30 days of spot history; perps with the prefix `1000` are excluded. Entry when the funding of the last 7 days is ≥ 0.35 % (≈ 18 % p.a.), exit at < 0.15 % or when the coin leaves the universe. At most 10 positions; with more candidates, the highest 7-day funding wins. Capital per position = spot notional plus 50 % margin for the perp, i.e. notional per leg = 1/15 of capital. Spot costs: 10 bp fee plus 1 bp half-spread plus 0.5 bp slippage per fill. |
| 13 | Short new listings, BTC-hedged | Event: a perp whose first trading day is after 2021-02-01. Entry at the close of the day after the listing day: short the coin and long BTCUSDT, 10 % of capital each. Holding period 30 days (earlier on delisting). At most 10 events at once; otherwise new ones are skipped (chronologically). Double costs on the listing leg because of wide spreads. |
| 14 | Cross-sectional momentum | Universe: top 100 by 30-day quote volume with 60 days of history. Weekly (as in variant 9), long the top and short the bottom quintile by 28-day return, inverse volatility (30 days) per leg, each leg ±0.5. |
| 15 | Combination | Fixed thirds of capital in variants 12, 13 and 14, daily return = mean of the three. No estimation of weights. |

Random benchmarks: variant 12 replaces the coin in each position with a random eligible coin that has a spot market, at the same points in time. Variant 13 shorts, instead of the new listing, a random established coin from the top 100 on the same day. Variant 14 shuffles the weights across the eligible coins, as in variant 9.

**Periods:** 2021-01 to 2023-12 and 2024-01 to the start of the holdout; both must pass. The holdout counts as informative only here (see change log).

**Two stages, both fixed in advance:**
1. **Paper candidate** (no money, forward paper trading only). Met if all three points hold:
    - In both periods the net return at costs ×1.5 is positive.
    - The block-bootstrap CI of the mean daily return over 2021-01 to 2026-07 combined lies above 0.
    - Sharpe ≥ 0.5 in both periods.
2. **Go for real money:** all daily criteria above (Sharpe ≥ 1, DSR ≥ 0.95 at n_trials = 16, random p < 0.05, costs ×2 positive) in both periods. After that, at least 3 months of forward paper trading with a positive result within expectations. And even then only with the owner's explicit approval.

## Phase 7: Capital-Efficient Funding Harvest on an EU Exchange (Variants 16 and 17, fixed in advance)
The signals remain exactly those of variant 12: the same universe, entry at a 7-day funding ≥ 0.35 %, exit below 0.15 % or on leaving the universe, at most 10 positions, ranked by 7-day funding. Only sizing, re-hedging and interest on cash are new. None of this is optimized.

| # | Variant | Rules |
|---|---|---|
| 16 | Capital-efficient funding harvest (Binance data) | **Sizing:** notional per leg q = min(1/4, 1/(2n)), with n = number of positions after the day's decisions. Each position therefore ties up spot q plus perp margin q; the perp has no leverage on its margin. From 2 positions on, capital is fully deployed; a single position uses half. **Re-hedge:** if a position's perp is ≥ 50 % above its price at the last re-hedge at the daily close, all positions are brought back to q at the same close. Gains from the spot thus flow into the margin, at normal costs. In addition, rebalancing happens on every change of positions, as in variant 12. **Cash:** the unallocated share 1 − 2·q·n earns interest daily at the 3-month T-bill rate (FRED `DTB3`, act/360), as if it were held in a money market fund. Spot and margin earn no interest. Costs, periods and random benchmark as in variant 12 (random coins at the same points in time, same sizing rule). |
| 17 | Variant 16 on Kraken (permitted for EU customers: perps via Kraken Derivatives EU, MiFID II) | Rules exactly as in 16, data and costs from Kraken (details below). |

**Kraken data for variant 17:**
- **Source:** public Kraken Futures API.
- **Instruments:** perps `PF_<BASE>USD` with the platform `europa`. These are the ones tradable for EU customers. The API only knows perps listed today, so delisted ones are missing. This survivorship error is known and is stated in the result.
- **Spot pair:** a coin is eligible only with a Kraken spot pair `<BASE>/USD`, according to today's `AssetPairs` list.
- **Prices:** for the perp, the daily `trade` candles (close, high) are used; for the spot leg, the daily `spot` candles of the Kraken index. The index thereby replaces the price of the Kraken spot order book.
- **Universe:** the top 60 by the 30-day mean of the perp's USD volume (candle volume × close), with 30 days of history for perp and index.

**Kraken funding:**
- **From 2025-10-06:** real, as hourly `relativeFundingRate`, summed per day. The API only provides the last 12 months.
- **Before that:** reconstructed from hourly candles, namely per hour clip((P/I − 1)/k, ±c). k is the `fundingRateCoefficient` and c the `maxRelativeFundingRate` of the instrument according to today's instrument list (correction of 2026-10-07, see change log).
    - I is the mean of open and close of the index candle (`spot`), P the same for the `mark` or `trade` candle.
    - Kraken computes funding as the hourly average premium divided by a coefficient, with a cap per instrument.
- **Validation:** on the 12 months with real funding, across all coins and days.
    - The candle type with the higher correlation of daily sums with the real funding is chosen.
    - The reconstruction passes at a correlation ≥ 0.9. In addition, the coin-days with an entry signal (7-day sum ≥ 0.35 %) must agree between real and reconstructed to at least 80 % (Jaccard index).
    - If the validation fails, there is no period A. Then no Go is possible, and variant 17 counts as informative only.

**Kraken costs per fill:**
- **Perp:** 5 bp taker plus 1 bp half-spread plus 0.5 bp slippage.
- **Spot:** 38 bp taker (Kraken Pro tier 3, applies from 20,000 USD in assets on the platform) plus 1 bp plus 0.5 bp. Below that, spot costs 60–80 bp; this is computed as a robustness check.

**Periods for variant 17:**
- A: 2022-06-01 to 2025-10-05, with reconstructed funding.
- B: 2025-10-06 to 2026-09-30, with real funding.
Both must pass. The Binance variants have never seen Kraken data. Forward data from 2026-10-01 remain untouched.

**Margin check (16 and 17):** on every day the worst case is computed: all held perps are simultaneously at their daily high, measured since the last re-hedge. The last price is used for this instead of the mark price, which is conservative. The futures account then holds Σ q·(1 − (High/Ref − 1)). A violation occurs if this falls below 5 % of the perp notional at these highs (maintenance margin). The days with a violation are counted. For comparison, the same figure is reported for variant 12 with its 50 % margin.

**Stages (n_trials = 18):**
1. **Paper candidate:**
    - all three points of phase 6, with the pooled CI over both periods of the respective variant;
    - in both periods a return above that of 100 % T-bills, with interest on cash included;
    - no day with a margin violation.
2. **Go for real money:**
    - the Go criteria of phase 6 in both periods, at n_trials = 18;
    - above T-bills even at costs ×2;
    - no margin violation;
    - after that, at least 3 months of forward paper trading, a tax review and the owner's explicit approval.
    - In Germany, perp and funding fall under § 20 EStG for tax purposes, the spot under § 23 EStG. Losses in one bucket do not offset gains in the other.

Variant 16 is not tradable for the owner, because Binance no longer serves EU customers. It only measures what the new sizing rule achieves on the known data. For real money, only variant 17 counts.

**Robustness checks (not selection candidates):**
- variant 12 with its old sizing on Kraken;
- Kraken spot at 60 bp (tier 2) and at 22 bp maker;
- trading delayed by 3 days (money first has to come out of the money market fund);
- return from an EUR perspective.

## Phase 5: Maker Fill Model (Variants 10 and 11, fixed in advance)
| # | Variant | Command |
|---|---|---|
| 10 | Signals of variant 6 (baseline 60m), executed passively, K = 1 bar (60 min) | `--strategy baseline --bars 60 --z-window 4320 --max-hold-days 7 --exec maker --maker-timeout-bars 1 --n-trials 12 --benchmark 200` |
| 11 | Signals of variant 5 (XGB 5m with meta-filter, seeds 0–4), executed passively, K = 3 bars (15 min) | `--strategy xgb --meta --seeds 0 1 2 3 4 --exec maker --maker-timeout-bars 3 --n-trials 12 --benchmark 200` |

Rules of the fill model:
- **Entry and take-profit exit:** for each leg, a limit order is placed at the closing price of the decision bar.
- **Fill condition:** a leg only counts as filled if the market trades at least 1 bp through the limit in one of the next K bars. Buy: Low < Limit · (1 − 1 bp). Sell: High > Limit · (1 + 1 bp).
- **Fill price:** fills are at the limit, without price improvement. The leg costs the maker fee of 2 bp, without spread, slippage or impact.
- **Entry without fill:** if no leg is filled after K bars, the entry expires and there is no trade.
- **Entry with only one leg:** if only one leg is filled, the other is completed as taker at the open of the next bar, with full taker costs. Leg risk is thus reflected in the prices.
- **Take-profit exit:** legs still open after K bars are closed by taker at the open of the next bar.
- **Immediately by taker:** stop-z, time stop and window end are always executed immediately by taker, as before.
- **No new decisions with open orders:** as long as an order is open for a pair, that pair makes no new decision.
- **Signals unchanged:** signals, models, thresholds and meta-filter remain exactly as in variant 5 and 6 respectively. The XGB validation continues to use taker costs.
- **Benchmark:** the random-entry benchmark uses the same fill model, with a random decision bar, the same holding period and the same exit type.
- **Go criteria:** unchanged criteria 1–8 above, i.e. including costs ×1.5 and, after that, the holdout once. For spread strategies the holdout is still unused.

## Forward Paper Trading (fixed in advance, before forward data is evaluated)
- **What:** `scripts/paper_forward.py` computes variant 12 (paper candidate) from 2026-10-01 day by day with the same code as the backtest. Only data published after the end of the research data is used. Until the monthly archive appears, funding is reconstructed from the 1-minute premium index (test on August 2026: error ≈ 2·10⁻⁶ per payment). After that, the official values apply. Results are stored under version control in `paper/`.
- **Observation without claim:** variants 9 and 14 run alongside, to see later whether their failure is confirmed.
- **Evaluation of variant 12:** at the earliest after 90 forward days, on at least 30 of which positions were held. Otherwise the test is still inconclusive and keeps running.
    - **Passed** if the net return is > 0 and the maximum drawdown is ≤ 5 % (2.5 % in the backtest).
    - **Failed** if the drawdown exceeds 5 % or the net return at the end is ≤ 0.
- **Real money**, even after a passed forward test, only with the owner's explicit approval, starting with a small amount. This requires order routing, position reconciliation and a kill switch, none of which exist yet.

## Results

All tables are generated unchanged from the result folders (`uv run python scripts/report.py results/<run_id> ...`). Each folder contains `manifest.json` with commit, data SHA-256 and parameters. The period is up to the start of the holdout (2026-07-03) unless stated otherwise. The check-mark columns test Go criteria 1, 2, 3, 4 and 6.

### Spread Strategies (Variants 1–6, Walk-Forward 2024-04 to 2026-07)
| run | seed | trades | net PnL USDT | gross bp/trade | net bp/trade [95% CI] | Sharpe | MaxDD | DSR | bench p | trades>=200 CI>0 Sharpe>=1 DSR>=0.95 benchmark p<0.05 |
|---|---|---|---|---|---|---|---|---|---|---|
| 20261005-120228_baseline_v1_baseline5m | 0 | 8056 | -879.98 | 3.1 | -10.2 [-12.8, -7.7] | -4.09 | 88.2% | 0.000 | 0.801 | ✓ ✗ ✗ ✗ ✗ |
| 20261005-123551_baseline_v2_baseline15m | seed0 | 6644 | -785.32 | 4.2 | -8.9 [-11.9, -6.0] | -3.28 | 79.4% | 0.000 | 0.343 | ✓ ✗ ✗ ✗ ✗ |
| 20261005-123709_baseline_v3_latency1 | seed0 | 8037 | -892.15 | 2.5 | -10.8 [-13.3, -8.3] | -4.85 | 89.5% | 0.000 | 0.905 | ✓ ✗ ✗ ✗ ✗ |
| 20261005-123835_baseline_v6_slow60m | seed0 | 2104 | -348.96 | 5.8 | -7.2 [-16.8, 2.1] | -0.87 | 37.9% | 0.001 | 0.308 | ✓ ✗ ✗ ✗ ✗ |
| 20261005-124100_xgb_v4v5_xgb | seed0 | 1156 | -287.47 | 2.5 | -10.8 [-22.8, 0.7] | -0.94 | 37.4% | 0.000 | 0.463 | ✓ ✗ ✗ ✗ ✗ |
| 20261005-124100_xgb_v4v5_xgb | seed0_meta | 1034 | -209.99 | 5.2 | -8.2 [-21.0, 3.4] | -0.67 | 31.4% | 0.001 | 0.179 | ✓ ✗ ✗ ✗ ✗ |
| 20261005-124100_xgb_v4v5_xgb | seed1 | 897 | -149.35 | 6.9 | -6.5 [-20.0, 6.3] | -0.65 | 28.6% | 0.003 | 0.114 | ✓ ✗ ✗ ✗ ✗ |
| 20261005-124100_xgb_v4v5_xgb | seed1_meta | 851 | -122.81 | 7.9 | -5.5 [-19.4, 8.0] | -0.53 | 26.7% | 0.005 | 0.075 | ✓ ✗ ✗ ✗ ✗ |
| 20261005-124100_xgb_v4v5_xgb | seed2 | 1159 | -149.67 | 8.5 | -4.8 [-15.9, 5.6] | -0.47 | 30.7% | 0.006 | 0.030 | ✓ ✗ ✗ ✗ ✓ |
| 20261005-124100_xgb_v4v5_xgb | seed2_meta | 1078 | -93.59 | 10.4 | -2.9 [-14.2, 8.1] | -0.27 | 27.9% | 0.017 | 0.020 | ✓ ✗ ✗ ✗ ✓ |
| 20261005-124100_xgb_v4v5_xgb | seed3 | 868 | -96.65 | 9.3 | -4.0 [-17.6, 9.2] | -0.40 | 24.1% | 0.011 | 0.035 | ✓ ✗ ✗ ✗ ✓ |
| 20261005-124100_xgb_v4v5_xgb | seed3_meta | 821 | -85.19 | 9.7 | -3.7 [-17.9, 9.7] | -0.35 | 23.3% | 0.014 | 0.035 | ✓ ✗ ✗ ✗ ✓ |
| 20261005-124100_xgb_v4v5_xgb | seed4 | 992 | -166.33 | 6.8 | -6.5 [-18.5, 4.7] | -0.60 | 31.4% | 0.002 | 0.070 | ✓ ✗ ✗ ✗ ✗ |
| 20261005-124100_xgb_v4v5_xgb | seed4_meta | 895 | -132.92 | 7.8 | -5.6 [-18.6, 6.1] | -0.48 | 27.6% | 0.004 | 0.114 | ✓ ✗ ✗ ✗ ✗ |
| 20261005-124100_xgb_v4v5_xgb | across_seeds | | mean -169.89 (min -287.47, max -96.65) | | | mean -0.61 | | | | positive seeds 0% |
| 20261005-124100_xgb_v4v5_xgb | across_seeds_meta | | mean -128.90 (min -209.99, max -85.19) | | | mean -0.46 | | | | positive seeds 0% |

**Verdict: all No-Go.** There is a small positive gross edge: baseline 2.5–5.8 bp and XGB 2.5–10.4 bp per trade. It is, however, clearly below the roughly 13 bp of costs per round trip. Criterion 7 (XGB, at least 4 of 5 seeds positive) is missed with 0/5 without and 0/5 with the meta-filter. For seeds 2 and 3, the XGB selection is better than random entries with the same holding period (p 0.02–0.035), but it nevertheless remains negative net. The meta-filter slightly improves every seed but makes none positive.

### Daily Strategies (Variants 7–9)
| run | days | total return | CAGR | Sharpe | MaxDD | mean bp/day [95% block CI] | DSR | random p | BTC long Sharpe | EW long Sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| 20261005-124433_daily_trend_v7v8 | 753 | -17.5% | -8.9% | -0.38 | 32.6% | -2.03 [-9.02, 5.70] | 0.016 | 0.002 | -0.01 | -0.36 |
| 20261005-125335_daily_trend_oop2021_2023 | 934 | 41.4% | 14.5% | 0.72 | 35.7% | 4.39 [-3.31, 12.74] | 0.337 | 0.056 | 0.26 | 0.57 |
| 20261005-124502_daily_carry_v7v8 | 906 | 41.4% | 15.0% | 0.88 | 31.8% | 4.27 [-3.24, 10.02] | 0.427 | 0.064 | 0.32 | -0.13 |
| 20261005-125301_daily_carry_oop2021_2023 | 1087 | 28.8% | 8.9% | 0.43 | 37.2% | 3.56 [-8.15, 14.84] | 0.206 | 0.126 | 0.13 | 0.98 |
| 20261005-131734_daily_carry_broad_v9_2024_2026 | 883 | 76.8% | 26.6% | 1.31 | 14.4% | 6.97 [-0.61, 14.11] | 0.676 | 0.016 | 0.41 | -0.28 |
| 20261005-131758_daily_carry_broad_v9_2021_2023 | 1064 | 74.9% | 21.1% | 1.27 | 13.1% | 5.61 [0.64, 11.25] | 0.724 | 0.010 | 0.25 | 0.38 |
| 20261005-131845_daily_carry_broad_v9_2024_2026_cost2 | 883 | 62.9% | 22.4% | 1.13 | 14.6% | 6.05 [-1.54, 13.18] | 0.574 | nan | 0.41 | -0.30 |
| 20261005-131850_daily_carry_broad_v9_2021_2023_cost2 | 1064 | 60.2% | 17.5% | 1.09 | 13.6% | 4.78 [-0.18, 10.40] | 0.609 | nan | 0.25 | 0.36 |

**Verdict: 7 and 8 No-Go.** Variant 9 was similarly good in both periods (Sharpe 1.31 and 1.27, above 1 even at double costs). However, it missed DSR ≥ 0.95, and for 2024–2026 the CI narrowly includes 0. It was therefore tested once on the holdout, as stated in the change log.

### Holdout (once, 2026-07-03 to 2026-09-30)
| run | days | total return | CAGR | Sharpe | MaxDD | mean bp/day [95% block CI] | DSR | random p | BTC long Sharpe | EW long Sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| 20261005-132450_daily_carry_broad_v9_holdout | 90 | -12.6% | -42.2% | -2.54 | 17.8% | -14.42 [-49.28, 4.19] | 0.001 | 0.884 | 3.22 | 0.91 |

**Variant 9 failed the holdout and is No-Go.** Net −12.6 % in 90 days, Sharpe −2.54, worse than 88 % of the random signals. The decomposition shows where this comes from:
- Funding contributed as expected: +5.5 %, i.e. about 6 bp per day, as in 2024–2026.
- The price side (long low, short high funding) lost −17.5 %. In the same quarter, BTC long rose by 33.6 % net.
- Costs were −0.95 %.

With a prior Sharpe of 1.3, a quarter at −2.5 is roughly a 2-sigma event: the standard error of an annualized Sharpe over 90 days is about 2.0. A single quarter therefore does not definitively refute the carry effect. But there is no longer any evidence that justifies real money. The holdout is used up for carry variants; from now on, only forward paper trading counts for them.

Known data limitation: 77 of 11,840 position-days of variant 9 have no funding in the archive and are computed with 0. Affected are ICP, TLM and BNX in 2021–2023. In 2024–2026 and in the holdout there are no such gaps.

### Phase 5: Passive Execution (Variants 10 and 11)
| run | seed | trades | net PnL USDT | gross bp/trade | net bp/trade [95% CI] | Sharpe | MaxDD | DSR | bench p | trades>=200 CI>0 Sharpe>=1 DSR>=0.95 benchmark p<0.05 |
|---|---|---|---|---|---|---|---|---|---|---|
| 20261005-134400_baseline_v10_maker60m | seed0 | 2046 | -290.89 | -1.2 | -5.8 [-15.6, 3.7] | -0.70 | 33.7% | 0.002 | 0.458 | ✓ ✗ ✗ ✗ ✗ |
| 20261005-134355_xgb_v11_xgb_maker | seed0_meta | 1031 | -111.46 | 1.1 | -4.0 [-16.8, 8.2] | -0.33 | 23.8% | 0.011 | 0.129 | ✓ ✗ ✗ ✗ ✗ |
| 20261005-134355_xgb_v11_xgb_maker | seed1_meta | 849 | -37.17 | 4.0 | -1.2 [-14.8, 12.1] | -0.14 | 21.6% | 0.029 | 0.055 | ✓ ✗ ✗ ✗ ✗ |
| 20261005-134355_xgb_v11_xgb_maker | seed2_meta | 1076 | 18.31 | 6.1 | 1.1 [-10.2, 11.7] | 0.13 | 24.2% | 0.075 | 0.005 | ✓ ✗ ✗ ✗ ✓ |
| 20261005-134355_xgb_v11_xgb_maker | seed3_meta | 819 | 22.25 | 6.7 | 1.6 [-12.2, 15.4] | 0.15 | 17.7% | 0.078 | 0.030 | ✓ ✗ ✗ ✗ ✓ |
| 20261005-134355_xgb_v11_xgb_maker | seed4_meta | 890 | -40.66 | 3.6 | -1.5 [-14.6, 10.3] | -0.11 | 22.6% | 0.030 | 0.025 | ✓ ✗ ✗ ✗ ✓ |
| 20261005-134355_xgb_v11_xgb_maker | across_seeds_meta | | mean -29.75 (min -111.46, max 22.25) | | | mean -0.06 | | | | positive seeds 40% |

**Verdict: both No-Go.** Limit orders cut execution costs from roughly 13 to roughly 5 bp per trade. The gross return, however, falls as well, because a resting order is filled mainly when the market moves against the position (adverse selection):
- Variant 10: from +5.8 to −1.2 bp.
- Variant 11: from 8.2 to 4.2 bp (mean over all trades of the 5 seeds).

Variant 11 comes closest to zero net: −0.8 bp per trade, 2 of 5 seeds positive with +18 and +22 USDT. But it misses criterion 7 and all significance criteria. The XGB selection beats random entries with the same execution for 3 of 5 seeds (p 0.005–0.03). So there is a small, genuine selection ability, but no return after costs. Without passing criteria 1–7, the holdout is not touched; it remains unused for spread strategies.

**Overall verdict after 11 variants: no Go, no real money.**

### Phase 6: New Strategy Families (Variants 12–15)
| run | days | total return | CAGR | Sharpe | MaxDD | mean bp/day [95% block CI] | DSR | random p | BTC long Sharpe | EW long Sharpe |
|---|---|---|---|---|---|---|---|---|---|---|
| 20261005-200851_daily_harvest_v6_harvest_2024_2026 | 913 | 6.2% | 2.4% | 1.24 | 2.5% | 0.66 [0.04, 1.38] | 0.563 | 0.976 | 0.37 | -0.21 |
| 20261005-200936_daily_harvest_v6_harvest_2021_2023 | 1034 | 24.9% | 8.2% | 5.34 | 0.4% | 2.15 [1.11, 3.37] | 1.000 | 0.002 | 0.06 | 0.25 |
| 20261005-201054_daily_listing_short_v6_listing_short_2024_2026 | 913 | -100.0% | -100.0% | -0.10 | 100.0% | -2.72 [-40.79, 30.81] | 0.023 | 0.870 | 0.37 | -0.21 |
| 20261005-201139_daily_listing_short_v6_listing_short_2021_2023 | 1034 | 184.4% | 44.6% | 1.00 | 51.3% | 13.37 [-0.71, 26.90] | 0.456 | 0.004 | 0.06 | 0.25 |
| 20261005-201240_daily_momentum_v6_momentum_2024_2026 | 913 | 61.0% | 21.0% | 0.81 | 25.5% | 6.32 [-2.21, 15.50] | 0.302 | 0.046 | 0.37 | -0.34 |
| 20261005-201307_daily_momentum_v6_momentum_2021_2023 | 1034 | -8.1% | -2.9% | -0.01 | 36.0% | -0.07 [-7.04, 7.03] | 0.035 | 0.267 | 0.06 | 0.24 |
| 20261005-201534_daily_harvest_v6_harvest_pooled_2021_2026 | 1948 | 33.2% | 5.5% | 3.14 | 2.5% | 1.47 [0.82, 2.21] | 1.000 | 0.790 | 0.20 | 0.06 |
| 20261005-201701_daily_harvest_v6_harvest_holdout_informative | 90 | -0.0% | -0.1% | -0.81 | 0.0% | -0.01 [-0.06, 0.04] | 0.014 | 0.002 | 3.22 | 3.07 |

Cost stress (total return 2021–2023 / 2024–2026):
- **Variant 12:** ×1.5 gives +22.9 % / +5.2 %, ×2 gives +21.0 % / +4.2 %.
- **Variant 13:** ×1.5 gives +174.2 % / −100 %.
- **Variant 14:** ×1.5 gives −12.2 % / +54.6 %.

Variant 15 (fixed thirds): 2021–2023 +62.7 %, Sharpe 1.05. 2024–2026 −4.3 %, Sharpe 0.15, MaxDD 61.2 %. The correlations of the three components lie between −0.01 and 0.08.

**Verdicts:**
- **Variant 12 (delta-neutral funding harvest): paper candidate, no Go.**
    - The paper-candidate stage is met: at costs ×1.5 both periods are positive, with Sharpe at 4.93 and 1.05 (5.34 and 1.24 at normal costs). The pooled 2021–2026 CI is [0.82, 2.21] bp/day, i.e. above 0.
    - For Go, a good deal is missing in 2024–2026: DSR 0.56 instead of 0.95, random p 0.976. Random coins in the same structure were not worse there.
    - The return is small: 5.5 % p.a. over 2021–2026 and 2.4 % p.a. in 2024–2026. On average, only about a third of capital is deployed.
    - Spot and perp move together: the daily basis return of the held pairs has a standard deviation of 0.12 %, with no deviation above 1 %.
    - The holdout (informative only) yielded 0.0 %, because no coin reached the entry threshold in the quarter.
- **Variant 13 (short new listings): No-Go.** 2021–2023 +184 %, but a total loss in 2024–2026: COAIUSDT (listed 2025-09-25) rose to 13.8 times its price within a few days, at over 3 billion USDT of daily volume. The first run showed −110 %, because the daily backtester had no notion of liquidation. This was corrected (ruin = −100 %, nothing afterwards; fixed in the original private repository history). Variant 9 is demonstrably unaffected by this.
- **Variant 14 (momentum): No-Go.** 2024–2026 +61 % (Sharpe 0.81, random p 0.046), but 2021–2023 −8.1 %.
- **Variant 15 (combination): No-Go.** 2024–2026 negative, because the listing third is ruined.

**Consequence per the protocol:** variant 12 goes into forward paper trading, without money. Forward data from 2026-10-01 are the only remaining clean test.

### Phase 7: Capital-Efficient Funding Harvest and on Kraken (Variants 16 and 17)
Generated from the result folders (`results/*p7_*/summary.json`) at a commit in the original private repository history. "vs T-Bills" is the total return relative to 100 % 3-month T-bills over the same days. "Margin" counts the days with a worst-case margin violation.

| Run | Period | Return | CAGR | Sharpe | MaxDD | bp/day [95% CI] | DSR | Random p | T-Bills | vs T-Bills | Margin |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 2021-03 to 2023-12 | +22.2% | 7.3% | 7.09 | 0.4% | 1.94 [1.26, 2.71] | 1.000 | 0.002 | +7.5% | +13.7% | 0 |
| 16, costs ×2 | 2021-03 to 2023-12 | +17.0% | 5.7% | 5.10 | 0.9% | 1.52 [0.84, 2.31] | 1.000 | – | +7.5% | +8.8% | 0 |
| 16 | 2024-01 to 2026-07 | +11.0% | 4.3% | 0.72 | 4.4% | 1.20 [0.01, 2.62] | 0.192 | 0.928 | +11.6% | −0.5% | 3 |
| 16, costs ×1.5 | 2024-01 to 2026-07 | +9.1% | 3.6% | 0.61 | 5.1% | 1.01 [−0.18, 2.42] | 0.143 | – | +11.6% | −2.2% | 3 |
| 16, costs ×2 | 2024-01 to 2026-07 | +7.3% | 2.8% | 0.49 | 5.7% | 0.82 [−0.38, 2.24] | 0.106 | – | +11.6% | −3.9% | 3 |
| 12 (comparison, 50% margin) | 2021-03 to 2023-12 | +24.9% | 8.2% | 5.34 | 0.4% | 2.15 [1.11, 3.37] | 1.000 | 0.002 | +7.5% | +16.2% | 12 |
| 12 (comparison, 50% margin) | 2024-01 to 2026-07 | +6.2% | 2.4% | 1.24 | 2.5% | 0.66 [0.04, 1.38] | 0.542 | 0.976 | +11.6% | −4.9% | 7 |
| 17 (Kraken, period B) | 2025-10 to 2026-09 | −12.6% | −12.8% | −4.48 | 12.7% | −3.73 [−5.37, −2.38] | 0.000 | 0.002 | +3.7% | −15.7% | 0 |
| 17, costs ×2 | 2025-10 to 2026-09 | −29.6% | −30.0% | −9.29 | 29.6% | −9.77 [−12.01, −7.87] | 0.000 | – | +3.7% | −32.2% | 0 |
| Robustness: 12 sizing on Kraken | 2025-10 to 2026-09 | −4.1% | −4.1% | −2.82 | 4.7% | −1.15 [−2.00, −0.34] | 0.000 | – | +3.7% | −7.5% | 0 |
| Robustness: 17 with spot 60 bp | 2025-10 to 2026-09 | −20.8% | −21.1% | −7.07 | 20.8% | −6.48 [−8.40, −4.92] | 0.000 | – | +3.7% | −23.7% | 0 |
| Robustness: 17 with spot 22 bp maker | 2025-10 to 2026-09 | −5.7% | −5.8% | −2.06 | 6.5% | −1.63 [−3.16, −0.40] | 0.000 | – | +3.7% | −9.1% | 0 |

**Validation of the Kraken reconstruction: failed.**
- Result: mark candles reach a correlation of daily sums of 0.13 and a Jaccard index of 0.46. Trade candles come to −0.07 and 0.22. Required were 0.9 and 0.8.
- A time-offset bug in the code is ruled out: for BTC, ETH and SOL the hourly correlation is highest at offset 0 (0.78–0.82). The hourly means are simply too noisy for the small coins.
- Consequence per the protocol: there is no period A for variant 17, so a Go is ruled out. The values from period B count as informative only.
- In the first days of period B, funding before 2025-10-06 is missing because the reconstruction was rejected. The first entries can therefore be delayed by up to 6 days.

**Verdicts:**
- **Variant 16: not a paper candidate, No-Go.**
    - The new sizing rule eliminates the margin violations in 2021–2023 (0 instead of 12) and reduces them to 3 (instead of 7) in 2024–2026.
    - In 2024–2026, however, at +11.0 % it is narrowly below T-bills (+11.6 %). Of this, 7.2 percentage points come from interest on cash alone. Funding and price together contribute 7.2 %, costs 3.5 %.
    - It thus misses two criteria: "above T-bills in both periods" and "no margin violation".
- **Variant 17 (Kraken): No-Go, and clearly negative even as informative result.**
    - Net −12.6 % in period B, with T-bills at +3.7 %.
    - Funding contributed 7.4 % of capital, costs took 21.7 %.
    - On Kraken, many more coins briefly exceed the threshold: 139 entries in one year against 54 in 2.5 years on Binance. Together with 39.5 bp of spot costs per fill, this eats up the return.
    - Even with maker orders (22 bp) and with the small sizing of variant 12, the result stays negative.

**Overall verdict after 17 variants:** on the exchange the owner is permitted to use, the funding harvest loses money under these rules. On Binance data it is no better than T-bills in normal years. No real money. Variant 12 continues as a forward paper test, even though it is not tradable on Binance.
