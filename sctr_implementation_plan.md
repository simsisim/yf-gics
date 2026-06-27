# SCTR Implementation Plan
# Sector & Industry Rotation — Research Findings and Full Plan

*Updated: 2026-06-27*

---

## 1. What StockCharts SCTR Actually Does (Confirmed)

### 1.1 The Six-Indicator Formula

| Timeframe | Indicator | Weight | Notes |
|---|---|---|---|
| **Long-term** | 14-month Rate of Change (ROC) | 30% | Monthly close, 14 bars back |
| **Long-term** | % above/below 200-day **SMA** | 15% | `(close - sma200) / sma200 * 100` |
| **Medium-term** | 9-month ROC | 15% | Monthly close, 9 bars back |
| **Medium-term** | % above/below 50-day **SMA** | 15% | `(close - sma50) / sma50 * 100` |
| **Short-term** | 3-month ROC | 10% | Monthly close, 3 bars back |
| **Short-term** | 3-day slope of PPO Histogram | 15% | PPO(12,26,9), slope = (hist[-1] - hist[-4]) / 3 |
| **Total** | | **100%** | Group weights: 45% long / 30% medium / 25% short |

**Critical detail:** The raw score is NOT a direct 0–99.9 value. StockCharts:
1. Computes the raw weighted score for every stock in a universe
2. **Ranks** all stocks by that score within their universe bucket
3. Assigns the **percentile rank** (0.0–99.9) as the SCTR

This means SCTR is always relative — a stock's SCTR changes when peers change, even if its own technicals are unchanged.

### 1.2 Universe Buckets — Hard Limits

StockCharts divides stocks into three universes by market cap, and each stock is ranked **only within its own bucket**:

| Universe | Market Cap Threshold | Observed in scraped data |
|---|---|---|
| Large-cap | > $10B | min ~$9.4B, max ~$4.97T, ~266 stocks (Jun 2026) |
| Mid-cap | $2B – $10B | min ~$1.1B, max ~$14B, ~178 stocks |
| Small-cap | $250M – $2B | min ~$166M, max ~$2.1B, ~59 stocks |

*Note: boundary overlaps in scraped data are expected — StockCharts uses point-in-time market cap at their calculation time.*

### 1.3 Industry SCTR — How It Works

StockCharts applies the **exact same 6-indicator SCTR formula** to the **Dow Jones US Industry Index price series** (`$DJUS...` / `^YH...` symbols). Each industry index is treated as a security. The 145 industry indexes are ranked against each other to produce the industry SCTR.

**This is NOT the median of member stocks' SCTRs.** It is computed on the index price itself.

Confirmed by: StockCharts industry summary output (`$DJUSAV`, `$DJUSBC`, etc.) and the `^YH...` symbol mapping in `industries.csv`.

---

## 2. What We Already Have

### 2.1 Data Assets

| Asset | Location | Count | History |
|---|---|---|---|
| Daily OHLCV (stocks) | `downloadData_v1/data/market_data/daily/` | 6,183 files | 2000–present |
| Weekly OHLCV (stocks) | `downloadData_v1/data/market_data/weekly/` | 6,027 files | 2000–present |
| Monthly OHLCV (stocks) | `downloadData_v1/data/market_data/monthly/` | 6,028 files | 2000–present |
| Daily OHLCV (**`^YH...` industry indexes**) | `downloadData_v1/data/market_data/daily/` | **145 files** | 2020–present |
| GICS stock → industry mapping | `yf-gics/stocks_by_industry.csv` | 18,837 stocks | current snapshot |
| GICS industry → `^YH...` mapping | `yf-gics/industries.csv` | 145 industries | current |
| Scraped StockCharts stock SCTR | `stockCharts/output/query_results_*.csv` | ~503 stocks | sparse (daily snapshots) |
| Scraped StockCharts industry SCTR | `stockCharts/output/industry_summary/` | 145 industries | ~daily since Jun 2026 |
| Ticker info (market cap, sector, industry) | `metaData_v1/data/tickers/combined_info_tickers_*.csv` | ~6,349 tickers | current snapshot |

**Key gap:** `^YH...` index files exist **only in daily** — not in weekly or monthly. Monthly ROC for industry SCTR must be computed from daily data (resample to month-end).

### 2.2 Existing Code That Applies

| Module | Location | What it does |
|---|---|---|
| `calculate_scooter()` | `metaData_v1/src/basic_calculations.py:751` | SCTR-like raw score (see Section 2.3) |
| `SystematicPercentileCalculator` | `metaData_v1/src/systematic_percentile_calculator.py` | Percentile ranking (IBD-style 1–99) |
| `rs_base.py`, `rs_ibd.py`, `rs_ma.py` | `metaData_v1/src/` | RS calculation framework |
| `PPO.py` | `metaData_v1/src/indicators/PPO.py` | PPO indicator |
| `scooter_screener.py` | `metaData_v1/src/screeners/` | Screener using stSCOOTER |

### 2.3 The Scooter Family — What It Is and Why It Exists

The Scooter models are **not an attempt to clone StockCharts SCTR**. They are a separate family of Murphy-inspired signals built entirely on daily price data, designed to produce an absolute technical score without needing a peer universe.

**stSCOOTER design intent:** Capture approximately **5 months of stock behavior** using daily bars only.
- `p_roc_l = 125` days = 25 trading days/month × 5 months — this is the core lookback
- `p_ema_l = 200` provides long-term trend context (above/below 200-EMA)
- Weights heavily biased long-term (60%) to identify **trend leaders**
- Output: `clamp(50 + 2.5 * raw, 0, 99.9)` — absolute score, no ranking needed

**fastSCOOTER design intent:** Detect **earlier rotation signals** by inverting the weights (10/30/60) and compressing the lookback periods (20-day ROC, 5-day ROC, 7-period RSI). Experiments to catch momentum turning before it shows in stSCOOTER.

| Aspect | True StockCharts SCTR | stSCOOTER | fastSCOOTER |
|---|---|---|---|
| Primary lookback | 14 months (monthly bars) | **5 months** (125 daily bars) | ~1 month (20 daily bars) |
| MA type | **SMA** | EMA | EMA |
| Short component | 3-month ROC + PPO slope | PPO slope + **RSI** | PPO slope + RSI |
| Weights (L/M/S) | 45/30/25 | **60/30/10** (trend-biased) | **10/30/60** (momentum-biased) |
| Output | Percentile rank within universe | Absolute score 0–99.9 | Absolute score 0–99.9 |
| Universe | large/mid/small buckets | All stocks together | All stocks together |
| Purpose | Validated ranking vs peers | Trend leader identification | Early rotation warning |

The key architectural difference: SCTR is always **relative** (a stock's score depends on its peers). Scooter is always **absolute** (score depends only on the stock's own price history). Both are useful; they answer different questions.

---

## 3. Three-Layer Output Design

Once built, the system produces three distinct SCTR signals, each useful for different purposes:

```
Layer 1: Industry SCTR (Option A)
  Input: ^YH... daily index prices (145 industries)
  Method: SCTR formula on index → percentile rank within 145 industries
  Use: direct rotation signal, validated against StockCharts

Layer 2: Stock SCTR (StockCharts methodology)
  Input: stock daily + monthly OHLCV (6,183 stocks)
  Method: SCTR formula on each stock → rank within large/mid/small bucket
  Use: find strongest stocks within any sector/industry

Layer 3: Median Stock SCTR per GICS Industry (derived)
  Input: Layer 2 stock SCTR results + stocks_by_industry.csv mapping
  Method: group stocks by GICS industry, compute median (and percentile spread)
  Use: breadth signal — is the whole industry strengthening or just a few names?
       complements Layer 1 (index-based) with a bottom-up view
```

These three layers answer different questions:
- Layer 1: "Is the Semiconductors industry index outperforming the others?" (index-level momentum)
- Layer 2: "Which stocks within Semiconductors have the strongest technicals right now?"
- Layer 3: "Is the whole Semiconductors industry broadly healthy, or just 2–3 big names pulling up the index?"

---

## 4. Implementation Plan

### Phase 1 — Exact SCTR Formula (the engine)

**Goal:** Implement the true SCTR six-indicator formula as a reusable function, applicable to any price series (stock or index).

**New file:** `yf-gics/src/sctr_engine.py`

```python
# Inputs: daily DataFrame (Date, Close columns), monthly DataFrame (Date, Close)
# Output: dict with raw component scores + weighted total

def compute_sctr_raw(daily_df, monthly_df) -> dict:
    """
    Returns raw SCTR score for a single security.
    Caller is responsible for percentile ranking across universe.

    Components:
      c1: 14-month ROC × 0.30   (monthly data)
      c2: % from 200-day SMA × 0.15
      c3: 9-month ROC × 0.15    (monthly data)
      c4: % from 50-day SMA × 0.15
      c5: 3-month ROC × 0.10    (monthly data)
      c6: PPO(12,26,9) histogram 3-day slope × 0.15
    """
```

**Important:** For `^YH...` index files (daily-only), monthly bars are derived by resampling to month-end. For stock files, use the dedicated monthly CSVs directly.

**Key differences from stSCOOTER to fix:**
- Use **SMA** (not EMA) for 200-day and 50-day MA distance
- Use **monthly bars** for the three ROC components
- Add **3-month ROC** as var5 (not RSI)
- PPO slope stays the same (already correct in stSCOOTER)
- Remove the `50 + 2.5 * raw` transformation — output raw score only

---

### Phase 2 — Industry SCTR (Option A: `^YH...` index approach)

**Goal:** Replicate StockCharts industry SCTR exactly, then validate against scraped data.

**New file:** `yf-gics/src/sctr_industry_a.py`

```
Steps:
1. Load all 145 ^YH daily files from downloadData_v1/data/market_data/daily/
2. For each: resample to month-end → get monthly close series
3. Run compute_sctr_raw() on each → raw score per industry
4. Rank 145 raw scores → assign SCTR percentile (0.0–99.9) within the 145-industry universe
5. Join with industries.csv to attach GICS sector/industry_name labels
6. Store result: (date, industry_key, gics_sector, sctr, raw_score, rank)

Validation:
- Compare against stockCharts/output/industry_summary/ for overlapping dates
- Target: correlation > 0.90 with StockCharts industry SCTR
```

**Output table schema:**

| date | symbol | industry_key | gics_sector | sctr_raw | sctr_rank | sctr |
|---|---|---|---|---|---|---|
| 2026-06-25 | ^YH31130020 | semiconductors | technology | 142.3 | 138 | 95.2 |

**Historical backfill:** Run from 2021-01-01 (enough monthly bars for 14-month ROC). Store in database or dated CSV files.

---

### Phase 3 — Stock SCTR (StockCharts universe buckets)

**Goal:** Calculate SCTR for all 6,183 stocks, ranked within large/mid/small cap buckets using StockCharts hard limits.

**New file:** `yf-gics/src/sctr_stocks.py`

```
Steps:
1. Load market cap from combined_info_tickers_*.csv → classify each ticker:
     large:  marketCap > 10_000_000_000
     mid:    2_000_000_000 <= marketCap <= 10_000_000_000
     small:  250_000_000 <= marketCap < 2_000_000_000
     nano:   < 250_000_000  → excluded (no SCTR, same as StockCharts)

2. For each stock: load daily CSV + monthly CSV → compute_sctr_raw()

3. For each bucket (large/mid/small): rank stocks by raw score within bucket
   → assign SCTR 0.0–99.9

4. Store result: (date, ticker, group, gics_industry_key, sctr_raw, sctr)

Validation:
- Compare against stockCharts/output/query_results_*.csv
- Check rank correlation per bucket per date
```

**Notes:**
- Market cap in CSVs is a current snapshot, not historical. For backtesting, approximate:
  `hist_market_cap ≈ shares_outstanding (from tickers CSV) × daily_close`
  Shares outstanding changes slowly; this is acceptable for rotation analysis.
- For the live (latest date) run, use the `marketCap` column from the daily CSV rows directly.

---

### Phase 4 — Median Stock SCTR per GICS Industry (derived Layer 3)

**Goal:** After Phase 3, aggregate stock SCTR to the GICS industry level for breadth analysis.

**New file:** `yf-gics/src/sctr_industry_breadth.py`

```
Steps:
1. Load Phase 3 output (stock SCTR for all dates)
2. Join with stocks_by_industry.csv on ticker → get industry_key per stock
3. Per (date, industry_key): compute:
     - median_sctr       → the typical stock in this industry
     - mean_sctr         → pulled by outliers (big caps), useful signal
     - pct_above_75      → % of stocks with SCTR > 75 (breadth strength)
     - pct_above_50      → % of stocks with SCTR > 50 (breadth neutral line)
     - pct_below_25      → % of stocks with SCTR < 25 (breadth weakness)
     - count             → number of stocks in universe for this industry

Output: (date, industry_key, median_sctr, mean_sctr, pct_above_75, pct_above_50, pct_below_25, count)
```

**Why this complements Option A (Layer 1):**

> Example: The Semiconductors index (`^YH31130020`) has SCTR = 85 (Layer 1) — the index
> is strong. But median stock SCTR in semiconductors is only 48 (Layer 3) — most stocks
> are average. This divergence reveals the index is being pulled up by 1–2 mega-caps
> (NVDA, TSMC) while the rest of the industry is not participating.
> This is a rotation warning: the index strength is fragile.

---

### Phase 5 — Option B: Synthetic GICS Industry Index

**Goal:** Build market-cap-weighted GICS industry price series from constituent stocks, then apply SCTR formula. This aligns the industry SCTR with your own stock universe (not Dow Jones composition).

**New file:** `yf-gics/src/sctr_industry_b.py`

```
Steps:
1. For each GICS industry_key in stocks_by_industry.csv:
   a. Get constituent tickers
   b. For each constituent and each date: load daily close + marketCap
   c. Compute weight_i = marketCap_i / sum(marketCap_all_constituents)
   d. synthetic_index_close = sum(weight_i * close_i)
   e. Resample synthetic daily to monthly for ROC components

2. Run compute_sctr_raw() on each synthetic index series

3. Rank 145+ GICS industries by raw score → GICS Industry SCTR B

Challenges:
- Stocks with missing data in a given period need handling (exclude or backfill)
- New stocks enter the industry over time → index composition changes
- Market cap for weighting: use close × shares_outstanding approximation

Start after Option A is validated, as Option A provides a baseline for comparison.
```

---

### Phase 6 — Dashboard Integration

**Extend:** `RRGs-Dashboard` and/or build standalone `yf-gics/dashboard/`

```
Panel 1: Industry SCTR Table (Layer 1)
  - All 145 industries, sorted by SCTR descending
  - Color: green > 75, yellow 40–75, red < 40
  - Show: current SCTR, 1-week change, 4-week trend (sparkline)
  - Filter by GICS sector

Panel 2: Industry Breadth (Layer 3)
  - For selected industry: show median_sctr + pct_above_75 over time
  - Flag divergences: index SCTR high but median SCTR low

Panel 3: Stock SCTR Leaderboard (Layer 2)
  - Top 20 stocks by SCTR in selected industry
  - Color by cap bucket (large/mid/small)

Panel 4: Rotation Strength Matrix
  - Rows: GICS sectors
  - Columns: weekly change in industry SCTR, 4-week change, 12-week change
  - Spot accelerating rotation in/out early

Panel 5: Correction Leader Filter (from literature_review_rotation.md)
  - Activated when benchmark is down >5% over N days
  - Shows stocks with closing range >60% + MA adherence + SCTR held
```

---

## 5. Validation Strategy

Before trusting the computed SCTR for analysis, validate each layer:

```
Layer 1 (Industry SCTR Option A):
  - Compare: computed industry SCTR vs stockCharts/output/industry_summary/2026-06-25.csv
  - Metric: Spearman rank correlation ≥ 0.90

Layer 2 (Stock SCTR):
  - Compare: computed stock SCTR vs stockCharts/output/query_results_2026-06-15.csv
  - Metric: Spearman rank correlation ≥ 0.85 per bucket (large/mid/small)
  - Acceptable gap: SMA vs EMA, snapshot timing differences

Layer 3 (Median SCTR breadth):
  - No external benchmark; validate internal consistency:
    * Industry with median_sctr > 70 should have pct_above_75 > 40%
    * Industry at ATH on monthly should have high median_sctr
```

---

## 6. Build Order

```
Week 1:  sctr_engine.py (Phase 1) — core formula, unit-tested against known AAPL SCTR values
Week 2:  sctr_industry_a.py (Phase 2) — 145 ^YH indexes, validate vs scraped data
Week 3:  sctr_stocks.py (Phase 3) — 6,183 stocks, large/mid/small buckets
Week 4:  sctr_industry_breadth.py (Phase 4) — derived breadth signal
Week 5+: Dashboard integration (Phase 6), then Option B (Phase 5)
```

---

## 7. Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| SMA vs EMA for MA distance | **SMA** | Matches StockCharts exactly |
| Monthly ROC source for `^YH...` | Resample daily to month-end | YH files only exist in daily |
| Monthly ROC source for stocks | Use monthly CSVs directly | Cleaner, already downloaded |
| Market cap universe classification | Current snapshot from CSV | Good enough for rotation analysis; historical cap adds little value |
| Percentile ranking method | Reuse `SystematicPercentileCalculator` | Already tested, avoids duplication |
| Historical backfill start date | 2021-01-01 | Needs 14 months of monthly bars; daily data starts Jan 2020 |
| Database vs CSV storage | Extend `stockCharts/stockcharts.db` or separate `sctr.db` | Keep scraped and computed separate for validation |

---

## 8. The Full Signal Family — How Everything Fits Together

Three distinct but complementary signal families. Nothing is being replaced.

### 8.1 Scooter Family (Murphy-inspired, daily-only, absolute score)

Built from daily price data, no peer universe needed, output is an absolute 0–99.9 score.

| Variant | Weights (L/M/S) | Core lookback | Design intent |
|---|---|---|---|
| **stSCOOTER** | 60 / 30 / 10 | 5 months (125-day ROC) | Identify trend leaders; the "slow" signal |
| **fastSCOOTER** | 10 / 30 / 60 | ~1 month (20-day ROC) | Catch rotation momentum early; the "fast" signal |

When stSCOOTER and fastSCOOTER are both high and rising: strong confirmation of leadership.
When fastSCOOTER rises before stSCOOTER: early warning — a stock (or industry) may be entering rotation.
When fastSCOOTER falls while stSCOOTER is still high: possible early exit signal from a leader.

Code: `metaData_v1/src/basic_calculations.py:751` (`calculate_scooter(variant='st'|'fast')`)

### 8.2 True SCTR Family (StockCharts methodology, percentile rank, peer-relative)

Built from daily + monthly price data, ranked within a cap-bucket universe. Output is a percentile rank 0–99.9 within peers.

| Layer | Input | Output |
|---|---|---|
| **Layer 1: Industry SCTR** | ^YH... index daily (resampled to monthly) | SCTR rank within 145 industries |
| **Layer 2: Stock SCTR** | Stock daily + monthly OHLCV | SCTR rank within large / mid / small bucket |
| **Layer 3: Industry Breadth** | Layer 2 aggregated by GICS industry | Median SCTR + % above 75/50/25 per industry |

Code: to be built in `yf-gics/src/` (see Phases 1–5 above).

### 8.3 How to Use Them Together

```
Scenario: Technology sector — is the recent pullback a rotation out, or noise?

1. Industry SCTR (Layer 1): Semiconductors at SCTR 72, 2 weeks ago was 88 → falling but not lagging yet
2. stSCOOTER of SOXX (semis ETF): score dropped from 81 to 68 → 5-month trend weakening
3. fastSCOOTER of SOXX: dropped sharply 2 weeks ago, now at 31 → momentum already negative
4. Industry Breadth (Layer 3): median stock SCTR fell from 74 to 58; pct_above_75 dropped from 55% to 28%
5. RRG (weekly): Semiconductors moving from Leading toward Weakening

Conclusion: fastSCOOTER led the signal (rotation warning), stSCOOTER confirmed it,
SCTR and RRG confirm the rotation is real, breadth shows it's broad (not just 1–2 stocks).
This is a Level 3–4 rotation (structural, not noise).
```
