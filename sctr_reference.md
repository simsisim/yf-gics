# SCTR Reference: Original Formula and Project Implementation

*Last updated: 2026-06-29*

---

## 1. The Original StockCharts SCTR Formula

Source: https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/stockcharts-technical-rank-sctr

SCTR (StockCharts Technical Rank) was designed by John Murphy and published by StockCharts. It is a **weighted composite of six technical indicators**, all using **daily bars only**, converted to a **percentile rank** within a universe of peers.

### 1.1 Six-Indicator Formula

| # | Key | Component | Calculation | Weight |
|---|---|---|---|---|
| 1 | `c_ema200_pct` | % above/below 200-day **EMA** | `(close − EMA200) / EMA200 × 100` | **30%** |
| 2 | `c_roc125` | 125-day Rate of Change | `(close_t − close_{t−125}) / close_{t−125} × 100` | **30%** |
| 3 | `c_ema50_pct` | % above/below 50-day **EMA** | `(close − EMA50) / EMA50 × 100` | **15%** |
| 4 | `c_roc20` | 20-day Rate of Change | `(close_t − close_{t−20}) / close_{t−20} × 100` | **15%** |
| 5 | `c_rsi14` | 14-day RSI | Wilder smoothing RSI, range 0–100 | **5%** |
| 6 | `c_ppo_slope` | PPO(12,26,9) histogram 3-day slope | Conditional normalization → 0–100 (see below) | **5%** |
| | | **Total** | | **100%** |

**Group weights:** Long-term 60% (1+2) / Medium-term 30% (3+4) / Short-term 10% (5+6)

### 1.2 Normalization — Raw Values × Weight (NOT per-component ranking)

ChartSchool states explicitly: *"raw numbers are multiplied by the weighting to calculate the indicator score."*

Example: a stock 15% above its 200-day EMA contributes `15 × 0.30 = 4.5 points`.

```
raw_score = c_ema200_pct × 0.30
          + c_roc125     × 0.30
          + c_ema50_pct  × 0.15
          + c_roc20      × 0.15
          + c_rsi14      × 0.05
          + c_ppo_slope  × 0.05
```

Then all raw scores in the universe are percentile-ranked → SCTR (0.0–99.9).

> Note: some third-party articles show a "double-ranking" approach (percentile-rank each component first, then weight and sum). **That is not what StockCharts does.** Raw values are used directly.

### 1.3 PPO Slope — Conditional Normalization

The slope `= (PPO_histogram[-1] − PPO_histogram[-4]) / 3` is normalized to 0–100 before being multiplied by 0.05:

| Slope condition | Score |
|---|---|
| slope ≥ +1 | 100 (full 5 points) |
| slope ≤ −1 | 0 (no contribution) |
| −1 < slope < +1 | `(slope + 1) × 50` → linear interpolation |

### 1.4 Two-Step Output — Raw Score → Percentile Rank

SCTR is **never** an absolute score. The pipeline has two steps:

```
Step 1: Compute raw_score for each security (formula above)

Step 2: Rank all securities within their universe by raw_score
  SCTR = percentile_rank × 99.9   →   range: 0.0 to 99.9
```

A stock's SCTR can change even if its own technicals are unchanged — because peers changed their raw scores.

### 1.5 Universe Buckets (Stocks Only)

StockCharts divides stocks into three universes by market cap. Each stock is **ranked only within its own bucket**:

| Bucket | Market Cap | Notes |
|---|---|---|
| Large-cap | > $10B | ~266 stocks on StockCharts (Jun 2026) |
| Mid-cap | $2B – $10B | ~178 stocks |
| Small-cap | $250M – $2B | ~59 stocks |
| Nano-cap | < $250M | **Excluded — no SCTR assigned** |

### 1.6 Industry SCTR — How StockCharts Does It

StockCharts applies the **identical six-indicator formula** to the **Dow Jones US Industry Index** price series (`$DJUS...` / `^YH...` symbols). Each of the 145 industry indexes is treated as if it were a stock, and the 145 raw scores are ranked against each other.

**This is NOT an average or median of member stocks' SCTRs.** It is computed on the index price series itself.

### 1.7 Key Implementation Details

| Detail | Value |
|---|---|
| MA type | **Exponential moving average (EMA)**, not SMA |
| ROC lookbacks | 125-day and 20-day (daily bars only) |
| Monthly bars needed | **None — all components use daily bars** |
| PPO parameters | EMA(12), EMA(26), Signal EMA(9) |
| PPO slope window | 3 days: `(hist[-1] − hist[-4]) / 3`, then normalized 0–100 |
| Minimum daily bars | 210 (200-day EMA convergence + buffer) |

---

## 2. SCTR Models Implemented in This Project

Three distinct families of scoring exist. Only the first is the true StockCharts SCTR.

---

### 2.1 True SCTR Family (StockCharts Methodology)

These modules implement the exact StockCharts formula. Output is a **percentile rank (0.0–99.9)** relative to peers.

#### Core Engine — `src/sctr_engine.py`

The shared formula. No I/O, no ranking — just computes one raw score for one security.

```python
compute_indicators(daily_df) → dict | None
```
Returns `{c_ema200_pct, c_roc125, c_ema50_pct, c_roc20, c_rsi14, c_ppo_slope, raw_score}` or `None` if data is insufficient. All components use daily bars only — no monthly data.

```python
rank_to_sctr(raw_scores: pd.Series) → pd.Series
```
Converts a Series of raw scores to percentile ranks 0.0–99.9.

---

#### Industry SCTR — Option A: `src/sctr_industry.py`

**Method:** Applies the SCTR formula to the **145 `^YH` Dow Jones Industry Index** price series.
Replicates what StockCharts shows for industry SCTR exactly.

**How to run:**
```python
from src.sctr_industry import run, save
df = run(config)          # compute SCTR for 145 industries
save(df, config)          # → results/industry_sctr/industry_sctr_YYYY-MM-DD.csv
```

**Output columns:** `symbol, industry_key, gics_sector, industry_name, c_ema200_pct, c_roc125, c_ema50_pct, c_roc20, c_rsi14, c_ppo_slope, raw_score, sctr, rank`

**Validation target:** Spearman correlation ≥ 0.90 vs StockCharts scraped industry SCTR.

---

#### Industry SCTR — Option B: `src/sctr_industry_gics.py`

**Method:** Builds a **market-cap-weighted synthetic price index** from constituent stocks for each GICS industry (145 industries), then applies the SCTR formula to each synthetic index.

No dependency on Dow Jones indexes — purely GICS-based. Useful when `^YH` coverage is incomplete or you want the industry SCTR to reflect your own stock universe.

**How to run:**
```python
from src.sctr_industry_gics import run, save
df = run(config)          # compute SCTR for all GICS industries
save(df, config)          # → results/industry_sctr/industry_sctr_gics_YYYY-MM-DD.csv
```

**Output columns:** `industry_key, sector_key, sector_name, industry_name, constituent_count, c_ema200_pct, c_roc125, c_ema50_pct, c_roc20, c_rsi14, c_ppo_slope, raw_score, sctr, rank`

**When to use Option A vs Option B:**

| | Option A (`sctr_industry.py`) | Option B (`sctr_industry_gics.py`) |
|---|---|---|
| Input | `^YH` Dow Jones index prices | Constituent stocks aggregated |
| Matches StockCharts | Yes (validated) | No (different composition) |
| Reflects your universe | No | Yes |
| Speed | Fast | Slow (loads all constituent stocks) |

---

#### Stock SCTR — `src/sctr_stocks.py`

**Method:** Applies the SCTR formula to all universe stocks, then ranks within large/mid/small cap buckets.

**How to run:**
```python
from src.sctr_stocks import run, save
df = run(config)          # score all stocks
save(df, config)          # → results/stock_sctr/stock_sctr_{large|mid|small}_YYYY-MM-DD.csv
```

**Output columns:** `ticker, cap_bucket, market_cap, c_ema200_pct, c_roc125, c_ema50_pct, c_roc20, c_rsi14, c_ppo_slope, raw_score, sctr, rank_in_bucket`

Universe source is controlled by `Config.universe_source`:
- `'sc'` — StockCharts DB universe (~5,200 stocks, with StockCharts group labels)
- `'tv'` — TradingView CSV (~3,300 stocks, group labels from market-cap thresholds)

---

### 2.2 Scooter Family (Murphy-Inspired, Absolute Score — NOT True SCTR)

Located in `metaData_v1/src/basic_calculations.py:751` (`calculate_scooter()`).

These are **not** an attempt to clone StockCharts SCTR. They are a separate signal family built entirely on **daily bars only**, producing an **absolute score** with no peer universe. They were named "Scooter" as an internal shorthand.

| Variant | Weights (L/M/S) | Core lookback | Key difference from SCTR |
|---|---|---|---|
| **stSCOOTER** | 60 / 30 / 10 | 125 days (~5 months, daily bars) | No monthly bars; uses EMA not SMA; trend-biased weights; absolute score |
| **fastSCOOTER** | 10 / 30 / 60 | 20 days (~1 month, daily bars) | Momentum-biased inverse weights; catches rotation earlier |

**Output:** `clamp(50 + 2.5 × raw, 0, 99.9)` — an absolute value, no ranking step.

**When to use Scooter vs True SCTR:**

| Question | Use |
|---|---|
| Is this stock outperforming its peers right now? | True SCTR (peer-relative) |
| Is this stock technically strong on its own merits? | stSCOOTER (absolute) |
| Is this stock starting to rotate in early? | fastSCOOTER (momentum-biased) |

---

### 2.3 Short-Term SCTR (ST-SCTR) — Used in `src/momentum_screen.py`

ST-SCTR is **not** a separate module — it is computed inside the Momentum Screen as a faster-reacting derivative of the full SCTR. It isolates the **true short-term bucket** of the ChartSchool formula (the bottom 10%) and ranks it independently.

**Formula:**

| Component | Key | Full SCTR weight | ST-SCTR weight |
|---|---|---|---|
| 14-day RSI | `c_rsi14` | 5% | **50%** |
| PPO(12,26,9) histogram 3-day slope | `c_ppo_slope` | 5% | **50%** |

```
sctr_st_raw = c_rsi14 × 0.50 + c_ppo_slope × 0.50
sctr_st     = percentile_rank(sctr_st_raw, all industries) → 0–99.9

st_lead   = sctr_st − sctr_full
st_signal = "ST Leading"  if st_lead ≥ +10  (early rotation entry forming)
            "ST Lagging"  if st_lead ≤ −10  (short-term fading, early warning)
            "flat"        otherwise
```

**Why `c_rsi14` + `c_ppo_slope` (not `c_roc20`):**
The 20-day ROC (`c_roc20`) carries 15% weight in the full SCTR under the **medium-term** tier. Using it in ST-SCTR would mix timeframes. The RSI-14 and PPO slope are the only indicators the ChartSchool formula classifies as short-term (the 10% tier). Both carry equal weight there, so 50/50 is the natural split.

**Role in the Momentum Screen composite:**
```
momentum_score = rs_pct_composite × 0.40
               + sctr_full        × 0.30
               + sctr_st          × 0.20   ← ST-SCTR
               + mg_pct           × 0.10
```

---

## 3. Signal Interaction — How to Read All Three Together

```
fastSCOOTER rises first        → early warning: momentum is turning
stSCOOTER confirms             → 5-month trend is joining the move
Industry SCTR (Layer 1) rises  → the whole industry index is strengthening
Stock SCTR rises               → individual stock confirmed as peer leader

Divergence example:
  Industry SCTR = 85  (index strong)
  Median stock SCTR = 48  (most stocks average)
  → Index is being pulled by 1-2 mega-caps. Rotation may be fragile.
```

---

## 4. Output Locations

| Module | Output path |
|---|---|
| `sctr_industry.py` | `results/industry_sctr/industry_sctr_YYYY-MM-DD.csv` |
| `sctr_industry_gics.py` | `results/industry_sctr/industry_sctr_gics_YYYY-MM-DD.csv` |
| `sctr_stocks.py` | `results/stock_sctr/stock_sctr_{large\|mid\|small}_YYYY-MM-DD.csv` |
| stSCOOTER / fastSCOOTER | `metaData_v1/` (separate project) |
