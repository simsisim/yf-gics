# GICS Industry Rotation Monitor — Stock Pipeline Architecture

> Date: 2026-06-29  
> Scope: Individual stock calculations only. For industry-level signals see `architecture_industry.md`.

---

## 1. No Synthetic Index — Ever

Individual stock calculations are always computed directly from each stock's own daily OHLCV file. There is no synthetic or ^YH index involved at any stage:

- Stock SCTR → computed from the stock's own daily close
- Stock stage → computed from the stock's own daily SMA structure
- Stock RS → stock/SPY ratio directly
- Industry filter → uses the momentum screen's Faber signal (which is already ^YH-based)

The only "index" a stock sees is **SPY** as the benchmark for RS calculations.

---

## 2. Stock SCTR

**File:** `src/sctr_stocks.py`  
**Engine:** `src/sctr_engine.py` — same formula as industry SCTR.

### Universe Buckets

```
Large cap:  market_cap >  $10B    ranked within large universe
Mid cap:    market_cap $2B–$10B   ranked within mid universe
Small cap:  market_cap $250M–$2B  ranked within small universe
```

Ranking within bucket avoids large-cap dominance. A small-cap stock's SCTR measures it against small-cap peers, not against Apple.

### Formula

Identical to industry SCTR — applied to the stock's own daily close series:

```
raw_score = c_ema200_pct × 0.30 + c_roc125 × 0.30 +
            c_ema50_pct  × 0.15 + c_roc20  × 0.15 +
            c_rsi14      × 0.05 + c_ppo_slope × 0.05

SCTR = percentile_rank(raw_score, bucket_universe) × 99.9
```

**Minimum:** 210 daily bars (EMA200 stabilization).

**Output:** `results/stocks_sctr/stock_sctr_large_YYYY-MM-DD.csv` etc.

---

## 3. Stock Screener

**File:** `src/stock_screener.py`  
**Output:** `results/stock_screener_YYYY-MM-DD.csv`

Drills down from leading industries into individual stocks. Runs after the full index pipeline.

### Step 1 — Industry Filter (^YH-based)

Only stocks in industries with Faber signal **STRONG BUY** or **BUY** are considered.

```
Faber signal comes from: momentum_screen_YYYY.csv  (^YH, non-synthetic)
Filter: faber_signal in {'STRONG BUY', 'BUY'}
```

In the 2026-06-29 run: 24 industries passed → 2,728 stocks screened.

### Step 2 — Stock-Level Filters

```
1. Stock must have daily OHLCV data (≥ 210 bars)
2. Stage must be 2A or 2B   (MA structure from stock's own price)
3. Minervini score ≥ 3/7    (criteria computed from stock's own price)
```

### Step 3 — Composite Ranking Score (0–100)

```
composite = rs_pct       × 0.35   (12-month RS vs SPY, ranked within screened universe)
           + stage_score × 0.30   (2B=100, 2A=85, 2C=70…)
           + (minervini/7 × 100) × 0.15
           + sctr        × 0.10   (stock SCTR from sctr_stocks.py)
           + (cr_score/5 × 100)   × 0.10   (IBD closing range score, 0 if missing)
```

Note: `rs_pct` here is the stock's individual RS percentile vs SPY, ranked within the screened universe — not the industry RS percentile.

### Stage Analysis for Stocks

Stage and Minervini criteria are computed directly from each stock's daily closes in the screener loop (not pre-computed separately). The same `_compute()` function used by `stage_analysis.py` for industries is called per stock.

```
C1: price > 150-day SMA
C2: price > 200-day SMA
C3: 150-SMA > 200-SMA
C4: 200-SMA slope > 0
C5: price ≥ 30% above 52-week low
C6: price within 25% of 52-week high
C7: stock RS percentile ≥ 70   (vs screened universe)
```

---

## 4. IBD Closing Range (Correction Leaders)

**File:** `src/closing_range.py`  
**Output:** `results/closing_range_YYYY-MM-DD.csv`

Identifies stocks that held up well during a market correction — O'Neil / IBD method.

### Inputs

```
--correction-start YYYY-MM-DD   (or auto-detected from market_clock)
--correction-end   YYYY-MM-DD   (or date of Follow-Through Day)
```

### Five Criteria (F1–F5)

```
F1: Closing range > 50%   (close near high of day, consistently)
    cr_score = (close - low) / (high - low)  averaged over correction period
F2: Above 200-SMA at correction end
F3: RS percentile ≥ 60 at correction end
F4: Volume > 20-day avg on up days during correction (institutional accumulation)
F5: Price within 30% of pre-correction high (not deeply damaged)

leader_score = count(F1–F5 passed)   → 0–5
```

A score ≥ 4 indicates a stock that institutions were accumulating while the market fell — these are the first stocks to break out when the uptrend resumes.

**Output:** filtered by `--min-score` (default 3). Optionally filtered to STRONG BUY / BUY industries with `--top-industries-only`.

---

## 5. Full Data Flow (Index → Stock)

```
^YH daily/weekly
    │
    ▼
[Index Pipeline — see architecture_industry.md]
    │
    ├─ momentum_screen_YYYY.csv   → industry Faber signals (STRONG BUY / BUY)
    │                                         ↓
    │                              stock_screener.py: filter by industry
    │
    ├─ market_clock_YYYY.csv      → market regime context
    │                                         ↓
    │                              closing_range.py: correction leader filter
    │
Stock daily CSVs (individual stocks)
    │
    ├─ sctr_stocks.py → stock SCTR by cap bucket
    │
    └─ stock_screener.py:
         - stage + Minervini from stock price
         - RS vs SPY from stock price
         - composite score = RS×0.35 + stage×0.30 + Minervini×0.15 + SCTR×0.10 + CR×0.10
         → stock_screener_YYYY.csv
```

---

## 6. Execution

```bash
# ── Composite modes ───────────────────────────────────────────────────────────
python main.py --mode update    # everything: industry pipeline + stocks pipeline
python main.py --mode stocks    # all stock steps: stock-sctr → stock-screener

# ── Individual stock steps ────────────────────────────────────────────────────
python main.py --mode stock-sctr        # stock SCTR by cap bucket
python main.py --mode stock-screener    # ranked stocks within top industries
python main.py --mode closing-range     # IBD correction leader filter

# ── Options ───────────────────────────────────────────────────────────────────
python main.py --mode stock-screener --top-industries-only --min-score 4
python main.py --mode closing-range --correction-start 2026-03-19 --correction-end 2026-04-08 --min-score 3
```

**Note:** stocks pipeline depends on the industry pipeline having run first — `--mode update` handles the correct order automatically.
