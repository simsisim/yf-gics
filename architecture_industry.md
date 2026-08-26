# GICS Industry Rotation Monitor — Architecture Report (Index Pipeline)

> Date: 2026-06-29  
> Model: `yf-gics` Python project  
> Scope: Industry-level signals only. For individual stock calculations see `architecture_stocks.md`.

---

## 1. Index Source — `^YH` Dow Jones Industry Indexes

All industry calculations default to the **published ^YH Dow Jones Industry Index** files.

```
downloadData_v1/data/market_data/
  daily/    ^YH31130020.csv  ^YH20101010.csv  ...  (145 files)
  weekly/   ^YH31130020.csv  ^YH20101010.csv  ...  (145 files)
```

These are the same price series StockCharts uses to compute industry SCTR. They are authoritative.

Each `^YH` ticker is mapped from `industries.csv` → `symbol` column.

**`source='synth'` (legacy research mode):** builds a market-cap-weighted daily/weekly index from constituent stock closes. Produces identical signal types but on a different price series. Used only for research comparison (`--mode momentum-compare`). Synth outputs go to `results/synth/`.

---

## 2. Data Flow

```
^YH daily/weekly CSVs  (145 industry indexes)
        │
        ├──▶ sctr_industry.py  → industry SCTR (daily close, all 6 components)
        │        industry_sctr_YYYY-MM-DD.csv
        │
        ├──▶ rrg_engine.py     → tactical (5/10/3) + trend (10/30/7) RRG profiles
        │        rrg_industry_YYYY-MM-DD.csv
        │
        ├──▶ rs_percentile.py  → RS composite rating 1–99
        │        rs_percentile_YYYY-MM-DD.csv
        │
        ├──▶ stage_analysis.py → Weinstein/Minervini stage + 7 criteria
        │        stage_analysis_YYYY-MM-DD.csv
        │
        ├──▶ ath_monitor.py    → all-time-high proximity (monthly resampled)
        │        ath_monitor_YYYY-MM-DD.csv
        │
        ├──▶ rs_ma_signals.py  → RS line vs 13-week MA (reads backfill history)
        │        rs_ma_signals_YYYY-MM-DD.csv
        │
        ├──▶ momentum_screen.py → Faber + Stage 6-level signal + composite score
        │        reads: rs_percentile + industry_sctr + stage_analysis
        │        momentum_screen_YYYY-MM-DD.csv
        │
        └──▶ rotation_severity.py → master severity label (6-component composite)
                 reads: industry_history.csv (backfill) + all above CSVs
                 rotation_severity_YYYY-MM-DD.csv
```

**Parallel reads:** `backfill.py` (separate step) builds `results/history/industry_history.csv` — a weekly time-series of SCTR + RRG for all industries. Severity depends on this for its 4 history-based components.

---

## 3. SCTR Engine

**File:** `src/sctr_engine.py`  
**Role:** Pure computation — no I/O. Called by `sctr_industry.py` (^YH) and `sctr_stocks.py` (individual stocks).

### Formula (StockCharts ChartSchool)

| Component     | Key            | Indicator                                       | Weight  |
| ------------- | -------------- | ----------------------------------------------- | ------- |
| Long-term 1   | `c_ema200_pct` | % above/below 200-day EMA                       | **30%** |
| Long-term 2   | `c_roc125`     | 125-day Rate of Change                          | **30%** |
| Medium-term 1 | `c_ema50_pct`  | % above/below 50-day EMA                        | **15%** |
| Medium-term 2 | `c_roc20`      | 20-day Rate of Change                           | **15%** |
| Short-term 1  | `c_rsi14`      | 14-day RSI (Wilder smoothing)                   | **5%**  |
| Short-term 2  | `c_ppo_slope`  | PPO(12,26,9) histogram 3-day slope (normalized) | **5%**  |

**Timeframe breakdown: 60% long / 30% medium / 10% short. All daily bars — no monthly data needed.**

### Component Formulas

```
c_ema200_pct = (close - EMA_200) / EMA_200 × 100
c_roc125     = (close_t - close_{t-125}) / close_{t-125} × 100
c_ema50_pct  = (close - EMA_50)  / EMA_50  × 100
c_roc20      = (close_t - close_{t-20}) / close_{t-20} × 100
c_rsi14      = 100 - 100 / (1 + avg_gain / avg_loss)   [Wilder EWM, alpha=1/14]

PPO         = (EMA_12 - EMA_26) / EMA_26 × 100
Signal      = EMA_9(PPO)
Histogram   = PPO - Signal
slope       = (hist[-1] - hist[-4]) / 3

c_ppo_slope = 100.0         if slope ≥ +1.0
              0.0            if slope ≤ −1.0
              (slope+1)×50   otherwise          [ChartSchool conditional]
```

### Normalization

```
raw_score = c_ema200_pct × 0.30 + c_roc125 × 0.30 +
            c_ema50_pct  × 0.15 + c_roc20  × 0.15 +
            c_rsi14      × 0.05 + c_ppo_slope × 0.05

SCTR = percentile_rank(raw_score, universe) × 99.9    → [0.0, 99.9]
```

Single final percentile rank (not per-component). Universe = all 145 ^YH industries ranked against each other.

**Output:** `results/industry_sctr/industry_sctr_YYYY-MM-DD.csv`  
**Minimum bars:** 210 daily (EMA200 stabilization).

---

## 4. RRG Engine

**File:** `src/rrg_engine.py`  
**Role:** Positions each industry in Relative Rotation Graph space vs SPY. Two weekly profiles computed in a single run.

### Core Formula — MACD Trend-Following

```
Step 1: Relative Strength
  RS = industry_weekly_close / SPY_weekly_close

Step 2: MACD (normalized)
  SMA_fast = RS.rolling(fast_window).mean()
  SMA_slow = RS.rolling(slow_window).mean()
  MACD     = (SMA_fast - SMA_slow) / SMA_slow × 100   ← % deviation

Step 3: RS-Ratio  (centered at 100)
  RS-Ratio = MACD × factor + 100
  > 100: fast SMA above slow SMA = outperforming trend

Step 4: Smooth RS-Ratio
  RS-Ratio_s = SMA(RS-Ratio, ratio_smooth)

Step 5: RS-Momentum  (first-difference, centered at 100)
  RS-Momentum = (RS-Ratio_s − RS-Ratio_s.shift(roc_period)) × factor + 100
  > 100: RS-Ratio is rising = momentum positive
```

*Model: MACD Trend-Following, validated against Julius de Kempenaer's published RRGs. Z-score normalization was tested and discarded.*

### Quadrant Assignment

```
RS-Ratio ≥ 100  AND  RS-Momentum ≥ 100  →  Leading    (strong, gaining)
RS-Ratio ≥ 100  AND  RS-Momentum < 100  →  Weakening  (strong, fading)
RS-Ratio < 100  AND  RS-Momentum < 100  →  Lagging    (weak, losing)
RS-Ratio < 100  AND  RS-Momentum ≥ 100  →  Improving  (weak, recovering)
```

Clockwise rotation (Improving → Leading → Weakening → Lagging → Improving) = bullish cycle.

### Two Weekly Profiles

Both run on weekly ^YH bars vs SPY. Output lands in one CSV file.

|                | Tactical                              | Trend                                       |
| -------------- | ------------------------------------- | ------------------------------------------- |
| `fast_window`  | 5 w                                   | 10 w                                        |
| `slow_window`  | 10 w                                  | 30 w                                        |
| `ratio_smooth` | 3 w                                   | 7 w                                         |
| Signal speed   | 1–3 weeks                             | 4–8 weeks (monthly-equivalent)              |
| Output columns | `rs_ratio`, `rs_momentum`, `quadrant` | `rs_ratio_m`, `rs_momentum_m`, `quadrant_m` |
| Role           | Rotation timing                       | Structural confirmation                     |

**Key insight:** Tactical Weakening + Trend Leading = pullback inside uptrend, not a structural rotation. Both timeframes must agree for high-conviction signals.

**Date normalization:** SPY weekly files use Saturday-labeled bars, ^YH uses Monday labels. `compute_rrg_macd()` normalizes both to Monday-of-week before alignment.

**Output:** `results/rrg/rrg_industry_YYYY-MM-DD.csv`  
**Minimum:** 60 weekly bars (covers Trend profile slow=30 + roc=6 + ratio_smooth=7 + buffer).

---

## 5. RS Percentile

**File:** `src/rs_percentile.py`  
**Input:** ^YH daily close vs SPY daily close.

### Formula

```
12 months → 4 non-overlapping quarterly windows (252 trading days):

  Q4 (0–63 days ago)     weight 2×   ← most recent quarter
  Q3 (63–126 days ago)   weight 1×
  Q2 (126–189 days ago)  weight 1×
  Q1 (189–252 days ago)  weight 1×

  excess_return_Qn = (industry_return - SPY_return) over that window

  composite = (Q4×2 + Q3 + Q2 + Q1) / 5

  rs_pct_composite = percentile_rank(composite, 145 industries) → 1–99
```

Heavy front-weighting (Q4 = 40%) emphasizes recent outperformance.

### RS Line and New Highs

```
RS_line = industry_close / SPY_close   (normalized to 1.0 at start)

rs_new_high      = RS_line ≥ 99% of 52-week peak  (within 1%)
rs_near_new_high = RS_line ≥ 97% of 52-week peak  (within 3%)
```

Minervini's confirmation: RS line making new highs before price breakout = highest-conviction entry timing.

**Output:** `results/rs_percentile_YYYY-MM-DD.csv`

---

## 6. Stage Analysis (Weinstein / Minervini)

**File:** `src/stage_analysis.py`  
**Input:** ^YH daily close. Uses 50/150/200-day SMAs.

### Stage Definitions

| Stage | Description                                               | Score |
| ----- | --------------------------------------------------------- | ----- |
| 2B    | All MAs rising and aligned (50 > 150 > 200, all slope up) | 100   |
| 2A    | Early uptrend — golden cross formed, 200-SMA turning up   | 85    |
| 2C    | Late uptrend — 200-SMA slope flattening                   | 70    |
| 1     | Basing — price flat near 200-SMA                          | 45    |
| 3     | Distribution / Topping — 50-SMA crossing below 200-SMA    | 25    |
| 4     | Decline — price below declining 200-SMA                   | 5     |

### Minervini 7 Criteria (0–7 score)

```
C1: price > 150-day SMA
C2: price > 200-day SMA
C3: 150-SMA > 200-SMA
C4: 200-SMA slope > 0  (trending up ≥ 4 weeks)
C5: price ≥ 30% above 52-week low
C6: price within 25% of 52-week high
C7: RS Percentile composite ≥ 70   ← loaded from rs_percentile output
```

**Output:** `results/stage_analysis_YYYY-MM-DD.csv`

---

## 7. ATH Monitor

**File:** `src/ath_monitor.py`  
**Input:** ^YH daily close, resampled to month-end.

### ATH Score Map

```
New ATH (at or above previous peak)               → score = 100
Holding Breakout (≤ 3 months ago, > prev×0.985)   → score = 85
Near ATH (within 3% of ATH)                       → score = 70
Recovering (3–15% below ATH, ≤ 12 months ago)     → score = 40 + (15 + pct_from_ath)/15 × 20
Below ATH (>15% below, or very old ATH)            → score = max(0, 20 + pct_from_ath)
```

**Output:** `results/ath_monitor_YYYY-MM-DD.csv`

---

## 8. RS-MA Signals

**File:** `src/rs_ma_signals.py`  
**Input:** `results/history/industry_history.csv` (built by backfill — weekly RS ratio time series).

### Signal Logic

```
RS_line = industry rs_ratio from backfill history (weekly)
MA_13   = SMA(RS_line, 13 weeks)

Signals (priority 0 = highest):
  0 — RS Cross Up       (crossed above MA in last 2 weeks)
  1 — RS Above MA (rising, slope > 0.005)
  2 — RS Above MA (flat)
  3 — RS Above MA (fading, slope < −0.005)
  4 — RS Below MA (recovering, slope > 0.005)
  5 — RS Cross Down     (crossed below MA in last 2 weeks)
  6 — RS Below MA (flat)
  7 — RS Below MA (falling, slope < −0.005)
```

**Output:** `results/rs_ma_signals_YYYY-MM-DD.csv`

---

## 9. Backfill (Historical Time Series)

**File:** `src/backfill.py`  
**Output:** `results/history/industry_history.csv`

Weekly time series of SCTR raw score + RRG (tactical profile) for every industry, from 2021 to present. This is the only module that builds history going back multiple years.

**What it uses:** Synthetic weekly industry indexes (constituent stocks) + SCTR formula applied at each historical date. Backfill predates the ^YH integration; switching it to ^YH weekly is a future improvement.

**Who reads it:** `rotation_severity.py` (for 4 history-based components) and `rs_ma_signals.py`.

---

## 10. Momentum Screen

**File:** `src/momentum_screen.py`  
**Output:** `results/momentum_screen_YYYY-MM-DD.csv`

Answers: "Should we rotate into this industry?" Combines three independent dimensions.

### Input files (all ^YH by default)

```
rs_percentile_YYYY.csv         → rs_pct_composite, rs_q1–q4
industry_sctr_YYYY.csv         → sctr, c_rsi14, c_ppo_slope, c_ema200_pct, c_roc20
stage_analysis_YYYY.csv        → stage, stage_score, minervini_count, golden_cross
```

### Layer 1 — Faber + Stage Signal

```
Faber rule: Hold if rs_pct_composite ≥ 80th percentile AND price above 200-SMA.

Signal hierarchy:
  STRONG BUY: quintile 1 + above 200-SMA + Stage 2B + Minervini ≥ 5
  BUY:        quintile 1 + above 200-SMA + Stage 2A or 2B
  HOLD:       quintile 2 + Stage 2A/2B  OR  quintile 1 + Stage 2C
  WATCH:      quintile 1–2 + Stage 1 (basing) + above 200-SMA
  CAUTION:    quintile 1 + below 200-SMA
  EXIT:       quintile 3–5 or Stage 3/4
```

### Layer 2 — Moskowitz-Grinblatt (1999) Industry Momentum

```
mg_9m_score = rs_q1 + rs_q2 + rs_q3   (9-month excess return, skip most recent quarter)
mg_pct      = percentile_rank(mg_9m_score) → 1–99
Signal:  Leader (≥80) / Above Average (60–79) / Average (40–59) /
         Below Average (20–39) / Laggard (<20)
```

### Layer 3 — Short-Term SCTR

```
Uses only the two true short-term ChartSchool components:

  sctr_st_raw = c_rsi14 × 0.50 + c_ppo_slope × 0.50

  sctr_st    = percentile_rank(sctr_st_raw) → 0–99.9

  st_lead   = sctr_st − sctr_full
  st_signal = "ST Leading"  if st_lead ≥ +10   (short-term outrunning long-term)
              "ST Lagging"  if st_lead ≤ −10   (early warning)
              "flat"        otherwise
```

### Composite Momentum Score

```
momentum_score = rs_pct_composite × 0.40
               + sctr_full        × 0.30
               + sctr_st          × 0.20
               + mg_pct           × 0.10

Range: 0–100.  momentum_pct = percentile_rank(momentum_score) → 1–99
```

---

## 11. Rotation Severity

**File:** `src/rotation_severity.py`  
**Output:** `results/rotation_severity_YYYY-MM-DD.csv`  

Master output — "how strong and confirmed is this rotation?" Six-component composite.

### Six Components

```
Component              Weight    Source
────────────────────────────────────────────────────────────
1. SCTR Momentum        20%      industry_history.csv (4w + 12w SCTR change)
2. RRG Persistence      18%      industry_history.csv (weeks in quadrant)
3. RRG Transition       18%      industry_history.csv (prior quadrant change)
4. Tail Velocity        17%      industry_history.csv (4-week RS-Ratio/Momentum vector)
5. MA Structure         12%      daily closes (above/below 50/200 SMA)
6. Timeframe Convergence 15%    rrg_industry_YYYY.csv (tactical + trend quadrant pair)
```

> **Note — Component 5:** Currently still builds a synthetic daily industry index for MA structure (50/200 SMA calculation). This is the last remaining synthetic usage in the index pipeline. Replacing it with ^YH daily data is a pending improvement.

### Component 1: SCTR Momentum (20%)

```
d4  = sctr_now − sctr_4w_ago
d12 = sctr_now − sctr_12w_ago

combined = clip(d4×0.6 + d12×0.4, −40, +40)
score    = (combined + 40) / 80 × 100
  → 0 (SCTR falling fast) to 50 (flat) to 100 (rising fast)
```

### Component 2: RRG Persistence (18%)

```
If Leading or Improving:
  weeks ≤ 2  → 65  (fresh entry)
  weeks 3–8  → 70 + min(weeks−2, 6) × 2.5   (peaks at 85)
  weeks 9+   → max(85 − (weeks−8) × 2, 50)  (rotation getting old)

If Weakening or Lagging:
  max(35 − weeks × 2, 10)
```

### Component 3: RRG Transition (18%)

```
Bullish (clockwise):
  Improving → Leading   → 90   (highest conviction)
  Lagging   → Improving → 75   (early signal)

Bearish:
  Leading   → Weakening → 25   (early warning)
  Weakening → Lagging   → 15   (confirmed exit)

Non-clockwise entry in Leading/Improving → 55
Stable Leading/Improving                 → 70
```

### Component 4: Tail Velocity (17%)

```
4-week movement vector in (RS-Ratio, RS-Momentum) space:
  dx = rs_ratio_now    − rs_ratio_4w_ago
  dy = rs_momentum_now − rs_momentum_4w_ago

Direction toward Leading center (101.5, 101.5):
  cos_similarity = dot(velocity, target) / (|velocity| × |target|)

speed_score = min(speed / 0.6, 1.0)
dir_score   = (cos_similarity + 1) / 2
score       = (speed_score × 0.5 + dir_score × 0.5) × 100
```

### Component 5: MA Structure (12%)

```
Using daily industry close vs 50-day and 200-day SMA:

  above_50 AND above_200: 75 + min(max(pct_above_50sma, 0), 10) × 1.5   [75–90]
  above_50 only:          55
  above_200 only:         40
  below both:             max(20 + pct_from_50sma, 5)
  Bonus: +10 if crossed above 50-SMA within last 15 trading days
```

### Component 6: Timeframe Convergence (15%)

```
Maps (tactical_quadrant, trend_quadrant) → score:

  Both Leading                           → 100
  Weekly Improving, Monthly Leading      → 85
  Weekly Leading, Monthly Improving      → 80
  Both Improving                         → 75
  Weekly Weakening, Monthly Leading      → 60   ← pullback in uptrend (Murphy)
  Weekly Lagging,   Monthly Leading      → 50   ← deeper pullback
  Both Weakening                         → 25
  Both Lagging                           → 0
```

Murphy's key insight: weekly Weakening + monthly (trend) Leading = noise. Only when both timeframes confirm is the rotation structurally significant.

### Final Score and Label

```
severity_score = 0.20×sctr_mom + 0.18×persistence + 0.18×transition +
                 0.17×velocity + 0.12×ma_structure + 0.15×timeframe

Labels:
  score ≥ 80, both_bullish          → "Both TF Confirmed"
  score ≥ 80, weekly_bullish        → "Strong Confirmed"
  score ≥ 65, weekly_bullish        → "Confirmed"
  score ≥ 55, weekly_bullish        → "Early Signal"
  pullback + score ≥ 50             → "Pullback in Uptrend"
  score ≥ 45                        → "Neutral / Watch"
  score < 30, both_bearish          → "Confirmed Exit"
  score < 40, weekly_bearish        → "Weakening"
  else                              → "Neutral / Watch"

Upgrades: " + ATH", " ★RS" (rs_pct ≥ 90 + new high), " (RS≥80)"
```

---

## 12. Market Breadth

**File:** `src/breadth.py`  
**Output:** `results/breadth_YYYY-MM-DD.csv`

### Composite Health Score (0–100)

```
health_score = stage2_pct   × 0.25   (% industries Stage 2A or 2B)
             + above_200    × 0.20   (% above 200-day SMA)
             + rs_positive  × 0.20   (% rs_pct_composite ≥ 50)
             + golden_cross × 0.20   (% with 50-SMA > 200-SMA)
             + sctr60       × 0.15   (% SCTR ≥ 60)

Labels: Bull (≥80) / Healthy (≥60) / Mixed (≥40) / Weak (≥20) / Bear (<20)
```

---

## 13. Execution Order

```bash
# ── Composite modes ──────────────────────────────────────────────────────────
python main.py --mode update      # everything: industry pipeline + stocks pipeline
python main.py --mode industry    # all industry steps in order (sctr → … → signal-delta)
python main.py --mode stocks      # all stock steps in order (stock-sctr → stock-screener)

# ── Individual industry steps (run in this order if running manually) ────────
python main.py --mode sctr            # ^YH SCTR  → industry_sctr_YYYY.csv
python main.py --mode rrg             # RRG (tactical + trend)  → rrg_industry_YYYY.csv
python main.py --mode ath             # ATH monitor  → ath_monitor_YYYY.csv
python main.py --mode rs-ma           # RS vs 13w MA signals  → rs_ma_signals_YYYY.csv
python main.py --mode market-clock    # regime detection  → market_clock_YYYY.csv
python main.py --mode rs-percentile   # RS rating  → rs_percentile_YYYY.csv
python main.py --mode stage           # Stage analysis  → stage_analysis_YYYY.csv
python main.py --mode momentum        # Faber + MG + ST-SCTR  → momentum_screen_YYYY.csv
python main.py --mode severity        # Master output  → rotation_severity_YYYY.csv
python main.py --mode market-breadth  # Health score  → breadth_YYYY.csv
python main.py --mode signal-delta    # What changed  → signal_delta_YYYY.csv

# ── Research / legacy ────────────────────────────────────────────────────────
python main.py --mode industry-synth    # synthetic SCTR (research only)
python main.py --mode momentum-compare  # yh vs synth comparison
python main.py --mode backfill          # rebuild weekly history from scratch
```

---

## 14. Timeframe Summary

```
TIMEFRAME   SOURCE        MODULE              SIGNAL
────────────────────────────────────────────────────────────────
Daily       ^YH daily     sctr_industry.py    SCTR (6 components)
                          stage_analysis.py   MA structure (50/150/200 SMA)
                          ath_monitor.py      Monthly ATH proximity
                          rs_percentile.py    RS rating (63/126/189/252d windows)

Weekly      ^YH weekly    rrg_engine.py       Tactical + Trend RRG profiles
            backfill      rs_ma_signals.py    RS line vs 13-week MA
            (synth)       industry_history    SCTR + RRG time series (4+ years)

Multi-week  history CSV   rotation_severity   SCTR momentum (4w/12w)
                                              RRG persistence + transition + velocity
```

---

## 15. Known Limitations

**Backfill SCTR formula:** `industry_history.csv` was built with the pre-correction SCTR formula. Historical SCTR momentum scores (Components 1) in severity are internally consistent but not comparable to the current `industry_sctr_*.csv` output.

**Severity Component 5 (MA Structure):** Still uses a synthetic daily index for the 50/200-SMA calculation. Switching to ^YH daily is a pending improvement.

**`rotation_composite.py` (standalone):** This legacy module reads `industry_sctr_synth_*.csv`. It is not part of the `--mode update` chain. May be deprecated or switched to ^YH.
