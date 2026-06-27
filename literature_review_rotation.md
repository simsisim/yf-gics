# Sector & Industry Rotation: Literature Review and Implementation Plan

*Generated: 2026-06-27*

---

## 1. The Core Problem (from ideas.txt)

You have SCTR data from StockCharts, an RRG dashboard, and want to answer:

- **How strong/severe is a rotation in or out of a sector/industry?**
- **Is a weekly tech pullback noise or a real monthly-timeframe rotation out?**
- **During a correction: which stocks are the true leaders, and which previous leaders are holding up?**
- **How do we find emerging industries and emerging stocks within them?**

The central difficulty: rotation is continuous and multi-scale. A signal that looks like a full reversal on the daily chart may be a tiny blip on the monthly. You need **convergence across timeframes** to assess severity.

---

## 2. Literature & Methodological Foundations

### 2.1 Relative Rotation Graphs (RRG) — Julius de Kempenaer

**Origin:** Developed 2004–2005 by Julius de Kempenaer (then at ABN Amro, Amsterdam) to solve the institutional problem of separating sector leaders from laggards amid information overload. Available on Bloomberg since 2011 (mnemonic: RRG), and later on StockCharts.

**Core metrics:**

| Metric | Definition | Threshold |
|---|---|---|
| RS-Ratio | Normalized relative performance vs benchmark | >100 = outperforming |
| RS-Momentum | Rate of change of RS-Ratio | >100 = accelerating |

Both metrics are normalized so they can be compared across different securities against the same benchmark.

**The four quadrants:**

```
                 RS-Momentum > 100
                        ↑
  Improving        |   Leading
  (low RS,         |   (high RS,
   rising mom)     |    rising mom)
  ─────────────────┼─────────────────  RS-Ratio 100
  Lagging          |   Weakening
  (low RS,         |   (high RS,
   falling mom)    |    falling mom)
                        ↓
                 RS-Momentum < 100
```

**Clockwise rotation** is the typical path: Leading → Weakening → Lagging → Improving → Leading.

**Measuring rotation strength:**
- **Tail length** = velocity of rotation (longer tail = faster move)
- **Tail curvature** = acceleration/deceleration
- **Distance from center (100,100)** = how extreme the relative positioning is
- **Timeframe of RRG** = weekly RRG vs monthly RRG resolves severity ambiguity

**Key insight for your use case:** Plot the same sector/industry on weekly AND monthly RRG simultaneously. If it's weakening only on the weekly but still in "Leading" on the monthly, the rotation is shallow (likely a pullback). If both timeframes show the same movement, the rotation is structural.

**Resources:** [relativerotationgraphs.com](https://relativerotationgraphs.com), [StockCharts RRG ChartSchool](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/chart-types/relative-rotation-graphs-rrg-charts)

---

### 2.2 SCTR — StockCharts Technical Rank (John Murphy)

**Design:** A percentile-based ranking (0–99.9) of a stock's *technical strength* within its peer universe (large-cap, mid-cap, small-cap). Devised by John Murphy and implemented by StockCharts.

**The six underlying indicators (multi-timeframe):**

| Timeframe | Indicator | Weight |
|---|---|---|
| Long-term | 14-month ROC | 30% |
| Long-term | Distance from 200-day SMA | 15% |
| Medium-term | 9-month ROC | 15% |
| Medium-term | Distance from 50-day SMA | 15% |
| Short-term | 3-month ROC | 10% |
| Short-term | PPO (14,1) histogram | 15% |

**Percentile buckets:** Universe split into equal buckets; a stock at SCTR 90+ is in the top 10% of its peer group technically.

**How to use for rotation:**
- Rising SCTR (especially crossing above 60, then 80) in an industry signals internal strength building before price breakout
- Falling SCTR in a previous leader signals rotation beginning even before the price breaks key MAs
- Use short-term SCTR variant (your idea from ideas.txt) to detect very early rotation signals
- Track SCTR percentile *rank* of entire GICS industries to find leading industries

**Resources:** [SCTR ChartSchool](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/stockcharts-technical-rank-sctr)

---

### 2.3 Murphy's Intermarket Analysis Framework

**Books:** *Intermarket Analysis* (2004), *Trading with Intermarket Analysis* (2012)

**Framework:** Four asset classes interact — stocks, bonds, commodities, currencies. The sequence:
1. Bond yields fall → money rotates into interest-rate-sensitive sectors (utilities, real estate)
2. Commodity prices rise → resource/energy sectors lead
3. Late cycle: inflation peaks, bonds sell off, defensive sectors lead

**For sector rotation monitoring:** Murphy's framework tells you *why* rotation is happening (macro regime). Useful as a cross-check: if the RRG/SCTR signals show technology weakening but bonds are also weakening and commodities rising, the rotation has a macro basis and is more likely to persist.

**SCTR in Murphy's context:** Murphy designed SCTR to combine his intermarket insights into one composite rank — the six-indicator blend already captures multi-timeframe momentum that aligns with the sector rotation cycle.

---

### 2.4 Momentum & Relative Strength — Academic Literature

**Moskowitz & Grinblatt (1999)** — *"Do Industries Explain Momentum?"* (Journal of Finance):
- Industry-based momentum strategies are *more profitable* than individual stock momentum
- Winners (past 12 months) at the industry level continue to outperform
- Fundamental finding: most individual stock momentum is actually captured by the industry trend

**Mebane Faber (2007)** — *"A Quantitative Approach to Tactical Asset Allocation"* (SSRN):
- Trend-following rule: hold asset if price > 10-month SMA, exit if below
- Sector rotation variant: rank sectors by 12-month momentum, hold top quintile
- Achieves equity-like returns with bond-like drawdowns
- [Paper (PDF)](https://mebfaber.com/wp-content/uploads/2016/05/SSRN-id962461.pdf)

**"Sector Rotation by Factor Model and Fundamental Analysis" (2024, arxiv:2401.00001):**
- Combines momentum factor + fundamental analysis for sector selection
- Confirms momentum is the dominant factor for sector rotation prediction
- Adds short-term reversion signal as a complement

**Two-Stage Sector Rotation via ML (arxiv:2108.02838):**
- Stage 1: ML predicts sector index returns
- Stage 2: Rank and hold top sectors
- Shows machine learning can improve on pure momentum, but pure momentum is still the baseline

---

### 2.5 Closing Range Analysis (IBD / O'Neil Framework)

**William O'Neil / IBD concept:** During a market correction, *closing range* (where the price closes within the day's range) reveals institutional conviction.

**Closing Range formula:**
```
Closing Range % = (Close - Low) / (High - Low) × 100
```

- **>60%**: Close in upper portion of range → institutions supporting the stock
- **<40%**: Close in lower portion → distribution pressure

**Your specific question from ideas.txt:** "What stocks had a closing range greater than 60% for 60% of the correction time?"

This is a powerful filter. If a stock consistently closes in the upper part of its range during a broad market correction, it signals:
1. Institutional accumulation (they buy the dip on that stock)
2. Relative strength (it resists selling pressure)
3. Likely future leader once the market recovers

**Related signals:** O'Neil's CAN SLIM system uses the Follow-Through Day (a +1.25% gain on volume above the prior day on day 4+ of a rally attempt) to confirm correction bottoms — useful for timing entry into your rotation leaders.

---

### 2.6 Moving Average Framework for Rotation Severity

The 10/21/50 EMA cascade (popularized by Mark Minervini and the Stage Analysis community):

```
10-EMA > 21-EMA > 50-SMA (all rising) = Stage 2 uptrend, strong
Price below 21-EMA but above 50-SMA = pullback in uptrend (buyable)
Price below 50-SMA = trend broken, potential Stage 3/4
50-SMA crossing below 200-SMA = confirmed bear (Death Cross)
```

**For rotation severity:**
- A sector ETF breaking below its 50-SMA on weekly chart = serious rotation signal
- Sector ETF breaking below 50-SMA on monthly chart = structural rotation, not noise
- **ATH breakout on monthly chart**: if a sector makes new all-time highs on monthly, the rotation INTO it is definitive — not a short-term fluctuation

**MA-based RS tracking (your idea):**
Apply MAs to the RS ratio line (not price):
- RS ratio crossing above its 50-day MA = rotation signal beginning
- RS ratio *sustaining* above its 50-day MA for multiple weeks = confirmed rotation

---

## 3. Synthesis: How to Measure Rotation Strength

Based on the literature, rotation strength can be scored along these dimensions:

| Dimension | Weak Rotation | Strong/Severe Rotation |
|---|---|---|
| **Timeframe** | Only on daily/weekly RRG | Weekly AND monthly RRG aligned |
| **RRG tail length** | Short tail | Long, fast-moving tail |
| **Distance from RRG center** | Near 100,100 | Far into Weakening or Lagging |
| **SCTR trend** | Declining slowly | SCTR < 40 and falling |
| **MA structure** | Below 10/21 but above 50 | Below 50-SMA on weekly |
| **Monthly chart** | Below recent highs, not ATH break | ATH breakout = confirmed strength IN |
| **RS line** | Below its 50-day MA | RS line trending down on all TFs |
| **Macro context (Murphy)** | Rotation without macro support | Rotation supported by bond/commodity signals |

**Rule of thumb:** If 4+ dimensions align, treat the rotation as structural (months-long). If 1–2, treat it as a tactical pullback.

---

## 4. Implementation Plan

### Phase 1: Multi-Timeframe RRG Scoring (extend `/home/imagda/_invest2024/python/RRGs-Dashboard`)

**Goal:** Generate a rotation score for each sector/industry that combines weekly and monthly RRG signals.

```
Steps:
1. Compute RS-Ratio and RS-Momentum on weekly prices
2. Compute RS-Ratio and RS-Momentum on monthly prices  
3. Assign quadrant label per timeframe: Leading/Weakening/Lagging/Improving
4. Score rotation severity:
   - Both Leading: +2 (strong in-rotation)
   - Weekly Weakening, Monthly Leading: -1 (shallow pullback)
   - Both Weakening or Weekly Lagging, Monthly Weakening: -3 (serious rotation out)
   - Both Lagging: -4 (confirmed rotation out)
5. Track tail length (speed of RS-Momentum change week-over-week)
6. Store scores in database alongside SCTR data
```

**Output:** Dashboard showing sector/industry rotation score + direction + speed

---

### Phase 2: SCTR Percentile Tracker (extend `/home/imagda/_invest2024/python/stockCharts`)

**Goal:** Track SCTR momentum at the industry level to find emerging leaders early.

```
Steps:
1. For each GICS industry: compute mean SCTR of all stocks in industry
2. Compute 4-week and 12-week moving average of the industry mean SCTR
3. Flag: industry mean SCTR crossing above its 4-week MA from below (improving)
4. Rank all industries by SCTR percentile — update weekly
5. Overlay on RRG quadrant assignment
6. "Short-term SCTR" variant: use a subset of SCTR indicators (3-month ROC + PPO only)
   for earlier signals
```

---

### Phase 3: Correction Leader Filter (new module in `/home/imagda/_invest2024/python/metaData_v1`)

**Goal:** During a market correction, identify stocks that are holding up best.

**Filters to implement:**

```python
# Filter 1: Closing Range
closing_range = (close - low) / (high - low) * 100
cr_strong = closing_range > 60  # upper range
cr_strong_pct = cr_strong.rolling(correction_days).mean() > 0.60  # 60% of correction days

# Filter 2: MA Adherence
above_10ema = close > ema(10)
above_21ema = close > ema(21)
above_50sma = close > sma(50)
held_21ema = above_21ema.rolling(correction_days).mean() > 0.70  # held 70% of time

# Filter 3: RS vs Benchmark
rs_line = close / benchmark_close
rs_line_positive = rs_line > rs_line.shift(correction_days)  # positive over correction

# Filter 4: SCTR >= 70 at start of correction AND hasn't fallen below 50 during
sctr_held = (sctr_at_correction_start >= 70) & (sctr_min_during_correction >= 50)

# Filter 5: Volume pattern (accumulation vs distribution)
up_volume = volume.where(close > close.shift(1), 0)
down_volume = volume.where(close < close.shift(1), 0)
accumulation = up_volume.rolling(correction_days).sum() > down_volume.rolling(correction_days).sum()

# Composite: stock passes 4 of 5
leader_score = (cr_strong_pct.astype(int) + held_21ema.astype(int) + 
                rs_line_positive.astype(int) + sctr_held.astype(int) + 
                accumulation.astype(int))
correction_leaders = leader_score >= 4
```

**Output:** Ranked list of stocks that held up, organized by GICS industry/sector

---

### Phase 4: Rotation Dashboard / Visualization

**Building on RRG Dashboard visualization strength:**

```
Panel 1: Multi-timeframe RRG (weekly + monthly overlay, same chart)
  - Show tail from last 4 weeks on weekly
  - Show separate dot for monthly position
  - Color: green = aligned leading, yellow = mixed, red = aligned lagging

Panel 2: Sector rotation score heatmap (sectors × timeframes)
  - Rows: 11 GICS sectors
  - Columns: daily / weekly / monthly
  - Cell: RRG quadrant + color

Panel 3: Industry Leaderboard
  - Top 5 improving industries (SCTR rising + RRG Improving quadrant)
  - Bottom 5 weakening industries

Panel 4: Correction Leaders (activates during market corrections)
  - Stocks passing correction leader filters
  - Sorted by composite score
  - Colored by GICS industry (cluster visualization)

Panel 5: Monthly ATH Breakout Monitor
  - Industries / sector ETFs at or breaking ATH on monthly chart
  - These are the highest-confidence rotation-in signals
```

---

### Phase 5: Rotation Severity Alert System

Define alert thresholds based on the scoring framework from Section 3:

```
ALERT LEVELS:
- Level 1 (Noise): 1–2 dimensions aligning → no action signal
- Level 2 (Watch): 3 dimensions → monitor closely, daily check
- Level 3 (Probable Rotation): 4–5 dimensions → begin position reduction / building
- Level 4 (Confirmed Rotation): 6+ dimensions OR monthly ATH breakout in destination sector
```

---

## 5. Data Sources and Tool Mapping

| Data | Source | Location in your system |
|---|---|---|
| SCTR scores (stocks + industries) | StockCharts (scraped) | `/home/imagda/_invest2024/python/stockCharts` |
| Historical OHLCV | Yahoo Finance (yfinance) | `/home/imagda/_invest2024/python/yf-gics` + `metaData_v1` |
| GICS classification | CSV mapping | `industries.csv`, `stocks_by_industry.csv` |
| RRG computation | Custom Python | `/home/imagda/_invest2024/python/RRGs-Dashboard/src` |
| RS / percentile / Minervini models | Existing module | `/home/imagda/_invest2024/python/metaData_v1/src` |

---

## 6. Key References

### Books
- Murphy, J.J. (2004). *Intermarket Analysis: Profiting from Global Market Relationships*. Wiley.
- Murphy, J.J. (2012). *Trading with Intermarket Analysis*. Wiley.
- O'Neil, W. (2009). *How to Make Money in Stocks*. McGraw-Hill. (CAN SLIM, closing range, follow-through day)
- de Kempenaer, J. (blog/research). [relativerotationgraphs.com](https://relativerotationgraphs.com)

### Papers
- Moskowitz, T. & Grinblatt, M. (1999). "Do Industries Explain Momentum?" *Journal of Finance*, 54(4), 1249–1290.
- Faber, M. (2007). "A Quantitative Approach to Tactical Asset Allocation." SSRN 962461. [PDF](https://mebfaber.com/wp-content/uploads/2016/05/SSRN-id962461.pdf)
- arxiv:2401.00001 (2024). "Sector Rotation by Factor Model and Fundamental Analysis." [Link](https://arxiv.org/abs/2401.00001)
- arxiv:2108.02838. "Two-Stage Sector Rotation Methodology Using Machine Learning." [Link](https://arxiv.org/pdf/2108.02838)

### Web Resources
- [RRG ChartSchool (StockCharts)](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/chart-types/relative-rotation-graphs-rrg-charts)
- [SCTR ChartSchool](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/stockcharts-technical-rank-sctr)
- [Faber's Sector Rotation Strategy (StockCharts)](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models/trading-strategies/fabers-sector-rotation-trading-strategy)
- [Sector Rotation Monitor (real-time)](https://sectorrotationmonitor.com/)
- [Quantpedia: Sector Momentum Rotational System](https://quantpedia.com/strategies/sector-momentum-rotational-system)
- [TrendSpider: How to Track Sector Rotation](https://trendspider.com/blog/sector-rotation-how-to-track-where-the-money-is-moving/)
- [EC Markets: Relative Strength Charts for Rotation](https://www.ecmarkets.com/insights/relative-strength-charts-picking-sectors-before-they-rotate/)
