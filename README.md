# GICS Industry Rotation Monitor

A top-down, quantitative market analysis system built around GICS industry groups. Combines StockCharts Technical Rank (SCTR), Relative Rotation Graphs (RRG), Weinstein/Minervini stage analysis, and Faber-style momentum signals to identify the strongest sectors, industries, and individual stocks at any point in time.

---

## Overview

The pipeline flows top-down:

```
Market Clock (SPY/QQQ health)
    └── Sector / Industry Rotation (RRG quadrant, RS, SCTR)
            └── Industry Stage Analysis (Weinstein/Minervini)
                    └── Momentum Screen (6-level Faber + Stage signal)
                            └── Stock Screener (individual stocks within top industries)
```

All outputs land in `results/` as timestamped CSVs and Markdown reports. A Streamlit dashboard (`app.py`) visualises everything.

---

## Architecture

```
yf-gics/
├── main.py                    # CLI entry point — all modes
├── app.py                     # Streamlit dashboard (5 tabs)
├── config.py                  # Paths and constants
├── industries.csv             # 143 GICS industry groups
├── stocks_by_industry.csv     # ~5,200 stocks mapped to industries
├── requirements.txt
└── src/
    ├── stage_analysis.py      # Weinstein/Minervini stage classification
    ├── momentum_screen.py     # 6-level Faber + Stage signal
    ├── rotation_severity.py   # Composite severity scoring
    ├── rotation_composite.py  # RRG + RS + ATH composite
    ├── rrg_engine.py          # Relative Rotation Graph engine
    ├── rrg_chart.py           # RRG chart generation
    ├── rs_percentile.py       # RS composite percentile ranking
    ├── rs_ma_signals.py       # RS moving average signals
    ├── ath_monitor.py         # All-time-high tracking
    ├── market_clock.py        # Market regime detector (FTD/distribution)
    ├── sctr_engine.py         # StockCharts Technical Rank formula
    ├── sctr_industry_synth.py  # Industry SCTR from ^YH indexes
    ├── sctr_industry.py       # Industry SCTR (alt method)
    ├── sctr_stocks.py         # Stock SCTR (large/mid/small cap)
    ├── sctr_breadth.py        # SCTR breadth by industry
    ├── breadth.py             # Market breadth health score
    ├── closing_range.py       # IBD correction leader filter
    ├── signal_delta.py        # Daily signal change detector
    ├── stock_screener.py      # Individual stock ranker
    ├── backfill.py            # Historical index builder (weekly SCTR + RRG, since 2021)
    ├── perf_screener.py       # Flat 1w/2w/1m/3m/6m return + RS + relret vs SPY & QQQ
    ├── history_metrics.py     # Shared vectorized computation for daily history tracking
    ├── history_backfill.py    # Daily SCTR/Stage-Minervini/RS/perf history (trailing N months)
    ├── history_tracker.py     # Go-forward daily append to that history
    ├── correction_filter.py   # Correction-period stock filter
    ├── data_loader.py         # OHLCV loader utilities
    └── universe_loader.py     # Stock universe management
```

---

## Data Requirements

This project reads OHLCV data from an external directory — it does **not** download data itself. Expected structure:

```
/path/to/market_data/
    daily/      # CSV per ticker:  Date, Open, High, Low, Close, Volume
    weekly/
    monthly/
```

Update `config.py` to point `_DOWNLOAD_ROOT` at your data root, and `_TICKER_INFO` at your ticker metadata CSV.

**Reference files included:**
- `industries.csv` — 143 GICS industry groups with sector mapping
- `stocks_by_industry.csv` — stock-to-industry mapping (~5,200 tickers)

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** `pandas`, `numpy`, `scipy`, `streamlit`, `plotly`, `yfinance`

---

## Usage

### Composite modes (recommended)

```bash
python main.py --mode update      # everything: industry pipeline + stocks pipeline
python main.py --mode industry    # all industry steps (sctr → rrg → … → signal-delta)
python main.py --mode stocks      # all stock steps (stock-sctr → stock-screener)
```

### Individual modes

| Mode | Level | Description |
|------|-------|-------------|
| `sctr` | industry | Industry SCTR from ^YH index files |
| `rrg` | industry | Weekly RRG — tactical (5/10/3) + trend (10/30/7) profiles |
| `ath` | industry | All-time-high tracking per industry |
| `rs-ma` | industry | RS line vs 13-week MA signals |
| `market-clock` | industry | Market regime via FTD + distribution days |
| `rs-percentile` | industry | RS composite percentile ranks (1–99) |
| `perf` | industry + sector | Flat 1w/2w/1m/3m/6m return, RS, and relative return vs SPY & QQQ; also adds SCTR for the 11 sector SPDR ETFs |
| `history-backfill` | industry + sector | Rebuilds trailing N months (default 6) of **daily** SCTR/Stage-Minervini/RS/performance history, one row per (date, symbol) |
| `history-append` | industry + sector | Cheap go-forward: computes and upserts just today's row into that same history file |
| `stage` | industry | Weinstein/Minervini stage classification |
| `momentum` | industry | 6-level Faber + Stage + MG momentum signal |
| `severity` | industry | Composite rotation severity score (master output) |
| `market-breadth` | industry | Breadth health score + sector breakdown |
| `signal-delta` | industry | What changed since the previous run |
| `stock-sctr` | stocks | Stock SCTR by cap bucket (large/mid/small) |
| `stock-screener` | stocks | Ranked stocks within STRONG BUY / BUY industries |
| `closing-range` | stocks | IBD correction leader filter (F1–F5 criteria) |

```bash
# Backtest as of a historical date
python main.py --mode update --date 2025-12-31

# Closing range with custom window
python main.py --mode closing-range --correction-start 2026-03-19 --correction-end 2026-04-08

# Stock screener — only top industries, min leader score 4
python main.py --mode stock-screener --top-industries-only --min-score 4
```

### Streamlit dashboard

```bash
streamlit run app.py
```

Five tabs: **Overview** (market clock, breadth gauge, sector health) · **Industries** (filterable severity table) · **Stage Map** (distribution charts, Minervini heatmap) · **Stocks** (scatter + ranked table) · **Signal Delta** (upgrades/downgrades/crossings)

---

## Signals

### Stage Analysis (Weinstein / Minervini)

Classifies each industry's synthetic price index using 50/150/200-day SMA structure:

| Stage | Meaning | Criteria |
|-------|---------|---------|
| **2B** | Mid uptrend | Above all 3 SMAs, 200-SMA rising, golden cross, all MAs aligned |
| **2A** | Early uptrend | Above 200-SMA + golden cross, but 150/50 not yet fully aligned |
| **2C** | Late uptrend | Above 200-SMA but 200-SMA flat (slope near 0) |
| **1**  | Basing | Below 200-SMA but 200-SMA flat — potential accumulation |
| **3**  | Distribution | Was above 200-SMA, now 200-SMA declining |
| **4**  | Downtrend | Below declining 200-SMA |

**Minervini 7 Criteria** (score 0–7):
1. Price > 150-SMA
2. Price > 200-SMA
3. 150-SMA > 200-SMA
4. 200-SMA slope > 0
5. Price ≥ 30% above 52-week low
6. Price within 25% of 52-week high
7. RS percentile ≥ 70

### Faber + Stage Signal (6 levels)

| Signal | Criteria |
|--------|---------|
| **STRONG BUY** | Quintile 1 + above 200-SMA + Stage 2B + Minervini ≥ 5/7 |
| **BUY** | Quintile 1 + above 200-SMA + Stage 2A or 2B |
| **HOLD** | Quintile 2 + Stage 2A/2B, or Quintile 1 + Stage 2C |
| **WATCH** | Quintile 1–2 + Stage 1 (basing) + above 200-SMA |
| **CAUTION** | Quintile 1 + below 200-SMA |
| **EXIT** | All other cases |

*Quintile is based on the Faber 10-month SMA timing model applied to the industry's 4-week RS composite.*

### Market Breadth Health Score (0–100)

Weighted composite of 5 metrics across all 143 industries:

| Metric | Weight |
|--------|--------|
| Stage 2 (2A + 2B) % | 25% |
| Above 200-SMA % | 20% |
| RS composite ≥ 50th percentile % | 20% |
| Golden Cross % | 20% |
| SCTR ≥ 60 % | 15% |

Labels: **Bull Market** (≥ 80) · **Healthy** (≥ 65) · **Mixed** (≥ 45) · **Weak** (≥ 30) · **Bear Market** (< 30)

### Stock Screener Composite Score (0–100)

| Component | Weight |
|-----------|--------|
| RS percentile (12m excess return vs SPY, ranked within screened universe) | 40% |
| Stage score (2B=100, 2A=85) | 35% |
| Minervini % (criteria_met / 7 × 100) | 15% |
| IBD closing range leader score (0–5) | 10% |

Filters: Stage 2A or 2B only · Minervini ≥ 3/7 · Industry in STRONG BUY or BUY

---

## Output Files

All outputs are written to `results/` with a date suffix:

| File | Contents |
|------|---------|
| `rotation_severity_DATE.csv` | Per-industry composite severity score + all sub-signals |
| `momentum_screen_DATE.csv` | Faber + Stage signal for all 143 industries |
| `stage_analysis_DATE.csv` | Stage classification + Minervini scores |
| `stock_screener_DATE.csv` | Ranked individual stocks within top industries |
| `breadth_DATE.csv` | Overall breadth health metrics |
| `breadth_sector_DATE.csv` | Same metrics broken down by GICS sector |
| `signal_delta_DATE.csv` | What changed vs previous run |
| `closing_range_DATE.csv` | IBD correction leaders with F1–F5 pass/fail |
| `market_clock_DATE.csv` | Regime state for SPY, QQQ, IWM |
| `rs_percentile_DATE.csv` | RS composite percentile ranks |
| `ath_monitor_DATE.csv` | ATH proximity and signal per industry |

---

## Key Concepts

- **RRG (Relative Rotation Graph)** — RS ratio and RS momentum axes produce 4 quadrants: Leading, Weakening, Lagging, Improving. MACD Trend-Following methodology (researched against Julius de Kempenaer's work in RRGs-Dashboard). Two weekly profiles: Tactical (5/10/3) for timing, Trend (10/30/7) for structural confirmation — both output to a single CSV.
- **SCTR** — StockCharts Technical Rank. Weighted blend of 6 technical indicators across 3 timeframes (short, medium, long).
- **RS Composite Percentile** — Proprietary formula weighting 12-month RS most heavily: `(Q4×2 + Q3×1 + Q2×1 + Q1×1) / 5`, converted to 1–99 percentile.
- **Market Clock** — Tracks Follow-Through Days (FTD) and distribution day counts per index to identify regime transitions.
- **Synthetic Industry Index** — Market-cap-weighted daily price index built from constituent stocks to apply technical analysis to GICS industry groups.

---

## Configuration

Edit `config.py`:

```python
_DOWNLOAD_ROOT = Path('/your/path/to/market_data')
_TICKER_INFO   = Path('/your/path/to/tickers_metadata.csv')
```

All other paths are relative to the project directory and require no changes.
