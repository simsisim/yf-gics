# CLI Command Reference

## Composite modes

```bash
# Everything: industry pipeline + stocks pipeline
python main.py --mode update

# All industry steps (sctr → rrg → ath → rs-ma → market-clock → rs-percentile → stage → momentum → severity → market-breadth → signal-delta)
python main.py --mode industry

# All stock steps (stock-sctr → stock-screener)
python main.py --mode stocks
```

---

## Setup / History

```bash
# Rebuild weekly history from scratch (run once before first update)
python main.py --mode backfill

# Rebuild from a specific start date (faster — 6 months is enough for daily use)
python main.py --mode backfill --backfill-start 2026-01-01
```

---

## Industry — individual steps

```bash
python main.py --mode sctr            # ^YH SCTR → industry_sctr_YYYY.csv
python main.py --mode rrg             # RRG tactical + trend → rrg_industry_YYYY.csv
python main.py --mode ath             # ATH monitor → ath_monitor_YYYY.csv
python main.py --mode rs-ma           # RS vs 13w MA → rs_ma_signals_YYYY.csv
python main.py --mode market-clock    # Regime (FTD / distribution days) → market_clock_YYYY.csv
python main.py --mode rs-percentile   # RS composite rating → rs_percentile_YYYY.csv
python main.py --mode stage           # Weinstein/Minervini stage → stage_analysis_YYYY.csv
python main.py --mode momentum        # Faber + MG + ST-SCTR → momentum_screen_YYYY.csv
python main.py --mode severity        # Master output → rotation_severity_YYYY.csv
python main.py --mode market-breadth  # Health score → breadth_YYYY.csv
python main.py --mode signal-delta    # What changed → signal_delta_YYYY.csv
```

---

## Stocks — individual steps

```bash
python main.py --mode stock-sctr       # Stock SCTR by cap bucket → stock_sctr_{large|mid|small}_YYYY.csv
python main.py --mode stock-screener   # Ranked stocks within top industries → stock_screener_YYYY.csv

# Closing range — correction leader filter (run after a correction, before re-running stock-screener)
python main.py --mode closing-range --correction-start 2026-06-12 --correction-end 2026-06-26

# After closing-range, re-run screener to pick up CR scores
python main.py --mode closing-range --correction-start 2026-06-12 --correction-end 2026-06-26 && python main.py --mode stock-screener

# Stock screener options
python main.py --mode stock-screener --top-industries-only   # only STRONG BUY / BUY industries
python main.py --mode stock-screener --min-score 4           # min closing range score
```

---

## Backtesting (as-of a historical date)

```bash
python main.py --mode update   --date 2025-12-31
python main.py --mode industry --date 2025-12-31
python main.py --mode stocks   --date 2025-12-31
python main.py --mode severity --date 2025-12-31
```

---

## Research / legacy

```bash
python main.py --mode industry-synth    # Synthetic SCTR (legacy research)
python main.py --mode momentum-compare  # yh vs synth side-by-side comparison
```

---

## Dashboard

```bash
streamlit run app.py
```
