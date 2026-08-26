# Dual Index Source Plan — `yh` (default) vs `synth`

*Last updated: 2026-06-29*

---

## Problem

All modules except `sctr_industry.py` use a **synthetic** market-cap-weighted index
built from constituent stocks. `sctr_industry.py` correctly uses the published `^YH`
Dow Jones Industry Index files. This inconsistency means SCTR and all other signals
are computed on different price series for the same industry.

## Decision

- **`source='yh'` (default)** — load `^YH` file from `daily_dir` / `weekly_dir` directly.
  One file per industry, same series as SCTR. Authoritative.
- **`source='synth'`** — build market-cap-weighted index from constituent stock closes.
  Existing approach. Reflects your exact stock universe.

## Output File Convention

| Source | Location | Example |
|--------|----------|---------|
| `yh` (default) | `results/` | `rs_percentile_2026-06-29.csv` |
| `synth` | `results/synth/` | `synth/rs_percentile_2026-06-29.csv` |
| `yh` SCTR | `results/industry_sctr/` | `industry_sctr_2026-06-29.csv` |
| `synth` SCTR | `results/industry_sctr/` | `industry_sctr_synth_2026-06-29.csv` |
| `yh` RRG | `results/rrg/` | `rrg_gics_2026-06-29.csv` |
| `synth` RRG | `results/rrg/` | `rrg_synth_2026-06-29.csv` |

Synth outputs in `results/synth/` means existing `_latest_file(dir, "*.csv")` globs
in `rotation_severity.py` and `rotation_composite.py` are not affected — they always
find the `yh` (default) files.

## New Module: `src/industry_index.py`

Single source of truth for industry price loading. All other modules import from here.
Replaces the inline `_load_all_closes + _build_industry_daily_index` pattern spread
across backfill, rs_percentile, stage_analysis, ath_monitor.

Key functions:
- `preload_price_matrix(tickers, daily_dir)` — batch load for synth (efficient)
- `build_synthetic_daily(tickers, weights, price_matrix)` — synth daily index
- `build_synthetic_weekly(tickers, weights, weekly_dir, cutoff, min_weeks)` — synth weekly
- `load_industry_daily(yh_symbol, tickers, weights, config, source, price_matrix, cutoff)`
- `load_industry_weekly(yh_symbol, tickers, weights, config, source, cutoff)`
- `synth_out_dir(base_dir)` — returns `base_dir/synth/`, creates if needed

## Renamed Output

`sctr_industry_gics.py`: `industry_sctr_gics_DATE.csv` → `industry_sctr_synth_DATE.csv`

Callers updated: `momentum_screen.py`, `rotation_composite.py`.

## Modules Updated

| Module | Change |
|--------|--------|
| `src/industry_index.py` | **new** — unified loader |
| `src/sctr_industry_gics.py` | rename output `_gics_` → `_synth_` |
| `src/rs_percentile.py` | `source` param; synth → `results/synth/` |
| `src/stage_analysis.py` | `source` param; synth → `results/synth/`; reads rs_pct from matching source dir |
| `src/ath_monitor.py` | `source` param; synth → `results/synth/` |
| `src/rrg_engine.py` | `source` param (default `yh`); synth RRG → `rrg_synth_DATE.csv` |
| `src/backfill.py` | import from `industry_index`; remove old inline builders |
| `src/momentum_screen.py` | `source` param; reads matching SCTR/RS/Stage files; `run_comparison()` |
| `src/rotation_composite.py` | glob `industry_sctr_gics_` → `industry_sctr_synth_` |
| `main.py` | `--source yh\|synth` flag; `momentum-compare` mode |

## Momentum Screener Comparison Output

`momentum_screen_comparison_DATE.csv` columns per industry:
- `rank_yh`, `momentum_score_yh`, `rs_pct_yh`, `sctr_yh`, `stage_yh`, `quadrant_yh`
- `rank_synth`, `momentum_score_synth`, `rs_pct_synth`, `sctr_synth`, `stage_synth`, `quadrant_synth`
- `rank_delta` (yh − synth), `score_delta`, `rs_pct_delta`, `sctr_delta`
- `stage_match` (True/False)

Industries with `|rank_delta| > 10` are the most research-relevant divergences.

## Execution

```bash
python main.py --mode rs-percentile                   # yh (default)
python main.py --mode rs-percentile --source synth    # synthetic

python main.py --mode rrg                             # yh weekly RRG (default)
python main.py --mode rrg --source synth              # synthetic weekly RRG

python main.py --mode momentum                        # yh composite (default)
python main.py --mode momentum --source synth         # synthetic composite
python main.py --mode momentum-compare                # both → comparison CSV

python main.py --mode update                          # full pipeline, yh default
python main.py --mode update --source synth           # full pipeline, synthetic
```
