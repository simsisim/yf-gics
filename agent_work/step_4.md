# Step 4 — Port the dashboard

Read `intro.md`, `step_1.md`, `step_2.md`, `step_3.md` first. Steps 2-3 are
done and verified: `data/yf_dashboard.db` has real, checked data in
`industry_summary`, `sector_summary`, `sctr_rankings`, `benchmarks`,
`universe`.

## `stockCharts` is a moving target — verify before trusting anything below

It's an actively-developed sister project (the user is working on it in
parallel). Every line number and tab count in this doc was true when
written, and was **already stale once during this same session** — the tab
count grew from 8 to 9 (a whole new "Screener" tab appeared) between the
first time this project was investigated and now. **Before doing anything
else, re-run:**

```
grep -n "^with tab_\|^tab_.*=.*st\.tabs" /home/imagda/_invest2024/python/stockCharts/app.py
```

and work from whatever that actually shows, not from the tab list below.

## What to port and what to exclude — a principle, not a fixed list

`intro.md`'s founding observation: stockCharts downloads data it doesn't
compute itself. That's the exclusion rule, applied per-tab: **any tab whose
data comes from something stockCharts downloads/scrapes from
stockcharts.com that this project has no pipeline to reproduce gets
excluded.** This currently means two tabs, for two different reasons:

- **`tab_nhl` ("📈 52W Highs/Lows")** — excluded per explicit user
  instruction from the start of this project (`intro.md`). Reads
  `available_dates_nhl()`/`load_date_nhl()`/`load_nhl_counts()`, all backed
  by data this project was never asked to compute.
- **`tab_screener` ("🧪 Screener")** — new discovery this session, same
  exclusion principle applies: it's built entirely around "StockCharts'
  predefined technical screens (141 daily-archived scans)" — its own
  docstring says so. Backed by `available_scan_report_dates()` /
  `load_scan_report()` / `load_scan_defs()` / `build_screen_universe()` /
  `available_index_codes()` / `load_sctr_universe_for_screener()` — all
  reading scraped stockcharts.com scan results this project has no
  equivalent for. Exclude for the same reason as `tab_nhl`, not because
  anyone said "no" to it specifically.

**If stockCharts has grown a tenth tab by the time this step actually runs,
apply the same test to it** — does its data ultimately come from a
stockcharts.com scrape/scan this project doesn't compute, or from
`sector_summary`/`industry_summary`/`sctr_rankings`/`benchmarks`/`universe`-
shaped data this project already has? Exclude the former, port the latter.

**As of this writing**, that leaves 7 tabs to port: `tab_sector`
("🌐 Sector Ranks"), `tab_ranks` ("🏆 Industry Ranks"), `tab_sctr`
("⚡ SCTR"), `tab_theme` ("📊 Visual Tracker"), `tab_heatmap`
("🔥 Leaders Heatmap"), `tab_rotation` ("🔄 Rotation Radar"), `tab_leaders`
("🔍 Industry Leaders").

## Why this port should be closer to mechanical than a rewrite

Checked directly: `app.py` does **not** go through the `StockChartsDB`
class in `db.py` at all — every loader function
(`load_industry`, `load_all_industry`, `load_sctr`, `load_sctr_enriched`,
`load_industry_stocks`, `load_sctr_leaders`, `load_benchmark_series`, etc.)
opens its own `sqlite3.connect(DB_PATH)` and runs raw SQL directly against
column names — `symbol, name, sector, industry, sctr, sctr_delta, close,
volume, market_cap, snapshot_date, group_name` — that are column-for-column
identical to `src/yf_industry_db.py`'s schema (verified in step 1, it's a
byte-faithful port of the same DDL). There is exactly **one** line tying the
whole file to stockCharts' own database:

```python
DB_PATH = "data/stockcharts.db"
```

So the mechanical part of this port really is: copy the file, change that
one line to `"data/yf_dashboard.db"`, delete the two excluded tabs and their
dedicated helper functions (`available_dates_nhl`, `load_date_nhl`,
`load_nhl_counts`, `build_nhl_trend_chart`, `available_scan_report_dates`,
`load_scan_report`, `load_scan_defs`, `build_screen_universe`,
`available_index_codes`, `load_sctr_universe_for_screener`, plus their entry
in the `st.tabs([...])` list and unpacking assignment). Everything else —
the 7 tabs, their shared helper functions, the chart-building functions —
should work against `yf_dashboard.db` without further changes, *if* the
data genuinely matches shape. That "if" still needs proving per tab, not
assumed:

1. One thing worth checking specifically: `load_industry_stocks` filters
   `sctr_rankings WHERE industry = ?` using the *exact string* passed in
   from `industry_summary.name`. Confirm `sctr_rankings.industry` values
   (set in step 3 from `tradingview_universe_yf.csv`'s `industry_yf`) match
   `industry_summary.name` (set in step 2 from `industries.csv`'s
   `industry_name`) character-for-character — if they diverge even slightly
   (e.g. one has a trailing space, different capitalization), the
   sector→industry→stock drill-down breaks silently (empty stock lists, not
   an error).
2. `available_dates_industry()`/`available_dates_sctr()` assume enough
   historical snapshot_dates exist for whatever date-range UI each tab
   offers (e.g. a "compare to N days ago" selector). Step 2 has 8 dates,
   step 3 has 5 — some UI controls may look sparse or need a wider backfill
   before they're genuinely useful, not necessarily broken.
3. `load_sctr_leaders`'s persistence-filter CTE and `load_sctr_enriched`'s
   benchmark-ratio adjustment both assume `sctr_rankings` has enough
   distinct `snapshot_date`s to look back over — same caveat as above.

## Where the ported file should live

Recommend: **replace** yf-gics's existing `app.py`, don't add a
differently-named file alongside it. The current `app.py` already reads
`rotation_severity_*.csv`/`momentum_screen_*.csv` (step 2 deleted the
modules that produced those) — it's already broken, not a working thing to
preserve in parallel. If there's a reason to keep the old one around for
reference, `git mv` it to something like `app_legacy.py` first rather than
leaving two "app.py"-shaped things with unclear precedence.

## Verification

1. `streamlit run app.py`, click through all 7 ported tabs, confirm each
   renders without a traceback and shows real numbers (not empty tables).
2. Specifically test the sector-drill-down flow end to end: pick a sector on
   `tab_sector`, drill into one of its industries, confirm the stock list
   that appears is non-empty and its industry matches what was clicked (see
   the string-matching risk above).
3. Confirm `tab_nhl` and `tab_screener` (and anything else excluded per the
   principle above) are genuinely gone from the tab bar, not just hidden/
   erroring silently.
4. Spot-check 2-3 numbers on screen against a direct `sqlite3` query against
   `yf_dashboard.db`, same as every previous step's verification — the
   dashboard should show exactly what's in the database, no silent
   transformation.

## Next

None planned yet — this was the last step in the original outline
(`intro.md`). Once this is verified, report back rather than assuming
there's a `step_5.md` to look for.
