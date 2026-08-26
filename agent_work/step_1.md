# Step 1 — Port the database schema

Read `intro.md` first for context.

## Objective

Create `yf-gics/src/yf_industry_db.py`: the same table schema, upsert logic, and
read helpers as stockCharts' `src/db.py`, so later steps can populate it and
the ported dashboard (final step) can read from it with no query changes.

## What to reuse

Port `/home/imagda/_invest2024/python/stockCharts/src/db.py` essentially as
a copy. It is pure schema + SQL, with no stockCharts-specific business logic
in it — nothing here needs to change to fit this project:

- `parse_symbols()` — unchanged, generic ticker-list parsing.
- Table DDL: `sector_summary`, `industry_summary`, `sctr_rankings`,
  `benchmarks`, `universe` — copy column-for-column, identical
  `UNIQUE(...)` constraints.
- `_clean()` — unchanged.
- `StockChartsDB` class — copy all methods (`upsert_*`, `load_*`,
  `available_dates_*`, `rebuild_universe`, `export_universe_csv`). Rename
  the class if you like (e.g. `YfIndustryDB`), but keep every method signature and
  every SQL statement identical — the dashboard port in the last step
  depends on these being drop-in compatible with stockCharts' `app.py`
  query functions (see `app.py` lines ~40-140 for how they're called).

## What changes

- Default `db_path`: use a new file, e.g. `data/yf_dashboard.db` (don't
  write into stockCharts' own `data/stockcharts.db` — these are separate
  projects/databases).
- Nothing else. Do not add new columns or tables yet, even if a later step's
  data source doesn't cleanly fill every column (e.g. `^YH` index series
  have no real trading volume, so `sector_summary.volume` /
  `industry_summary`-equivalent volume fields may end up `NULL`/`0` —
  that's fine, leave the schema untouched and handle it in the populator
  step instead of changing the table).

## Verification

1. Import `yf_industry_db.py`, instantiate the DB class (creates the file + tables
   if missing).
2. Confirm all 5 tables exist with `sqlite3 data/yf_dashboard.db ".tables"`.
3. Round-trip smoke test: build a tiny fake DataFrame matching one upsert
   method's expected columns (e.g. `upsert_benchmarks` with 1-2 rows), call
   it, then call the matching `load_*` method and confirm the row comes
   back unchanged. Delete the test DB file/rows afterward if this was just a
   smoke test, not real data (real population happens in later steps).

## Next

`step_2.md` — populate `industry_summary` / `sector_summary` from `^YH`
data.
