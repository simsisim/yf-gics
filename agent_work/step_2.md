# Step 2 — Compute and track industry/sector data. No dashboard work here.

Read `intro.md` and `step_1.md` first for context. `step_1.md` should already
be done (`src/yf_industry_db.py` exists and is verified).

## Why this step is scoped the way it is

stockCharts solved the *presentation* problem (a working Streamlit
dashboard) but never had to solve the *computation* problem — it just
downloads numbers stockcharts.com already calculated. This project has to
solve computation from scratch. That's the hard, novel part, and it's the
priority. **The dashboard is explicitly out of scope until this step and
step_3 are done and the numbers in `yf_industry_db.py` have been checked and
trusted.** Don't get pulled toward UI work before there's real, validated
data to show.

**Don't build this by patching or chaining together the existing
`sctr_industry.py` / `perf_screener.py` / `industry_index.py` / `backfill.py`
/ `rs_percentile.py` / `stage_analysis.py` / `history_*.py` modules.** They're
from an earlier architecture and running one of them live during planning
already surfaced two real bugs on the first try (see below) — proof they
aren't a safe foundation to extend. Write one new, clean module instead.
Import only the pieces below that are genuinely validated primitives, not
orchestration logic:

- `src/data_loader.py::load_daily()` — pure I/O (archive/current tiering,
  batch-cache stitching). Keep using this, it's not business logic.
- `src/sctr_engine.py::compute_indicators()` / `rank_to_sctr()` — the actual
  ChartSchool SCTR formula, corrected twice across past sessions and
  cross-checked by the separate `dashboard-screener` project. This is the
  one piece of "old stuff" worth keeping outright.
- `src/yf_industry_db.py` (from step_1) — already new, clean code written for this
  project; not part of the old architecture.
- `industries.csv` / `stocks_by_industry.csv` — reference data, not code.

Everything else that computes industry/sector numbers today should be left
alone and ignored, not built upon.

## What running the old code once already proved, live

Running `sctr_industry.py` for real during planning returned **156** rows,
not 145 — it ranks the 145 industries and the 11 `^YH3XX` sector tickers
together in one pool, because both share the `^YH` prefix and nothing
separates them. Rank #1 (SCTR 99.9) was `^YH10200010` (Auto & Truck
Dealerships) — checked its actual Close series: it jumped from ~150k-range
to 245,966 on 2026-07-29 and never came back down (263,502 as of the latest
close, still climbing). That's a permanent step-change in Yahoo's own index
data, not a transient glitch — and since SCTR's two biggest components
(125-day ROC, %-above-200-day-EMA, 60% combined weight) are still measuring
"today vs. before the jump," a large chunk of that 99.9 score is very likely
the artifact, not organic strength. **Both of these need to be designed
around from the start in the new module, not bolted on after.**

## The new module

Write `src/yf_industry_compute.py` (or similar — pick one clear name). Design:

1. **Load `^YH` prices.** For each of the 145 industries (`industries.csv`'s
   `symbol` column) and the 11 sectors (`^YH3XX` — see `intro.md`), load
   daily Close via `data_loader.load_daily()`.
2. **Two separate ranking pools, always.** Compute SCTR
   (`compute_indicators` + `rank_to_sctr`) for the 145 industries as one
   pool, and the 11 sectors as a second, separate pool. Never combine them
   before ranking — this is the bug that just got caught live.
3. **Returns.** For each instrument, compute `chg_<period>` (absolute) and
   `pct_<period>` (percent) for periods `1d, 1w, 1m, 3m, 6m, 1y, ytd`,
   using calendar-day lookback (nearest available close on/before
   `as_of - N days`; `ytd` looks back to `as_of.replace(month=1, day=1)`
   instead of a fixed day-count). This is a simple, self-contained helper —
   a few lines, write it fresh in the new module rather than importing
   `perf_screener.py`'s version.
4. **Data-quality check, applied before anything gets trusted.** For each
   instrument, check for an implausible single-day jump in the trailing
   ~15 days (a real diversified industry index essentially never moves
   >15-20% in one session — `^YH10200010`'s +63% and the previously-observed
   `^YH31110020` +32% overnight are the confirmed real-world cases this is
   for). Flag, don't silently drop or "correct" — this project doesn't know
   the true value, only that the reported one is implausible. Since
   `yf_industry_db.py`'s schema (ported from stockCharts) has no column for this,
   log flagged symbols clearly (console output and/or a small companion
   CSV/log file) alongside the DB write, so a flagged instrument sitting at
   the top of a ranking is visible, not hidden.
5. **`child_count`.** For industries: count of `stocks_by_industry.csv` rows
   per `industry_key`. For sectors: count of `industries.csv` rows per
   `sector_key` (i.e. how many industries belong to that sector — matches
   the small 6-12 range in stockCharts' real sample data, not a stock
   count).
6. **Write to `yf_industry_db.py`.** Call `upsert_industry_summary()` /
   `upsert_sector_summary()` for the computed `snapshot_date`. Leave
   `volume`/`market_cap` `NULL` where `^YH` has no real trading data to
   report — don't fabricate a number.

Make this runnable as a script/CLI entrypoint that takes an `--as-of DATE`
(defaulting to latest available), so it can be run once for today **and**
re-run across a range of past trading dates to build up real tracking
history in `yf_industry_db.py` — that's the actual point of a *tracking* database:
multiple snapshot_dates accumulating over time, not just today's numbers.

## Verification

1. Run for one date, confirm exactly 145 industry rows and 11 sector rows
   land in their respective tables — not 156 in one pool.
2. Confirm `^YH10200010` (or whatever the current top-ranked industry is)
   shows up in the flagged-jump log if its score is still being driven by
   an unreverted jump — don't just check that the flag *exists* in code,
   check that it actually fires on this real case.
3. Backfill a handful of past trading dates (5-10) and confirm
   `available_dates_industry()` / `available_dates_sctr()`-style queries
   against `yf_industry_db.py` return all of them, with no duplicate rows on
   re-running the same date twice (upsert, not insert).
4. Spot-check 3-4 rows by hand: pick an industry, independently recompute
   its `pct_1m` and SCTR from the raw `^YH<code>.csv`, confirm it matches
   what's in `yf_industry_db.py`.

## Next

`step_3.md` — `sctr_rankings` (large/mid/small stock groups, using the same
"compute fresh, don't patch old modules" approach) and `benchmarks`. Still no
dashboard work.
