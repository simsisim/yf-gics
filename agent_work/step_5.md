# Step 5 — Cleanup + open questions for the user

Read `step_4.md` first. This is a small follow-up, not a new phase — one
confirmed cleanup task, plus a list of decisions that have come up across
this project but haven't been settled yet. Don't guess on the questions
below; report back and wait for an answer rather than picking one.

## A — Cleanup (confirmed, just do it)

Four definitions in `app.py` are orphaned leftovers from the two excluded
tabs (`tab_nhl`, `tab_screener`) — not a functional bug (nothing calls
them, nothing breaks), but dead code that should have gone with the tabs
that used them. Verified directly against upstream `stockCharts/app.py`:
every call site for all four lives inside `tab_nhl` or `tab_screener`
(lines 3011/3152/3157/3162 there, both past `tab_nhl`'s start at 2963) —
none of the 7 kept tabs reference any of them:

- `SCREEN_CATEGORIES` (the large RSI/MACD/Moving Averages/... dict)
- `_sector_filter_values()`
- `_market_cap_bucket_value()`
- `_INDEX_TAG_CODES`

Delete all four from `yf-gics/app.py`. Re-run the same regex/AST dead-
reference check used in step 4 afterward to confirm nothing else now
references them (it shouldn't — they were already confirmed unused).

**Leave these alone, even though they're also unused** — checked, they're
unused in *upstream* `stockCharts/app.py` too (pre-existing dead code
there, e.g. `load_sctr` superseded by `load_sctr_enriched`), not something
this port introduced or is responsible for cleaning up:
`load_sctr`, `load_index_benchmarks`, `_def_cols`.

## B — Open questions (need the user's answer, don't assume)

1. **Empty drill-downs for 3 industries.** Step 4's verification found
   Infrastructure Operations, Pharmaceutical Retailers, and Silver have
   zero member stocks in `sctr_rankings` — not a bug, the curated
   `tradingview_universe_yf.csv` universe (step 3's revision) simply
   contains no stock tagged with those Yahoo industry labels. Leave as
   empty drill-downs (matches "curated universe" being the deliberate
   choice), or is this worth widening the universe for specifically these
   three?

2. **Snapshot history is still shallow.** 8 dates in `industry_summary`/
   `sector_summary`, 5 in `sctr_rankings`. Date-range UI controls in the
   ported dashboard (compare-to-N-days-ago selectors, etc.) work but look
   sparse. Backfill a longer history now (how far back?), or let it
   accumulate naturally from daily runs going forward?

3. **RS Percentile (`src/rs_percentile.py`) — add it or not?** Raised
   earlier this session: a real IBD/Minervini-style RS Rating (1-99
   composite vs SPY, different from SCTR) already exists in yf-gics but
   isn't wired into `yf_industry_db.py`'s schema or the dashboard —
   stockCharts itself has no equivalent at all, so this would be going
   beyond parity, not filling a gap. Still undecided. If yes: as a new
   column on `industry_summary`/`sector_summary`, or its own addition to
   the dashboard?

4. **The missing-large-cap gap** (GOOGL, TSM, BRK-B, and others in
   `agent_work/missing_price_tickers.csv`) is confirmed out of scope for
   this project (a `downloadData_v1` ticker-universe fix, not yf-gics
   code) — no action needed here, just confirming this is still the
   answer and not something that quietly needs revisiting.

## Next

Report back on A once done. For B, list out the answers received (or
"still open") rather than assuming — no `step_6.md` until these are
resolved one way or another.
