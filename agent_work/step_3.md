# Step 3 — sctr_rankings (stocks + industry) and benchmarks

**REVISION (after step 3 was already executed once):** the stock universe
source changed from `stocks_by_industry.csv` to `tradingview_universe_yf.csv`
— see the updated "foundation" section below for why. `src/yf_stock_compute.py`
needs to be re-pointed at the new universe source and rerun; the rest of
step 3's design (separate cap-bucket pools, `sctr_delta`, industry reshape,
`etf` group skipped, benchmarks) is unchanged.

Read `intro.md`, `step_1.md`, `step_2.md` first. Step 2 is done and verified:
`src/yf_industry_compute.py` populates `industry_summary`/`sector_summary` in
`data/yf_dashboard.db`, 145/11 rows per date, two-tier data-quality flagging
confirmed working against real cases.

Same rule as step 2: **write a fresh module, don't patch/reuse
`sctr_stocks.py` or `universe_loader.py`.** They're tangled in the
TradingView-vs-Yahoo taxonomy problem this project already spent real effort
resolving (see `intro.md` point 8's investigation) — `universe_loader.py`'s
`'sc'`/`'tv'` modes source sector/industry from TradingView or a
StockCharts-DB blend, neither of which is the Yahoo-native taxonomy `^YH`
and `industries.csv` use. Reusing them would reintroduce exactly the
drill-down mismatch already ruled out for the sector tier in step 2.

## What's already the right foundation for this: `tradingview_universe_yf.csv`

`/home/imagda/_invest2024/python/downloadData_v1/data/tickers/tradingview_universe_yf.csv`
(4,010 rows) — **not** `stocks_by_industry.csv` (18,809 rows). Reasoning,
checked live this session:

- `stocks_by_industry.csv` is built from Yahoo's full equity screener per
  industry (no cap — one industry has 1,004 members) and is much bigger than
  what's actually useful here: it includes many thinly-traded/illiquid
  names, and since it's a separate universe from what `downloadData_v1`
  actually downloads price history for, a large chunk of it (5,510 tickers,
  including large caps like GOOGL/TSM/BRK-B) has no local price data at all
  — a real, but out-of-scope-for-this-project gap (see below).
- `tradingview_universe_yf.csv` is the curated ~4,000-ticker universe this
  project already downloads full price history for, **and** it already
  carries Yahoo-native `sector_yf`/`industry_yf` columns (built by
  `downloadData_v1/src/enrich_universe_yf.py` earlier this session) — so it
  keeps the same taxonomy consistency as `stocks_by_industry.csv` without
  the bloat. Use `sector_yf`/`industry_yf` for classification (never
  `sector_tv`/`industry_tv` — confirmed genuinely different taxonomies
  earlier this session, e.g. AAPL is "Technology / Consumer Electronics" per
  Yahoo vs. "Electronic technology / Telecommunications equipment" per
  TradingView).
- Ticker column is `ticker` (not `symbol`).
- **Market cap for cap-bucket thresholds: use `tradingview_universe_yf.csv`'s
  `market_cap_yf` column, not its `Market capitalization` column.**
  `market_cap_yf` was added to that file this session (see
  `downloadData_v1/src/enrich_universe_yf.py`) — it's read from each ticker's
  own daily price file's `marketCap_asOfDownload` column, which
  `update_individual_stock_data()` refreshes on every incremental price
  update, unlike TradingView's `Market capitalization` (only as fresh as the
  last manual quarterly re-export — see `intro.md`). A stock sitting near a
  $10B/$2B/$250M bucket boundary could otherwise sit in the wrong bucket for
  a whole quarter. Since this now lives directly in the universe file, no
  per-ticker lookup is needed on the yf-gics side — just read the column
  when loading the universe, same as `sector_yf`/`industry_yf`. 115/4,010
  rows have `market_cap_yf` NaN (some overlap with the 83 missing
  `sector_yf`/`industry_yf`, some additional tickers whose price file
  predates the `marketCap_asOfDownload` field) — exclude those from
  bucketing, same as the NaN filter below. (Note: `data_loader.load_daily()`
  would *not* give you this column even if you tried reading it from the
  per-ticker price file directly — it deliberately filters every frame down
  to `_OHLCV = ['Open','High','Low','Close','Volume']` — but that's moot now
  since `market_cap_yf` is already sitting in the universe file.)
- Checked: this file already has `GOOG` but not `GOOGL`, and `BRK-A` but not
  `BRK-B` — TradingView's own curation already avoids most multi-share-class
  duplication. (For the record, if both classes of a stock ever do show up
  together somewhere: that's not a data bug to dedupe — Class A/C shares
  trade as genuinely separate securities with independent price/volume,
  and StockCharts itself tracks both when both are in its universe.)

**Filter before bucketing:** 83/4,010 rows have `sector_yf`/`industry_yf`
NaN (no local price file to source it from, at enrichment time) — exclude
those. Then bucket by `config.py`'s existing thresholds (`large_cap_min`=$10B,
`mid_cap_min`=$2B, `small_cap_min`=$250M — already defined, matches
StockCharts' own cutoffs).

**Known, accepted, out-of-scope gap:** major names like GOOGL, TSM, BRK-B,
SK Hynix, and ASML have no local price history because `downloadData_v1`'s
own ticker universe never included them. That's a `downloadData_v1`-side
fix (adding tickers there, downloading their history), not something this
project's code should try to work around. Don't build any special-case
handling for it here — if/when those tickers get added upstream, rerunning
`yf_stock_compute.py` picks them up automatically, no code changes needed.
Anything below `small_cap_min` gets no SCTR, same as StockCharts.

## Building `sctr_rankings`

Schema (from `src/yf_industry_db.py`, ported from stockCharts): one row per
`(snapshot_date, group_name, symbol)` — `group_name`, `symbol`, `name`,
`sector`, `industry`, `sctr`, `sctr_delta`, `close`, `volume`, `market_cap`.

### `group_name in {large, mid, small}`

For each bucket, separately: load each stock's daily Close
(`data_loader.load_daily`), compute SCTR (`sctr_engine.compute_indicators` +
`rank_to_sctr`), **rank within that bucket only** — large stocks compete
against large stocks, not against small caps (this is stockCharts'
methodology, and it's the same "don't mix pools" rule step 2 already
enforced for industries vs. sectors — apply it a third time here across the
three cap tiers). `sector`/`industry` come straight from
`tradingview_universe_yf.csv`'s `sector_yf`/`industry_yf` columns (not
`sector_tv`/`industry_tv`). `close`/`volume`/`market_cap` are real numbers
here (unlike the industry/sector tiers) since these are real stocks, not
composite indexes — use the actual latest Close and Volume from
`data_loader.load_daily()`, and `market_cap` from
`tradingview_universe_yf.csv`'s `market_cap_yf` column (same source as
cap-bucket assignment above, not its `Market capitalization` column).

~4,000 stocks (before the 83-row NaN filter) is a much smaller job than the
18,809-row universe originally planned here — should run comfortably.
`src/yf_industry_compute.py` from step 2 already had to solve "load and
compute over many symbols efficiently" for 156 tickers; the same
`data_loader.load_daily()` primitive is what both need, just reuse that
pattern rather than designing batching from nothing.

`sctr_delta`: change in SCTR since some prior reference point (stockCharts
calls this out as a real column, per the earlier fetcher research —
`sctr.py`'s `item.get('delta')`). Since this project now has multiple
snapshot_dates in `yf_industry_db.py` after step 2's backfill, compute it as
`sctr(today) - sctr(previous available snapshot_date for the same
symbol/group)` where available; `NULL` for the first date with no prior
snapshot to compare against.

### `group_name = 'industry'`

This is the *same* 145-industry SCTR already computed and stored in
`industry_summary` by step 2's `yf_industry_compute.py` — don't recompute
it, reshape it into `sctr_rankings`' row format. `close` = `last` from
`industry_summary`. `market_cap` = sum of `stocks_by_industry.csv`'s
`marketCap` per `industry_key` (industries don't have their own market cap
as a security, but StockCharts' real sample data does populate this field
for the industry group, so approximate it this way rather than leaving it
NULL). Deliberately use the fuller `stocks_by_industry.csv` for this one
aggregate sum, not the smaller `tradingview_universe_yf.csv` — this is a
"how big is this industry, total" figure, not a ranking universe, so
completeness matters more here than it does for the sctr_rankings pools
above. `volume` stays `NULL` (no real volume for a `^YH` index, same as
`industry_summary`).

### `group_name = 'etf'` — explicit scope decision, don't guess

stockCharts covers ~3,900 ETFs for this group. This project has no
equivalent broad ETF universe or a validated way to compute SCTR for
one beyond the 11 sector `^YH3XX` composites (already covered under
`sector_summary`, arguably redundant with an `etf` group here). **Skip this
group entirely for now** rather than inventing a partial ETF universe —
note it as explicitly out of scope in whatever you write next, don't
silently leave it half-done.

## Building `benchmarks`

Schema: `(snapshot_date, symbol, close)`. Target symbols: `^GSPC`, `^NDX`,
`^DJI`, `^RUT` (matches stockCharts' own benchmark set).

**Data-acquisition prerequisite, not a yf-gics problem:** checked this
session — `^NDX` and `^DJI` already have downloaded price history;
`^GSPC` and `^RUT` do not. This is the exact same situation the 11 sector
`^YH3XX` tickers were in earlier in this project. Fix it the same way: add
`^GSPC` and `^RUT` to
`/home/imagda/_invest2024/python/downloadData_v1/user_input/indexes_tickers.csv`
(they belong there — that file already holds `^NDX`, so this is consistent,
not a new pattern), then trigger
`python main.py --ticker-choice 5` in `downloadData_v1` to download them.
Do this before writing the benchmarks populator, not as an afterthought.

Once all 4 are on disk, this is a straightforward loop:
`data_loader.load_daily()` each, take the latest Close per `snapshot_date`,
`upsert_benchmarks()`.

## Rebuilding `universe`

Mechanical — call `YfIndustryDB.rebuild_universe()` (already ported
verbatim in step 1) after `sctr_rankings` is populated for at least one
date. No new code needed here.

## Verification

1. Confirm three separate ranking pools for large/mid/small — spot-check
   that a stock's rank is relative to its own bucket, not the full universe
   (e.g. pick a large-cap stock, confirm its rank denominator matches the
   large-cap bucket's size, not the full ~4,000-stock universe).
2. Confirm `group_name='industry'` rows in `sctr_rankings` match
   `industry_summary` exactly for the same date (same SCTR values — this is
   a reshape, not a recomputation, so any mismatch is a bug).
3. Confirm `benchmarks` has all 4 symbols for at least one date.
4. Run `rebuild_universe()`, confirm `load_universe()` returns a sensible
   row count and `group_name` distribution across large/mid/small.
5. Same idempotency check as step 2: re-run one date twice, confirm no
   duplicate rows.

## Next

`step_4.md` — the dashboard port. Not before this step's numbers are
checked and trusted (see `step_2.md`'s framing — this still applies).
