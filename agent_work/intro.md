# Goal

Reconstruct a dashboard/SCTR system inside yf-gics equivalent to
`/home/imagda/_invest2024/python/stockCharts`, **excluding the 52-Week
Highs/Lows tab** (`tab_nhl` in stockCharts' `app.py`) — that tab is under
active development in the stockCharts project itself and is out of scope
here.

## Key differences from stockCharts

1. **stockCharts downloads already-computed data; it doesn't compute
   anything itself.** Its fetchers (`src/fetchers/*.py`) scrape
   stockcharts.com's own API for SCTR scores and Dow Jones industry index
   values that StockCharts already calculated, then just organize the result
   into a SQLite database (`src/db.py`) and CSV mirrors.

2. **This project computes everything itself, from raw price history**,
   because that data isn't available as a pre-computed download from Yahoo
   Finance. SCTR, industry-level indexes, and rankings all need to be built
   from OHLCV history already sitting in `downloadData_v1`.

3. **The industry/sector "index" input data is different.** stockCharts uses
   Dow Jones's `$DJUSxx` industry indices and the 11 SPDR sector ETFs. This
   project uses Yahoo Finance's own per-industry/per-sector index tickers
   instead — `^YH<code>` (e.g. `^YH31130020` = Semiconductors,
   `^YH311` = Technology sector). These are real, live Yahoo Finance
   `quoteType: INDEX` instruments (verified via `yfinance`), not something
   this project invented. 145 industry tickers + 11 sector tickers are
   already downloaded and verified clean (see `downloadData_v1/src/
   verify_index_tickers.py`) — no further data-source work needed, this is
   a solved problem.

4. **Per-stock SCTR uses the same three cap-bucket groups stockCharts uses:
   large / mid / small**, split by the same market-cap thresholds
   StockCharts itself uses ($10B / $2B / $250M — already encoded as
   `large_cap_min` / `mid_cap_min` / `small_cap_min` in yf-gics's
   `config.py`). This is already implemented in `src/sctr_stocks.py`.

5. **How does StockCharts compute SCTR for an industry index — on the index
   itself, or by averaging the SCTR of its member stocks?**
   *Answered.* Checked directly against stockCharts' own fetcher
   (`stockCharts/src/fetchers/sctr.py`): it calls StockCharts' API with
   `view=I` (DJ US Industries) and gets back one `SCTR` value per industry,
   **computed on the industry index's own price series** — the exact same
   SCTR formula applied to a single instrument, treating the industry index
   as if it were one stock. It is *not* an average of constituent stocks'
   individual SCTR scores. So for this project: run the standard SCTR
   formula (`src/sctr_engine.py`) directly on each `^YH<code>` index's own
   Close series — no different from how it's already applied to a stock.

6. **Is `yf.Industry(key).top_companies` (the constituent list Yahoo returns
   for an industry) ranked by market cap?**
   *Answered.* Verified live: the returned DataFrame is sorted descending by
   a `market weight` column — each company's share of the industry's total
   market cap (e.g. for Semiconductors: NVDA=0.49, AVGO=0.16, ...), and
   `yf.Industry(key).overview['market_cap']` gives the industry's total
   market cap. Yes — market-cap-weighted and market-cap-ranked. Note it only
   returns the top ~49 of the industry's full constituent count (e.g. 49 of
   59 for Semiconductors) — the largest names, which is what carries almost
   all the weight anyway.

7. **Once the database mirrors stockCharts' schema with equivalent data in
   it, the dashboard itself should be close to a 1:1 port** of
   `stockCharts/app.py` — its tabs mostly just read from the DB via a small
   set of query functions (see `app.py` lines ~40-140) and render; they
   don't know or care whether the underlying numbers came from a scrape or a
   local computation, as long as the schema and column meanings match.

8. **Are the index-membership flags in `tradingview_universe_yf.csv`
   (SP500, NASDAQ100, Russell1000, etc.) TradingView-based or Yahoo-based?
   Can Yahoo Finance provide/refresh this instead?**
   *Answered — this is a dead end via Yahoo, and out of scope for the SCTR
   dashboard work below; noted here only so it isn't re-investigated later.*
   - Those boolean columns are 100% TradingView-sourced today: they come
     from `tradingview_ticker_processor.py`'s `create_boolean_index_columns()`,
     which parses TradingView's own `Index` column (a string like
     `"S&P 500, Russell 3000, Russell 1000"`). Nothing from Yahoo is
     involved in these specific columns — they're a separate concern from
     the `sector_tv`/`sector_yf` split done in `enrich_universe_yf.py`.
   - Checked live, three separate ways, on NVDA: (1) `t.info` — all 189 keys
     read in full and manually reviewed (not a keyword filter) — the only
     S&P-500-related field is `SandP52WeekChange`, which is the *index's*
     own 52-week return used as a comparison benchmark, not a membership
     flag; (2) `t.fast_info` (17 keys) — no membership field; (3)
     `t.get_history_metadata()` — exchange/trading-hours metadata only,
     nothing there either. Yahoo's own website does show "Component of:
     S&P 500" on a stock's page, but that's rendered from a backend module
     `yfinance` doesn't expose through any of these surfaces — confirmed
     not reachable via `.info`, `.fast_info`, or history metadata.
   - Checked live: Yahoo doesn't expose full index constituent lists either.
     The closest thing is an index-tracking ETF's holdings
     (`yf.Ticker('SPY').funds_data.top_holdings`) — but that only returns
     the **top 10** holdings by weight, nowhere near the ~500 S&P 500
     members (unlike `yf.Industry`/`yf.Sector`'s `top_companies`, which
     covers ~80-95% of an industry's much smaller membership, e.g. 49/59 for
     Semiconductors — broad indices are a different scale problem).
   - **Conclusion: there is no reliable way to get or refresh full S&P
     500 / Russell 3000 / NASDAQ 100 / etc. membership from Yahoo Finance.**
   - **Decision: settled, no code change needed.** Keep the index-membership
     flags TradingView-sourced, as they are today. The only maintenance this
     needs is re-exporting `user_input/tradingview_universe.csv` from
     TradingView by hand every ~3 months or so (index reconstitutions are
     infrequent) and letting `tradingview_ticker_processor.py` +
     `enrich_universe_yf.py` reprocess it as usual — no automation to build,
     nothing further to investigate here.

## Where to start

See `step_1.md` for the first concrete implementation step.
