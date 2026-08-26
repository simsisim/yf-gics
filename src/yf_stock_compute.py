"""
Compute stock-level SCTR rankings (large/mid/small cap tiers) + benchmarks
→ YfIndustryDB (`sctr_rankings`, `benchmarks`, `universe`).

Fresh module per step_3's rule — does NOT reuse sctr_stocks.py /
universe_loader.py (TradingView-taxonomy entanglement).

Stock universe (step_3 REVISION): downloadData_v1's curated
tradingview_universe_yf.csv (~4k tickers — exactly what downloadData_v1
downloads full price history for), NOT stocks_by_industry.csv (Yahoo's raw
per-industry screener dump, 18.8k rows, mostly without local price data).
Classification uses its Yahoo-native `sector_yf`/`industry_yf` columns
(never `sector_tv`/`industry_tv` — genuinely different taxonomies), and cap
buckets use `market_cap_yf` (refreshed from each ticker's price file on
every incremental update; TradingView's own "Market capitalization" column
is only as fresh as the last manual quarterly re-export).
Known, accepted gap: names absent from downloadData_v1's universe entirely
(GOOGL, TSM, BRK-B, ASML, ...) stay unscored until added upstream — no
special-casing here; rerunning picks them up automatically.

Methodology (matches StockCharts):
  - Rows with sector_yf/industry_yf/market_cap_yf NaN are excluded.
  - Buckets by config.py thresholds: >= $10B large, >= $2B mid,
    >= $250M small, below $250M → no SCTR at all.
  - Each bucket is its own ranking pool — large caps compete against large
    caps only. Same "don't mix pools" rule step 2 enforced for
    industries-vs-sectors, applied across the three cap tiers.
  - group 'industry' is a RESHAPE of step 2's industry_summary rows (same
    SCTR values, not recomputed); market_cap approximated as the sum of
    member stocks' marketCap from stocks_by_industry.csv — deliberately the
    fuller screener file for this one aggregate ("how big is the industry
    in total"), completeness beats ranking-universe consistency here;
    volume stays NULL (^YH index has none).
  - group 'etf' is explicitly OUT OF SCOPE (step_3): no validated broad ETF
    universe exists here; don't half-invent one.

sctr_delta = SCTR vs the previous snapshot_date present in sctr_rankings for
the same (group_name, symbol); NULL on first appearance.

Benchmarks: ^GSPC, ^NDX, ^DJI, ^RUT latest close per snapshot date.

CLI:
  python -m src.yf_stock_compute                    # latest session
  python -m src.yf_stock_compute --as-of 2026-08-25
  python -m src.yf_stock_compute --last 5           # last 5 sessions
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from config import Config
from src.data_loader import load_daily
from src.sctr_engine import _MIN_DAILY as MIN_SCTR_BARS
from src.sctr_engine import compute_indicators, rank_to_sctr
from src.yf_industry_db import YfIndustryDB

logger = logging.getLogger(__name__)

BENCHMARKS = ["^GSPC", "^NDX", "^DJI", "^RUT"]

DEFAULT_DB_PATH = "data/yf_dashboard.db"

# Stock universe (downloadData_v1's enriched TradingView export)
TV_UNIVERSE_YF_CSV = Path(
    "/home/imagda/_invest2024/python/downloadData_v1/data/tickers/tradingview_universe_yf.csv"
)

# tradingview_universe_yf.csv column names
COL_SYM, COL_NAME = "ticker", "Description"
COL_CAP = "market_cap_yf"                 # NOT 'Market capitalization' (stale TV value)
COL_SECTOR, COL_INDUSTRY = "sector_yf", "industry_yf"   # NOT *_tv

# stocks_by_industry.csv column names (used ONLY for the industry-group
# aggregate market-cap sum — see docstring)
SBI_CSV_COL_KEY, SBI_CSV_COL_CAP = "industry_key", "marketCap"


# ---------------------------------------------------------------------------
# Universe + price loading
# ---------------------------------------------------------------------------

def load_universe() -> pd.DataFrame:
    """
    tradingview_universe_yf.csv filtered for usable rows and labeled with
    their cap tier ('large'/'mid'/'small').

    Filters (step_3 revision): rows with sector_yf/industry_yf NaN (no local
    price file at enrichment time) or market_cap_yf NaN/<= 0 are excluded.
    Below-small-cap rows are dropped entirely (they get no SCTR, same as
    StockCharts).
    """
    uni = pd.read_csv(TV_UNIVERSE_YF_CSV)
    n0 = len(uni)

    uni = uni[uni[COL_SECTOR].notna() & uni[COL_INDUSTRY].notna()].copy()
    uni = uni[uni[COL_CAP].notna() & (uni[COL_CAP] > 0)].copy()
    logger.info("Universe: %d rows -> %d usable (%d dropped: NaN sector_yf/"
                "industry_yf or market_cap_yf)", n0, len(uni), n0 - len(uni))

    cfg = Config()

    def tier(cap: float) -> str | None:
        if cap >= cfg.large_cap_min:
            return "large"
        if cap >= cfg.mid_cap_min:
            return "mid"
        if cap >= cfg.small_cap_min:
            return "small"
        return None

    uni["group_name"] = uni[COL_CAP].map(tier)
    uni = uni[uni["group_name"].notna()].copy()

    # NOT NULL columns downstream — never let NaN strings through
    for c in [COL_NAME, COL_SECTOR, COL_INDUSTRY]:
        uni[c] = uni[c].fillna("")
    dist = uni["group_name"].value_counts().reindex(["large", "mid", "small"])
    logger.info("Cap buckets: %s", dist.to_dict())
    return uni


def load_industry_cap_sums(config: Config) -> pd.Series:
    """
    Total member market-cap per industry_key, summed from stocks_by_industry.csv
    (deliberately the fuller screener file — see module docstring). Feeds the
    'industry' group's market_cap field only.
    """
    sbi = pd.read_csv(config.stocks_by_industry_csv,
                      usecols=[SBI_CSV_COL_KEY, SBI_CSV_COL_CAP])
    sbi = sbi[sbi[SBI_CSV_COL_CAP].notna() & (sbi[SBI_CSV_COL_CAP] > 0)]
    return sbi.groupby(SBI_CSV_COL_KEY)[SBI_CSV_COL_CAP].sum()


def load_quotes(symbols: list[str], config: Config) -> dict[str, tuple[pd.Series, pd.Series]]:
    """
    Load (Close, Volume) daily series per symbol via data_loader.load_daily().
    Deduped, sorted ascending, floats. Logs progress every 500 symbols.
    """
    quotes: dict[str, tuple[pd.Series, pd.Series]] = {}
    t0 = time.time()
    for i, sym in enumerate(symbols, 1):
        df = load_daily(sym, config.daily_dir)
        if df is None or df.empty or "Close" not in df.columns:
            quotes[sym] = (pd.Series(dtype=float), pd.Series(dtype=float))
        else:
            close = df["Close"].astype(float).dropna()
            vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(dtype=float)
            close = close[~close.index.duplicated(keep="last")].sort_index()
            vol = vol[~vol.index.duplicated(keep="last")].sort_index()
            quotes[sym] = (close, vol)
        if i % 500 == 0:
            logger.info("  loaded %d/%d (%.0fs)", i, len(symbols), time.time() - t0)
    logger.info("Loaded quotes for %d symbols in %.0fs", len(symbols), time.time() - t0)
    return quotes


def session_calendar(quotes: dict[str, tuple[pd.Series, pd.Series]]) -> pd.DatetimeIndex:
    """Union of all session dates across every loaded close series, ascending."""
    idx: set = set()
    for close, _ in quotes.values():
        idx.update(close.index)
    return pd.DatetimeIndex(sorted(idx))


def snap_to_session(ts: pd.Timestamp, calendar: pd.DatetimeIndex) -> pd.Timestamp:
    le = calendar[calendar <= ts]
    if len(le) == 0:
        raise ValueError(f"No session on/before {ts.date()} in the loaded data")
    return le[-1]


# ---------------------------------------------------------------------------
# Snapshot computation
# ---------------------------------------------------------------------------

def _stock_rows_for_bucket(
    bucket: str,
    universe: pd.DataFrame,
    quotes: dict[str, tuple[pd.Series, pd.Series]],
    as_of: pd.Timestamp,
) -> list[dict]:
    """SCTRed rows for ONE cap tier — ranked within the bucket only."""
    sub = universe.loc[universe["group_name"] == bucket]
    raws: dict[str, float] = {}
    meta: dict[str, dict] = {}

    for r in sub[[COL_SYM, COL_NAME, COL_SECTOR, COL_INDUSTRY, COL_CAP]].itertuples(index=False):
        close, vol = quotes[r.ticker]
        trunc = close.loc[:as_of]
        if trunc.empty:
            continue
        comps = compute_indicators(pd.DataFrame({"Close": trunc}))
        if comps is None:
            continue                      # insufficient history -> no SCTR
        raws[r.ticker] = comps["raw_score"]
        last_dt = trunc.index[-1]
        v = vol.loc[last_dt] if last_dt in vol.index else None
        meta[r.ticker] = {
            "name": r.Description,
            "sector": r.sector_yf,
            "industry": r.industry_yf,
            "market_cap": float(r.market_cap_yf),
            "last": float(trunc.iloc[-1]),
            "volume": float(v) if v is not None and pd.notna(v) else None,
        }

    sctr = rank_to_sctr(pd.Series(raws, dtype=float))
    rows = []
    for sym, raw in raws.items():
        m = meta[sym]
        rows.append({
            "group": bucket,
            "symbol": sym,
            "name": m["name"],
            "sector": m["sector"],
            "industry": m["industry"],
            "sctr": float(sctr[sym]),
            "sctr_delta": None,           # filled against prior snapshot later
            "close": m["last"],
            "volume": m["volume"],
            "market_cap": m["market_cap"],
        })
    return rows


def _industry_rows(
    db: YfIndustryDB,
    as_of: pd.Timestamp,
    industries: pd.DataFrame,
    ind_cap_sum: pd.Series,
) -> list[dict]:
    """
    Reshape step 2's industry_summary rows into sctr_rankings format —
    same SCTR values, deliberately NOT recomputed.

    ind_cap_sum — per-industry_key member market-cap sums from
                  load_industry_cap_sums() (stocks_by_industry.csv).
    """
    ind_sum = db.load_date_industry(as_of.date().isoformat())
    if ind_sum.empty:
        logger.warning("[%s] industry_summary empty — run src.yf_industry_compute first",
                       as_of.date())
        return []

    key_of = industries.set_index("symbol")["industry_key"]

    rows = []
    for _, row in ind_sum.iterrows():
        sym = row["symbol"]
        ind_key = key_of.get(sym)
        mc = float(ind_cap_sum[ind_key]) if ind_key is not None and ind_key in ind_cap_sum.index else None
        rows.append({
            "group": "industry",
            "symbol": sym,
            "name": row["name"],
            "sector": row["sector"],
            "industry": row["name"],
            "sctr": float(row["sctr"]) if pd.notna(row["sctr"]) else None,
            "sctr_delta": None,
            "close": float(row["last"]) if pd.notna(row["last"]) else None,
            "volume": None,               # ^YH carries no real volume
            "market_cap": mc,
        })
    return rows


def _benchmark_rows(
    quotes: dict[str, tuple[pd.Series, pd.Series]],
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    recs = []
    for sym in BENCHMARKS:
        close, _ = quotes.get(sym, (pd.Series(dtype=float), pd.Series(dtype=float)))
        trunc = close.loc[:as_of]
        if trunc.empty:
            logger.warning("No benchmark data for %s on/before %s", sym, as_of.date())
            continue
        recs.append({"snapshot_date": as_of.date().isoformat(),
                     "symbol": sym, "close": float(trunc.iloc[-1])})
    return pd.DataFrame(recs)


def _prior_sctr_map(db: YfIndustryDB, as_of: pd.Timestamp) -> dict[tuple[str, str], float]:
    """{(group_name, symbol): sctr} from the latest snapshot_date strictly before as_of."""
    row = db._conn().execute(
        "SELECT MAX(snapshot_date) FROM sctr_rankings WHERE snapshot_date < ?",
        (as_of.date().isoformat(),),
    ).fetchone()
    if not row or row[0] is None:
        return {}
    df = pd.read_sql_query(
        "SELECT group_name, symbol, sctr FROM sctr_rankings WHERE snapshot_date = ?",
        db._conn(), params=(row[0],),
    )
    return {(g, s): float(v) for g, s, v in df.itertuples(index=False) if pd.notna(v)}


def apply_deltas(rows: list[dict], prior: dict[tuple[str, str], float]) -> None:
    """Fill sctr_delta in place vs the prior snapshot (None when absent)."""
    for r in rows:
        if r["sctr"] is None:
            r["sctr_delta"] = None
            continue
        prev = prior.get((r["group"], r["symbol"]))
        r["sctr_delta"] = round(r["sctr"] - prev, 1) if prev is not None else None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_snapshot(
    as_of: pd.Timestamp,
    universe: pd.DataFrame,
    quotes: dict[str, tuple[pd.Series, pd.Series]],
    db: YfIndustryDB,
    industries: pd.DataFrame,
    ind_cap_sum: pd.Series,
) -> int:
    """Compute + persist one snapshot date's sctr_rankings and benchmarks."""
    t0 = time.time()
    prior = _prior_sctr_map(db, as_of)

    rows: list[dict] = []
    for bucket in ("large", "mid", "small"):
        b_rows = _stock_rows_for_bucket(bucket, universe, quotes, as_of)
        logger.info("[%s] %s: %d stocks ranked", as_of.date(), bucket, len(b_rows))
        rows.extend(b_rows)

    ind_rows = _industry_rows(db, as_of, industries, ind_cap_sum)
    logger.info("[%s] industry: %d rows reshaped from industry_summary",
                as_of.date(), len(ind_rows))
    rows.extend(ind_rows)

    apply_deltas(rows, prior)
    df = pd.DataFrame(rows)
    db.upsert_sctr(df, snapshot_date=as_of.date().isoformat())

    bench = _benchmark_rows(quotes, as_of)
    if not bench.empty:
        db.upsert_benchmarks(bench)

    logger.info("[%s] done: %d ranking rows, %d benchmark rows (%.0fs)",
                as_of.date(), len(df), len(bench), time.time() - t0)
    return len(df)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--as-of", action="append", default=[],
                    help="snapshot date YYYY-MM-DD (repeatable; snaps to last session on/before)")
    ap.add_argument("--last", type=int, default=None, metavar="N",
                    help="run the last N trading sessions instead of --as-of")
    ap.add_argument("--db", default=DEFAULT_DB_PATH, help=f"SQLite path (default {DEFAULT_DB_PATH})")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    config = Config()
    universe = load_universe()
    industries = pd.read_csv(config.industries_csv)
    ind_cap_sum = load_industry_cap_sums(config)

    symbols = sorted(set(universe[COL_SYM]) | set(BENCHMARKS))
    logger.info("Loading quotes: %d stocks + %d benchmarks...", len(universe), len(BENCHMARKS))
    quotes = load_quotes(symbols, config)

    calendar = session_calendar(quotes)

    # Coverage visibility: even in the curated universe a few tickers can lack
    # local price data, and young listings have <210 bars. Report both every
    # run so the effective scored universe is never a surprise.
    n_empty = sum(1 for c, _ in quotes.values() if c.empty)
    n_short = sum(1 for c, _ in quotes.values()
                  if not c.empty and len(c) < MIN_SCTR_BARS)
    logger.info("Coverage: %d symbols | no local price data: %d | <210 bars: %d | scoreable: %d",
                len(quotes), n_empty, n_short, len(quotes) - n_empty - n_short)

    if args.last is not None:
        targets = list(calendar[-args.last:])
    elif args.as_of:
        targets = sorted({snap_to_session(pd.Timestamp(d), calendar) for d in args.as_of})
    else:
        targets = [calendar[-1]]
    logger.info("Snapshot dates: %s", [str(t.date()) for t in targets])

    db = YfIndustryDB(args.db)
    for t in targets:
        run_snapshot(t, universe, quotes, db, industries, ind_cap_sum)

    n = db.rebuild_universe()
    dist = pd.read_sql_query(
        "SELECT group_name, COUNT(*) AS n FROM universe GROUP BY group_name ORDER BY group_name",
        db._conn(),
    )
    logger.info("Universe rebuilt: %d symbols | %s", n, dist.to_dict("records"))

    nd = db._conn().execute("SELECT COUNT(DISTINCT snapshot_date) FROM sctr_rankings").fetchone()[0]
    nb = db._conn().execute("SELECT COUNT(DISTINCT snapshot_date) FROM benchmarks").fetchone()[0]
    logger.info("Done. DB=%s | sctr_rankings dates=%d | benchmark dates=%d", args.db, nd, nb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
