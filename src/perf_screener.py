"""
Performance Screener — simple flat multi-period returns + dual-benchmark
relative return, for the 145 GICS industries and the 11 sector SPDR ETFs.

Fills three gaps the rest of the pipeline doesn't cover:
  1. QQQ as a second RS/relative-return benchmark (rs_percentile.py only uses SPY).
  2. Plain calendar-day % return for 1w/2w/1m/3m/6m (rs_percentile.py only has
     IBD-style non-overlapping trading-day quarters folded into one composite).
  3. SCTR for the 11 sector SPDR ETFs themselves (sctr_industry.py only scores
     the 145 ^YH industry indexes; sector-level view previously only existed
     as breadth *counts* rolled up from industries, in breadth.py).

Universe:
  - 145 industries, priced via their ^YH index (same series sctr_industry.py
    uses — see src/industry_index.py).
  - 11 sector SPDR ETFs (XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY),
    priced directly (they're real tradable tickers, not synthetic).
  - SPY and QQQ, included as reference rows (their own relret vs themselves
    is always 0 — a built-in sanity check) but excluded from RS ranking pools
    and left without an SCTR (no natural peer group of two).

RS is a straight percentile rank of raw return, computed *within* each level
(industries ranked against industries, sectors against sectors) — not
IBD-style composite, and not mixed across levels.

Output file: results/perf_screener_YYYY-MM-DD.csv
"""

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from config import Config
from src.data_loader import load_daily
from src.sctr_engine import compute_indicators, rank_to_sctr
import src.sctr_industry as sctr_industry

logger = logging.getLogger(__name__)

BENCHMARKS = ["SPY", "QQQ"]

# Calendar-day lookback per period (nearest available close on/before this
# many days before the as-of date is used).
PERIODS = {
    "1w": 7,
    "2w": 14,
    "1m": 30,
    "3m": 91,
    "6m": 182,
}

# Sector SPDR ETF -> yf-gics sector_key (industries.csv uses the same 11 keys)
SECTOR_ETF_MAP = {
    "XLK":  "technology",
    "XLF":  "financial-services",
    "XLY":  "consumer-cyclical",
    "XLI":  "industrials",
    "XLV":  "healthcare",
    "XLP":  "consumer-defensive",
    "XLE":  "energy",
    "XLRE": "real-estate",
    "XLB":  "basic-materials",
    "XLC":  "communication-services",
    "XLU":  "utilities",
}


def _last_close_on_or_before(series: pd.Series, target: pd.Timestamp):
    valid = series[series.index <= target]
    if valid.empty:
        return None, None
    return valid.index[-1], valid.iloc[-1]


# A real diversified industry index essentially never moves this much in one
# session. A hit here almost always means an index rebase/reconstitution
# artifact or a bad print in the source data, not a real market event -- e.g.
# ^YH10200010 (Auto & Truck Dealerships, 60 constituents) jumped +63%
# overnight on 2026-07-29 and *stayed* at the new level, which inflated its
# 1w/1m return figures until this check caught it. Verified against a fresh,
# independent yfinance pull for all 7 tickers flagged in production: 6 of 7
# reproduce byte-for-byte from Yahoo directly (so it's Yahoo's own index data,
# not a bug in this repo's download/stitch pipeline), and none of the 7 are
# thin/illiquid industries (constituent counts ranged 17-926, several in the
# hundreds) -- so this isn't "one stock dominates a tiny sample" either; it
# looks like Yahoo's industry-index methodology itself isn't a continuously
# chain-linked index. Flagged, not silently dropped or corrected -- this
# project doesn't know the true value, only that the reported one is
# implausible.
DATA_QUALITY_LOOKBACK_DAYS = 15
DATA_QUALITY_JUMP_THRESHOLD = 20.0  # % move between calendar-adjacent rows that triggers a flag
DATA_QUALITY_MAX_GAP_DAYS = 5       # rows further apart than this aren't "adjacent" (see below)
DATA_QUALITY_GAP_FLAG_DAYS = 30     # a gap this large within the lookback window is itself flagged


def _max_single_day_move(series: pd.Series, lookback: int = DATA_QUALITY_LOOKBACK_DAYS):
    """
    Largest absolute % change between calendar-adjacent rows (gap <=
    DATA_QUALITY_MAX_GAP_DAYS, allowing for weekends/holidays) in the
    trailing `lookback` rows, and the date it occurred. (None, None) if no
    such pair exists.

    Gap-aware on purpose: a naive `series.tail(n).pct_change()` treats any
    two consecutive *rows* as adjacent regardless of how far apart their
    dates are, so a ticker with a multi-month data gap (e.g.
    ^YH31040020 / Infrastructure Operations, which jumps straight from
    2026-03-24 to 2026-07-31 in this repo's data) would misreport that gap as
    an implausible "single-day move" -- it's a missing-data problem, not a
    price move, and needs a different fix (see _max_recent_gap below).
    """
    tail = series.tail(lookback + 1)
    if len(tail) < 2:
        return None, None
    gap_days = tail.index.to_series().diff().dt.days
    pct = tail.pct_change() * 100.0
    valid = pct[(gap_days <= DATA_QUALITY_MAX_GAP_DAYS) & pct.notna()]
    if valid.empty:
        return None, None
    idx = valid.abs().idxmax()
    return float(valid.loc[idx]), idx


def _max_recent_gap(series: pd.Series, lookback_rows: int = 10):
    """
    Largest calendar-day gap between consecutive rows in the trailing
    `lookback_rows` rows. (None, None) if not enough data.

    Positional, not date-filtered: filtering to "the last N *days*" first
    would only look inside a window that's already assumed gap-free, so a
    gap bigger than the window (^YH31040020 has one ~129 days wide) would
    simply exclude the lone row on the far side and vanish undetected. A
    positional tail always keeps whatever the most recent rows actually are,
    however far back the one before the newest actually falls.
    """
    tail = series.tail(lookback_rows)
    if len(tail) < 2:
        return None, None
    gaps = tail.index.to_series().diff().dt.days.dropna()
    if gaps.empty:
        return None, None
    idx = gaps.idxmax()
    return int(gaps.loc[idx]), idx


def _build_universe(config: Config) -> pd.DataFrame:
    industries = pd.read_csv(config.industries_csv)
    sector_names = industries[["sector_key", "sector_name"]].drop_duplicates().set_index("sector_key")["sector_name"]

    rows = []
    for _, r in industries.iterrows():
        rows.append({
            "symbol": r["symbol"], "level": "industry",
            "sector_key": r["sector_key"], "sector_name": r["sector_name"],
            "industry_key": r["industry_key"], "industry_name": r["industry_name"],
        })
    for etf, sector_key in SECTOR_ETF_MAP.items():
        rows.append({
            "symbol": etf, "level": "sector",
            "sector_key": sector_key, "sector_name": sector_names.get(sector_key, ""),
            "industry_key": "", "industry_name": "",
        })
    for bench in BENCHMARKS:
        rows.append({
            "symbol": bench, "level": "benchmark",
            "sector_key": "", "sector_name": "", "industry_key": "", "industry_name": "",
        })
    return pd.DataFrame(rows)


def _compute_returns(symbols: list[str], daily_dir: Path, as_of: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        daily = load_daily(symbol, daily_dir)
        row = {"symbol": symbol}
        series = daily["Close"].dropna().sort_index() if (daily is not None and not daily.empty) else None

        if series is None or series.empty:
            latest_date, latest_close = None, None
        else:
            latest_date, latest_close = _last_close_on_or_before(series, as_of)

        row["close"] = latest_close
        row["price_date"] = latest_date

        if series is not None and not series.empty:
            flags = []
            move, move_date = _max_single_day_move(series)
            if move is not None and abs(move) >= DATA_QUALITY_JUMP_THRESHOLD:
                flags.append(
                    f"{move:+.0f}% move on {move_date.date()} (vs. the prior available close) "
                    f"in trailing {DATA_QUALITY_LOOKBACK_DAYS}d -- implausible for a diversified "
                    f"index, verify source data"
                )
            gap, gap_end = _max_recent_gap(series)
            if gap is not None and gap > DATA_QUALITY_GAP_FLAG_DAYS:
                flags.append(f"{gap}-day data gap ending {gap_end.date()} -- history incomplete near this date")
            row["data_flag"] = "; ".join(flags) if flags else None
        else:
            row["data_flag"] = None

        for period, days in PERIODS.items():
            if latest_close is None:
                row[f"return_{period}"] = None
                continue
            _, past_close = _last_close_on_or_before(series, as_of - pd.Timedelta(days=days))
            if past_close is None or past_close == 0:
                row[f"return_{period}"] = None
            else:
                row[f"return_{period}"] = (latest_close / past_close - 1.0) * 100.0

        rows.append(row)
    return pd.DataFrame(rows)


def _add_rs_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Percentile-rank return within each level's own peer group (industries
    ranked against industries, sectors against sectors). Benchmark rows are
    excluded from ranking (no peer group of two)."""
    df = df.copy()
    for period in PERIODS:
        col, rs_col = f"return_{period}", f"rs_{period}"
        df[rs_col] = pd.NA
        for level in ("industry", "sector"):
            mask = df["level"] == level
            df.loc[mask, rs_col] = (df.loc[mask, col].rank(pct=True, na_option="keep") * 99).round(1)
    return df


def _add_relative_return_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for benchmark in BENCHMARKS:
        bench_rows = df[df["symbol"] == benchmark]
        if bench_rows.empty:
            raise ValueError(f"Benchmark {benchmark!r} not found in the universe table")
        bench_row = bench_rows.iloc[0]
        for period in PERIODS:
            col = f"return_{period}"
            bench_return = bench_row[col]
            out_col = f"relret_{benchmark.lower()}_{period}"
            df[out_col] = None if pd.isna(bench_return) else df[col] - bench_return
    return df


def _sector_etf_sctr(config: Config, as_of: date | None) -> pd.DataFrame:
    """SCTR for the 11 sector SPDR ETFs, ranked against each other, using the
    same formula (sctr_engine) as industry/stock SCTR elsewhere in this repo."""
    records = []
    for etf in SECTOR_ETF_MAP:
        daily = load_daily(etf, config.daily_dir)
        if daily is None or daily.empty:
            continue
        if as_of is not None:
            daily = daily[daily.index <= pd.Timestamp(as_of)]
            if daily.empty:
                continue
        result = compute_indicators(daily)
        if result is None:
            continue
        records.append({"symbol": etf, **result})

    if not records:
        return pd.DataFrame(columns=["symbol", "sctr"])

    df = pd.DataFrame(records)
    raw = df.set_index("symbol")["raw_score"]
    df["sctr"] = rank_to_sctr(raw).values
    return df[["symbol", "sctr"]]


def run(config: Config, as_of: date | None = None) -> pd.DataFrame:
    universe = _build_universe(config)
    as_of_ts = pd.Timestamp(as_of) if as_of else _latest_industry_date(config)

    logger.info(f"Perf screener: {len(universe)} tickers "
                f"({universe['level'].value_counts().to_dict()}), as-of {as_of_ts.date()}")

    ret_df = _compute_returns(universe["symbol"].tolist(), config.daily_dir, as_of_ts)
    merged = universe.merge(ret_df, on="symbol", how="left")

    merged = _add_rs_columns(merged)
    merged = _add_relative_return_columns(merged)

    industry_sctr = sctr_industry.run(config, as_of=as_of)
    sector_sctr = _sector_etf_sctr(config, as_of)
    sctr_lookup = pd.concat([
        industry_sctr[["symbol", "sctr"]] if not industry_sctr.empty else pd.DataFrame(columns=["symbol", "sctr"]),
        sector_sctr,
    ], ignore_index=True)
    merged = merged.merge(sctr_lookup, on="symbol", how="left")

    cols = (
        ["symbol", "level", "sector_key", "sector_name", "industry_key", "industry_name",
         "sctr", "close", "price_date", "data_flag"]
        + [f"return_{p}" for p in PERIODS]
        + [f"rs_{p}" for p in PERIODS]
        + [f"relret_{b.lower()}_{p}" for b in BENCHMARKS for p in PERIODS]
    )
    merged = merged[cols].sort_values(["level", "sctr"], ascending=[True, False]).reset_index(drop=True)
    return merged


def _latest_industry_date(config: Config) -> pd.Timestamp:
    """
    Anchor the whole report on the industries' own latest date, not SPY/QQQ's.

    ^YH industry index files are never batch-supplemented (see data_loader.py:
    "batch files don't contain them"), so they can lag several calendar days
    behind SPY/QQQ/sector ETFs, which do get same-day batch updates. Anchoring
    on the fresher date would silently truncate every industry's "latest
    close" to a stale one while still subtracting a lookback of N days from
    *that* stale date -- if the lag happens to equal one of the lookback
    windows (it did: 7 days late, exactly the 1w window) every industry's
    return for that period comes out as an exact 0.0 (comparing the same
    stale close to itself), which reads as valid data but isn't. Anchoring on
    the industries' own latest date keeps every row's "today" and "N days
    ago" on the same, real, calendar.
    """
    industries = pd.read_csv(config.industries_csv)
    for symbol in industries["symbol"]:
        daily = load_daily(symbol, config.daily_dir)
        if daily is not None and not daily.empty:
            return daily.index.max()
    raise RuntimeError("No ^YH industry price history found -- cannot resolve a price date")


def save(df: pd.DataFrame, config: Config, as_of: date | None = None) -> Path:
    config.setup_dirs()
    label = str(as_of) if as_of else pd.Timestamp.today().strftime('%Y-%m-%d')
    out = config.results_dir / f"perf_screener_{label}.csv"
    df.to_csv(out, index=False)
    logger.info(f"Saved perf screener -> {out}")
    return out


def print_report(df: pd.DataFrame, top_n: int = 15) -> None:
    DIVIDER = '─' * 100
    cols = ["symbol", "sctr", "return_1w", "return_1m", "return_3m", "return_6m",
            "rs_1m", "rs_3m", "relret_spy_1m", "relret_qqq_1m"]
    cols = [c for c in cols if c in df.columns]

    flagged = df[df["data_flag"].notna()] if "data_flag" in df.columns else pd.DataFrame()
    if not flagged.empty:
        print(f"\n{DIVIDER}\n  ⚠ DATA QUALITY -- {len(flagged)} ticker(s) flagged "
              f"(verify before trusting return figures)\n{DIVIDER}")
        for _, r in flagged.iterrows():
            name = r.get("industry_name") or r["symbol"]
            print(f"  {r['symbol']:<14} {name:<40} {r['data_flag']}")

    for level, title in [("sector", "SECTOR ETFs"), ("industry", "INDUSTRIES (top by SCTR)")]:
        sub = df[df["level"] == level].sort_values("sctr", ascending=False)
        print(f"\n{DIVIDER}\n  {title}  ({len(sub)})\n{DIVIDER}")
        print(sub[cols].head(top_n).to_string(index=False))

    bench = df[df["level"] == "benchmark"]
    if not bench.empty:
        print(f"\n{DIVIDER}\n  BENCHMARKS\n{DIVIDER}")
        print(bench[cols].to_string(index=False))
    print(DIVIDER)
