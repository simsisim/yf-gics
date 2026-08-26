"""
Compute industry & sector daily snapshots from ^YH index prices → YfIndustryDB.

New, self-contained computation pipeline (deliberately NOT built on the old
sctr_industry/perf_screener orchestration modules):

  1. Loads daily Close for the 145 industry ^YH<code> indexes (industries.csv)
     and the 11 sector ^YH3XX indexes via data_loader.load_daily().
     The sector ticker set is derived from industries.csv itself (the first 3
     digits of an industry's code are its sector's code), not hardcoded.
  2. Ranks industries and sectors in TWO SEPARATE SCTR pools — never combined.
     (Running them as one pool was caught live producing 156 ranked rows with
     a data artifact at #1.)
  3. Computes chg_<p> (absolute) and pct_<p> (percent) for
     1d/1w/1m/3m/6m/1y/ytd using calendar-day lookback (nearest close on or
     before as_of - N days; ytd looks back to Jan 1 of the snapshot year).
  4. Flags implausible single-day moves (>15%) — the confirmed Yahoo ^YH
     step-change cases (^YH10200010 +63%, ^YH31110020 +32%) are permanent
     feed artifacts. Two tiers: 'fresh' (jump in the trailing ~15 sessions)
     and 'unreverted' (older jump whose level never came back and is still
     skewing ROC125/EMA200, e.g. ^YH10200010's #1 rank). Flagged instruments
     stay in the DB unchanged; flags go to console plus a companion CSV so a
     flagged top rank is visible, not hidden.
  5. child_count: industries = stock rows in stocks_by_industry.csv;
     sectors = industry count per sector (industries.csv).
  6. Writes industry_summary / sector_summary via yf_industry_db.YfIndustryDB
     upserts. volume/market_cap stay NULL — ^YH carries no real volume.

CLI:
  python -m src.yf_industry_compute                     # latest session
  python -m src.yf_industry_compute --as-of 2026-08-21
  python -m src.yf_industry_compute --last 8            # last 8 sessions
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from config import Config
from src.data_loader import load_daily
from src.sctr_engine import compute_indicators, rank_to_sctr
from src.yf_industry_db import YfIndustryDB

logger = logging.getLogger(__name__)

# Calendar-day lookback windows ('ytd' handled separately: Jan 1 of as_of year)
RETURN_WINDOWS = {
    "1d": 1,
    "1w": 7,
    "1m": 30,
    "3m": 91,
    "6m": 182,
    "1y": 365,
}

# A diversified industry/sector index essentially never moves this much in
# one session — observed offenders were Yahoo feed artifacts, not markets.
JUMP_THRESHOLD_PCT = 15.0

# Two detection tiers:
#   'fresh'      — jump inside the trailing FRESH_LOOKBACK_SESSIONS sessions
#                  (recent feed weirdness, even if it has since round-tripped).
#   'unreverted' — older jump whose price LEVEL never came back: the step
#                  between the pre-jump close and the current close still
#                  exceeds UNREVERTED_PCT. These are what permanently skew
#                  SCTR's ROC125/EMA200 components (^YH10200010's +63% on
#                  2026-07-29 sits far outside the fresh window but still
#                  drives its #1 score months later), so they stay flagged as
#                  long as the displacement lasts.
FRESH_LOOKBACK_SESSIONS = 15
STEP_SCAN_SESSIONS = 252     # score-relevant horizon (ROC125 = 126 bars)
UNREVERTED_PCT = 5.0

DEFAULT_DB_PATH = "data/yf_dashboard.db"
DEFAULT_QUALITY_LOG = "results/yf_index_quality_flags.csv"

_CHG_COLS = [f"chg_{w}" for w in RETURN_WINDOWS] + ["chg_ytd"]
_PCT_COLS = [f"pct_{w}" for w in RETURN_WINDOWS] + ["pct_ytd"]


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

def load_reference(config: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """
    Load industries.csv + stocks_by_industry.csv and derive the sector map.

    Returns:
        industries      — industries.csv as-is
        stocks_by_ind   — stocks_by_industry.csv[['industry_key', 'symbol']]
        sector_map      — {sector_key: '^YH3XX' sector index symbol}
                          Derived from the industry code prefixes; asserts the
                          derivation is unambiguous and covers exactly the
                          11 sector indexes expected on disk.
    """
    industries = pd.read_csv(config.industries_csv)
    stocks_by_ind = pd.read_csv(
        config.stocks_by_industry_csv, usecols=["industry_key", "symbol"]
    )

    ind = industries.copy()
    ind["sector_code"] = ind["symbol"].str.extract(r"^\^YH(\d{3})")
    per_sector = ind.groupby("sector_key")["sector_code"].agg(["nunique", "first"])
    bad = per_sector[per_sector["nunique"] != 1]
    if len(bad):
        raise ValueError(f"Sector code derivation ambiguous for: {bad.index.tolist()}")

    sector_map = {
        sector_key: f"^YH{code}"
        for sector_key, code in per_sector["first"].items()
    }
    if len(sector_map) != 11:
        raise ValueError(f"Expected 11 sectors, derived {len(sector_map)}: {sorted(sector_map)}")

    return industries, stocks_by_ind, sector_map


# ---------------------------------------------------------------------------
# Price loading + primitives
# ---------------------------------------------------------------------------

def load_closes(symbols: list[str], config: Config) -> dict[str, pd.Series]:
    """Load daily Close for each symbol via data_loader.load_daily()."""
    closes: dict[str, pd.Series] = {}
    for sym in symbols:
        df = load_daily(sym, config.daily_dir)
        if df is None or df.empty or "Close" not in df.columns:
            logger.warning("No price data loaded for %s — skipped", sym)
            continue
        s = df["Close"].dropna()
        s = s[~s.index.duplicated(keep="last")].sort_index()
        if s.empty:
            logger.warning("Empty Close series for %s — skipped", sym)
            continue
        closes[sym] = s.astype(float)
    return closes


def session_calendar(closes: dict[str, pd.Series]) -> pd.DatetimeIndex:
    """Union of all session dates across every loaded series, ascending."""
    return pd.DatetimeIndex(sorted(set().union(*[s.index for s in closes.values()])))


def snap_to_session(ts: pd.Timestamp, calendar: pd.DatetimeIndex) -> pd.Timestamp:
    """Snap ts to the last session on or before it."""
    le = calendar[calendar <= ts]
    if len(le) == 0:
        raise ValueError(f"No session on/before {ts.date()} in the loaded data")
    return le[-1]


def _last_on_or_before(s: pd.Series, ts: pd.Timestamp) -> float | None:
    """Last close strictly on or before ts (s must be sorted ascending)."""
    sub = s.loc[:ts]
    return None if sub.empty else float(sub.iloc[-1])


def compute_returns(s: pd.Series, as_of: pd.Timestamp) -> dict[str, float]:
    """
    chg_<w>/pct_<w> vs the close N calendar days earlier (nearest session on
    or before that date); *_ytd vs the last close before Jan 1 of as_of's
    year. Missing base (insufficient history) → NaN for that window.
    """
    out: dict[str, float] = {}
    latest = _last_on_or_before(s, as_of)
    if latest is None:
        return {**{c: float("nan") for c in _CHG_COLS},
                **{p: float("nan") for p in _PCT_COLS}}

    bases = {}
    for w, days in RETURN_WINDOWS.items():
        bases[w] = _last_on_or_before(s, as_of - pd.Timedelta(days=days))
    bases["ytd"] = _last_on_or_before(s, pd.Timestamp(year=as_of.year, month=1, day=1))

    for w, base in bases.items():
        if base is None or base == 0:
            out[f"chg_{w}"] = float("nan")
            out[f"pct_{w}"] = float("nan")
        else:
            out[f"chg_{w}"] = latest - base
            out[f"pct_{w}"] = (latest - base) / base * 100.0
    return out


def flag_quality(s: pd.Series) -> list[tuple[pd.Timestamp, float, float, str]]:
    """
    Data-quality flags for one instrument's close series (as_of = last bar).

    Returns [(jump_date, pct_move, pct_unreverted, kind), ...] where kind is
    'fresh', 'unreverted', or 'fresh+unreverted'. Flags only — callers decide
    visibility; nothing is dropped or corrected.
    """
    if len(s) < 2:
        return []
    rets = s.pct_change().dropna() * 100.0
    last = float(s.iloc[-1])
    fresh_idx = set(rets.index[-FRESH_LOOKBACK_SESSIONS:])

    out: list[tuple[pd.Timestamp, float, float, str]] = []
    for dt, move in rets.iloc[-STEP_SCAN_SESSIONS:].items():
        if abs(move) <= JUMP_THRESHOLD_PCT:
            continue
        loc = s.index.get_loc(dt)
        if loc == 0:
            continue
        pre = float(s.iloc[loc - 1])          # close before the jump
        unreverted = (last / pre - 1.0) * 100.0
        is_fresh = dt in fresh_idx
        unrev = abs(unreverted) > UNREVERTED_PCT
        if not (is_fresh or unrev):
            continue
        kind = ("fresh+unreverted" if is_fresh and unrev
                else "fresh" if is_fresh else "unreverted")
        out.append((dt, float(move), float(unreverted), kind))
    return out


# ---------------------------------------------------------------------------
# Snapshot computation
# ---------------------------------------------------------------------------

def compute_snapshot(
    as_of: pd.Timestamp,
    closes: dict[str, pd.Series],
    industries: pd.DataFrame,
    stocks_by_ind: pd.DataFrame,
    sector_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute one snapshot date's industry_summary / sector_summary rows plus
    the data-quality flag table.

    Returns:
        ind_df, sect_df — frames whose columns match what
                          YfIndustryDB.upsert_* methods read
                          (note: industry rows carry camelCase 'childCount',
                           matching the DB module's contract).
        flags_df        — snapshot_date, pool, symbol, name, jump_date, pct_move
    """
    ind_child = stocks_by_ind.groupby("industry_key").size().to_dict()
    sect_child = industries.groupby("sector_key").size().to_dict()
    name_of = industries.set_index("industry_key")[["industry_name", "sector_name"]]

    # ---- per-instrument metrics ------------------------------------------
    ind_raw: dict[str, float] = {}
    sect_raw: dict[str, float] = {}
    ind_rows: list[dict] = []
    sect_rows: list[dict] = []
    flags: list[dict] = []

    def _metrics(sym: str, pool: str, name: str, sector: str | None,
                 child_count: int) -> tuple[dict, float | None]:
        s = closes[sym]
        trunc = s.loc[:as_of]
        if trunc.empty:
            logger.warning("%s (%s): no data on/before %s", sym, pool, as_of.date())
            return {}, None

        row: dict = {"symbol": sym, "name": name, "last": float(trunc.iloc[-1])}
        if pool == "industry":
            row["sector"] = sector
            row["childCount"] = int(child_count)
        else:
            row["volume"] = None       # ^YH indexes carry no real volume…
            row["market_cap"] = None   # …and no market cap — leave NULL
            row["child_count"] = int(child_count)

        row.update(compute_returns(trunc, as_of))

        comps = compute_indicators(pd.DataFrame({"Close": trunc}))
        raw = None if comps is None else comps["raw_score"]

        for dt, move, unreverted, kind in flag_quality(trunc):
            flags.append({
                "snapshot_date": as_of.date().isoformat(),
                "pool": pool,
                "symbol": sym,
                "name": name,
                "jump_date": dt.date().isoformat(),
                "pct_move": round(move, 2),
                "pct_unreverted": round(unreverted, 2),
                "kind": kind,
            })

        return row, raw

    # Industries — one ranking pool
    for _, r in industries.iterrows():
        meta = name_of.loc[r["industry_key"]]
        row, raw = _metrics(
            r["symbol"], "industry", meta["industry_name"], meta["sector_name"],
            int(ind_child.get(r["industry_key"], 0)),
        )
        if row:
            ind_rows.append(row)
            if raw is not None:
                ind_raw[r["symbol"]] = raw

    # Sectors — a second, separate pool (NEVER merged with industries)
    for sector_key, sym in sector_map.items():
        sector_name = industries.loc[industries["sector_key"] == sector_key,
                                     "sector_name"].iloc[0]
        row, raw = _metrics(sym, "sector", sector_name, None, int(sect_child[sector_key]))
        if row:
            sect_rows.append(row)
            if raw is not None:
                sect_raw[sym] = raw

    # ---- percentile-rank each pool separately -----------------------------
    ind_sctr = rank_to_sctr(pd.Series(ind_raw, dtype=float))
    sect_sctr = rank_to_sctr(pd.Series(sect_raw, dtype=float))
    for rows, sctr in ((ind_rows, ind_sctr), (sect_rows, sect_sctr)):
        for row in rows:
            row["sctr"] = float(sctr.get(row["symbol"])) if row["symbol"] in sctr.index else float("nan")

    flags_df = pd.DataFrame(flags, columns=[
        "snapshot_date", "pool", "symbol", "name", "jump_date",
        "pct_move", "pct_unreverted", "kind",
    ])
    return pd.DataFrame(ind_rows), pd.DataFrame(sect_rows), flags_df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def write_quality_log(flags_df: pd.DataFrame, path: str | Path) -> Path | None:
    """Append flags to the companion CSV, deduped on (snapshot_date, symbol, jump_date, kind)."""
    if flags_df.empty:
        return None
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    old = pd.read_csv(out, dtype=str) if out.exists() else pd.DataFrame()
    combo = pd.concat([old, flags_df.astype(str)], ignore_index=True)
    combo = combo.drop_duplicates(
        subset=["snapshot_date", "symbol", "jump_date", "kind"], keep="last"
    )
    combo.to_csv(out, index=False)
    return out


def run_snapshot(
    as_of: pd.Timestamp,
    closes: dict[str, pd.Series],
    ref: tuple[pd.DataFrame, pd.DataFrame, dict[str, str]],
    db: YfIndustryDB,
    quality_log: str | Path = DEFAULT_QUALITY_LOG,
    quiet_db: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute one snapshot and persist it (+ quality log). Returns (ind_df, sect_df)."""
    ind_df, sect_df, flags_df = compute_snapshot(as_of, closes, *ref)

    db.upsert_industry_summary(ind_df, snapshot_date=as_of.date().isoformat())
    db.upsert_sector_summary(sect_df, snapshot_date=as_of.date().isoformat())

    if not quiet_db:
        if not flags_df.empty:
            logger.warning(
                "DATA QUALITY: %d flag(s) on %s — see %s (flagged scores may be artifact-driven)",
                len(flags_df), flags_df["symbol"].unique().tolist(), quality_log,
            )
        for _, f in flags_df.iterrows():
            logger.warning(
                "  FLAG [%s] %s (%s): %+.1f%% single-day move on %s | kind=%s, still %.1f%% vs pre-jump",
                f["pool"], f["symbol"], f["name"], f["pct_move"], f["jump_date"],
                f["kind"], f["pct_unreverted"],
            )
        top_ind = ind_df.dropna(subset=["sctr"]).sort_values("sctr", ascending=False).head(3)
        logger.info("[%s] industries ranked: %d | sectors ranked: %d | top industries: %s",
                    as_of.date(), int(ind_df["sctr"].notna().sum()), int(sect_df["sctr"].notna().sum()),
                    ", ".join(f"{r.symbol}={r.sctr}" for r in top_ind.itertuples()))

    write_quality_log(flags_df, quality_log)
    return ind_df, sect_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--as-of", action="append", default=[],
                    help="snapshot date YYYY-MM-DD (repeatable; snaps to last session on/before)")
    ap.add_argument("--last", type=int, default=None, metavar="N",
                    help="run the last N trading sessions instead of --as-of")
    ap.add_argument("--db", default=DEFAULT_DB_PATH, help=f"SQLite path (default {DEFAULT_DB_PATH})")
    ap.add_argument("--quality-log", default=DEFAULT_QUALITY_LOG,
                    help=f"data-quality flag CSV (default {DEFAULT_QUALITY_LOG})")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    config = Config()
    industries, stocks_by_ind, sector_map = load_reference(config)
    ref = (industries, stocks_by_ind, sector_map)

    symbols = industries["symbol"].tolist() + list(sector_map.values())
    logger.info("Loading prices: %d industries + %d sectors...", len(industries), len(sector_map))
    closes = load_closes(symbols, config)
    if len(closes) < len(symbols):
        logger.warning("Loaded %d/%d price series", len(closes), len(symbols))

    calendar = session_calendar(closes)
    if args.last is not None:
        targets = list(calendar[-args.last:])
    elif args.as_of:
        targets = sorted({snap_to_session(pd.Timestamp(d), calendar) for d in args.as_of})
    else:
        targets = [calendar[-1]]
    logger.info("Snapshot dates: %s", [str(t.date()) for t in targets])

    db = YfIndustryDB(args.db)
    for t in targets:
        run_snapshot(t, closes, ref, db, quality_log=args.quality_log)

    n_sect_dates = db._conn().execute("SELECT COUNT(DISTINCT snapshot_date) FROM sector_summary").fetchone()[0]
    logger.info("Done. DB=%s | industry dates=%d | sector dates=%d",
                args.db, len(db.available_dates_industry()), n_sect_dates)
    return 0


if __name__ == "__main__":
    sys.exit(main())
