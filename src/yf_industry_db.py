"""
SQLite store for GICS dashboard daily snapshots.

Schema-compatible port of stockCharts' src/db.py — same tables, same
columns, same UNIQUE constraints, same SQL — so the dashboard port can
read it with no query changes. Only the default db_path differs
(data/yf_dashboard.db instead of data/stockcharts.db) and the
new_highs_lows / index_membership pieces are omitted (the 52-Week
Highs/Lows tab they serve is out of scope for this project).

Tables:
    sector_summary    — one row per (snapshot_date, symbol)  — sector indexes
    industry_summary  — one row per (snapshot_date, symbol)
    sctr_rankings     — one row per (snapshot_date, group, symbol)
    benchmarks        — one row per (snapshot_date, symbol)  — EOD close
    universe          — one row per symbol (latest group label + metadata)
                        rebuilt from sctr_rankings via rebuild_universe()

Upsert on conflict — safe to re-run the same day.
"""

import math
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd


def parse_symbols(source) -> list[str]:
    """
    Return a clean list of ticker symbols from any of these inputs:

    - Python list:          ['AAPL', 'TSLA', 'NVDA']
    - TradingView string:   'NASDAQ:AAPL,NYSE:TSLA,NASDAQ:NVDA'
    - Newline-separated:    'AAPL\\nTSLA\\nNVDA'
    - CSV file path (str):  'watchlist.csv'  — uses the first column

    Exchange prefixes (NASDAQ:, NYSE:, etc.) are stripped automatically.
    """
    if isinstance(source, list):
        raw = source
    elif isinstance(source, str) and Path(source).is_file():
        import csv
        with open(source, newline='') as f:
            reader = csv.reader(f)
            next(reader, None)          # skip header if present
            raw = [row[0] for row in reader if row]
    else:
        # TradingView comma-separated or newline-separated string
        raw = [s.strip() for s in source.replace('\n', ',').split(',')]

    # strip exchange prefix: 'NASDAQ:AAPL' → 'AAPL'
    return [s.split(':')[-1].strip().upper() for s in raw if s]


# ── sector_summary ───────────────────────────────────────────────────────────

_SECT_COLS = [
    "snapshot_date", "symbol", "name", "last",
    "chg_1d", "chg_1w", "chg_1m", "chg_3m", "chg_6m", "chg_1y", "chg_ytd",
    "pct_1d", "pct_1w", "pct_1m", "pct_3m", "pct_6m", "pct_1y", "pct_ytd",
    "sctr", "volume", "market_cap", "child_count",
]

_CREATE_SECTOR = """
CREATE TABLE IF NOT EXISTS sector_summary (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    name          TEXT NOT NULL,
    last          REAL,
    chg_1d REAL, chg_1w REAL, chg_1m REAL, chg_3m REAL, chg_6m REAL, chg_1y REAL, chg_ytd REAL,
    pct_1d REAL, pct_1w REAL, pct_1m REAL, pct_3m REAL, pct_6m REAL, pct_1y REAL, pct_ytd REAL,
    sctr          REAL,
    volume        REAL,
    market_cap    REAL,
    child_count   INTEGER,
    UNIQUE(snapshot_date, symbol)
);
"""

# ── industry_summary ──────────────────────────────────────────────────────────

_IND_COLS = [
    "snapshot_date", "sector", "symbol", "name", "last",
    "chg_1d", "chg_1w", "chg_1m", "chg_3m", "chg_6m", "chg_1y", "chg_ytd",
    "pct_1d", "pct_1w", "pct_1m", "pct_3m", "pct_6m", "pct_1y", "pct_ytd",
    "sctr", "child_count",
]

_CREATE_INDUSTRY = """
CREATE TABLE IF NOT EXISTS industry_summary (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date TEXT NOT NULL,
    sector        TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    name          TEXT NOT NULL,
    last          REAL,
    chg_1d REAL, chg_1w REAL, chg_1m REAL, chg_3m REAL, chg_6m REAL, chg_1y REAL, chg_ytd REAL,
    pct_1d REAL, pct_1w REAL, pct_1m REAL, pct_3m REAL, pct_6m REAL, pct_1y REAL, pct_ytd REAL,
    sctr          REAL,
    child_count   INTEGER,
    UNIQUE(snapshot_date, symbol)
);
"""

# ── sctr_rankings ─────────────────────────────────────────────────────────────

_SCTR_COLS = [
    "snapshot_date", "group_name", "symbol", "name",
    "sector", "industry", "sctr", "sctr_delta",
    "close", "volume", "market_cap",
]

_CREATE_SCTR = """
CREATE TABLE IF NOT EXISTS sctr_rankings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date TEXT NOT NULL,
    group_name    TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    name          TEXT NOT NULL,
    sector        TEXT,
    industry      TEXT,
    sctr          REAL,
    sctr_delta    REAL,
    close         REAL,
    volume        REAL,
    market_cap    REAL,
    UNIQUE(snapshot_date, group_name, symbol)
);
"""


# ── benchmarks ───────────────────────────────────────────────────────────────

_BENCH_COLS = ["snapshot_date", "symbol", "close"]

_CREATE_BENCHMARKS = """
CREATE TABLE IF NOT EXISTS benchmarks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    close         REAL,
    UNIQUE(snapshot_date, symbol)
);
"""


# ── universe ──────────────────────────────────────────────────────────────────

_CREATE_UNIVERSE = """
CREATE TABLE IF NOT EXISTS universe (
    symbol      TEXT PRIMARY KEY,
    group_name  TEXT NOT NULL,
    name        TEXT,
    sector      TEXT,
    industry    TEXT,
    market_cap  REAL,
    first_seen  TEXT,
    last_seen   TEXT
);
"""


def _clean(v):
    """Convert NaN / None to None for SQLite."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


class YfIndustryDB:

    def __init__(self, db_path: str = "data/yf_dashboard.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(_CREATE_SECTOR)
            conn.execute(_CREATE_INDUSTRY)
            conn.execute(_CREATE_SCTR)
            conn.execute(_CREATE_BENCHMARKS)
            conn.execute(_CREATE_UNIVERSE)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    # ── write ─────────────────────────────────────────────────────────────────

    def upsert_sector_summary(self, df: pd.DataFrame, snapshot_date: str | None = None) -> None:
        today = snapshot_date or date.today().isoformat()
        rows = []
        for _, r in df.iterrows():
            rows.append({
                "snapshot_date": today,
                "symbol":      _clean(r.get("symbol", "")),
                "name":        _clean(r.get("name", "")),
                "last":        _clean(r.get("last")),
                "chg_1d":      _clean(r.get("chg_1d")),
                "chg_1w":      _clean(r.get("chg_1w")),
                "chg_1m":      _clean(r.get("chg_1m")),
                "chg_3m":      _clean(r.get("chg_3m")),
                "chg_6m":      _clean(r.get("chg_6m")),
                "chg_1y":      _clean(r.get("chg_1y")),
                "chg_ytd":     _clean(r.get("chg_ytd")),
                "pct_1d":      _clean(r.get("pct_1d")),
                "pct_1w":      _clean(r.get("pct_1w")),
                "pct_1m":      _clean(r.get("pct_1m")),
                "pct_3m":      _clean(r.get("pct_3m")),
                "pct_6m":      _clean(r.get("pct_6m")),
                "pct_1y":      _clean(r.get("pct_1y")),
                "pct_ytd":     _clean(r.get("pct_ytd")),
                "sctr":        _clean(r.get("sctr")),
                "volume":      _clean(r.get("volume")),
                "market_cap":  _clean(r.get("market_cap")),
                "child_count": _clean(r.get("child_count")),
            })

        ph  = ", ".join(f":{c}" for c in _SECT_COLS)
        upd = ", ".join(f"{c}=excluded.{c}" for c in _SECT_COLS if c not in ("snapshot_date", "symbol"))
        sql = (
            f"INSERT INTO sector_summary ({', '.join(_SECT_COLS)}) "
            f"VALUES ({ph}) "
            f"ON CONFLICT(snapshot_date, symbol) DO UPDATE SET {upd}"
        )
        with self._conn() as conn:
            conn.executemany(sql, rows)
        print(f"  DB: {len(rows)} sector rows upserted  [{today}]")

    def upsert_industry_summary(self, df: pd.DataFrame, snapshot_date: str | None = None) -> None:
        today = snapshot_date or date.today().isoformat()
        rows = []
        for _, r in df.iterrows():
            rows.append({
                "snapshot_date": today,
                "sector":      _clean(r.get("sector", "")),
                "symbol":      _clean(r.get("symbol", "")),
                "name":        _clean(r.get("name", "")),
                "last":        _clean(r.get("last")),
                "chg_1d":      _clean(r.get("chg_1d")),
                "chg_1w":      _clean(r.get("chg_1w")),
                "chg_1m":      _clean(r.get("chg_1m")),
                "chg_3m":      _clean(r.get("chg_3m")),
                "chg_6m":      _clean(r.get("chg_6m")),
                "chg_1y":      _clean(r.get("chg_1y")),
                "chg_ytd":     _clean(r.get("chg_ytd")),
                "pct_1d":      _clean(r.get("pct_1d")),
                "pct_1w":      _clean(r.get("pct_1w")),
                "pct_1m":      _clean(r.get("pct_1m")),
                "pct_3m":      _clean(r.get("pct_3m")),
                "pct_6m":      _clean(r.get("pct_6m")),
                "pct_1y":      _clean(r.get("pct_1y")),
                "pct_ytd":     _clean(r.get("pct_ytd")),
                "sctr":        _clean(r.get("sctr")),
                # NOTE: source column really is camelCase 'childCount' here,
                # matching stockCharts' own industry fetcher output.
                "child_count": _clean(r.get("childCount")),
            })

        ph  = ", ".join(f":{c}" for c in _IND_COLS)
        upd = ", ".join(f"{c}=excluded.{c}" for c in _IND_COLS if c not in ("snapshot_date", "symbol"))
        sql = (
            f"INSERT INTO industry_summary ({', '.join(_IND_COLS)}) "
            f"VALUES ({ph}) "
            f"ON CONFLICT(snapshot_date, symbol) DO UPDATE SET {upd}"
        )
        with self._conn() as conn:
            conn.executemany(sql, rows)
        print(f"  DB: {len(rows)} industry rows upserted  [{today}]")

    def upsert_sctr(self, df: pd.DataFrame, snapshot_date: str | None = None) -> None:
        today = snapshot_date or date.today().isoformat()
        rows = []
        for _, r in df.iterrows():
            rows.append({
                "snapshot_date": today,
                "group_name":   _clean(r.get("group", "")),
                "symbol":       _clean(r.get("symbol", "")),
                "name":         _clean(r.get("name", "")),
                "sector":       _clean(r.get("sector", "")),
                "industry":     _clean(r.get("industry", "")),
                "sctr":         _clean(r.get("sctr")),
                "sctr_delta":   _clean(r.get("sctr_delta")),
                "close":        _clean(r.get("close")),
                "volume":       _clean(r.get("volume")),
                "market_cap":   _clean(r.get("market_cap")),
            })

        ph  = ", ".join(f":{c}" for c in _SCTR_COLS)
        upd = ", ".join(f"{c}=excluded.{c}" for c in _SCTR_COLS if c not in ("snapshot_date", "group_name", "symbol"))
        sql = (
            f"INSERT INTO sctr_rankings ({', '.join(_SCTR_COLS)}) "
            f"VALUES ({ph}) "
            f"ON CONFLICT(snapshot_date, group_name, symbol) DO UPDATE SET {upd}"
        )
        with self._conn() as conn:
            conn.executemany(sql, rows)
        print(f"  DB: {len(rows)} SCTR rows upserted  [{today}]")

    # ── read ──────────────────────────────────────────────────────────────────

    def load_latest_industry(self) -> pd.DataFrame:
        sql = """
            SELECT * FROM industry_summary
            WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM industry_summary)
            ORDER BY sctr DESC
        """
        with self._conn() as conn:
            return pd.read_sql_query(sql, conn)

    def load_latest_sctr(self, group_name: str) -> pd.DataFrame:
        sql = """
            SELECT * FROM sctr_rankings
            WHERE group_name = ?
              AND snapshot_date = (
                  SELECT MAX(snapshot_date) FROM sctr_rankings WHERE group_name = ?
              )
            ORDER BY sctr DESC
        """
        with self._conn() as conn:
            return pd.read_sql_query(sql, conn, params=(group_name, group_name))

    def load_date_industry(self, snapshot_date: str) -> pd.DataFrame:
        """Industry summary for a specific date, e.g. '2026-06-13'."""
        sql = """
            SELECT * FROM industry_summary
            WHERE snapshot_date = ?
            ORDER BY sctr DESC
        """
        with self._conn() as conn:
            return pd.read_sql_query(sql, conn, params=(snapshot_date,))

    def load_date_sctr(self, snapshot_date: str, group_name: str) -> pd.DataFrame:
        """SCTR rankings for a specific date and group, e.g. '2026-06-13', 'large'."""
        sql = """
            SELECT * FROM sctr_rankings
            WHERE snapshot_date = ? AND group_name = ?
            ORDER BY sctr DESC
        """
        with self._conn() as conn:
            return pd.read_sql_query(sql, conn, params=(snapshot_date, group_name))

    def load_sctr_for_symbols(
        self,
        symbols,
        snapshot_date: str | None = None,
        group_name: str | None = None,
    ) -> pd.DataFrame:
        """
        Return SCTR data for a list of symbols on a given date.

        symbols       — list, TradingView string, newline string, or CSV file path
        snapshot_date — 'YYYY-MM-DD', or None for the latest available date
        group_name    — 'large' / 'mid' / 'small' / 'etf' / 'industry', or None to search all

        Example:
            db.load_sctr_for_symbols('NASDAQ:AAPL,NYSE:TSLA', '2026-06-13')
            db.load_sctr_for_symbols(['AAPL', 'NVDA', 'MSFT'])
            db.load_sctr_for_symbols('watchlist.csv', group_name='large')
        """
        tickers = parse_symbols(symbols)
        if not tickers:
            return pd.DataFrame()

        if snapshot_date is None:
            base = "WHERE group_name = ?" if group_name else "WHERE 1=1"
            params_date = (group_name,) if group_name else ()
            date_sql = f"SELECT MAX(snapshot_date) FROM sctr_rankings {base}"
            with self._conn() as conn:
                snapshot_date = conn.execute(date_sql, params_date).fetchone()[0]

        ph = ", ".join("?" * len(tickers))
        group_filter = "AND group_name = ?" if group_name else ""
        sql = f"""
            SELECT * FROM sctr_rankings
            WHERE snapshot_date = ?
              {group_filter}
              AND symbol IN ({ph})
            ORDER BY sctr DESC
        """
        params = [snapshot_date]
        if group_name:
            params.append(group_name)
        params.extend(tickers)

        with self._conn() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def load_history_symbol(self, symbol: str) -> pd.DataFrame:
        """All snapshot dates for a single symbol across both tables."""
        sql = """
            SELECT snapshot_date, sector, symbol, name,
                   pct_1d, pct_1w, pct_1m, pct_3m, pct_6m, pct_1y, pct_ytd, sctr
            FROM industry_summary
            WHERE symbol = ?
            ORDER BY snapshot_date
        """
        with self._conn() as conn:
            return pd.read_sql_query(sql, conn, params=(symbol,))

    def available_dates_industry(self) -> list[str]:
        with self._conn() as conn:
            return [r[0] for r in conn.execute(
                "SELECT DISTINCT snapshot_date FROM industry_summary ORDER BY snapshot_date DESC"
            ).fetchall()]

    def available_dates_sctr(self, group_name: str) -> list[str]:
        with self._conn() as conn:
            return [r[0] for r in conn.execute(
                "SELECT DISTINCT snapshot_date FROM sctr_rankings WHERE group_name = ? ORDER BY snapshot_date DESC",
                (group_name,),
            ).fetchall()]

    # ── benchmarks ────────────────────────────────────────────────────────────

    def upsert_benchmarks(self, df: pd.DataFrame) -> None:
        rows = [
            {
                "snapshot_date": str(r["snapshot_date"]),
                "symbol":        str(r["symbol"]),
                "close":         _clean(r["close"]),
            }
            for _, r in df.iterrows()
        ]
        ph  = ", ".join(f":{c}" for c in _BENCH_COLS)
        upd = "close=excluded.close"
        sql = (
            f"INSERT INTO benchmarks ({', '.join(_BENCH_COLS)}) "
            f"VALUES ({ph}) "
            f"ON CONFLICT(snapshot_date, symbol) DO UPDATE SET {upd}"
        )
        with self._conn() as conn:
            conn.executemany(sql, rows)
        symbols = df["symbol"].unique().tolist() if not df.empty else []
        print(f"  DB: {len(rows)} benchmark rows upserted  {symbols}")

    def load_benchmark(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """EOD close series for one benchmark symbol, newest first."""
        conditions = ["symbol = ?"]
        params: list = [symbol]
        if start:
            conditions.append("snapshot_date >= ?")
            params.append(start)
        if end:
            conditions.append("snapshot_date <= ?")
            params.append(end)
        where = " AND ".join(conditions)
        sql = f"SELECT snapshot_date, symbol, close FROM benchmarks WHERE {where} ORDER BY snapshot_date"
        with self._conn() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def available_benchmarks(self) -> list[str]:
        """Distinct benchmark symbols stored in the DB."""
        with self._conn() as conn:
            return [r[0] for r in conn.execute(
                "SELECT DISTINCT symbol FROM benchmarks ORDER BY symbol"
            ).fetchall()]

    # ── universe ──────────────────────────────────────────────────────────────

    def rebuild_universe(self) -> int:
        """
        Rebuild the universe table from sctr_rankings.

        For each unique symbol, keeps the group_name from the latest snapshot.
        Name, sector, industry, market_cap are also taken from the latest snapshot.
        first_seen and last_seen track the date range the symbol appeared.

        Returns the number of symbols in the rebuilt universe.
        Call this after each populate run to keep the universe current.
        """
        # group_name comes from the LATEST snapshot per symbol;
        # first_seen / last_seen span the full history.
        sql = """
            INSERT INTO universe (symbol, group_name, name, sector, industry,
                                  market_cap, first_seen, last_seen)
            SELECT
                r.symbol,
                r.group_name,
                r.name,
                r.sector,
                r.industry,
                r.market_cap,
                hist.first_seen,
                hist.last_seen
            FROM sctr_rankings r
            INNER JOIN (
                SELECT symbol,
                       MAX(snapshot_date) AS last_seen,
                       MIN(snapshot_date) AS first_seen
                FROM sctr_rankings
                GROUP BY symbol
            ) hist ON r.symbol = hist.symbol
                   AND r.snapshot_date = hist.last_seen
            GROUP BY r.symbol
            ON CONFLICT(symbol) DO UPDATE SET
                group_name  = excluded.group_name,
                name        = excluded.name,
                sector      = excluded.sector,
                industry    = excluded.industry,
                market_cap  = excluded.market_cap,
                first_seen  = MIN(universe.first_seen, excluded.first_seen),
                last_seen   = excluded.last_seen
        """
        with self._conn() as conn:
            conn.execute(sql)
            count = conn.execute("SELECT COUNT(*) FROM universe").fetchone()[0]
        print(f"  DB: universe rebuilt — {count} symbols")
        return count

    def load_universe(self, group_name: str | None = None) -> pd.DataFrame:
        """
        Return the full universe (or a single cap bucket) as a DataFrame.

        Args:
            group_name: 'large', 'mid', 'small', or None for all.

        Returns:
            DataFrame with columns: symbol, group_name, name, sector, industry,
                                    market_cap, first_seen, last_seen
        """
        if group_name:
            sql = "SELECT * FROM universe WHERE group_name = ? ORDER BY symbol"
            params = (group_name,)
        else:
            sql = "SELECT * FROM universe ORDER BY group_name, symbol"
            params = ()
        with self._conn() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def export_universe_csv(self, path: str | Path) -> Path:
        """
        Export the universe table to a CSV file.
        Useful for consuming in other projects without a direct DB dependency.
        """
        df = self.load_universe()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"  Universe exported → {out}  ({len(df)} symbols)")
        return out
