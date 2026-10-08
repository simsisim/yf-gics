"""
Load key-index EOD closes from downloadData_v1 → YfIndustryDB (`key_indices`).

Feeds the dashboard's Key Index Ranks tab: macro indices + size/style/bond
ETFs listed in input/key_indices.csv. Only raw closes are stored — the
dashboard computes pct_1d…pct_ytd from the close series at read time
(app.load_all_key_index), same as stockCharts.

History starts at --since (default 2025-05-01): enough for a full 1Y lookback
from the dashboard's first snapshot (2026-05-27), no deeper.

CLI:
  python -m src.yf_key_indices_compute                    # since 2025-05-01
  python -m src.yf_key_indices_compute --since 2026-01-01
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from config import PROJECT_DIR, Config
from src.data_loader import load_daily
from src.yf_industry_db import YfIndustryDB

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/yf_dashboard.db"
DEFAULT_SINCE = "2025-05-01"
KEY_INDICES_CSV = PROJECT_DIR / "input" / "key_indices.csv"


def load_key_index_symbols(path: Path = KEY_INDICES_CSV) -> list[str]:
    u = pd.read_csv(path, comment="#")
    u.columns = u.columns.str.strip()
    return u["symbol"].astype(str).str.strip().tolist()


def compute_key_indices(symbols: list[str], config: Config, since: str) -> pd.DataFrame:
    """One row per (snapshot_date, symbol, close) for every session >= since."""
    frames = []
    for sym in symbols:
        df = load_daily(sym, config.daily_dir)
        if df is None or df.empty or "Close" not in df.columns:
            logger.warning("No price data for %s — skipped", sym)
            continue
        s = df["Close"].dropna().astype(float)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        s = s[s.index >= pd.Timestamp(since)]
        frames.append(pd.DataFrame({
            "snapshot_date": s.index.strftime("%Y-%m-%d"),
            "symbol": sym,
            "close": s.values,
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["snapshot_date", "symbol", "close"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--since", default=DEFAULT_SINCE,
                    help=f"first date to store, YYYY-MM-DD (default {DEFAULT_SINCE})")
    ap.add_argument("--db", default=DEFAULT_DB_PATH, help=f"SQLite path (default {DEFAULT_DB_PATH})")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    symbols = load_key_index_symbols()
    df = compute_key_indices(symbols, Config(), args.since)
    if df.empty:
        logger.error("No key-index data loaded")
        return 1

    last = df.groupby("symbol")["snapshot_date"].max()
    stale = last[last < last.max()]
    if not stale.empty:
        logger.warning("Symbols behind the latest session %s: %s", last.max(), stale.to_dict())

    YfIndustryDB(args.db).upsert_key_indices(df)
    logger.info("Done. %d symbols | %s → %s", df["symbol"].nunique(),
                df["snapshot_date"].min(), df["snapshot_date"].max())
    return 0


if __name__ == "__main__":
    sys.exit(main())
