"""
Universe loader — provides the list of stocks and their cap bucket.

Supports two sources, selectable via Config.universe_source:

  'sc'  — StockCharts universe, read from the stockcharts.db universe table.
          group_name (large/mid/small) comes directly from StockCharts' own
          classification, making SCTR rankings directly comparable to their output.

  'tv'  — TradingView universe, read from tradingview_universe_bool.csv.
          group_name is derived by applying StockCharts hard market-cap limits
          ($10B / $2B / $250M) to the market_cap column in the file.

Both return a DataFrame with at least:
    ticker, group_name, market_cap
"""

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from config import Config

logger = logging.getLogger(__name__)

_SC_DB      = Path('/home/imagda/_invest2024/python/stockCharts/data/stockcharts.db')
_SC_TV_FILE = Path('/home/imagda/_invest2024/python/stockCharts/output/stockcharts_universe_tv.csv')
_TV_FILE    = Path('/home/imagda/_invest2024/python/downloadData_v1/data/tickers/tradingview_universe_bool.csv')

_STOCK_GROUPS = {'large', 'mid', 'small'}


def load(config: Config) -> pd.DataFrame:
    """
    Load the stock universe for SCTR calculation.

    Returns a DataFrame with columns: ticker, group_name, market_cap
    Only includes stocks in large / mid / small groups (ETFs and indexes excluded).
    """
    source = getattr(config, 'universe_source', 'sc')

    if source == 'sc':
        return _load_sc(config)
    elif source == 'tv':
        return _load_tv(config)
    else:
        raise ValueError(f"Unknown universe_source '{source}'. Use 'sc' or 'tv'.")


def _load_sc(config: Config) -> pd.DataFrame:
    """
    Load StockCharts universe from the blended CSV
    (SC group_name + TV sector/industry — no Dow Jones industry mapping).

    Falls back to reading directly from the DB if the blended file is missing
    (regenerate it with: cd stockCharts && python src/export_sc_universe.py).
    """
    if _SC_TV_FILE.exists():
        df = pd.read_csv(_SC_TV_FILE)
        df = df.rename(columns={'symbol': 'ticker'})
        df = df[df['group_name'].isin(_STOCK_GROUPS)].copy()
        logger.info(
            f"StockCharts universe loaded from CSV: {len(df)} stocks  "
            f"({df.groupby('group_name').size().to_dict()})"
        )
        return df

    # Fallback: read from DB (sector/industry will be SC/DJ, not TV)
    logger.warning(
        f"Blended universe CSV not found ({_SC_TV_FILE}). "
        f"Run: cd stockCharts && python src/export_sc_universe.py"
    )
    if not _SC_DB.exists():
        raise FileNotFoundError(f"StockCharts DB not found: {_SC_DB}")

    conn = sqlite3.connect(_SC_DB)
    df = pd.read_sql_query(
        "SELECT symbol, group_name, name, market_cap "
        "FROM universe "
        "WHERE group_name IN ('large','mid','small') "
        "ORDER BY group_name, symbol",
        conn,
    )
    conn.close()
    df = df.rename(columns={'symbol': 'ticker'})
    logger.info(
        f"StockCharts universe loaded from DB: {len(df)} stocks  "
        f"({df.groupby('group_name').size().to_dict()})"
    )
    return df


def _load_tv(config: Config) -> pd.DataFrame:
    """Load from TradingView universe CSV, classify cap bucket by market-cap thresholds."""
    if not _TV_FILE.exists():
        raise FileNotFoundError(f"TradingView universe file not found: {_TV_FILE}")

    df = pd.read_csv(_TV_FILE, usecols=['ticker', 'market_cap'])

    # Apply StockCharts hard limits
    df['group_name'] = None
    df.loc[df.market_cap >= config.large_cap_min,                                      'group_name'] = 'large'
    df.loc[(df.market_cap >= config.mid_cap_min)   & (df.market_cap < config.large_cap_min), 'group_name'] = 'mid'
    df.loc[(df.market_cap >= config.small_cap_min) & (df.market_cap < config.mid_cap_min),   'group_name'] = 'small'

    df = df[df.group_name.notna()].copy()
    logger.info(
        f"TradingView universe loaded: {len(df)} stocks  "
        f"({df.groupby('group_name').size().to_dict()})"
    )
    return df


def summary(universe_df: pd.DataFrame) -> None:
    """Print a quick summary of the loaded universe."""
    print(f"\nUniverse: {len(universe_df)} stocks")
    for group, grp in universe_df.groupby('group_name'):
        print(f"  {group:<8} {len(grp):>5} stocks")
