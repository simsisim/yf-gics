"""
Industry SCTR — Option A (StockCharts methodology).

Applies the true SCTR formula directly to the ^YH Dow Jones Industry Index
price series, exactly as StockCharts does. The 145 industry indexes are
ranked against each other to produce the industry SCTR.

^YH files only exist as daily downloads; monthly bars are derived by
resampling to month-end.

Output is labeled with GICS sector/industry names from industries.csv.
"""

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from config import Config
from src.data_loader import load_daily, load_monthly, list_tickers
from src.sctr_engine import compute_sctr_raw, rank_to_sctr

logger = logging.getLogger(__name__)


def run(config: Config, as_of: date | None = None) -> pd.DataFrame:
    """
    Compute SCTR for all ^YH industry indexes and return a ranked DataFrame.

    Args:
        config:  Project Config instance.
        as_of:   If given, truncate data to this date (useful for backtesting).
                 If None, uses all available data (latest date in files).

    Returns:
        DataFrame columns:
            symbol, industry_key, gics_sector, industry_name,
            c1..c6, raw_score, sctr, rank  (sorted by sctr desc)
    """
    industries = pd.read_csv(config.industries_csv)
    # Build lookup: symbol → (sector_key, sector_name, industry_key, industry_name)
    sym_to_gics = {
        row['symbol']: row
        for _, row in industries.iterrows()
    }

    yh_tickers = list_tickers(config.daily_dir, prefix='^YH')
    logger.info(f"Found {len(yh_tickers)} ^YH industry index files")

    records = []
    for ticker in yh_tickers:
        daily = load_daily(ticker, config.daily_dir)
        if daily is None or daily.empty:
            continue

        if as_of is not None:
            cutoff = pd.Timestamp(as_of)
            daily = daily[daily.index <= cutoff]
            if daily.empty:
                continue

        # Monthly bars: resample from daily (no monthly file for ^YH symbols)
        monthly = load_monthly(ticker, config.monthly_dir, config.daily_dir)
        if monthly is None or monthly.empty:
            continue

        if as_of is not None:
            monthly = monthly[monthly.index <= cutoff]

        result = compute_sctr_raw(daily, monthly)
        if result is None:
            logger.debug(f"Skipping {ticker}: insufficient data")
            continue

        gics = sym_to_gics.get(ticker, {})
        records.append({
            'symbol':        ticker,
            'industry_key':  gics.get('industry_key', ''),
            'gics_sector':   gics.get('sector_name', ''),
            'industry_name': gics.get('industry_name', ticker),
            **result,
        })

    if not records:
        logger.warning("No industry SCTR records computed — check data availability")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    raw = df.set_index('symbol')['raw_score']
    df['sctr'] = rank_to_sctr(raw).values
    df['rank'] = df['sctr'].rank(ascending=False, method='min').astype(int)
    df = df.sort_values('sctr', ascending=False).reset_index(drop=True)

    logger.info(f"Industry SCTR computed for {len(df)} indexes")
    return df


def save(df: pd.DataFrame, config: Config, as_of: date | None = None) -> Path:
    """Save the industry SCTR result to a dated CSV."""
    config.setup_dirs()
    label = str(as_of) if as_of else pd.Timestamp.today().strftime('%Y-%m-%d')
    out = config.industry_sctr_dir / f"industry_sctr_{label}.csv"
    df.to_csv(out, index=False)
    logger.info(f"Saved industry SCTR → {out}")
    return out
