"""
Industry SCTR breadth — derived from stock-level SCTR (Layer 3).

Aggregates stock SCTR scores per GICS industry to answer:
"Is the whole industry strengthening, or just a few names pulling the index up?"

Complements the ^YH index-based industry SCTR (Layer 1) with a bottom-up view.
Key divergence signal: index SCTR high but median stock SCTR low → fragile leadership.
"""

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from config import Config

logger = logging.getLogger(__name__)


def run(stock_sctr_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Aggregate stock SCTR to GICS industry breadth metrics.

    Args:
        stock_sctr_df:  Output of sctr_stocks.run() — must have 'ticker' and 'sctr' columns.
        config:         Project Config instance.

    Returns:
        DataFrame columns:
            industry_key, gics_sector, industry_name,
            count, median_sctr, mean_sctr,
            pct_above_75, pct_above_50, pct_below_25
        Sorted by median_sctr descending.
    """
    if stock_sctr_df.empty or 'sctr' not in stock_sctr_df.columns:
        logger.warning("Empty or incomplete stock SCTR input")
        return pd.DataFrame()

    sbi = pd.read_csv(config.stocks_by_industry_csv, usecols=['symbol', 'sector_key', 'sector_name', 'industry_key', 'industry_name'])
    sbi = sbi.rename(columns={'symbol': 'ticker'})

    merged = stock_sctr_df[['ticker', 'sctr']].merge(sbi, on='ticker', how='left')
    merged = merged.dropna(subset=['industry_key'])

    def _agg(g: pd.DataFrame) -> pd.Series:
        s = g['sctr']
        return pd.Series({
            'count':        len(s),
            'median_sctr':  round(s.median(), 1),
            'mean_sctr':    round(s.mean(), 1),
            'pct_above_75': round((s > 75).mean() * 100, 1),
            'pct_above_50': round((s > 50).mean() * 100, 1),
            'pct_below_25': round((s < 25).mean() * 100, 1),
        })

    breadth = merged.groupby(['industry_key', 'sector_name', 'industry_name']).apply(_agg).reset_index()
    breadth = breadth.sort_values('median_sctr', ascending=False).reset_index(drop=True)

    logger.info(f"Industry breadth computed for {len(breadth)} GICS industries")
    return breadth


def save(df: pd.DataFrame, config: Config, as_of: date | None = None) -> Path:
    """Save breadth results to a dated CSV."""
    config.setup_dirs()
    label = str(as_of) if as_of else pd.Timestamp.today().strftime('%Y-%m-%d')
    out = config.breadth_dir / f"industry_breadth_{label}.csv"
    df.to_csv(out, index=False)
    logger.info(f"Saved industry breadth → {out}")
    return out
