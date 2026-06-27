"""
Stock SCTR — StockCharts methodology with large/mid/small cap buckets.

Each stock is ranked within its own cap bucket (not against all stocks).

Universe source is controlled by Config.universe_source:
  'sc' — StockCharts DB universe table (~5,200 stocks, group labels from StockCharts)
  'tv' — TradingView universe CSV (~3,300 stocks, group labels from market-cap limits)
"""

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from config import Config
from src.data_loader import load_daily, load_monthly
from src.sctr_engine import compute_sctr_raw, rank_to_sctr
from src.universe_loader import load as load_universe, summary as universe_summary

logger = logging.getLogger(__name__)


def run(config: Config, as_of: date | None = None) -> pd.DataFrame:
    """
    Compute SCTR for all universe stocks, ranked within large/mid/small buckets.

    Args:
        config:  Project Config instance (universe_source controls which list to use).
        as_of:   Truncate data to this date for backtesting. None = latest.

    Returns:
        DataFrame columns:
            ticker, cap_bucket, market_cap,
            c1..c6, raw_score, sctr, rank_in_bucket
    """
    universe = load_universe(config)
    universe_summary(universe)

    cutoff = pd.Timestamp(as_of) if as_of else None
    records = []
    skipped = 0

    for i, row in enumerate(universe.itertuples(index=False)):
        ticker     = row.ticker
        cap_bucket = row.group_name
        market_cap = row.market_cap

        if i % 500 == 0 and i > 0:
            logger.info(f"  {i}/{len(universe)} processed  ({len(records)} scored so far)...")

        daily = load_daily(ticker, config.daily_dir)
        if daily is None or daily.empty:
            skipped += 1
            continue

        if cutoff is not None:
            daily = daily[daily.index <= cutoff]
            if daily.empty:
                skipped += 1
                continue

        monthly = load_monthly(ticker, config.monthly_dir, config.daily_dir)
        if monthly is None or monthly.empty:
            skipped += 1
            continue

        if cutoff is not None:
            monthly = monthly[monthly.index <= cutoff]

        result = compute_sctr_raw(daily, monthly)
        if result is None:
            skipped += 1
            continue

        records.append({
            'ticker':     ticker,
            'cap_bucket': cap_bucket,
            'market_cap': market_cap,
            **result,
        })

    logger.info(
        f"Raw scores computed: {len(records)} stocks  "
        f"(skipped {skipped} — no data or insufficient history)"
    )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Rank within each cap bucket separately
    sctr_parts = []
    for bucket, group in df.groupby('cap_bucket'):
        raw  = group.set_index('ticker')['raw_score']
        sctr = rank_to_sctr(raw).rename('sctr').reset_index()
        sctr_parts.append(sctr)

    df = df.merge(pd.concat(sctr_parts), on='ticker')

    df['rank_in_bucket'] = (
        df.groupby('cap_bucket')['sctr']
        .rank(ascending=False, method='min')
        .astype(int)
    )

    df = df.sort_values(['cap_bucket', 'sctr'], ascending=[True, False]).reset_index(drop=True)
    logger.info(f"SCTR assigned: {df.groupby('cap_bucket').size().to_dict()}")
    return df


def save(df: pd.DataFrame, config: Config, as_of: date | None = None) -> list[Path]:
    """
    Save stock SCTR results as three separate files — one per cap bucket.

    Returns list of paths: [large_path, mid_path, small_path]
    """
    config.setup_dirs()
    label = str(as_of) if as_of else pd.Timestamp.today().strftime('%Y-%m-%d')

    paths = []
    for bucket in ['large', 'mid', 'small']:
        subset = df[df['cap_bucket'] == bucket].copy()
        out = config.stock_sctr_dir / f"stock_sctr_{bucket}_{label}.csv"
        subset.to_csv(out, index=False)
        logger.info(f"Saved {bucket} SCTR → {out}  ({len(subset)} stocks)")
        paths.append(out)

    return paths
