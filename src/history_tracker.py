"""
Daily History Tracker — the cheap go-forward path.

Computes today's row only (per industry/sector) via history_backfill's shared
compute_history(months_back=None), then upserts it into the same long-format
history file history_backfill.py writes: re-running the same date replaces
that date's rows rather than duplicating them, so this is safe to run more
than once on the same day (e.g. after a data correction).
"""

import logging
from pathlib import Path

import pandas as pd

from config import Config
from src.history_backfill import compute_history, save as save_history

logger = logging.getLogger(__name__)

HISTORY_FILE = 'industry_sector_metrics_daily.csv'


def run(config: Config) -> pd.DataFrame:
    """Compute today's row per symbol and upsert into the persisted history file."""
    today_df = compute_history(config, months_back=None)
    if today_df.empty:
        logger.error("No rows computed for today -- nothing to append")
        return today_df

    history_path = config.results_dir / 'history' / HISTORY_FILE
    if history_path.exists():
        existing = pd.read_csv(history_path)
        existing['date'] = pd.to_datetime(existing['date']).dt.date
        today_dates = set(today_df['date'])
        before = len(existing)
        existing = existing[~existing['date'].isin(today_dates)]
        dropped = before - len(existing)
        if dropped:
            logger.info(f"Replacing {dropped} existing row(s) for date(s) {sorted(today_dates)}")
        combined = pd.concat([existing, today_df], ignore_index=True)
    else:
        combined = today_df

    combined = combined.sort_values(['date', 'level', 'symbol']).reset_index(drop=True)
    logger.info(f"History after append: {len(combined)} rows "
                f"({combined['date'].nunique()} dates x {combined['symbol'].nunique()} symbols)")
    return combined


def save(df: pd.DataFrame, config: Config) -> dict[str, Path]:
    """Same save/pivot logic as the full backfill -- reused so both write identically shaped files."""
    return save_history(df, config)
