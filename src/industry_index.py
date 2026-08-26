"""
Synthetic industry index builder.

Builds a market-cap-weighted synthetic daily/weekly index from constituent
stock closes for one GICS industry. This is the deliberate, permanent
alternative to the published ^YH Dow Jones Industry Index ticker: this
project found real, verified data defects in Yahoo's own ^YH industry-index
feed (permanent step-change jumps that never revert — see history_metrics.py's
module docstring for the confirmed cases), so history-tracking code that
can't tolerate those glitches baking in permanently builds its own index from
constituents instead.

This module no longer offers a source='yh'|'synth' dispatch (that dual-source
abstraction was part of an earlier, abandoned architecture) — callers that
want the ^YH series directly should call src.data_loader.load_daily() with
the industry's '^YH<code>' symbol themselves (see sctr_industry.py,
stage_analysis.py, rs_percentile.py for that pattern).
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data_loader import load_daily

logger = logging.getLogger(__name__)

MIN_CONSTITUENTS = 3


# ---------------------------------------------------------------------------
# Synthetic index builders
# ---------------------------------------------------------------------------

def preload_price_matrix(tickers: list[str], daily_dir: Path) -> pd.DataFrame:
    """
    Load Close series for all tickers into one wide DataFrame.
    Use in batch mode (backfill, rs_percentile, stage, ath) to avoid
    reopening files per industry.
    """
    closes = {}
    for i, ticker in enumerate(tickers):
        if i % 500 == 0 and i > 0:
            logger.info(f"  Loading closes: {i}/{len(tickers)} ({len(closes)} loaded)...")
        df = load_daily(ticker, daily_dir)
        if df is not None and not df.empty and len(df) >= 100:
            closes[ticker] = df['Close']
    matrix = pd.DataFrame(closes).sort_index()
    logger.info(f"Price matrix: {len(matrix)} rows × {len(matrix.columns)} tickers")
    return matrix


def build_synthetic_daily(
    tickers: list[str],
    weights: dict[str, float],
    price_matrix: pd.DataFrame,
) -> Optional[pd.Series]:
    """Market-cap-weighted daily close index from a pre-loaded price matrix."""
    available = [t for t in tickers if t in price_matrix.columns]
    if len(available) < MIN_CONSTITUENTS:
        return None
    total_cap = sum(weights.get(t, 0) for t in available)
    if total_cap == 0:
        w = pd.Series({t: 1.0 / len(available) for t in available})
    else:
        w = pd.Series({t: weights.get(t, 0) / total_cap for t in available})
    sub     = price_matrix[available].mask(price_matrix[available] <= 0).ffill(limit=5)
    returns = sub.pct_change(fill_method=None)
    idx_ret = returns.mul(w, axis=1).sum(axis=1)
    idx     = (1 + idx_ret).cumprod() * 1000.0
    idx.iloc[0] = 1000.0
    return idx.dropna()


def _read_weekly_tier(path: Path, ticker: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col='Date', parse_dates=False)
        df.index = pd.to_datetime(df.index.str.split(' ').str[0], errors='coerce')
        return df[df.index.notna()].sort_index()
    except Exception as e:
        logger.debug(f"Could not load weekly {ticker} from {path}: {e}")
        return None


def _load_weekly_file(ticker: str, weekly_dir: Path) -> Optional[pd.Series]:
    """
    Load weekly close for a single ticker. Reads archive/+current/ directly
    (current wins on any overlapping date), falling back to the flat legacy
    cache only if neither tier has this ticker.
    """
    frames = [
        f for f in (
            _read_weekly_tier(weekly_dir / "archive" / f"{ticker}.csv", ticker),
            _read_weekly_tier(weekly_dir / "current" / f"{ticker}.csv", ticker),
        ) if f is not None and not f.empty
    ]
    if frames:
        df = pd.concat(frames) if len(frames) > 1 else frames[0]
        df = df[~df.index.duplicated(keep='last')].sort_index()
    else:
        df = _read_weekly_tier(weekly_dir / f"{ticker}.csv", ticker)

    if df is None or 'Close' not in df.columns:
        return None
    return df['Close'].dropna()


def build_synthetic_weekly(
    tickers: list[str],
    weights: dict[str, float],
    weekly_dir: Path,
    cutoff: Optional[pd.Timestamp] = None,
    min_weeks: int = 60,
) -> Optional[pd.Series]:
    """Market-cap-weighted weekly close index from constituent stocks."""
    closes = {}
    for ticker in tickers:
        s = _load_weekly_file(ticker, weekly_dir)
        if s is None or len(s) < min_weeks:
            continue
        closes[ticker] = s
    if len(closes) < MIN_CONSTITUENTS:
        return None
    price_df = pd.DataFrame(closes).sort_index().ffill(limit=2)
    if cutoff is not None:
        price_df = price_df[price_df.index <= cutoff]
    if len(price_df) < min_weeks:
        return None
    available = list(closes.keys())
    total_cap = sum(weights.get(t, 0) for t in available)
    if total_cap == 0:
        w = pd.Series({t: 1.0 / len(available) for t in available})
    else:
        w = pd.Series({t: weights.get(t, 0) / total_cap for t in available})
    returns       = price_df[available].pct_change(fill_method=None)
    index_returns = returns.mul(w, axis=1).sum(axis=1)
    index_close   = (1 + index_returns).cumprod() * 1000.0
    index_close.iloc[0] = 1000.0
    return index_close.dropna()


