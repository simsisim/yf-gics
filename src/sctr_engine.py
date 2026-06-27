"""
True StockCharts SCTR formula — 6 components, percentile ranking.

Formula (John Murphy / StockCharts):
  c1 = 14-month ROC                     weight 30%   (monthly bars)
  c2 = % above/below 200-day SMA        weight 15%   (daily bars)
  c3 = 9-month ROC                      weight 15%   (monthly bars)
  c4 = % above/below 50-day SMA         weight 15%   (daily bars)
  c5 = 3-month ROC                      weight 10%   (monthly bars)
  c6 = PPO(12,26,9) histogram 3-day slope  weight 15%  (daily bars)

  raw_score = c1*0.30 + c2*0.15 + c3*0.15 + c4*0.15 + c5*0.10 + c6*0.15

SCTR = percentile rank of raw_score within the universe (0.0 – 99.9).

This module only computes scores and ranks — it does not do I/O.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Component weights (must sum to 1.0)
_W = dict(c1=0.30, c2=0.15, c3=0.15, c4=0.15, c5=0.10, c6=0.15)

# Minimum bar counts
_MIN_MONTHLY = 15   # 14-month ROC needs index t-14
_MIN_DAILY   = 210  # 200-day SMA needs 200 bars + some buffer


def _roc(series: pd.Series, n: int) -> Optional[float]:
    """Rate of change: (close_t - close_{t-n}) / close_{t-n} * 100."""
    if len(series) <= n:
        return None
    base = series.iloc[-(n + 1)]
    if base == 0 or pd.isna(base):
        return None
    return (series.iloc[-1] - base) / base * 100.0


def _pct_from_sma(close: pd.Series, period: int) -> Optional[float]:
    """(close - SMA_period) / SMA_period * 100."""
    if len(close) < period:
        return None
    sma = close.rolling(period).mean().iloc[-1]
    if pd.isna(sma) or sma == 0:
        return None
    return (close.iloc[-1] - sma) / sma * 100.0


def _ppo_histogram_slope(close: pd.Series) -> float:
    """
    3-day slope of PPO(12,26,9) histogram.
    slope = (hist[-1] - hist[-4]) / 3, clamped to [-50, 50].
    """
    if len(close) < 35:
        return 0.0
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    ppo   = (ema12 - ema26) / ema26.replace(0, np.nan) * 100.0
    sig   = ppo.ewm(span=9, adjust=False).mean()
    hist  = ppo - sig
    if len(hist) < 4 or pd.isna(hist.iloc[-1]) or pd.isna(hist.iloc[-4]):
        return 0.0
    slope = (hist.iloc[-1] - hist.iloc[-4]) / 3.0
    return float(np.clip(slope, -50.0, 50.0))


def compute_sctr_raw(
    daily_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
) -> Optional[dict]:
    """
    Compute the raw (un-ranked) SCTR score for a single security.

    Args:
        daily_df:   DataFrame with at least a 'Close' column, daily frequency,
                    sorted ascending, tz-naive DatetimeIndex.
        monthly_df: Same but monthly frequency (month-end closes).

    Returns:
        dict with keys c1..c6, raw_score, and component details.
        Returns None if there is insufficient data.
    """
    if daily_df is None or monthly_df is None:
        return None
    if len(daily_df) < _MIN_DAILY or len(monthly_df) < _MIN_MONTHLY:
        return None
    if 'Close' not in daily_df.columns or 'Close' not in monthly_df.columns:
        return None

    d = daily_df['Close'].dropna()
    m = monthly_df['Close'].dropna()

    if len(d) < _MIN_DAILY or len(m) < _MIN_MONTHLY:
        return None

    # Monthly ROC components
    c1 = _roc(m, 14)   # 14-month
    c3 = _roc(m,  9)   # 9-month
    c5 = _roc(m,  3)   # 3-month

    if c1 is None or c3 is None or c5 is None:
        return None

    # Daily SMA distance components
    c2 = _pct_from_sma(d, 200)
    c4 = _pct_from_sma(d, 50)

    if c2 is None or c4 is None:
        return None

    # Daily PPO histogram slope
    c6 = _ppo_histogram_slope(d)

    raw = c1*_W['c1'] + c2*_W['c2'] + c3*_W['c3'] + c4*_W['c4'] + c5*_W['c5'] + c6*_W['c6']

    return {
        'c1_14m_roc':   round(c1, 4),
        'c2_sma200_pct': round(c2, 4),
        'c3_9m_roc':    round(c3, 4),
        'c4_sma50_pct': round(c4, 4),
        'c5_3m_roc':    round(c5, 4),
        'c6_ppo_slope': round(c6, 4),
        'raw_score':    round(raw, 4),
    }


def rank_to_sctr(raw_scores: pd.Series) -> pd.Series:
    """
    Convert a Series of raw SCTR scores to percentile ranks 0.0 – 99.9.

    Uses the same bucket approach as StockCharts: stocks are ranked by their
    raw score within the universe, then assigned a percentile.

    Args:
        raw_scores: pd.Series indexed by ticker symbol, values are raw scores.

    Returns:
        pd.Series with the same index, values in [0.0, 99.9].
    """
    if raw_scores.empty:
        return raw_scores.copy()
    # rank(pct=True) maps lowest → ~0, highest → 1.0  (method='average' for ties)
    return (raw_scores.rank(method='average', pct=True) * 99.9).round(1)
