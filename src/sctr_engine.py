"""
True StockCharts SCTR formula — 6 components, percentile ranking.

Source: https://chartschool.stockcharts.com/table-of-contents/
        technical-indicators-and-overlays/technical-indicators/
        stockcharts-technical-rank-sctr

Formula (StockCharts / John Murphy):
  Long-term  (60%):
    c_ema200_pct  = % above/below 200-day EMA          weight 30%
    c_roc125      = 125-day Rate of Change              weight 30%
  Medium-term (30%):
    c_ema50_pct   = % above/below 50-day EMA            weight 15%
    c_roc20       = 20-day Rate of Change               weight 15%
  Short-term  (10%):
    c_rsi14       = 14-day RSI                          weight  5%
    c_ppo_slope   = PPO(12,26,9) histogram 3-day slope  weight  5%
                    (normalized 0-100 via ChartSchool conditional logic)

  raw_score = each component × its weight, summed directly (raw values, not
              per-component percentile ranks — per ChartSchool documentation).

SCTR = percentile rank of raw_score within the universe (0.0 – 99.9).

All components use daily bars only — no monthly data needed.

This module only computes scores and ranks — it does not do I/O.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Component weights (must sum to 1.0)
_W = {
    'c_ema200_pct': 0.30,
    'c_roc125':     0.30,
    'c_ema50_pct':  0.15,
    'c_roc20':      0.15,
    'c_rsi14':      0.05,
    'c_ppo_slope':  0.05,
}

# Minimum daily bars: 200-day EMA needs ~200 bars to converge; 125-day ROC
# needs 126 bars. Keep 210 as a conservative minimum.
_MIN_DAILY = 210


def _roc(series: pd.Series, n: int) -> Optional[float]:
    """Rate of change: (close_t - close_{t-n}) / close_{t-n} * 100."""
    if len(series) <= n:
        return None
    base = series.iloc[-(n + 1)]
    if base == 0 or pd.isna(base):
        return None
    return (series.iloc[-1] - base) / base * 100.0


def _pct_from_ema(close: pd.Series, period: int) -> Optional[float]:
    """(close - EMA_period) / EMA_period * 100."""
    if len(close) < period:
        return None
    ema = close.ewm(span=period, adjust=False).mean().iloc[-1]
    if pd.isna(ema) or ema == 0:
        return None
    return (close.iloc[-1] - ema) / ema * 100.0


def _rsi(close: pd.Series, period: int = 14) -> Optional[float]:
    """
    14-day RSI using Wilder smoothing (equivalent to EWM with alpha=1/period).
    Returns a value in [0, 100].
    """
    if len(close) < period + 1:
        return None
    delta = close.diff().dropna()
    if len(delta) < period:
        return None
    gains  = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    avg_gain = gains.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]
    avg_loss = losses.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _ppo_slope_score(close: pd.Series) -> float:
    """
    PPO(12,26,9) histogram 3-day slope, normalized to a 0–100 score.

    ChartSchool conditional logic:
      slope >= +1  → 100  (full points)
      slope <= -1  →   0  (no points)
      otherwise    → (slope + 1) * 50   (linear interpolation on [-1, +1])

    Returns 50.0 (neutral) when insufficient data.
    """
    if len(close) < 35:
        return 50.0
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    ppo   = (ema12 - ema26) / ema26.replace(0, np.nan) * 100.0
    sig   = ppo.ewm(span=9, adjust=False).mean()
    hist  = ppo - sig
    if len(hist) < 4 or pd.isna(hist.iloc[-1]) or pd.isna(hist.iloc[-4]):
        return 50.0
    slope = (hist.iloc[-1] - hist.iloc[-4]) / 3.0
    if slope >= 1.0:
        return 100.0
    if slope <= -1.0:
        return 0.0
    return (slope + 1.0) * 50.0


def compute_indicators(daily_df: pd.DataFrame) -> Optional[dict]:
    """
    Compute the 6 raw SCTR indicator values for a single security.

    Args:
        daily_df: DataFrame with at least a 'Close' column, daily frequency,
                  sorted ascending, tz-naive DatetimeIndex.
                  Minimum _MIN_DAILY bars required.

    Returns:
        dict with keys:
            c_ema200_pct, c_roc125, c_ema50_pct, c_roc20,
            c_rsi14, c_ppo_slope (normalized 0-100), raw_score
        Returns None if insufficient data.
    """
    if daily_df is None or daily_df.empty:
        return None
    if 'Close' not in daily_df.columns:
        return None

    d = daily_df['Close'].dropna()
    if len(d) < _MIN_DAILY:
        return None

    c_ema200 = _pct_from_ema(d, 200)
    c_roc125 = _roc(d, 125)
    c_ema50  = _pct_from_ema(d, 50)
    c_roc20  = _roc(d, 20)
    c_rsi14  = _rsi(d, 14)

    if any(v is None or (isinstance(v, float) and np.isnan(v))
           for v in (c_ema200, c_roc125, c_ema50, c_roc20, c_rsi14)):
        return None

    c_ppo = _ppo_slope_score(d)

    raw = (
        c_ema200 * _W['c_ema200_pct'] +
        c_roc125 * _W['c_roc125']     +
        c_ema50  * _W['c_ema50_pct']  +
        c_roc20  * _W['c_roc20']      +
        c_rsi14  * _W['c_rsi14']      +
        c_ppo    * _W['c_ppo_slope']
    )

    if np.isnan(raw):
        return None

    return {
        'c_ema200_pct': round(c_ema200, 4),
        'c_roc125':     round(c_roc125, 4),
        'c_ema50_pct':  round(c_ema50,  4),
        'c_roc20':      round(c_roc20,  4),
        'c_rsi14':      round(c_rsi14,  4),
        'c_ppo_slope':  round(c_ppo,    4),
        'raw_score':    round(raw,       4),
    }


def rank_to_sctr(raw_scores: pd.Series) -> pd.Series:
    """
    Convert a Series of raw SCTR scores to percentile ranks 0.0 – 99.9.

    Stocks are ranked by their raw score within the universe, then assigned
    a percentile (same approach as StockCharts).

    Args:
        raw_scores: pd.Series indexed by ticker symbol, values are raw scores.

    Returns:
        pd.Series with the same index, values in [0.0, 99.9].
    """
    if raw_scores.empty:
        return raw_scores.copy()
    return (raw_scores.rank(method='average', pct=True) * 99.9).round(1)
