"""
RS-Line vs Moving Average signals.

Murphy's principle: the RS line (relative strength ratio vs benchmark) has
its OWN momentum structure. Applying a 13-week MA to the RS-Ratio series
gives an earlier rotation signal than waiting for the RRG quadrant to change.

A cross of RS-Ratio above its 13-week MA often precedes entry into the
'Improving' quadrant by 1-3 weeks.  A cross below often confirms an exit
from 'Leading' before the quadrant label changes.

Signals computed per industry (from weekly history):
  rs_ma13           — 13-week rolling mean of rs_ratio
  rs_above_ma       — True when rs_ratio > rs_ma13 (positive RS trend)
  weeks_above_ma    — consecutive weeks above rs_ma13 (0 = currently below)
  rs_cross_up       — rs_ratio crossed above rs_ma13 within last 2 weeks
  rs_cross_down     — rs_ratio crossed below rs_ma13 within last 2 weeks
  rs_ma_slope       — slope of rs_ma13: (current - 4wk_ago) / 4  (pos = rising)
  rs_ma_signal      — human-readable summary label

Output: results/rs_ma_signals_YYYY-MM-DD.csv
"""

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config

logger = logging.getLogger(__name__)

MA_PERIOD   = 13   # weeks — ~quarter
CROSS_WEEKS = 2    # how recently a cross must have happened to be flagged "fresh"


def _rs_ma_for_industry(series: pd.Series) -> dict:
    """
    Compute RS-MA signals from a weekly rs_ratio series (most-recent last).
    Returns a dict of scalar metrics as of the latest date.
    """
    if len(series) < MA_PERIOD + 4:
        return {}

    ma = series.rolling(MA_PERIOD, min_periods=MA_PERIOD).mean()

    # Current state
    current_rs  = float(series.iloc[-1])
    current_ma  = float(ma.iloc[-1])
    above_ma    = current_rs > current_ma

    # Consecutive weeks above MA
    weeks_above = 0
    if above_ma:
        for v, m in zip(reversed(series.tolist()), reversed(ma.tolist())):
            if pd.isna(m):
                break
            if v > m:
                weeks_above += 1
            else:
                break
    else:
        weeks_above = 0  # not above MA

    # Fresh cross detection (last CROSS_WEEKS periods)
    cross_up   = False
    cross_down = False
    window     = min(CROSS_WEEKS + 1, len(series))
    for i in range(1, window):
        prev_rs = series.iloc[-(i + 1)]
        prev_ma = ma.iloc[-(i + 1)]
        curr_rs = series.iloc[-i]
        curr_ma = ma.iloc[-i]
        if pd.isna(prev_ma) or pd.isna(curr_ma):
            continue
        if prev_rs <= prev_ma and curr_rs > curr_ma:
            cross_up = True
        if prev_rs >= prev_ma and curr_rs < curr_ma:
            cross_down = True

    # MA slope over last 4 weeks
    if len(ma.dropna()) >= 5:
        ma_now  = float(ma.iloc[-1])
        ma_4ago = float(ma.dropna().iloc[-5]) if len(ma.dropna()) >= 5 else ma_now
        rs_ma_slope = (ma_now - ma_4ago) / 4.0
    else:
        rs_ma_slope = 0.0

    # Signal label — priority order
    if cross_up:
        signal = 'RS Cross Up'
    elif cross_down:
        signal = 'RS Cross Down'
    elif above_ma and rs_ma_slope > 0.005:
        signal = 'RS Above MA (rising)'
    elif above_ma and rs_ma_slope < -0.005:
        signal = 'RS Above MA (fading)'
    elif above_ma:
        signal = 'RS Above MA'
    elif not above_ma and rs_ma_slope < -0.005:
        signal = 'RS Below MA (falling)'
    elif not above_ma and rs_ma_slope > 0.005:
        signal = 'RS Below MA (recovering)'
    else:
        signal = 'RS Below MA'

    return {
        'rs_ratio_now':  round(current_rs, 3),
        'rs_ma13':       round(current_ma, 3),
        'rs_above_ma':   above_ma,
        'weeks_above_ma': weeks_above,
        'rs_cross_up':   cross_up,
        'rs_cross_down': cross_down,
        'rs_ma_slope':   round(rs_ma_slope, 4),
        'rs_ma_signal':  signal,
    }


def run(config: Config, as_of: date | None = None) -> pd.DataFrame:
    """
    Compute RS-MA signals for all GICS industries from the backfill history.

    Reads results/history/industry_history.csv  (written by backfill.run()).
    Returns one row per industry with rs_ma signal columns.
    """
    hist_path = config.results_dir / 'history' / 'industry_history.csv'
    if not hist_path.exists():
        logger.error("industry_history.csv not found — run --mode backfill first.")
        return pd.DataFrame()

    history = pd.read_csv(hist_path, parse_dates=['date'])
    logger.info(f"Loaded history: {len(history)} rows, "
                f"{history['industry_key'].nunique()} industries")

    if as_of:
        history = history[history['date'] <= pd.Timestamp(as_of)]

    records = []
    for ind_key, grp in history.groupby('industry_key', sort=True):
        grp = grp.sort_values('date')
        rs_series = grp['rs_ratio'].dropna()
        if len(rs_series) < MA_PERIOD + 4:
            continue

        signals = _rs_ma_for_industry(rs_series)
        if not signals:
            continue

        # Carry over latest meta
        last = grp.iloc[-1]
        records.append({
            'industry_key':  ind_key,
            'sector_name':   last.get('sector_name', ''),
            'industry_name': last.get('industry_name', ''),
            'quadrant':      last.get('quadrant', ''),
            **signals,
        })

    if not records:
        logger.error("No RS-MA records computed.")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Sort: fresh crosses first, then above-MA by weeks, then below-MA
    priority = {
        'RS Cross Up':                 0,
        'RS Above MA (rising)':        1,
        'RS Above MA':                 2,
        'RS Above MA (fading)':        3,
        'RS Below MA (recovering)':    4,
        'RS Cross Down':               5,
        'RS Below MA':                 6,
        'RS Below MA (falling)':       7,
    }
    df['_sort_key'] = df['rs_ma_signal'].map(priority).fillna(9)
    df = (df.sort_values(['_sort_key', 'weeks_above_ma'], ascending=[True, False])
            .drop(columns=['_sort_key'])
            .reset_index(drop=True))
    df.insert(0, 'rank', df.index + 1)

    counts = df['rs_ma_signal'].value_counts().to_dict()
    logger.info(f"RS-MA signal distribution: {counts}")
    return df


def save(df: pd.DataFrame, config: Config, as_of: date | None = None) -> Path:
    config.setup_dirs()
    label = str(as_of) if as_of else pd.Timestamp.today().strftime('%Y-%m-%d')
    out   = config.results_dir / f"rs_ma_signals_{label}.csv"
    df.to_csv(out, index=False)
    logger.info(f"Saved RS-MA signals → {out}  ({len(df)} industries)")
    return out
