"""
Monthly ATH (All-Time High) Breakout Monitor.

From the literature review (Murphy): an industry making new all-time highs on the
monthly chart is definitively rotating in — it is the highest-confidence signal
in intermarket analysis because it confirms the rotation has macro support and is
not mean-reversion noise.

For each GICS industry this module:
  1. Builds the market-cap-weighted synthetic daily index (reuses backfill logic).
  2. Resamples to monthly closes.
  3. Computes ATH metrics against the full history since 2020.
  4. Labels the breakout status and how long it has been holding.

Signal labels (in descending conviction):
  New ATH          — current month is a new all-time high
  Holding Breakout — was at new ATH within last 3 months, still above prev ATH level
  Near ATH         — within 3% of ATH but not yet broken through
  Recovering       — below ATH by 3–15%, was at ATH within 12 months
  Below ATH        — more than 15% below ATH

Output: results/ath_monitor_YYYY-MM-DD.csv
"""

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from src.backfill import _load_all_closes, _build_industry_daily_index

logger = logging.getLogger(__name__)

# Thresholds
NEAR_ATH_PCT   = 3.0    # within 3% = "Near ATH"
RECOVER_PCT    = 15.0   # within 15% = "Recovering" (if previously at ATH)
RECENT_MONTHS  = 3      # ATH within this many months = "recent"
HOLD_MONTHS    = 12     # ATH within this many months still counts as "recovering"


def _ath_metrics(monthly: pd.Series) -> dict:
    """
    Compute all ATH metrics for a monthly close series.

    Returns a dict with keys:
        current_close, monthly_ath, ath_date, pct_from_ath, months_since_ath,
        new_ath, recent_ath, prev_ath_level, months_holding_above_prev_ath,
        ath_signal, ath_score (0–100)
    """
    if monthly.empty or len(monthly) < 6:
        return {}

    current       = float(monthly.iloc[-1])
    current_date  = monthly.index[-1]

    # Rolling ATH up to (but not including) current month → previous ATH
    # This lets us detect if current month is a NEW ATH
    prior_monthly  = monthly.iloc[:-1]
    prev_ath_level = float(prior_monthly.max())
    prev_ath_date  = prior_monthly.idxmax()

    # Is current month a new all-time high?
    new_ath = current > prev_ath_level

    # Overall ATH including current month
    ath_level = max(current, prev_ath_level)
    ath_date  = current_date if new_ath else prev_ath_date
    pct_from_ath = (current / ath_level - 1) * 100

    # Months since ATH
    months_since_ath = 0
    if not new_ath:
        # Count months from ath_date to current_date
        months_since_ath = (
            (current_date.year - ath_date.year) * 12 +
            (current_date.month - ath_date.month)
        )

    # How many consecutive months has the index been ABOVE the prev_ath_level?
    # (measures how well it's holding the breakout)
    months_holding = 0
    if new_ath or pct_from_ath > -NEAR_ATH_PCT:
        # Walk backwards from current month
        for close in reversed(monthly.iloc[:-1 if not new_ath else len(monthly)].tolist()):
            if close >= prev_ath_level * 0.995:   # 0.5% tolerance for rounding
                months_holding += 1
            else:
                break

    # Signal label
    if new_ath:
        signal = 'New ATH'
        score  = 100.0
    elif months_since_ath <= RECENT_MONTHS and current >= prev_ath_level * 0.985:
        signal = 'Holding Breakout'
        score  = 85.0
    elif pct_from_ath >= -NEAR_ATH_PCT:
        signal = 'Near ATH'
        score  = 70.0
    elif months_since_ath <= HOLD_MONTHS and pct_from_ath >= -RECOVER_PCT:
        signal = 'Recovering'
        score  = 40.0 + (RECOVER_PCT + pct_from_ath) / RECOVER_PCT * 20  # 40–60
    else:
        signal = 'Below ATH'
        score  = max(0.0, 20.0 + pct_from_ath)   # falls toward 0 the further below

    return {
        'current_monthly':           round(current, 2),
        'monthly_ath':               round(ath_level, 2),
        'ath_date':                  ath_date.date(),
        'pct_from_ath':              round(pct_from_ath, 2),
        'months_since_ath':          months_since_ath,
        'new_ath':                   new_ath,
        'recent_ath':                months_since_ath <= RECENT_MONTHS,
        'prev_ath_level':            round(prev_ath_level, 2),
        'months_holding_above_prev': months_holding,
        'ath_signal':                signal,
        'ath_score':                 round(score, 1),
    }


def run(config: Config, as_of: date | None = None) -> pd.DataFrame:
    """
    Compute ATH status for all GICS industries.

    Args:
        config: Project Config.
        as_of:  Cutoff date (for backtesting). None = latest.

    Returns:
        DataFrame sorted by ath_score desc (New ATH first), one row per industry.
    """
    sbi        = pd.read_csv(config.stocks_by_industry_csv)
    industries = pd.read_csv(config.industries_csv)
    ind_keys   = sorted(sbi['industry_key'].dropna().unique())

    cutoff = pd.Timestamp(as_of) if as_of else None

    # Load all constituent closes once
    all_tickers = sbi['symbol'].dropna().unique().tolist()
    logger.info(f"Loading {len(all_tickers)} constituent closes for ATH monitor...")
    price_matrix = _load_all_closes(all_tickers, config.daily_dir)

    if price_matrix.empty:
        logger.error("No daily data — check daily_dir")
        return pd.DataFrame()

    if cutoff is not None:
        price_matrix = price_matrix[price_matrix.index <= cutoff]

    records = []
    skipped = []

    for i, ind_key in enumerate(ind_keys):
        if i % 30 == 0 and i > 0:
            logger.info(f"  {i}/{len(ind_keys)} processed ({len(records)} scored)...")

        members = sbi[sbi['industry_key'] == ind_key]
        tickers = members['symbol'].dropna().tolist()
        weights = {
            row['symbol']: float(row['marketCap'])
            for _, row in members.iterrows()
            if pd.notna(row.get('marketCap')) and row['marketCap'] > 0
        }

        idx_daily = _build_industry_daily_index(tickers, weights, price_matrix)
        if idx_daily is None or len(idx_daily) < 60:
            skipped.append(ind_key)
            continue

        monthly = idx_daily.resample('ME').last().dropna()
        if len(monthly) < 6:
            skipped.append(ind_key)
            continue

        metrics = _ath_metrics(monthly)
        if not metrics:
            skipped.append(ind_key)
            continue

        meta          = industries[industries['industry_key'] == ind_key]
        sector_name   = meta['sector_name'].iloc[0]   if len(meta) else ''
        industry_name = meta['industry_name'].iloc[0] if len(meta) else ind_key

        records.append({
            'industry_key':  ind_key,
            'sector_name':   sector_name,
            'industry_name': industry_name,
            **metrics,
        })

    if skipped:
        logger.info(f"Skipped {len(skipped)} industries (insufficient data)")

    if not records:
        logger.error("No ATH records computed")
        return pd.DataFrame()

    df = (
        pd.DataFrame(records)
        .sort_values(['ath_score', 'pct_from_ath'], ascending=[False, False])
        .reset_index(drop=True)
    )
    df.insert(0, 'rank', df.index + 1)

    counts = df['ath_signal'].value_counts()
    logger.info(f"ATH signal distribution: {counts.to_dict()}")
    return df


def save(df: pd.DataFrame, config: Config, as_of: date | None = None) -> Path:
    config.setup_dirs()
    label = str(as_of) if as_of else pd.Timestamp.today().strftime('%Y-%m-%d')
    out   = config.results_dir / f"ath_monitor_{label}.csv"
    df.to_csv(out, index=False)
    logger.info(f"Saved ATH monitor → {out}  ({len(df)} industries)")
    return out
