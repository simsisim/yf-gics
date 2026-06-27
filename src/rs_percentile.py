"""
RS Percentile — IBD/Minervini Relative Strength Rating for GICS Industries.

This module computes a pure price-performance rank for every industry synthetic
index relative to SPY, then assigns each industry a 1–99 percentile score within
the universe of 143 GICS industries.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY THIS IS DIFFERENT FROM SCTR AND RS-RATIO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SCTR       — technical indicator composite (SMA%, ROC, RSI, PPO)
  RS-Ratio   — RRG smoothed relative momentum vs benchmark
  RS Pct     — raw price performance vs SPY, ranked across all peers (THIS)

SCTR measures HOW TECHNICALLY STRONG an industry is.
RS Percentile measures HOW MUCH IT HAS OUTPERFORMED ITS PEERS.

Minervini's requirement: RS Percentile ≥ 80 before buying a stock in that
industry. High SCTR + High RS Percentile = leadership.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IBD/MINERVINI RS RATING FORMULA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  12 months split into 4 non-overlapping quarterly windows:
    Q4 (0-3m ago)   ← most recent, weighted 2×
    Q3 (3-6m ago)   ← weight 1×
    Q2 (6-9m ago)   ← weight 1×
    Q1 (9-12m ago)  ← weight 1×

  Each window: excess_return = industry_return - spy_return
  Composite  = (Q4×2 + Q3 + Q2 + Q1) / 5
  Rank composite across all 143 industries → 1-99 (IBD scale)

RS NEW HIGH (Minervini confirmation signal):
  The RS LINE = daily_industry_close / daily_spy_close (both normalized to 1.0)
  rs_new_high      = RS line within 1% of its 52-week high
  rs_near_new_high = RS line within 3% of its 52-week high

Outputs:
  rs_pct_3m        — 3-month relative performance percentile (1-99)
  rs_pct_6m        — 6-month relative performance percentile (1-99)
  rs_pct_12m       — 12-month relative performance percentile (1-99)
  rs_pct_composite — IBD/Minervini weighted composite percentile (1-99)
  rs_excess_3m     — raw 3-month excess return vs SPY (%)
  rs_excess_6m     — raw 6-month excess return vs SPY (%)
  rs_excess_12m    — raw 12-month excess return vs SPY (%)
  rs_excess_comp   — raw composite excess return (%)
  rs_line_vs_52w   — RS line as % of its 52-week high
  rs_new_high      — True if RS line within 1% of 52-week high
  rs_near_new_high — True if RS line within 3% of 52-week high

Output file: results/rs_percentile_YYYY-MM-DD.csv
"""

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from src.backfill import _load_all_closes, _build_industry_daily_index
from src.data_loader import load_daily

logger = logging.getLogger(__name__)

# ── Lookback windows (trading days) ───────────────────────────────────────
DAYS_Q4 = 63    # 0–3 months  (most recent, 2× weight)
DAYS_Q3 = 126   # 3–6 months
DAYS_Q2 = 189   # 6–9 months
DAYS_Q1 = 252   # 9–12 months

W4, W3, W2, W1 = 2.0, 1.0, 1.0, 1.0
TOTAL_WEIGHT = W4 + W3 + W2 + W1   # = 5.0

RS_NEW_HIGH_WINDOW  = 252   # 52-week window for RS line high
RS_NEW_HIGH_PCT     = 1.0   # within 1% = "at new high"
RS_NEAR_HIGH_PCT    = 3.0   # within 3% = "near new high"

MIN_DAYS_REQUIRED   = DAYS_Q1 + 10   # need at least 12 months + a cushion


def _excess_return(prices: pd.Series, spy: pd.Series, lookback: int) -> float | None:
    """
    Compute excess return of prices vs spy over the last `lookback` trading days.
    Returns (industry_return - spy_return) as a percentage, or None if insufficient data.
    """
    # align
    common = prices.index.intersection(spy.index)
    if len(common) < lookback + 1:
        return None
    p = prices.reindex(common)
    s = spy.reindex(common)
    try:
        ind_ret = (float(p.iloc[-1]) / float(p.iloc[-lookback - 1]) - 1) * 100
        spy_ret = (float(s.iloc[-1]) / float(s.iloc[-lookback - 1]) - 1) * 100
        return round(ind_ret - spy_ret, 4)
    except (ZeroDivisionError, IndexError):
        return None


def _quarterly_excess(prices: pd.Series, spy: pd.Series, newer_days: int, older_days: int) -> float | None:
    """
    Excess return for a non-overlapping quarterly window.

    newer_days: the more-recent boundary expressed as days-ago (e.g. 0 = today, 63 = 3m ago)
    older_days: the older boundary expressed as days-ago  (e.g. 63 = 3m ago, 126 = 6m ago)

    Return = (price at newer_days ago) / (price at older_days ago) - 1
    e.g. Q4: newer=0, older=63  → today / 63d ago - 1  (most recent quarter)
         Q3: newer=63, older=126 → 63d ago / 126d ago - 1
    """
    common = prices.index.intersection(spy.index)
    if len(common) < older_days + 1:
        return None
    p = prices.reindex(common)
    s = spy.reindex(common)
    try:
        # iloc[-1] = today, iloc[-64] = 63 days ago, etc.
        newer_idx = -1          if newer_days == 0 else -(newer_days + 1)
        older_idx = -(older_days + 1)
        ind_ret = (float(p.iloc[newer_idx]) / float(p.iloc[older_idx]) - 1) * 100
        spy_ret = (float(s.iloc[newer_idx]) / float(s.iloc[older_idx]) - 1) * 100
        return round(ind_ret - spy_ret, 4)
    except (ZeroDivisionError, IndexError):
        return None


def _rs_line_metrics(prices: pd.Series, spy: pd.Series) -> dict:
    """
    Compute the RS line (industry / SPY, normalized) and its 52-week high status.
    """
    common = prices.index.intersection(spy.index)
    if len(common) < 20:
        return {
            'rs_line_vs_52w': None,
            'rs_new_high':    False,
            'rs_near_new_high': False,
        }
    p = prices.reindex(common).ffill()
    s = spy.reindex(common).ffill()

    # RS line: normalized so that it starts at 1.0 on the first available date
    rs_line = (p / p.iloc[0]) / (s / s.iloc[0])

    window = min(RS_NEW_HIGH_WINDOW, len(rs_line))
    rs_52w_high = float(rs_line.iloc[-window:].max())
    rs_current  = float(rs_line.iloc[-1])

    pct_from_high = (rs_current / rs_52w_high - 1) * 100

    return {
        'rs_line_vs_52w':  round(pct_from_high, 2),
        'rs_new_high':     abs(pct_from_high) <= RS_NEW_HIGH_PCT,
        'rs_near_new_high': abs(pct_from_high) <= RS_NEAR_HIGH_PCT,
    }


def _compute_metrics(
    ind_daily: pd.Series,
    spy: pd.Series,
) -> dict | None:
    """Compute all RS metrics for one industry's daily price index."""
    common = ind_daily.index.intersection(spy.index)
    if len(common) < MIN_DAYS_REQUIRED:
        return None

    p = ind_daily.reindex(common).ffill()
    s = spy.reindex(common).ffill()

    # Overlapping period excess returns (for percentile ranking)
    exc_3m  = _excess_return(p, s, DAYS_Q4)
    exc_6m  = _excess_return(p, s, DAYS_Q3)
    exc_12m = _excess_return(p, s, DAYS_Q1)

    # Non-overlapping quarterly windows (IBD/Minervini formula)
    # Q4 (0–3m): today / 63d-ago     — most recent, gets 2× weight
    # Q3 (3–6m): 63d-ago / 126d-ago
    # Q2 (6–9m): 126d-ago / 189d-ago
    # Q1 (9–12m): 189d-ago / 252d-ago
    q4 = _quarterly_excess(p, s, newer_days=0,        older_days=DAYS_Q4)
    q3 = _quarterly_excess(p, s, newer_days=DAYS_Q4,  older_days=DAYS_Q3)
    q2 = _quarterly_excess(p, s, newer_days=DAYS_Q3,  older_days=DAYS_Q2)
    q1 = _quarterly_excess(p, s, newer_days=DAYS_Q2,  older_days=DAYS_Q1)

    if any(v is None for v in [q4, q3, q2, q1]):
        exc_comp = None
    else:
        exc_comp = round((q4 * W4 + q3 * W3 + q2 * W2 + q1 * W1) / TOTAL_WEIGHT, 4)

    rs_meta = _rs_line_metrics(p, s)

    return {
        'rs_excess_3m':   exc_3m,
        'rs_excess_6m':   exc_6m,
        'rs_excess_12m':  exc_12m,
        'rs_excess_comp': exc_comp,
        'rs_q4':          q4,
        'rs_q3':          q3,
        'rs_q2':          q2,
        'rs_q1':          q1,
        **rs_meta,
    }


def _rank_to_percentile(series: pd.Series) -> pd.Series:
    """Convert raw values to IBD-style 1-99 integer percentile ranks."""
    ranks = series.rank(pct=True, na_option='keep')
    return (ranks * 98 + 1).round().astype('Int64')


def run(config: Config, as_of: date | None = None) -> pd.DataFrame:
    """
    Compute RS Percentile ratings for all GICS industries.

    Returns a DataFrame with one row per industry, sorted by rs_pct_composite desc.
    """
    sbi        = pd.read_csv(config.stocks_by_industry_csv)
    industries = pd.read_csv(config.industries_csv)
    ind_keys   = sorted(sbi['industry_key'].dropna().unique())

    cutoff = pd.Timestamp(as_of) if as_of else None

    # Load SPY once
    spy_daily = load_daily('SPY', config.daily_dir)
    if spy_daily is None or spy_daily.empty:
        logger.error("SPY daily data not found in daily_dir")
        return pd.DataFrame()
    spy = spy_daily['Close'].sort_index()
    if cutoff is not None:
        spy = spy[spy.index <= cutoff]

    # Load all constituent closes once (same as ATH monitor)
    all_tickers = sbi['symbol'].dropna().unique().tolist()
    logger.info(f"Loading {len(all_tickers)} constituent closes for RS Percentile...")
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
            logger.info(f"  {i}/{len(ind_keys)} industries processed...")

        members = sbi[sbi['industry_key'] == ind_key]
        tickers = members['symbol'].dropna().tolist()
        weights = {
            row['symbol']: float(row['marketCap'])
            for _, row in members.iterrows()
            if pd.notna(row.get('marketCap')) and row['marketCap'] > 0
        }

        idx_daily = _build_industry_daily_index(tickers, weights, price_matrix)
        if idx_daily is None or len(idx_daily) < MIN_DAYS_REQUIRED:
            skipped.append(ind_key)
            continue

        metrics = _compute_metrics(idx_daily, spy)
        if metrics is None:
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
        logger.error("No RS Percentile records computed")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # ── Compute percentile ranks across the universe ──────────────────────
    for col, pct_col in [
        ('rs_excess_3m',   'rs_pct_3m'),
        ('rs_excess_6m',   'rs_pct_6m'),
        ('rs_excess_12m',  'rs_pct_12m'),
        ('rs_excess_comp', 'rs_pct_composite'),
    ]:
        valid = df[col].dropna()
        if not valid.empty:
            df[pct_col] = _rank_to_percentile(df[col])
        else:
            df[pct_col] = pd.NA

    # ── Sort by composite percentile ──────────────────────────────────────
    df = (df.sort_values('rs_pct_composite', ascending=False, na_position='last')
            .reset_index(drop=True))
    df.insert(0, 'rank', df.index + 1)

    top5 = df[['industry_name', 'rs_pct_composite', 'rs_pct_3m', 'rs_new_high']].head(5).to_dict('records')
    logger.info(f"RS Percentile computed for {len(df)} industries. Top 5: {top5}")

    return df


def save(df: pd.DataFrame, config: Config, as_of: date | None = None) -> Path:
    config.setup_dirs()
    label = str(as_of) if as_of else pd.Timestamp.today().strftime('%Y-%m-%d')
    out   = config.results_dir / f"rs_percentile_{label}.csv"

    cols = [
        'rank', 'industry_key', 'sector_name', 'industry_name',
        'rs_pct_composite', 'rs_pct_3m', 'rs_pct_6m', 'rs_pct_12m',
        'rs_excess_comp', 'rs_excess_3m', 'rs_excess_6m', 'rs_excess_12m',
        'rs_q4', 'rs_q3', 'rs_q2', 'rs_q1',
        'rs_line_vs_52w', 'rs_new_high', 'rs_near_new_high',
    ]
    cols_present = [c for c in cols if c in df.columns]
    df[cols_present].to_csv(out, index=False)
    logger.info(f"Saved RS Percentile → {out}  ({len(df)} industries)")
    return out


def print_report(df: pd.DataFrame, top_n: int = 20) -> None:
    """Print a ranked table of RS Percentile results."""
    DIVIDER = '─' * 90

    print(f"\n{DIVIDER}")
    print(f"  RS PERCENTILE RANKING (IBD/Minervini Formula)  —  {len(df)} industries")
    print(f"  Composite weights: Q4 (0-3m) ×2, Q3 (3-6m) ×1, Q2 (6-9m) ×1, Q1 (9-12m) ×1")
    print(DIVIDER)
    print(f"  {'Rnk':>3}  {'Industry':<32}  {'Comp':>4}  {'3m':>4}  {'6m':>4}  "
          f"{'12m':>4}  {'Exc%':>7}  {'RsLine%':>8}  {'Signal'}")
    print(DIVIDER)

    for _, row in df.head(top_n).iterrows():
        nh = '★' if row.get('rs_new_high') else ('◈' if row.get('rs_near_new_high') else ' ')
        comp_val  = int(row['rs_pct_composite'])  if pd.notna(row.get('rs_pct_composite'))  else  0
        pct_3m    = int(row['rs_pct_3m'])         if pd.notna(row.get('rs_pct_3m'))         else  0
        pct_6m    = int(row['rs_pct_6m'])         if pd.notna(row.get('rs_pct_6m'))         else  0
        pct_12m   = int(row['rs_pct_12m'])        if pd.notna(row.get('rs_pct_12m'))        else  0
        exc       = row.get('rs_excess_comp', 0) or 0
        rs_line   = row.get('rs_line_vs_52w', 0) or 0
        print(f"  {int(row['rank']):>3}  {str(row['industry_name']):<32}  "
              f"{comp_val:>4}  {pct_3m:>4}  {pct_6m:>4}  {pct_12m:>4}  "
              f"{exc:>+7.1f}%  {rs_line:>+7.1f}%  {nh}")

    print(DIVIDER)
    if len(df) > top_n:
        print(f"  ... {len(df) - top_n} more industries in CSV")

    # ── Summary buckets ───────────────────────────────────────────────────
    print()
    thresholds = [(90, 99, 'Elite    (90-99)'),
                  (80, 89, 'Leaders  (80-89)'),
                  (70, 79, 'Strong   (70-79)'),
                  (50, 69, 'Average  (50-69)'),
                  (1,  49, 'Lagging  ( <50) ')]
    comp_col = df['rs_pct_composite'].dropna().astype(int)
    new_hi_n = df['rs_new_high'].sum() if 'rs_new_high' in df.columns else 0
    near_hi_n = df['rs_near_new_high'].sum() if 'rs_near_new_high' in df.columns else 0

    print(f"  {'Bucket':<22}  Count")
    for lo, hi, label in thresholds:
        n = ((comp_col >= lo) & (comp_col <= hi)).sum()
        bar = '█' * (n // 2)
        print(f"  {label}  {n:>3}  {bar}")

    print(f"\n  ★ RS New High   : {int(new_hi_n)} industries")
    print(f"  ◈ Near New High : {int(near_hi_n)} industries")
    print(DIVIDER)
