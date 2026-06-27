"""
Correction Leader Filter — identifies stocks showing institutional accumulation
during a market pullback.

Five signals from the literature review (O'Neil / IBD / Minervini):

  1. Closing Range (CR) > 60%  [O'Neil]
       CR = (Close - Low) / (High - Low) * 100
       A stock closing in the upper 40% of its daily range during a down market
       signals institutional buyers absorbing the selling. We require CR > 60%
       on at least 50% of correction days.

  2. 21-EMA Adherence  [Minervini]
       Stock held above its 21-day EMA on ≥70% of correction days.
       The 21-EMA is Minervini's key short-term trend line; stocks that stay
       above it during corrections are in the strongest institutional hands.
       Also checked: above 50-SMA and 200-SMA at end of correction.

  3. RS Line vs Benchmark  [IBD]
       The stock's RS line (Close / SPY_Close) must be higher at the end of
       the correction than at the start. If the RS line rises while the market
       falls, institutions are accumulating relative to the market.
       rs_change_pct = (rs_end / rs_start - 1) * 100

  4. Pre-Correction Strength  [O'Neil SCTR proxy]
       Was the stock already technically strong BEFORE the correction started?
       Proxy: above 50-SMA + positive 3-month ROC at correction_start.
       Combined with current SCTR still ≥ threshold.
       A stock that was weak before the correction and is "holding up" may just
       be low-beta, not a genuine rotation leader.

  5. Volume Accumulation  [O'Neil]
       Sum of volume on up-days (Close > prev Close) vs down-days during the correction.
       acc_ratio = up_volume / down_volume. > 1.0 = net accumulation.

Composite: weighted score across all five signals.
Hard filter: must pass Signal 1 (CR) + Signal 2 (above 50-SMA at end).

Correction period: auto-detected from SPY (highest close in last 60 days →
correction starts the next trading day). Can be overridden with
--correction-start YYYY-MM-DD.

Output: results/correction_leaders_YYYY-MM-DD.csv
"""

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from src.data_loader import load_daily

logger = logging.getLogger(__name__)

# Thresholds
CR_THRESHOLD        = 60.0   # closing range % to count as "institutional day"
CR_MIN_PCT_DAYS     = 0.50   # at least 50% of correction days must meet CR threshold
EMA21_MIN_PCT_DAYS  = 0.70   # 21-EMA adherence: held above for this fraction of days
MIN_CORRECTION_DAYS = 3      # need at least this many days to run the filter
SCTR_MIN            = 60.0   # minimum current SCTR to include in results
SPY_DROP_THRESHOLD  = 0.02   # SPY must be down at least 2% from recent peak


def detect_correction_window(
    daily_dir: Path,
    lookback_days: int = 60,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """
    Find the most recent correction window in SPY.

    Returns (correction_start, correction_end) where correction_start is
    the first day after the recent peak closed > SPY_DROP_THRESHOLD below it.
    Returns None if SPY is not in a correction.
    """
    spy = load_daily('SPY', daily_dir)
    if spy is None or spy.empty:
        return None

    recent = spy.tail(lookback_days)
    peak_idx   = recent['Close'].idxmax()
    peak_price = recent['Close'].max()
    last_close = recent['Close'].iloc[-1]

    if (peak_price - last_close) / peak_price < SPY_DROP_THRESHOLD:
        logger.info(
            f"SPY not in correction: peak {peak_price:.2f} on {peak_idx.date()}, "
            f"current {last_close:.2f} ({(last_close/peak_price-1)*100:.1f}%)"
        )
        return None

    days_after_peak = recent[recent.index > peak_idx]
    if days_after_peak.empty:
        return None

    corr_start = days_after_peak.index[0]
    corr_end   = recent.index[-1]

    drop_pct = (last_close / peak_price - 1) * 100
    logger.info(
        f"Correction detected: SPY peak {peak_price:.2f} on {peak_idx.date()}, "
        f"current {last_close:.2f} ({drop_pct:.1f}%)  "
        f"Window: {corr_start.date()} → {corr_end.date()} "
        f"({len(days_after_peak)} trading days)"
    )
    return corr_start, corr_end


def _closing_range(df: pd.DataFrame) -> pd.Series:
    """CR = (Close - Low) / (High - Low) * 100. NaN where High == Low."""
    hl = df['High'] - df['Low']
    cr = (df['Close'] - df['Low']) / hl * 100
    return cr.where(hl > 0)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, min_periods=span, adjust=False).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def _load_latest_stock_sctr(config: Config, as_of: date | None) -> pd.DataFrame | None:
    label = str(as_of) if as_of else None
    parts = []
    for bucket in ['large', 'mid', 'small']:
        if label:
            p = config.stock_sctr_dir / f"stock_sctr_{bucket}_{label}.csv"
        else:
            files = sorted(config.stock_sctr_dir.glob(f"stock_sctr_{bucket}_*.csv"))
            p = files[-1] if files else None
        if p and p.exists():
            parts.append(pd.read_csv(p))
    if not parts:
        logger.error("No stock SCTR files found. Run --mode stocks first.")
        return None
    df = pd.concat(parts, ignore_index=True)
    logger.info(f"Loaded stock SCTR: {len(df)} stocks")
    return df


def _load_stocks_by_industry(config: Config) -> pd.DataFrame:
    return pd.read_csv(config.stocks_by_industry_csv,
                       usecols=['symbol', 'sector_name', 'industry_name', 'industry_key'])


def run(
    config: Config,
    correction_start: pd.Timestamp | None = None,
    correction_end:   pd.Timestamp | None = None,
    as_of:            date | None = None,
    sctr_min:         float = SCTR_MIN,
    min_cr_pct_days:  float = CR_MIN_PCT_DAYS,
) -> pd.DataFrame:
    """
    Screen stocks for correction leadership signals (5-filter O'Neil framework).

    Args:
        config:           Project Config.
        correction_start: Start of correction window (auto-detected if None).
        correction_end:   End of correction window (auto-detected if None).
        as_of:            Date for loading SCTR file. None = latest.
        sctr_min:         Minimum current SCTR to enter candidate pool (default 60).
        min_cr_pct_days:  Fraction of correction days needing CR > 60% (default 0.50).

    Returns:
        DataFrame of stocks sorted by composite leader score.
    """
    # --- Correction window ---
    if correction_start is None or correction_end is None:
        window = detect_correction_window(config.daily_dir)
        if window is None:
            logger.warning(
                "No active correction detected in SPY. "
                "Use --correction-start to specify a manual window."
            )
            return pd.DataFrame()
        correction_start, correction_end = window

    logger.info(f"Correction window: {correction_start.date()} → {correction_end.date()}")

    # --- Load SPY once for RS-line computation ---
    spy_daily = load_daily('SPY', config.daily_dir)
    spy_close = None
    if spy_daily is not None and not spy_daily.empty:
        spy_close = spy_daily['Close']
        logger.info(f"Loaded SPY for RS-line computation ({len(spy_close)} days)")
    else:
        logger.warning("SPY data unavailable — RS-line filter will be skipped")

    # --- Stock universe (SCTR filtered) ---
    sctr_df = _load_latest_stock_sctr(config, as_of)
    if sctr_df is None:
        return pd.DataFrame()

    candidates = sctr_df[sctr_df['sctr'] >= sctr_min][['ticker', 'cap_bucket', 'sctr']].copy()
    logger.info(f"Candidates (SCTR >= {sctr_min}): {len(candidates)} stocks")

    # --- GICS mapping ---
    sbi = _load_stocks_by_industry(config)

    # --- Score each candidate ---
    records = []
    total = len(candidates)

    for i, row in enumerate(candidates.itertuples(index=False)):
        ticker = row.ticker
        if i % 200 == 0 and i > 0:
            logger.info(f"  {i}/{total} screened ({len(records)} passing)...")

        daily = load_daily(ticker, config.daily_dir)
        if daily is None or daily.empty:
            continue

        # Full history up to correction end (for MA computation)
        full = daily[daily.index <= correction_end]
        if len(full) < 200:
            continue

        # Correction-window slice
        corr = full[(full.index >= correction_start)].copy()
        if len(corr) < MIN_CORRECTION_DAYS:
            continue

        last_close = float(full['Close'].iloc[-1])

        # ------------------------------------------------------------------
        # Signal 1: Closing Range > 60% on ≥50% of correction days
        # ------------------------------------------------------------------
        cr = _closing_range(corr).dropna()
        if cr.empty:
            continue
        cr_above_pct = float((cr > CR_THRESHOLD).sum() / len(cr))

        # ------------------------------------------------------------------
        # Signal 2: 21-EMA Adherence (≥70% of correction days above EMA21)
        #           + 50-SMA and 200-SMA position at end
        # ------------------------------------------------------------------
        ema21_full = _ema(full['Close'], 21)
        sma50_full = _sma(full['Close'], 50)
        sma200_full= _sma(full['Close'], 200)

        # 21-EMA during correction window
        ema21_corr = ema21_full.reindex(corr.index)
        days_above_ema21 = (corr['Close'] > ema21_corr).sum()
        ema21_pct = float(days_above_ema21 / len(corr))
        held_21ema = ema21_pct >= EMA21_MIN_PCT_DAYS

        # MA positions at end of correction
        above_50sma  = bool(last_close > sma50_full.iloc[-1])
        above_200sma = bool(last_close > sma200_full.iloc[-1])
        above_21ema  = bool(last_close > ema21_full.iloc[-1])

        # ------------------------------------------------------------------
        # Signal 3: RS Line vs SPY (positive change over correction window)
        # ------------------------------------------------------------------
        rs_positive   = False
        rs_change_pct = np.nan

        if spy_close is not None:
            # Align stock close to SPY dates within correction window
            stock_corr = corr['Close']
            spy_corr   = spy_close.reindex(corr.index)
            valid      = spy_corr.dropna()

            if len(valid) >= 2:
                # RS line = stock_close / spy_close
                rs_line  = stock_corr.reindex(valid.index) / valid
                rs_start = float(rs_line.iloc[0])
                rs_end   = float(rs_line.iloc[-1])
                if rs_start > 0:
                    rs_change_pct = (rs_end / rs_start - 1) * 100
                    rs_positive   = rs_end > rs_start

        # ------------------------------------------------------------------
        # Signal 4: Pre-Correction Strength (SCTR proxy at correction start)
        #
        # Check the stock's technical condition on the last trading day
        # BEFORE the correction started. Proxy components:
        #   a) Above 50-SMA at correction start (key SCTR driver)
        #   b) 3-month ROC positive at correction start (trend confirmation)
        #   c) Current SCTR still ≥ threshold (held its rank despite correction)
        # ------------------------------------------------------------------
        pre_full = daily[daily.index < correction_start]

        was_above_50sma_pre = False
        roc_3m_positive_pre = False

        if len(pre_full) >= 65:
            sma50_pre = _sma(pre_full['Close'], 50)
            close_pre = pre_full['Close'].iloc[-1]
            was_above_50sma_pre = bool(close_pre > sma50_pre.iloc[-1])

            # 3-month ROC ≈ 63 trading days
            close_63ago = pre_full['Close'].iloc[-64] if len(pre_full) >= 64 else np.nan
            if not np.isnan(close_63ago) and close_63ago > 0:
                roc_3m_positive_pre = bool(close_pre > close_63ago)

        # Strength score: 0–2 pre-correction signals
        pre_strength_signals = int(was_above_50sma_pre) + int(roc_3m_positive_pre)
        was_strong_pre = pre_strength_signals >= 2

        # ------------------------------------------------------------------
        # Signal 5: Volume Accumulation
        # ------------------------------------------------------------------
        if 'Volume' in corr.columns and len(corr) >= 2:
            prev_close  = corr['Close'].shift(1)
            up_days     = corr['Close'] > prev_close
            down_days   = corr['Close'] < prev_close
            up_vol      = corr.loc[up_days,   'Volume'].sum()
            down_vol    = corr.loc[down_days, 'Volume'].sum()
            acc_ratio   = float(up_vol / down_vol) if down_vol > 0 else 2.0
            is_accumulating = acc_ratio >= 1.0
        else:
            acc_ratio       = np.nan
            is_accumulating = False

        # ------------------------------------------------------------------
        # Drawdown and pre-correction context
        # ------------------------------------------------------------------
        corr_high   = float(corr['High'].max())
        drawdown    = (last_close / corr_high - 1) * 100

        pre_start   = correction_start - pd.Timedelta(days=30)
        pre_window  = daily[(daily.index >= pre_start) & (daily.index < correction_start)]
        pre_high    = float(pre_window['Close'].max()) if not pre_window.empty else np.nan
        vs_pre_high = (last_close / pre_high - 1) * 100 if not np.isnan(pre_high) else np.nan

        # ------------------------------------------------------------------
        # Composite leader score
        #
        # Weights (O'Neil priority order):
        #   CR adherence         25%  — signature institutional signal
        #   RS vs SPY            22%  — relative outperformance during correction
        #   21-EMA adherence     18%  — short-term trend held
        #   Pre-correction strength 15% — was already a leader before correction
        #   Volume accumulation  12%  — institutional buying vs selling
        #   Current SCTR          8%  — still ranked highly
        # ------------------------------------------------------------------
        # CR score: fraction of days with CR > 60%, mapped to 0–100
        cr_score  = cr_above_pct * 100

        # RS score: positive change → good; scored relative to magnitude
        if not np.isnan(rs_change_pct):
            rs_score = min(50 + rs_change_pct * 10, 100)   # +1% RS → 60, +5% RS → 100
            rs_score = max(rs_score, 0)
        else:
            rs_score = 50.0

        # 21-EMA score: fraction above EMA × 100, with bonus for held_21ema
        ema_score = ema21_pct * 100

        # Pre-correction strength score
        pre_score = pre_strength_signals / 2.0 * 100   # 0, 50, or 100

        # Volume accumulation score
        if not np.isnan(acc_ratio):
            vol_score = min(60 + (acc_ratio - 1.0) * 30, 100)
            vol_score = max(vol_score, 0)
        else:
            vol_score = 50.0

        # Current SCTR score (already ≥ sctr_min to be in candidates)
        sctr_score = min(row.sctr, 99.9)

        composite = (
            0.25 * cr_score    +
            0.22 * rs_score    +
            0.18 * ema_score   +
            0.15 * pre_score   +
            0.12 * vol_score   +
            0.08 * sctr_score
        )

        # ------------------------------------------------------------------
        # Filters passed count (for signal_count column)
        # ------------------------------------------------------------------
        filters_passed = sum([
            cr_above_pct  >= min_cr_pct_days,   # Filter 1
            held_21ema,                           # Filter 2
            rs_positive,                          # Filter 3
            was_strong_pre,                       # Filter 4
            is_accumulating,                      # Filter 5
        ])

        # ------------------------------------------------------------------
        # GICS labels
        # ------------------------------------------------------------------
        gics          = sbi[sbi['symbol'] == ticker]
        sector_name   = gics['sector_name'].iloc[0]   if not gics.empty else ''
        industry_name = gics['industry_name'].iloc[0] if not gics.empty else ''
        industry_key  = gics['industry_key'].iloc[0]  if not gics.empty else ''

        records.append({
            'ticker':              ticker,
            'cap_bucket':          row.cap_bucket,
            'sector_name':         sector_name,
            'industry_name':       industry_name,
            'industry_key':        industry_key,
            'sctr':                round(row.sctr, 1),
            'corr_days':           len(corr),
            # Signal 1
            'cr_above_60_pct':     round(cr_above_pct * 100, 1),
            # Signal 2
            'ema21_pct':           round(ema21_pct * 100, 1),
            'held_21ema':          held_21ema,
            'above_21ema':         above_21ema,
            'above_50sma':         above_50sma,
            'above_200sma':        above_200sma,
            # Signal 3
            'rs_change_pct':       round(rs_change_pct, 2) if not np.isnan(rs_change_pct) else None,
            'rs_positive':         rs_positive,
            # Signal 4
            'was_strong_pre':      was_strong_pre,
            'was_above_50sma_pre': was_above_50sma_pre,
            'roc_3m_positive_pre': roc_3m_positive_pre,
            # Signal 5
            'acc_ratio':           round(acc_ratio, 2) if not np.isnan(acc_ratio) else None,
            'is_accumulating':     is_accumulating,
            # Summary
            'filters_passed':      filters_passed,
            'drawdown_pct':        round(drawdown, 1),
            'vs_pre_high_pct':     round(vs_pre_high, 1) if not np.isnan(vs_pre_high) else None,
            'composite':           round(composite, 1),
        })

    if not records:
        logger.warning("No stocks passed the correction leader filter.")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Hard filters: must pass CR + above 50-SMA at end
    df = df[df['above_50sma']].copy()
    df = df[df['cr_above_60_pct'] >= min_cr_pct_days * 100].copy()

    df = df.sort_values('composite', ascending=False).reset_index(drop=True)
    df.insert(0, 'rank', df.index + 1)

    # Distribution log
    f_counts = df['filters_passed'].value_counts().sort_index(ascending=False)
    logger.info(
        f"Correction leaders: {len(df)} stocks pass hard filters  "
        f"(window: {correction_start.date()} → {correction_end.date()}, "
        f"SCTR >= {sctr_min}, CR >= {min_cr_pct_days*100:.0f}% of days)"
    )
    logger.info(f"Filters passed distribution: {f_counts.to_dict()}")
    return df


def save(
    df: pd.DataFrame,
    config: Config,
    as_of: date | None = None,
) -> Path:
    config.setup_dirs()
    label = str(as_of) if as_of else pd.Timestamp.today().strftime('%Y-%m-%d')
    out   = config.results_dir / f"correction_leaders_{label}.csv"
    df.to_csv(out, index=False)
    logger.info(f"Saved correction leaders → {out}  ({len(df)} stocks)")
    return out
