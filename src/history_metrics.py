"""
Shared computation for historical SCTR / Stage-Minervini / RS / performance
tracking across the 145 GICS industries + 11 sector SPDR ETFs.

Deliberately reuses the exact logic already tested elsewhere in this repo
(industry_index.py, sctr_engine.py, stage_analysis.py, rs_percentile.py,
perf_screener.py) instead of reimplementing it, so a historical row for date
D matches what a live run would have produced if it had been run on D.

Uses each industry's market-cap-weighted SYNTHETIC constituent index
(industry_index.py's build_synthetic_daily), not the published ^YH ticker:
this session found real, verified data defects in Yahoo's own ^YH
industry-index feed (e.g. ^YH31110020 jumped +32% overnight on 2026-07-17 and
Open/High/Low silently zeroed out afterward -- see perf_screener.py's
data_flag logic). A historical record built on ^YH would bake those glitches
in permanently; the synthetic index avoids that. Consequence: this history's
SCTR/Stage/RS numbers will not exactly match the ^YH-based "current snapshot"
files (sctr_industry.py, stage_analysis.py, rs_percentile.py all read ^YH
directly) -- expected, not a bug.

Two cross-sectional ranking pools, matching perf_screener.py's convention:
industries are ranked against industries, sector ETFs against sector ETFs.
"""

import logging

import numpy as np
import pandas as pd

from config import Config
from src.data_loader import load_daily
from src.industry_index import preload_price_matrix, build_synthetic_daily
from src.sctr_engine import _W as _SCTR_WEIGHTS, _MIN_DAILY as MIN_DAILY_HISTORY
from src.stage_analysis import _compute as _stage_compute, MIN_DAYS as STAGE_MIN_DAYS
from src.rs_percentile import DAYS_Q4, DAYS_Q3, DAYS_Q2, DAYS_Q1, W4, W3, W2, W1, TOTAL_WEIGHT, MIN_DAYS_REQUIRED
from src.perf_screener import PERIODS, BENCHMARKS, SECTOR_ETF_MAP

logger = logging.getLogger(__name__)


def _sctr_components_series(daily_close: pd.Series) -> pd.DataFrame:
    """
    Vectorized SCTR raw-score time series for one price series.

    Fresh implementation (not ported from the old backfill.py), but uses the
    exact same weights/lookbacks as sctr_engine.compute_indicators() -- via
    the imported _SCTR_WEIGHTS dict -- evaluated at every bar instead of just
    the last one, so a historical row for date D matches what
    compute_indicators() would have produced had it been run on D.
    """
    c = daily_close

    ema200       = c.ewm(span=200, adjust=False).mean()
    c_ema200_pct = (c - ema200) / ema200 * 100
    c_roc125     = c.pct_change(125) * 100

    ema50       = c.ewm(span=50, adjust=False).mean()
    c_ema50_pct = (c - ema50) / ema50 * 100
    c_roc20     = c.pct_change(20) * 100

    delta    = c.diff()
    gains    = delta.clip(lower=0)
    losses   = (-delta).clip(lower=0)
    avg_gain = gains.ewm(alpha=1.0 / 14, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / 14, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    c_rsi14  = 100.0 - (100.0 / (1.0 + rs))

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    ppo   = (ema12 - ema26) / ema26.replace(0, np.nan) * 100.0
    sig   = ppo.ewm(span=9, adjust=False).mean()
    hist  = ppo - sig
    slope = (hist - hist.shift(3)) / 3.0
    c_ppo_slope = np.where(
        slope >= 1.0,  100.0,
        np.where(slope <= -1.0, 0.0, (slope + 1.0) * 50.0)
    )
    c_ppo_slope = pd.Series(c_ppo_slope, index=c.index).where(slope.notna(), other=np.nan)

    raw_score = (
        c_ema200_pct * _SCTR_WEIGHTS['c_ema200_pct'] +
        c_roc125     * _SCTR_WEIGHTS['c_roc125']     +
        c_ema50_pct  * _SCTR_WEIGHTS['c_ema50_pct']  +
        c_roc20      * _SCTR_WEIGHTS['c_roc20']      +
        c_rsi14      * _SCTR_WEIGHTS['c_rsi14']      +
        c_ppo_slope  * _SCTR_WEIGHTS['c_ppo_slope']
    )

    # Zero out rows with insufficient history (before bar MIN_DAILY_HISTORY)
    mask = pd.Series(False, index=c.index)
    mask.iloc[:MIN_DAILY_HISTORY] = True
    raw_score = raw_score.where(~mask, other=np.nan)

    return pd.DataFrame({
        'c_ema200_pct': c_ema200_pct,
        'c_roc125':     c_roc125,
        'c_ema50_pct':  c_ema50_pct,
        'c_roc20':      c_roc20,
        'c_rsi14':      c_rsi14,
        'c_ppo_slope':  c_ppo_slope,
        'raw_score':    raw_score,
    }, index=c.index)


def build_price_series(config: Config) -> tuple[dict[str, pd.Series], list[str], list[str]]:
    """
    One daily Close series per symbol:
      - industries: synthetic market-cap-weighted index from constituents, keyed by industry_key
      - sector ETFs + SPY + QQQ: direct daily Close, keyed by ticker

    Returns (price_series, industry_keys, sector_keys).
    """
    sbi = pd.read_csv(config.stocks_by_industry_csv)
    industry_keys = sorted(sbi['industry_key'].dropna().unique())

    all_tickers = sbi['symbol'].dropna().unique().tolist()
    logger.info(f"Loading {len(all_tickers)} constituent daily closes for synthetic indexes...")
    price_matrix = preload_price_matrix(all_tickers, config.daily_dir)

    series: dict[str, pd.Series] = {}
    skipped = []
    for i, ind_key in enumerate(industry_keys):
        if i % 30 == 0 and i > 0:
            logger.info(f"  Synthetic index: {i}/{len(industry_keys)}...")
        members = sbi[sbi['industry_key'] == ind_key]
        tickers = members['symbol'].dropna().tolist()
        weights = {
            row['symbol']: float(row['marketCap'])
            for _, row in members.iterrows()
            if pd.notna(row.get('marketCap')) and row['marketCap'] > 0
        }
        idx = build_synthetic_daily(tickers, weights, price_matrix)
        if idx is not None and len(idx) >= MIN_DAILY_HISTORY:
            series[ind_key] = idx
        else:
            skipped.append(ind_key)

    if skipped:
        logger.warning(f"Skipped {len(skipped)} industries (insufficient constituent data): {skipped}")

    sector_keys = list(SECTOR_ETF_MAP.keys())
    for etf in sector_keys:
        daily = load_daily(etf, config.daily_dir)
        if daily is not None and not daily.empty:
            series[etf] = daily['Close'].dropna().sort_index()

    for bench in BENCHMARKS:
        daily = load_daily(bench, config.daily_dir)
        if daily is not None and not daily.empty:
            series[bench] = daily['Close'].dropna().sort_index()

    logger.info(f"Price series built: {len(series)} symbols "
                f"({len(industry_keys)} industries + {len(sector_keys)} sectors + {len(BENCHMARKS)} benchmarks)")
    return series, industry_keys, sector_keys


def sctr_wide_matrix(price_series: dict[str, pd.Series], keys: list[str],
                      trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily SCTR (0-99.9) for each key, ranked cross-sectionally among `keys` only."""
    raw = {}
    for key in keys:
        if key not in price_series:
            continue
        raw[key] = _sctr_components_series(price_series[key])['raw_score']
    if not raw:
        return pd.DataFrame(index=trading_dates)
    raw_df = pd.DataFrame(raw).reindex(trading_dates, method='ffill', tolerance=pd.Timedelta('5d'))
    return raw_df.rank(axis=1, method='average', pct=True).mul(99.9).round(1)


def _qexcess_series(p: pd.Series, s: pd.Series, newer: int, older: int) -> pd.Series:
    """
    Vectorized form of rs_percentile._quarterly_excess(), evaluated at every row
    instead of just the last one. p.shift(k) at row i is the value k rows before
    row i -- identical to _quarterly_excess's iloc[-(k+1)] convention when i is
    the last row (and shift(0) is the identity, matching its newer_days=0 case).
    """
    ind = (p.shift(newer) / p.shift(older) - 1) * 100
    spy_r = (s.shift(newer) / s.shift(older) - 1) * 100
    return ind - spy_r


def rs_composite_wide(price_series: dict[str, pd.Series], keys: list[str], spy_key: str,
                       trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Daily RS percentile (1-99, IBD-weighted composite vs SPY), ranked
    cross-sectionally among `keys` only. Mirrors rs_percentile.py's formula
    exactly (Q4 x2, Q3/Q2/Q1 x1, /5), just evaluated at every date at once.
    """
    spy = price_series.get(spy_key)
    if spy is None:
        return pd.DataFrame(index=trading_dates)

    composite = {}
    for key in keys:
        p = price_series.get(key)
        if p is None:
            continue
        common = p.index.intersection(spy.index)
        if len(common) < MIN_DAYS_REQUIRED:
            continue
        pa = p.reindex(common)
        sa = spy.reindex(common)
        q4 = _qexcess_series(pa, sa, 0, DAYS_Q4)
        q3 = _qexcess_series(pa, sa, DAYS_Q4, DAYS_Q3)
        q2 = _qexcess_series(pa, sa, DAYS_Q3, DAYS_Q2)
        q1 = _qexcess_series(pa, sa, DAYS_Q2, DAYS_Q1)
        composite[key] = (q4 * W4 + q3 * W3 + q2 * W2 + q1 * W1) / TOTAL_WEIGHT

    if not composite:
        return pd.DataFrame(index=trading_dates)

    raw_df = pd.DataFrame(composite).reindex(trading_dates, method='ffill', tolerance=pd.Timedelta('5d'))
    ranks = raw_df.rank(axis=1, method='average', pct=True)
    return (ranks * 98 + 1).round()


def flat_returns_wide(price_series: dict[str, pd.Series], keys: list[str],
                       trading_dates: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """
    {period: wide DataFrame} for each of perf_screener.PERIODS, evaluated at
    every date in trading_dates. Calendar-day lookback (not trading-day), same
    semantics as perf_screener._last_close_on_or_before: reindex onto a daily
    calendar grid and forward-fill, so shift(N) lands on the last available
    close N calendar days back.
    """
    out = {p: {} for p in PERIODS}
    for key in keys:
        s = price_series.get(key)
        if s is None or s.empty:
            continue
        full_idx = pd.date_range(s.index.min(), s.index.max(), freq='D')
        s_cal = s.reindex(full_idx).ffill()
        for period, days in PERIODS.items():
            shifted = s_cal.shift(days)
            ret = (s_cal / shifted - 1) * 100
            out[period][key] = ret.reindex(trading_dates)
    return {p: pd.DataFrame(d) for p, d in out.items()}


def assemble_rows(price_series: dict[str, pd.Series], keys: list[str],
                   sctr_wide: pd.DataFrame, rs_wide: pd.DataFrame,
                   returns_wide: dict[str, pd.DataFrame],
                   trading_dates: pd.DatetimeIndex) -> list[dict]:
    """
    The one loop: per (date, key), slice the price series to <= date and call
    stage_analysis._compute() directly -- reusing tested logic rather than a
    bespoke vectorized reimplementation. ~130-180 dates x ~156 symbols is a
    few tens of thousands of calls, all in-memory, well under a minute.
    """
    rows = []
    for key in keys:
        s = price_series.get(key)
        if s is None or len(s) < STAGE_MIN_DAYS:
            continue
        for date in trading_dates:
            sliced = s[s.index <= date]
            if len(sliced) < STAGE_MIN_DAYS:
                continue

            rs_pct = None
            if key in rs_wide.columns and date in rs_wide.index:
                v = rs_wide.at[date, key]
                rs_pct = None if pd.isna(v) else float(v)

            metrics = _stage_compute(sliced, rs_pct=rs_pct)
            if metrics is None:
                continue

            row = {'date': date, 'key': key}
            row['close'] = round(float(sliced.iloc[-1]), 2)
            row['sctr'] = (sctr_wide.at[date, key]
                           if key in sctr_wide.columns and date in sctr_wide.index
                           and not pd.isna(sctr_wide.at[date, key]) else None)
            row['rs_pct_composite'] = rs_pct
            row.update({
                'stage': metrics['stage'],
                'minervini_count': metrics['minervini_count'],
                'minervini_c1_above_150': metrics['minervini_c1_above_150'],
                'minervini_c2_above_200': metrics['minervini_c2_above_200'],
                'minervini_c3_150_gt_200': metrics['minervini_c3_150_gt_200'],
                'minervini_c4_200_rising': metrics['minervini_c4_200_rising'],
                'minervini_c5_30pct_low': metrics['minervini_c5_30pct_low'],
                'minervini_c6_near_high': metrics['minervini_c6_near_high'],
                'minervini_c7_rs70': metrics['minervini_c7_rs70'],
            })

            for period in PERIODS:
                df = returns_wide.get(period)
                v = df.at[date, key] if (df is not None and key in df.columns and date in df.index) else None
                row[f'return_{period}'] = None if v is None or pd.isna(v) else round(float(v), 2)
                for bench in BENCHMARKS:
                    bdf = returns_wide.get(period)
                    bv = (bdf.at[date, bench]
                          if (bdf is not None and bench in bdf.columns and date in bdf.index) else None)
                    rel = None
                    if row[f'return_{period}'] is not None and bv is not None and not pd.isna(bv):
                        rel = round(row[f'return_{period}'] - float(bv), 2)
                    row[f'relret_{bench.lower()}_{period}'] = rel

            rows.append(row)
    return rows
