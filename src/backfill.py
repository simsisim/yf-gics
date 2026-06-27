"""
Historical backfill — builds complete weekly time series of industry SCTR + RRG.

Strategy (vectorized):
  - Load all constituent closes once → wide price matrix (much faster than per-date runs)
  - Build each industry's synthetic daily index as a full time series
  - Compute all 6 SCTR components as rolling series, then rank across industries at
    each weekly Friday → SCTR time series
  - Compute full RS-Ratio / RS-Momentum weekly time series via compute_rrg()
  - Join SCTR + RRG at shared weekly dates → one row per (date, industry)

Output:
  results/history/industry_history.csv     — long format, all industries × all dates
  results/history/industry_history_wide_sctr.csv  — pivot: rows=dates, cols=industries
  results/history/industry_history_wide_rrg.csv

Date range: from earliest viable date (needs ~15 monthly bars + 200 daily bars ≈
2021-01) to the latest data available.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from src.data_loader import load_daily
from src.rrg_engine import _load_weekly, _build_weekly_index, compute_rrg, assign_quadrant

logger = logging.getLogger(__name__)

# Minimum history needed before the first score is valid
MIN_DAILY_HISTORY  = 210   # 200-day SMA needs ~200 bars
MIN_MONTHLY_BARS   = 15    # 14-month ROC needs index t-14

HISTORY_START = pd.Timestamp('2021-01-01')   # earliest date to include in output


# ---------------------------------------------------------------------------
# Vectorized SCTR components
# ---------------------------------------------------------------------------

def _build_industry_daily_index(
    tickers: list[str],
    weights: dict[str, float],
    price_matrix: pd.DataFrame,
) -> pd.Series | None:
    """Build market-cap-weighted daily close index from the pre-loaded price matrix."""
    available = [t for t in tickers if t in price_matrix.columns]
    if len(available) < 3:
        return None

    total_cap = sum(weights.get(t, 0) for t in available)
    if total_cap == 0:
        w = pd.Series({t: 1.0 / len(available) for t in available})
    else:
        w = pd.Series({t: weights.get(t, 0) / total_cap for t in available})

    sub     = price_matrix[available].ffill(limit=5)
    returns = sub.pct_change(fill_method=None)
    idx_ret = returns.mul(w, axis=1).sum(axis=1)
    idx     = (1 + idx_ret).cumprod() * 1000.0
    idx.iloc[0] = 1000.0
    return idx.dropna()


def _sctr_components_series(daily_close: pd.Series) -> pd.DataFrame:
    """
    Compute all 6 SCTR components as time series aligned to daily_close.index.

    Returns a DataFrame with columns: c1, c2, c3, c4, c5, c6, raw_score.
    Values before sufficient history are NaN.
    """
    # Monthly close (last trading day of each month)
    monthly = daily_close.resample('ME').last().dropna()

    # C1, C3, C5: monthly ROC forward-filled to daily
    c1_m = monthly.pct_change(14) * 100
    c3_m = monthly.pct_change(9)  * 100
    c5_m = monthly.pct_change(3)  * 100

    c1 = c1_m.reindex(daily_close.index, method='ffill')
    c3 = c3_m.reindex(daily_close.index, method='ffill')
    c5 = c5_m.reindex(daily_close.index, method='ffill')

    # C2: % above/below 200-day SMA
    sma200 = daily_close.rolling(200, min_periods=200).mean()
    c2     = (daily_close - sma200) / sma200 * 100

    # C4: % above/below 50-day SMA
    sma50 = daily_close.rolling(50, min_periods=50).mean()
    c4    = (daily_close - sma50) / sma50 * 100

    # C6: PPO(12,26,9) histogram 3-day slope
    ema12 = daily_close.ewm(span=12, adjust=False).mean()
    ema26 = daily_close.ewm(span=26, adjust=False).mean()
    ppo   = (ema12 - ema26) / ema26 * 100
    sig   = ppo.ewm(span=9, adjust=False).mean()
    hist  = ppo - sig
    # 3-day slope: (hist[t] - hist[t-3]) / 3, clamped
    c6 = ((hist - hist.shift(3)) / 3.0).clip(-50, 50)

    raw = c1 * 0.30 + c2 * 0.15 + c3 * 0.15 + c4 * 0.15 + c5 * 0.10 + c6 * 0.15

    return pd.DataFrame({'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4,
                         'c5': c5, 'c6': c6, 'raw_score': raw},
                        index=daily_close.index)


def _weekly_fridays(index: pd.DatetimeIndex, start: pd.Timestamp) -> pd.DatetimeIndex:
    """
    Return the last available date in each calendar week (Mon–Sun) at or after start.
    This handles cases where Friday is a holiday by falling back to Thursday/Wednesday.
    """
    s = pd.Series(index, index=index)
    s = s[s >= start]
    # Group by ISO year-week, take the last date in each week
    weeks = s.groupby(s.dt.isocalendar().week.astype(str) + '_' + s.dt.isocalendar().year.astype(str)).last()
    return pd.DatetimeIndex(weeks.values).sort_values()


# ---------------------------------------------------------------------------
# Load all constituents once
# ---------------------------------------------------------------------------

def _load_all_closes(tickers: list[str], daily_dir: Path) -> pd.DataFrame:
    """Load Close series for all tickers into a single wide DataFrame."""
    closes = {}
    for i, ticker in enumerate(tickers):
        if i % 500 == 0 and i > 0:
            logger.info(f"  Loading closes: {i}/{len(tickers)} ({len(closes)} loaded)...")
        daily = load_daily(ticker, daily_dir)
        if daily is not None and not daily.empty and len(daily) >= 100:
            closes[ticker] = daily['Close']

    matrix = pd.DataFrame(closes).sort_index()
    logger.info(f"Loaded price matrix: {len(matrix)} rows × {len(matrix.columns)} tickers")
    return matrix


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(config: Config, start_date: str = '2021-01-01') -> pd.DataFrame:
    """
    Build complete weekly history of industry SCTR + RRG.

    Returns long-format DataFrame with columns:
        date, industry_key, sector_name, industry_name,
        sctr, raw_score, rs_ratio, rs_momentum, quadrant, tail_direction
    """
    start_ts = max(pd.Timestamp(start_date), HISTORY_START)

    sbi        = pd.read_csv(config.stocks_by_industry_csv)
    industries = pd.read_csv(config.industries_csv)
    ind_keys   = sorted(sbi['industry_key'].dropna().unique())

    # --- Load all constituent daily closes once ---
    all_tickers = sbi['symbol'].dropna().unique().tolist()
    logger.info(f"Loading {len(all_tickers)} constituent daily closes...")
    price_matrix = _load_all_closes(all_tickers, config.daily_dir)

    if price_matrix.empty:
        logger.error("No daily data loaded — check daily_dir")
        return pd.DataFrame()

    # Determine weekly snapshots from the available daily dates
    weekly_dates = _weekly_fridays(price_matrix.index, start_ts)
    logger.info(
        f"Weekly snapshot dates: {weekly_dates[0].date()} → {weekly_dates[-1].date()} "
        f"({len(weekly_dates)} weeks)"
    )

    # --- Build SCTR raw_score time series for each industry ---
    logger.info("Building SCTR component time series...")
    raw_score_by_industry = {}   # {ind_key: pd.Series indexed on daily dates}

    for i, ind_key in enumerate(ind_keys):
        if i % 20 == 0 and i > 0:
            logger.info(f"  SCTR series: {i}/{len(ind_keys)}...")

        members = sbi[sbi['industry_key'] == ind_key]
        tickers = members['symbol'].dropna().tolist()
        weights = {
            row['symbol']: float(row['marketCap'])
            for _, row in members.iterrows()
            if pd.notna(row.get('marketCap')) and row['marketCap'] > 0
        }

        idx_daily = _build_industry_daily_index(tickers, weights, price_matrix)
        if idx_daily is None or len(idx_daily) < MIN_DAILY_HISTORY:
            continue

        comps = _sctr_components_series(idx_daily)
        raw_score_by_industry[ind_key] = comps['raw_score']

    logger.info(f"SCTR series built for {len(raw_score_by_industry)} industries")

    if not raw_score_by_industry:
        logger.error("No SCTR series computed — aborting")
        return pd.DataFrame()

    # --- Rank across industries at each weekly date → SCTR ---
    # Build a wide matrix: rows = weekly_dates, cols = ind_keys
    raw_matrix = pd.DataFrame(raw_score_by_industry)
    # Snap to weekly dates using nearest available daily close (ffill)
    raw_weekly = raw_matrix.reindex(weekly_dates, method='ffill', tolerance=pd.Timedelta('5d'))

    # Rank: within each row (date), rank all industries → 0–99.9
    sctr_matrix = raw_weekly.rank(axis=1, method='average', pct=True).mul(99.9).round(1)

    logger.info(
        f"SCTR matrix: {len(sctr_matrix)} weekly dates × {len(sctr_matrix.columns)} industries"
    )

    # --- Build RRG time series for each industry ---
    logger.info("Building RRG time series...")
    spy_weekly = _load_weekly('SPY', config.weekly_dir)
    if spy_weekly is None:
        logger.error("SPY weekly data not found — RRG columns will be empty")
        spy_weekly = pd.Series(dtype=float)

    rrg_by_industry = {}   # {ind_key: (rs_ratio_series, rs_mom_series)}

    for i, ind_key in enumerate(ind_keys):
        if i % 20 == 0 and i > 0:
            logger.info(f"  RRG series: {i}/{len(ind_keys)}...")

        members = sbi[sbi['industry_key'] == ind_key]
        tickers = members['symbol'].dropna().tolist()
        weights = {
            row['symbol']: float(row['marketCap'])
            for _, row in members.iterrows()
            if pd.notna(row.get('marketCap')) and row['marketCap'] > 0
        }

        idx_weekly = _build_weekly_index(tickers, weights, config.weekly_dir)
        if idx_weekly is None:
            continue
        if spy_weekly.empty:
            continue

        try:
            rs_r, rs_m = compute_rrg(idx_weekly, spy_weekly)
            rrg_by_industry[ind_key] = (rs_r, rs_m)
        except Exception as e:
            logger.debug(f"RRG series failed for {ind_key}: {e}")

    logger.info(f"RRG series built for {len(rrg_by_industry)} industries")

    # --- Assemble long-format output ---
    logger.info("Assembling output...")
    ind_meta = industries.set_index('industry_key')[['sector_name', 'industry_name']]

    records = []
    industries_with_data = set(raw_score_by_industry.keys()) | set(rrg_by_industry.keys())

    for ind_key in sorted(industries_with_data):
        meta    = ind_meta.loc[ind_key] if ind_key in ind_meta.index else None
        sec_name = meta['sector_name']   if meta is not None else ''
        ind_name = meta['industry_name'] if meta is not None else ind_key

        # SCTR series (already weekly snapped)
        sctr_series  = sctr_matrix[ind_key]  if ind_key in sctr_matrix.columns  else pd.Series(dtype=float)
        raw_s        = raw_weekly[ind_key]   if ind_key in raw_weekly.columns   else pd.Series(dtype=float)

        # RRG series (re-snap to weekly dates)
        if ind_key in rrg_by_industry:
            rs_r_s, rs_m_s = rrg_by_industry[ind_key]
            rs_r_w = rs_r_s.reindex(weekly_dates, method='ffill', tolerance=pd.Timedelta('30d'))
            rs_m_w = rs_m_s.reindex(weekly_dates, method='ffill', tolerance=pd.Timedelta('30d'))
        else:
            rs_r_w = pd.Series(np.nan, index=weekly_dates)
            rs_m_w = pd.Series(np.nan, index=weekly_dates)

        for dt in weekly_dates:
            sc   = sctr_series.get(dt, np.nan)
            raw  = raw_s.get(dt, np.nan)
            rs_r = rs_r_w.get(dt, np.nan)
            rs_m = rs_m_w.get(dt, np.nan)

            # Skip rows where both SCTR and RRG are NaN (not yet enough history)
            if pd.isna(sc) and pd.isna(rs_r):
                continue

            quad = assign_quadrant(rs_r, rs_m) if not (pd.isna(rs_r) or pd.isna(rs_m)) else ''

            records.append({
                'date':         dt.date(),
                'industry_key': ind_key,
                'sector_name':  sec_name,
                'industry_name': ind_name,
                'sctr':         round(sc,   1) if not pd.isna(sc)   else np.nan,
                'raw_score':    round(raw,  4) if not pd.isna(raw)  else np.nan,
                'rs_ratio':     round(rs_r, 3) if not pd.isna(rs_r) else np.nan,
                'rs_momentum':  round(rs_m, 3) if not pd.isna(rs_m) else np.nan,
                'quadrant':     quad,
            })

    df = pd.DataFrame(records)

    if df.empty:
        logger.error("No records assembled")
        return df

    df = df.sort_values(['date', 'industry_key']).reset_index(drop=True)

    # Add tail_direction (comparing current to 4 weeks ago per industry)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['industry_key', 'date'])

    def _tail_dir(grp: pd.DataFrame) -> pd.Series:
        r4 = grp['rs_ratio'].shift(4)
        m4 = grp['rs_momentum'].shift(4)
        ratio_dir = np.where(grp['rs_ratio'] > r4, 'right', 'left')
        mom_dir   = np.where(grp['rs_momentum'] > m4, 'up', 'down')
        result    = pd.Series(
            [f"{r}-{m}" for r, m in zip(ratio_dir, mom_dir)],
            index=grp.index,
        )
        result[r4.isna()] = 'unknown'
        return result

    df['tail_direction'] = df.groupby('industry_key', group_keys=False).apply(
        lambda g: _tail_dir(g), include_groups=False
    )

    df = df.sort_values(['date', 'industry_key']).reset_index(drop=True)
    df['date'] = df['date'].dt.date

    n_dates      = df['date'].nunique()
    n_industries = df['industry_key'].nunique()
    logger.info(
        f"History assembled: {len(df)} rows  "
        f"({n_industries} industries × {n_dates} dates)"
    )
    return df


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save(df: pd.DataFrame, config: Config) -> dict[str, Path]:
    """
    Save history to results/history/.

    Files:
        industry_history.csv          — long format (all industries × all dates)
        industry_sctr_wide.csv        — pivot: rows=dates, cols=industries (SCTR)
        industry_rs_ratio_wide.csv    — pivot: rows=dates, cols=industries (RS-Ratio)
        industry_rs_momentum_wide.csv — pivot: rows=dates, cols=industries (RS-Momentum)
    """
    out_dir = config.results_dir / 'history'
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Long format
    p = out_dir / 'industry_history.csv'
    df.to_csv(p, index=False)
    paths['long'] = p
    logger.info(f"Saved long-format history → {p}  ({len(df)} rows)")

    # Wide pivots (easier for time-series charting)
    for col, fname in [
        ('sctr',         'industry_sctr_wide.csv'),
        ('rs_ratio',     'industry_rs_ratio_wide.csv'),
        ('rs_momentum',  'industry_rs_momentum_wide.csv'),
    ]:
        try:
            wide = df.pivot(index='date', columns='industry_name', values=col)
            p2   = out_dir / fname
            wide.to_csv(p2)
            paths[col] = p2
            logger.info(f"Saved {col} wide pivot → {p2}")
        except Exception as e:
            logger.warning(f"Could not save {col} wide pivot: {e}")

    return paths
