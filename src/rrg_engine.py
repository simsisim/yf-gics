"""
RRG (Relative Rotation Graph) for GICS industries.

Methodology mirrors the default "stockcharts" method from RRGs-Dashboard project,
which was found to produce results closest to Julius de Kempenaer's charts.

Formula (weekly closes, benchmark = SPY):
  RS         = 100 * (industry_close / SPY_close)
  RS-Ratio   = 100 + z-score(RS, window)              — centered at 100
  RS-MomROC  = 100 * ((RS-Ratio / RS-Ratio.shift(roc_period)) - 1)
  RS-Momentum= 100 + z-score(RS-MomROC, window)
  Post-smooth: SMA(smoothing_period) on both series

Quadrants (axes cross at 100, 100):
  Leading:   RS-Ratio > 100, RS-Momentum > 100  — strong and gaining
  Weakening: RS-Ratio > 100, RS-Momentum < 100  — strong but fading
  Lagging:   RS-Ratio < 100, RS-Momentum < 100  — weak and losing
  Improving: RS-Ratio < 100, RS-Momentum > 100  — weak but turning

Input:  weekly OHLCV from market_data/weekly/ + constituent mapping
Output: results/rrg/rrg_gics_YYYY-MM-DD.csv
"""

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config

logger = logging.getLogger(__name__)

# --- RRG default parameters (matching RRGs-Dashboard "stockcharts" defaults for weekly) ---
WINDOW          = 10   # z-score rolling window (10 weeks ≈ 2.5 months)
ROC_PERIOD      = 10   # shift for RS-Ratio ROC
SMOOTHING       = 3    # post-smoothing SMA period (light smoothing for weekly)
MIN_WEEKS       = 60   # minimum weekly bars needed (WINDOW + ROC_PERIOD + tail + buffer)
MIN_CONSTITUENTS = 3


# ---------------------------------------------------------------------------
# Core RRG formula  (extracted from RRGs-Dashboard calculate_rrg_components)
# ---------------------------------------------------------------------------

def _zscore_centered(series: pd.Series, window: int) -> pd.Series:
    """100 + rolling z-score (ddof=0), centered at 100."""
    mu  = series.rolling(window=window).mean()
    sig = series.rolling(window=window).std(ddof=0)
    return 100 + (series - mu) / sig


def compute_rrg(
    asset_weekly: pd.Series,
    benchmark_weekly: pd.Series,
    window: int = WINDOW,
    roc_period: int = ROC_PERIOD,
    smoothing: int = SMOOTHING,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute RS-Ratio and RS-Momentum for one asset vs benchmark.

    Args:
        asset_weekly:     Weekly close prices for the industry index.
        benchmark_weekly: Weekly close prices for SPY.
        window:           Rolling window for z-score normalization.
        roc_period:       Shift period for RS-Ratio rate-of-change.
        smoothing:        Post-calculation SMA smoothing period.

    Returns:
        (rs_ratio, rs_momentum) — aligned Series on the same DatetimeIndex.
    """
    asset, bench = asset_weekly.align(benchmark_weekly, join='inner')
    asset   = asset.dropna()
    bench   = bench.dropna()
    asset, bench = asset.align(bench, join='inner')

    rs          = 100 * (asset / bench)
    rs_ratio    = _zscore_centered(rs, window).dropna()

    rs_roc      = 100 * ((rs_ratio / rs_ratio.shift(roc_period)) - 1)
    rs_momentum = _zscore_centered(rs_roc, window).dropna()

    if smoothing > 1:
        rs_ratio    = rs_ratio.rolling(window=smoothing).mean().dropna()
        rs_momentum = rs_momentum.rolling(window=smoothing).mean().dropna()

    # Align to common index
    common = rs_ratio.index.intersection(rs_momentum.index)
    return rs_ratio.loc[common], rs_momentum.loc[common]


def assign_quadrant(rs_ratio: float, rs_momentum: float) -> str:
    if rs_ratio >= 100 and rs_momentum >= 100:
        return 'Leading'
    elif rs_ratio >= 100 and rs_momentum < 100:
        return 'Weakening'
    elif rs_ratio < 100 and rs_momentum < 100:
        return 'Lagging'
    else:
        return 'Improving'


# ---------------------------------------------------------------------------
# Synthetic weekly industry index builder
# ---------------------------------------------------------------------------

def _load_monthly(ticker: str, monthly_dir: Path) -> pd.Series | None:
    """
    Load monthly close for a single ticker.
    Monthly files use timezone-aware timestamps: '2026-06-01 00:00:00-04:00'.
    Strip the time/zone component and keep just the date.
    """
    path = monthly_dir / f"{ticker}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col='Date', parse_dates=False)
        df.index = pd.to_datetime(df.index.str.split(' ').str[0], errors='coerce')
        df = df[df.index.notna()].sort_index()
        if 'Close' not in df.columns:
            return None
        df = df[~df.index.duplicated(keep='last')]
        return df['Close'].dropna()
    except Exception as e:
        logger.debug(f"Could not load monthly {ticker}: {e}")
        return None


def _build_monthly_index(
    tickers: list[str],
    weights: dict[str, float],
    monthly_dir: Path,
    cutoff: pd.Timestamp | None = None,
    min_months: int = 24,
) -> pd.Series | None:
    """
    Build a market-cap-weighted monthly close index from constituent stocks.
    Returns a Series of monthly closes starting at 1000, or None if insufficient data.
    """
    closes = {}
    for ticker in tickers:
        s = _load_monthly(ticker, monthly_dir)
        if s is None or len(s) < min_months:
            continue
        s = s[~s.index.duplicated(keep='last')]
        closes[ticker] = s

    if len(closes) < MIN_CONSTITUENTS:
        return None

    price_df = pd.DataFrame(closes).sort_index().ffill(limit=2)

    if cutoff is not None:
        price_df = price_df[price_df.index <= cutoff]

    if len(price_df) < min_months:
        return None

    available = list(closes.keys())
    total_cap  = sum(weights.get(t, 0) for t in available)
    if total_cap == 0:
        w = pd.Series({t: 1.0 / len(available) for t in available})
    else:
        w = pd.Series({t: weights.get(t, 0) / total_cap for t in available})

    returns       = price_df[available].pct_change(fill_method=None)
    index_returns = returns.mul(w, axis=1).sum(axis=1)
    index_close   = (1 + index_returns).cumprod() * 1000.0
    index_close.iloc[0] = 1000.0
    return index_close.dropna()


def _load_weekly(ticker: str, weekly_dir: Path) -> pd.Series | None:
    """
    Load weekly close for a single ticker.
    Cannot use _read_csv because that filters to weekday < 5 — weekly bars
    are indexed on Saturdays/Sundays (start-of-week convention in yfinance).
    """
    path = weekly_dir / f"{ticker}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col='Date', parse_dates=False)
        df.index = pd.to_datetime(df.index.str.split(' ').str[0], errors='coerce')
        df = df[df.index.notna()].sort_index()
        if 'Close' not in df.columns:
            return None
        # Drop duplicate dates
        df = df[~df.index.duplicated(keep='last')]
        return df['Close'].dropna()
    except Exception as e:
        logger.debug(f"Could not load weekly {ticker}: {e}")
        return None


def _build_weekly_index(
    tickers: list[str],
    weights: dict[str, float],
    weekly_dir: Path,
    cutoff: pd.Timestamp | None = None,
) -> pd.Series | None:
    """
    Build a market-cap-weighted weekly close index from constituent stocks.
    Returns a Series of weekly closes starting at 1000, or None if insufficient data.
    """
    closes = {}
    for ticker in tickers:
        s = _load_weekly(ticker, weekly_dir)
        if s is None or len(s) < MIN_WEEKS:
            continue
        # Drop duplicate dates before storing
        s = s[~s.index.duplicated(keep='last')]
        closes[ticker] = s

    if len(closes) < MIN_CONSTITUENTS:
        return None

    price_df = pd.DataFrame(closes).sort_index().ffill(limit=2)

    if cutoff is not None:
        price_df = price_df[price_df.index <= cutoff]

    if len(price_df) < MIN_WEEKS:
        return None

    available  = list(closes.keys())
    total_cap  = sum(weights.get(t, 0) for t in available)
    if total_cap == 0:
        w = pd.Series({t: 1.0 / len(available) for t in available})
    else:
        w = pd.Series({t: weights.get(t, 0) / total_cap for t in available})

    returns       = price_df[available].pct_change(fill_method=None)
    index_returns = returns.mul(w, axis=1).sum(axis=1)
    index_close   = (1 + index_returns).cumprod() * 1000.0
    index_close.iloc[0] = 1000.0
    return index_close.dropna()


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(config: Config, as_of: date | None = None, tail_weeks: int = 4) -> pd.DataFrame:
    """
    Compute RRG position for all GICS industries.

    Args:
        config:     Project Config.
        as_of:      Cutoff date (for backtesting). None = latest.
        tail_weeks: How many weeks of tail history to include in output.

    Returns:
        DataFrame sorted by rs_ratio descending with columns:
            industry_key, sector_key, sector_name, industry_name,
            constituent_count, rs_ratio, rs_momentum, quadrant,
            rs_ratio_{N}w, rs_momentum_{N}w  (tail columns for each tail week),
            tail_direction
    """
    sbi        = pd.read_csv(config.stocks_by_industry_csv)
    industries = pd.read_csv(config.industries_csv)
    weekly_dir = config.weekly_dir

    # Load SPY benchmark
    spy = _load_weekly('SPY', weekly_dir)
    if spy is None:
        logger.error("SPY weekly data not found — cannot compute RRG")
        return pd.DataFrame()

    cutoff = pd.Timestamp(as_of) if as_of else None
    if cutoff is not None:
        spy = spy[spy.index <= cutoff]

    records  = []
    skipped  = []
    ind_keys = sorted(sbi['industry_key'].dropna().unique())

    logger.info(f"Computing RRG for {len(ind_keys)} GICS industries vs SPY (weekly)...")

    for i, ind_key in enumerate(ind_keys):
        if i % 20 == 0 and i > 0:
            logger.info(f"  {i}/{len(ind_keys)} processed ({len(records)} scored)...")

        members = sbi[sbi['industry_key'] == ind_key]
        tickers = members['symbol'].dropna().tolist()

        weights = {
            row['symbol']: float(row['marketCap'])
            for _, row in members.iterrows()
            if pd.notna(row.get('marketCap')) and row['marketCap'] > 0
        }

        idx_weekly = _build_weekly_index(tickers, weights, weekly_dir, cutoff)
        if idx_weekly is None:
            skipped.append(ind_key)
            continue

        try:
            rs_ratio_s, rs_mom_s = compute_rrg(idx_weekly, spy)
        except Exception as e:
            logger.debug(f"RRG failed for {ind_key}: {e}")
            skipped.append(ind_key)
            continue

        if rs_ratio_s.empty or rs_mom_s.empty:
            skipped.append(ind_key)
            continue

        # Current values
        rs_r = float(rs_ratio_s.iloc[-1])
        rs_m = float(rs_mom_s.iloc[-1])

        # Tail history
        tail_data = {}
        for w in range(1, tail_weeks + 1):
            idx = -(w + 1)
            tail_data[f'rs_ratio_{w}w']   = float(rs_ratio_s.iloc[idx]) if len(rs_ratio_s) > w else np.nan
            tail_data[f'rs_momentum_{w}w'] = float(rs_mom_s.iloc[idx])  if len(rs_mom_s)  > w else np.nan

        # Tail direction: is ratio improving (moving right) over last 4 weeks?
        if len(rs_ratio_s) >= 5:
            ratio_4w_ago   = float(rs_ratio_s.iloc[-5])
            mom_4w_ago     = float(rs_mom_s.iloc[-5])
            ratio_trending = 'right' if rs_r > ratio_4w_ago else 'left'
            mom_trending   = 'up'    if rs_m > mom_4w_ago   else 'down'
            tail_direction = f"{ratio_trending}-{mom_trending}"
        else:
            tail_direction = 'unknown'

        meta          = industries[industries['industry_key'] == ind_key]
        sector_key    = meta['sector_key'].iloc[0]    if len(meta) else ''
        sector_name   = meta['sector_name'].iloc[0]   if len(meta) else ''
        industry_name = meta['industry_name'].iloc[0] if len(meta) else ind_key

        available_count = sum(1 for t in tickers if (weekly_dir / f'{t}.csv').exists())

        records.append({
            'industry_key':      ind_key,
            'sector_key':        sector_key,
            'sector_name':       sector_name,
            'industry_name':     industry_name,
            'constituent_count': available_count,
            'rs_ratio':          round(rs_r, 3),
            'rs_momentum':       round(rs_m, 3),
            'quadrant':          assign_quadrant(rs_r, rs_m),
            'tail_direction':    tail_direction,
            **{k: round(v, 3) if not np.isnan(v) else np.nan for k, v in tail_data.items()},
        })

    if skipped:
        logger.info(f"Skipped {len(skipped)} industries: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values('rs_ratio', ascending=False).reset_index(drop=True)
    logger.info(f"RRG computed for {len(df)} industries")

    counts = df['quadrant'].value_counts()
    logger.info(f"Quadrant distribution: {counts.to_dict()}")
    return df


def run_monthly(config: Config, as_of: date | None = None, tail_months: int = 4) -> pd.DataFrame:
    """
    Compute monthly RRG position for all GICS industries vs SPY.

    Same formula as run() but on monthly bars.  Monthly RRG resolves the
    "pullback vs structural rotation" ambiguity: if weekly is Weakening but
    monthly is still Leading, the move is shallow noise.

    Args:
        config:      Project Config.
        as_of:       Cutoff date (for backtesting).  None = latest.
        tail_months: Months of tail history to include in output.

    Returns:
        DataFrame with columns: industry_key, sector_name, industry_name,
        rs_ratio_m, rs_momentum_m, quadrant_m, tail_direction_m,
        rs_ratio_m_{N}mo, rs_momentum_m_{N}mo  (tail columns).
    """
    sbi        = pd.read_csv(config.stocks_by_industry_csv)
    industries = pd.read_csv(config.industries_csv)
    monthly_dir = config.monthly_dir

    spy = _load_monthly('SPY', monthly_dir)
    if spy is None:
        logger.error("SPY monthly data not found — cannot compute monthly RRG")
        return pd.DataFrame()

    cutoff = pd.Timestamp(as_of) if as_of else None
    if cutoff is not None:
        spy = spy[spy.index <= cutoff]

    records = []
    skipped = []
    ind_keys = sorted(sbi['industry_key'].dropna().unique())

    logger.info(f"Computing monthly RRG for {len(ind_keys)} GICS industries vs SPY...")

    for i, ind_key in enumerate(ind_keys):
        if i % 20 == 0 and i > 0:
            logger.info(f"  {i}/{len(ind_keys)} processed ({len(records)} scored)...")

        members = sbi[sbi['industry_key'] == ind_key]
        tickers = members['symbol'].dropna().tolist()
        weights = {
            row['symbol']: float(row['marketCap'])
            for _, row in members.iterrows()
            if pd.notna(row.get('marketCap')) and row['marketCap'] > 0
        }

        idx_monthly = _build_monthly_index(tickers, weights, monthly_dir, cutoff)
        if idx_monthly is None:
            skipped.append(ind_key)
            continue

        try:
            rs_ratio_s, rs_mom_s = compute_rrg(
                idx_monthly, spy,
                window=10, roc_period=10, smoothing=3,
            )
        except Exception as e:
            logger.debug(f"Monthly RRG failed for {ind_key}: {e}")
            skipped.append(ind_key)
            continue

        if rs_ratio_s.empty or rs_mom_s.empty:
            skipped.append(ind_key)
            continue

        rs_r = float(rs_ratio_s.iloc[-1])
        rs_m = float(rs_mom_s.iloc[-1])

        tail_data = {}
        for w in range(1, tail_months + 1):
            tail_data[f'rs_ratio_m_{w}mo']   = float(rs_ratio_s.iloc[-(w + 1)]) if len(rs_ratio_s) > w else np.nan
            tail_data[f'rs_momentum_m_{w}mo'] = float(rs_mom_s.iloc[-(w + 1)])  if len(rs_mom_s)  > w else np.nan

        if len(rs_ratio_s) >= 5:
            ratio_4m_ago  = float(rs_ratio_s.iloc[-5])
            mom_4m_ago    = float(rs_mom_s.iloc[-5])
            ratio_trend   = 'right' if rs_r > ratio_4m_ago else 'left'
            mom_trend     = 'up'    if rs_m > mom_4m_ago   else 'down'
            tail_dir      = f"{ratio_trend}-{mom_trend}"
        else:
            tail_dir = 'unknown'

        meta          = industries[industries['industry_key'] == ind_key]
        sector_name   = meta['sector_name'].iloc[0]   if len(meta) else ''
        industry_name = meta['industry_name'].iloc[0] if len(meta) else ind_key

        records.append({
            'industry_key':    ind_key,
            'sector_name':     sector_name,
            'industry_name':   industry_name,
            'rs_ratio_m':      round(rs_r, 3),
            'rs_momentum_m':   round(rs_m, 3),
            'quadrant_m':      assign_quadrant(rs_r, rs_m),
            'tail_direction_m': tail_dir,
            **{k: round(v, 3) if not np.isnan(v) else np.nan for k, v in tail_data.items()},
        })

    if skipped:
        logger.info(f"Skipped {len(skipped)} industries: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values('rs_ratio_m', ascending=False).reset_index(drop=True)
    counts = df['quadrant_m'].value_counts()
    logger.info(f"Monthly RRG: {len(df)} industries  quadrants={counts.to_dict()}")
    return df


def save(df: pd.DataFrame, config: Config, as_of: date | None = None) -> Path:
    rrg_dir = config.results_dir / 'rrg'
    rrg_dir.mkdir(parents=True, exist_ok=True)
    label = str(as_of) if as_of else pd.Timestamp.today().strftime('%Y-%m-%d')
    out   = rrg_dir / f"rrg_gics_{label}.csv"
    df.to_csv(out, index=False)
    logger.info(f"Saved RRG → {out}")
    return out


def save_monthly(df: pd.DataFrame, config: Config, as_of: date | None = None) -> Path:
    rrg_dir = config.results_dir / 'rrg'
    rrg_dir.mkdir(parents=True, exist_ok=True)
    label = str(as_of) if as_of else pd.Timestamp.today().strftime('%Y-%m-%d')
    out   = rrg_dir / f"rrg_monthly_gics_{label}.csv"
    df.to_csv(out, index=False)
    logger.info(f"Saved monthly RRG → {out}")
    return out
