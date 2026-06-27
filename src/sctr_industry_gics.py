"""
Industry SCTR — Option B (GICS, synthetic index from constituent stocks).

For each GICS industry:
  1. Identify constituent stocks from stocks_by_industry.csv
  2. Load their daily close prices from downloadData_v1
  3. Build a market-cap-weighted daily price index (chained returns)
  4. Resample to month-end for ROC components
  5. Run the true SCTR formula on the synthetic index
  6. Rank all 145 GICS industries against each other

No mapping to StockCharts or Dow Jones indexes — purely GICS-based.
"""

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from src.data_loader import load_daily
from src.sctr_engine import compute_sctr_raw, rank_to_sctr

logger = logging.getLogger(__name__)

MIN_CONSTITUENTS = 3   # minimum stocks with data to build a valid index


def _build_synthetic_index(tickers: list[str], weights: dict[str, float],
                            daily_dir: Path) -> pd.DataFrame | None:
    """
    Build a market-cap-weighted daily price series from constituent stocks.

    Returns a DataFrame with columns [Open, High, Low, Close, Volume] on a
    daily DatetimeIndex, or None if fewer than MIN_CONSTITUENTS have data.

    The index is constructed as a chained return series:
        index_t = index_{t-1} * weighted_average(close_t / close_{t-1})

    Starting level is 1000.0. Absolute level doesn't matter for SCTR
    (all components use ratios / ROC), but a round start aids debugging.
    """
    closes = {}
    for ticker in tickers:
        daily = load_daily(ticker, daily_dir)
        if daily is None or daily.empty or 'Close' not in daily.columns:
            continue
        closes[ticker] = daily['Close'].dropna()

    if len(closes) < MIN_CONSTITUENTS:
        return None

    # Align on common date index (union), forward-fill gaps ≤ 5 days
    price_df = pd.DataFrame(closes)
    price_df = price_df.sort_index().ffill(limit=5)

    # Recompute weights using only constituents that loaded
    available = list(closes.keys())
    total_cap = sum(weights.get(t, 0) for t in available)
    if total_cap == 0:
        # Equal-weight fallback if no market cap data
        w = pd.Series({t: 1.0 / len(available) for t in available})
    else:
        w = pd.Series({t: weights.get(t, 0) / total_cap for t in available})

    # Daily returns per stock, weighted sum → index return
    returns = price_df[available].pct_change(fill_method=None)
    index_returns = returns.mul(w, axis=1).sum(axis=1)

    # Chain returns into a price series starting at 1000
    index_close = (1 + index_returns).cumprod() * 1000.0
    index_close.iloc[0] = 1000.0  # anchor first value

    # Wrap as OHLCV DataFrame (High = Low = Open = Close for index)
    idx_df = pd.DataFrame({
        'Open':   index_close,
        'High':   index_close,
        'Low':    index_close,
        'Close':  index_close,
        'Volume': 0.0,
    })
    idx_df = idx_df.dropna(subset=['Close'])
    return idx_df if len(idx_df) >= 210 else None


def run(config: Config, as_of: date | None = None) -> pd.DataFrame:
    """
    Compute SCTR for all 145 GICS industries using synthetic constituent indexes.

    Args:
        config:  Project Config instance.
        as_of:   Truncate data to this date for backtesting. None = latest.

    Returns:
        DataFrame sorted by sctr descending with columns:
            industry_key, sector_key, sector_name, industry_name,
            constituent_count, c1..c6, raw_score, sctr, rank
    """
    sbi = pd.read_csv(config.stocks_by_industry_csv)
    industries = pd.read_csv(config.industries_csv)

    cutoff = pd.Timestamp(as_of) if as_of else None
    records = []
    skipped = []

    industry_keys = sbi['industry_key'].unique()
    logger.info(f"Building synthetic indexes for {len(industry_keys)} GICS industries...")

    for i, ind_key in enumerate(sorted(industry_keys)):
        if i % 20 == 0 and i > 0:
            logger.info(f"  {i}/{len(industry_keys)} industries processed ({len(records)} scored)...")

        members = sbi[sbi['industry_key'] == ind_key]
        tickers = members['symbol'].dropna().tolist()

        # Build market-cap weights from stocks_by_industry.csv snapshot
        weights = {}
        for _, row in members.iterrows():
            mc = row.get('marketCap')
            if pd.notna(mc) and mc > 0:
                weights[row['symbol']] = float(mc)

        # Build synthetic daily index
        synthetic = _build_synthetic_index(tickers, weights, config.daily_dir)
        if synthetic is None:
            skipped.append(ind_key)
            logger.debug(f"Skipping {ind_key}: insufficient constituent data")
            continue

        if cutoff is not None:
            synthetic = synthetic[synthetic.index <= cutoff]
            if synthetic.empty:
                skipped.append(ind_key)
                continue

        # Monthly series: resample synthetic daily to month-end
        monthly = synthetic[['Close']].resample('ME').last().rename(columns={'Close': 'Close'})
        monthly['Open'] = monthly['Close']
        monthly['High'] = monthly['Close']
        monthly['Low'] = monthly['Close']
        monthly['Volume'] = 0.0
        monthly = monthly.dropna(subset=['Close'])

        result = compute_sctr_raw(synthetic, monthly)
        if result is None:
            skipped.append(ind_key)
            continue

        # Fetch GICS labels
        meta = industries[industries['industry_key'] == ind_key]
        sector_key   = meta['sector_key'].iloc[0]   if len(meta) else ''
        sector_name  = meta['sector_name'].iloc[0]  if len(meta) else ''
        industry_name = meta['industry_name'].iloc[0] if len(meta) else ind_key

        # Count how many constituents actually had data
        available_count = sum(
            1 for t in tickers if (config.daily_dir / f'{t}.csv').exists()
        )

        records.append({
            'industry_key':      ind_key,
            'sector_key':        sector_key,
            'sector_name':       sector_name,
            'industry_name':     industry_name,
            'constituent_count': available_count,
            **result,
        })

    if skipped:
        logger.info(f"Skipped {len(skipped)} industries (insufficient data): {skipped[:10]}{'...' if len(skipped)>10 else ''}")

    if not records:
        logger.error("No GICS industry SCTR computed — check data availability")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    raw = df.set_index('industry_key')['raw_score']
    df['sctr'] = rank_to_sctr(raw).values
    df['rank'] = df['sctr'].rank(ascending=False, method='min').astype(int)
    df = df.sort_values('sctr', ascending=False).reset_index(drop=True)

    logger.info(f"GICS industry SCTR computed for {len(df)} industries")
    return df


def save(df: pd.DataFrame, config: Config, as_of: date | None = None) -> Path:
    config.setup_dirs()
    label = str(as_of) if as_of else pd.Timestamp.today().strftime('%Y-%m-%d')
    out = config.industry_sctr_dir / f"industry_sctr_gics_{label}.csv"
    df.to_csv(out, index=False)
    logger.info(f"Saved GICS industry SCTR → {out}")
    return out
