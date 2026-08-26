"""
Daily History Backfill — rebuilds the last N months of SCTR / Stage-Minervini /
RS / performance history for the 145 GICS industries + 11 sector SPDR ETFs.

Safe to re-run: full overwrite of the trailing window each time (not
incremental) -- see history_tracker.py for the cheap go-forward append that
runs daily without repeating this work. Both share compute_history() below so
the metadata/assembly logic exists in exactly one place.

Output: results/history/industry_sector_metrics_daily.csv (long format) plus
sctr_wide.csv / minervini_count_wide.csv (pivots for quick eyeballing).
"""

import logging
from pathlib import Path

import pandas as pd

from config import Config
from src.history_metrics import (
    build_price_series, sctr_wide_matrix, rs_composite_wide, flat_returns_wide,
    assemble_rows,
)
from src.perf_screener import SECTOR_ETF_MAP

logger = logging.getLogger(__name__)

DEFAULT_MONTHS_BACK = 6


def compute_history(config: Config, months_back: int | None = DEFAULT_MONTHS_BACK) -> pd.DataFrame:
    """
    Shared core for both the full backfill and the go-forward tracker.

    months_back=6 (or any int): trading_dates = the trailing N months of SPY's
    calendar (the backfill case).
    months_back=None: trading_dates = just the single latest available date
    (the cheap go-forward case -- history_tracker.py calls this).
    """
    price_series, industry_keys, sector_keys = build_price_series(config)

    spy = price_series.get('SPY')
    if spy is None or spy.empty:
        logger.error("No SPY price history -- cannot resolve a trading calendar")
        return pd.DataFrame()

    end_date = spy.index.max()
    if months_back is None:
        trading_dates = pd.DatetimeIndex([end_date])
        logger.info(f"Computing single date: {end_date.date()}")
    else:
        start_date = end_date - pd.DateOffset(months=months_back)
        trading_dates = spy.index[(spy.index >= start_date) & (spy.index <= end_date)]
        logger.info(f"Backfill window: {trading_dates[0].date()} -> {trading_dates[-1].date()} "
                    f"({len(trading_dates)} trading days)")

    all_keys = industry_keys + sector_keys

    logger.info("Computing SCTR (ranked within industries, and separately within sectors)...")
    sctr_ind = sctr_wide_matrix(price_series, industry_keys, trading_dates)
    sctr_sec = sctr_wide_matrix(price_series, sector_keys, trading_dates)
    sctr_all = pd.concat([sctr_ind, sctr_sec], axis=1)

    logger.info("Computing RS percentile composite (ranked within industries, and within sectors)...")
    rs_ind = rs_composite_wide(price_series, industry_keys, 'SPY', trading_dates)
    rs_sec = rs_composite_wide(price_series, sector_keys, 'SPY', trading_dates)
    rs_all = pd.concat([rs_ind, rs_sec], axis=1)

    logger.info("Computing flat 1w/2w/1m/3m/6m returns + relative return vs SPY & QQQ...")
    returns_wide = flat_returns_wide(price_series, all_keys + ['SPY', 'QQQ'], trading_dates)

    logger.info(f"Computing Stage/Minervini per (date, symbol) -- "
                f"{len(trading_dates)} dates x {len(all_keys)} symbols...")
    rows = assemble_rows(price_series, all_keys, sctr_all, rs_all, returns_wide, trading_dates)

    if not rows:
        logger.error("No rows assembled -- check data availability")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # --- attach metadata (sector/industry names) ---
    industries = pd.read_csv(config.industries_csv)
    ind_meta = industries.set_index('industry_key')[['sector_key', 'sector_name', 'industry_name']]
    sector_names = industries[['sector_key', 'sector_name']].drop_duplicates().set_index('sector_key')['sector_name']

    def _meta(key: str) -> dict:
        if key in ind_meta.index:
            m = ind_meta.loc[key]
            return {'level': 'industry', 'sector_key': m['sector_key'], 'sector_name': m['sector_name'],
                    'industry_key': key, 'industry_name': m['industry_name'], 'symbol': key}
        if key in SECTOR_ETF_MAP:
            sk = SECTOR_ETF_MAP[key]
            return {'level': 'sector', 'sector_key': sk, 'sector_name': sector_names.get(sk, ''),
                    'industry_key': '', 'industry_name': '', 'symbol': key}
        return {'level': '', 'sector_key': '', 'sector_name': '', 'industry_key': '', 'industry_name': '', 'symbol': key}

    meta_df = pd.DataFrame([_meta(k) for k in df['key']])
    df = pd.concat([meta_df.reset_index(drop=True), df.drop(columns=['key']).reset_index(drop=True)], axis=1)

    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df.sort_values(['date', 'level', 'symbol']).reset_index(drop=True)

    n_dates = df['date'].nunique()
    n_symbols = df['symbol'].nunique()
    logger.info(f"History assembled: {len(df)} rows ({n_symbols} symbols x {n_dates} dates)")
    return df


def run(config: Config, months_back: int = DEFAULT_MONTHS_BACK) -> pd.DataFrame:
    return compute_history(config, months_back=months_back)


def save(df: pd.DataFrame, config: Config) -> dict[str, Path]:
    out_dir = config.results_dir / 'history'
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    p = out_dir / 'industry_sector_metrics_daily.csv'
    df.to_csv(p, index=False)
    paths['long'] = p
    logger.info(f"Saved long-format daily history -> {p}  ({len(df)} rows)")

    for col, fname in [('sctr', 'sctr_wide.csv'), ('minervini_count', 'minervini_count_wide.csv')]:
        try:
            label_col = df['industry_name'].where(df['level'] == 'industry', df['symbol'])
            wide = df.assign(_label=label_col).pivot(index='date', columns='_label', values=col)
            p2 = out_dir / fname
            wide.to_csv(p2)
            paths[col] = p2
            logger.info(f"Saved {col} wide pivot -> {p2}")
        except Exception as e:
            logger.warning(f"Could not save {col} wide pivot: {e}")

    return paths
