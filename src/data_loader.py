"""
Data loader for OHLCV files from downloadData_v1.

Reads individual ticker CSVs and returns clean DataFrames.
Handles both stock files (daily + monthly available) and ^YH industry
index files (daily only — monthly is derived by resampling).

Stitching: historical market_data/daily files are supplemented with
newer rows from market_data_batch/daily/prices_1d_YYYY-MM-DD.csv files.
Batch files are loaded once into a module-level cache (BatchCache) and
appended transparently in load_daily().
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_OHLCV = ['Open', 'High', 'Low', 'Close', 'Volume']

# Batch data directory relative to the historical data root
_BATCH_SUBDIR = 'market_data_batch/daily'


# ---------------------------------------------------------------------------
# Batch supplement cache
# ---------------------------------------------------------------------------

class _BatchCache:
    """
    Loads all prices_1d_*.csv batch files once and indexes them by ticker.
    Thread-unsafe but fine for single-process use.
    """

    def __init__(self):
        self._data: dict[str, pd.DataFrame] = {}   # ticker → DataFrame(OHLCV)
        self._loaded_dir: Path | None = None

    def load(self, batch_dir: Path, after: pd.Timestamp) -> None:
        if self._loaded_dir == batch_dir:
            return
        self._loaded_dir = batch_dir
        self._data.clear()

        if not batch_dir.exists():
            logger.debug(f"Batch dir not found: {batch_dir}")
            return

        cols_needed = {'Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume'}
        files = sorted(batch_dir.glob('prices_1d_*.csv'))
        files_used = [f for f in files if _batch_file_date(f) > after]

        if not files_used:
            logger.info("BatchCache: no batch files newer than historical cutoff")
            return

        frames = []
        for f in files_used:
            try:
                df = pd.read_csv(f)
                missing = cols_needed - set(df.columns)
                if missing:
                    logger.warning(f"Batch file {f.name} missing columns {missing}, skipping")
                    continue
                frames.append(df[list(cols_needed)])
            except Exception as e:
                logger.warning(f"Could not read batch file {f.name}: {e}")

        if not frames:
            return

        combined = pd.concat(frames, ignore_index=True)
        combined['Date'] = pd.to_datetime(combined['Date'].astype(str).str.split(' ').str[0],
                                          errors='coerce')
        combined = combined[combined['Date'].notna() & (combined['Date'] > after)]
        combined['Symbol'] = combined['Symbol'].str.upper()

        for sym, grp in combined.groupby('Symbol'):
            sub = grp.drop(columns='Symbol').set_index('Date').sort_index()
            sub = sub[sub.index.weekday < 5]
            self._data[sym] = sub.astype(float, errors='ignore')

        logger.info(
            f"BatchCache: {len(files_used)} file(s) loaded, "
            f"{len(self._data)} symbols beyond {after.date()}"
        )

    def get(self, ticker: str) -> Optional[pd.DataFrame]:
        sym = ticker.upper()
        for key in (sym, sym.replace('.', '-'), sym.replace('-', '.')):
            result = self._data.get(key)
            if result is not None:
                return result
        return None


def _batch_file_date(path: Path) -> pd.Timestamp:
    """Extract date from prices_1d_YYYY-MM-DD.csv filename."""
    try:
        return pd.Timestamp(path.stem.split('_')[-1])
    except Exception:
        return pd.Timestamp.min


_BATCH_CACHE = _BatchCache()


# ---------------------------------------------------------------------------
# Internal CSV reader
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    """Read a single OHLCV CSV into a clean, tz-naive, date-indexed DataFrame."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col='Date', parse_dates=False)
        df.index = pd.to_datetime(df.index.str.split(' ').str[0], errors='coerce')
        df = df[df.index.notna()]
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.sort_index()
        df = df[df.index.weekday < 5]
        cols = [c for c in _OHLCV if c in df.columns]
        return df[cols] if cols else df
    except Exception as e:
        logger.warning(f"Could not read {path.name}: {e}")
        return None


def _ensure_batch_cache(daily_dir: Path) -> None:
    """
    Bootstrap the batch cache on first call.
    Finds the historical cutoff from SPY/AAPL, then loads all newer batch files.
    """
    batch_dir = daily_dir.parent.parent / _BATCH_SUBDIR
    if _BATCH_CACHE._loaded_dir == batch_dir:
        return

    # Determine historical cutoff from a reference ticker
    cutoff = pd.Timestamp('2000-01-01')
    for ref in ['SPY', 'AAPL', 'MSFT', 'QQQ']:
        df = _read_csv(daily_dir / f'{ref}.csv')
        if df is not None and not df.empty:
            cutoff = df.index.max()
            break

    _BATCH_CACHE.load(batch_dir, after=cutoff)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_daily(ticker: str, daily_dir: Path) -> Optional[pd.DataFrame]:
    """Load daily OHLCV, stitching in any newer batch rows automatically."""
    _ensure_batch_cache(daily_dir)

    df = _read_csv(daily_dir / f"{ticker}.csv")

    # Skip batch supplement for index symbols (^YH...) — batch files don't contain them
    if not isinstance(ticker, str) or ticker.startswith('^'):
        return df

    batch = _BATCH_CACHE.get(ticker)
    if batch is not None and df is not None and not df.empty:
        new_rows = batch[batch.index > df.index[-1]]
        if not new_rows.empty:
            cols = [c for c in _OHLCV if c in new_rows.columns]
            df = pd.concat([df, new_rows[cols]])

    return df


def load_monthly(ticker: str, monthly_dir: Path, daily_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load monthly OHLCV. Uses the dedicated monthly file if it exists.
    Falls back to resampling the daily file (needed for ^YH... index symbols
    which are only downloaded at daily frequency).
    """
    monthly_path = monthly_dir / f"{ticker}.csv"
    if monthly_path.exists():
        # For monthly files, also stitch batch data via daily resample
        monthly = _read_csv(monthly_path)
        # Supplement: re-resample from stitched daily if ticker has batch rows
        if not ticker.startswith('^'):
            daily = load_daily(ticker, daily_dir)
            if daily is not None and monthly is not None and not monthly.empty:
                last_monthly = monthly.index[-1]
                new_daily = daily[daily.index > last_monthly]
                if not new_daily.empty:
                    agg = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
                    if 'Volume' in new_daily.columns:
                        agg['Volume'] = 'sum'
                    extra = new_daily.resample('ME').agg(
                        {k: v for k, v in agg.items() if k in new_daily.columns}
                    ).dropna(subset=['Close'])
                    if not extra.empty:
                        monthly = pd.concat([monthly, extra])
        return monthly

    # Resample daily → month-end (includes stitched batch data)
    daily = load_daily(ticker, daily_dir)
    if daily is None or daily.empty:
        return None

    agg = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
    if 'Volume' in daily.columns:
        agg['Volume'] = 'sum'

    monthly = daily.resample('ME').agg({k: v for k, v in agg.items() if k in daily.columns})
    monthly = monthly.dropna(subset=['Close'])
    return monthly if not monthly.empty else None


def get_market_cap(ticker: str, daily_dir: Path) -> Optional[float]:
    """
    Read the most recent marketCap value from the daily CSV.
    Returns None if the column is missing or the file doesn't exist.
    """
    path = daily_dir / f"{ticker}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=['marketCap'])
        vals = df['marketCap'].dropna()
        return float(vals.iloc[-1]) if not vals.empty else None
    except Exception:
        return None


def list_tickers(directory: Path, prefix: str = '') -> list[str]:
    """List all ticker symbols available in a directory, optionally filtered by prefix."""
    tickers = [f.stem for f in directory.glob('*.csv') if f.stem.startswith(prefix)]
    return sorted(tickers)
