"""
Data loader for OHLCV files from downloadData_v1.

Reads individual ticker CSVs and returns clean DataFrames.
Handles both stock files (daily + monthly available) and ^YH industry
index files (daily only — monthly is derived by resampling).

Storage: downloadData_v1 splits each ticker's history into
archive/{ticker}.csv (frozen, through last year-end) and current/{ticker}.csv
(this year only). It also maintains a flat {ticker}.csv as a locally
materialized archive+current cache, but that cache only refreshes when
downloadData_v1's own pipeline runs for that ticker on this machine — if
archive/+current/ were synced from elsewhere without a local run afterward,
the flat file can be stale or entirely missing (this has happened: after the
archive/current migration, 0 of ~6,300 flat files existed here). Every reader
below reads archive/+current/ directly and only falls back to the flat file
for a pre-migration checkout that doesn't have the tier subfolders at all —
same approach as marketHealth/metaData_v1's data_reader.py.

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


def _read_ticker_frame(directory: Path, ticker: str) -> Optional[pd.DataFrame]:
    """
    Read archive/+current/ for one ticker directly (current wins on any
    overlapping date), falling back to the flat legacy cache only if neither
    tier subfolder has this ticker at all.
    """
    frames = [
        f for f in (
            _read_csv(directory / "archive" / f"{ticker}.csv"),
            _read_csv(directory / "current" / f"{ticker}.csv"),
        ) if f is not None and not f.empty
    ]
    if frames:
        combined = pd.concat(frames) if len(frames) > 1 else frames[0]
        return combined[~combined.index.duplicated(keep='last')].sort_index()
    return _read_csv(directory / f"{ticker}.csv")


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
        df = _read_ticker_frame(daily_dir, ref)
        if df is not None and not df.empty:
            cutoff = df.index.max()
            break

    _BATCH_CACHE.load(batch_dir, after=cutoff)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_daily(ticker: str, daily_dir: Path) -> Optional[pd.DataFrame]:
    """Load daily OHLCV, stitching in any newer batch rows automatically.

    Index symbols (^YH...) used to skip this entirely on the assumption
    that downloadData_v1's batch pipeline didn't fetch them - that assumption
    is now stale: downloadData_v1's ticker universe (combined_tickers_0-5.csv)
    includes them and its batch pipeline fetches them fine via yf.download().
    What actually had no ^YH coverage was the SLOW pipeline, because its
    ticker_choice was set to a universe (0-8) that excluded all ^-prefixed
    tickers - since fixed at the source (user_input/user_data.csv). Until the
    slow pipeline backfills real history for these under the corrected
    universe, batch is the ONLY source of data for them - handled below by
    falling back to batch alone when there's no slow-pipeline file at all,
    rather than returning None."""
    _ensure_batch_cache(daily_dir)

    df = _read_ticker_frame(daily_dir, ticker)
    batch = _BATCH_CACHE.get(ticker) if isinstance(ticker, str) else None

    if batch is None:
        return df
    if df is None or df.empty:
        return batch  # no slow-pipeline history at all (e.g. ^YH not yet backfilled) - batch alone is what we've got

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
    monthly = _read_ticker_frame(monthly_dir, ticker)
    if monthly is not None and not monthly.empty:
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
    Read the most recent marketCap value, checking current/ (freshest) then
    archive/, then the flat legacy cache as a last resort.
    Returns None if the column is missing everywhere or no file exists.
    """
    for path in (daily_dir / "current" / f"{ticker}.csv",
                 daily_dir / "archive" / f"{ticker}.csv",
                 daily_dir / f"{ticker}.csv"):
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, usecols=['marketCap'])
            vals = df['marketCap'].dropna()
            if not vals.empty:
                return float(vals.iloc[-1])
        except Exception:
            continue
    return None


def _latest_file(directory: Path, pattern: str) -> Optional[Path]:
    """Return the most recent file matching `pattern` in `directory` (sorted by
    filename, so patterns should embed a sortable YYYY-MM-DD date), or None if
    no file matches."""
    files = sorted(directory.glob(pattern))
    return files[-1] if files else None


def list_tickers(directory: Path, prefix: str = '') -> list[str]:
    """
    List all ticker symbols available in a directory, optionally filtered by
    prefix. Checks archive/+current/ tiers as well as the flat directory, so
    this doesn't silently under-count if the flat legacy cache is stale.
    """
    tickers = {f.stem for f in directory.glob('*.csv') if f.stem.startswith(prefix)}
    for tier in ("archive", "current"):
        d = directory / tier
        if d.exists():
            tickers.update(f.stem for f in d.glob('*.csv') if f.stem.startswith(prefix))
    return sorted(tickers)
