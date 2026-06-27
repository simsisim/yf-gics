"""
Project configuration — paths and SCTR constants.
All paths pointing to downloadData_v1 are read-only.
"""

from pathlib import Path
from dataclasses import dataclass

# Absolute root of this project
PROJECT_DIR = Path(__file__).resolve().parent

# Root of the downloaded OHLCV data (read-only)
_DOWNLOAD_ROOT = Path('/home/imagda/_invest2024/python/downloadData_v1/data/market_data')

# Ticker metadata from metaData_v1 (read-only)
_TICKER_INFO = Path('/home/imagda/_invest2024/python/metaData_v1/data/tickers/combined_info_tickers_clean_0.csv')


@dataclass
class Config:
    # --- Source data directories (read-only) ---
    daily_dir:   Path = _DOWNLOAD_ROOT / 'daily'
    weekly_dir:  Path = _DOWNLOAD_ROOT / 'weekly'
    monthly_dir: Path = _DOWNLOAD_ROOT / 'monthly'

    # --- Reference files (this project) ---
    industries_csv:         Path = PROJECT_DIR / 'industries.csv'
    stocks_by_industry_csv: Path = PROJECT_DIR / 'stocks_by_industry.csv'
    ticker_info_csv:        Path = _TICKER_INFO

    # --- Output directories ---
    results_dir:       Path = PROJECT_DIR / 'results'
    industry_sctr_dir: Path = PROJECT_DIR / 'results' / 'industry_sctr'
    stock_sctr_dir:    Path = PROJECT_DIR / 'results' / 'stock_sctr'
    breadth_dir:       Path = PROJECT_DIR / 'results' / 'breadth'

    # --- StockCharts universe hard limits (market cap) ---
    large_cap_min:  int = 10_000_000_000   # > $10B
    mid_cap_min:    int = 2_000_000_000    # $2B – $10B
    small_cap_min:  int = 250_000_000      # $250M – $2B
    # below small_cap_min → excluded (no SCTR, same as StockCharts)

    # --- Universe source for stock SCTR ---
    # 'sc' = StockCharts DB universe table (group labels from StockCharts, ~5,200 stocks)
    # 'tv' = TradingView universe CSV (group labels from market-cap hard limits, ~3,300 stocks)
    universe_source: str = 'sc'

    # --- SCTR formula minimum data requirements ---
    min_daily_bars:   int = 210   # 200-day SMA needs 200+ bars
    min_monthly_bars: int = 15    # 14-month ROC needs 15 monthly closes

    def setup_dirs(self) -> None:
        for d in [self.results_dir, self.industry_sctr_dir, self.stock_sctr_dir, self.breadth_dir]:
            d.mkdir(parents=True, exist_ok=True)
