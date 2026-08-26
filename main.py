"""
SCTR Analysis — entry point.

Usage:
    python main.py --mode industry          # Layer 1: industry SCTR from ^YH indexes
    python main.py --mode stocks            # Layer 2: stock SCTR (large/mid/small)
    python main.py --mode breadth           # Layer 3: industry breadth (needs stocks first)
    python main.py --mode all               # all three in sequence

    python main.py --mode industry --date 2025-06-01   # backtest as-of a date
    python main.py --mode stocks   --date 2025-06-01

    python main.py --mode validate          # compare industry SCTR vs scraped StockCharts
    python main.py --mode perf              # 1w/2w/1m/3m/6m return + RS + relret vs SPY & QQQ
                                             #   (industries, sector SPDR ETFs, SPY/QQQ reference rows)
    python main.py --mode history-backfill  # rebuild trailing 6mo daily SCTR/Stage/RS/perf history
    python main.py --mode history-backfill --months-back 12
    python main.py --mode history-append    # cheap: append just today's row (chain into cron)
"""

import argparse
import logging
import sys
from datetime import date

import pandas as pd
from pathlib import Path

# Allow running from the project root directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
import src.sctr_industry      as sctr_industry
import src.sctr_stocks        as sctr_stocks
import src.sctr_breadth       as sctr_breadth
import src.correction_filter   as correction_filter
import src.rs_ma_signals         as rs_ma_signals
import src.market_clock          as market_clock
import src.rs_percentile         as rs_percentile
import src.stage_analysis        as stage_analysis
import src.closing_range         as closing_range
import src.breadth               as breadth
import src.signal_delta          as signal_delta
import src.stock_screener        as stock_screener
import src.perf_screener         as perf_screener
import src.history_backfill      as history_backfill
import src.history_tracker       as history_tracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-7s  %(name)s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def parse_date(s: str | None) -> date | None:
    if s is None:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        logger.error(f"Invalid date '{s}' — expected YYYY-MM-DD")
        sys.exit(1)


def run_industry(cfg: Config, as_of: date | None) -> None:
    logger.info("=== Layer 1: Industry SCTR ===")
    df = sctr_industry.run(cfg, as_of=as_of)
    if df.empty:
        logger.error("No industry SCTR results — check ^YH files in daily_dir")
        return
    path = sctr_industry.save(df, cfg, as_of=as_of)

    print(f"\nTop 15 industries by SCTR  (as of {as_of or 'latest'}):")
    print(df[['rank', 'industry_name', 'gics_sector', 'sctr', 'raw_score']].head(15).to_string(index=False))
    print(f"\nSaved → {path}")


def run_stocks(cfg: Config, as_of: date | None) -> None:
    logger.info("=== Layer 2: Stock SCTR ===")
    df = sctr_stocks.run(cfg, as_of=as_of)
    if df.empty:
        logger.error("No stock SCTR results — check daily/monthly data availability")
        return
    paths = sctr_stocks.save(df, cfg, as_of=as_of)

    cols = ['ticker', 'cap_bucket', 'sctr', 'rank_in_bucket', 'raw_score']
    for bucket in ['large', 'mid', 'small']:
        sub = df[df['cap_bucket'] == bucket]
        print(f"\n--- {bucket.upper()} cap  ({len(sub)} stocks) — Top 15 ---")
        print(sub[cols].head(15).to_string(index=False))

    print("\nBucket summary:")
    summary = (
        df.groupby('cap_bucket')
        .agg(count=('ticker', 'count'), median_sctr=('sctr', 'median'), top_sctr=('sctr', 'max'))
        .reset_index()
    )
    print(summary.to_string(index=False))
    print("\nSaved:")
    for p in paths:
        print(f"  {p}")


def run_breadth(cfg: Config, as_of: date | None) -> None:
    logger.info("=== Layer 3: Industry Breadth ===")

    # Try to load the most recent stock SCTR result
    label = str(as_of) if as_of else None
    if label:
        stock_path = cfg.stock_sctr_dir / f"stock_sctr_large_{label}.csv"
    else:
        files = sorted(cfg.stock_sctr_dir.glob("stock_sctr_large_*.csv"))
        stock_path = files[-1] if files else None

    if not stock_path or not stock_path.exists():
        logger.error(f"Stock SCTR file not found. Run --mode stocks first.")
        return

    import pandas as pd
    label_str = str(as_of) if as_of else stock_path.stem.replace('stock_sctr_large_', '')
    parts = []
    for bucket in ['large', 'mid', 'small']:
        p = cfg.stock_sctr_dir / f"stock_sctr_{bucket}_{label_str}.csv"
        if p.exists():
            parts.append(pd.read_csv(p))
    if not parts:
        logger.error("No stock SCTR files found. Run --mode stocks first.")
        return
    stock_df = pd.concat(parts, ignore_index=True)
    logger.info(f"Loaded stock SCTR: {len(stock_df)} stocks ({label_str})")

    df = sctr_breadth.run(stock_df, cfg)
    if df.empty:
        return
    path = sctr_breadth.save(df, cfg, as_of=as_of)

    print(f"\nTop 20 GICS industries by breadth  (as of {as_of or 'latest'}):")
    cols = ['industry_name', 'sector_name', 'count', 'median_sctr', 'pct_above_75', 'pct_below_25']
    print(df[cols].head(20).to_string(index=False))
    print(f"\nSaved → {path}")


def run_validate(cfg: Config) -> None:
    """
    Compare computed industry SCTR against the scraped StockCharts data.
    Prints Spearman rank correlation.
    """
    import pandas as pd
    from scipy import stats

    scraped_dir = Path('/home/imagda/_invest2024/python/stockCharts/output/industry_summary')
    scraped_files = sorted(scraped_dir.glob('*_industry_summary.csv'))
    if not scraped_files:
        logger.error(f"No scraped industry summary files found in {scraped_dir}")
        return
    scraped_file = scraped_files[-1]
    logger.info(f"Using scraped data: {scraped_file.name}")
    scraped = pd.read_csv(scraped_file)[['symbol', 'sctr']].rename(columns={'sctr': 'sctr_scraped'})

    computed_files = sorted(f for f in cfg.industry_sctr_dir.glob('industry_sctr_*.csv')
                            if 'gics' not in f.name)
    if not computed_files:
        logger.error("No computed industry SCTR files found. Run --mode industry first.")
        return
    computed_file = computed_files[-1]
    logger.info(f"Using computed data: {computed_file.name}")
    computed = pd.read_csv(computed_file)[['symbol', 'sctr']].rename(columns={'sctr': 'sctr_computed'})

    # Scraped uses $DJUS... (StockCharts/Dow Jones) symbols; computed uses ^YH... (Yahoo).
    # Different index families — match on industry name (partial overlap expected, ~10–30 names).
    scraped['name_lc']  = scraped['sctr_scraped'].index if False else \
                          pd.read_csv(scraped_file)['name'].str.lower().str.strip()
    scraped = pd.read_csv(scraped_file)[['name', 'sctr']].rename(columns={'sctr': 'sctr_scraped'})
    scraped['name_lc'] = scraped['name'].str.lower().str.strip()

    computed = pd.read_csv(computed_file)[['industry_name', 'sctr']].rename(columns={'sctr': 'sctr_computed'})
    computed['name_lc'] = computed['industry_name'].str.lower().str.strip()

    merged = scraped.merge(computed, on='name_lc', how='inner')

    if merged.empty:
        logger.warning("No name matches between scraped and computed — indexes use different naming")
        return

    corr, pval = stats.spearmanr(merged['sctr_scraped'], merged['sctr_computed'])
    print(f"\nValidation: {len(merged)} industries matched (DJ vs YF indexes — rough comparison)")
    print(f"Spearman rank correlation: {corr:.3f}  (p={pval:.4f})")
    print("\nSample comparison (top 10 by scraped SCTR):")
    top = merged.sort_values('sctr_scraped', ascending=False).head(10)
    print(top[['name', 'sctr_scraped', 'sctr_computed']].to_string(index=False))


def _run_steps(steps: list, label: str, title: str) -> list[tuple[str, str]]:
    """Run a list of (name, fn) steps, log each, return results."""
    results: list[tuple[str, str]] = []
    for name, fn in steps:
        logger.info(f"── {title}: {name} ──")
        try:
            out = fn()
            if out is None:
                results.append((name, 'SKIP (empty result)'))
            else:
                paths = out if isinstance(out, (list, tuple)) else [out]
                results.append((name, f"OK  {', '.join(str(p) for p in paths if p)}"))
        except Exception as exc:
            logger.error(f"{name} failed: {exc}", exc_info=True)
            results.append((name, f"FAIL: {exc}"))
    return results


def _print_summary(results: list[tuple[str, str]], label: str, title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title} — {label}")
    print(f"{'─'*60}")
    for name, status in results:
        icon = '✓' if status.startswith('OK') else ('⚠' if status.startswith('SKIP') else '✗')
        print(f"  {icon}  {name:<18} {status[:60]}")
    print(f"{'─'*60}")


def run_industry_pipeline(cfg: Config, as_of: date | None, source: str = 'yh') -> None:
    """All industry-level steps in dependency order."""
    label = str(as_of) if as_of else pd.Timestamp.today().strftime('%Y-%m-%d')
    steps = [
        ('sctr',          lambda: sctr_industry.save(sctr_industry.run(cfg, as_of=as_of), cfg, as_of=as_of)),
        ('rs-ma',         lambda: rs_ma_signals.save(rs_ma_signals.run(cfg, as_of=as_of), cfg, as_of=as_of)),
        ('market-clock',  lambda: market_clock.save(market_clock.run(cfg, as_of=as_of)[0], cfg, as_of=as_of)),
        ('rs-percentile', lambda: rs_percentile.save(rs_percentile.run(cfg, as_of=as_of, source=source), cfg, as_of=as_of, source=source)),
        ('stage',         lambda: stage_analysis.save(stage_analysis.run(cfg, as_of=as_of, source=source), cfg, as_of=as_of, source=source)),
        ('market-breadth',lambda: breadth.save(*breadth.run(cfg, as_of=as_of), cfg, as_of=as_of)),
        ('signal-delta',  lambda: signal_delta.save(signal_delta.run(cfg, as_of=as_of), cfg, as_of=as_of)),
    ]
    results = _run_steps(steps, label, 'industry')
    _print_summary(results, label, 'Industry pipeline')


def run_stocks_pipeline(cfg: Config, as_of: date | None) -> None:
    """All stock-level steps in dependency order."""
    label = str(as_of) if as_of else pd.Timestamp.today().strftime('%Y-%m-%d')
    steps = [
        ('stock-sctr',     lambda: sctr_stocks.save(sctr_stocks.run(cfg, as_of=as_of), cfg, as_of=as_of)),
        ('stock-screener', lambda: stock_screener.save(stock_screener.run(cfg, as_of=as_of), cfg, as_of=as_of)),
    ]
    results = _run_steps(steps, label, 'stocks')
    _print_summary(results, label, 'Stocks pipeline')


def _history_append_step(cfg: Config) -> list[Path] | None:
    df = history_tracker.run(cfg)
    if df.empty:
        return None
    return list(history_tracker.save(df, cfg).values())


def run_update(cfg: Config, as_of: date | None, source: str = 'yh') -> None:
    """Full pipeline — industry, then stocks, then append today's row to the daily history."""
    run_industry_pipeline(cfg, as_of, source=source)
    run_stocks_pipeline(cfg, as_of)
    label = str(as_of) if as_of else pd.Timestamp.today().strftime('%Y-%m-%d')
    results = _run_steps([('history-append', lambda: _history_append_step(cfg))], label, 'history')
    _print_summary(results, label, 'History tracker')


def main() -> None:
    parser = argparse.ArgumentParser(description='SCTR Analysis System')
    parser.add_argument('--mode',
                        choices=[
                            # ── composite modes ──────────────────────────────
                            'update',          # industry + stocks (everything)
                            'industry',        # all industry steps
                            'stocks',          # all stock steps
                            # ── industry individual steps ────────────────────
                            'sctr',            # industry SCTR only (^YH)
                            'rs-ma',           # RS line vs 13w MA signals
                            'market-clock',    # Follow-Through Day / distribution days
                            'rs-percentile',   # RS composite percentile
                            'perf',            # flat 1w/2w/1m/3m/6m return + RS + relret vs SPY & QQQ
                            'history-backfill',# rebuild trailing 6mo daily SCTR/Stage/RS/perf history
                            'history-append',  # cheap go-forward: append today's row to that history
                            'stage',           # Weinstein/Minervini stage
                            'market-breadth',  # market health breadth score
                            'signal-delta',    # what changed vs previous run
                            # ── stock individual steps ───────────────────────
                            'stock-sctr',      # stock SCTR (large/mid/small)
                            'stock-screener',  # ranked stocks within top industries
                            'closing-range',   # IBD correction leader filter
                            # ── research / legacy ────────────────────────────
                            'correction',      # legacy correction filter
                            'validate',        # compare vs StockCharts scraped data
                        ],
                        default='industry',
                        help='Which analysis to run')
    parser.add_argument('--source', choices=['yh', 'synth'], default='yh',
                        help='Index source: yh=^YH published indexes (default). synth is no longer '
                             'supported (see stage_analysis.py / rs_percentile.py docstrings) and is '
                             'accepted only for compatibility.')
    parser.add_argument('--months-back', type=int, default=history_backfill.DEFAULT_MONTHS_BACK,
                        help=f'history-backfill: trailing months to rebuild (default: {history_backfill.DEFAULT_MONTHS_BACK})')
    parser.add_argument('--correction-start', default=None,
                        help='Manual correction start date YYYY-MM-DD (closing-range: auto-detected if omitted)')
    parser.add_argument('--correction-end', default=None,
                        help='Manual correction end date YYYY-MM-DD (closing-range: FTD date if omitted)')
    parser.add_argument('--min-score', type=int, default=3,
                        help='Minimum leader score 0-5 for closing-range output (default 3)')
    parser.add_argument('--prev-date', default=None,
                        help='signal-delta: explicit previous run date YYYY-MM-DD (auto-detected if omitted)')
    parser.add_argument('--top-industries-only', action='store_true',
                        help='closing-range: only score stocks in STRONG BUY / BUY industries')
    parser.add_argument('--sctr-min', type=float, default=60.0,
                        help='Minimum SCTR for correction filter (default 60)')
    parser.add_argument('--universe', choices=['sc', 'tv'], default='sc',
                        help='Stock universe: sc=StockCharts DB (~5200), tv=TradingView CSV (~3300)')
    parser.add_argument('--date', default=None,
                        help='As-of date YYYY-MM-DD for backtesting (default: latest available)')
    args = parser.parse_args()

    cfg = Config()
    cfg.universe_source = args.universe
    as_of  = parse_date(args.date)
    source = args.source

    if args.mode == 'update':
        logger.info(f"=== Full Pipeline Update (source={source}) ===")
        run_update(cfg, as_of, source=source)
    elif args.mode == 'industry':
        logger.info(f"=== Industry Pipeline (source={source}) ===")
        run_industry_pipeline(cfg, as_of, source=source)
    elif args.mode == 'stocks':
        logger.info("=== Stocks Pipeline ===")
        run_stocks_pipeline(cfg, as_of)
    elif args.mode == 'sctr':
        run_industry(cfg, as_of)
    elif args.mode == 'stock-sctr':
        run_stocks(cfg, as_of)
    elif args.mode == 'rs-percentile':
        logger.info(f"=== RS Percentile Rating (IBD/Minervini, source={source}) ===")
        df = rs_percentile.run(cfg, as_of=as_of, source=source)
        if df.empty:
            logger.error("No RS Percentile results computed")
        else:
            path = rs_percentile.save(df, cfg, as_of=as_of, source=source)
            rs_percentile.print_report(df)
            print(f"\nSaved → {path}")
    elif args.mode == 'stage':
        logger.info(f"=== Stage Analysis (Weinstein / Minervini, source={source}) ===")
        df = stage_analysis.run(cfg, as_of=as_of, source=source)
        if df.empty:
            logger.error("No stage analysis results computed")
        else:
            csv_path, md_path = stage_analysis.save(df, cfg, as_of=as_of, source=source)
            stage_analysis.print_report(df)
            print(f"\nCSV → {csv_path}")
            print(f"MD  → {md_path}")
    elif args.mode == 'stock-screener':
        logger.info("=== Stock Screener (within top GICS industries) ===")
        df = stock_screener.run(cfg, as_of=as_of)
        if df.empty:
            logger.warning("No stock screener results — run --mode update first")
        else:
            csv_path, md_path = stock_screener.save(df, cfg, as_of=as_of)
            stock_screener.print_report(df)
            print(f"\nCSV → {csv_path}")
            print(f"MD  → {md_path}")
    elif args.mode == 'signal-delta':
        logger.info("=== Signal Delta ===")
        delta_df = signal_delta.run(cfg, as_of=as_of, prev_label=args.prev_date)
        if delta_df.empty:
            logger.info("Nothing to report — either no previous file or no changes detected")
        else:
            csv_path, md_path = signal_delta.save(delta_df, cfg, as_of=as_of)
            signal_delta.print_report(delta_df)
            print(f"\nCSV → {csv_path}")
            print(f"MD  → {md_path}")
    elif args.mode == 'market-breadth':
        logger.info("=== Market Breadth ===")
        overall, sector_df = breadth.run(cfg, as_of=as_of)
        if not overall:
            logger.error("No breadth results — run --mode stage first")
        else:
            csv_path, md_path = breadth.save(overall, sector_df, cfg, as_of=as_of)
            breadth.print_report(overall, sector_df)
            print(f"\nCSV → {csv_path}")
            print(f"MD  → {md_path}")
    elif args.mode == 'closing-range':
        logger.info("=== Closing Range Analysis (IBD / O'Neil Correction Leader Filter) ===")
        corr_start_arg = parse_date(args.correction_start)
        corr_end_arg   = parse_date(args.correction_end)

        # Optionally limit to STRONG BUY / BUY industry stocks
        ind_filter: list[str] | None = None
        if args.top_industries_only:
            label = str(as_of) if as_of else None
            mom_path = (cfg.results_dir / f"momentum_screen_{label}.csv") if label else \
                       closing_range._latest_file(cfg.results_dir, "momentum_screen_*.csv")
            if mom_path and mom_path.exists():
                mom = pd.read_csv(mom_path)
                ind_filter = mom.loc[
                    mom['faber_signal'].isin(['STRONG BUY', 'BUY']), 'industry_key'
                ].tolist()
                logger.info(f"Top-industries-only: {len(ind_filter)} STRONG BUY/BUY industries")

        cr_df, c_start, c_end = closing_range.run(
            cfg, as_of=as_of,
            correction_start=corr_start_arg,
            correction_end=corr_end_arg,
            industry_keys=ind_filter,
            min_score=args.min_score,
        )
        if cr_df.empty:
            logger.warning("No closing range leaders found — try --min-score 2 or check correction window")
        else:
            csv_path, md_path = closing_range.save(cr_df, cfg, c_start, c_end, as_of=as_of)
            closing_range.print_report(cr_df, c_start, c_end)
            print(f"\nCSV → {csv_path}")
            print(f"MD  → {md_path}")
    elif args.mode == 'market-clock':
        logger.info("=== Market Clock: Follow-Through Day & Distribution Day ===")
        result = market_clock.run(cfg, as_of=as_of)
        if result is None or (isinstance(result, tuple) and result[0].empty):
            logger.error("No market clock results — check SPY/QQQ/IWM in daily_dir")
        else:
            df, detail = result
            csv_path, md_path = market_clock.save(df, cfg, records=detail, as_of=as_of)
            market_clock.print_report(detail)
            print(f"\nCSV → {csv_path}")
            print(f"MD  → {md_path}")
    elif args.mode == 'rs-ma':
        logger.info("=== RS-Line vs 13-week MA Signals ===")
        df = rs_ma_signals.run(cfg, as_of=as_of)
        if df.empty:
            logger.error("No RS-MA results — run --mode backfill first")
        else:
            path = rs_ma_signals.save(df, cfg, as_of=as_of)
            display_cols = ['rank', 'industry_name', 'sector_name',
                            'quadrant', 'rs_ratio_now', 'rs_ma13',
                            'weeks_above_ma', 'rs_ma_slope', 'rs_ma_signal']
            for sig in ['RS Cross Up', 'RS Above MA (rising)', 'RS Above MA',
                        'RS Above MA (fading)', 'RS Below MA (recovering)',
                        'RS Cross Down', 'RS Below MA', 'RS Below MA (falling)']:
                sub = df[df['rs_ma_signal'] == sig]
                if sub.empty:
                    continue
                print(f"\n{'─'*72}")
                print(f"  {sig}  ({len(sub)})")
                print(f"{'─'*72}")
                print(sub[display_cols].to_string(index=False))
            print(f"\nSaved → {path}")
    elif args.mode == 'correction':
        logger.info("=== Correction Leader Filter ===")
        corr_start = pd.Timestamp(args.correction_start) if args.correction_start else None
        df = correction_filter.run(cfg, correction_start=corr_start,
                                   as_of=as_of, sctr_min=args.sctr_min)
        if df.empty:
            logger.warning("No results — check correction window or lower --sctr-min")
        else:
            path = correction_filter.save(df, cfg, as_of=as_of)
            cols = ['rank', 'ticker', 'cap_bucket', 'sector_name', 'industry_name',
                    'sctr', 'cr_above_60_pct', 'ema21_pct', 'held_21ema',
                    'rs_change_pct', 'rs_positive',
                    'was_strong_pre', 'acc_ratio', 'is_accumulating',
                    'filters_passed', 'drawdown_pct', 'composite']
            print(f"\nTop 40 correction leaders:")
            print(df[cols].head(40).to_string(index=False))
            print(f"\nTotal passing: {len(df)}  |  Saved → {path}")
    elif args.mode == 'validate':
        run_validate(cfg)
    elif args.mode == 'perf':
        logger.info("=== Performance Screener (1w/2w/1m/3m/6m, RS, relret vs SPY & QQQ) ===")
        df = perf_screener.run(cfg, as_of=as_of)
        if df.empty:
            logger.error("No perf screener results computed")
        else:
            path = perf_screener.save(df, cfg, as_of=as_of)
            perf_screener.print_report(df)
            print(f"\nSaved → {path}")
    elif args.mode == 'history-backfill':
        logger.info(f"=== History Backfill (trailing {args.months_back} months, daily) ===")
        df = history_backfill.run(cfg, months_back=args.months_back)
        if df.empty:
            logger.error("No history computed")
        else:
            paths = history_backfill.save(df, cfg)
            print(f"\n{df['symbol'].nunique()} symbols x {df['date'].nunique()} dates = {len(df)} rows")
            print(f"Saved → {paths['long']}")
    elif args.mode == 'history-append':
        logger.info("=== History Append (today only) ===")
        df = history_tracker.run(cfg)
        if df.empty:
            logger.error("No history computed")
        else:
            paths = history_tracker.save(df, cfg)
            print(f"\nHistory now: {df['symbol'].nunique()} symbols x {df['date'].nunique()} dates = {len(df)} rows")
            print(f"Saved → {paths['long']}")


if __name__ == '__main__':
    main()
