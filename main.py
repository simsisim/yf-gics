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
import src.sctr_industry_gics as sctr_industry_gics
import src.sctr_stocks        as sctr_stocks
import src.sctr_breadth       as sctr_breadth
import src.rrg_engine          as rrg_engine
import src.rrg_chart           as rrg_chart
import src.rotation_composite  as rotation_composite
import src.correction_filter   as correction_filter
import src.backfill             as backfill
import src.rotation_severity    as rotation_severity
import src.ath_monitor           as ath_monitor
import src.rs_ma_signals         as rs_ma_signals
import src.market_clock          as market_clock
import src.rs_percentile         as rs_percentile
import src.momentum_screen       as momentum_screen
import src.stage_analysis        as stage_analysis
import src.closing_range         as closing_range
import src.breadth               as breadth
import src.signal_delta          as signal_delta
import src.stock_screener        as stock_screener

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

    computed_files = sorted(cfg.industry_sctr_dir.glob('industry_sctr_*.csv'))
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


def run_update(cfg: Config, as_of: date | None) -> None:
    """
    Full pipeline in dependency order. Each step is run independently;
    failures are logged but do not abort subsequent steps.
    """
    label = str(as_of) if as_of else pd.Timestamp.today().strftime('%Y-%m-%d')
    steps = [
        ('industry-gics',  lambda: sctr_industry_gics.save(sctr_industry_gics.run(cfg, as_of=as_of), cfg, as_of=as_of)),
        ('rrg',            lambda: rrg_engine.save(rrg_engine.run(cfg, as_of=as_of), cfg, as_of=as_of)),
        ('rrg-monthly',    lambda: rrg_engine.save_monthly(rrg_engine.run_monthly(cfg, as_of=as_of), cfg, as_of=as_of)),
        ('ath',            lambda: ath_monitor.save(ath_monitor.run(cfg, as_of=as_of), cfg, as_of=as_of)),
        ('rs-ma',          lambda: rs_ma_signals.save(rs_ma_signals.run(cfg, as_of=as_of), cfg, as_of=as_of)),
        ('market-clock',   lambda: market_clock.save(market_clock.run(cfg, as_of=as_of), cfg, as_of=as_of)),
        ('rs-percentile',  lambda: rs_percentile.save(rs_percentile.run(cfg, as_of=as_of), cfg, as_of=as_of)),
        ('stage',          lambda: stage_analysis.save(stage_analysis.run(cfg, as_of=as_of), cfg, as_of=as_of)),
        ('momentum',       lambda: momentum_screen.save(momentum_screen.run(cfg, as_of=as_of), cfg, as_of=as_of)),
        ('severity',       lambda: rotation_severity.save(rotation_severity.run(cfg, as_of=as_of), cfg, as_of=as_of)),
        ('market-breadth', lambda: breadth.save(*breadth.run(cfg, as_of=as_of), cfg, as_of=as_of)),
        ('signal-delta',   lambda: signal_delta.save(signal_delta.run(cfg, as_of=as_of), cfg, as_of=as_of)),
        ('stock-screener', lambda: stock_screener.save(stock_screener.run(cfg, as_of=as_of), cfg, as_of=as_of)),
    ]

    results: list[tuple[str, str]] = []
    for name, fn in steps:
        logger.info(f"── update: {name} ──")
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

    print(f"\n{'─'*60}")
    print(f"  Update summary — {label}")
    print(f"{'─'*60}")
    for name, status in results:
        icon = '✓' if status.startswith('OK') else ('⚠' if status.startswith('SKIP') else '✗')
        print(f"  {icon}  {name:<18} {status[:60]}")
    print(f"{'─'*60}")


def main() -> None:
    parser = argparse.ArgumentParser(description='SCTR Analysis System')
    parser.add_argument('--mode',
                        choices=['industry', 'industry-gics', 'stocks', 'breadth',
                                 'rrg', 'rrg-monthly', 'composite', 'correction', 'backfill',
                                 'severity', 'ath', 'rs-ma', 'rs-percentile', 'market-clock',
                                 'momentum', 'stage', 'closing-range', 'market-breadth',
                                 'signal-delta', 'stock-screener', 'update', 'all', 'validate'],
                        default='industry-gics',
                        help='Which analysis to run')
    parser.add_argument('--backfill-start', default='2021-01-01',
                        help='Start date for historical backfill (default: 2021-01-01)')
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
    as_of = parse_date(args.date)

    if args.mode == 'industry':
        run_industry(cfg, as_of)
    elif args.mode == 'industry-gics':
        logger.info("=== Layer 1b: GICS Industry SCTR (synthetic constituent index) ===")
        df = sctr_industry_gics.run(cfg, as_of=as_of)
        if df.empty:
            logger.error("No results")
        else:
            path = sctr_industry_gics.save(df, cfg, as_of=as_of)
            cols = ['rank', 'industry_name', 'sector_name', 'constituent_count', 'sctr', 'raw_score']
            print(f"\nTop 20 GICS industries by SCTR  (as of {as_of or 'latest'}):")
            print(df[cols].head(20).to_string(index=False))
            print(f"\nBottom 10:")
            print(df[cols].tail(10).to_string(index=False))
            print(f"\nSaved → {path}")
    elif args.mode == 'stocks':
        run_stocks(cfg, as_of)
    elif args.mode == 'breadth':
        run_breadth(cfg, as_of)
    elif args.mode == 'all':
        run_industry(cfg, as_of)
        run_stocks(cfg, as_of)
        run_breadth(cfg, as_of)
    elif args.mode == 'rrg':
        logger.info("=== RRG: GICS Industries vs SPY (weekly) ===")
        df = rrg_engine.run(cfg, as_of=as_of)
        if df.empty:
            logger.error("No RRG results")
        else:
            label     = str(as_of) if as_of else pd.Timestamp.today().strftime('%Y-%m-%d')
            csv_path  = rrg_engine.save(df, cfg, as_of=as_of)
            rrg_dir   = cfg.results_dir / 'rrg' / label
            png_paths = rrg_chart.plot_all_sectors(df, rrg_dir, label)
            logger.info(f"Generated {len(png_paths)} PNG charts → {rrg_dir}")
            cols = ['industry_name', 'sector_name', 'constituent_count',
                    'rs_ratio', 'rs_momentum', 'quadrant', 'tail_direction']
            for quad in ['Leading', 'Improving', 'Weakening', 'Lagging']:
                sub = df[df['quadrant'] == quad][cols]
                print(f"\n--- {quad} ({len(sub)}) ---")
                print(sub.head(10).to_string(index=False))
            print(f"\nCSV  → {csv_path}")
            print(f"PNGs → {rrg_dir}/")
            for p in png_paths:
                print(f"       {p.name}")
    elif args.mode == 'rrg-monthly':
        logger.info("=== Monthly RRG: GICS Industries vs SPY ===")
        df = rrg_engine.run_monthly(cfg, as_of=as_of)
        if df.empty:
            logger.error("No monthly RRG results")
        else:
            path = rrg_engine.save_monthly(df, cfg, as_of=as_of)
            cols = ['industry_name', 'sector_name', 'rs_ratio_m', 'rs_momentum_m',
                    'quadrant_m', 'tail_direction_m']
            for quad in ['Leading', 'Improving', 'Weakening', 'Lagging']:
                sub = df[df['quadrant_m'] == quad][cols]
                print(f"\n--- {quad} ({len(sub)}) ---")
                print(sub.head(12).to_string(index=False))
            print(f"\nSaved → {path}")
    elif args.mode == 'composite':
        logger.info("=== Rotation Composite ===")
        df = rotation_composite.run(cfg, as_of=as_of)
        if df.empty:
            logger.error("No composite results — run industry-gics, rrg, and breadth first")
        else:
            path = rotation_composite.save(df, cfg, as_of=as_of)
            display_cols = ['rank', 'industry_name', 'sector_name',
                            'sctr', 'quadrant', 'composite', 'signal', 'convergence']

            for sig in ['Strong Rotation In', 'Rotation In', 'Neutral',
                        'Rotation Out', 'Strong Rotation Out']:
                sub = df[df['signal'] == sig][display_cols]
                if sub.empty:
                    continue
                print(f"\n{'─'*70}")
                print(f"  {sig}  ({len(sub)})")
                print(f"{'─'*70}")
                print(sub.to_string(index=False))

            print(f"\nSaved → {path}")
    elif args.mode == 'ath':
        logger.info("=== Monthly ATH Breakout Monitor ===")
        df = ath_monitor.run(cfg, as_of=as_of)
        if df.empty:
            logger.error("No ATH results")
        else:
            path = ath_monitor.save(df, cfg, as_of=as_of)
            cols = ['rank', 'industry_name', 'sector_name',
                    'pct_from_ath', 'ath_date', 'months_since_ath',
                    'months_holding_above_prev', 'ath_signal', 'ath_score']
            for sig in ['New ATH', 'Holding Breakout', 'Near ATH', 'Recovering', 'Below ATH']:
                sub = df[df['ath_signal'] == sig]
                if sub.empty:
                    continue
                print(f"\n{'─'*72}")
                print(f"  {sig}  ({len(sub)})")
                print(f"{'─'*72}")
                print(sub[cols].head(20).to_string(index=False))
            print(f"\nSaved → {path}")
    elif args.mode == 'severity':
        logger.info("=== Rotation Severity Scoring ===")
        df = rotation_severity.run(cfg, as_of=as_of)
        if df.empty:
            logger.error("No severity results — run --mode backfill first")
        else:
            path = rotation_severity.save(df, cfg, as_of=as_of)
            display_cols = ['rank', 'industry_name', 'sector_name',
                            'quadrant', 'quadrant_m', 'timeframe_label',
                            'ath_signal', 'pct_from_ath',
                            'rs_ma_signal', 'rs_weeks_above_ma',
                            'rs_pct_composite', 'rs_pct_3m', 'rs_new_high',
                            'stage', 'minervini_count',
                            'momentum_score', 'faber_signal',
                            'sctr', 'sctr_trend', 'weeks_in_quadrant', 'transition',
                            'severity_score', 'severity_label']

            sev_labels = [
                'Both TF Confirmed + ATH', 'Both TF Confirmed',
                'Strong Confirmed + ATH', 'Strong Confirmed',
                'Confirmed + ATH', 'Confirmed',
                'Early Signal', 'Pullback in Uptrend',
                'Neutral / Watch', 'Weakening', 'Confirmed Exit',
            ]
            # Also include dynamic labels with RS≥80 / ★RS suffixes
            existing = df['severity_label'].unique().tolist()
            sev_labels += [l for l in existing if l not in sev_labels]

            for label in sev_labels:
                sub = df[df['severity_label'] == label]
                if sub.empty:
                    continue
                print(f"\n{'─'*80}")
                print(f"  {label}  ({len(sub)})")
                print(f"{'─'*80}")
                print(sub[display_cols].head(20).to_string(index=False))

            print(f"\nSaved → {path}")
    elif args.mode == 'rs-percentile':
        logger.info("=== RS Percentile Rating (IBD/Minervini) ===")
        df = rs_percentile.run(cfg, as_of=as_of)
        if df.empty:
            logger.error("No RS Percentile results computed")
        else:
            path = rs_percentile.save(df, cfg, as_of=as_of)
            rs_percentile.print_report(df)
            print(f"\nSaved → {path}")
    elif args.mode == 'stage':
        logger.info("=== Stage Analysis (Weinstein / Minervini) ===")
        df = stage_analysis.run(cfg, as_of=as_of)
        if df.empty:
            logger.error("No stage analysis results computed")
        else:
            csv_path, md_path = stage_analysis.save(df, cfg, as_of=as_of)
            stage_analysis.print_report(df)
            print(f"\nCSV → {csv_path}")
            print(f"MD  → {md_path}")
    elif args.mode == 'momentum':
        logger.info("=== Momentum Screen (Faber + Moskowitz-Grinblatt + ST SCTR) ===")
        df = momentum_screen.run(cfg, as_of=as_of)
        if df.empty:
            logger.error("No momentum screen results — run --mode rs-percentile and --mode industry-gics first")
        else:
            csv_path, md_path = momentum_screen.save(df, cfg, as_of=as_of)
            momentum_screen.print_report(df)
            print(f"\nCSV → {csv_path}")
            print(f"MD  → {md_path}")
    elif args.mode == 'update':
        logger.info("=== Full Pipeline Update ===")
        run_update(cfg, as_of)
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
    elif args.mode == 'backfill':
        logger.info("=== Historical Backfill: Industry SCTR + RRG ===")
        df = backfill.run(cfg, start_date=args.backfill_start)
        if df.empty:
            logger.error("Backfill produced no results")
        else:
            paths = backfill.save(df, cfg)
            n_dates = df['date'].nunique()
            n_inds  = df['industry_key'].nunique()
            print(f"\nBackfill complete: {len(df)} rows  ({n_inds} industries × {n_dates} weeks)")
            print(f"Date range: {df['date'].min()} → {df['date'].max()}")
            print(f"\nSample (latest week, top 15 by SCTR):")
            latest = df[df['date'] == df['date'].max()]
            cols = ['industry_name', 'sector_name', 'sctr', 'rs_ratio', 'rs_momentum', 'quadrant']
            print(latest.sort_values('sctr', ascending=False)[cols].head(15).to_string(index=False))
            print("\nFiles saved:")
            for name, p in paths.items():
                print(f"  {name}: {p}")
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


if __name__ == '__main__':
    main()
