"""
Stock Screener — individual stock ranking within top GICS industries.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PURPOSE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Completes the top-down pipeline:  market → sector → industry → stock.

Given the STRONG BUY / BUY industries identified by the momentum screen,
this module drills into individual constituent stocks and ranks them by
the same criteria used at the industry level:

  Stage score     Weinstein/Minervini MA structure (50/150/200-SMA)
  RS rank         12-month excess return vs SPY, percentile-ranked within
                  the screened universe
  SCTR            StockCharts Technical Rank (0–99.9) from stock_sctr CSV
  Closing range   IBD correction leader score (F1–F5) if available from
                  the most recent closing_range CSV
  Minervini score 0–7 criteria met

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPOSITE SCORE  (0–100)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RS rank (12m)       × 35%
  Stage score         × 30%   (2B=100, 2A=85, 2C=70, 1=45, 3=25, 4=5)
  Minervini %         × 15%   (minervini_count / 7 × 100)
  SCTR                × 10%   (0–99.9, neutral 50 if not available)
  Closing range score × 10%   (leader_score / 5 × 100, 0 if no CR data)

Filters applied before ranking:
  • Stage must be 2A or 2B              (in confirmed uptrend)
  • Minervini count ≥ 3                 (at least 3/7 criteria)
  • Industry faber_signal in STRONG BUY / BUY

Output: results/stock_screener_YYYY-MM-DD.csv + .md
"""

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from src.rotation_composite import _latest_file
from src.stage_analysis import _compute as _stage_compute, MIN_DAYS, SLOPE_WINDOW

logger = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────────
MIN_STAGE_RANK   = {'2A', '2B'}    # only stage 2 stocks
MIN_MINERVINI    = 3               # 3+ Minervini criteria
TOP_FABER        = {'STRONG BUY', 'BUY'}
WARMUP_DAYS      = 260             # calendar days of history needed

_STAGE_SCORE = {'2B': 100, '2A': 85, '2C': 70, '1': 45, '3': 25, '4': 5}

# ── RS helpers (12-month excess return vs SPY) ─────────────────────────────

def _excess_12m(close: pd.Series, spy_close: pd.Series) -> float | None:
    """12-month excess return of stock vs SPY."""
    aligned = close.align(spy_close, join='inner')[0]
    spy_al  = spy_close.reindex(aligned.index)
    if len(aligned) < 200:
        return None
    s_ret  = float(aligned.iloc[-1] / aligned.iloc[-252] - 1) if len(aligned) >= 252 \
             else float(aligned.iloc[-1] / aligned.iloc[0] - 1)
    spy_ret = float(spy_al.iloc[-1] / spy_al.iloc[-252] - 1)  if len(spy_al) >= 252 \
              else float(spy_al.iloc[-1] / spy_al.iloc[0] - 1)
    return round((s_ret - spy_ret) * 100, 2)


def _rank_to_pct(series: pd.Series) -> pd.Series:
    return (series.rank(pct=True, na_option='keep') * 98 + 1).round().astype('Int64')


# ── OHLCV loader (reuse pattern from closing_range) ────────────────────────

def _load_close(ticker: str, daily_dir: Path) -> pd.Series | None:
    p = daily_dir / f"{ticker}.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, parse_dates=['Date'])
    except Exception:
        return None
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
    df = df.sort_values('Date')
    if 'Close' not in df.columns or len(df) < MIN_DAYS:
        return None
    return df.set_index('Date')['Close'].astype(float)


# ── Main ──────────────────────────────────────────────────────────────────

def run(
    config: Config,
    as_of: date | None = None,
    faber_filter: set[str] | None = None,
) -> pd.DataFrame:
    """
    Screen individual stocks within top GICS industries.

    faber_filter : set of faber_signal values to include.
                   Defaults to {'STRONG BUY', 'BUY'}.
    """
    label        = str(as_of) if as_of else None
    faber_filter = faber_filter or TOP_FABER

    # ── Load industry list (filtered by Faber signal) ─────────────────────
    if label:
        mom_path = config.results_dir / f"momentum_screen_{label}.csv"
    else:
        mom_path = _latest_file(config.results_dir, "momentum_screen_*.csv")
    if mom_path is None or not mom_path.exists():
        logger.error("momentum_screen CSV not found — run --mode momentum first")
        return pd.DataFrame()

    mom = pd.read_csv(mom_path)
    top_ind = mom[mom['faber_signal'].isin(faber_filter)]['industry_key'].tolist()
    logger.info(f"Top industries ({', '.join(faber_filter)}): {len(top_ind)}")

    # ── Load closing range scores (optional) ──────────────────────────────
    cr_map: dict[str, int] = {}
    if label:
        cr_path = config.results_dir / f"closing_range_{label}.csv"
    else:
        cr_path = _latest_file(config.results_dir, "closing_range_*.csv")
    if cr_path and cr_path.exists():
        cr = pd.read_csv(cr_path)
        cr_map = cr.set_index('ticker')['leader_score'].to_dict()
        logger.info(f"Loaded closing range scores: {len(cr_map)} stocks")

    # ── Load individual stock SCTR (optional) ─────────────────────────────
    sctr_map: dict[str, float] = {}
    if label:
        sctr_path = config.stock_sctr_dir / f"stock_sctr_{label}.csv"
    else:
        sctr_path = _latest_file(config.stock_sctr_dir, "stock_sctr_*.csv")
    if sctr_path and sctr_path.exists():
        try:
            sctr_df = pd.read_csv(sctr_path)
            sctr_map = sctr_df.set_index('ticker')['sctr'].to_dict()
            logger.info(f"Loaded stock SCTR scores: {len(sctr_map)} stocks")
        except Exception as e:
            logger.warning(f"Could not load stock SCTR: {e}")

    # ── Load industry meta ─────────────────────────────────────────────────
    sbi        = pd.read_csv(config.stocks_by_industry_csv)
    industries = pd.read_csv(config.industries_csv)
    ind_meta   = industries.set_index('industry_key')[['industry_name', 'sector_name']].to_dict('index')

    # Load industry-level faber/stage context for enrichment
    ind_faber = mom.set_index('industry_key')['faber_signal'].to_dict()
    ind_stage: dict[str, str] = {}
    if label:
        st_path = config.results_dir / f"stage_analysis_{label}.csv"
    else:
        st_path = _latest_file(config.results_dir, "stage_analysis_*.csv")
    if st_path and st_path.exists():
        st_df = pd.read_csv(st_path)
        ind_stage = st_df.set_index('industry_key')['stage'].to_dict()

    # ── Load SPY close ─────────────────────────────────────────────────────
    spy_close: pd.Series = pd.Series(dtype=float)
    spy_raw = _load_close('SPY', config.daily_dir)
    if spy_raw is not None:
        spy_close = spy_raw

    # ── Score each stock ───────────────────────────────────────────────────
    universe = sbi[sbi['industry_key'].isin(top_ind)][['symbol', 'industry_key']].dropna()
    total = len(universe)
    logger.info(f"Screening {total} stocks across {len(top_ind)} industries...")

    raw_records = []   # all stocks with stage data (for RS ranking within universe)
    skipped = 0

    for i, (_, row) in enumerate(universe.iterrows()):
        if i % 300 == 0 and i > 0:
            logger.info(f"  {i}/{total} ({len(raw_records)} with data)...")
        ticker  = str(row['symbol'])
        ind_key = str(row['industry_key'])

        close = _load_close(ticker, config.daily_dir)
        if close is None:
            skipped += 1
            continue

        if not spy_close.empty:
            rs_12m = _excess_12m(close, spy_close)
        else:
            rs_12m = None

        metrics = _stage_compute(close, rs_pct=None)   # Minervini C7 left open
        if metrics is None:
            skipped += 1
            continue

        stage         = metrics['stage']
        minervini_cnt = metrics['minervini_count']
        stage_score   = _STAGE_SCORE.get(stage, 45)
        cr_score      = int(cr_map.get(ticker, 0))
        sctr_score    = float(sctr_map.get(ticker, 50.0))  # neutral 50 if missing

        meta = ind_meta.get(ind_key, {})
        raw_records.append({
            'ticker':         ticker,
            'industry_key':   ind_key,
            'industry_name':  meta.get('industry_name', ind_key),
            'sector_name':    meta.get('sector_name', ''),
            'ind_faber':      ind_faber.get(ind_key, ''),
            'ind_stage':      ind_stage.get(ind_key, ''),
            # stock-level
            'stage':          stage,
            'stage_score':    stage_score,
            'minervini_count': minervini_cnt,
            'rs_12m':         rs_12m,
            'cr_score':       cr_score,
            'sctr':           sctr_score,
            # from stage metrics
            'pct_vs_sma200':  metrics.get('pct_vs_sma200', np.nan),
            'pct_vs_sma150':  metrics.get('pct_vs_sma150', np.nan),
            'sma200_slope':   metrics.get('sma200_slope', np.nan),
            'range_pos_52w':  metrics.get('range_pos_52w', np.nan),
            'pct_from_52w_high': metrics.get('pct_from_52w_high', np.nan),
            'golden_cross':   metrics.get('golden_cross', False),
        })

    logger.info(f"Loaded {len(raw_records)} stocks  (skipped {skipped})")

    if not raw_records:
        return pd.DataFrame()

    df = pd.DataFrame(raw_records)

    # ── RS percentile rank within this screened universe ──────────────────
    df['rs_pct'] = _rank_to_pct(pd.to_numeric(df['rs_12m'], errors='coerce'))

    # ── Apply filters ─────────────────────────────────────────────────────
    pre_filter = len(df)
    df = df[
        df['stage'].isin(MIN_STAGE_RANK) &
        (df['minervini_count'] >= MIN_MINERVINI)
    ].copy()
    logger.info(f"After filters (Stage 2A/2B + Minervini≥{MIN_MINERVINI}): "
                f"{len(df)} / {pre_filter} stocks")

    if df.empty:
        return pd.DataFrame()

    # ── Composite score ───────────────────────────────────────────────────
    rs_w, st_w, mc_w, sctr_w, cr_w = 0.35, 0.30, 0.15, 0.10, 0.10

    df['composite'] = (
        df['rs_pct'].astype(float).fillna(50)  * rs_w +
        df['stage_score'].astype(float)         * st_w +
        (df['minervini_count'] / 7 * 100)       * mc_w +
        df['sctr'].astype(float).fillna(50)     * sctr_w +
        (df['cr_score'] / 5 * 100)              * cr_w
    ).round(1)

    df['composite_pct'] = _rank_to_pct(df['composite'])

    # ── Sort and rank ─────────────────────────────────────────────────────
    df = (df.sort_values('composite', ascending=False)
            .reset_index(drop=True))
    df.insert(0, 'rank', df.index + 1)

    dist = df['stage'].value_counts().to_dict()
    logger.info(f"Stage distribution: {dist}")
    logger.info(f"Top 5: {df[['ticker','industry_name','composite']].head(5).to_dict('records')}")

    return df


# ── Save ──────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame, config: Config, as_of: date | None = None) -> tuple[Path, Path]:
    config.setup_dirs()
    label = str(as_of) if as_of else pd.Timestamp.today().strftime('%Y-%m-%d')

    csv_cols = [
        'rank', 'ticker', 'industry_key', 'sector_name', 'industry_name',
        'ind_faber', 'ind_stage',
        'composite', 'composite_pct',
        'stage', 'stage_score', 'minervini_count',
        'rs_12m', 'rs_pct', 'sctr', 'cr_score',
        'pct_vs_sma200', 'pct_vs_sma150', 'sma200_slope',
        'range_pos_52w', 'pct_from_52w_high', 'golden_cross',
    ]
    present = [c for c in csv_cols if c in df.columns]

    csv_out = config.results_dir / f"stock_screener_{label}.csv"
    df[present].to_csv(csv_out, index=False)
    logger.info(f"Saved stock screener CSV → {csv_out}  ({len(df)} stocks)")

    md_out = config.results_dir / f"stock_screener_{label}.md"
    md_out.write_text(_build_md(df, label), encoding='utf-8')
    logger.info(f"Saved stock screener MD  → {md_out}")

    return csv_out, md_out


# ── Markdown report ────────────────────────────────────────────────────────

def _build_md(df: pd.DataFrame, label: str) -> str:
    lines = []
    lines.append(f"# Stock Screener — {label}")
    lines.append("")
    lines.append(f"> {len(df)} stocks  ·  Stage 2A/2B  ·  Minervini ≥ 3/7  ·  "
                 f"STRONG BUY or BUY industries")
    lines.append("")

    lines.append("## Score Distribution")
    lines.append("")
    lines.append("| Bucket | Stocks |")
    lines.append("|--------|:------:|")
    for lo, hi, lbl in [(80, 100, 'Elite (80-100)'), (65, 80, 'Strong (65-79)'),
                        (50, 65, 'Good (50-64)'),   (0, 50, 'Marginal (<50)')]:
        n = ((df['composite'] >= lo) & (df['composite'] < hi)).sum()
        lines.append(f"| {lbl} | {n} |")
    lines.append("")

    # Group by industry
    for ind_key, grp in df.groupby('industry_key'):
        ind_name = grp['industry_name'].iloc[0]
        ind_fab  = grp['ind_faber'].iloc[0]
        lines.append(f"## {ind_name} ({ind_fab})  — {len(grp)} stocks")
        lines.append("")
        lines.append("| Rank | Ticker | Score | Stage | M# | RS% | SCTR | CR | vs200% | vs150% | Slope | 52wPos | HiDiff% |")
        lines.append("|-----:|--------|:-----:|:-----:|:--:|:---:|:----:|:--:|-------:|-------:|:-----:|:------:|--------:|")
        for _, r in grp.iterrows():
            gc = '✓' if r.get('golden_cross') else ''
            sctr_val = f"{r['sctr']:.1f}" if pd.notna(r.get('sctr')) else '–'
            lines.append(
                f"| {int(r['rank'])} | **{r['ticker']}** "
                f"| {r['composite']:.0f} "
                f"| {r['stage']} "
                f"| {int(r['minervini_count'])}/7 "
                f"| {int(r['rs_pct']) if pd.notna(r.get('rs_pct')) else '–'} "
                f"| {sctr_val} "
                f"| {int(r['cr_score'])}/5 "
                f"| {r['pct_vs_sma200']:+.1f}% "
                f"| {r['pct_vs_sma150']:+.1f}% "
                f"| {r['sma200_slope']:+.2f}% "
                f"| {r['range_pos_52w']:.0f}% "
                f"| {r['pct_from_52w_high']:+.1f}% {gc} |"
            )
        lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append("**Composite** = RS rank (35%) + Stage score (30%) + Minervini% (15%) + SCTR (10%) + CR score (10%)")
    lines.append("")
    lines.append("**RS rank** — 12-month excess return vs SPY, percentile-ranked within this screened universe.")
    lines.append("")
    lines.append("**Stage score** — Weinstein/Minervini MA structure: 2B=100, 2A=85, filtered to Stage 2A/2B only.")
    lines.append("")
    lines.append("**Minervini %** — (criteria_met / 7) × 100. Filtered to ≥ 3/7.")
    lines.append("")
    lines.append("**CR score** — IBD closing range leader score (0–5) from correction window analysis.")
    lines.append("")
    return '\n'.join(lines)


# ── Terminal report ────────────────────────────────────────────────────────

def print_report(df: pd.DataFrame, top_n: int = 50) -> None:
    DIVIDER = '─' * 106
    print(f"\n{DIVIDER}")
    print(f"  Stock Screener  |  {len(df)} stocks  |  Stage 2A/2B  |  Minervini ≥ 3/7")
    print(DIVIDER)

    prev_ind = None
    shown = 0
    for _, r in df.iterrows():
        if shown >= top_n:
            print(f"\n  ... {len(df) - shown} more (see CSV/MD for full list)")
            break
        ind = r['industry_name']
        if ind != prev_ind:
            print(f"\n  ── {ind}  [{r['ind_faber']}]  (ind stage: {r['ind_stage']}) ──")
            print(f"  {'Rnk':>4}  {'Ticker':<8} {'Score':>6}  {'Stg':>3}  {'M#':>4}  "
                  f"{'RS%':>4}  {'SCTR':>5}  {'CR':>3}  {'vs200':>6}  {'vs150':>6}  {'Slp200':>7}  "
                  f"{'52wPos':>6}  {'HiDiff':>7}  {'GX'}")
            prev_ind = ind
        gc = '✓' if r.get('golden_cross') else ' '
        sctr_val = f"{r['sctr']:5.1f}" if pd.notna(r.get('sctr')) else '   – '
        print(
            f"  {int(r['rank']):>4}  {str(r['ticker']):<8} {r['composite']:>6.1f}  "
            f"{str(r['stage']):>3}  "
            f"{int(r['minervini_count']):>2}/7  "
            f"{int(r['rs_pct']) if pd.notna(r.get('rs_pct')) else 0:>4}  "
            f"{sctr_val}  "
            f"{int(r['cr_score']):>3}/5  "
            f"{r['pct_vs_sma200']:>+6.1f}%  "
            f"{r['pct_vs_sma150']:>+6.1f}%  "
            f"{r['sma200_slope']:>+7.2f}%  "
            f"{r['range_pos_52w']:>6.0f}%  "
            f"{r['pct_from_52w_high']:>+7.1f}%  {gc}"
        )
        shown += 1
    print(f"\n{DIVIDER}")
