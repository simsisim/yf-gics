"""
Stage Analysis — Weinstein / Minervini Stage Classification for GICS Industries.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE DEFINITIONS  (Weinstein 1988 + Minervini refinements)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Stage 1 — Basing / Accumulation
    Price flat near 200-SMA (within ±5%), 200-SMA flat or just turning up.
    Institutions quietly accumulating. Not actionable yet.

  Stage 2A — Early Uptrend  ← ENTRY ZONE (Weinstein / Minervini)
    Price broke above 200-SMA. 50-SMA just crossed above 200-SMA (golden
    cross) OR crossing imminently. 200-SMA beginning to slope upward.
    This is Weinstein's primary buy signal.

  Stage 2B — Mid Uptrend  ← CORE HOLDING ZONE
    All three MAs aligned and rising: price > 150-SMA > 200-SMA.
    50-SMA well above 200-SMA. 200-SMA trending up ≥ 4 weeks.
    Minervini's full Stage 2 confirmation.

  Stage 2C — Late Uptrend  ← REDUCE / TIGHTEN STOPS
    Price extended, 200-SMA slope slowing or flattening.
    First signs of distribution: price struggles to make new highs,
    or 50-SMA begins to flatten toward 200-SMA.

  Stage 3 — Distribution / Topping
    Price still near old highs but below its 50-SMA, OR 50-SMA has
    crossed below 200-SMA (death cross forming). 200-SMA flattening.
    Avoid new positions. Exit longs.

  Stage 4 — Decline
    Price below declining 200-SMA. 50-SMA below 200-SMA (death cross).
    Full downtrend. No long positions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MINERVINI'S 7 STAGE 2 CRITERIA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Price above 150-day SMA (30-week)
  2. Price above 200-day SMA (40-week)
  3. 150-day SMA above 200-day SMA
  4. 200-day SMA trending up for ≥ 4 weeks (slope positive)
  5. Price ≥ 30% above its 52-week low
  6. Price within 25% of its 52-week high
  7. RS Percentile (composite) ≥ 70

  Score 0–7. Minervini requires all 7 for individual stocks.
  For industries: ≥ 5 = strong Stage 2, ≥ 3 = developing.

Key MAs:
  sma50  — 50-day  (10-week)  short-term trend
  sma150 — 150-day (30-week)  Weinstein's primary reference
  sma200 — 200-day (40-week)  long-term trend / institutional baseline

Output: results/stage_analysis_YYYY-MM-DD.csv + .md
"""

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from src.backfill import _load_all_closes, _build_industry_daily_index
from src.rotation_composite import _latest_file

logger = logging.getLogger(__name__)

# ── MA periods ─────────────────────────────────────────────────────────────
SMA50  = 50
SMA150 = 150
SMA200 = 200
SLOPE_WINDOW = 20    # 4 weeks of trading days for 200-SMA slope
MIN_DAYS = 210       # need ≥ 210 days for all MAs

# ── Stage 2 thresholds (Minervini) ─────────────────────────────────────────
M_ABOVE_52W_LOW  = 1.30    # price ≥ 30% above 52-week low
M_FROM_52W_HIGH  = 0.75    # price ≥ 75% of 52-week high (within 25%)
M_RS_MIN         = 70      # RS percentile composite ≥ 70

# ── Stage score map (for sorting / severity integration) ───────────────────
_STAGE_SCORE = {
    '2B': 100,
    '2A': 85,
    '2C': 70,
    '1':  45,
    '3':  25,
    '4':  5,
}


def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()


def _slope_pct(series: pd.Series, window: int) -> float | None:
    """Annualised slope as % of current value over `window` bars."""
    s = series.dropna()
    if len(s) < window + 1:
        return None
    old = float(s.iloc[-window - 1])
    new = float(s.iloc[-1])
    if old <= 0:
        return None
    return round((new / old - 1) * 100, 3)


def _classify(
    price: float,
    s50: float,
    s150: float,
    s200: float,
    slope200: float | None,
) -> str:
    """
    Assign stage label from current MA values and 200-SMA slope.
    Priority: check Stage 2 family first, then 3, 4, 1.
    """
    above200    = price > s200
    above150    = price > s150
    above50     = price > s50
    golden_x    = s50 > s200          # 50 above 200
    s150_gt_200 = s150 > s200
    s200_rising = (slope200 or 0) > 0.05   # 200-SMA trending up
    s200_flat   = abs(slope200 or 0) <= 0.05

    # ── Stage 2 family ────────────────────────────────────────────────
    if above200 and golden_x and s200_rising:
        if above150 and s150_gt_200:
            # Check if late: 200-SMA slope slowing (slope < 0.15%)
            if (slope200 or 0) < 0.15 and not above50:
                return '2C'
            return '2B'
        else:
            return '2A'   # early: 150-SMA hasn't caught up

    if above200 and golden_x and s200_flat:
        return '2C'       # topping out: still structured but 200-SMA flattening

    # ── Stage 3 ───────────────────────────────────────────────────────
    if above200 and not golden_x:
        return '3'        # price above 200 but 50 already below → distributing

    if not above200 and s50 > s200 * 0.98 and s200_rising:
        return '3'        # price just dipped below 200-SMA, golden cross still intact

    # ── Stage 4 ───────────────────────────────────────────────────────
    if not above200 and not golden_x and (slope200 or 0) < 0:
        return '4'

    if not above200 and not golden_x and s200_flat:
        return '4'        # price and MAs all pointing down / flat at lows

    # ── Stage 1 ───────────────────────────────────────────────────────
    return '1'


def _compute(close: pd.Series, rs_pct: float | None) -> dict | None:
    """Compute all stage metrics from a daily close series."""
    if len(close) < MIN_DAYS:
        return None

    s50  = _sma(close, SMA50)
    s150 = _sma(close, SMA150)
    s200 = _sma(close, SMA200)

    if s200.dropna().empty:
        return None

    price = float(close.iloc[-1])
    v50   = float(s50.iloc[-1])
    v150  = float(s150.iloc[-1])
    v200  = float(s200.iloc[-1])

    slope200 = _slope_pct(s200, SLOPE_WINDOW)

    # 52-week metrics
    w52 = close.iloc[-252:] if len(close) >= 252 else close
    high52 = float(w52.max())
    low52  = float(w52.min())
    range52 = high52 - low52
    range_pos = (price - low52) / range52 * 100 if range52 > 0 else 50.0

    pct_from_high52 = (price / high52 - 1) * 100
    pct_from_low52  = (price / low52  - 1) * 100 if low52 > 0 else 0.0

    # Distances from MAs
    pct_vs_s50  = (price / v50  - 1) * 100 if v50  > 0 else np.nan
    pct_vs_s150 = (price / v150 - 1) * 100 if v150 > 0 else np.nan
    pct_vs_s200 = (price / v200 - 1) * 100 if v200 > 0 else np.nan
    pct_s50_vs_s200 = (v50 / v200 - 1) * 100 if v200 > 0 else np.nan

    stage = _classify(price, v50, v150, v200, slope200)

    # Minervini 7 criteria
    c1 = price > v150
    c2 = price > v200
    c3 = v150 > v200
    c4 = (slope200 or 0) > 0
    c5 = pct_from_low52 >= (M_ABOVE_52W_LOW - 1) * 100   # ≥ 30% above 52w low
    c6 = price >= high52 * M_FROM_52W_HIGH                # within 25% of 52w high
    c7 = (rs_pct or 0) >= M_RS_MIN

    minervini_count = sum([c1, c2, c3, c4, c5, c6, c7])

    # Stage score
    stage_score = _STAGE_SCORE.get(stage, 45)

    return {
        'stage':              stage,
        'stage_score':        stage_score,
        'minervini_count':    minervini_count,
        'minervini_c1_above_150': c1,
        'minervini_c2_above_200': c2,
        'minervini_c3_150_gt_200': c3,
        'minervini_c4_200_rising': c4,
        'minervini_c5_30pct_low':  c5,
        'minervini_c6_near_high':  c6,
        'minervini_c7_rs70':       c7,
        # MAs
        'sma50':          round(v50,  2),
        'sma150':         round(v150, 2),
        'sma200':         round(v200, 2),
        'sma200_slope':   round(slope200, 3) if slope200 is not None else np.nan,
        'pct_vs_sma50':   round(pct_vs_s50,  2),
        'pct_vs_sma150':  round(pct_vs_s150, 2),
        'pct_vs_sma200':  round(pct_vs_s200, 2),
        'pct_sma50_vs_sma200': round(pct_s50_vs_s200, 2),
        # 52-week
        'high_52w':       round(high52, 2),
        'low_52w':        round(low52,  2),
        'range_pos_52w':  round(range_pos, 1),
        'pct_from_52w_high': round(pct_from_high52, 2),
        'pct_from_52w_low':  round(pct_from_low52,  2),
        # Boolean flags
        'above_sma50':    price > v50,
        'above_sma150':   price > v150,
        'above_sma200':   price > v200,
        'golden_cross':   v50 > v200,
        'sma150_gt_sma200': v150 > v200,
    }


def run(config: Config, as_of: date | None = None) -> pd.DataFrame:
    """
    Compute Stage Analysis for all GICS industries.

    Loads RS Percentile for criterion 7 (rs_pct_composite ≥ 70).
    Rebuilds synthetic daily indexes from constituent stock files.
    """
    sbi        = pd.read_csv(config.stocks_by_industry_csv)
    industries = pd.read_csv(config.industries_csv)
    ind_keys   = sorted(sbi['industry_key'].dropna().unique())

    cutoff = pd.Timestamp(as_of) if as_of else None
    label  = str(as_of) if as_of else None

    # Load RS percentile for Minervini criterion 7
    if label:
        rsp_path = config.results_dir / f"rs_percentile_{label}.csv"
    else:
        rsp_path = _latest_file(config.results_dir, "rs_percentile_*.csv")
    rs_pct_map: dict[str, float] = {}
    if rsp_path and rsp_path.exists():
        rsp = pd.read_csv(rsp_path).set_index('industry_key')
        rs_pct_map = rsp['rs_pct_composite'].dropna().to_dict()
        logger.info(f"Loaded RS Percentile for Minervini C7: {len(rs_pct_map)} industries")
    else:
        logger.warning("RS Percentile not found — Minervini C7 will be False for all industries")

    # Load all constituent closes once
    all_tickers = sbi['symbol'].dropna().unique().tolist()
    logger.info(f"Loading {len(all_tickers)} constituent closes for Stage Analysis...")
    price_matrix = _load_all_closes(all_tickers, config.daily_dir)
    if price_matrix.empty:
        logger.error("No daily data — check daily_dir")
        return pd.DataFrame()
    if cutoff is not None:
        price_matrix = price_matrix[price_matrix.index <= cutoff]

    records = []
    skipped = []

    for i, ind_key in enumerate(ind_keys):
        if i % 30 == 0 and i > 0:
            logger.info(f"  {i}/{len(ind_keys)} industries processed...")

        members = sbi[sbi['industry_key'] == ind_key]
        tickers = members['symbol'].dropna().tolist()
        weights = {
            row['symbol']: float(row['marketCap'])
            for _, row in members.iterrows()
            if pd.notna(row.get('marketCap')) and row['marketCap'] > 0
        }

        idx_daily = _build_industry_daily_index(tickers, weights, price_matrix)
        if idx_daily is None or len(idx_daily) < MIN_DAYS:
            skipped.append(ind_key)
            continue

        rs_pct = rs_pct_map.get(ind_key)
        metrics = _compute(idx_daily, rs_pct)
        if metrics is None:
            skipped.append(ind_key)
            continue

        meta          = industries[industries['industry_key'] == ind_key]
        sector_name   = meta['sector_name'].iloc[0]   if len(meta) else ''
        industry_name = meta['industry_name'].iloc[0] if len(meta) else ind_key

        records.append({
            'industry_key':  ind_key,
            'sector_name':   sector_name,
            'industry_name': industry_name,
            **metrics,
        })

    if skipped:
        logger.info(f"Skipped {len(skipped)} industries (insufficient data)")

    if not records:
        logger.error("No Stage Analysis records computed")
        return pd.DataFrame()

    df = (pd.DataFrame(records)
            .sort_values(['stage_score', 'minervini_count'], ascending=[False, False])
            .reset_index(drop=True))
    df.insert(0, 'rank', df.index + 1)

    dist = df['stage'].value_counts().to_dict()
    logger.info(f"Stage distribution: {dist}")
    full_s2 = (df['minervini_count'] == 7).sum()
    logger.info(f"Full Minervini Stage 2 (all 7 criteria): {full_s2} industries")

    return df


def save(df: pd.DataFrame, config: Config, as_of: date | None = None) -> tuple[Path, Path]:
    config.setup_dirs()
    label = str(as_of) if as_of else pd.Timestamp.today().strftime('%Y-%m-%d')

    csv_cols = [
        'rank', 'industry_key', 'sector_name', 'industry_name',
        'stage', 'stage_score', 'minervini_count',
        'minervini_c1_above_150', 'minervini_c2_above_200', 'minervini_c3_150_gt_200',
        'minervini_c4_200_rising', 'minervini_c5_30pct_low', 'minervini_c6_near_high',
        'minervini_c7_rs70',
        'pct_vs_sma50', 'pct_vs_sma150', 'pct_vs_sma200',
        'pct_sma50_vs_sma200', 'sma200_slope',
        'range_pos_52w', 'pct_from_52w_high', 'pct_from_52w_low',
        'above_sma50', 'above_sma150', 'above_sma200',
        'golden_cross', 'sma150_gt_sma200',
    ]
    cols_present = [c for c in csv_cols if c in df.columns]

    csv_out = config.results_dir / f"stage_analysis_{label}.csv"
    df[cols_present].to_csv(csv_out, index=False)
    logger.info(f"Saved stage analysis CSV → {csv_out}  ({len(df)} industries)")

    md_out = config.results_dir / f"stage_analysis_{label}.md"
    md_out.write_text(_build_md(df, label), encoding='utf-8')
    logger.info(f"Saved stage analysis MD  → {md_out}")

    return csv_out, md_out


_STAGE_LABEL = {
    '2B': '🟢 Stage 2B — Mid Uptrend',
    '2A': '🟢 Stage 2A — Early Uptrend',
    '2C': '🟡 Stage 2C — Late Uptrend',
    '1':  '⚪ Stage 1  — Basing',
    '3':  '🟠 Stage 3  — Distribution',
    '4':  '🔴 Stage 4  — Decline',
}


def _build_md(df: pd.DataFrame, label: str) -> str:
    lines = []
    lines.append(f"# Stage Analysis — {label}")
    lines.append("")
    lines.append("> Weinstein (1988) Stage 1/2/3/4 · Minervini 7-Criteria Score")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Stage | Count | Description |")
    lines.append("|-------|:-----:|-------------|")
    for s in ['2B', '2A', '2C', '1', '3', '4']:
        n = (df['stage'] == s).sum()
        if n == 0:
            continue
        desc = _STAGE_LABEL[s].split('—')[1].strip()
        lines.append(f"| {_STAGE_LABEL[s].split('—')[0].strip()} | {n} | {desc} |")
    lines.append("")

    full7 = (df['minervini_count'] == 7).sum()
    min5  = ((df['minervini_count'] >= 5) & (df['minervini_count'] < 7)).sum()
    lines.append(f"**Minervini all-7 criteria:** {full7} industries  ")
    lines.append(f"**Minervini 5-6 criteria:** {min5} industries")
    lines.append("")

    # Per-stage detail tables
    for s in ['2B', '2A', '2C', '1', '3', '4']:
        sub = df[df['stage'] == s]
        if sub.empty:
            continue
        lines.append(f"## {_STAGE_LABEL[s]} ({len(sub)})")
        lines.append("")
        lines.append("| Rank | Industry | Sector | M# | vs200% | vs150% | SMA200slope | 52wPos | PctHigh |")
        lines.append("|-----:|----------|--------|:--:|-------:|-------:|:-----------:|:------:|--------:|")
        for _, r in sub.iterrows():
            criteria_str = f"{int(r['minervini_count'])}/7"
            lines.append(
                f"| {int(r['rank'])} | {r['industry_name']} | {r['sector_name']} "
                f"| {criteria_str} "
                f"| {r['pct_vs_sma200']:+.1f}% "
                f"| {r['pct_vs_sma150']:+.1f}% "
                f"| {r['sma200_slope']:+.2f}% "
                f"| {r['range_pos_52w']:.0f}% "
                f"| {r['pct_from_52w_high']:+.1f}% |"
            )
        lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append("**Moving averages:** 50-day, 150-day (Weinstein's 30-week), 200-day (40-week).  ")
    lines.append("**200-SMA slope:** % change over 20 trading days (~4 weeks).  ")
    lines.append("**52-week range position:** 0% = at yearly low, 100% = at yearly high.")
    lines.append("")
    lines.append("**Minervini criteria:** (1) price > 150-SMA, (2) price > 200-SMA, "
                 "(3) 150-SMA > 200-SMA, (4) 200-SMA slope > 0 for ≥4 weeks, "
                 "(5) price ≥ 30% above 52w low, (6) price within 25% of 52w high, "
                 "(7) RS Percentile ≥ 70.")
    lines.append("")
    lines.append("*Sources: Weinstein, S. (1988). Secrets for Profiting in Bull and Bear Markets. "
                 "Minervini, M. (2013). Trade Like a Stock Market Wizard.*")
    lines.append("")
    return '\n'.join(lines)


def print_report(df: pd.DataFrame) -> None:
    DIVIDER = '─' * 96

    for s in ['2B', '2A', '2C', '1', '3', '4']:
        sub = df[df['stage'] == s]
        if sub.empty:
            continue
        print(f"\n{DIVIDER}")
        print(f"  {_STAGE_LABEL[s]}  ({len(sub)} industries)")
        print(DIVIDER)
        print(f"  {'Rnk':>3}  {'Industry':<34} {'M#':>3}  {'vs200':>6}  {'vs150':>6}  "
              f"{'Slp200':>7}  {'52wPos':>6}  {'Hi%':>6}  {'Golden'}")
        print(DIVIDER)
        for _, r in sub.iterrows():
            gc = '✓' if r['golden_cross'] else ' '
            print(
                f"  {int(r['rank']):>3}  {str(r['industry_name']):<34} "
                f"{int(r['minervini_count']):>3}  "
                f"{r['pct_vs_sma200']:>+6.1f}%  "
                f"{r['pct_vs_sma150']:>+6.1f}%  "
                f"{r['sma200_slope']:>+7.2f}%  "
                f"{r['range_pos_52w']:>6.0f}%  "
                f"{r['pct_from_52w_high']:>+6.1f}%  {gc}"
            )
