"""
Momentum Screen — Faber Tactical Ranking + Short-Term SCTR.

Combines two analytical layers:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER 1 — FABER MOMENTUM SCREEN + STAGE FILTER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Original Faber rule: hold asset if price > 10-month SMA AND 12-month
relative momentum is in the top quintile of the universe.

Stage analysis (Weinstein/Minervini) adds a second dimension: instead of
a binary above/below 200-SMA flag, the full MA structure (50/150/200-SMA
alignment, slopes, Minervini 7-criteria score) is used to classify each
industry as Stage 1 / 2A / 2B / 2C / 3 / 4.

Six-level signal hierarchy:

  STRONG BUY : top quintile + above 200-SMA + Stage 2B + Minervini ≥ 5
               → ideal setup: strong RS AND full MA structure confirmed
  BUY        : top quintile + above 200-SMA + Stage 2A or 2B
               → standard Faber BUY with positive MA structure
  HOLD       : (quintile 2 + Stage 2A/2B) OR (quintile 1 + Stage 2C)
               → either rank is weaker, or trend is late-stage
  WATCH      : quintile 1-2 + above 200-SMA + Stage 1
               → basing near 200-SMA with strong RS — potential entry on breakout
  CAUTION    : top quintile + below 200-SMA (any stage 3/4)
               → high RS but absolute trend is broken
  EXIT       : quintile 3-5, or Stage 3/4 with quintile ≥ 2

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER 2 — MOSKOWITZ-GRINBLATT INDUSTRY MOMENTUM (1999)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Finding: most individual-stock momentum is explained by industry momentum.
Industries that outperformed over the past 9-12 months continue to do so.

Implementation uses the non-overlapping quarterly excess returns already
computed in rs_percentile (Q1+Q2+Q3 = 9-month, which approximates the
standard "12m skip last 1m" rule used at the industry level):
  mg_9m_score   — Q3+Q2+Q1 excess return (skip most recent quarter)
  mg_rank       — rank of mg_9m_score across all industries (1=best)
  mg_signal     — Leader / Above Average / Average / Below Average / Laggard

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER 3 — SHORT-TERM SCTR  (short-term technical momentum)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The full SCTR uses six indicators spanning long/medium/short horizons.
The short-term SCTR uses only the two short-term components:
  c5 = 3-month ROC (weight 10/25 of short-term total)
  c6 = PPO(12,26,9) histogram slope (weight 15/25)
  sctr_st = rank(c5 * 0.40 + c6 * 0.60)  → 0–99.9

Why this is useful: the short-term SCTR reacts faster than the full SCTR
because it excludes the 9-month and 14-month ROC anchors. When sctr_st
rises sharply while full sctr is still low, an early rotation entry signal
is forming before the full SCTR picks it up.
  st_lead  = sctr_st - sctr_full   (positive = ST leading, early entry)
  st_signal = accelerating / decelerating / flat

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPOSITE SCORE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
momentum_score = rs_pct_composite * 0.40     (IBD/Minervini weighted RS)
              + sctr_full        * 0.30     (full technical rank)
              + sctr_st          * 0.20     (short-term technical)
              + mg_pct           * 0.10     (9m Moskowitz-Grinblatt)

Range: 0–100. Ranked → momentum_pct (1–99 percentile).

Output: results/momentum_screen_YYYY-MM-DD.csv + .md
"""

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from src.rotation_composite import _latest_file

logger = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────────
FABER_TOP_QUINTILE  = 80   # rs_pct_12m ≥ 80 = top quintile
FABER_HOLD_MIN      = 60   # rs_pct_12m ≥ 60 = hold zone

# Short-term SCTR component weights (normalised to sum to 1.0)
ST_W_ROC  = 0.40   # c5: 3-month ROC
ST_W_PPO  = 0.60   # c6: PPO histogram slope


def _rank_to_percentile(series: pd.Series) -> pd.Series:
    ranks = series.rank(pct=True, na_option='keep')
    return (ranks * 98 + 1).round().astype('Int64')


def _quintile(pct: float) -> int:
    """Convert 1-99 percentile to quintile (1=best, 5=worst)."""
    if pd.isna(pct):
        return 5
    p = int(pct)
    if p >= 80: return 1
    if p >= 60: return 2
    if p >= 40: return 3
    if p >= 20: return 4
    return 5


def _faber_signal(quintile: int, above_200: bool, stage: str, minervini: int) -> str:
    """
    Six-level Faber signal with stage (Weinstein/Minervini) as second dimension.

    Stage 2B = fully confirmed uptrend (50>150>200, all rising, within 25% of 52w high).
    Stage 2A = early uptrend (golden cross just formed or forming).
    Stage 2C = late/extended (200-SMA slope flattening).
    Stage 1  = basing (potential future Stage 2A breakout).
    Stage 3  = distribution (topping, 50-SMA crossed below 200-SMA).
    Stage 4  = decline (below declining 200-SMA).
    """
    in_stage2  = stage in ('2A', '2B')
    late       = stage == '2C'
    basing     = stage == '1'

    if quintile == 1 and above_200 and stage == '2B' and minervini >= 5:
        return 'STRONG BUY'
    if quintile == 1 and above_200 and in_stage2:
        return 'BUY'
    if (quintile == 2 and above_200 and in_stage2) or (quintile == 1 and above_200 and late):
        return 'HOLD'
    if quintile <= 2 and basing and above_200:
        return 'WATCH'
    if quintile == 1 and not above_200:
        return 'CAUTION'
    return 'EXIT'


def _mg_signal(pct: float) -> str:
    if pd.isna(pct):
        return 'Unknown'
    p = int(pct)
    if p >= 80: return 'Leader'
    if p >= 60: return 'Above Average'
    if p >= 40: return 'Average'
    if p >= 20: return 'Below Average'
    return 'Laggard'


def _st_signal(st_lead: float) -> str:
    if pd.isna(st_lead):
        return 'flat'
    if st_lead >= 10:
        return 'ST Leading'     # short-term outrunning long-term → early entry
    if st_lead <= -10:
        return 'ST Lagging'     # short-term fading → early warning
    return 'flat'


def _load_stage_data(config: Config, label: str | None) -> pd.DataFrame | None:
    if label:
        p = config.results_dir / f"stage_analysis_{label}.csv"
    else:
        p = _latest_file(config.results_dir, "stage_analysis_*.csv")
    if p is None or not p.exists():
        logger.warning("stage_analysis file not found — run --mode stage first. "
                       "Stage filter will default all to Stage '?' (treated as 2B for backward compat).")
        return None
    df = pd.read_csv(p)[['industry_key', 'stage', 'minervini_count',
                          'stage_score', 'pct_vs_sma200', 'sma200_slope',
                          'range_pos_52w', 'pct_from_52w_high', 'golden_cross']]
    logger.info(f"Loaded stage analysis: {p.name}  ({len(df)} industries)")
    return df.set_index('industry_key')


def run(config: Config, as_of: date | None = None) -> pd.DataFrame:
    """
    Build momentum screen by joining rs_percentile + SCTR component CSVs.
    No price recomputation — all inputs are pre-computed.
    """
    label = str(as_of) if as_of else None

    # ── Load RS Percentile ───────────────────────────────────────────────
    if label:
        rs_path = config.results_dir / f"rs_percentile_{label}.csv"
    else:
        rs_path = _latest_file(config.results_dir, "rs_percentile_*.csv")
    if rs_path is None or not rs_path.exists():
        logger.error("rs_percentile file not found — run --mode rs-percentile first.")
        return pd.DataFrame()
    rs = pd.read_csv(rs_path).set_index('industry_key')
    logger.info(f"Loaded RS Percentile: {rs_path.name}  ({len(rs)} industries)")

    # ── Load SCTR GICS (has c5_3m_roc, c6_ppo_slope, c2_sma200_pct) ────
    sctr_dir = config.results_dir / 'industry_sctr'
    if label:
        sc_path = sctr_dir / f"industry_sctr_gics_{label}.csv"
    else:
        sc_path = _latest_file(sctr_dir, "industry_sctr_gics_*.csv")
    if sc_path is None or not sc_path.exists():
        logger.error("industry_sctr_gics file not found — run --mode industry-gics first.")
        return pd.DataFrame()
    sc = pd.read_csv(sc_path).set_index('industry_key')
    logger.info(f"Loaded SCTR GICS: {sc_path.name}  ({len(sc)} industries)")

    # ── Merge on industry_key ────────────────────────────────────────────
    df = rs[['sector_name', 'industry_name',
             'rs_pct_composite', 'rs_pct_3m', 'rs_pct_6m', 'rs_pct_12m',
             'rs_excess_comp', 'rs_excess_3m', 'rs_excess_12m',
             'rs_q4', 'rs_q3', 'rs_q2', 'rs_q1',
             'rs_line_vs_52w', 'rs_new_high', 'rs_near_new_high']].copy()

    df['sctr']          = sc['sctr'].reindex(df.index)
    df['c5_3m_roc']     = sc['c5_3m_roc'].reindex(df.index)
    df['c6_ppo_slope']  = sc['c6_ppo_slope'].reindex(df.index)
    df['c2_sma200_pct'] = sc['c2_sma200_pct'].reindex(df.index)

    # ── Stage Analysis ───────────────────────────────────────────────────
    stage_df = _load_stage_data(config, label)
    if stage_df is not None:
        df['stage']            = stage_df['stage'].reindex(df.index).fillna('?')
        df['minervini_count']  = stage_df['minervini_count'].reindex(df.index).fillna(0).astype(int)
        df['stage_score']      = stage_df['stage_score'].reindex(df.index)
        df['range_pos_52w']    = stage_df['range_pos_52w'].reindex(df.index)
        df['pct_from_52w_high']= stage_df['pct_from_52w_high'].reindex(df.index)
        df['sma200_slope']     = stage_df['sma200_slope'].reindex(df.index)
    else:
        df['stage']           = '2B'   # fallback: assume stage 2B (preserves old BUY behaviour)
        df['minervini_count'] = 0
        df['stage_score']     = 100
        df['range_pos_52w']   = np.nan
        df['pct_from_52w_high'] = np.nan
        df['sma200_slope']    = np.nan

    # ── Layer 1: Faber + Stage ────────────────────────────────────────────
    df['above_200sma']    = df['c2_sma200_pct'] > 0
    df['faber_quintile']  = df['rs_pct_12m'].apply(_quintile)
    df['faber_signal']    = df.apply(
        lambda r: _faber_signal(
            r['faber_quintile'],
            bool(r['above_200sma']),
            str(r['stage']),
            int(r['minervini_count']),
        ), axis=1
    )

    # ── Layer 2: Moskowitz-Grinblatt (9m = Q1+Q2+Q3, skip most recent Q) ─
    df['mg_9m_score'] = (
        df['rs_q1'].fillna(0) +
        df['rs_q2'].fillna(0) +
        df['rs_q3'].fillna(0)
    )
    mg_valid = df['rs_q1'].notna() & df['rs_q2'].notna() & df['rs_q3'].notna()
    df.loc[~mg_valid, 'mg_9m_score'] = np.nan

    df['mg_pct']    = _rank_to_percentile(df['mg_9m_score'])
    df['mg_rank']   = df['mg_9m_score'].rank(ascending=False, method='min').astype('Int64')
    df['mg_signal'] = df['mg_pct'].apply(_mg_signal)

    # ── Layer 3: Short-term SCTR ─────────────────────────────────────────
    df['sctr_st_raw'] = (
        df['c5_3m_roc'].fillna(0)  * ST_W_ROC +
        df['c6_ppo_slope'].fillna(0) * ST_W_PPO
    )
    # Normalise raw scores to 0-99.9 percentile rank within this universe
    valid_st   = df['c5_3m_roc'].notna() & df['c6_ppo_slope'].notna()
    sctr_st    = _rank_to_percentile(df.loc[valid_st, 'sctr_st_raw'])
    df['sctr_st'] = np.nan
    df.loc[valid_st, 'sctr_st'] = sctr_st.astype(float)

    df['st_lead']   = df['sctr_st'] - df['sctr']
    df['st_signal'] = df['st_lead'].apply(_st_signal)

    # ── Composite momentum score ──────────────────────────────────────────
    rs_w, sc_w, st_w, mg_w = 0.40, 0.30, 0.20, 0.10

    df['momentum_score'] = np.nan
    mask = (df['rs_pct_composite'].notna() &
            df['sctr'].notna() &
            df['sctr_st'].notna() &
            df['mg_pct'].notna())
    df.loc[mask, 'momentum_score'] = (
        df.loc[mask, 'rs_pct_composite'].astype(float) * rs_w +
        df.loc[mask, 'sctr'].astype(float)             * sc_w +
        df.loc[mask, 'sctr_st'].astype(float)          * st_w +
        df.loc[mask, 'mg_pct'].astype(float)           * mg_w
    ).round(1)

    df['momentum_pct'] = _rank_to_percentile(df['momentum_score'])

    # ── Sort and rank ─────────────────────────────────────────────────────
    df = (df.sort_values('momentum_score', ascending=False, na_position='last')
            .reset_index())
    df.insert(0, 'rank', df.index + 1)
    df = df.rename(columns={'index': 'industry_key'})

    signal_counts = df['faber_signal'].value_counts().to_dict()
    mg_counts     = df['mg_signal'].value_counts().to_dict()
    logger.info(f"Faber signals: {signal_counts}")
    logger.info(f"M-G signals:   {mg_counts}")

    return df


def save(df: pd.DataFrame, config: Config, as_of: date | None = None) -> tuple[Path, Path]:
    """Save CSV + markdown report."""
    config.setup_dirs()
    label = str(as_of) if as_of else pd.Timestamp.today().strftime('%Y-%m-%d')

    out_cols = [
        'rank', 'industry_key', 'sector_name', 'industry_name',
        'momentum_score', 'momentum_pct',
        'faber_signal', 'faber_quintile', 'above_200sma',
        'stage', 'minervini_count', 'stage_score',
        'range_pos_52w', 'pct_from_52w_high', 'sma200_slope',
        'rs_pct_composite', 'rs_pct_12m', 'rs_pct_3m',
        'rs_excess_12m', 'rs_excess_3m', 'rs_line_vs_52w', 'rs_new_high',
        'mg_pct', 'mg_rank', 'mg_signal', 'mg_9m_score',
        'sctr', 'sctr_st', 'st_lead', 'st_signal',
        'c2_sma200_pct',
    ]
    cols_present = [c for c in out_cols if c in df.columns]

    csv_out = config.results_dir / f"momentum_screen_{label}.csv"
    df[cols_present].to_csv(csv_out, index=False)
    logger.info(f"Saved momentum screen CSV → {csv_out}  ({len(df)} industries)")

    md_out  = config.results_dir / f"momentum_screen_{label}.md"
    md_out.write_text(_build_md(df, label), encoding='utf-8')
    logger.info(f"Saved momentum screen MD  → {md_out}")

    return csv_out, md_out


_SIG_META = {
    'STRONG BUY': ('🟢', 'STRONG BUY — Stage 2B + Minervini ≥ 5 + Top Quintile'),
    'BUY':        ('🟢', 'BUY — Stage 2A/2B + Top Quintile'),
    'HOLD':       ('🟡', 'HOLD — Quintile 2 Stage 2A/2B, or Quintile 1 Stage 2C'),
    'WATCH':      ('🔵', 'WATCH — Basing (Stage 1) with Strong RS — potential entry on breakout'),
    'CAUTION':    ('⚠️',  'CAUTION — Top Quintile but Absolute Trend Broken'),
    'EXIT':       ('🔴', 'EXIT'),
}

_SIG_ORDER = ['STRONG BUY', 'BUY', 'HOLD', 'WATCH', 'CAUTION']


def _r_int(r, col):
    v = r.get(col)
    return int(v) if pd.notna(v) else '–'


def _build_md(df: pd.DataFrame, label: str) -> str:
    lines = []
    lines.append(f"# Momentum Screen — {label}")
    lines.append("")
    lines.append("> Faber (2007) + Weinstein/Minervini Stage · Moskowitz-Grinblatt (1999) · Short-Term SCTR")
    lines.append("")

    # ── Summary counts ────────────────────────────────────────────────────
    counts = df['faber_signal'].value_counts().to_dict()
    lines.append("## Faber + Stage Signal Summary")
    lines.append("")
    lines.append("| Signal | Count | Criteria |")
    lines.append("|--------|:-----:|----------|")
    for sig, (em, _) in _SIG_META.items():
        n = counts.get(sig, 0)
        if sig == 'STRONG BUY':
            desc = 'Top quintile + Stage 2B + Minervini ≥ 5 + above 200-SMA'
        elif sig == 'BUY':
            desc = 'Top quintile + Stage 2A or 2B + above 200-SMA'
        elif sig == 'HOLD':
            desc = 'Quintile 2 + Stage 2A/2B, OR quintile 1 + Stage 2C'
        elif sig == 'WATCH':
            desc = 'Quintile 1-2 + Stage 1 (basing) + above 200-SMA'
        elif sig == 'CAUTION':
            desc = 'Top quintile + below 200-SMA (Stage 3/4)'
        else:
            desc = 'Quintile 3-5, or Stage 3/4 with quintile ≥ 2'
        lines.append(f"| {em} {sig} | {n} | {desc} |")
    lines.append("")

    # ── Per-signal detail tables ──────────────────────────────────────────
    for sig in _SIG_ORDER:
        em, header = _SIG_META[sig]
        sub = df[df['faber_signal'] == sig]
        if sub.empty:
            continue
        lines.append(f"## {em} {header} ({len(sub)})")
        lines.append("")
        lines.append("| Rank | Industry | Sector | Stage | M# | Mom | RS12 | RS3 | NwHi | M-G | SCTR | ST | STsig |")
        lines.append("|-----:|----------|--------|:-----:|:--:|:---:|:----:|:---:|:----:|:---:|:----:|:--:|-------|")
        for _, r in sub.iterrows():
            nh = '★' if r.get('rs_new_high') else ('◈' if r.get('rs_near_new_high') else '')
            lines.append(
                f"| {int(r['rank'])} | {r['industry_name']} | {r['sector_name']} "
                f"| {r.get('stage', '?')} "
                f"| {_r_int(r, 'minervini_count')}/7 "
                f"| {r['momentum_score']:.0f} "
                f"| {_r_int(r, 'rs_pct_12m')} "
                f"| {_r_int(r, 'rs_pct_3m')} "
                f"| {nh} "
                f"| {_r_int(r, 'mg_pct')} "
                f"| {r['sctr']:.0f} "
                f"| {r['sctr_st']:.0f} "
                f"| {r['st_signal']} |"
            )
        lines.append("")

    # ── Short-term SCTR leaders ───────────────────────────────────────────
    st_leaders = df[df['st_signal'] == 'ST Leading'].sort_values('st_lead', ascending=False).head(15)
    if not st_leaders.empty:
        lines.append("## ⚡ Short-Term SCTR Leaders (ST outrunning full SCTR)")
        lines.append("")
        lines.append("| Rank | Industry | Sector | Stage | ST SCTR | Full SCTR | ST Lead | Faber | RS 3m |")
        lines.append("|-----:|----------|--------|:-----:|:-------:|:---------:|:-------:|-------|:-----:|")
        for _, r in st_leaders.iterrows():
            lines.append(
                f"| {int(r['rank'])} | {r['industry_name']} | {r['sector_name']} "
                f"| {r.get('stage', '?')} "
                f"| {r['sctr_st']:.0f} | {r['sctr']:.0f} "
                f"| +{r['st_lead']:.0f} "
                f"| {r['faber_signal']} "
                f"| {_r_int(r, 'rs_pct_3m')} |"
            )
        lines.append("")

    # ── M-G Leaders ───────────────────────────────────────────────────────
    mg_top = df[df['mg_signal'] == 'Leader'].sort_values('mg_pct', ascending=False).head(15)
    if not mg_top.empty:
        lines.append("## 📊 Moskowitz-Grinblatt Leaders (9-month momentum, skip last quarter)")
        lines.append("")
        lines.append("| Rank | Industry | Sector | Stage | M-G Pct | 9m Score | RS 12m | Faber |")
        lines.append("|-----:|----------|--------|:-----:|:-------:|:--------:|:------:|-------|")
        for _, r in mg_top.iterrows():
            lines.append(
                f"| {int(r['rank'])} | {r['industry_name']} | {r['sector_name']} "
                f"| {r.get('stage', '?')} "
                f"| {_r_int(r, 'mg_pct')} "
                f"| {r['mg_9m_score']:+.1f}% "
                f"| {_r_int(r, 'rs_pct_12m')} "
                f"| {r['faber_signal']} |"
            )
        lines.append("")

    # ── Methodology ───────────────────────────────────────────────────────
    lines.append("## Methodology")
    lines.append("")
    lines.append("**Faber Rank** — 12-month relative performance vs SPY, ranked 1–99. "
                 "Absolute filter: above 200-day SMA (≈ 10-month SMA, Faber 2007).")
    lines.append("")
    lines.append("**Stage Filter (Weinstein/Minervini)** — Stage 2B requires 50>150>200-SMA "
                 "all rising, within 25% of 52w high, and Minervini 7-criteria score. "
                 "Upgrades BUY → STRONG BUY when fully confirmed; demotes late-stage (2C) "
                 "from BUY to HOLD; WATCH for Stage 1 basing with strong RS.")
    lines.append("")
    lines.append("**Moskowitz-Grinblatt (1999)** — 9-month excess return (Q1+Q2+Q3), skipping "
                 "the most recent quarter.")
    lines.append("")
    lines.append("**Short-Term SCTR** — 3-month ROC (40%) + PPO histogram slope (60%). "
                 "Reacts faster than the full SCTR.")
    lines.append("")
    lines.append("**Composite** — rs_pct_composite × 40% + SCTR × 30% + ST SCTR × 20% + "
                 "M-G percentile × 10%")
    lines.append("")
    lines.append("*Sources: Faber (2007) SSRN 962461 · Moskowitz & Grinblatt (1999) JoF 54(4) · "
                 "Weinstein (1988) · Minervini (2013)*")
    lines.append("")

    return '\n'.join(lines)


def print_report(df: pd.DataFrame) -> None:
    DIVIDER = '─' * 110

    for sig in _SIG_ORDER:
        sub = df[df['faber_signal'] == sig]
        if sub.empty:
            continue
        em, header = _SIG_META[sig]
        print(f"\n{DIVIDER}")
        print(f"  {header}  ({len(sub)})")
        print(DIVIDER)
        print(f"  {'Rnk':>3}  {'Industry':<34} {'Stg':>4} {'M#':>3}  {'Mom':>4} "
              f"{'RS12':>5} {'RS3':>4} {'MG':>4} {'SCTR':>5} {'ST':>4} {'STlead':>7}  "
              f"{'STsig':<14} {'NwHi'}")
        print(DIVIDER)
        for _, r in sub.iterrows():
            nh = '★' if r.get('rs_new_high') else ('◈' if r.get('rs_near_new_high') else ' ')
            print(
                f"  {int(r['rank']):>3}  {str(r['industry_name']):<34} "
                f"{str(r.get('stage', '?')):>4}  "
                f"{int(r.get('minervini_count', 0)):>2}/7  "
                f"{r['momentum_score']:>4.0f} "
                f"{int(r['rs_pct_12m']) if pd.notna(r.get('rs_pct_12m')) else 0:>5} "
                f"{int(r['rs_pct_3m']) if pd.notna(r.get('rs_pct_3m')) else 0:>4} "
                f"{int(r['mg_pct']) if pd.notna(r.get('mg_pct')) else 0:>4} "
                f"{r['sctr']:>5.1f} "
                f"{r['sctr_st']:>4.0f} "
                f"{r['st_lead']:>+7.0f}  "
                f"{r['st_signal']:<14} {nh}"
            )

    # Short-term leaders
    st_lead = df[df['st_signal'] == 'ST Leading'].sort_values('st_lead', ascending=False).head(10)
    if not st_lead.empty:
        print(f"\n{DIVIDER}")
        print(f"  Short-Term SCTR Leaders (ST outrunning full SCTR — early entry signal)")
        print(DIVIDER)
        print(f"  {'Rnk':>3}  {'Industry':<34} {'Stg':>4}  {'ST':>4} {'Full':>5} "
              f"{'Lead':>6}  {'Faber':<10} {'RS3m':>4}")
        print(DIVIDER)
        for _, r in st_lead.iterrows():
            print(
                f"  {int(r['rank']):>3}  {str(r['industry_name']):<34} "
                f"{str(r.get('stage', '?')):>4}  "
                f"{r['sctr_st']:>4.0f} {r['sctr']:>5.1f} {r['st_lead']:>+6.0f}  "
                f"{r['faber_signal']:<10} "
                f"{int(r['rs_pct_3m']) if pd.notna(r.get('rs_pct_3m')) else 0:>4}"
            )
    print(f"\n{DIVIDER}")
