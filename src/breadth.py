"""
Market Breadth — Internal health of the GICS industry universe.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY BREADTH MATTERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A rising index driven by a handful of mega-caps can hide a sick
market underneath. Breadth measures whether the advance is broad
(healthy) or narrow (fragile). Cross-referencing market_clock
(which only sees SPY/QQQ/IWM) against industry-level internals
resolves the ambiguity:

  market_clock = Uptrend  +  breadth HIGH → confirmed bull, add exposure
  market_clock = Uptrend  +  breadth LOW  → narrow leadership, caution
  market_clock = Rally    +  breadth HIGH → FTD likely to hold
  market_clock = Rally    +  breadth LOW  → suspect rally, wait for confirmation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METRICS  (all computed from pre-existing CSVs)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage breadth     % of industries in Stage 2A or 2B  (Weinstein uptrend)
SMA200 breadth    % above their 200-day SMA
SMA150 breadth    % above their 150-day SMA (Weinstein 30-week)
SMA50 breadth     % above their 50-day SMA
Golden cross      % where 50-SMA > 200-SMA
RS breadth        % with rs_pct_composite ≥ 50 (outperforming SPY)
RS new highs      % with RS line at 52-week high
SCTR breadth      % with SCTR ≥ 60  (technically healthy)
SCTR strong       % with SCTR ≥ 80  (leading)
Faber BUY         % with STRONG BUY or BUY faber signal
ATH breadth       % at or within 5% of all-time high
Minervini avg     average Minervini 7-criteria score across all industries
Minervini≥5       % with ≥ 5/7 Minervini criteria (Stage 2 quality)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPOSITE HEALTH SCORE  (0–100)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Stage 2 breadth    × 25%
  Above 200-SMA      × 20%
  RS breadth ≥ 50    × 20%
  Golden cross       × 20%
  SCTR ≥ 60          × 15%

  80-100 : Bull — broad internal confirmation
  60–79  : Healthy — majority in uptrend, selective caution
  40–59  : Mixed — narrowing leadership, breadth warning
  20–39  : Weak — bearish internals, defensive
   0–19  : Bear — broad distribution / decline

Output: results/breadth_YYYY-MM-DD.csv + .md
"""

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from src.rotation_composite import _latest_file

logger = logging.getLogger(__name__)

# ── Health score weights ───────────────────────────────────────────────────
_W_STAGE2   = 0.25
_W_SMA200   = 0.20
_W_RS50     = 0.20
_W_GOLDEN   = 0.20
_W_SCTR60   = 0.15


def _pct(mask: pd.Series) -> float:
    """Non-NaN fraction → 0-100."""
    valid = mask.dropna()
    if len(valid) == 0:
        return 0.0
    return round(valid.sum() / len(valid) * 100, 1)


# ── Input loading ──────────────────────────────────────────────────────────

def _load_inputs(config: Config, as_of: date | None) -> dict[str, pd.DataFrame]:
    label = str(as_of) if as_of else None

    def latest(pattern):
        if label:
            p = config.results_dir / pattern.replace('*', label)
            if p.exists():
                return p
        return _latest_file(config.results_dir, pattern)

    data: dict[str, pd.DataFrame] = {}

    for key, pattern, cols in [
        ('stage',    'stage_analysis_*.csv',
         ['industry_key', 'sector_name', 'industry_name', 'stage',
          'above_sma50', 'above_sma150', 'above_sma200', 'golden_cross',
          'minervini_count', 'stage_score']),
        ('rs',       'rs_percentile_*.csv',
         ['industry_key', 'sector_name', 'rs_pct_composite',
          'rs_new_high', 'rs_near_new_high']),
        ('momentum', 'momentum_screen_*.csv',
         ['industry_key', 'sector_name', 'faber_signal', 'sctr', 'sctr_st']),
        ('ath',      'ath_monitor_*.csv',
         ['industry_key', 'sector_name', 'ath_signal', 'pct_from_ath']),
        ('mc',       'market_clock_*.csv',
         ['ticker', 'state', 'dd_count_25d', 'last_ftd_date']),
    ]:
        p = latest(pattern)
        if p is None or not p.exists():
            logger.warning(f"Breadth input not found: {pattern}")
            data[key] = pd.DataFrame()
            continue
        df = pd.read_csv(p)
        present = [c for c in cols if c in df.columns]
        data[key] = df[present].copy()
        logger.info(f"Loaded {key}: {p.name}  ({len(df)} rows)")

    return data


# ── Merge into one industry-level frame ───────────────────────────────────

def _merge(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = data.get('stage', pd.DataFrame())
    if base.empty:
        logger.error("stage_analysis CSV required but missing")
        return pd.DataFrame()

    df = base.copy()

    for key, on in [('rs', 'industry_key'), ('momentum', 'industry_key'), ('ath', 'industry_key')]:
        src = data.get(key, pd.DataFrame())
        if src.empty:
            continue
        merge_cols = [c for c in src.columns if c not in df.columns or c == 'industry_key']
        df = df.merge(src[merge_cols], on='industry_key', how='left')

    return df


# ── Overall breadth metrics ────────────────────────────────────────────────

def _compute_overall(df: pd.DataFrame) -> dict:
    n = len(df)

    stage2_mask   = df['stage'].isin(['2A', '2B'])
    stage2b_mask  = df['stage'] == '2B'
    stage34_mask  = df['stage'].isin(['3', '4'])

    m = {
        'n_industries':     n,
        # Stage
        'stage2_pct':       _pct(stage2_mask),
        'stage2b_pct':      _pct(stage2b_mask),
        'stage1_pct':       _pct(df['stage'] == '1'),
        'stage2c_pct':      _pct(df['stage'] == '2C'),
        'stage3_pct':       _pct(df['stage'] == '3'),
        'stage4_pct':       _pct(df['stage'] == '4'),
        # SMA breadth
        'above_sma200_pct': _pct(df.get('above_sma200', pd.Series(dtype=bool))),
        'above_sma150_pct': _pct(df.get('above_sma150', pd.Series(dtype=bool))),
        'above_sma50_pct':  _pct(df.get('above_sma50',  pd.Series(dtype=bool))),
        'golden_cross_pct': _pct(df.get('golden_cross', pd.Series(dtype=bool))),
        # Minervini
        'minervini_avg':    round(df['minervini_count'].mean(), 2) if 'minervini_count' in df else np.nan,
        'minervini5_pct':   _pct(df['minervini_count'] >= 5) if 'minervini_count' in df else 0.0,
        'minervini7_pct':   _pct(df['minervini_count'] == 7) if 'minervini_count' in df else 0.0,
    }

    # RS breadth
    if 'rs_pct_composite' in df.columns:
        m['rs_positive_pct']  = _pct(df['rs_pct_composite'] >= 50)
        m['rs_strong_pct']    = _pct(df['rs_pct_composite'] >= 80)
        m['rs_newhigh_pct']   = _pct(df['rs_new_high'].fillna(False))
        m['rs_nearhigh_pct']  = _pct(df['rs_near_new_high'].fillna(False) | df['rs_new_high'].fillna(False))
    else:
        m.update({'rs_positive_pct': 0, 'rs_strong_pct': 0,
                  'rs_newhigh_pct': 0, 'rs_nearhigh_pct': 0})

    # SCTR breadth
    if 'sctr' in df.columns:
        m['sctr60_pct']  = _pct(df['sctr'] >= 60)
        m['sctr80_pct']  = _pct(df['sctr'] >= 80)
        m['sctr_median'] = round(df['sctr'].median(), 1)
    else:
        m.update({'sctr60_pct': 0, 'sctr80_pct': 0, 'sctr_median': np.nan})

    # Faber
    if 'faber_signal' in df.columns:
        m['faber_strongbuy_pct'] = _pct(df['faber_signal'] == 'STRONG BUY')
        m['faber_buy_pct']       = _pct(df['faber_signal'].isin(['STRONG BUY', 'BUY']))
        m['faber_exit_pct']      = _pct(df['faber_signal'] == 'EXIT')
    else:
        m.update({'faber_strongbuy_pct': 0, 'faber_buy_pct': 0, 'faber_exit_pct': 0})

    # ATH breadth
    if 'pct_from_ath' in df.columns:
        m['near_ath_pct']  = _pct(df['pct_from_ath'] >= -5)    # within 5%
        m['at_ath_pct']    = _pct(df['pct_from_ath'] >= -0.5)  # at ATH
    else:
        m.update({'near_ath_pct': 0, 'at_ath_pct': 0})

    # Composite health score
    m['health_score'] = round(
        m['stage2_pct']       * _W_STAGE2 +
        m['above_sma200_pct'] * _W_SMA200 +
        m['rs_positive_pct']  * _W_RS50   +
        m['golden_cross_pct'] * _W_GOLDEN +
        m['sctr60_pct']       * _W_SCTR60,
        1
    )

    return m


def _health_label(score: float) -> str:
    if score >= 80: return 'Bull Market'
    if score >= 60: return 'Healthy'
    if score >= 40: return 'Mixed'
    if score >= 20: return 'Weak'
    return 'Bear Market'


# ── Sector breakdown ───────────────────────────────────────────────────────

def _compute_by_sector(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for sector, grp in df.groupby('sector_name'):
        n = len(grp)
        rec = {
            'sector_name':      sector,
            'n_industries':     n,
            'stage2_pct':       _pct(grp['stage'].isin(['2A', '2B'])),
            'stage2b_pct':      _pct(grp['stage'] == '2B'),
            'above_sma200_pct': _pct(grp.get('above_sma200', pd.Series(dtype=bool))),
            'golden_cross_pct': _pct(grp.get('golden_cross', pd.Series(dtype=bool))),
        }
        if 'rs_pct_composite' in grp.columns:
            rec['rs_positive_pct'] = _pct(grp['rs_pct_composite'] >= 50)
            rec['rs_strong_pct']   = _pct(grp['rs_pct_composite'] >= 80)
        if 'sctr' in grp.columns:
            rec['sctr60_pct']  = _pct(grp['sctr'] >= 60)
            rec['sctr_median'] = round(grp['sctr'].median(), 1)
        if 'minervini_count' in grp.columns:
            rec['minervini_avg'] = round(grp['minervini_count'].mean(), 1)
        if 'faber_signal' in grp.columns:
            rec['faber_buy_pct'] = _pct(grp['faber_signal'].isin(['STRONG BUY', 'BUY']))

        # Sector health score (same formula)
        rec['health_score'] = round(
            rec['stage2_pct']       * _W_STAGE2 +
            rec['above_sma200_pct'] * _W_SMA200 +
            rec.get('rs_positive_pct', 0) * _W_RS50   +
            rec['golden_cross_pct'] * _W_GOLDEN +
            rec.get('sctr60_pct', 0)      * _W_SCTR60,
            1
        )
        records.append(rec)

    sector_df = (pd.DataFrame(records)
                   .sort_values('health_score', ascending=False)
                   .reset_index(drop=True))
    sector_df.insert(0, 'rank', sector_df.index + 1)
    return sector_df


# ── Historical delta ───────────────────────────────────────────────────────

def _load_prev_breadth(config: Config, current_label: str) -> dict | None:
    """Load the most recent previous breadth CSV (not today's) for delta."""
    files = sorted(config.results_dir.glob("breadth_*.csv"))
    files = [f for f in files if current_label not in f.name]
    if not files:
        return None
    prev = pd.read_csv(files[-1])
    # The summary CSV has metric/value columns
    if 'metric' in prev.columns and 'value' in prev.columns:
        return prev.set_index('metric')['value'].to_dict()
    return None


# ── Main entry point ───────────────────────────────────────────────────────

def run(config: Config, as_of: date | None = None) -> tuple[dict, pd.DataFrame]:
    """
    Compute market breadth from pre-computed CSVs.
    Returns (overall_metrics_dict, sector_df).
    """
    data       = _load_inputs(config, as_of)
    merged     = _merge(data)
    if merged.empty:
        return {}, pd.DataFrame()

    overall    = _compute_overall(merged)
    sector_df  = _compute_by_sector(merged)

    # Attach market_clock context
    mc = data.get('mc', pd.DataFrame())
    if not mc.empty:
        overall['mc_spy_state']  = mc.loc[mc['ticker'] == 'SPY', 'state'].iloc[0]  \
                                   if 'ticker' in mc.columns and (mc['ticker'] == 'SPY').any() else ''
        overall['mc_qqq_state']  = mc.loc[mc['ticker'] == 'QQQ', 'state'].iloc[0]  \
                                   if (mc['ticker'] == 'QQQ').any() else ''
        overall['mc_spy_dds']    = int(mc.loc[mc['ticker'] == 'SPY', 'dd_count_25d'].iloc[0]) \
                                   if (mc['ticker'] == 'SPY').any() else 0

    overall['health_label'] = _health_label(overall['health_score'])
    overall['as_of']        = str(as_of) if as_of else pd.Timestamp.today().strftime('%Y-%m-%d')

    logger.info(f"Health score: {overall['health_score']:.1f}  ({overall['health_label']})")
    logger.info(f"Stage 2 breadth: {overall['stage2_pct']}%  |  "
                f"Above 200-SMA: {overall['above_sma200_pct']}%  |  "
                f"RS positive: {overall['rs_positive_pct']}%")

    return overall, sector_df


# ── Save ───────────────────────────────────────────────────────────────────

def save(
    overall: dict,
    sector_df: pd.DataFrame,
    config: Config,
    as_of: date | None = None,
) -> tuple[Path, Path]:
    config.setup_dirs()
    label = str(as_of) if as_of else pd.Timestamp.today().strftime('%Y-%m-%d')

    # Summary CSV — one row per metric (easy to load for delta comparison)
    summary_rows = [{'metric': k, 'value': v} for k, v in overall.items()]
    csv_out = config.results_dir / f"breadth_{label}.csv"
    pd.DataFrame(summary_rows).to_csv(csv_out, index=False)

    # Sector CSV appended to the same file as a second sheet isn't supported in CSV,
    # so write sector breakdown as breadth_sector_DATE.csv
    if not sector_df.empty:
        sec_out = config.results_dir / f"breadth_sector_{label}.csv"
        sector_df.to_csv(sec_out, index=False)
        logger.info(f"Saved breadth sector CSV → {sec_out}")

    md_out = config.results_dir / f"breadth_{label}.md"
    prev   = _load_prev_breadth(config, label)
    md_out.write_text(_build_md(overall, sector_df, label, prev), encoding='utf-8')

    logger.info(f"Saved breadth CSV → {csv_out}")
    logger.info(f"Saved breadth MD  → {md_out}")
    return csv_out, md_out


# ── Markdown report ────────────────────────────────────────────────────────

def _delta_str(current: float, prev: dict | None, key: str, fmt: str = '.1f') -> str:
    """Return '+X.X' delta string if previous value available, else ''."""
    if prev is None or key not in prev:
        return ''
    try:
        d = current - float(prev[key])
        sign = '+' if d >= 0 else ''
        return f" ({sign}{d:{fmt}})"
    except (TypeError, ValueError):
        return ''


def _build_md(
    m: dict,
    sector_df: pd.DataFrame,
    label: str,
    prev: dict | None,
) -> str:
    hs   = m['health_score']
    hlbl = m['health_label']

    bar_filled = int(hs / 5)
    bar = '█' * bar_filled + '░' * (20 - bar_filled)

    lines = []
    lines.append(f"# Market Breadth — {label}")
    lines.append("")
    lines.append(f"> {m.get('n_industries', '?')} GICS industries  ·  "
                 f"Market Clock: SPY={m.get('mc_spy_state', 'n/a')} "
                 f"({m.get('mc_spy_dds', '?')} DDs)  ·  "
                 f"QQQ={m.get('mc_qqq_state', 'n/a')}")
    lines.append("")

    # Health score banner
    lines.append("## Overall Health Score")
    lines.append("")
    lines.append(f"```")
    lines.append(f"  {hs:5.1f} / 100   {hlbl}")
    lines.append(f"  [{bar}]")
    lines.append(f"```")
    lines.append("")

    # Breadth snapshot table
    def row(label_str, value, key, fmt='.1f', suffix='%'):
        d = _delta_str(value, prev, key, fmt)
        return f"| {label_str} | {value:{fmt}}{suffix}{d} |"

    lines.append("## Breadth Snapshot")
    lines.append("")
    lines.append("| Indicator | Value |")
    lines.append("|-----------|------:|")
    lines.append(row("Stage 2 (2A+2B)",        m['stage2_pct'],       'stage2_pct'))
    lines.append(row("  Stage 2B (mid uptrend)",m['stage2b_pct'],      'stage2b_pct'))
    lines.append(row("  Stage 2A (early)",      m.get('stage2_pct',0) - m.get('stage2b_pct',0), 'stage2a_pct'))
    lines.append(row("Stage 1 (basing)",        m['stage1_pct'],       'stage1_pct'))
    lines.append(row("Stage 3 (distribution)",  m['stage3_pct'],       'stage3_pct'))
    lines.append(row("Stage 4 (decline)",       m['stage4_pct'],       'stage4_pct'))
    lines.append("| | |")
    lines.append(row("Above 200-SMA",           m['above_sma200_pct'], 'above_sma200_pct'))
    lines.append(row("Above 150-SMA",           m['above_sma150_pct'], 'above_sma150_pct'))
    lines.append(row("Above 50-SMA",            m['above_sma50_pct'],  'above_sma50_pct'))
    lines.append(row("Golden Cross (50>200)",   m['golden_cross_pct'], 'golden_cross_pct'))
    lines.append("| | |")
    lines.append(row("RS Composite ≥ 50",       m['rs_positive_pct'],  'rs_positive_pct'))
    lines.append(row("RS Composite ≥ 80",       m['rs_strong_pct'],    'rs_strong_pct'))
    lines.append(row("RS at 52w High",          m['rs_newhigh_pct'],   'rs_newhigh_pct'))
    lines.append("| | |")
    lines.append(row("SCTR ≥ 60",              m['sctr60_pct'],       'sctr60_pct'))
    lines.append(row("SCTR ≥ 80",              m['sctr80_pct'],       'sctr80_pct'))
    lines.append(f"| SCTR Median | {m['sctr_median']:.1f} |")
    lines.append("| | |")
    lines.append(row("Minervini ≥ 5/7",        m['minervini5_pct'],   'minervini5_pct'))
    lines.append(row("Minervini all 7",         m['minervini7_pct'],   'minervini7_pct'))
    lines.append(f"| Minervini Avg Score | {m['minervini_avg']:.2f}/7 |")
    lines.append("| | |")
    lines.append(row("Faber STRONG BUY+BUY",   m['faber_buy_pct'],    'faber_buy_pct'))
    lines.append(row("Faber EXIT",              m['faber_exit_pct'],   'faber_exit_pct'))
    lines.append("| | |")
    lines.append(row("Within 5% of ATH",        m['near_ath_pct'],     'near_ath_pct'))
    lines.append(row("At ATH",                  m['at_ath_pct'],       'at_ath_pct'))
    lines.append("")

    # Market clock interpretation
    mc_state = m.get('mc_spy_state', '')
    lines.append("## Market Clock × Breadth Interpretation")
    lines.append("")
    interp = _interpret(mc_state, hs, m)
    lines.append(interp)
    lines.append("")

    # Sector breakdown
    if not sector_df.empty:
        lines.append("## Sector Breakdown")
        lines.append("")
        cols_md = ['sector_name', 'n_industries', 'health_score',
                   'stage2_pct', 'above_sma200_pct', 'golden_cross_pct',
                   'rs_positive_pct', 'sctr60_pct', 'sctr_median']
        present = [c for c in cols_md if c in sector_df.columns]

        hdr = " | ".join(c.replace('_pct', '%').replace('_', ' ').title() for c in present)
        sep = "|" + "|".join([":---" if i == 0 else "---:" for i in range(len(present))]) + "|"
        lines.append("| " + hdr + " |")
        lines.append(sep)
        for _, r in sector_df.iterrows():
            cells = []
            for c in present:
                v = r[c]
                if c in ('sector_name',):
                    cells.append(str(v))
                elif c in ('n_industries',):
                    cells.append(str(int(v)))
                elif pd.isna(v):
                    cells.append('–')
                else:
                    cells.append(f"{v:.1f}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # Methodology
    lines.append("## Health Score Formula")
    lines.append("")
    lines.append("| Component | Weight | Current |")
    lines.append("|-----------|:------:|--------:|")
    lines.append(f"| Stage 2 breadth (2A+2B) | 25% | {m['stage2_pct']:.1f}% |")
    lines.append(f"| Above 200-SMA | 20% | {m['above_sma200_pct']:.1f}% |")
    lines.append(f"| RS composite ≥ 50 | 20% | {m['rs_positive_pct']:.1f}% |")
    lines.append(f"| Golden cross (50>200) | 20% | {m['golden_cross_pct']:.1f}% |")
    lines.append(f"| SCTR ≥ 60 | 15% | {m['sctr60_pct']:.1f}% |")
    lines.append(f"| **Composite** | **100%** | **{hs:.1f}** |")
    lines.append("")

    return '\n'.join(lines)


def _interpret(mc_state: str, health: float, m: dict) -> str:
    """Generate a plain-English interpretation of clock × breadth."""
    in_uptrend   = 'Uptrend' in mc_state
    in_rally     = 'Rally' in mc_state
    in_correction= 'Correction' in mc_state
    under_stress = 'Stress' in mc_state or 'Pressure' in mc_state

    s2  = m['stage2_pct']
    s4  = m['stage4_pct']
    sma = m['above_sma200_pct']
    rs  = m['rs_positive_pct']
    dds = m.get('mc_spy_dds', 0)

    lines = []

    if in_uptrend and health >= 60:
        lines.append(f"**Confirmed Bull — broad internal support.** "
                     f"{s2:.0f}% of industries are in Stage 2 (uptrend) and "
                     f"{sma:.0f}% are above their 200-SMA. "
                     f"Breadth confirms the market_clock Uptrend reading.")
        if under_stress:
            lines.append(f"  \nDistribution day count is elevated ({dds} DDs). "
                         f"While internals remain healthy, {dds} distribution days "
                         f"warrant tighter stops on extended positions.")

    elif in_uptrend and health < 60:
        lines.append(f"**Narrow leadership — breadth warning.** "
                     f"Market clock shows Uptrend but only {s2:.0f}% of industries "
                     f"are in Stage 2 and only {sma:.0f}% are above their 200-SMA. "
                     f"The index advance is being carried by a small number of leaders. "
                     f"Avoid adding broad exposure; concentrate in STRONG BUY industries.")

    elif in_rally and health >= 55:
        lines.append(f"**Rally with internal support — FTD likely to hold.** "
                     f"{s2:.0f}% of industries in Stage 2 provides a foundation for "
                     f"the current rally attempt. A Follow-Through Day confirmation "
                     f"would be meaningful here.")

    elif in_rally and health < 55:
        lines.append(f"**Suspect rally — thin internal breadth.** "
                     f"Only {s2:.0f}% of industries are in Stage 2 and {rs:.0f}% "
                     f"have positive RS. Rally attempts with low breadth frequently "
                     f"fail. Wait for breadth expansion before adding exposure.")

    elif in_correction:
        lines.append(f"**Correction underway.** "
                     f"Stage 4 industries: {s4:.0f}%. "
                     f"{'Breadth is still elevated — correction may be selective.' if health >= 50 else 'Breadth is weak — broad-based decline.'} "
                     f"Watch for breadth stabilisation (Stage 2 > 50%) before re-entry.")

    else:
        lines.append(f"Market clock state: {mc_state or 'unknown'}. "
                     f"Health score: {health:.1f} ({_health_label(health)}). "
                     f"Stage 2: {s2:.0f}%  |  Above 200-SMA: {sma:.0f}%  |  RS positive: {rs:.0f}%.")

    return '\n'.join(lines)


# ── Terminal report ────────────────────────────────────────────────────────

def print_report(overall: dict, sector_df: pd.DataFrame) -> None:
    hs   = overall['health_score']
    hlbl = overall['health_label']
    n    = overall.get('n_industries', '?')

    bar_filled = int(hs / 5)
    bar = '█' * bar_filled + '░' * (20 - bar_filled)

    DIVIDER = '─' * 72
    print(f"\n{DIVIDER}")
    print(f"  Market Breadth — {overall.get('as_of', '')}  |  {n} GICS industries")
    print(f"  Market Clock: SPY={overall.get('mc_spy_state', 'n/a')}  "
          f"QQQ={overall.get('mc_qqq_state', 'n/a')}  "
          f"DDs={overall.get('mc_spy_dds', '?')}")
    print(DIVIDER)
    print(f"  Health Score  {hs:5.1f} / 100   {hlbl}")
    print(f"  [{bar}]")
    print(DIVIDER)

    rows = [
        ("Stage 2 (2A+2B)",         f"{overall['stage2_pct']:.1f}%"),
        ("  Stage 2B",              f"{overall['stage2b_pct']:.1f}%"),
        ("  Stage 2C (late)",       f"{overall['stage2c_pct']:.1f}%"),
        ("  Stage 1 (basing)",      f"{overall['stage1_pct']:.1f}%"),
        ("  Stage 3/4",             f"{overall['stage3_pct'] + overall['stage4_pct']:.1f}%"),
        ("",                        ""),
        ("Above 200-SMA",           f"{overall['above_sma200_pct']:.1f}%"),
        ("Above 150-SMA",           f"{overall['above_sma150_pct']:.1f}%"),
        ("Above 50-SMA",            f"{overall['above_sma50_pct']:.1f}%"),
        ("Golden Cross (50>200)",   f"{overall['golden_cross_pct']:.1f}%"),
        ("",                        ""),
        ("RS Composite ≥ 50",       f"{overall['rs_positive_pct']:.1f}%"),
        ("RS Composite ≥ 80",       f"{overall['rs_strong_pct']:.1f}%"),
        ("RS at 52w High",          f"{overall['rs_newhigh_pct']:.1f}%"),
        ("",                        ""),
        ("SCTR ≥ 60",              f"{overall['sctr60_pct']:.1f}%"),
        ("SCTR ≥ 80",              f"{overall['sctr80_pct']:.1f}%"),
        ("SCTR Median",            f"{overall['sctr_median']:.1f}"),
        ("",                        ""),
        ("Minervini ≥ 5/7",        f"{overall['minervini5_pct']:.1f}%"),
        ("Minervini Avg Score",    f"{overall['minervini_avg']:.2f}/7"),
        ("",                        ""),
        ("Faber STRONG BUY + BUY", f"{overall['faber_buy_pct']:.1f}%"),
        ("Faber EXIT",             f"{overall['faber_exit_pct']:.1f}%"),
        ("",                        ""),
        ("Within 5% of ATH",       f"{overall['near_ath_pct']:.1f}%"),
        ("At ATH",                 f"{overall['at_ath_pct']:.1f}%"),
    ]
    for lbl, val in rows:
        if not lbl:
            print()
            continue
        print(f"  {lbl:<28} {val:>8}")

    print(f"\n{DIVIDER}")
    print("  Interpretation")
    print(DIVIDER)
    mc_state = overall.get('mc_spy_state', '')
    print(_interpret(mc_state, hs, overall))

    if not sector_df.empty:
        print(f"\n{DIVIDER}")
        print("  Sector Health Scores")
        print(DIVIDER)
        print(f"  {'Sector':<32} {'Score':>6}  {'Stg2%':>6}  {'SMA200':>7}  {'RS+':>5}  {'SCTR60':>7}")
        print(DIVIDER)
        for _, r in sector_df.iterrows():
            print(
                f"  {str(r['sector_name']):<32} {r['health_score']:>6.1f}  "
                f"{r['stage2_pct']:>6.1f}%  "
                f"{r.get('above_sma200_pct', 0):>6.1f}%  "
                f"{r.get('rs_positive_pct', 0):>4.1f}%  "
                f"{r.get('sctr60_pct', 0):>6.1f}%"
            )
    print(f"\n{DIVIDER}")
