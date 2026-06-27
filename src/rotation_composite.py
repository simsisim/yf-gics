"""
Rotation Composite — joins SCTR + RRG + Breadth into one ranked signal table.

Each GICS industry gets a composite rotation score (0–100) from:
  - SCTR score         (40%) — absolute technical strength of the synthetic index
  - RRG quadrant       (30%) — relative rotation position vs SPY
  - RRG tail direction (15%) — is momentum accelerating or decelerating
  - RS-Momentum        (15%) — magnitude of momentum vs benchmark

Signal labels:
  Strong Rotation In  : composite > 70, quadrant Leading or Improving
  Rotation In         : composite 55–70, quadrant Leading or Improving
  Neutral             : composite 45–55
  Rotation Out        : composite 30–45, quadrant Weakening or Lagging
  Strong Rotation Out : composite < 30, quadrant Weakening or Lagging

Convergence flags are added to highlight industries where multiple signals agree.
"""

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from config import Config

logger = logging.getLogger(__name__)

_QUAD_SCORE = {'Leading': 100, 'Improving': 67, 'Weakening': 33, 'Lagging': 0}
_TAIL_SCORE = {'right-up': 100, 'right-down': 50, 'left-up': 50,
               'left-down': 0, 'unknown': 50}

_W_SCTR  = 0.40
_W_QUAD  = 0.30
_W_TAIL  = 0.15
_W_RSMOM = 0.15


def _latest_file(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern))
    return files[-1] if files else None


def _load_sctr(config: Config, as_of: date | None) -> pd.DataFrame | None:
    label = str(as_of) if as_of else None
    if label:
        p = config.industry_sctr_dir / f"industry_sctr_gics_{label}.csv"
    else:
        p = _latest_file(config.industry_sctr_dir, "industry_sctr_gics_*.csv")
    if p is None or not p.exists():
        logger.error(f"Industry SCTR file not found. Run --mode industry-gics first.")
        return None
    df = pd.read_csv(p)[['industry_key', 'sector_name', 'industry_name',
                          'constituent_count', 'sctr', 'raw_score']]
    logger.info(f"Loaded SCTR: {p.name} ({len(df)} industries)")
    return df


def _load_rrg(config: Config, as_of: date | None) -> pd.DataFrame | None:
    rrg_dir = config.results_dir / 'rrg'
    label   = str(as_of) if as_of else None
    if label:
        p = rrg_dir / f"rrg_gics_{label}.csv"
    else:
        p = _latest_file(rrg_dir, "rrg_gics_*.csv")
    if p is None or not p.exists():
        logger.error(f"RRG file not found. Run --mode rrg first.")
        return None
    df = pd.read_csv(p)[['industry_key', 'rs_ratio', 'rs_momentum',
                          'quadrant', 'tail_direction']]
    logger.info(f"Loaded RRG: {p.name} ({len(df)} industries)")
    return df


def _load_rrg_monthly(config: Config, as_of: date | None) -> pd.DataFrame | None:
    rrg_dir = config.results_dir / 'rrg'
    label   = str(as_of) if as_of else None
    if label:
        p = rrg_dir / f"rrg_monthly_gics_{label}.csv"
    else:
        p = _latest_file(rrg_dir, "rrg_monthly_gics_*.csv")
    if p is None or not p.exists():
        logger.warning("Monthly RRG file not found — run --mode rrg-monthly to add convergence column.")
        return None
    df = pd.read_csv(p)[['industry_key', 'rs_ratio_m', 'rs_momentum_m',
                          'quadrant_m', 'tail_direction_m']]
    logger.info(f"Loaded monthly RRG: {p.name} ({len(df)} industries)")
    return df


_CONVERGENCE_LABEL = {
    ('Leading',   'Leading'):   'Both Leading',
    ('Improving', 'Leading'):   'Weekly Imp / Mo Leading',
    ('Leading',   'Improving'): 'Weekly Lead / Mo Imp',
    ('Improving', 'Improving'): 'Both Improving',
    ('Weakening', 'Leading'):   'Pullback (Mo Leading)',
    ('Lagging',   'Leading'):   'Deep Pullback (Mo Lead)',
    ('Weakening', 'Improving'): 'Weekly Weak / Mo Imp',
    ('Lagging',   'Improving'): 'Weekly Lag / Mo Imp',
    ('Leading',   'Weakening'): 'Weekly Lead / Mo Weak',
    ('Improving', 'Weakening'): 'Weekly Imp / Mo Weak',
    ('Weakening', 'Weakening'): 'Both Weakening',
    ('Lagging',   'Weakening'): 'Both Bearish',
    ('Leading',   'Lagging'):   'Weekly Lead / Mo Lag',
    ('Improving', 'Lagging'):   'Weekly Imp / Mo Lag',
    ('Weakening', 'Lagging'):   'Both Bearish',
    ('Lagging',   'Lagging'):   'Both Lagging',
}

_TIMEFRAME_SCORE = {
    'Both Leading':              100,
    'Weekly Imp / Mo Leading':    85,
    'Weekly Lead / Mo Imp':       80,
    'Both Improving':             75,
    'Pullback (Mo Leading)':      60,   # key insight: shallow pullback only
    'Deep Pullback (Mo Lead)':    50,
    'Weekly Weak / Mo Imp':       45,
    'Weekly Lag / Mo Imp':        40,
    'Weekly Lead / Mo Weak':      35,
    'Weekly Imp / Mo Weak':       30,
    'Both Weakening':             25,
    'Both Bearish':               10,
    'Weekly Lead / Mo Lag':       30,
    'Weekly Imp / Mo Lag':        25,
    'Both Lagging':                0,
}


def _load_breadth(config: Config, as_of: date | None) -> pd.DataFrame | None:
    label = str(as_of) if as_of else None
    if label:
        p = config.breadth_dir / f"industry_breadth_{label}.csv"
    else:
        p = _latest_file(config.breadth_dir, "industry_breadth_*.csv")
    if p is None or not p.exists():
        logger.warning("Breadth file not found — breadth columns will be empty.")
        return None
    df = pd.read_csv(p)[['industry_key', 'count', 'median_sctr',
                          'pct_above_75', 'pct_above_50', 'pct_below_25']]
    logger.info(f"Loaded breadth: {p.name} ({len(df)} industries)")
    return df


def _rs_momentum_norm(rs_momentum: pd.Series) -> pd.Series:
    """Scale RS-Momentum distance from 100 into a 0–100 score."""
    dist = rs_momentum - 100
    # clip to ±3 range (covers 99%+ of values), then map to 0–100
    dist_clipped = dist.clip(-3, 3)
    return (dist_clipped + 3) / 6 * 100


def _composite_score(df: pd.DataFrame) -> pd.Series:
    sctr_norm  = df['sctr'].clip(0, 99.9) / 99.9 * 100
    quad_score = df['quadrant'].map(_QUAD_SCORE).fillna(50)
    tail_score = df['tail_direction'].map(_TAIL_SCORE).fillna(50)
    rsmom_norm = _rs_momentum_norm(df['rs_momentum'])

    return (
        _W_SCTR  * sctr_norm  +
        _W_QUAD  * quad_score +
        _W_TAIL  * tail_score +
        _W_RSMOM * rsmom_norm
    ).round(1)


def _signal_label(row: pd.Series) -> str:
    c = row['composite']
    q = row['quadrant']
    if c > 70 and q in ('Leading', 'Improving'):
        return 'Strong Rotation In'
    elif c > 55 and q in ('Leading', 'Improving'):
        return 'Rotation In'
    elif c >= 45:
        return 'Neutral'
    elif c >= 30 and q in ('Weakening', 'Lagging'):
        return 'Rotation Out'
    else:
        return 'Strong Rotation Out'


def _convergence_flags(row: pd.Series) -> str:
    """
    Short string summarising how many signals agree.
    Example: 'SCTR↑ RRG↑ BRD↑' means all three point the same direction.
    """
    flags = []

    # SCTR signal
    if row['sctr'] >= 75:
        flags.append('SCTR↑')
    elif row['sctr'] <= 25:
        flags.append('SCTR↓')

    # RRG signal
    if row['quadrant'] in ('Leading', 'Improving'):
        flags.append('RRG↑')
    elif row['quadrant'] in ('Weakening', 'Lagging'):
        flags.append('RRG↓')

    # Tail direction
    td = row.get('tail_direction', '')
    if 'right-up' in td:
        flags.append('TAIL↑')
    elif 'left-down' in td:
        flags.append('TAIL↓')

    # Breadth (if available)
    pct75 = row.get('pct_above_75', None)
    if pd.notna(pct75):
        if pct75 >= 50:
            flags.append('BRD↑')
        elif pct75 <= 20:
            flags.append('BRD↓')

    return ' '.join(flags)


def run(config: Config, as_of: date | None = None) -> pd.DataFrame:
    """
    Build the rotation composite table.

    Returns a DataFrame sorted by composite score (descending) with columns:
        industry_key, sector_name, industry_name, constituent_count,
        sctr, rs_ratio, rs_momentum, quadrant, tail_direction,
        median_sctr, pct_above_75, pct_below_25,
        composite, signal, convergence
    """
    sctr        = _load_sctr(config, as_of)
    rrg         = _load_rrg(config, as_of)
    rrg_monthly = _load_rrg_monthly(config, as_of)
    breadth     = _load_breadth(config, as_of)

    if sctr is None or rrg is None:
        return pd.DataFrame()

    # Join SCTR + weekly RRG
    df = sctr.merge(rrg, on='industry_key', how='inner')
    logger.info(f"After SCTR+RRG join: {len(df)} industries")

    # Optionally join monthly RRG
    if rrg_monthly is not None:
        df = df.merge(rrg_monthly, on='industry_key', how='left')
        df['timeframe_label'] = df.apply(
            lambda r: _CONVERGENCE_LABEL.get(
                (r['quadrant'], r.get('quadrant_m', '')), ''
            ), axis=1
        )
        df['timeframe_score'] = df['timeframe_label'].map(_TIMEFRAME_SCORE).fillna(50)
    else:
        df['quadrant_m']      = None
        df['rs_ratio_m']      = None
        df['rs_momentum_m']   = None
        df['tail_direction_m']= None
        df['timeframe_label'] = ''
        df['timeframe_score'] = 50.0

    # Optionally join breadth
    if breadth is not None:
        df = df.merge(breadth, on='industry_key', how='left')
    else:
        df['median_sctr']  = None
        df['pct_above_75'] = None
        df['pct_above_50'] = None
        df['pct_below_25'] = None
        df['count']        = None

    # Composite score, signal, convergence
    df['composite']   = _composite_score(df)
    df['signal']      = df.apply(_signal_label, axis=1)
    df['convergence'] = df.apply(_convergence_flags, axis=1)

    df = df.sort_values('composite', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1

    col_order = [
        'rank', 'industry_key', 'sector_name', 'industry_name', 'constituent_count',
        'sctr', 'rs_ratio', 'rs_momentum', 'quadrant', 'tail_direction',
        'rs_ratio_m', 'rs_momentum_m', 'quadrant_m', 'tail_direction_m',
        'timeframe_label', 'timeframe_score',
        'median_sctr', 'pct_above_75', 'pct_below_25',
        'composite', 'signal', 'convergence',
    ]
    col_order = [c for c in col_order if c in df.columns]
    return df[col_order]


def save(df: pd.DataFrame, config: Config, as_of: date | None = None) -> Path:
    config.setup_dirs()
    label = str(as_of) if as_of else pd.Timestamp.today().strftime('%Y-%m-%d')
    out   = config.results_dir / f"rotation_composite_{label}.csv"
    df.to_csv(out, index=False)
    logger.info(f"Saved rotation composite → {out}  ({len(df)} industries)")
    return out
