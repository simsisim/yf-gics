"""
Rotation Severity Scoring — measures HOW STRONG and HOW CONFIRMED a rotation signal is.

Answers the question: is this rotation just starting, fully confirmed, or running out of steam?

Six components (all scored 0–100):

  1. SCTR Momentum (20%)
       4-week SCTR change (short momentum) + 12-week SCTR change (medium trend)
       Both rising = acceleration, divergence = caution

  2. RRG Quadrant Persistence (18%)
       How long has the industry been in its current quadrant?
       0 weeks in quadrant = just entered (fresh), 4+ weeks = confirmed
       Penalised if it's been weakening/lagging for many weeks

  3. RRG Quadrant Transition (18%)
       What quadrant did it come from, and is the trajectory correct?
       Improving → Leading = highest conviction (clockwise cycle)
       Lagging → Improving = early signal
       Leading → Weakening = early warning

  4. Tail Velocity (17%)
       How fast is the industry moving in (RS-Ratio, RS-Momentum) space?
       Direction: toward Leading quadrant (top-right) = positive velocity
       High speed + correct direction = acceleration signal

  5. MA Structure (12%)
       Synthetic industry index vs 50/200-day SMA
       Above both: full structure
       Recent 50-SMA cross (within 3 weeks): early entry signal

  6. Timeframe Convergence (15%)  ← new
       Weekly RRG quadrant vs Monthly RRG quadrant alignment.
       Both Leading = 100; Pullback in monthly uptrend = 60 (not a real rotation out);
       Both Lagging = 0.
       Key insight from Murphy: if weekly is Weakening but monthly is still Leading,
       the rotation is noise (pullback). Only when both timeframes agree is it structural.

Output: results/rotation_severity_YYYY-MM-DD.csv
"""

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from src.rotation_composite import _CONVERGENCE_LABEL, _TIMEFRAME_SCORE, _latest_file

logger = logging.getLogger(__name__)

# Quadrant traversal order (clockwise = bullish cycle)
_CLOCKWISE = {
    'Lagging':   'Improving',
    'Improving': 'Leading',
    'Leading':   'Weakening',
    'Weakening': 'Lagging',   # back to lagging = bearish
}
_BULLISH_ENTRY = {('Improving', 'Leading'), ('Lagging', 'Improving')}
_BEARISH_ENTRY = {('Leading', 'Weakening'), ('Weakening', 'Lagging')}


def _load_history(config: Config) -> pd.DataFrame | None:
    p = config.results_dir / 'history' / 'industry_history.csv'
    if not p.exists():
        logger.error("No history file found. Run --mode backfill first.")
        return None
    df = pd.read_csv(p, parse_dates=['date'])
    logger.info(f"Loaded history: {len(df)} rows, "
                f"{df['industry_key'].nunique()} industries, "
                f"{df['date'].nunique()} weeks")
    return df


def _load_synthetic_daily_closes(config: Config, industries: list[str]) -> dict[str, pd.Series]:
    """
    Rebuild the synthetic daily close for each industry (needed for MA structure).
    Reuses the same logic as the backfill module.
    """
    from src.backfill import _load_all_closes, _build_industry_daily_index
    from src.data_loader import load_daily

    sbi = pd.read_csv(config.stocks_by_industry_csv)
    all_tickers = sbi[sbi['industry_key'].isin(industries)]['symbol'].dropna().unique().tolist()

    logger.info(f"Loading daily closes for MA structure ({len(all_tickers)} tickers)...")
    price_matrix = _load_all_closes(all_tickers, config.daily_dir)

    closes = {}
    for ind_key in industries:
        members = sbi[sbi['industry_key'] == ind_key]
        tickers = members['symbol'].dropna().tolist()
        weights = {
            row['symbol']: float(row['marketCap'])
            for _, row in members.iterrows()
            if pd.notna(row.get('marketCap')) and row['marketCap'] > 0
        }
        idx = _build_industry_daily_index(tickers, weights, price_matrix)
        if idx is not None:
            closes[ind_key] = idx
    logger.info(f"Synthetic daily closes built: {len(closes)} industries")
    return closes


# ---------------------------------------------------------------------------
# Component scorers (all return float 0–100)
# ---------------------------------------------------------------------------

def _score_sctr_momentum(history: pd.DataFrame, ind_key: str, as_of: pd.Timestamp) -> dict:
    """4-week and 12-week SCTR change → acceleration score."""
    ind_h = history[history['industry_key'] == ind_key].sort_values('date')
    ind_h = ind_h[ind_h['date'] <= as_of]

    if len(ind_h) < 13:
        return {'sctr_4w_change': np.nan, 'sctr_12w_change': np.nan,
                'sctr_trend': 'insufficient', 'sctr_momentum_score': 50.0}

    sctr_now = ind_h['sctr'].iloc[-1]
    sctr_4w  = ind_h['sctr'].iloc[-5]  if len(ind_h) >= 5  else np.nan
    sctr_12w = ind_h['sctr'].iloc[-13] if len(ind_h) >= 13 else np.nan

    d4  = sctr_now - sctr_4w  if not np.isnan(sctr_4w)  else 0
    d12 = sctr_now - sctr_12w if not np.isnan(sctr_12w) else 0

    # Trend classification
    if d4 > 2 and d12 > 5:
        trend = 'accelerating'
    elif d4 > 2 or d12 > 5:
        trend = 'rising'
    elif d4 < -2 and d12 < -5:
        trend = 'decelerating'
    elif d4 < -2 or d12 < -5:
        trend = 'falling'
    else:
        trend = 'flat'

    # Score: map (d4 + d12) to 0–100, centered at 50 for no change
    # d4 range: typically -30..+30; d12 range: typically -50..+50
    combined = np.clip(d4 * 0.6 + d12 * 0.4, -40, 40)
    score    = (combined + 40) / 80 * 100

    return {
        'sctr_now':        round(sctr_now, 1),
        'sctr_4w_change':  round(d4,  1),
        'sctr_12w_change': round(d12, 1),
        'sctr_trend':      trend,
        'sctr_momentum_score': round(score, 1),
    }


def _score_rrg_persistence(history: pd.DataFrame, ind_key: str, as_of: pd.Timestamp) -> dict:
    """Count consecutive weeks in current quadrant → persistence score."""
    ind_h = history[history['industry_key'] == ind_key].sort_values('date')
    ind_h = ind_h[ind_h['date'] <= as_of].dropna(subset=['quadrant'])

    if ind_h.empty:
        return {'quadrant_now': '', 'weeks_in_quadrant': 0, 'persistence_score': 50.0}

    current_quad = ind_h['quadrant'].iloc[-1]
    if not current_quad:
        return {'quadrant_now': '', 'weeks_in_quadrant': 0, 'persistence_score': 50.0}

    # Count consecutive weeks in current quad (going backwards)
    weeks = 1
    for q in reversed(ind_h['quadrant'].iloc[:-1].tolist()):
        if q == current_quad:
            weeks += 1
        else:
            break

    # Score:
    # Leading/Improving: confirmed at 4-6 weeks, mature at 10+ weeks
    # Weakening/Lagging: longer = more bearish conviction
    if current_quad in ('Leading', 'Improving'):
        # Fresh (1-2 wks) = 65, confirmed (3-8 wks) = 85, mature (9+ wks) = 70 (rotation may be old)
        if weeks <= 2:
            score = 65.0
        elif weeks <= 8:
            score = 70.0 + min(weeks - 2, 6) * 2.5   # peaks at 85
        else:
            score = max(85.0 - (weeks - 8) * 2, 50.0)  # slowly fades
    else:
        # Weakening/Lagging: more weeks = more bearish = lower score
        score = max(35.0 - weeks * 2, 10.0)

    return {
        'quadrant_now':      current_quad,
        'weeks_in_quadrant': weeks,
        'persistence_score': round(score, 1),
    }


def _score_rrg_transition(history: pd.DataFrame, ind_key: str, as_of: pd.Timestamp) -> dict:
    """What quadrant did it come from? Clockwise = bullish."""
    ind_h = history[history['industry_key'] == ind_key].sort_values('date')
    ind_h = ind_h[ind_h['date'] <= as_of].dropna(subset=['quadrant'])

    if len(ind_h) < 2:
        return {'prev_quadrant': '', 'transition': 'insufficient', 'transition_score': 50.0}

    quads = ind_h['quadrant'].tolist()
    current = quads[-1]

    # Find the most recent quadrant change
    prev = ''
    for q in reversed(quads[:-1]):
        if q != current:
            prev = q
            break

    if not prev:
        transition = f'stable-{current}'
        score = 70.0 if current in ('Leading', 'Improving') else 30.0
    elif (prev, current) in _BULLISH_ENTRY:
        transition = f'{prev}→{current}'
        score = 90.0 if current == 'Leading' else 75.0
    elif (prev, current) in _BEARISH_ENTRY:
        transition = f'{prev}→{current}'
        score = 15.0 if current == 'Lagging' else 25.0
    elif current in ('Leading', 'Improving'):
        transition = f'{prev}→{current}'
        score = 55.0   # non-clockwise entry — less conviction
    else:
        transition = f'{prev}→{current}'
        score = 35.0

    return {
        'prev_quadrant':   prev,
        'transition':      transition,
        'transition_score': round(score, 1),
    }


def _score_tail_velocity(history: pd.DataFrame, ind_key: str, as_of: pd.Timestamp) -> dict:
    """Compute speed and direction of movement in (RS-Ratio, RS-Momentum) space."""
    ind_h = (
        history[history['industry_key'] == ind_key]
        .sort_values('date')
        .dropna(subset=['rs_ratio', 'rs_momentum'])
    )
    ind_h = ind_h[ind_h['date'] <= as_of]

    if len(ind_h) < 5:
        return {'tail_dx': np.nan, 'tail_dy': np.nan,
                'tail_speed': np.nan, 'tail_toward_leading': False,
                'tail_velocity_score': 50.0}

    x_now  = float(ind_h['rs_ratio'].iloc[-1])
    y_now  = float(ind_h['rs_momentum'].iloc[-1])
    x_4w   = float(ind_h['rs_ratio'].iloc[-5])   if len(ind_h) >= 5 else x_now
    y_4w   = float(ind_h['rs_momentum'].iloc[-5]) if len(ind_h) >= 5 else y_now

    dx    = x_now - x_4w
    dy    = y_now - y_4w
    speed = np.sqrt(dx**2 + dy**2)

    # "Toward leading" = both RS-Ratio increasing AND RS-Momentum increasing
    toward_leading = dx > 0 and dy > 0

    # Dot product with direction vector toward Leading quadrant center
    # Leading center ≈ (101.5, 101.5); current pos → center vector
    target_x, target_y = 101.5, 101.5
    dir_x = target_x - x_now
    dir_y = target_y - y_now
    dot = dx * dir_x + dy * dir_y
    # Normalise dot product into -1..+1
    mag_v   = speed + 1e-9
    mag_dir = np.sqrt(dir_x**2 + dir_y**2) + 1e-9
    cos_sim = dot / (mag_v * mag_dir)

    # Score: speed × direction
    # max typical speed ~0.8, min 0
    speed_score = min(speed / 0.6, 1.0)          # 0–1, saturates at 0.6 speed
    dir_score   = (cos_sim + 1) / 2               # 0–1

    score = (speed_score * 0.5 + dir_score * 0.5) * 100

    return {
        'tail_dx':              round(dx,    3),
        'tail_dy':              round(dy,    3),
        'tail_speed':           round(speed, 3),
        'tail_toward_leading':  toward_leading,
        'tail_velocity_score':  round(score, 1),
    }


def _score_ma_structure(daily_close: pd.Series | None, as_of: pd.Timestamp) -> dict:
    """Above 50/200 SMA? Recent cross?"""
    empty = {'above_50sma': False, 'above_200sma': False,
             'recent_50sma_cross': False, 'pct_from_50sma': np.nan,
             'pct_from_200sma': np.nan, 'ma_score': 50.0}

    if daily_close is None:
        return empty

    d = daily_close[daily_close.index <= as_of]
    if len(d) < 200:
        return empty

    sma50  = d.rolling(50,  min_periods=50).mean()
    sma200 = d.rolling(200, min_periods=200).mean()

    last   = d.iloc[-1]
    s50    = sma50.iloc[-1]
    s200   = sma200.iloc[-1]

    above50  = bool(last > s50)
    above200 = bool(last > s200)

    pct50  = (last / s50  - 1) * 100 if s50  > 0 else np.nan
    pct200 = (last / s200 - 1) * 100 if s200 > 0 else np.nan

    # Recent 50-SMA cross: was it below SMA50 3 weeks ago (15 trading days)?
    if len(d) >= 215 and len(sma50) >= 215:
        was_below = d.iloc[-15] < sma50.iloc[-15]
        recent_cross = above50 and was_below
    else:
        recent_cross = False

    # Score
    if above50 and above200:
        score = 75.0 + min(max(pct50, 0), 10) * 1.5   # bonus for how far above
    elif above50:
        score = 55.0
    elif above200:
        score = 40.0
    else:
        score = max(20.0 + (pct50 if not np.isnan(pct50) else 0), 5.0)

    if recent_cross:
        score = min(score + 10, 100)

    return {
        'above_50sma':        above50,
        'above_200sma':       above200,
        'recent_50sma_cross': recent_cross,
        'pct_from_50sma':     round(pct50,  2) if not np.isnan(pct50)  else np.nan,
        'pct_from_200sma':    round(pct200, 2) if not np.isnan(pct200) else np.nan,
        'ma_score':           round(score,  1),
    }


# ---------------------------------------------------------------------------
# Timeframe convergence scorer (Component 6)
# ---------------------------------------------------------------------------

def _load_ath(config: Config, as_of: date | None) -> pd.DataFrame | None:
    label = str(as_of) if as_of else None
    if label:
        p = config.results_dir / f"ath_monitor_{label}.csv"
    else:
        p = _latest_file(config.results_dir, "ath_monitor_*.csv")
    if p is None or not p.exists():
        logger.warning("ATH monitor file not found — run --mode ath first.")
        return None
    df = pd.read_csv(p)[['industry_key', 'pct_from_ath', 'ath_date',
                          'months_since_ath', 'new_ath', 'ath_signal', 'ath_score']]
    logger.info(f"Loaded ATH monitor: {p.name}  ({len(df)} industries)")
    return df.set_index('industry_key')


def _load_rs_ma(config: Config, as_of: date | None) -> pd.DataFrame | None:
    label = str(as_of) if as_of else None
    if label:
        p = config.results_dir / f"rs_ma_signals_{label}.csv"
    else:
        p = _latest_file(config.results_dir, "rs_ma_signals_*.csv")
    if p is None or not p.exists():
        logger.warning("RS-MA signals file not found — run --mode rs-ma first.")
        return None
    df = pd.read_csv(p)[['industry_key', 'rs_above_ma', 'weeks_above_ma',
                          'rs_cross_up', 'rs_cross_down', 'rs_ma_slope', 'rs_ma_signal']]
    logger.info(f"Loaded RS-MA signals: {p.name}  ({len(df)} industries)")
    return df.set_index('industry_key')


def _load_rs_pct(config: Config, as_of: date | None) -> pd.DataFrame | None:
    label = str(as_of) if as_of else None
    if label:
        p = config.results_dir / f"rs_percentile_{label}.csv"
    else:
        p = _latest_file(config.results_dir, "rs_percentile_*.csv")
    if p is None or not p.exists():
        logger.warning("RS Percentile file not found — run --mode rs-percentile first.")
        return None
    df = pd.read_csv(p)[['industry_key', 'rs_pct_composite', 'rs_pct_3m',
                          'rs_pct_6m', 'rs_pct_12m',
                          'rs_excess_comp', 'rs_line_vs_52w',
                          'rs_new_high', 'rs_near_new_high']]
    logger.info(f"Loaded RS Percentile: {p.name}  ({len(df)} industries)")
    return df.set_index('industry_key')


def _load_stage(config: Config, as_of: date | None) -> pd.DataFrame | None:
    label = str(as_of) if as_of else None
    if label:
        p = config.results_dir / f"stage_analysis_{label}.csv"
    else:
        p = _latest_file(config.results_dir, "stage_analysis_*.csv")
    if p is None or not p.exists():
        logger.warning("Stage analysis file not found — run --mode stage first.")
        return None
    df = pd.read_csv(p)[['industry_key', 'stage', 'stage_score', 'minervini_count',
                          'pct_vs_sma200', 'sma200_slope', 'range_pos_52w',
                          'golden_cross', 'pct_from_52w_high']]
    logger.info(f"Loaded stage analysis: {p.name}  ({len(df)} industries)")
    return df.set_index('industry_key')


def _load_momentum_screen(config: Config, as_of: date | None) -> pd.DataFrame | None:
    label = str(as_of) if as_of else None
    if label:
        p = config.results_dir / f"momentum_screen_{label}.csv"
    else:
        p = _latest_file(config.results_dir, "momentum_screen_*.csv")
    if p is None or not p.exists():
        logger.warning("Momentum screen file not found — run --mode momentum first.")
        return None
    df = pd.read_csv(p)[['industry_key', 'momentum_score', 'momentum_pct',
                          'faber_signal', 'sctr_st', 'st_lead', 'st_signal']]
    logger.info(f"Loaded momentum screen: {p.name}  ({len(df)} industries)")
    return df.set_index('industry_key')


def _load_monthly_rrg(config: Config, as_of: date | None) -> pd.DataFrame | None:
    rrg_dir = config.results_dir / 'rrg'
    label   = str(as_of) if as_of else None
    if label:
        p = rrg_dir / f"rrg_monthly_gics_{label}.csv"
    else:
        p = _latest_file(rrg_dir, "rrg_monthly_gics_*.csv")
    if p is None or not p.exists():
        logger.warning("Monthly RRG not found — run --mode rrg-monthly. Timeframe score = 50.")
        return None
    df = pd.read_csv(p)[['industry_key', 'quadrant_m', 'rs_ratio_m', 'rs_momentum_m']]
    logger.info(f"Loaded monthly RRG: {p.name}  ({len(df)} industries)")
    return df.set_index('industry_key')


def _score_timeframe_convergence(quad_weekly: str, quad_monthly: str) -> dict:
    """Score how well weekly and monthly RRG quadrants agree (0–100)."""
    label = _CONVERGENCE_LABEL.get((quad_weekly, quad_monthly), '')
    score = float(_TIMEFRAME_SCORE.get(label, 50))
    return {'timeframe_label': label, 'timeframe_score': score}


# ---------------------------------------------------------------------------
# Composite severity
# ---------------------------------------------------------------------------

_W = dict(sctr_mom=0.20, persistence=0.18, transition=0.18,
          velocity=0.17, ma=0.12, timeframe=0.15)


def _severity_label(score: float, quad_w: str, quad_m: str) -> str:
    both_up  = quad_w in ('Leading', 'Improving') and quad_m in ('Leading', 'Improving')
    pullback = quad_w in ('Weakening', 'Lagging')  and quad_m in ('Leading', 'Improving')
    both_dn  = quad_w in ('Weakening', 'Lagging')  and quad_m in ('Weakening', 'Lagging')

    if score >= 80 and both_up:
        return 'Both TF Confirmed'
    elif score >= 80 and quad_w in ('Leading', 'Improving'):
        return 'Strong Confirmed'
    elif score >= 65 and quad_w in ('Leading', 'Improving'):
        return 'Confirmed'
    elif score >= 55 and quad_w in ('Leading', 'Improving'):
        return 'Early Signal'
    elif pullback and score >= 50:
        return 'Pullback in Uptrend'     # weekly weak, monthly still leading → noise
    elif score >= 45:
        return 'Neutral / Watch'
    elif score < 30 and both_dn:
        return 'Confirmed Exit'
    elif score < 40 and quad_w in ('Weakening', 'Lagging'):
        return 'Weakening'
    else:
        return 'Neutral / Watch'


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(config: Config, as_of: date | None = None) -> pd.DataFrame:
    """
    Compute rotation severity for all industries.

    Args:
        config: Project Config.
        as_of:  Snapshot date. None = latest date in history.

    Returns:
        DataFrame sorted by severity_score desc, one row per industry.
    """
    history = _load_history(config)
    if history is None:
        return pd.DataFrame()

    as_of_ts = pd.Timestamp(as_of) if as_of else history['date'].max()
    logger.info(f"Computing rotation severity as of {as_of_ts.date()}")

    industries_meta  = pd.read_csv(config.industries_csv).set_index('industry_key')
    monthly_rrg      = _load_monthly_rrg(config, as_of)
    ath_data         = _load_ath(config, as_of)
    rs_ma_data       = _load_rs_ma(config, as_of)
    rs_pct_data      = _load_rs_pct(config, as_of)
    momentum_data    = _load_momentum_screen(config, as_of)
    stage_data       = _load_stage(config, as_of)
    ind_keys         = sorted(history['industry_key'].dropna().unique())

    # Build synthetic daily closes for MA structure (batch load once)
    daily_closes = _load_synthetic_daily_closes(config, ind_keys)

    records = []
    for ind_key in ind_keys:
        sm  = _score_sctr_momentum(history, ind_key, as_of_ts)
        per = _score_rrg_persistence(history, ind_key, as_of_ts)
        tra = _score_rrg_transition(history, ind_key, as_of_ts)
        vel = _score_tail_velocity(history, ind_key, as_of_ts)
        ma  = _score_ma_structure(daily_closes.get(ind_key), as_of_ts)

        # Monthly RRG quadrant for this industry
        quad_m = ''
        rs_r_m = np.nan
        rs_m_m = np.nan
        if monthly_rrg is not None and ind_key in monthly_rrg.index:
            row_m  = monthly_rrg.loc[ind_key]
            quad_m = str(row_m.get('quadrant_m', ''))
            rs_r_m = float(row_m.get('rs_ratio_m', np.nan))
            rs_m_m = float(row_m.get('rs_momentum_m', np.nan))

        tf  = _score_timeframe_convergence(per['quadrant_now'], quad_m)

        composite = (
            _W['sctr_mom']    * sm['sctr_momentum_score']  +
            _W['persistence'] * per['persistence_score']   +
            _W['transition']  * tra['transition_score']    +
            _W['velocity']    * vel['tail_velocity_score'] +
            _W['ma']          * ma['ma_score']             +
            _W['timeframe']   * tf['timeframe_score']
        )

        meta          = industries_meta.loc[ind_key] if ind_key in industries_meta.index else None
        sector_name   = meta['sector_name']   if meta is not None else ''
        industry_name = meta['industry_name'] if meta is not None else ind_key

        # RS-MA data
        rs_ma_signal  = ''
        rs_cross_up_v = False
        rs_cross_dn_v = False
        weeks_above   = 0
        if rs_ma_data is not None and ind_key in rs_ma_data.index:
            rm             = rs_ma_data.loc[ind_key]
            rs_ma_signal   = str(rm.get('rs_ma_signal', ''))
            rs_cross_up_v  = bool(rm.get('rs_cross_up', False))
            rs_cross_dn_v  = bool(rm.get('rs_cross_down', False))
            weeks_above    = int(rm.get('weeks_above_ma', 0))

        # ATH data
        ath_signal  = ''
        ath_score_v = 50.0
        pct_from_ath = np.nan
        if ath_data is not None and ind_key in ath_data.index:
            ar          = ath_data.loc[ind_key]
            ath_signal  = str(ar.get('ath_signal', ''))
            ath_score_v = float(ar.get('ath_score', 50.0))
            pct_from_ath = float(ar.get('pct_from_ath', np.nan))

        # Stage analysis data
        stage_v          = ''
        stage_score_v    = np.nan
        minervini_v      = np.nan
        if stage_data is not None and ind_key in stage_data.index:
            sd            = stage_data.loc[ind_key]
            stage_v       = str(sd.get('stage', ''))
            stage_score_v = float(sd.get('stage_score', np.nan))
            minervini_v   = int(sd.get('minervini_count', 0))

        # Momentum screen data
        momentum_score_v = np.nan
        momentum_pct_v   = np.nan
        faber_signal_v   = ''
        sctr_st_v        = np.nan
        st_lead_v        = np.nan
        st_signal_v      = ''
        if momentum_data is not None and ind_key in momentum_data.index:
            md               = momentum_data.loc[ind_key]
            momentum_score_v = md.get('momentum_score', np.nan)
            momentum_pct_v   = md.get('momentum_pct', np.nan)
            faber_signal_v   = str(md.get('faber_signal', ''))
            sctr_st_v        = md.get('sctr_st', np.nan)
            st_lead_v        = md.get('st_lead', np.nan)
            st_signal_v      = str(md.get('st_signal', ''))

        # RS Percentile data
        rs_pct_comp    = np.nan
        rs_pct_3m_v    = np.nan
        rs_pct_12m_v   = np.nan
        rs_excess_v    = np.nan
        rs_line_52w_v  = np.nan
        rs_new_high_v  = False
        rs_near_high_v = False
        if rs_pct_data is not None and ind_key in rs_pct_data.index:
            rp              = rs_pct_data.loc[ind_key]
            rs_pct_comp     = rp.get('rs_pct_composite', np.nan)
            rs_pct_3m_v     = rp.get('rs_pct_3m', np.nan)
            rs_pct_12m_v    = rp.get('rs_pct_12m', np.nan)
            rs_excess_v     = rp.get('rs_excess_comp', np.nan)
            rs_line_52w_v   = rp.get('rs_line_vs_52w', np.nan)
            rs_new_high_v   = bool(rp.get('rs_new_high', False))
            rs_near_high_v  = bool(rp.get('rs_near_new_high', False))

        records.append({
            'industry_key':       ind_key,
            'sector_name':        sector_name,
            'industry_name':      industry_name,
            # RS-MA
            'rs_ma_signal':       rs_ma_signal,
            'rs_cross_up':        rs_cross_up_v,
            'rs_cross_down':      rs_cross_dn_v,
            'rs_weeks_above_ma':  weeks_above,
            # ATH
            'ath_signal':         ath_signal,
            'ath_score':          ath_score_v,
            'pct_from_ath':       round(pct_from_ath, 2) if not np.isnan(pct_from_ath) else np.nan,
            # Stage analysis (Weinstein / Minervini)
            'stage':              stage_v,
            'stage_score':        stage_score_v,
            'minervini_count':    minervini_v,
            # Momentum screen (Faber + M-G + ST SCTR)
            'momentum_score':     round(float(momentum_score_v), 1) if not np.isnan(momentum_score_v) else np.nan,
            'momentum_pct':       int(momentum_pct_v)  if not np.isnan(momentum_pct_v)  else np.nan,
            'faber_signal':       faber_signal_v,
            'sctr_st':            round(float(sctr_st_v), 1) if not np.isnan(sctr_st_v) else np.nan,
            'st_lead':            round(float(st_lead_v), 1) if not np.isnan(st_lead_v) else np.nan,
            'st_signal':          st_signal_v,
            # RS Percentile (IBD/Minervini)
            'rs_pct_composite':   int(rs_pct_comp)   if not np.isnan(rs_pct_comp)   else np.nan,
            'rs_pct_3m':          int(rs_pct_3m_v)   if not np.isnan(rs_pct_3m_v)   else np.nan,
            'rs_pct_12m':         int(rs_pct_12m_v)  if not np.isnan(rs_pct_12m_v)  else np.nan,
            'rs_excess_comp':     round(rs_excess_v, 1)  if not np.isnan(rs_excess_v)  else np.nan,
            'rs_line_vs_52w':     round(rs_line_52w_v, 1) if not np.isnan(rs_line_52w_v) else np.nan,
            'rs_new_high':        rs_new_high_v,
            'rs_near_new_high':   rs_near_high_v,
            # Weekly RRG
            'quadrant':           per['quadrant_now'],
            'weeks_in_quadrant':  per['weeks_in_quadrant'],
            'prev_quadrant':      tra['prev_quadrant'],
            'transition':         tra['transition'],
            # Monthly RRG
            'quadrant_m':         quad_m,
            'rs_ratio_m':         round(rs_r_m, 3) if not np.isnan(rs_r_m) else np.nan,
            'rs_momentum_m':      round(rs_m_m, 3) if not np.isnan(rs_m_m) else np.nan,
            # Timeframe convergence
            'timeframe_label':    tf['timeframe_label'],
            'timeframe_score':    tf['timeframe_score'],
            # SCTR momentum
            'sctr':               sm.get('sctr_now'),
            'sctr_4w_change':     sm['sctr_4w_change'],
            'sctr_12w_change':    sm['sctr_12w_change'],
            'sctr_trend':         sm['sctr_trend'],
            # Tail velocity
            'tail_dx':            vel['tail_dx'],
            'tail_dy':            vel['tail_dy'],
            'tail_speed':         vel['tail_speed'],
            'toward_leading':     vel['tail_toward_leading'],
            # MA structure
            'above_50sma':        ma['above_50sma'],
            'above_200sma':       ma['above_200sma'],
            'pct_from_50sma':     ma['pct_from_50sma'],
            'pct_from_200sma':    ma['pct_from_200sma'],
            'recent_50sma_cross': ma['recent_50sma_cross'],
            # Component scores
            'score_sctr_mom':     sm['sctr_momentum_score'],
            'score_persistence':  per['persistence_score'],
            'score_transition':   tra['transition_score'],
            'score_velocity':     vel['tail_velocity_score'],
            'score_ma':           ma['ma_score'],
            'score_timeframe':    tf['timeframe_score'],
            # Final
            'severity_score':     round(composite, 1),
        })

    df = pd.DataFrame(records)
    df['severity_label'] = df.apply(
        lambda r: _severity_label(r['severity_score'], r['quadrant'], r['quadrant_m']), axis=1
    )

    # Upgrade label: New ATH
    def _upgrade_for_ath(row: pd.Series) -> str:
        lbl = row['severity_label']
        if row.get('ath_signal') == 'New ATH' and lbl in ('Both TF Confirmed', 'Strong Confirmed', 'Confirmed'):
            return lbl + ' + ATH'
        return lbl

    df['severity_label'] = df.apply(_upgrade_for_ath, axis=1)

    # Upgrade label: RS Percentile ≥ 90 (Minervini elite) + RS New High
    def _upgrade_for_rs_pct(row: pd.Series) -> str:
        lbl = row['severity_label']
        pct = row.get('rs_pct_composite')
        if pd.isna(pct):
            return lbl
        if int(pct) >= 90 and row.get('rs_new_high') and lbl in ('Both TF Confirmed', 'Strong Confirmed', 'Confirmed',
                                                                   'Both TF Confirmed + ATH', 'Strong Confirmed + ATH'):
            return lbl + ' ★RS'   # RS new high on elite performer
        elif int(pct) >= 80 and lbl in ('Both TF Confirmed', 'Strong Confirmed', 'Confirmed',
                                         'Both TF Confirmed + ATH', 'Strong Confirmed + ATH'):
            return lbl + ' (RS≥80)'
        return lbl

    df['severity_label'] = df.apply(_upgrade_for_rs_pct, axis=1)

    df = df.sort_values('severity_score', ascending=False).reset_index(drop=True)
    df.insert(0, 'rank', df.index + 1)

    label_counts = df['severity_label'].value_counts()
    logger.info(f"Severity distribution: {label_counts.to_dict()}")
    return df


def save(df: pd.DataFrame, config: Config, as_of: date | None = None) -> Path:
    config.setup_dirs()
    label = str(as_of) if as_of else pd.Timestamp.today().strftime('%Y-%m-%d')
    out   = config.results_dir / f"rotation_severity_{label}.csv"
    df.to_csv(out, index=False)
    logger.info(f"Saved rotation severity → {out}  ({len(df)} industries)")
    return out
