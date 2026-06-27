"""
Signal Delta — Change detector for rotation severity output.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PURPOSE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Diffs two consecutive rotation_severity CSVs and surfaces only what
actually moved — avoiding the need to scan 143 rows manually each day.

Each run emits three sections:

  UPGRADES   — industries whose severity score improved most
  DOWNGRADES — industries whose severity score fell most
  CROSSINGS  — threshold transitions that matter regardless of score delta:
               new STRONG BUY, new Stage 2A/2B entry, new Stage 4 drop,
               RRG quadrant changes, Faber signal changes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRACKED FIELDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  severity_label    main signal text label
  severity_score    numeric composite score (float)
  faber_signal      STRONG BUY / BUY / HOLD / WATCH / CAUTION / EXIT
  stage             2B / 2A / 2C / 1 / 3 / 4
  minervini_count   0 – 7
  quadrant          weekly RRG quadrant
  quadrant_m        monthly RRG quadrant
  rs_pct_composite  1 – 99
  rs_pct_3m         1 – 99
  sctr              0 – 99.9
  ath_signal        New ATH / Holding Breakout / Near ATH / …

Output: results/signal_delta_YYYY-MM-DD.csv + .md
"""

import logging
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config

logger = logging.getLogger(__name__)

# ── Ordinal rankings (higher index = more bullish) ─────────────────────────

_SEV_RANK: dict[str, int] = {
    'Confirmed Exit':               0,
    'Weakening':                    1,
    'Neutral / Watch':              2,
    'Pullback in Uptrend':          3,
    'Early Signal':                 4,
    'Confirmed':                    5,
    'Confirmed + ATH':              6,
    'Strong Confirmed':             7,
    'Strong Confirmed + ATH':       8,
    'Both TF Confirmed':            9,
    'Both TF Confirmed + ATH':      10,
}

_FABER_RANK: dict[str, int] = {
    'EXIT': 0, 'CAUTION': 1, 'WATCH': 2, 'HOLD': 3, 'BUY': 4, 'STRONG BUY': 5,
}

_STAGE_RANK: dict[str, int] = {
    '4': 0, '3': 1, '1': 2, '2C': 3, '2A': 4, '2B': 5,
}

_QUAD_RANK: dict[str, int] = {
    'Lagging': 0, 'Weakening': 1, 'Improving': 2, 'Leading': 3,
}

_ATH_RANK: dict[str, int] = {
    'Below ATH': 0, 'Recovering': 1, 'Near ATH': 2,
    'Holding Breakout': 3, 'New ATH': 4,
}


def _sev_rank(label: str) -> int:
    """Rank a severity label, stripping dynamic suffixes like (RS≥80) or ★RS."""
    clean = re.sub(r'\s*[\(★].*$', '', str(label)).strip()
    return _SEV_RANK.get(clean, 2)   # default to Neutral


def _rank(val: str, mapping: dict[str, int], default: int = 0) -> int:
    return mapping.get(str(val), default)


# ── File loading ───────────────────────────────────────────────────────────

def _find_severity_files(config: Config, current_label: str) -> list[Path]:
    """Return all rotation_severity CSVs sorted by date, newest last."""
    return sorted(config.results_dir.glob("rotation_severity_*.csv"))


def _load_severity(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalise: strip whitespace from string columns
    for c in ['severity_label', 'faber_signal', 'stage', 'quadrant', 'quadrant_m', 'ath_signal']:
        if c in df.columns:
            df[c] = df[c].fillna('').astype(str).str.strip()
    return df


# ── Core diff ─────────────────────────────────────────────────────────────

def _diff(curr: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    """
    Merge current and previous severity DataFrames on industry_key.
    Returns one row per industry with delta columns.
    """
    key = 'industry_key'

    # Select columns to carry forward
    track = [
        'severity_label', 'severity_score',
        'faber_signal', 'stage', 'minervini_count',
        'quadrant', 'quadrant_m',
        'rs_pct_composite', 'rs_pct_3m',
        'sctr', 'ath_signal',
    ]
    meta  = ['industry_name', 'sector_name']

    def safe_cols(df, cols):
        return [c for c in cols if c in df.columns]

    curr_cols = [key] + safe_cols(curr, meta) + safe_cols(curr, track)
    prev_cols = [key] + safe_cols(prev, track)

    merged = curr[curr_cols].merge(
        prev[prev_cols], on=key, how='outer', suffixes=('', '_prev')
    )

    # Fill metadata from current side
    for c in meta:
        if c in merged.columns:
            merged[c] = merged[c].fillna('')

    # ── Numeric deltas ─────────────────────────────────────────────────────
    def num_delta(col):
        if col in merged.columns and f'{col}_prev' in merged.columns:
            merged[f'{col}_delta'] = (
                pd.to_numeric(merged[col], errors='coerce') -
                pd.to_numeric(merged[f'{col}_prev'], errors='coerce')
            ).round(2)

    for col in ['severity_score', 'rs_pct_composite', 'rs_pct_3m', 'sctr', 'minervini_count']:
        num_delta(col)

    # ── Ordinal rank deltas ────────────────────────────────────────────────
    def rank_delta(col, mapping, default=0):
        now_col  = col
        prev_col = f'{col}_prev'
        if now_col in merged.columns and prev_col in merged.columns:
            merged[f'{col}_rank_delta'] = (
                merged[now_col].apply(lambda v: _rank(v, mapping, default)) -
                merged[prev_col].apply(lambda v: _rank(v, mapping, default))
            )

    rank_delta('faber_signal',  _FABER_RANK)
    rank_delta('stage',         _STAGE_RANK)
    rank_delta('quadrant',      _QUAD_RANK)
    rank_delta('quadrant_m',    _QUAD_RANK)
    rank_delta('ath_signal',    _ATH_RANK)

    # ── Severity label rank delta ──────────────────────────────────────────
    if 'severity_label' in merged.columns and 'severity_label_prev' in merged.columns:
        merged['sev_rank_delta'] = (
            merged['severity_label'].apply(_sev_rank) -
            merged['severity_label_prev'].apply(_sev_rank)
        )
    else:
        merged['sev_rank_delta'] = 0

    # ── Existence flags ────────────────────────────────────────────────────
    merged['is_new']     = merged['severity_label_prev'].isna() | (merged['severity_label_prev'] == '')
    merged['is_dropped'] = merged['severity_label'].isna()      | (merged['severity_label'] == '')

    # ── Change flag: at least one tracked field changed ───────────────────
    change_cols = [
        'sev_rank_delta', 'faber_signal_rank_delta',
        'stage_rank_delta', 'quadrant_rank_delta',
    ]
    present_change = [c for c in change_cols if c in merged.columns]
    merged['any_change'] = merged[present_change].abs().sum(axis=1) > 0

    # Also flag numeric moves ≥ threshold
    if 'severity_score_delta' in merged.columns:
        merged['any_change'] |= merged['severity_score_delta'].abs() >= 2.0
    if 'rs_pct_composite_delta' in merged.columns:
        merged['any_change'] |= merged['rs_pct_composite_delta'].abs() >= 5.0

    return merged


# ── Change narrative ───────────────────────────────────────────────────────

def _narrative(row: pd.Series) -> str:
    """One-line human-readable summary of what changed for an industry."""
    parts = []

    # Severity label change
    sl_now  = row.get('severity_label', '')
    sl_prev = row.get('severity_label_prev', '')
    if sl_now != sl_prev and sl_prev:
        parts.append(f"Severity: {sl_prev} → {sl_now}")

    # Faber change
    fb_now  = row.get('faber_signal', '')
    fb_prev = row.get('faber_signal_prev', '')
    if fb_now != fb_prev and fb_prev:
        parts.append(f"Faber: {fb_prev} → {fb_now}")

    # Stage change
    st_now  = row.get('stage', '')
    st_prev = row.get('stage_prev', '')
    if st_now != st_prev and st_prev:
        parts.append(f"Stage: {st_prev} → {st_now}")

    # Quadrant change (weekly)
    q_now  = row.get('quadrant', '')
    q_prev = row.get('quadrant_prev', '')
    if q_now != q_prev and q_prev:
        parts.append(f"RRG: {q_prev} → {q_now}")

    # Monthly quadrant change
    qm_now  = row.get('quadrant_m', '')
    qm_prev = row.get('quadrant_m_prev', '')
    if qm_now != qm_prev and qm_prev:
        parts.append(f"RRG-M: {qm_prev} → {qm_now}")

    # RS delta
    rs_d = row.get('rs_pct_composite_delta', np.nan)
    if pd.notna(rs_d) and abs(rs_d) >= 5:
        rs_now = row.get('rs_pct_composite', np.nan)
        rs_pr  = row.get('rs_pct_composite_prev', np.nan)
        parts.append(f"RS: {rs_pr:.0f}→{rs_now:.0f} ({rs_d:+.0f})")

    # Minervini delta
    mc_d = row.get('minervini_count_delta', np.nan)
    if pd.notna(mc_d) and abs(mc_d) >= 1:
        parts.append(f"Minervini: {row.get('minervini_count_prev', '?'):.0f}→"
                     f"{row.get('minervini_count', '?'):.0f}/7 ({mc_d:+.0f})")

    return '  |  '.join(parts) if parts else 'score change only'


# ── Main ──────────────────────────────────────────────────────────────────

def run(
    config: Config,
    as_of: date | None = None,
    prev_label: str | None = None,
) -> pd.DataFrame:
    """
    Diff current vs previous rotation_severity CSV.

    prev_label : explicit previous run date string YYYY-MM-DD.
                 If None, uses the second-most-recent file automatically.

    Returns a DataFrame of changed industries, empty if no previous file found.
    """
    label  = str(as_of) if as_of else None
    files  = sorted(config.results_dir.glob("rotation_severity_*.csv"))

    if not files:
        logger.error("No rotation_severity files found — run --mode severity first")
        return pd.DataFrame()

    # Current file
    if label:
        curr_path = config.results_dir / f"rotation_severity_{label}.csv"
        if not curr_path.exists():
            logger.error(f"rotation_severity_{label}.csv not found")
            return pd.DataFrame()
    else:
        curr_path = files[-1]
        label     = curr_path.stem.replace('rotation_severity_', '')

    # Previous file
    if prev_label:
        prev_path = config.results_dir / f"rotation_severity_{prev_label}.csv"
        if not prev_path.exists():
            logger.error(f"rotation_severity_{prev_label}.csv not found")
            return pd.DataFrame()
    else:
        candidates = [f for f in files if f != curr_path]
        if not candidates:
            logger.warning("Only one severity file found — no previous run to compare against. "
                           "Run --mode severity again after data updates to generate a second snapshot.")
            return pd.DataFrame()
        prev_path = candidates[-1]

    prev_label_str = prev_path.stem.replace('rotation_severity_', '')
    logger.info(f"Diffing  {curr_path.name}  vs  {prev_path.name}")

    curr = _load_severity(curr_path)
    prev = _load_severity(prev_path)

    diff = _diff(curr, prev)

    # Keep only changed rows (exclude new/dropped for main table — handle separately)
    changed = diff[diff['any_change'] & ~diff['is_new'] & ~diff['is_dropped']].copy()

    if changed.empty:
        logger.info("No changes detected between the two runs")
        return pd.DataFrame()

    # Build narrative
    changed['change_summary'] = changed.apply(_narrative, axis=1)
    changed['prev_date']      = prev_label_str
    changed['curr_date']      = label

    # ── Sort key: severity_score_delta descending (upgrades first) ────────
    ssd = 'severity_score_delta'
    if ssd not in changed.columns:
        changed[ssd] = 0.0
    changed = changed.sort_values(ssd, ascending=False).reset_index(drop=True)
    changed.insert(0, 'rank', changed.index + 1)

    n_up   = (changed[ssd] > 0).sum()
    n_dn   = (changed[ssd] < 0).sum()
    n_flat = (changed[ssd] == 0).sum()
    logger.info(f"Changes: {len(changed)} industries  "
                f"({n_up} upgraded, {n_dn} downgraded, {n_flat} lateral)")

    return changed


# ── Save ──────────────────────────────────────────────────────────────────

def save(
    df: pd.DataFrame,
    config: Config,
    as_of: date | None = None,
) -> tuple[Path, Path]:
    config.setup_dirs()
    label = str(as_of) if as_of else pd.Timestamp.today().strftime('%Y-%m-%d')

    csv_cols = [
        'rank', 'industry_key', 'sector_name', 'industry_name',
        'curr_date', 'prev_date',
        'severity_label', 'severity_label_prev', 'severity_score_delta', 'sev_rank_delta',
        'faber_signal', 'faber_signal_prev', 'faber_signal_rank_delta',
        'stage', 'stage_prev', 'stage_rank_delta',
        'minervini_count', 'minervini_count_prev', 'minervini_count_delta',
        'quadrant', 'quadrant_prev', 'quadrant_rank_delta',
        'quadrant_m', 'quadrant_m_prev', 'quadrant_m_rank_delta',
        'rs_pct_composite', 'rs_pct_composite_prev', 'rs_pct_composite_delta',
        'rs_pct_3m', 'rs_pct_3m_prev', 'rs_pct_3m_delta',
        'sctr', 'sctr_prev', 'sctr_delta',
        'ath_signal', 'ath_signal_prev',
        'change_summary',
    ]
    present = [c for c in csv_cols if c in df.columns]

    csv_out = config.results_dir / f"signal_delta_{label}.csv"
    df[present].to_csv(csv_out, index=False)
    logger.info(f"Saved signal delta CSV → {csv_out}  ({len(df)} changes)")

    md_out = config.results_dir / f"signal_delta_{label}.md"
    md_out.write_text(_build_md(df, label), encoding='utf-8')
    logger.info(f"Saved signal delta MD  → {md_out}")

    return csv_out, md_out


# ── Markdown report ────────────────────────────────────────────────────────

_UP_ARROW   = '▲'
_DOWN_ARROW = '▼'
_FLAT_ARROW = '◆'


def _build_md(df: pd.DataFrame, label: str) -> str:
    if df.empty:
        return f"# Signal Delta — {label}\n\n*No changes detected.*\n"

    prev_date = df['prev_date'].iloc[0] if 'prev_date' in df.columns else '?'
    curr_date = df['curr_date'].iloc[0] if 'curr_date' in df.columns else label
    ssd = 'severity_score_delta'

    upgrades   = df[df[ssd] > 0].head(20)
    downgrades = df[df[ssd] < 0].sort_values(ssd).head(20)
    lateral    = df[df[ssd] == 0].head(10)

    lines = []
    lines.append(f"# Signal Delta — {curr_date}")
    lines.append("")
    lines.append(f"> Comparing **{curr_date}** vs **{prev_date}**  ·  "
                 f"{len(df)} industries changed  ·  "
                 f"{(df[ssd] > 0).sum()} upgraded  ·  "
                 f"{(df[ssd] < 0).sum()} downgraded")
    lines.append("")

    def section(title, sub, arrow):
        if sub.empty:
            return
        lines.append(f"## {arrow} {title} ({len(sub)})")
        lines.append("")
        lines.append("| # | Industry | Sector | Δ Score | Severity | Faber | Stage | Changes |")
        lines.append("|--:|----------|--------|:-------:|----------|-------|:-----:|---------|")
        for _, r in sub.iterrows():
            delta    = r.get(ssd, 0)
            sign     = '+' if delta >= 0 else ''
            sev_now  = r.get('severity_label', '')
            sev_prev = r.get('severity_label_prev', '')
            sev_str  = f"{sev_prev} → {sev_now}" if sev_now != sev_prev else sev_now
            fab_now  = r.get('faber_signal', '')
            fab_prev = r.get('faber_signal_prev', '')
            fab_str  = f"{fab_prev}→{fab_now}" if fab_now != fab_prev else fab_now
            stg_now  = r.get('stage', '')
            stg_prev = r.get('stage_prev', '')
            stg_str  = f"{stg_prev}→{stg_now}" if stg_now != stg_prev else stg_now
            lines.append(
                f"| {int(r['rank'])} | {r['industry_name']} | {r['sector_name']} "
                f"| {sign}{delta:.1f} "
                f"| {sev_str} "
                f"| {fab_str} "
                f"| {stg_str} "
                f"| {r.get('change_summary', '')} |"
            )
        lines.append("")

    section("Upgrades",   upgrades,   _UP_ARROW)
    section("Downgrades", downgrades, _DOWN_ARROW)
    if not lateral.empty:
        section("Lateral Changes (label / stage / Faber only)", lateral, _FLAT_ARROW)

    # ── Key threshold crossings ────────────────────────────────────────────
    lines.append("## Key Threshold Crossings")
    lines.append("")

    crossings = {
        "New STRONG BUY":  df[(df.get('faber_signal', '') == 'STRONG BUY') &
                               (df.get('faber_signal_prev', '') != 'STRONG BUY')] if 'faber_signal' in df else pd.DataFrame(),
        "Lost STRONG BUY": df[(df.get('faber_signal', '') != 'STRONG BUY') &
                               (df.get('faber_signal_prev', '') == 'STRONG BUY')] if 'faber_signal' in df else pd.DataFrame(),
        "New Stage 2A/2B": df[df.get('stage', pd.Series()).isin(['2A','2B']) &
                               ~df.get('stage_prev', pd.Series()).isin(['2A','2B'])] if 'stage' in df else pd.DataFrame(),
        "Dropped to Stage 3/4": df[df.get('stage', pd.Series()).isin(['3','4']) &
                                    ~df.get('stage_prev', pd.Series()).isin(['3','4'])] if 'stage' in df else pd.DataFrame(),
        "New Confirmed+":  df[df.get('severity_label', pd.Series()).str.contains('Confirmed', na=False) &
                               ~df.get('severity_label_prev', pd.Series()).str.contains('Confirmed', na=False)] if 'severity_label' in df else pd.DataFrame(),
        "Lost Confirmed":  df[~df.get('severity_label', pd.Series()).str.contains('Confirmed', na=False) &
                                df.get('severity_label_prev', pd.Series()).str.contains('Confirmed', na=False)] if 'severity_label' in df else pd.DataFrame(),
    }

    any_crossing = False
    for label_str, sub in crossings.items():
        if sub.empty:
            continue
        any_crossing = True
        names = ', '.join(sub['industry_name'].tolist())
        lines.append(f"**{label_str}** ({len(sub)}): {names}  ")
    if not any_crossing:
        lines.append("*No major threshold crossings.*")
    lines.append("")

    return '\n'.join(lines)


# ── Terminal report ────────────────────────────────────────────────────────

def print_report(df: pd.DataFrame, top_n: int = 15) -> None:
    DIVIDER = '─' * 100

    if df.empty:
        print("\n  No changes detected between the two runs.")
        return

    prev_date = df['prev_date'].iloc[0] if 'prev_date' in df.columns else '?'
    curr_date = df['curr_date'].iloc[0] if 'curr_date' in df.columns else '?'
    ssd = 'severity_score_delta'

    upgrades   = df[df[ssd] > 0].head(top_n)
    downgrades = df[df[ssd] < 0].sort_values(ssd).head(top_n)

    print(f"\n{DIVIDER}")
    print(f"  Signal Delta  |  {prev_date} → {curr_date}  "
          f"|  {len(df)} changes  "
          f"({(df[ssd]>0).sum()} up  {(df[ssd]<0).sum()} down  {(df[ssd]==0).sum()} lateral)")
    print(DIVIDER)

    HDR = (f"  {'#':>3}  {'Industry':<34} {'ΔScr':>6}  "
           f"{'Severity':<28} {'Faber':<12} {'Stg':>4}  {'Changes'}")

    for subset, arrow, title in [
        (upgrades,   _UP_ARROW,   'Upgrades'),
        (downgrades, _DOWN_ARROW, 'Downgrades'),
    ]:
        if subset.empty:
            continue
        print(f"\n  {arrow} {title} ({len(subset)})")
        print(DIVIDER)
        print(HDR)
        print(DIVIDER)
        for _, r in subset.iterrows():
            delta   = r.get(ssd, 0)
            sign    = '+' if delta >= 0 else ''
            sev_now = r.get('severity_label', '')
            sev_pre = r.get('severity_label_prev', '')
            sev_str = f"{sev_pre[:14]}→{sev_now[:13]}" if sev_now != sev_pre else sev_now[:28]
            fab_now = r.get('faber_signal', '')
            fab_pre = r.get('faber_signal_prev', '')
            fab_str = f"{fab_pre[:5]}→{fab_now[:6]}" if fab_now != fab_pre else fab_now[:12]
            stg_now = r.get('stage', '')
            stg_pre = r.get('stage_prev', '')
            stg_str = f"{stg_pre}→{stg_now}" if stg_now != stg_pre else stg_now
            print(
                f"  {int(r['rank']):>3}  {str(r['industry_name']):<34} "
                f"{sign}{delta:>5.1f}  "
                f"{sev_str:<28} "
                f"{fab_str:<12} "
                f"{stg_str:>4}  "
                f"{str(r.get('change_summary', ''))[:42]}"
            )

    # Key crossings
    print(f"\n{DIVIDER}")
    print(f"  {_FLAT_ARROW} Key Threshold Crossings")
    print(DIVIDER)

    crossings_found = False
    crossing_defs = [
        ("New STRONG BUY",
         lambda d: (d.get('faber_signal', pd.Series(dtype=str)) == 'STRONG BUY') &
                   (d.get('faber_signal_prev', pd.Series(dtype=str)) != 'STRONG BUY')),
        ("Lost STRONG BUY",
         lambda d: (d.get('faber_signal', pd.Series(dtype=str)) != 'STRONG BUY') &
                   (d.get('faber_signal_prev', pd.Series(dtype=str)) == 'STRONG BUY')),
        ("New Stage 2A/2B",
         lambda d: d.get('stage', pd.Series(dtype=str)).isin(['2A','2B']) &
                   ~d.get('stage_prev', pd.Series(dtype=str)).isin(['2A','2B'])),
        ("Dropped Stage 3/4",
         lambda d: d.get('stage', pd.Series(dtype=str)).isin(['3','4']) &
                   ~d.get('stage_prev', pd.Series(dtype=str)).isin(['3','4'])),
        ("New Confirmed",
         lambda d: d.get('severity_label', pd.Series(dtype=str)).str.contains('Confirmed', na=False) &
                   ~d.get('severity_label_prev', pd.Series(dtype=str)).str.contains('Confirmed', na=False)),
        ("Lost Confirmed",
         lambda d: ~d.get('severity_label', pd.Series(dtype=str)).str.contains('Confirmed', na=False) &
                    d.get('severity_label_prev', pd.Series(dtype=str)).str.contains('Confirmed', na=False)),
    ]

    for label_str, mask_fn in crossing_defs:
        try:
            sub = df[mask_fn(df)]
        except Exception:
            continue
        if sub.empty:
            continue
        crossings_found = True
        names = ', '.join(sub['industry_name'].tolist())
        print(f"  {label_str:<22} ({len(sub)})  {names}")

    if not crossings_found:
        print("  No major threshold crossings.")

    print(f"\n{DIVIDER}")
