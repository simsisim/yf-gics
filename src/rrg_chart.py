"""
RRG static chart — saves a PNG of the Relative Rotation Graph.

Quadrant colors match the RRGs-Dashboard convention:
  Leading   (top-right)  — green
  Weakening (bottom-right)— yellow/orange
  Lagging   (bottom-left) — red
  Improving (top-left)    — blue

Tail lines show the last N weeks of RS-Ratio / RS-Momentum movement.
Arrow at the current endpoint indicates direction of travel.
"""

import textwrap
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Quadrant fill colours (light, semi-transparent)
_Q_COLORS = {
    'Leading':   '#d4edda',   # light green
    'Weakening': '#fff3cd',   # light yellow
    'Lagging':   '#f8d7da',   # light red
    'Improving': '#cce5ff',   # light blue
}

# Dot / text colours per quadrant
_DOT_COLORS = {
    'Leading':   '#1a7a3a',
    'Weakening': '#c47a00',
    'Lagging':   '#9e1c26',
    'Improving': '#0055a4',
}

_TAIL_ALPHA   = 0.45
_DOT_SIZE     = 40
_FONT_SIZE    = 6.5
_ARROW_HEAD   = 0.015   # fraction of axis range


def _short_name(name: str, max_len: int = 22) -> str:
    """Abbreviate long industry names for chart readability."""
    abbreviations = {
        'Semiconductor Equipment & Materials': 'Semis Equipment',
        'Electronics & Computer Distribution': 'Elec & Comp Dist',
        'Other Industrial Metals & Mining':    'Other Metals',
        'Drug Manufacturers - Specialty & Generic': 'Drug Mfg Spec',
        'Specialty Industrial Machinery':      'Indus Machinery',
        'Farm & Heavy Construction Machinery': 'Farm/Const Mach',
        'Specialty Business Services':         'Spec Bus Services',
        'Internet Content & Information':      'Internet Content',
        'Utilities - Regulated Electric':      'Util - Electric',
        'Utilities - Regulated Gas':           'Util - Gas',
        'Utilities - Regulated Water':         'Util - Water',
        'Financial Data & Stock Exchanges':    'Fin Data/Exch',
        'Insurance - Property & Casualty':     'Ins P&C',
        'Insurance - Diversified':             'Ins Diversified',
        'Real Estate Services':                'RE Services',
        'Communication Equipment':             'Comm Equipment',
        'Software - Application':              'Software App',
        'Software - Infrastructure':           'Software Infra',
        'Scientific & Technical Instruments':  'Sci & Tech Instr',
        'Medical Instruments & Supplies':      'Med Instruments',
        'Medical Care Facilities':             'Med Care Fac',
        'Pharmaceutical Retailers':            'Pharma Retail',
        'Health Information Services':         'Health Info Svc',
        'Drug Manufacturers - General':        'Drug Mfg Gen',
    }
    name = abbreviations.get(name, name)
    return name if len(name) <= max_len else name[:max_len - 1] + '…'


def plot_all_sectors(
    df: pd.DataFrame,
    out_dir: Path,
    label: str,
    tail_weeks: int = 4,
) -> list[Path]:
    """Generate one PNG per GICS sector plus one summary (all sectors)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # Per-sector charts
    for sector in sorted(df['sector_name'].dropna().unique()):
        sub = df[df['sector_name'] == sector]
        if len(sub) < 2:
            continue
        fname = sector.lower().replace(' ', '_').replace('/', '_') + f"_{label}.png"
        p = plot(sub, out_dir / fname,
                 title=f"RRG — {sector}  [{label}]",
                 tail_weeks=tail_weeks)
        paths.append(p)

    # All-industry summary (no labels — just dots with quadrant colours)
    p = plot(df, out_dir / f"rrg_all_{label}.png",
             title=f"GICS All Industries — vs SPY (weekly)  [{label}]",
             tail_weeks=tail_weeks, show_labels=False)
    paths.append(p)
    return paths


def plot(
    df: pd.DataFrame,
    output_path: Path,
    title: str = 'GICS Industry RRG — vs SPY (weekly)',
    tail_weeks: int = 4,
    sector_filter: str | None = None,
    show_labels: bool = True,
    pad: float | None = None,
) -> Path:
    """
    Render and save the RRG chart as a PNG.

    Args:
        df:            Output DataFrame from rrg_engine.run().
        output_path:   Where to save the PNG.
        title:         Chart title.
        tail_weeks:    How many weeks of tail to draw (must match rrg_engine tail_weeks).
        sector_filter: If set, only plot industries in this sector_name.
        show_labels:   Whether to draw text labels next to each dot.
        pad:           Axis padding beyond the data range (auto if None).
    """
    if sector_filter:
        df = df[df['sector_name'] == sector_filter].copy()
        title = f"{title}\n[Sector: {sector_filter}]"

    if df.empty:
        raise ValueError("No data to plot after filtering.")

    n = len(df)
    figsize = (16, 12) if n <= 20 else (20, 15)

    # ------------------------------------------------------------------ layout
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#f9f9f9')
    ax.set_facecolor('#f9f9f9')

    # Axis range — auto-pad based on data spread, minimum ±0.4 around 100
    all_x = [df['rs_ratio']] + [df[f'rs_ratio_{w}w'].dropna() for w in range(1, tail_weeks + 1) if f'rs_ratio_{w}w' in df.columns]
    all_y = [df['rs_momentum']] + [df[f'rs_momentum_{w}w'].dropna() for w in range(1, tail_weeks + 1) if f'rs_momentum_{w}w' in df.columns]
    x_vals = pd.concat(all_x).dropna()
    y_vals = pd.concat(all_y).dropna()

    _pad = pad if pad is not None else max(0.4, (x_vals.max() - x_vals.min()) * 0.12, (y_vals.max() - y_vals.min()) * 0.12)
    x_min = min(x_vals.min(), 100) - _pad
    x_max = max(x_vals.max(), 100) + _pad
    y_min = min(y_vals.min(), 100) - _pad
    y_max = max(y_vals.max(), 100) + _pad

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # ---------------------------------------------------------------- quadrant fills
    ax.axvspan(100, x_max, ymin=(100 - y_min) / (y_max - y_min), ymax=1,
               color=_Q_COLORS['Leading'],   alpha=0.6, zorder=0)
    ax.axvspan(100, x_max, ymin=0,           ymax=(100 - y_min) / (y_max - y_min),
               color=_Q_COLORS['Weakening'], alpha=0.6, zorder=0)
    ax.axvspan(x_min, 100, ymin=0,           ymax=(100 - y_min) / (y_max - y_min),
               color=_Q_COLORS['Lagging'],   alpha=0.6, zorder=0)
    ax.axvspan(x_min, 100, ymin=(100 - y_min) / (y_max - y_min), ymax=1,
               color=_Q_COLORS['Improving'], alpha=0.6, zorder=0)

    # ---------------------------------------------------------------- grid lines at 100
    ax.axvline(100, color='#888888', linewidth=1.2, linestyle='--', zorder=1)
    ax.axhline(100, color='#888888', linewidth=1.2, linestyle='--', zorder=1)

    # ---------------------------------------------------------------- quadrant labels
    label_kw = dict(fontsize=13, fontweight='bold', alpha=0.25, zorder=2,
                    ha='center', va='center')
    mid_x_r = (100 + x_max) / 2
    mid_x_l = (x_min + 100) / 2
    mid_y_t = (100 + y_max) / 2
    mid_y_b = (y_min + 100) / 2
    ax.text(mid_x_r, mid_y_t, 'LEADING',   color=_DOT_COLORS['Leading'],   **label_kw)
    ax.text(mid_x_r, mid_y_b, 'WEAKENING', color=_DOT_COLORS['Weakening'], **label_kw)
    ax.text(mid_x_l, mid_y_b, 'LAGGING',   color=_DOT_COLORS['Lagging'],   **label_kw)
    ax.text(mid_x_l, mid_y_t, 'IMPROVING', color=_DOT_COLORS['Improving'], **label_kw)

    # ---------------------------------------------------------------- tails + dots
    for _, row in df.iterrows():
        quad  = row['quadrant']
        color = _DOT_COLORS.get(quad, '#555555')

        # Build tail path: oldest → newest
        xs = []
        ys = []
        for w in range(tail_weeks, 0, -1):
            xc = f'rs_ratio_{w}w'
            yc = f'rs_momentum_{w}w'
            if xc in row and yc in row and pd.notna(row[xc]) and pd.notna(row[yc]):
                xs.append(float(row[xc]))
                ys.append(float(row[yc]))
        xs.append(float(row['rs_ratio']))
        ys.append(float(row['rs_momentum']))

        # Draw tail line
        if len(xs) > 1:
            ax.plot(xs, ys, color=color, linewidth=0.9, alpha=_TAIL_ALPHA, zorder=3)
            # Arrow from second-last to last point
            if len(xs) >= 2:
                dx = xs[-1] - xs[-2]
                dy = ys[-1] - ys[-2]
                ax.annotate(
                    '', xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.1),
                    zorder=4,
                )

        # Current dot
        ax.scatter(row['rs_ratio'], row['rs_momentum'],
                   c=color, s=_DOT_SIZE, zorder=5, edgecolors='white', linewidths=0.5)

        # Label
        if show_labels:
            lbl = _short_name(row['industry_name'])
            ax.text(
                float(row['rs_ratio']) + (x_max - x_min) * 0.008,
                float(row['rs_momentum']) + (y_max - y_min) * 0.008,
                lbl,
                fontsize=_FONT_SIZE if n > 20 else 8.5,
                color=color,
                zorder=6,
                clip_on=True,
            )

    # ---------------------------------------------------------------- axes / title
    ax.set_xlabel('RS-Ratio  (relative strength vs SPY)', fontsize=11, labelpad=8)
    ax.set_ylabel('RS-Momentum  (rate of change of RS-Ratio)', fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=14)
    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle=':', linewidth=0.4, alpha=0.5, zorder=0)

    # ---------------------------------------------------------------- legend
    legend_patches = [
        mpatches.Patch(color=_DOT_COLORS[q], label=q) for q in ['Leading', 'Improving', 'Weakening', 'Lagging']
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=9, framealpha=0.85)

    # ---------------------------------------------------------------- industry count annotation
    counts = df['quadrant'].value_counts()
    count_str = '  |  '.join(f"{q}: {counts.get(q, 0)}" for q in ['Leading', 'Improving', 'Weakening', 'Lagging'])
    fig.text(0.5, 0.01, count_str, ha='center', fontsize=9, color='#555555')

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path
