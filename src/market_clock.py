"""
Market Clock — Follow-Through Day & Distribution Day analysis.

Implements William O'Neil's (IBD) market timing methodology for determining
whether the broad market is in a confirmed uptrend, rally attempt, or correction.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MARKET STATES  (in priority order)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Confirmed Uptrend       — FTD occurred, ≤4 DDs in rolling 25 sessions
  Uptrend Under Pressure  — 5 DDs in rolling 25 sessions
  Uptrend Under Stress    — 6+ DDs in rolling 25 sessions (near correction)
  Rally Attempt Day N     — correction low set, Day N of attempt (FTD pending day 4+)
  Correction              — no FTD, or FTD invalidated (rally low undercut)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FOLLOW-THROUGH DAY (FTD)  — O'Neil definition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Occurs on Day 4 or later of a rally attempt (Days 1-3 are unreliable)
  • Index gains ≥ 1.25% from prior close (IBD uses 1.25% for S&P 500 / NASDAQ)
  • Volume is HIGHER than the prior session
  • Optimal window: Days 4–7 (beyond day 10 = lower conviction)
  • FTD is INVALIDATED when index closes below the rally-attempt Day 1 low

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISTRIBUTION DAYS (DD)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Index closes DOWN ≥ 0.2% from prior close
  • Volume is HIGHER than the prior session
  • Counted in a rolling 25-trading-session window only
  • DDs expire automatically after 25 sessions
  • 5 DDs → "Under Pressure"; 6+ DDs → "Under Stress" (near correction)

STALL DAYS (light distribution variant):
  • Index closes flat to slightly up (0% to +0.4%) BUT volume ≥ 1.5× prior session
  • Signals institutions distributing stock into a rising market (selling into buyers)
  • Counts as a distribution day in the IBD count

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RALLY ATTEMPT LOGIC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Day 1: First day index closes ABOVE the prior close after being in a correction.
         The intraday low of Day 1 becomes the "rally-attempt low" — a reference
         level the index must not undercut or the attempt is invalidated.
  Day 2+: Index may pull back but must not close below the Day 1 low.
  Day 4+: FTD window opens. Any day with ≥1.25% gain on higher volume = FTD.

Output: results/market_clock_YYYY-MM-DD.csv
"""

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from src.data_loader import load_daily

logger = logging.getLogger(__name__)

# ── Thresholds (O'Neil / IBD) ──────────────────────────────────────────────
FTD_PRICE_GAIN    = 1.25    # % gain vs prior close required for FTD
FTD_DAY_MIN       = 4       # earliest rally-attempt day a FTD can occur
FTD_DAY_OPTIMAL   = 7       # beyond this day the FTD is less reliable
DD_PRICE_DROP     = -0.20   # % decline to qualify as distribution day
DD_WINDOW         = 25      # rolling session window for DD count
DD_UNDER_PRESSURE = 5       # DDs that trigger "Under Pressure"
DD_UNDER_STRESS   = 6       # DDs that trigger "Under Stress"
STALL_PRICE_MAX   = 0.40    # max gain % to count as a stall day
STALL_VOL_RATIO   = 1.50    # volume must be ≥ this × prior session for stall
CORRECTION_DROP   = 7.0     # % drop from recent peak = "in correction" for state init
LOOKBACK_DAYS     = 300     # days of history to analyse


# ── States ─────────────────────────────────────────────────────────────────
class _State:
    CORRECTION       = 'Correction'
    RALLY            = 'Rally Attempt'   # suffix: ' Day N'
    UPTREND          = 'Confirmed Uptrend'
    UNDER_PRESSURE   = 'Uptrend Under Pressure'
    UNDER_STRESS     = 'Uptrend Under Stress'


def _analyse(df: pd.DataFrame, ticker: str) -> dict:
    """
    Run the O'Neil FTD/DD state machine over a daily OHLCV DataFrame.

    Returns a dict with:
      state, rally_day, rally_day1_low, dd_count_25d, last_ftd_date,
      last_ftd_gain_pct, last_ftd_vol_ratio, last_ftd_day,
      distribution_days   (list of recent DDs),
      follow_through_days (list of all FTDs in the window),
      stall_days          (list of recent stall days),
    """
    df = df.copy().sort_index()
    df['pct_chg']    = df['Close'].pct_change() * 100
    df['vol_ratio']  = df['Volume'] / df['Volume'].shift(1)   # vs PRIOR session

    n = len(df)

    # ── Per-row outputs ──────────────────────────────────────────────────
    states        = [''] * n
    rally_days    = [0] * n
    rally_lows    = [np.nan] * n
    dd_flags      = [False] * n
    stall_flags   = [False] * n
    ftd_flags     = [False] * n

    # ── State machine variables ──────────────────────────────────────────
    state          = _State.CORRECTION
    rally_day      = 0
    rally_day1_low = np.nan
    correction_low = np.nan          # lowest close seen during current correction

    for i in range(1, n):
        row       = df.iloc[i]
        prev      = df.iloc[i - 1]
        pct       = float(row['pct_chg'])
        vol_ratio = float(row['vol_ratio']) if not np.isnan(row['vol_ratio']) else 1.0

        # ── Distribution / Stall detection (valid whenever uptrend) ─────
        is_dd    = False
        is_stall = False

        if state in (_State.UPTREND, _State.UNDER_PRESSURE, _State.UNDER_STRESS):
            # Distribution day: down ≥ 0.2% AND volume > prior session
            if pct <= DD_PRICE_DROP and vol_ratio > 1.0:
                is_dd = True
            # Stall day: nearly flat / slight gain on very heavy volume
            elif 0.0 <= pct <= STALL_PRICE_MAX and vol_ratio >= STALL_VOL_RATIO:
                is_stall = True

        dd_flags[i]    = is_dd
        stall_flags[i] = is_stall

        # ── State transitions ────────────────────────────────────────────
        if state == _State.CORRECTION:
            # Track the lowest close in the correction
            if np.isnan(correction_low) or float(row['Close']) < correction_low:
                correction_low = float(row['Close'])

            # Day 1 of rally attempt: first close UP from prior close
            if pct > 0:
                state          = _State.RALLY
                rally_day      = 1
                rally_day1_low = float(row['Low'])   # intraday low of Day 1
                correction_low = np.nan

        elif state == _State.RALLY:
            # Invalidation: index closes below the Day 1 intraday low
            if float(row['Close']) < rally_day1_low:
                state          = _State.CORRECTION
                rally_day      = 0
                correction_low = min(float(row['Close']),
                                     rally_day1_low if not np.isnan(rally_day1_low) else float(row['Close']))
                rally_day1_low = np.nan
            else:
                rally_day += 1
                # FTD check: Day 4+, gain ≥ 1.25%, volume above prior session
                if rally_day >= FTD_DAY_MIN and pct >= FTD_PRICE_GAIN and vol_ratio > 1.0:
                    ftd_flags[i]  = True
                    # Store the day number BEFORE resetting
                    rally_days[i] = rally_day
                    state         = _State.UPTREND
                    rally_day     = 0
                    rally_day1_low = np.nan
                    states[i]      = state
                    rally_lows[i]  = rally_day1_low
                    continue

        elif state in (_State.UPTREND, _State.UNDER_PRESSURE, _State.UNDER_STRESS):
            # Count DDs in rolling 25-session window
            dd_window_start = max(0, i - DD_WINDOW + 1)
            dd_count = sum(dd_flags[dd_window_start: i + 1]) + sum(stall_flags[dd_window_start: i + 1])

            if dd_count >= DD_UNDER_STRESS:
                state = _State.UNDER_STRESS
            elif dd_count >= DD_UNDER_PRESSURE:
                state = _State.UNDER_PRESSURE
            else:
                state = _State.UPTREND

            # Heavy selloff (>7% from any recent 25-day high) resets to correction
            high_25 = df['Close'].iloc[max(0, i - 24): i + 1].max()
            if float(row['Close']) < high_25 * (1 - CORRECTION_DROP / 100):
                state          = _State.CORRECTION
                rally_day      = 0
                correction_low = float(row['Close'])
                rally_day1_low = np.nan

        states[i]     = state
        rally_days[i] = rally_day
        rally_lows[i] = rally_day1_low

    df['state']          = states
    df['rally_day']      = rally_days
    df['rally_day1_low'] = rally_lows
    df['is_dd']          = dd_flags
    df['is_stall']       = stall_flags
    df['is_ftd']         = ftd_flags

    # ── Rolling 25-session DD count at current date ──────────────────────
    recent = df.tail(DD_WINDOW)
    dd_count_25d = int(recent['is_dd'].sum() + recent['is_stall'].sum())

    # ── Last FTD in the lookback window ─────────────────────────────────
    ftd_rows = df[df['is_ftd']]
    if not ftd_rows.empty:
        last_ftd = ftd_rows.iloc[-1]
        last_ftd_date      = last_ftd.name.date() if hasattr(last_ftd.name, 'date') else last_ftd.name
        last_ftd_gain_pct  = round(float(last_ftd['pct_chg']), 2)
        last_ftd_vol_ratio = round(float(last_ftd['vol_ratio']), 2)
        last_ftd_day       = int(last_ftd['rally_day']) if last_ftd['rally_day'] else None
    else:
        last_ftd_date = last_ftd_gain_pct = last_ftd_vol_ratio = last_ftd_day = None

    # ── Current state (last row) ─────────────────────────────────────────
    cur_state     = df['state'].iloc[-1] or _State.CORRECTION
    cur_rally_day = int(df['rally_day'].iloc[-1])
    cur_date      = df.index[-1].date() if hasattr(df.index[-1], 'date') else df.index[-1]
    cur_close     = float(df['Close'].iloc[-1])

    if cur_state == _State.RALLY and cur_rally_day > 0:
        display_state = f"{_State.RALLY} {cur_rally_day}"
    else:
        display_state = cur_state

    # ── Build signal lists (most-recent first) ───────────────────────────
    dd_list = []
    for ts, row in df[df['is_dd']].iterrows():
        dd_list.append({
            'date':        ts.date() if hasattr(ts, 'date') else ts,
            'pct_chg':     round(float(row['pct_chg']), 2),
            'vol_ratio':   round(float(row['vol_ratio']), 2),
            'close':       round(float(row['Close']), 2),
            'type':        'Distribution',
            'severity':    ('Severe'   if abs(float(row['pct_chg'])) >= 1.0 and float(row['vol_ratio']) >= 1.4
                           else 'Moderate' if abs(float(row['pct_chg'])) >= 0.5 and float(row['vol_ratio']) >= 1.3
                           else 'Mild'),
        })
    dd_list.reverse()

    stall_list = []
    for ts, row in df[df['is_stall']].iterrows():
        stall_list.append({
            'date':      ts.date() if hasattr(ts, 'date') else ts,
            'pct_chg':   round(float(row['pct_chg']), 2),
            'vol_ratio': round(float(row['vol_ratio']), 2),
            'close':     round(float(row['Close']), 2),
            'type':      'Stall',
            'severity':  'Mild',
        })
    stall_list.reverse()

    ftd_list = []
    for ts, row in df[df['is_ftd']].iterrows():
        ftd_list.append({
            'date':      ts.date() if hasattr(ts, 'date') else ts,
            'pct_chg':   round(float(row['pct_chg']), 2),
            'vol_ratio': round(float(row['vol_ratio']), 2),
            'close':     round(float(row['Close']), 2),
            'rally_day': int(row['rally_day']) if row['rally_day'] else None,
            'quality':   ('Strong'   if float(row['pct_chg']) >= 2.0 and float(row['vol_ratio']) >= 1.5
                         else 'Good' if float(row['pct_chg']) >= 1.5
                         else 'Weak'),
        })
    ftd_list.reverse()

    return {
        'ticker':              ticker,
        'as_of_date':          cur_date,
        'close':               cur_close,
        'state':               display_state,
        'raw_state':           cur_state,
        'rally_day':           cur_rally_day,
        'dd_count_25d':        dd_count_25d,
        'dd_under_pressure':   dd_count_25d >= DD_UNDER_PRESSURE,
        'last_ftd_date':       last_ftd_date,
        'last_ftd_gain_pct':   last_ftd_gain_pct,
        'last_ftd_vol_ratio':  last_ftd_vol_ratio,
        'last_ftd_day':        last_ftd_day,
        'follow_through_days': ftd_list,
        'distribution_days':   dd_list,
        'stall_days':          stall_list,
    }


def run(
    config: Config,
    tickers: list[str] | None = None,
    as_of: date | None = None,
) -> pd.DataFrame:
    """
    Run FTD/DD market clock analysis for the specified indexes.

    Args:
        config:  Project Config.
        tickers: List of index tickers. Default: ['SPY', 'QQQ', 'IWM'].
        as_of:   Cutoff date for backtesting. None = latest.

    Returns:
        DataFrame with one row per ticker, plus detail lists attached.
    """
    if tickers is None:
        tickers = ['SPY', 'QQQ', 'IWM']

    cutoff = pd.Timestamp(as_of) if as_of else None
    records = []

    for ticker in tickers:
        daily = load_daily(ticker, config.daily_dir)
        if daily is None or daily.empty:
            logger.warning(f"No daily data for {ticker} — skipping")
            continue

        df = daily.tail(LOOKBACK_DAYS).copy()
        if cutoff is not None:
            df = df[df.index <= cutoff]
        if len(df) < 50:
            logger.warning(f"{ticker}: insufficient data ({len(df)} rows)")
            continue

        result = _analyse(df, ticker)
        records.append(result)
        logger.info(
            f"{ticker}: {result['state']}  "
            f"| DDs(25d)={result['dd_count_25d']}  "
            f"| last FTD={result['last_ftd_date']}  "
            f"| rally_day={result['rally_day']}"
        )

    if not records:
        return pd.DataFrame()

    summary_cols = ['ticker', 'as_of_date', 'close', 'state', 'rally_day',
                    'dd_count_25d', 'dd_under_pressure',
                    'last_ftd_date', 'last_ftd_gain_pct',
                    'last_ftd_vol_ratio', 'last_ftd_day']

    df_out = pd.DataFrame([{k: r[k] for k in summary_cols} for r in records])
    return df_out, records


def save(df: pd.DataFrame, config: Config,
         records: list[dict] | None = None,
         as_of: date | None = None) -> tuple[Path, Path]:
    """Save CSV summary and markdown report. Returns (csv_path, md_path)."""
    config.setup_dirs()
    label = str(as_of) if as_of else pd.Timestamp.today().strftime('%Y-%m-%d')

    csv_out = config.results_dir / f"market_clock_{label}.csv"
    df.to_csv(csv_out, index=False)
    logger.info(f"Saved market clock CSV → {csv_out}  ({len(df)} indexes)")

    md_out = config.results_dir / f"market_clock_{label}.md"
    if records:
        md_text = _build_report(records, as_of=as_of)
        md_out.write_text(md_text, encoding='utf-8')
        logger.info(f"Saved market clock MD  → {md_out}")

    return csv_out, md_out


def _state_badge(state: str) -> str:
    """Return a short markdown badge label for a market state."""
    badges = {
        _State.UPTREND:        '🟢 Confirmed Uptrend',
        _State.UNDER_PRESSURE: '🟡 Uptrend Under Pressure',
        _State.UNDER_STRESS:   '🔴 Uptrend Under Stress',
        _State.CORRECTION:     '🔴 Correction',
    }
    for key, badge in badges.items():
        if state.startswith(key):
            return badge
    if state.startswith(_State.RALLY):
        return f'🟡 {state}'
    return state


def _build_report(records: list[dict], as_of: date | None = None) -> str:
    """Build the full markdown report string."""
    today = str(as_of) if as_of else str(records[0]['as_of_date']) if records else ''
    lines = []

    lines.append(f"# Market Clock Report — {today}")
    lines.append("")
    lines.append("> O'Neil / IBD Framework: Follow-Through Days & Distribution Days")
    lines.append("")

    # ── Summary table ────────────────────────────────────────────────────
    lines.append("## Summary")
    lines.append("")
    lines.append("| Index | State | DDs (25d) | Last FTD | FTD Gain | FTD Day |")
    lines.append("|-------|-------|:---------:|----------|:--------:|:-------:|")
    for r in records:
        dd_warn = ' ⚠️' if r['dd_count_25d'] >= DD_UNDER_PRESSURE else ''
        ftd_str  = str(r['last_ftd_date'])  if r['last_ftd_date']  else '—'
        gain_str = f"+{r['last_ftd_gain_pct']}%" if r['last_ftd_gain_pct'] else '—'
        day_str  = str(r['last_ftd_day'])   if r['last_ftd_day']   else '—'
        lines.append(
            f"| **{r['ticker']}** | {_state_badge(r['state'])} "
            f"| {r['dd_count_25d']}{dd_warn} "
            f"| {ftd_str} | {gain_str} | {day_str} |"
        )
    lines.append("")

    # ── State legend ─────────────────────────────────────────────────────
    lines.append("### State Definitions")
    lines.append("")
    lines.append("| State | Meaning |")
    lines.append("|-------|---------|")
    lines.append("| 🟢 Confirmed Uptrend | FTD occurred, ≤4 distribution days in rolling 25 sessions |")
    lines.append("| 🟡 Uptrend Under Pressure | 5 distribution days in rolling 25 sessions |")
    lines.append("| 🔴 Uptrend Under Stress | 6+ distribution days — nearing correction |")
    lines.append("| 🟡 Rally Attempt Day N | Correction low set, Day N of attempt; FTD eligible on Day 4+ |")
    lines.append("| 🔴 Correction | No valid FTD, or FTD invalidated (rally low undercut) |")
    lines.append("")

    # ── Per-index detail ─────────────────────────────────────────────────
    lines.append("---")
    lines.append("")

    for r in records:
        lines.append(f"## {r['ticker']} — {_state_badge(r['state'])}")
        lines.append("")
        lines.append(f"- **Close:** {r['close']:.2f}  &nbsp;·&nbsp;  **As of:** {r['as_of_date']}")
        dd_note = ' ⚠️ above threshold' if r['dd_count_25d'] >= DD_UNDER_PRESSURE else ''
        lines.append(f"- **Distribution Days (rolling 25 sessions):** {r['dd_count_25d']}{dd_note}")

        if r['last_ftd_date']:
            quality = next((f['quality'] for f in r['follow_through_days']
                           if str(f['date']) == str(r['last_ftd_date'])), '')
            lines.append(
                f"- **Last FTD:** {r['last_ftd_date']}  ·  "
                f"+{r['last_ftd_gain_pct']}%  ·  "
                f"Vol×{r['last_ftd_vol_ratio']}  ·  "
                f"Day {r['last_ftd_day']}  ·  [{quality}]"
            )
        else:
            lines.append("- **Last FTD:** none in lookback window")
        lines.append("")

        # FTDs table
        ftds = r['follow_through_days'][:8]
        if ftds:
            lines.append(f"### Follow-Through Days ({len(r['follow_through_days'])} in window)")
            lines.append("")
            lines.append("| Date | Gain % | Vol × Prior | Rally Day | Quality |")
            lines.append("|------|-------:|:-----------:|:---------:|---------|")
            for f in ftds:
                day_lbl = str(f['rally_day']) if f['rally_day'] else '?'
                opt_lbl = ' ✓' if f['rally_day'] and FTD_DAY_MIN <= f['rally_day'] <= FTD_DAY_OPTIMAL else ''
                lines.append(
                    f"| {f['date']} | +{f['pct_chg']:.2f}% | {f['vol_ratio']:.2f}× "
                    f"| {day_lbl}{opt_lbl} | {f['quality']} |"
                )
            lines.append("")

        # DDs + stalls table
        dds = (r['distribution_days'] + r['stall_days'])
        dds.sort(key=lambda x: str(x['date']), reverse=True)
        dds = dds[:15]
        if dds:
            n_dd    = len(r['distribution_days'])
            n_stall = len(r['stall_days'])
            lines.append(f"### Distribution + Stall Days ({n_dd} DD, {n_stall} stall in window)")
            lines.append("")
            lines.append("| Date | Chg % | Vol × Prior | Type | Severity |")
            lines.append("|------|------:|:-----------:|------|----------|")
            for d in dds:
                lines.append(
                    f"| {d['date']} | {d['pct_chg']:+.2f}% | {d['vol_ratio']:.2f}× "
                    f"| {d['type']} | {d['severity']} |"
                )
            lines.append("")

        lines.append("---")
        lines.append("")

    # ── Methodology note ─────────────────────────────────────────────────
    lines.append("## Methodology")
    lines.append("")
    lines.append(f"- **FTD criteria:** gain ≥ {FTD_PRICE_GAIN}% from prior close, "
                 f"volume above prior session, Day {FTD_DAY_MIN}+ of rally attempt")
    lines.append(f"- **Optimal FTD window:** Days {FTD_DAY_MIN}–{FTD_DAY_OPTIMAL}")
    lines.append(f"- **DD criteria:** decline ≥ {abs(DD_PRICE_DROP)}% from prior close, "
                 f"volume above prior session")
    lines.append(f"- **Stall day:** gain 0–{STALL_PRICE_MAX}% but volume ≥ {STALL_VOL_RATIO}× prior session")
    lines.append(f"- **DD window:** rolling {DD_WINDOW} trading sessions (DDs expire after {DD_WINDOW} sessions)")
    lines.append(f"- **Thresholds:** ≥{DD_UNDER_PRESSURE} DDs = Under Pressure · "
                 f"≥{DD_UNDER_STRESS} DDs = Under Stress")
    lines.append("")
    lines.append(f"*Source: O'Neil, W. (2009). How to Make Money in Stocks. IBD methodology.*")
    lines.append("")

    return '\n'.join(lines)


def print_report(records: list[dict]) -> None:
    """Print a readable market clock report to stdout."""
    DIVIDER = '─' * 72

    for r in records:
        print(f"\n{DIVIDER}")
        print(f"  {r['ticker']}  |  {r['state']}  |  close {r['close']:.2f}  |  as of {r['as_of_date']}")
        print(DIVIDER)

        dd_color = ('⚠️ ' if r['dd_count_25d'] >= DD_UNDER_PRESSURE else '')
        print(f"  Distribution Days (rolling 25 sessions): {dd_color}{r['dd_count_25d']}")

        if r['last_ftd_date']:
            quality = next((f['quality'] for f in r['follow_through_days']
                           if str(f['date']) == str(r['last_ftd_date'])), '')
            print(f"  Last FTD: {r['last_ftd_date']}  "
                  f"+{r['last_ftd_gain_pct']}%  "
                  f"vol×{r['last_ftd_vol_ratio']}  "
                  f"Day {r['last_ftd_day']}  [{quality}]")
        else:
            print("  Last FTD: none in lookback window")

        ftds = r['follow_through_days'][:5]
        if ftds:
            print(f"\n  Follow-Through Days (last {len(r['follow_through_days'])} in window, showing 5):")
            print(f"  {'Date':<12} {'Gain%':>7} {'Vol×':>6} {'Day':>4} {'Quality':<8}")
            for f in ftds:
                print(f"  {str(f['date']):<12} {f['pct_chg']:>+7.2f} {f['vol_ratio']:>6.2f}"
                      f" {str(f['rally_day'] or '?'):>4}  {f['quality']}")

        dds = (r['distribution_days'] + r['stall_days'])
        dds.sort(key=lambda x: str(x['date']), reverse=True)
        dds = dds[:10]
        if dds:
            print(f"\n  Distribution + Stall Days (last {len(r['distribution_days'])} DD, "
                  f"{len(r['stall_days'])} stall, showing 10):")
            print(f"  {'Date':<12} {'Chg%':>7} {'Vol×':>6} {'Type':<14} {'Severity'}")
            for d in dds:
                print(f"  {str(d['date']):<12} {d['pct_chg']:>+7.2f} {d['vol_ratio']:>6.2f}"
                      f"  {d['type']:<14} {d['severity']}")
