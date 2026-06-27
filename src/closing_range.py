"""
Closing Range Analysis — IBD / O'Neil Correction Leader Filter.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IT MEASURES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
During a market correction, institutional money does not vanish — it
rotates into stocks being accumulated. These stocks consistently
close in the upper portion of their daily range even as the broad
market sells off. O'Neil's insight: stocks that close >60% of the
way through their daily range (High – Low) for most correction days
are being actively supported.

Closing Range % = (Close − Low) / (High − Low) × 100
  ≥ 80% : closed near the high — strong institutional buying
  60–80%: upper range — mild support
  40–60%: middle — neutral
  < 40% : lower range — distribution pressure

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5-FILTER LEADER SCORE (0 – 5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
F1  Closing Range   CR ≥ 60% on ≥ 60% of correction days
F2  MA Adherence    Price above 21-EMA on ≥ 70% of correction days
F3  RS vs SPY       RS line (stock / SPY) positive over the window
F4  Above 50-SMA    Price above 50-SMA at correction end
F5  Volume Accum    Up-day volume > down-day volume over window

Score 5 : perfect correction leader — all criteria met
Score 4 : strong leader
Score 3 : developing leader — worth watching
Score ≤2: laggard during correction

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CORRECTION WINDOW AUTO-DETECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If no manual dates are provided:
  end   = last_ftd_date from market_clock CSV for SPY
          (the Follow-Through Day when the correction was confirmed over)
  start = the day the SPY rolling-peak drawdown first exceeded –5%
          looking backward from the FTD date

Override with --correction-start / --correction-end on the CLI.

Output: results/closing_range_YYYY-MM-DD.csv + .md
"""

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from src.rotation_composite import _latest_file

logger = logging.getLogger(__name__)

# ── Filter thresholds ──────────────────────────────────────────────────────
CR_THRESHOLD       = 60.0    # closing range % to count as "strong close"
CR_DAY_PCT         = 0.60    # fraction of days that must be strong closes (F1)
EMA21_DAY_PCT      = 0.70    # fraction of days price must be above 21-EMA (F2)
MIN_DAYS           = 8       # skip stocks with fewer correction-window bars
LOOKBACK_EXTRA     = 60      # extra calendar days before window to warm up EMAs

# ── EMA / SMA helpers ─────────────────────────────────────────────────────

def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()


# ── Correction window auto-detection ──────────────────────────────────────

def detect_correction_window(
    config: Config,
    as_of: date | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Return (correction_start, correction_end) from market_clock CSV + SPY daily.

    correction_end  = SPY's last_ftd_date
    correction_start = first day SPY drawdown from rolling peak exceeded –5%,
                       searching backward from correction_end.
    """
    label = str(as_of) if as_of else None

    # ── Load market_clock for FTD date ────────────────────────────────────
    if label:
        mc_path = config.results_dir / f"market_clock_{label}.csv"
    else:
        mc_path = _latest_file(config.results_dir, "market_clock_*.csv")

    ftd_date: pd.Timestamp | None = None
    if mc_path and mc_path.exists():
        mc = pd.read_csv(mc_path)
        spy_row = mc[mc['ticker'] == 'SPY']
        if not spy_row.empty and pd.notna(spy_row.iloc[0].get('last_ftd_date')):
            ftd_date = pd.Timestamp(spy_row.iloc[0]['last_ftd_date'])
    if ftd_date is None:
        ftd_date = pd.Timestamp(as_of) if as_of else pd.Timestamp.today().normalize()
        logger.warning(f"No FTD date found in market_clock — using {ftd_date.date()} as correction end")

    # ── Load SPY daily to find correction start ────────────────────────────
    spy_path = config.daily_dir / 'SPY.csv'
    if not spy_path.exists():
        # Fall back to 60 calendar days before FTD
        corr_start = ftd_date - pd.Timedelta(days=60)
        logger.warning(f"SPY daily not found — using {corr_start.date()} as correction start (fallback)")
        return corr_start, ftd_date

    spy = pd.read_csv(spy_path, parse_dates=['Date'])
    spy['Date'] = pd.to_datetime(spy['Date'], utc=True).dt.tz_localize(None)
    spy = spy.sort_values('Date')

    # Work with data up to and including FTD
    spy = spy[spy['Date'] <= ftd_date]
    if spy.empty:
        corr_start = ftd_date - pd.Timedelta(days=60)
        return corr_start, ftd_date

    # Rolling peak drawdown
    peak   = spy['Close'].cummax()
    dd_pct = (spy['Close'] - peak) / peak * 100

    # Find the most recent stretch where drawdown <= –5% before FTD
    in_correction = dd_pct <= -5.0
    if not in_correction.any():
        corr_start = ftd_date - pd.Timedelta(days=30)
        logger.warning(f"SPY never drew down –5% — using 30-day window ending at FTD")
        return corr_start, ftd_date

    # Correction start = first day of the last contiguous below-peak-5% run
    # Work backward from FTD to find the run
    idx_series = spy.index[in_correction]
    # Take the run that includes the period just before the FTD
    # Find last day before FTD with drawdown ≤ –5%
    before_ftd = spy[spy['Date'] < ftd_date]
    dd_before  = dd_pct[before_ftd.index]
    last_corr_idx = dd_before[dd_before <= -5.0].index[-1] if (dd_before <= -5.0).any() else None

    if last_corr_idx is None:
        corr_start = ftd_date - pd.Timedelta(days=30)
        return corr_start, ftd_date

    # Walk backward to the start of this correction run
    i = last_corr_idx
    while i > spy.index[0] and dd_pct.loc[i - 1 if i - 1 in dd_pct.index else i] <= -5.0:
        candidates = [x for x in dd_pct.index if x < i]
        if not candidates:
            break
        prev = candidates[-1]
        if dd_pct.loc[prev] <= -5.0:
            i = prev
        else:
            break

    corr_start = spy.loc[i, 'Date']
    logger.info(f"Auto-detected correction window: {corr_start.date()} → {ftd_date.date()}")
    return corr_start, ftd_date


# ── Per-stock OHLCV loader ─────────────────────────────────────────────────

def _load_ohlcv(
    ticker: str,
    daily_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    warmup_start: pd.Timestamp,
) -> pd.DataFrame | None:
    """
    Load OHLCV for a ticker from warmup_start through end.
    Returns normalised DataFrame with Date (tz-naive), O/H/L/C/V columns.
    """
    p = daily_dir / f"{ticker}.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, parse_dates=['Date'])
    except Exception:
        return None

    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
    df = df.sort_values('Date')
    df = df[(df['Date'] >= warmup_start) & (df['Date'] <= end)]

    needed = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(c in df.columns for c in needed):
        return None
    df = df[needed].copy()

    # Drop rows where High == Low (flat / no trade) to avoid division by zero
    df = df[df['High'] > df['Low']].copy()
    if df.empty:
        return None

    df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].astype(float)
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
    return df.reset_index(drop=True)


# ── Per-stock scoring ──────────────────────────────────────────────────────

def _score_stock(
    df: pd.DataFrame,
    spy_close: pd.Series,
    corr_start: pd.Timestamp,
    corr_end: pd.Timestamp,
) -> dict | None:
    """
    Compute all 5 filters + supporting metrics for one stock.

    df          — full OHLCV DataFrame from warmup_start through corr_end
    spy_close   — SPY close indexed by date (pd.Series, tz-naive dates as index)
    corr_start  — first trading day of correction window
    corr_end    — last trading day (FTD date)
    """
    # Compute indicators over full (warmup) history
    df = df.copy()
    df['ema21']  = _ema(df['Close'], 21)
    df['sma50']  = _sma(df['Close'], 50)
    df['cr_pct'] = (df['Close'] - df['Low']) / (df['High'] - df['Low']) * 100

    # Align with SPY for RS computation
    df = df.set_index('Date')
    spy_aligned = spy_close.reindex(df.index, method='ffill')
    df['rs_line'] = df['Close'] / spy_aligned.replace(0, np.nan)

    # Slice to correction window
    window = df[(df.index >= corr_start) & (df.index <= corr_end)]
    if len(window) < MIN_DAYS:
        return None

    n = len(window)

    # ── F1: Closing Range ≥ 60% on ≥ CR_DAY_PCT of days ──────────────────
    cr_strong_days = (window['cr_pct'] >= CR_THRESHOLD).sum()
    f1 = (cr_strong_days / n) >= CR_DAY_PCT

    # ── F2: Above 21-EMA on ≥ EMA21_DAY_PCT of days ──────────────────────
    above_ema21 = (window['Close'] > window['ema21']).sum()
    f2 = (above_ema21 / n) >= EMA21_DAY_PCT

    # ── F3: RS line positive over full window ─────────────────────────────
    rs_start = float(window['rs_line'].iloc[0])
    rs_end   = float(window['rs_line'].iloc[-1])
    f3 = (rs_end > rs_start) if (rs_start > 0 and not np.isnan(rs_start)) else False

    # ── F4: Above 50-SMA at correction end ───────────────────────────────
    close_end = float(window['Close'].iloc[-1])
    sma50_end = float(window['sma50'].iloc[-1])
    f4 = (close_end > sma50_end) if not np.isnan(sma50_end) else False

    # ── F5: Volume accumulation (up-day vol > down-day vol) ───────────────
    prev_close = window['Close'].shift(1)
    up_mask    = window['Close'] > prev_close
    dn_mask    = window['Close'] < prev_close
    up_vol = window.loc[up_mask,  'Volume'].sum()
    dn_vol = window.loc[dn_mask, 'Volume'].sum()
    f5 = up_vol > dn_vol

    score = sum([f1, f2, f3, f4, f5])

    # Supporting metrics for output
    cr_mean = float(window['cr_pct'].mean())
    ema21_pct = float(above_ema21 / n * 100)
    rs_chg_pct = float((rs_end / rs_start - 1) * 100) if rs_start > 0 else np.nan
    pct_vs_sma50 = float((close_end / sma50_end - 1) * 100) if sma50_end > 0 else np.nan
    up_vol_pct = float(up_vol / (up_vol + dn_vol) * 100) if (up_vol + dn_vol) > 0 else np.nan

    return {
        'leader_score':  score,
        'f1_cr':         f1,
        'f2_ema21':      f2,
        'f3_rs_spy':     f3,
        'f4_above_50sma': f4,
        'f5_accum':      f5,
        # metrics
        'cr_mean':       round(cr_mean, 1),
        'cr_strong_days': int(cr_strong_days),
        'corr_days':     n,
        'ema21_held_pct': round(ema21_pct, 1),
        'rs_chg_pct':    round(rs_chg_pct, 2) if not np.isnan(rs_chg_pct) else np.nan,
        'pct_vs_sma50':  round(pct_vs_sma50, 2) if not np.isnan(pct_vs_sma50) else np.nan,
        'up_vol_pct':    round(up_vol_pct, 1) if not np.isnan(up_vol_pct) else np.nan,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def run(
    config: Config,
    as_of: date | None = None,
    correction_start: date | None = None,
    correction_end: date | None = None,
    industry_keys: list[str] | None = None,
    min_score: int = 3,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """
    Score all GICS stocks as correction leaders.

    industry_keys : optional whitelist of industry_key values (e.g. STRONG BUY industries)
    min_score     : only include stocks with leader_score >= min_score in output

    Returns (df, corr_start_ts, corr_end_ts).
    """
    label = str(as_of) if as_of else None

    # ── Correction window ─────────────────────────────────────────────────
    if correction_start and correction_end:
        corr_start = pd.Timestamp(correction_start)
        corr_end   = pd.Timestamp(correction_end)
        logger.info(f"Manual correction window: {corr_start.date()} → {corr_end.date()}")
    else:
        corr_start, corr_end = detect_correction_window(config, as_of)

    warmup_start = corr_start - pd.Timedelta(days=LOOKBACK_EXTRA)

    # ── Load SPY close for RS computation ─────────────────────────────────
    spy_path = config.daily_dir / 'SPY.csv'
    if not spy_path.exists():
        logger.error("SPY.csv not found — RS filter (F3) will be disabled")
        spy_close = pd.Series(dtype=float)
    else:
        spy_raw = pd.read_csv(spy_path, parse_dates=['Date'])
        spy_raw['Date'] = pd.to_datetime(spy_raw['Date'], utc=True).dt.tz_localize(None)
        spy_raw = spy_raw.sort_values('Date')
        spy_raw = spy_raw[(spy_raw['Date'] >= warmup_start) & (spy_raw['Date'] <= corr_end)]
        spy_close = spy_raw.set_index('Date')['Close'].astype(float)

    # ── Load stock universe ───────────────────────────────────────────────
    sbi        = pd.read_csv(config.stocks_by_industry_csv)
    industries = pd.read_csv(config.industries_csv)
    ind_map    = industries.set_index('industry_key')[['industry_name', 'sector_name']].to_dict('index')

    if industry_keys:
        sbi = sbi[sbi['industry_key'].isin(industry_keys)]
        logger.info(f"Filtered to {len(industry_keys)} industries: {len(sbi)} stocks")

    # Load momentum/faber context for industry-level enrichment
    mom_map: dict[str, str] = {}
    if label:
        mom_path = config.results_dir / f"momentum_screen_{label}.csv"
    else:
        mom_path = _latest_file(config.results_dir, "momentum_screen_*.csv")
    if mom_path and mom_path.exists():
        mom = pd.read_csv(mom_path).set_index('industry_key')
        mom_map = mom['faber_signal'].to_dict()

    tickers_df = sbi[['symbol', 'industry_key']].dropna(subset=['symbol']).drop_duplicates()
    total = len(tickers_df)
    logger.info(f"Scoring {total} stocks over correction window "
                f"{corr_start.date()} → {corr_end.date()}  "
                f"({(corr_end - corr_start).days} calendar days)")

    records = []
    skipped = 0

    for i, (_, row) in enumerate(tickers_df.iterrows()):
        if i % 500 == 0 and i > 0:
            logger.info(f"  {i}/{total} stocks processed ({len(records)} scored)...")

        ticker   = str(row['symbol'])
        ind_key  = str(row['industry_key'])

        ohlcv = _load_ohlcv(ticker, config.daily_dir, corr_start, corr_end, warmup_start)
        if ohlcv is None:
            skipped += 1
            continue

        metrics = _score_stock(ohlcv, spy_close, corr_start, corr_end)
        if metrics is None:
            skipped += 1
            continue

        if metrics['leader_score'] < min_score:
            continue

        meta = ind_map.get(ind_key, {})
        records.append({
            'ticker':        ticker,
            'industry_key':  ind_key,
            'industry_name': meta.get('industry_name', ind_key),
            'sector_name':   meta.get('sector_name', ''),
            'faber_signal':  mom_map.get(ind_key, ''),
            **metrics,
        })

    logger.info(f"Scored: {len(records)} leaders (score ≥ {min_score}),  "
                f"skipped: {skipped} (no data / insufficient window)")

    if not records:
        return pd.DataFrame(), corr_start, corr_end

    df = (pd.DataFrame(records)
            .sort_values(['leader_score', 'cr_mean'], ascending=[False, False])
            .reset_index(drop=True))
    df.insert(0, 'rank', df.index + 1)

    score_dist = df['leader_score'].value_counts().sort_index(ascending=False).to_dict()
    logger.info(f"Leader score distribution: {score_dist}")

    return df, corr_start, corr_end


# ── Save ──────────────────────────────────────────────────────────────────

def save(
    df: pd.DataFrame,
    config: Config,
    corr_start: pd.Timestamp,
    corr_end: pd.Timestamp,
    as_of: date | None = None,
) -> tuple[Path, Path]:
    config.setup_dirs()
    label = str(as_of) if as_of else pd.Timestamp.today().strftime('%Y-%m-%d')

    csv_cols = [
        'rank', 'ticker', 'industry_key', 'sector_name', 'industry_name', 'faber_signal',
        'leader_score', 'f1_cr', 'f2_ema21', 'f3_rs_spy', 'f4_above_50sma', 'f5_accum',
        'cr_mean', 'cr_strong_days', 'corr_days',
        'ema21_held_pct', 'rs_chg_pct', 'pct_vs_sma50', 'up_vol_pct',
    ]
    cols_present = [c for c in csv_cols if c in df.columns]

    csv_out = config.results_dir / f"closing_range_{label}.csv"
    df[cols_present].to_csv(csv_out, index=False)
    logger.info(f"Saved closing range CSV → {csv_out}  ({len(df)} stocks)")

    md_out = config.results_dir / f"closing_range_{label}.md"
    md_out.write_text(_build_md(df, label, corr_start, corr_end), encoding='utf-8')
    logger.info(f"Saved closing range MD  → {md_out}")

    return csv_out, md_out


# ── Markdown report ────────────────────────────────────────────────────────

_FABER_ORDER = ['STRONG BUY', 'BUY', 'HOLD', 'WATCH', 'CAUTION', 'EXIT', '']


def _build_md(
    df: pd.DataFrame,
    label: str,
    corr_start: pd.Timestamp,
    corr_end: pd.Timestamp,
) -> str:
    n_days = len(df['corr_days'].mode()) and int(df['corr_days'].median())
    lines = []
    lines.append(f"# Closing Range Analysis — {label}")
    lines.append("")
    lines.append(f"> Correction window: **{corr_start.date()}** → **{corr_end.date()}**  "
                 f"({n_days} trading days median)")
    lines.append("")
    lines.append("> IBD / O'Neil Correction Leader Filter — 5 criteria, score 0–5")
    lines.append("")

    # Summary
    score_dist = df['leader_score'].value_counts().sort_index(ascending=False)
    lines.append("## Score Distribution")
    lines.append("")
    lines.append("| Score | Stocks | Meaning |")
    lines.append("|:-----:|:------:|---------|")
    desc = {5: 'Perfect leader — all 5 criteria', 4: 'Strong leader',
            3: 'Developing — worth watching', 2: 'Partial', 1: 'Weak'}
    for s, n in score_dist.items():
        lines.append(f"| {s}/5 | {n} | {desc.get(s, '')} |")
    lines.append("")

    # Per-faber-signal groups, within each: per-score sub-groups
    for fsig in _FABER_ORDER:
        sub_f = df[df['faber_signal'] == fsig]
        if sub_f.empty:
            continue
        em = {'STRONG BUY': '🟢', 'BUY': '🟢', 'HOLD': '🟡',
              'WATCH': '🔵', 'CAUTION': '⚠️', 'EXIT': '🔴', '': '⚪'}.get(fsig, '⚪')
        lines.append(f"## {em} {fsig or 'Unknown'} industries ({len(sub_f)} stocks)")
        lines.append("")
        lines.append("| Rank | Ticker | Industry | Score | CR% | CR days | EMA21% | RS chg | vs50SMA | UpVol% |")
        lines.append("|-----:|--------|----------|:-----:|:---:|:-------:|:------:|:------:|:-------:|:------:|")
        for _, r in sub_f.sort_values(['leader_score', 'cr_mean'], ascending=[False, False]).iterrows():
            f_str = ''.join([
                '✓' if r['f1_cr'] else '✗',
                '✓' if r['f2_ema21'] else '✗',
                '✓' if r['f3_rs_spy'] else '✗',
                '✓' if r['f4_above_50sma'] else '✗',
                '✓' if r['f5_accum'] else '✗',
            ])
            lines.append(
                f"| {int(r['rank'])} | {r['ticker']} | {r['industry_name']} "
                f"| {int(r['leader_score'])}/5 `{f_str}` "
                f"| {r['cr_mean']:.1f}% "
                f"| {int(r['cr_strong_days'])}/{int(r['corr_days'])} "
                f"| {r['ema21_held_pct']:.0f}% "
                f"| {r['rs_chg_pct']:+.1f}% "
                f"| {r['pct_vs_sma50']:+.1f}% "
                f"| {r['up_vol_pct']:.0f}% |"
            )
        lines.append("")

    # Methodology
    lines.append("## Methodology")
    lines.append("")
    lines.append("**F1 Closing Range** — (Close − Low) / (High − Low) × 100. "
                 "Counts as strong if ≥ 60%. Leader if ≥ 60% of days are strong closes.")
    lines.append("")
    lines.append("**F2 MA Adherence** — Stock closes above its 21-day EMA on ≥ 70% of correction days. "
                 "21-EMA is computed with full lookback (60+ days warmup) before the window.")
    lines.append("")
    lines.append("**F3 RS vs SPY** — RS line (stock / SPY) at correction end > RS line at correction start. "
                 "Positive RS change = outperformed SPY during the selloff.")
    lines.append("")
    lines.append("**F4 Above 50-SMA** — Close above 50-day SMA at the correction end date.")
    lines.append("")
    lines.append("**F5 Volume Accumulation** — Total up-day volume (days where Close > prev Close) "
                 "exceeds total down-day volume over the correction window.")
    lines.append("")
    lines.append("*Source: O'Neil, W. (2009). How to Make Money in Stocks. McGraw-Hill.*")
    lines.append("")
    return '\n'.join(lines)


# ── Terminal report ────────────────────────────────────────────────────────

def print_report(
    df: pd.DataFrame,
    corr_start: pd.Timestamp,
    corr_end: pd.Timestamp,
    top_n: int = 30,
) -> None:
    DIVIDER = '─' * 108
    n_days = int(df['corr_days'].median()) if not df.empty else 0
    print(f"\n  Correction Leader Filter  |  {corr_start.date()} → {corr_end.date()}  "
          f"({n_days} trading days)  |  {len(df)} stocks scored")

    for fsig in _FABER_ORDER:
        sub = df[df['faber_signal'] == fsig]
        if sub.empty:
            continue
        print(f"\n{DIVIDER}")
        label = fsig or 'Other'
        print(f"  {label}  ({len(sub)} stocks)")
        print(DIVIDER)
        print(f"  {'Rnk':>4}  {'Ticker':<7} {'Industry':<34} {'Scr':>4}  "
              f"{'Fltr':>5}  {'CR%':>5}  {'CR days':>7}  {'EMA21':>6}  "
              f"{'RS chg':>7}  {'vs50':>5}  {'UpV%':>5}")
        print(DIVIDER)
        shown = 0
        for _, r in sub.sort_values(['leader_score', 'cr_mean'], ascending=[False, False]).iterrows():
            if shown >= top_n:
                remaining = len(sub) - shown
                if remaining > 0:
                    print(f"  ... {remaining} more")
                break
            f_str = ''.join([
                '✓' if r['f1_cr'] else '·',
                '✓' if r['f2_ema21'] else '·',
                '✓' if r['f3_rs_spy'] else '·',
                '✓' if r['f4_above_50sma'] else '·',
                '✓' if r['f5_accum'] else '·',
            ])
            rs_str = f"{r['rs_chg_pct']:+.1f}%" if pd.notna(r['rs_chg_pct']) else '  n/a'
            s50_str = f"{r['pct_vs_sma50']:+.1f}%" if pd.notna(r['pct_vs_sma50']) else '  n/a'
            print(
                f"  {int(r['rank']):>4}  {str(r['ticker']):<7} "
                f"{str(r['industry_name']):<34} "
                f"{int(r['leader_score'])}/5  "
                f"{f_str}  "
                f"{r['cr_mean']:>5.1f}%  "
                f"{int(r['cr_strong_days']):>3}/{int(r['corr_days']):<3}  "
                f"{r['ema21_held_pct']:>5.0f}%  "
                f"{rs_str:>7}  "
                f"{s50_str:>5}  "
                f"{r['up_vol_pct']:>5.1f}%"
            )
            shown += 1
    print(f"\n{DIVIDER}")
