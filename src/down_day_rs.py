"""
Down-Day Relative Strength — which stocks show strength specifically on the
days the benchmark falls.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY THIS IS DIFFERENT FROM THE CORRECTION-LEADER FILTER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
closing_range.py / correction_filter.py measure how a stock behaved over a
correction *window* as a whole (closing range, RS-line start-vs-end, EMA
adherence). A low-beta name that simply drifts sideways scores well there
even if it never actually outperformed on the red days.

This module conditions on the benchmark's *daily* direction: it looks only
at the sessions where SPY closed down and asks how the stock did on exactly
those days.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METRICS  (per trailing window, and over the active correction window)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  down_capture     mean(stock_ret on SPY-down days) / mean(SPY_ret on those days)
                   1.0 = matches the market · <1 = cushioned · <=0 = rose while
                   the market fell (the signal we want)
  down_winrate     fraction of SPY-down days the stock's return beat SPY's
  down_green       fraction of SPY-down days the stock itself closed green
  down_alpha       mean(stock_ret - SPY_ret) on SPY-down days, in % per day
  up_capture       same ratio on SPY-up days
  capture_spread   up_capture - down_capture  (O'Neil ideal: high up, low down)

SIGNAL COUNT (0-4): cushioned (down_capture < 0.80) + positive_on_red
  (down_capture <= 0) + outperform (down_winrate >= 0.60) + good_spread
  (capture_spread >= 0.30).

RANKING: down_alpha, inverted down_capture and capture_spread are each
percentile-ranked across the scored universe; down_day_rank =
0.50 * alpha_pct + 0.30 * inv_capture_pct + 0.20 * spread_pct, re-ranked 0-99.
The same blend is also re-ranked within cap bucket and within industry.

Correction window (optional context — trailing windows are the primary signal):
the peak in the trailing 90 calendar days down to the last bar, when SPY is
>= 3% below that peak. `_corr`-suffixed metric columns appear only when SPY is
actually in a pullback. Override with --correction-start / --correction-end.

Output: results/down_day_rs_YYYY-MM-DD.csv + .md
"""

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from src.data_loader import load_daily, _latest_file

logger = logging.getLogger(__name__)

# ── Windows / thresholds ──────────────────────────────────────────────────
DEFAULT_WINDOWS   = (21, 42, 63)   # trailing trading-day windows
RANK_WINDOW       = 42             # which window feeds the cross-sectional rank
MIN_DOWN_DAYS     = 6              # need at least this many benchmark-down days
MIN_OVERLAP       = 10             # min aligned stock/benchmark observations
WARMUP_EXTRA      = 15             # extra bars beyond the widest window
MAX_ABS_DAY_RET   = 40.0           # skip a ticker if any single-day |return| exceeds this
                                   # in the window (split / bad print / delisting artifact)

# Capture ratios are unstable when the benchmark's mean down-day move is small
# (a two-day pop in a thin name can print down_capture = -500). The cap-bucket
# universe filter removes the worst offenders; this wide clip just fences off
# genuinely broken values before ranking. The raw ratio is kept in the CSV.
CAPTURE_CLIP      = (-10.0, 10.0)

DOWN_CAPTURE_MAX  = 0.80           # "cushioned" threshold
DOWN_WINRATE_MIN  = 0.60           # "outperform" threshold
SPREAD_MIN        = 0.30           # "good spread" threshold

CORR_LOOKBACK     = 90             # calendar days back to look for a correction peak
CORR_DROP_MIN     = 0.03           # SPY must be ≥3% below the trailing peak to count

BENCHMARK         = "SPY"


def _detect_correction(spy_close: pd.Series) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Most recent SPY drawdown window: peak in the trailing CORR_LOOKBACK days
    down to the last bar, when that drop is ≥ CORR_DROP_MIN. (None, None) if SPY
    is not currently in a pullback. Uses the already-`as_of`-sliced series."""
    recent = spy_close.tail(CORR_LOOKBACK)
    if len(recent) < 10:
        return None, None
    peak_idx = recent.idxmax()
    peak, last = float(recent.max()), float(recent.iloc[-1])
    if peak <= 0 or (peak - last) / peak < CORR_DROP_MIN:
        return None, None
    after_peak = recent[recent.index > peak_idx]
    if after_peak.empty:
        return None, None
    return after_peak.index[0], recent.index[-1]


# ── Core metric (pure — also imported by app.py) ──────────────────────────

def down_day_metrics(
    stock_ret: pd.Series,
    bench_ret: pd.Series,
    down_threshold: float = 0.0,
) -> dict | None:
    """
    Down-day capture statistics for one stock vs one benchmark.

    stock_ret / bench_ret : daily % returns (e.g. close.pct_change() * 100),
                            indexed by date. They are inner-joined on date, so
                            they need not be pre-aligned or the same length.
    down_threshold        : a session counts as a "down day" when the benchmark
                            return is strictly below this (0.0 = any red close;
                            -0.5 = only meaningful down days).

    Returns a dict of scalars, or None if there is not enough overlap / not
    enough down days to be meaningful.
    """
    df = pd.concat({"s": stock_ret, "b": bench_ret}, axis=1).dropna()
    if len(df) < MIN_OVERLAP:
        return None

    down = df["b"] < down_threshold
    up   = df["b"] > 0.0
    n_down = int(down.sum())
    n_up   = int(up.sum())
    if n_down < MIN_DOWN_DAYS:
        return None

    s_down = df.loc[down, "s"]
    b_down = df.loc[down, "b"]
    mean_b_down = float(b_down.mean())

    down_capture = float(s_down.mean() / mean_b_down) if mean_b_down != 0 else np.nan
    down_winrate = float((s_down > b_down).mean())
    down_green   = float((s_down > 0).mean())
    down_alpha   = float((s_down - b_down).mean())

    if n_up > 0:
        mean_b_up  = float(df.loc[up, "b"].mean())
        up_capture = float(df.loc[up, "s"].mean() / mean_b_up) if mean_b_up != 0 else np.nan
    else:
        up_capture = np.nan

    capture_spread = (up_capture - down_capture) if not (
        np.isnan(up_capture) or np.isnan(down_capture)
    ) else np.nan

    signal_count = int(
        (not np.isnan(down_capture) and down_capture < DOWN_CAPTURE_MAX)
        + (not np.isnan(down_capture) and down_capture <= 0.0)
        + (down_winrate >= DOWN_WINRATE_MIN)
        + (not np.isnan(capture_spread) and capture_spread >= SPREAD_MIN)
    )

    return {
        "n_down_days":    n_down,
        "n_up_days":      n_up,
        "down_capture":   round(down_capture, 3) if not np.isnan(down_capture) else np.nan,
        "down_winrate":   round(down_winrate, 3),
        "down_green":     round(down_green, 3),
        "down_alpha":     round(down_alpha, 3),
        "up_capture":     round(up_capture, 3) if not np.isnan(up_capture) else np.nan,
        "capture_spread": round(capture_spread, 3) if not np.isnan(capture_spread) else np.nan,
        "signal_count":   signal_count,
    }


# ── Universe helpers ─────────────────────────────────────────────────────

def _load_stock_sctr(config: Config, as_of: date | None) -> pd.DataFrame:
    """cap_bucket + current SCTR per ticker from the stock_sctr CSVs (optional
    enrichment — stocks without a row are still scored, just uncapped)."""
    label = str(as_of) if as_of else None
    parts = []
    for bucket in ("large", "mid", "small"):
        if label:
            p = config.stock_sctr_dir / f"stock_sctr_{bucket}_{label}.csv"
        else:
            files = sorted(config.stock_sctr_dir.glob(f"stock_sctr_{bucket}_*.csv"))
            p = files[-1] if files else None
        if p and p.exists():
            df = pd.read_csv(p)
            keep = [c for c in ("ticker", "cap_bucket", "sctr") if c in df.columns]
            parts.append(df[keep])
    if not parts:
        return pd.DataFrame(columns=["ticker", "cap_bucket", "sctr"])
    return pd.concat(parts, ignore_index=True).drop_duplicates("ticker")


def _load_universe(config: Config, industry_keys: list[str] | None) -> pd.DataFrame:
    sbi = pd.read_csv(
        config.stocks_by_industry_csv,
        usecols=["symbol", "sector_name", "industry_name", "industry_key"],
    ).dropna(subset=["symbol"]).drop_duplicates("symbol")
    if industry_keys:
        sbi = sbi[sbi["industry_key"].isin(industry_keys)]
    return sbi


def _clip_capture(s: pd.Series) -> pd.Series:
    return s.clip(*CAPTURE_CLIP)


def blend_rank(
    down_alpha: pd.Series,
    down_capture: pd.Series,
    capture_spread: pd.Series,
) -> pd.Series:
    """0–99 down-day-RS rank from three aligned metric Series.

    down_alpha (%/day of outperformance on benchmark-down days) is the stable
    primary; inverted down_capture and the up/down spread refine it. Each is
    percentile-ranked, blended 0.50 / 0.30 / 0.20, then re-ranked 0–99.

    Shared by the CLI screener (down_day_rs.run) and the dashboard tab so both
    surfaces use exactly one definition of the rank.
    """
    a  = down_alpha.rank(pct=True, na_option="keep")
    ic = 1.0 - _clip_capture(down_capture).rank(pct=True, na_option="keep")
    sp = _clip_capture(capture_spread).rank(pct=True, na_option="keep")
    blended = (
        0.50 * a.fillna(a.mean())
        + 0.30 * ic.fillna(ic.mean())
        + 0.20 * sp.fillna(sp.mean())
    )
    return (blended.rank(pct=True) * 99).round(1)


# ── Main ─────────────────────────────────────────────────────────────────

def run(
    config: Config,
    as_of: date | None = None,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    correction_start: date | None = None,
    correction_end: date | None = None,
    industry_keys: list[str] | None = None,
    min_signals: int = 2,
    sctr_min: float | None = None,
    down_threshold: float = 0.0,
    require_cap_bucket: bool = True,
) -> tuple[pd.DataFrame, pd.Timestamp | None, pd.Timestamp | None]:
    """
    Score every GICS stock on down-day relative strength vs SPY.

    Returns (df, corr_start, corr_end). corr_start/end are None when no
    correction window could be resolved (trailing-window metrics still run).
    """
    cutoff = pd.Timestamp(as_of) if as_of else None
    widest = max(windows)

    # ── Benchmark ────────────────────────────────────────────────────────
    spy = load_daily(BENCHMARK, config.daily_dir)
    if spy is None or spy.empty:
        logger.error(f"No {BENCHMARK} data — cannot compute down-day RS")
        return pd.DataFrame(), None, None
    if cutoff is not None:
        spy = spy[spy.index <= cutoff]
    spy_ret = spy["Close"].pct_change(fill_method=None) * 100.0

    # ── Correction window (best-effort context; trailing windows are primary) ─
    if correction_start and correction_end:
        corr_start, corr_end = pd.Timestamp(correction_start), pd.Timestamp(correction_end)
    else:
        corr_start, corr_end = _detect_correction(spy["Close"])
    if corr_start is not None:
        logger.info(f"Correction window: {corr_start.date()} → {corr_end.date()}")
    else:
        logger.info("SPY not in a pullback — reporting trailing-window metrics only")

    # ── Universe ────────────────────────────────────────────────────────
    universe = _load_universe(config, industry_keys)
    sctr = _load_stock_sctr(config, as_of)
    universe = universe.merge(sctr, left_on="symbol", right_on="ticker", how="left")

    if require_cap_bucket:
        if sctr.empty:
            logger.warning("No stock_sctr files — cannot apply cap-bucket quality filter "
                           "(run --mode stocks). Scoring the full universe.")
        else:
            before = len(universe)
            universe = universe[universe["cap_bucket"].notna()]
            logger.info(f"Cap-bucket filter (real SCTR universe): {before} → {len(universe)} stocks")
    if sctr_min is not None and "sctr" in universe.columns:
        universe = universe[universe["sctr"].fillna(-1) >= sctr_min]
    logger.info(f"Scoring {len(universe)} stocks over windows {windows} vs {BENCHMARK}")

    records = []
    skipped = 0
    for i, row in enumerate(universe.itertuples(index=False)):
        if i % 500 == 0 and i > 0:
            logger.info(f"  {i}/{len(universe)} processed ({len(records)} scored)...")

        daily = load_daily(row.symbol, config.daily_dir)
        if daily is None or daily.empty:
            skipped += 1
            continue
        if cutoff is not None:
            daily = daily[daily.index <= cutoff]
        if len(daily) < widest + WARMUP_EXTRA:
            skipped += 1
            continue

        stock_ret = daily["Close"].pct_change(fill_method=None) * 100.0
        if stock_ret.tail(widest).abs().max() > MAX_ABS_DAY_RET:
            skipped += 1   # split / bad print / delisting artifact in the window
            continue

        rec: dict = {
            "ticker":        row.symbol,
            "sector_name":   row.sector_name,
            "industry_name": row.industry_name,
            "industry_key":  row.industry_key,
            "cap_bucket":    getattr(row, "cap_bucket", "") if isinstance(getattr(row, "cap_bucket", ""), str) else "",
            "sctr":          round(row.sctr, 1) if not pd.isna(getattr(row, "sctr", np.nan)) else np.nan,
        }

        any_window = False
        for w in windows:
            m = down_day_metrics(stock_ret.tail(w), spy_ret.tail(w), down_threshold)
            if m is None:
                continue
            any_window = True
            for k, v in m.items():
                rec[f"{k}_{w}"] = v

        if corr_start is not None:
            m = down_day_metrics(
                stock_ret[(stock_ret.index >= corr_start) & (stock_ret.index <= corr_end)],
                spy_ret[(spy_ret.index >= corr_start) & (spy_ret.index <= corr_end)],
                down_threshold,
            )
            if m is not None:
                for k, v in m.items():
                    rec[f"{k}_corr"] = v

        if not any_window:
            skipped += 1
            continue

        records.append(rec)

    logger.info(f"Scored {len(records)} stocks · skipped {skipped} (no data / short history)")
    if not records:
        return pd.DataFrame(), corr_start, corr_end

    df = pd.DataFrame(records)

    # ── Cross-sectional ranking (on RANK_WINDOW, falling back if absent) ──
    rw = RANK_WINDOW if f"down_capture_{RANK_WINDOW}" in df.columns else max(
        (w for w in windows if f"down_capture_{w}" in df.columns), default=None
    )
    if rw is None:
        logger.warning("No usable window for ranking — returning unranked frame")
        return df, corr_start, corr_end

    cap_col, spread_col, alpha_col = (
        f"down_capture_{rw}", f"capture_spread_{rw}", f"down_alpha_{rw}"
    )
    df["rank_window"] = rw

    _mcols = [alpha_col, cap_col, spread_col]

    def _rank(frame: pd.DataFrame) -> pd.Series:
        return blend_rank(frame[alpha_col], frame[cap_col], frame[spread_col])

    df["down_day_rank"] = _rank(df)
    df["ddr_in_bucket"] = (
        df.groupby("cap_bucket", group_keys=False)[_mcols].apply(_rank)
        if df["cap_bucket"].ne("").any() else np.nan
    )
    df["ddr_in_industry"] = df.groupby("industry_key", group_keys=False)[_mcols].apply(
        lambda g: _rank(g) if len(g) >= 3 else pd.Series(np.nan, index=g.index)
    )

    sig_col = f"signal_count_{rw}"
    if sig_col in df.columns:
        df["signal_count"] = df[sig_col]
        df = df[df["signal_count"].fillna(0) >= min_signals].copy()

    df = df.sort_values("down_day_rank", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)

    if "signal_count" in df.columns:
        logger.info(f"Signal-count distribution: {df['signal_count'].value_counts().sort_index(ascending=False).to_dict()}")
    return df, corr_start, corr_end


# ── Save ─────────────────────────────────────────────────────────────────

_PRIMARY_COLS = [
    "rank", "ticker", "sector_name", "industry_name", "cap_bucket", "sctr",
    "rank_window", "down_day_rank", "ddr_in_bucket", "ddr_in_industry", "signal_count",
]


def save(
    df: pd.DataFrame,
    config: Config,
    corr_start: pd.Timestamp | None = None,
    corr_end: pd.Timestamp | None = None,
    as_of: date | None = None,
) -> tuple[Path, Path]:
    config.setup_dirs()
    label = str(as_of) if as_of else pd.Timestamp.today().strftime("%Y-%m-%d")

    csv_out = config.results_dir / f"down_day_rs_{label}.csv"
    ordered = [c for c in _PRIMARY_COLS if c in df.columns] + \
              [c for c in df.columns if c not in _PRIMARY_COLS]
    df[ordered].to_csv(csv_out, index=False)
    logger.info(f"Saved down-day RS CSV → {csv_out}  ({len(df)} stocks)")

    md_out = config.results_dir / f"down_day_rs_{label}.md"
    md_out.write_text(_build_md(df, label, corr_start, corr_end), encoding="utf-8")
    logger.info(f"Saved down-day RS MD  → {md_out}")
    return csv_out, md_out


def _build_md(
    df: pd.DataFrame,
    label: str,
    corr_start: pd.Timestamp | None,
    corr_end: pd.Timestamp | None,
) -> str:
    rw = int(df["rank_window"].iloc[0]) if "rank_window" in df.columns and not df.empty else RANK_WINDOW
    L: list[str] = []
    L.append(f"# Down-Day Relative Strength — {label}")
    L.append("")
    if corr_start is not None:
        L.append(f"> Correction window: **{corr_start.date()}** → **{corr_end.date()}**  ·  "
                 f"ranking window: trailing **{rw}** trading days  ·  benchmark **{BENCHMARK}**")
    else:
        L.append(f"> Ranking window: trailing **{rw}** trading days  ·  benchmark **{BENCHMARK}**  "
                 f"(no correction window resolved)")
    L.append("")
    L.append("> `down_capture` = stock's mean return on SPY-down days ÷ SPY's mean return on "
             "those days. **< 1 = cushioned, ≤ 0 = rose while the market fell.**")
    L.append("")

    dc, dw, sp = f"down_capture_{rw}", f"down_winrate_{rw}", f"capture_spread_{rw}"
    uc, nd = f"up_capture_{rw}", f"n_down_days_{rw}"

    for sector in df["sector_name"].dropna().unique():
        sub = df[df["sector_name"] == sector].sort_values("down_day_rank", ascending=False)
        if sub.empty:
            continue
        L.append(f"## {sector}  ({len(sub)})")
        L.append("")
        L.append("| Rank | Ticker | Industry | DDR | Bucket DDR | Sig | "
                 "DownCap | UpCap | Spread | WinRate | DnDays | SCTR |")
        L.append("|-----:|--------|----------|:---:|:----------:|:---:|"
                 "--------:|------:|-------:|--------:|-------:|-----:|")
        for _, r in sub.iterrows():
            L.append(
                f"| {int(r['rank'])} | {r['ticker']} | {r['industry_name']} "
                f"| {_num(r.get('down_day_rank'))} | {_num(r.get('ddr_in_bucket'))} "
                f"| {_int(r.get('signal_count'))} "
                f"| {_num(r.get(dc), 2)} | {_num(r.get(uc), 2)} | {_num(r.get(sp), 2)} "
                f"| {_pct(r.get(dw))} | {_int(r.get(nd))} | {_num(r.get('sctr'))} |"
            )
        L.append("")

    L.append("## Methodology")
    L.append("")
    L.append(f"- **Down day** — a session where {BENCHMARK} closed below 0%. Metrics use only those days.")
    L.append(f"- **down_capture** — mean stock return on down days ÷ mean {BENCHMARK} return on down days.")
    L.append("- **capture_spread** — up_capture − down_capture. High up-participation with a low "
             "down-capture is the O'Neil correction-leader profile.")
    L.append(f"- **signal_count (0–4)** — down_capture < {DOWN_CAPTURE_MAX}; down_capture ≤ 0; "
             f"down_winrate ≥ {DOWN_WINRATE_MIN}; capture_spread ≥ {SPREAD_MIN}.")
    L.append("- **down_day_rank** — 0.50 × down_alpha percentile + 0.30 × inverted down_capture "
             "percentile + 0.20 × spread percentile, re-ranked 0–99 across all scored stocks. "
             "`ddr_in_bucket` / `ddr_in_industry` are the same blend re-ranked within peer group.")
    L.append("")
    return "\n".join(L)


def _num(v, nd: int = 1) -> str:
    return f"{v:.{nd}f}" if pd.notna(v) else "—"


def _int(v) -> str:
    return f"{int(v)}" if pd.notna(v) else "—"


def _pct(v) -> str:
    return f"{v*100:.0f}%" if pd.notna(v) else "—"


# ── Terminal report ──────────────────────────────────────────────────────

def print_report(df: pd.DataFrame, corr_start=None, corr_end=None, top_n: int = 30) -> None:
    DIV = "─" * 104
    if df.empty:
        print("\n  Down-Day RS: no stocks passed the filter.")
        return
    rw = int(df["rank_window"].iloc[0])
    dc, dw, sp = f"down_capture_{rw}", f"down_winrate_{rw}", f"capture_spread_{rw}"
    uc, nd = f"up_capture_{rw}", f"n_down_days_{rw}"

    win = f"  |  correction {corr_start.date()} → {corr_end.date()}" if corr_start is not None else ""
    print(f"\n  Down-Day Relative Strength  |  rank window {rw}d  |  {len(df)} stocks{win}")

    for sector in df["sector_name"].dropna().unique():
        sub = df[df["sector_name"] == sector].sort_values("down_day_rank", ascending=False)
        if sub.empty:
            continue
        print(f"\n{DIV}\n  {sector}  ({len(sub)})\n{DIV}")
        print(f"  {'Rnk':>4}  {'Ticker':<7} {'Industry':<32} {'DDR':>5}  {'Bkt':>5}  "
              f"{'Sig':>3}  {'DnCap':>6}  {'UpCap':>6}  {'Sprd':>6}  {'Win':>5}  {'Dn':>3}")
        print(DIV)
        for j, (_, r) in enumerate(sub.iterrows()):
            if j >= top_n:
                print(f"  ... {len(sub) - top_n} more")
                break
            print(
                f"  {int(r['rank']):>4}  {str(r['ticker']):<7} {str(r['industry_name'])[:32]:<32} "
                f"{_num(r.get('down_day_rank')):>5}  {_num(r.get('ddr_in_bucket')):>5}  "
                f"{_int(r.get('signal_count')):>3}  "
                f"{_num(r.get(dc), 2):>6}  {_num(r.get(uc), 2):>6}  {_num(r.get(sp), 2):>6}  "
                f"{_pct(r.get(dw)):>5}  {_int(r.get(nd)):>3}"
            )
    print(f"\n{DIV}")
