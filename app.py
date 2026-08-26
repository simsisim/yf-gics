"""
StockCharts interactive dashboard.
Run:  streamlit run app.py
"""

import glob
import json
import re
import sqlite3

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="StockCharts Dashboard",
    page_icon="📈",
    layout="wide",
)

# ── palette (mirrors renderer.py dark theme) ──────────────────────────────────
_BG      = "#0d1117"
_BG_CARD = "#161b22"
_BORDER  = "#30363d"
_FG      = "#e6edf3"
_DIM     = "#8b949e"
_GREEN   = "#2ea44f"
_RED     = "#cf222e"

st.markdown(
    f"""
    <style>
        .stApp {{ background-color: {_BG}; color: {_FG}; }}
        section[data-testid="stSidebar"] {{ background-color: {_BG_CARD}; }}
        div[data-testid="stSidebar"] .stMarkdown p {{ color: {_DIM}; font-size: 0.8rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

DB_PATH = "data/yf_dashboard.db"

# ── screener: curated category taxonomy over the 141 scanIDs archived daily by
# src/fetchers/scan_reports.py — NOT StockCharts' own 20 groups (those mix
# unrelated signals together, e.g. "Popular Bullish Scans" bundles RSI, MACD,
# Bollinger, CCI, Parabolic SAR, Aroon, ADX and Chaikin Money Flow into one
# bucket, and splits RSI's bullish/bearish scans into two different groups).
# Built once by scanID naming pattern (see to_do/stockcharts_api_surface.md
# "Naming conventions"), cross-checked against the live 141 — all accounted for.
_MARKET_CAP_BUCKETS = ["0", "100M", "500M", "1B", "2B", "5B", "10B", "20B", "50B", "100B", "500B", "∞"]


# ── DB helpers (cached 5 min) ─────────────────────────────────────────────────

@st.cache_data(ttl=300)
def available_dates_industry() -> list[str]:
    with sqlite3.connect(DB_PATH) as c:
        return [r[0] for r in c.execute(
            "SELECT DISTINCT snapshot_date FROM industry_summary ORDER BY snapshot_date DESC"
        ).fetchall()]


def _group_where(group_name: "str | tuple[str, ...]") -> "tuple[str, list, int]":
    """SQL WHERE fragment + params for one or more sctr_rankings group_name values."""
    if isinstance(group_name, tuple):
        placeholders = ",".join("?" for _ in group_name)
        return f"group_name IN ({placeholders})", list(group_name), len(group_name)
    return "group_name=?", [group_name], 1


def _dedup_symbols(
    df: pd.DataFrame, group_name: "str | tuple[str, ...]", subset: "list[str] | None" = None,
) -> pd.DataFrame:
    """Drop duplicate symbols in a combined-group frame (a stock can briefly appear in two
    cap-tier groups at once, e.g. mid-reclassification) — keeps the row from whichever group
    is listed first in the combo, so the result stays one row per symbol per date."""
    if not isinstance(group_name, tuple) or "group_name" not in df.columns:
        return df
    subset = subset or ["symbol"]
    if not df.duplicated(subset=subset).any():
        return df
    pref = {g: i for i, g in enumerate(group_name)}
    return (
        df.assign(_pref=df["group_name"].map(pref))
          .sort_values("_pref")
          .drop_duplicates(subset=subset, keep="first")
          .drop(columns="_pref")
    )


@st.cache_data(ttl=300)
def available_dates_sctr(group_name: "str | tuple[str, ...]") -> list[str]:
    where, params, n_groups = _group_where(group_name)
    # for a combo of groups, only offer dates where every group in the combo has data
    having = f" GROUP BY snapshot_date HAVING COUNT(DISTINCT group_name) = {n_groups}" if n_groups > 1 else ""
    select = "snapshot_date" if n_groups > 1 else "DISTINCT snapshot_date"
    with sqlite3.connect(DB_PATH) as c:
        return [r[0] for r in c.execute(
            f"SELECT {select} FROM sctr_rankings WHERE {where}{having} ORDER BY snapshot_date DESC",
            params,
        ).fetchall()]


@st.cache_data(ttl=300)
def load_industry(snapshot_date: str) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as c:
        return pd.read_sql_query(
            "SELECT * FROM industry_summary WHERE snapshot_date=? ORDER BY sctr DESC",
            c, params=(snapshot_date,),
        )


@st.cache_data(ttl=300)
def load_all_industry() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as c:
        return pd.read_sql_query(
            "SELECT * FROM industry_summary ORDER BY snapshot_date DESC, sctr DESC",
            c,
        )


@st.cache_data(ttl=300)
def available_sctr_groups() -> list[str]:
    with sqlite3.connect(DB_PATH) as c:
        return [r[0] for r in c.execute(
            "SELECT DISTINCT group_name FROM sctr_rankings ORDER BY group_name"
        ).fetchall()]


@st.cache_data(ttl=300)
def load_sctr(snapshot_date: str, group_name: str) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as c:
        return pd.read_sql_query(
            "SELECT * FROM sctr_rankings WHERE snapshot_date=? AND group_name=? ORDER BY sctr DESC",
            c, params=(snapshot_date, group_name),
        )


@st.cache_data(ttl=300)
def load_all_sctr_for_group(group_name: "str | tuple[str, ...]") -> pd.DataFrame:
    where, params, _ = _group_where(group_name)
    with sqlite3.connect(DB_PATH) as c:
        df = pd.read_sql_query(
            f"""SELECT snapshot_date, symbol, name, sector, industry,
                      sctr, sctr_delta, close, group_name
               FROM sctr_rankings WHERE {where}
               ORDER BY snapshot_date DESC, sctr DESC""",
            c, params=params,
        )
    return _dedup_symbols(df, group_name, subset=["snapshot_date", "symbol"])


def build_compact_sctr_df(
    all_df: pd.DataFrame,
    today: str,
    dates: list[str],
    bench_ser: "pd.Series | None" = None,
) -> pd.DataFrame:
    """Rolling-window compact table for SCTR — rank + delta columns, no per-date columns.

    bench_ser: optional Series indexed by snapshot_date; when set, all price-% columns
    are ratio-adjusted vs the benchmark. SCTR columns are unaffected.
    """
    all_dates = [today] + dates

    def _delta_str(old_rank, new_rank):
        if pd.isna(old_rank) or pd.isna(new_rank):
            return "—"
        d = int(old_rank) - int(new_rank)
        return f"↑{d}" if d > 0 else (f"↓{abs(d)}" if d < 0 else "—")

    # compute rank series per date
    rank_frames: dict[str, pd.Series] = {}
    for d in all_dates:
        day = all_df[all_df["snapshot_date"] == d][["symbol", "sctr"]].copy()
        day = day.sort_values("sctr", ascending=False, na_position="last").reset_index(drop=True)
        rank_frames[d] = pd.Series(range(1, len(day) + 1), index=day["symbol"])

    today_ranks = rank_frames[today]

    base = all_df[all_df["snapshot_date"] == today][
        ["symbol", "name", "sector", "industry", "sctr", "sctr_delta", "close", "group_name"]
    ].copy()
    base = base.sort_values("sctr", ascending=False, na_position="last").reset_index(drop=True)
    base.insert(0, "Rank", base["symbol"].map(today_ranks))

    # all available dates for calendar-based period lookups (may extend beyond window)
    all_avail_desc = sorted(all_df["snapshot_date"].unique(), reverse=True)
    all_avail_desc = [d for d in all_avail_desc if d < today]   # exclude today itself

    # SCTR deltas — calendar-based
    for dcol, delta in [("Δ SCTR 1W", 7), ("Δ SCTR 1M", 30)]:
        ref = _snap_for_delta(today, all_avail_desc, delta)
        if ref is None:
            continue
        past_sctr = all_df[all_df["snapshot_date"] == ref][["symbol", "sctr"]].set_index("symbol")["sctr"]
        base[dcol] = base.apply(
            lambda r, p=past_sctr: round(r["sctr"] - p[r["symbol"]], 1)
            if r["symbol"] in p.index else float("nan"),
            axis=1,
        )

    # price pct changes — calendar-based, extended periods
    bench_today = _bench_at(bench_ser, today) if bench_ser is not None else None
    _price_specs = [
        ("1D%",  1), ("1W%",  7), ("1M%", 30),
        ("3M%", 90), ("YTD%", None),
    ]
    for pcol, delta in _price_specs:
        ref = (
            _snap_for_ytd(today, all_avail_desc) if delta is None
            else _snap_for_delta(today, all_avail_desc, delta)
        )
        if ref is None:
            continue
        past_close = all_df[all_df["snapshot_date"] == ref][["symbol", "close"]].set_index("symbol")["close"]
        bench_ref = _bench_at(bench_ser, ref) if bench_ser is not None else None

        if bench_ser is not None and bench_today is not None and bench_ref is not None:
            base[pcol] = base.apply(
                lambda r, p=past_close, bn=bench_today, bp=bench_ref: (
                    _rel_pct(r["close"], p[r["symbol"]], bn, bp)
                    if r["symbol"] in p.index else float("nan")
                ), axis=1,
            )
        else:
            base[pcol] = base.apply(
                lambda r, p=past_close: (
                    (r["close"] - p[r["symbol"]]) / p[r["symbol"]] * 100
                    if r["symbol"] in p.index and p[r["symbol"]] != 0 else float("nan")
                ), axis=1,
            )

    # rank delta columns
    if len(dates) >= 1:
        prev = rank_frames[dates[0]]
        base["Δ Rank 1d"] = base["symbol"].apply(
            lambda s: _delta_str(prev.get(s), today_ranks.get(s))
        )
    if len(dates) >= 5:
        w1 = rank_frames[dates[4]]
        base["Δ Rank 1w"] = base["symbol"].apply(
            lambda s: _delta_str(w1.get(s), today_ranks.get(s))
        )
    if len(dates) >= 1:
        oldest = rank_frames[dates[-1]]
        base[f"Δ Rank vs {_fmt_date(dates[-1])}"] = base["symbol"].apply(
            lambda s: _delta_str(oldest.get(s), today_ranks.get(s))
        )

    return base


@st.cache_data(ttl=300)
def load_sctr_enriched(
    snapshot_date: str,
    group_name: "str | tuple[str, ...]",
    all_dates: tuple[str, ...],
    compare_date: str | None = None,
    bench_symbol: str | None = None,
) -> pd.DataFrame:
    """SCTR snapshot enriched with computed pct/delta columns and optional rank comparison.

    When bench_symbol is set, all price-% columns are ratio-adjusted relative to that benchmark.
    SCTR columns are never adjusted.
    """
    hist = [d for d in all_dates if d < snapshot_date]
    where, params, n_groups = _group_where(group_name)

    # pre-load benchmark series once (cached separately)
    bench_ser: "pd.Series | None" = None
    if bench_symbol:
        _b = load_benchmark_series(bench_symbol)
        if not _b.empty:
            bench_ser = _b.set_index("snapshot_date")["close"]

    with sqlite3.connect(DB_PATH) as c:
        group_cols = ", group_name" if n_groups > 1 else ""
        df = pd.read_sql_query(
            f"""SELECT symbol, name, sector, industry, sctr, sctr_delta, close, volume, market_cap{group_cols}
               FROM sctr_rankings WHERE snapshot_date=? AND {where} ORDER BY sctr DESC""",
            c, params=[snapshot_date] + params,
        )
        df = _dedup_symbols(df, group_name).sort_values(
            "sctr", ascending=False, na_position="last"
        ).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))

        def _past(date: str) -> pd.DataFrame:
            p = pd.read_sql_query(
                f"SELECT symbol, sctr, close{group_cols} FROM sctr_rankings WHERE snapshot_date=? AND {where}",
                c, params=[date] + params,
            )
            return _dedup_symbols(p, group_name).set_index("symbol")

        bench_today = _bench_at(bench_ser, snapshot_date) if bench_ser is not None else None

        # fixed periods: calendar-based lookup so gaps don't skew the window
        _period_specs = [
            ("1D%",   None,          1),
            ("1W%",   "Δ SCTR 1W",   7),
            ("1M%",   "Δ SCTR 1M",  30),
            ("3M%",   None,          90),
            ("YTD%",  None,         None),   # handled separately
        ]
        for pct_col, dsctr_col, delta in _period_specs:
            ref = (
                _snap_for_ytd(snapshot_date, hist) if delta is None
                else _snap_for_delta(snapshot_date, hist, delta)
            )
            if ref is None:
                continue
            p = _past(ref)
            bench_ref = _bench_at(bench_ser, ref) if bench_ser is not None else None

            if bench_ser is not None and bench_today is not None and bench_ref is not None:
                df[pct_col] = df.apply(
                    lambda r, p=p, bn=bench_today, bp=bench_ref: (
                        _rel_pct(r["close"], p.at[r["symbol"], "close"], bn, bp)
                        if r["symbol"] in p.index else float("nan")
                    ), axis=1,
                )
            else:
                df[pct_col] = df.apply(
                    lambda r, p=p: (
                        (r["close"] - p.at[r["symbol"], "close"]) / p.at[r["symbol"], "close"] * 100
                        if r["symbol"] in p.index and p.at[r["symbol"], "close"] != 0
                        else float("nan")
                    ), axis=1,
                )
            if dsctr_col:
                df[dsctr_col] = df.apply(
                    lambda r, p=p: (
                        r["sctr"] - p.at[r["symbol"], "sctr"]
                        if r["symbol"] in p.index else float("nan")
                    ), axis=1,
                )

        if compare_date:
            past_cmp = pd.read_sql_query(
                f"SELECT symbol, sctr, close{group_cols} FROM sctr_rankings WHERE snapshot_date=? AND {where} ORDER BY sctr DESC",
                c, params=[compare_date] + params,
            )
            past_cmp = _dedup_symbols(past_cmp, group_name).sort_values(
                "sctr", ascending=False, na_position="last"
            ).reset_index(drop=True)
            past_ranks   = pd.Series(range(1, len(past_cmp) + 1), index=past_cmp["symbol"])
            past_sctr_s  = past_cmp.set_index("symbol")["sctr"]
            past_close_s = past_cmp.set_index("symbol")["close"]

            lbl = _fmt_date(compare_date)
            df["Δ Rank"] = df.apply(
                lambda r: (
                    int(past_ranks[r["symbol"]]) - r["Rank"]
                    if r["symbol"] in past_ranks.index else float("nan")
                ), axis=1,
            ).astype("Int64")

            bench_cmp = _bench_at(bench_ser, compare_date) if bench_ser is not None else None
            if bench_ser is not None and bench_today is not None and bench_cmp is not None:
                df[f"Δ Price% {lbl}"] = df.apply(
                    lambda r, pc=past_close_s, bn=bench_today, bp=bench_cmp: (
                        _rel_pct(r["close"], pc[r["symbol"]], bn, bp)
                        if r["symbol"] in pc.index and pc[r["symbol"]] != 0 else float("nan")
                    ), axis=1,
                )
            else:
                df[f"Δ Price% {lbl}"] = df.apply(
                    lambda r, pc=past_close_s: (
                        (r["close"] - pc[r["symbol"]]) / pc[r["symbol"]] * 100
                        if r["symbol"] in pc.index and pc[r["symbol"]] != 0 else float("nan")
                    ), axis=1,
                )

            df[f"Δ SCTR {lbl}"] = df.apply(
                lambda r, ps=past_sctr_s: (
                    r["sctr"] - ps[r["symbol"]]
                    if r["symbol"] in ps.index else float("nan")
                ), axis=1,
            )

    return df


@st.cache_data(ttl=300)
def available_benchmarks() -> list[str]:
    """All symbols in the benchmarks table (indices + stocks)."""
    with sqlite3.connect(DB_PATH) as c:
        try:
            return [r[0] for r in c.execute(
                "SELECT DISTINCT symbol FROM benchmarks ORDER BY symbol"
            ).fetchall()]
        except sqlite3.OperationalError:
            return []


@st.cache_data(ttl=300)
def load_index_benchmarks() -> list[str]:
    """Only index symbols (^GSPC, ^NDX …) — used as ratio denominators."""
    with sqlite3.connect(DB_PATH) as c:
        try:
            return [r[0] for r in c.execute(
                "SELECT DISTINCT symbol FROM benchmarks WHERE symbol LIKE '^%' ORDER BY symbol"
            ).fetchall()]
        except sqlite3.OperationalError:
            return []



@st.cache_data(ttl=300)
def load_benchmark_series(symbol: str) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as c:
        return pd.read_sql_query(
            "SELECT snapshot_date, close FROM benchmarks WHERE symbol=? ORDER BY snapshot_date",
            c, params=(symbol,),
        )








# ── screener ──────────────────────────────────────────────────────────────────













@st.cache_data(ttl=300)
def load_industry_stocks(
    industry_name: str,
    snapshot_date: str,
    all_dates: tuple[str, ...],
) -> pd.DataFrame:
    """Stocks in an industry with pct changes computed from historical close prices."""
    hist = [d for d in all_dates if d < snapshot_date]

    with sqlite3.connect(DB_PATH) as c:
        df = pd.read_sql_query(
            """
            SELECT symbol, name, group_name, sctr, sctr_delta, close, volume, market_cap
            FROM sctr_rankings
            WHERE snapshot_date = ? AND industry = ?
            ORDER BY sctr DESC
            """,
            c, params=(snapshot_date, industry_name),
        ).drop_duplicates(subset=["symbol"], keep="first")

        if df.empty:
            return df

        def _closes_at(date: str) -> pd.Series:
            h = pd.read_sql_query(
                "SELECT symbol, close FROM sctr_rankings WHERE snapshot_date=? AND industry=?",
                c, params=(date, industry_name),
            ).drop_duplicates("symbol").set_index("symbol")["close"]
            return h

        timeframes = [("pct_1d", 0), ("pct_1w", 4), ("pct_1m", 19)]
        for col, idx in timeframes:
            if len(hist) > idx:
                past = _closes_at(hist[idx])
                df[col] = df.apply(
                    lambda r, p=past: (
                        (r["close"] - p[r["symbol"]]) / p[r["symbol"]] * 100
                        if r["symbol"] in p and p[r["symbol"]] != 0 else float("nan")
                    ),
                    axis=1,
                )

    return df


@st.cache_data(ttl=300)
def load_all_sctr_multi(groups: tuple[str, ...]) -> pd.DataFrame:
    """Per-stock sctr/industry/market_cap/group_name across multiple groups, all dates."""
    ph = ",".join("?" * len(groups))
    with sqlite3.connect(DB_PATH) as c:
        return pd.read_sql_query(
            f"""SELECT snapshot_date, symbol, sector, industry, sctr, market_cap, group_name
                FROM sctr_rankings WHERE group_name IN ({ph})""",
            c, params=groups,
        )


@st.cache_data(ttl=300)
def _sctr_avail_dates(group_name: str) -> list[str]:
    with sqlite3.connect(DB_PATH) as c:
        return [r[0] for r in c.execute(
            "SELECT DISTINCT snapshot_date FROM sctr_rankings WHERE group_name=? ORDER BY snapshot_date DESC",
            (group_name,),
        ).fetchall()]


@st.cache_data(ttl=300)
def load_sctr_leaders(
    snapshot_date: str,
    groups: tuple[str, ...],
    min_sctr: float,
    persist_n: int = 1,
    ref_1w: str | None = None,
    ref_1m: str | None = None,
    ref_3m: str | None = None,
) -> pd.DataFrame:
    ph = ",".join("?" * len(groups))
    _NEVER = "1900-01-01"  # sentinel: LEFT JOIN returns NULL when ref date absent

    # ── optional CTE for persistence filter ──────────────────────────────────
    if persist_n > 1:
        cte = f"""
WITH persisted AS (
    SELECT symbol, group_name
    FROM sctr_rankings
    WHERE group_name IN ({ph})
      AND sctr >= ?
      AND snapshot_date IN (
          SELECT DISTINCT snapshot_date
          FROM sctr_rankings
          WHERE group_name IN ({ph})
          ORDER BY snapshot_date DESC
          LIMIT ?
      )
    GROUP BY symbol, group_name
    HAVING COUNT(*) >= ?
)"""
        persist_join = "JOIN persisted pers ON pers.symbol=t.symbol AND pers.group_name=t.group_name"
        cte_params   = [*groups, min_sctr, *groups, persist_n, persist_n]
    else:
        cte = ""
        persist_join = ""
        cte_params   = []

    sql = f"""
{cte}
SELECT
    t.snapshot_date, t.group_name, t.symbol, t.name,
    t.sector, t.industry, t.sctr, t.sctr_delta,
    t.close, t.volume, t.market_cap,
    CASE WHEN prev.close > 0
         THEN ROUND((t.close - prev.close) / prev.close * 100, 2)
         ELSE NULL END AS pct_1d,
    CASE WHEN w.close > 0
         THEN ROUND((t.close - w.close) / w.close * 100, 2)
         ELSE NULL END AS pct_1w,
    CASE WHEN m.close > 0
         THEN ROUND((t.close - m.close) / m.close * 100, 2)
         ELSE NULL END AS pct_1m,
    CASE WHEN q.close > 0
         THEN ROUND((t.close - q.close) / q.close * 100, 2)
         ELSE NULL END AS pct_3m
FROM sctr_rankings t
LEFT JOIN sctr_rankings prev
    ON  prev.symbol     = t.symbol
    AND prev.group_name = t.group_name
    AND prev.snapshot_date = (
        SELECT MAX(p.snapshot_date) FROM sctr_rankings p
        WHERE p.symbol=t.symbol AND p.group_name=t.group_name
          AND p.snapshot_date < t.snapshot_date
    )
LEFT JOIN sctr_rankings w
    ON w.symbol=t.symbol AND w.group_name=t.group_name AND w.snapshot_date=?
LEFT JOIN sctr_rankings m
    ON m.symbol=t.symbol AND m.group_name=t.group_name AND m.snapshot_date=?
LEFT JOIN sctr_rankings q
    ON q.symbol=t.symbol AND q.group_name=t.group_name AND q.snapshot_date=?
{persist_join}
WHERE t.snapshot_date=? AND t.group_name IN ({ph}) AND t.sctr>=?
ORDER BY t.sector, t.industry, t.sctr DESC
"""
    params = [
        *cte_params,
        ref_1w or _NEVER, ref_1m or _NEVER, ref_3m or _NEVER,
        snapshot_date, *groups, min_sctr,
    ]
    with sqlite3.connect(DB_PATH) as c:
        return pd.read_sql_query(sql, c, params=params)


# ── constants ─────────────────────────────────────────────────────────────────

PERIOD_MAP = {
    "1 Day":    "pct_1d",
    "1 Week":   "pct_1w",
    "1 Month":  "pct_1m",
    "3 Months": "pct_3m",
    "6 Months": "pct_6m",
    "1 Year":   "pct_1y",
    "YTD":      "pct_ytd",
    "Custom":   None,
}

RANK_BY_MAP = {
    "SCTR":     "sctr",
    "1 Month%": "pct_1m",
    "3 Month%": "pct_3m",
    "YTD%":     "pct_ytd",
    "1 Year%":  "pct_1y",
}


# ── chart / table builders ────────────────────────────────────────────────────

def _base_layout(fig: go.Figure, height: int = 500) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG_CARD,
        font=dict(color=_FG, size=11),
        margin=dict(l=10, r=90, t=35, b=10),
        height=height,
        xaxis=dict(
            gridcolor=_BORDER,
            zerolinecolor=_BORDER,
            tickfont=dict(color=_DIM, size=9),
            tickformat="+.1f",
            title_font=dict(color=_DIM),
        ),
        yaxis=dict(
            gridcolor=_BORDER,
            tickfont=dict(size=9.5),
        ),
        hoverlabel=dict(bgcolor=_BG_CARD, bordercolor=_BORDER, font_size=12),
    )
    return fig


def build_theme_chart(df: pd.DataFrame, pct_col: str, ascending: bool) -> go.Figure:
    df = df.dropna(subset=[pct_col]).copy()
    df = df.sort_values(pct_col, ascending=ascending)

    colors = [_GREEN if v >= 0 else _RED for v in df[pct_col]]
    labels = df[pct_col].apply(lambda v: f"+{v:.2f}%" if v >= 0 else f"{v:.2f}%")

    pct_hover_cols = ["pct_1d", "pct_1w", "pct_1m", "pct_3m", "pct_ytd"]
    available_pct = [c for c in pct_hover_cols if c in df.columns]
    for c in pct_hover_cols:
        if c not in df.columns:
            df[c] = float("nan")
    hover_cols = df[["sector", "sctr"] + pct_hover_cols].fillna(float("nan"))

    if available_pct:
        hovertemplate = (
            "<b>%{y}</b><br>"
            "Sector: %{customdata[0]}<br>"
            "SCTR: %{customdata[1]:.1f}<br>"
            "1D: %{customdata[2]:+.2f}%  "
            "1W: %{customdata[3]:+.2f}%<br>"
            "1M: %{customdata[4]:+.2f}%  "
            "3M: %{customdata[5]:+.2f}%<br>"
            "YTD: %{customdata[6]:+.2f}%"
            "<extra></extra>"
        )
    else:
        hovertemplate = (
            "<b>%{y}</b><br>"
            "Sector: %{customdata[0]}<br>"
            "SCTR: %{customdata[1]:.1f}<br>"
            "<extra></extra>"
        )

    fig = go.Figure(go.Bar(
        x=df[pct_col],
        y=df["name"],
        orientation="h",
        marker_color=colors,
        text=labels,
        textposition="outside",
        textfont=dict(size=9, color=_FG),
        customdata=hover_cols.values,
        hovertemplate=hovertemplate,
    ))

    n = len(df)
    _base_layout(fig, height=max(420, n * 23 + 60))
    fig.update_layout(
        xaxis_title="% Change",
        bargap=0.3,
        shapes=[dict(
            type="line", x0=0, x1=0, y0=-0.5, y1=n - 0.5,
            line=dict(color=_BORDER, width=1),
        )],
    )
    return fig


def build_theme_diverging_chart(df: pd.DataFrame, pct_col: str) -> go.Figure:
    """Vertical diverging bar chart: gainers rise above the zero line, losers drop below.

    Industries sorted by pct value descending (best on the left, worst on the right)
    and plotted as a single column chart, colored green/red by sign.
    """
    df = df.dropna(subset=[pct_col]).copy()
    df = df.sort_values(pct_col, ascending=False)

    colors = [_GREEN if v >= 0 else _RED for v in df[pct_col]]
    labels = df[pct_col].apply(lambda v: f"+{v:.2f}%" if v >= 0 else f"{v:.2f}%")

    pct_hover_cols = ["pct_1d", "pct_1w", "pct_1m", "pct_3m", "pct_ytd"]
    available_pct = [c for c in pct_hover_cols if c in df.columns]
    for c in pct_hover_cols:
        if c not in df.columns:
            df[c] = float("nan")
    hover_cols = df[["sector", "sctr"] + pct_hover_cols].fillna(float("nan"))

    if available_pct:
        hovertemplate = (
            "<b>%{x}</b><br>"
            "Sector: %{customdata[0]}<br>"
            "SCTR: %{customdata[1]:.1f}<br>"
            "1D: %{customdata[2]:+.2f}%  "
            "1W: %{customdata[3]:+.2f}%<br>"
            "1M: %{customdata[4]:+.2f}%  "
            "3M: %{customdata[5]:+.2f}%<br>"
            "YTD: %{customdata[6]:+.2f}%"
            "<extra></extra>"
        )
    else:
        hovertemplate = (
            "<b>%{x}</b><br>"
            "Sector: %{customdata[0]}<br>"
            "SCTR: %{customdata[1]:.1f}"
            "<extra></extra>"
        )

    fig = go.Figure(go.Bar(
        x=df["name"],
        y=df[pct_col],
        marker_color=colors,
        text=labels,
        textposition="outside",
        textfont=dict(size=9, color=_FG),
        customdata=hover_cols.values,
        hovertemplate=hovertemplate,
    ))

    n = len(df)
    _base_layout(fig, height=560)
    fig.update_layout(
        yaxis_title="% Change",
        bargap=0.25,
        margin=dict(l=10, r=10, t=35, b=140),
    )
    fig.update_xaxes(
        tickangle=-45, tickfont=dict(color=_DIM, size=9), gridcolor=_BORDER,
    )
    fig.update_yaxes(
        tickformat="+.1f", tickfont=dict(size=9.5), gridcolor=_BORDER,
        zerolinecolor=_BORDER, zerolinewidth=1.5,
    )
    return fig




def _fmt_date(d: str) -> str:
    """'2026-06-15' → 'Jun 15'"""
    import datetime
    return datetime.date.fromisoformat(d).strftime("%b %-d")


def _nearest_earlier_date(picked: "datetime.date", available: list[str]) -> str | None:
    """Return the latest available snapshot date that is <= picked (available is DESC)."""
    iso = picked.isoformat()
    candidates = [d for d in available if d <= iso]
    return candidates[0] if candidates else None


def _rel_pct(now: float, past: float, bench_now: float, bench_past: float) -> float:
    """Ratio-based relative return vs benchmark, in percent.

    Returns ((now/past) / (bench_now/bench_past) - 1) * 100.
    Any NaN or zero-division produces NaN.
    """
    import math
    try:
        if past == 0 or bench_past == 0:
            return float("nan")
        v = ((now / past) / (bench_now / bench_past) - 1) * 100
        return float("nan") if math.isnan(v) or math.isinf(v) else v
    except Exception:
        return float("nan")


def _bench_at(bench_ser: "pd.Series", target_date: str) -> float | None:
    """Closest benchmark close on or before target_date.

    bench_ser: pandas Series indexed by snapshot_date strings (any order).
    """
    avail_desc = sorted(bench_ser.index, reverse=True)
    candidates = [d for d in avail_desc if d <= target_date]
    return float(bench_ser[candidates[0]]) if candidates else None


def _snap_for_delta(today_iso: str, available_desc: list[str], delta_days: int) -> str | None:
    """Return the most recent snapshot whose date <= (today − delta_days calendar days).

    Uses calendar days so that 1W → 7 days, 1M → 30 days, etc., regardless of how
    many trading-day snapshots are available. `available_desc` must be sorted DESC.
    """
    import datetime
    target = (datetime.date.fromisoformat(today_iso) - datetime.timedelta(days=delta_days)).isoformat()
    candidates = [d for d in available_desc if d <= target]
    return candidates[0] if candidates else None


def _snap_for_ytd(today_iso: str, available_desc: list[str]) -> str | None:
    """Return the last snapshot from the previous calendar year (Dec 31 or earlier)."""
    year = int(today_iso[:4])
    target = f"{year - 1}-12-31"
    candidates = [d for d in available_desc if d <= target]
    return candidates[0] if candidates else None


def build_ranks_df(
    all_df: pd.DataFrame,
    rank_by: str,
    today: str,
    compare: list[str],
    bench_ser: "pd.Series | None" = None,
) -> pd.DataFrame:
    use_dates = [today] + compare

    # compute rank for each date
    rank_frames: dict[str, pd.DataFrame] = {}
    for d in use_dates:
        day = all_df[all_df["snapshot_date"] == d][["symbol", rank_by]].copy()
        day = day.sort_values(rank_by, ascending=False, na_position="last").reset_index(drop=True)
        col = _fmt_date(d)
        day[col] = range(1, len(day) + 1)
        rank_frames[d] = day[["symbol", col]]

    base = all_df[all_df["snapshot_date"] == today][
        ["symbol", "name", "sector", "sctr", "last", "pct_1m", "pct_3m", "pct_ytd"]
    ].copy()
    base = base.sort_values(rank_by, ascending=False, na_position="last").reset_index(drop=True)

    for d in use_dates:
        base = base.merge(rank_frames[d], on="symbol", how="left")

    today_col  = _fmt_date(today)
    oldest_col = _fmt_date(use_dates[-1])
    if today_col != oldest_col:
        base["Δ Rank"] = (base[oldest_col].fillna(0) - base[today_col].fillna(0)).astype("Int64")

        # Δ Price% vs the compare date
        cmp_date = use_dates[-1]
        lbl = _fmt_date(cmp_date)
        past_last = all_df[all_df["snapshot_date"] == cmp_date][["symbol", "last"]].set_index("symbol")["last"]

        bench_today = _bench_at(bench_ser, today)    if bench_ser is not None else None
        bench_cmp   = _bench_at(bench_ser, cmp_date) if bench_ser is not None else None

        if bench_ser is not None and bench_today is not None and bench_cmp is not None:
            base[f"Δ Price% {lbl}"] = base.apply(
                lambda r, p=past_last, bn=bench_today, bp=bench_cmp: (
                    _rel_pct(r["last"], p[r["symbol"]], bn, bp)
                    if r["symbol"] in p.index and p[r["symbol"]] != 0 else float("nan")
                ), axis=1,
            )
        else:
            base[f"Δ Price% {lbl}"] = base.apply(
                lambda r, p=past_last: (
                    (r["last"] - p[r["symbol"]]) / p[r["symbol"]] * 100
                    if r["symbol"] in p.index and p[r["symbol"]] != 0 else float("nan")
                ), axis=1,
            )

    base.insert(0, "Rank", range(1, len(base) + 1))
    base = base.rename(columns={
        "name": "Name", "sector": "Sector",
        "sctr": "SCTR", "pct_1m": "1M%",
        "pct_3m": "3M%", "pct_ytd": "YTD%",
    }).drop(columns=["symbol", "last"])

    return base


_SECTOR_COLORS = {
    "Communication Services Sector": "#4e9fd4",
    "Consumer Discretionary Sector": "#f0b429",
    "Consumer Staples Sector":       "#82c91e",
    "Energy Sector":                 "#e8590c",
    "Financial Sector":              "#5c7cfa",
    "Health Care Sector":            "#f06595",
    "Industrial Sector":             "#74c0fc",
    "Materials Sector":              "#a9e34b",
    "Real Estate Sector":            "#e599f7",
    "Technology Sector":             "#63e6be",
    "Utilities Sector":              "#ffa94d",
}
_DEFAULT_SECTOR_COLOR = _DIM


def build_compact_ranks_df(
    all_df: pd.DataFrame,
    rank_by: str,
    today: str,
    dates: list[str],
    bench_ser: "pd.Series | None" = None,
) -> pd.DataFrame:
    """Wide-format rank table for Rolling Window mode.

    Returns today's industries sorted by rank, with delta columns instead of
    per-date rank columns. `dates` is sorted newest→oldest (today excluded).
    When bench_ser is set, pct columns are ratio-adjusted vs the benchmark.
    """
    all_dates = [today] + dates

    rank_frames: dict[str, pd.Series] = {}
    for d in all_dates:
        day = all_df[all_df["snapshot_date"] == d][["symbol", rank_by]].copy()
        day = day.sort_values(rank_by, ascending=False, na_position="last").reset_index(drop=True)
        rank_frames[d] = pd.Series(range(1, len(day) + 1), index=day["symbol"])

    today_ranks = rank_frames[today]

    _pct_src = ["pct_1d", "pct_1w", "pct_1m", "pct_3m", "pct_6m", "pct_1y", "pct_ytd"]
    _avail   = [c for c in _pct_src if c in all_df.columns]
    base = all_df[all_df["snapshot_date"] == today][
        ["symbol", "name", "sector", "sctr"] + _avail
    ].copy()
    base = base.sort_values(rank_by, ascending=False, na_position="last").reset_index(drop=True)
    base["Rank"] = base["symbol"].map(today_ranks)

    def _delta_str(old_rank, new_rank):
        if pd.isna(old_rank) or pd.isna(new_rank):
            return "—"
        d = int(old_rank) - int(new_rank)
        if d > 0:
            return f"↑{d}"
        if d < 0:
            return f"↓{abs(d)}"
        return "—"

    if len(dates) >= 1:
        prev = rank_frames[dates[0]]
        base["Δ Rank 1d"] = base["symbol"].apply(
            lambda s: _delta_str(prev.get(s), today_ranks.get(s))
        )
    if len(dates) >= 5:
        w1 = rank_frames[dates[4]]
        base["Δ Rank 1w"] = base["symbol"].apply(
            lambda s: _delta_str(w1.get(s), today_ranks.get(s))
        )
    if len(dates) >= 1:
        oldest = rank_frames[dates[-1]]
        lbl = _fmt_date(dates[-1])
        base[f"Δ Rank vs {lbl}"] = base["symbol"].apply(
            lambda s: _delta_str(oldest.get(s), today_ranks.get(s))
        )

    # benchmark adjustment — recompute pct columns from `last` prices
    if bench_ser is not None and "last" in all_df.columns:
        bench_today = _bench_at(bench_ser, today)
        all_avail_desc = sorted(all_df["snapshot_date"].unique(), reverse=True)
        all_avail_desc = [d for d in all_avail_desc if d < today]
        _ind_price_periods = [
            ("pct_1d",  1), ("pct_1w",  7), ("pct_1m", 30),
            ("pct_3m", 90), ("pct_6m", 180), ("pct_1y", 365),
            ("pct_ytd", None),
        ]
        today_last = all_df[all_df["snapshot_date"] == today][["symbol", "last"]].set_index("symbol")["last"]
        for src_col, delta in _ind_price_periods:
            if src_col not in _avail:
                continue
            ref = (
                _snap_for_ytd(today, all_avail_desc) if delta is None
                else _snap_for_delta(today, all_avail_desc, delta)
            )
            if ref is None:
                continue
            bench_ref = _bench_at(bench_ser, ref)
            if bench_today is None or bench_ref is None:
                continue
            past_last = all_df[all_df["snapshot_date"] == ref][["symbol", "last"]].set_index("symbol")["last"]
            base[src_col] = base.apply(
                lambda r, pl=past_last, tl=today_last, bn=bench_today, bp=bench_ref: (
                    _rel_pct(tl.get(r["symbol"], float("nan")),
                             pl.get(r["symbol"], float("nan")), bn, bp)
                ), axis=1,
            )

    _rename = {
        "name": "Name", "sector": "Sector", "sctr": "SCTR",
        "pct_1d": "1D%", "pct_1w": "1W%", "pct_1m": "1M%",
        "pct_3m": "3M%", "pct_6m": "6M%", "pct_1y": "1Y%", "pct_ytd": "YTD%",
    }
    base = base.rename(columns=_rename).drop(columns=["symbol"])

    delta_cols = [c for c in base.columns if c.startswith("Δ")]
    # return all metric columns; caller filters to user selection
    metric_cols = ["SCTR"] + [_rename[c] for c in _avail]
    ordered = ["Rank", "Name", "Sector"] + metric_cols + delta_cols
    return base[[c for c in ordered if c in base.columns]]


def build_bump_chart(
    all_df: pd.DataFrame,
    rank_by: str,
    today: str,
    dates: list[str],
    top_n: int = 20,
    bottom_n: int = 0,
) -> go.Figure:
    """Bump (slope) chart: rank of selected industries over the date window.

    top_n    — include the top-ranked N industries (by today's rank).
    bottom_n — also include the bottom-ranked N industries (worst ranks).
    Y-axis is inverted so rank 1 appears at the top.
    Lines are coloured by sector.
    `dates` sorted newest→oldest (today excluded).
    """
    all_dates = sorted([today] + dates)

    rank_by_date: dict[str, pd.Series] = {}
    for d in all_dates:
        day = all_df[all_df["snapshot_date"] == d][["symbol", rank_by]].copy()
        day = day.sort_values(rank_by, ascending=False, na_position="last").reset_index(drop=True)
        rank_by_date[d] = pd.Series(range(1, len(day) + 1), index=day["symbol"])

    today_ranks = rank_by_date[today]
    top_symbols    = list(today_ranks.nsmallest(top_n).index) if top_n > 0 else []
    bottom_symbols = list(today_ranks.nlargest(bottom_n).index) if bottom_n > 0 else []
    symbols = list(dict.fromkeys(top_symbols + bottom_symbols))  # preserve order, deduplicate

    meta = all_df[all_df["snapshot_date"] == today][
        ["symbol", "name", "sector", "sctr"]
    ].set_index("symbol")

    x_labels = [_fmt_date(d) for d in all_dates]

    fig = go.Figure()

    for sym in symbols:
        if sym not in meta.index:
            continue
        row   = meta.loc[sym]
        name  = row["name"]
        sector = row.get("sector", "")
        sctr  = row.get("sctr", float("nan"))
        color = _SECTOR_COLORS.get(sector, _DEFAULT_SECTOR_COLOR)

        y_vals = [rank_by_date[d].get(sym, None) for d in all_dates]

        fig.add_trace(go.Scatter(
            x=x_labels,
            y=y_vals,
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=1.5),
            marker=dict(color=color, size=6),
            customdata=[[sector, sctr]] * len(all_dates),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Rank: %{y}<br>"
                "Date: %{x}<br>"
                "Sector: %{customdata[0]}<br>"
                "SCTR: %{customdata[1]:.1f}"
                "<extra></extra>"
            ),
            text=[name] * len(all_dates),
            showlegend=False,
        ))

    _base_layout(fig, height=520)
    fig.update_layout(
        yaxis=dict(
            autorange="reversed",
            title="Rank",
            dtick=5,
            gridcolor=_BORDER,
            tickfont=dict(size=9),
        ),
        xaxis=dict(
            title="",
            gridcolor=_BORDER,
            tickfont=dict(color=_DIM, size=9),
            tickformat="",
        ),
        margin=dict(l=50, r=10, t=35, b=40),
        hovermode="closest",
    )
    return fig


def compute_nday_pct(
    today_date: str,
    n_days: int,
    all_df: pd.DataFrame,
) -> pd.DataFrame | None:
    """Return % change of `last` price over the past N calendar days using DB snapshots."""
    import datetime
    today_dt  = datetime.date.fromisoformat(today_date)
    target_dt = today_dt - datetime.timedelta(days=n_days)

    dates = all_df["snapshot_date"].unique().tolist()
    past  = sorted([d for d in dates if datetime.date.fromisoformat(d) <= target_dt], reverse=True)
    if not past:
        return None

    past_date = past[0]
    df_now  = all_df[all_df["snapshot_date"] == today_date][["symbol", "name", "sector", "sctr", "last"]].copy()
    df_old  = all_df[all_df["snapshot_date"] == past_date][["symbol", "last"]].rename(columns={"last": "last_old"})

    merged = df_now.merge(df_old, on="symbol", how="inner")
    merged = merged[merged["last_old"] != 0]
    merged["pct_custom"] = (merged["last"] - merged["last_old"]) / merged["last_old"] * 100
    merged.attrs["past_date"] = past_date
    return merged


def compute_rotation_radar(
    all_df: pd.DataFrame,
    as_of: str,
    n_days: int,
    pct_threshold: float = 10.0,
) -> "tuple[pd.DataFrame | None, str | None]":
    """Per-industry breadth of SCTR movers over a trailing window.

    Compares equal-weight avg ΔSCTR (breadth — every stock counts once) against
    cap-weight avg ΔSCTR (dominated by the biggest names) per industry. A gap
    between the two — high breadth while the cap-weighted number lags — flags
    rotation building in the smaller names before it shows up in the aggregate.

    Returns (industry_df, ref_date, n_snapshots), or (None, None, None)/(None, ref_date, None)
    if there's not enough history or no overlapping stocks. `n_snapshots` is the number of
    actual trading-day snapshots spanned between ref_date and as_of — the lookback window is
    calendar days, so this can be well below `n_days` across weekends/holidays/data gaps.
    """
    all_avail_asc = sorted(all_df["snapshot_date"].unique())
    avail_desc = [d for d in reversed(all_avail_asc) if d < as_of]
    ref_date = _snap_for_delta(as_of, avail_desc, n_days)
    if ref_date is None:
        return None, None, None

    n_snapshots = sum(1 for d in all_avail_asc if ref_date <= d <= as_of) - 1

    now = all_df[all_df["snapshot_date"] == as_of][
        ["symbol", "sector", "industry", "sctr", "market_cap"]
    ]
    past = all_df[all_df["snapshot_date"] == ref_date][["symbol", "sctr"]].rename(
        columns={"sctr": "sctr_past"}
    )
    merged = now.merge(past, on="symbol", how="inner").dropna(subset=["sctr", "sctr_past"])
    if merged.empty:
        return None, ref_date, n_snapshots

    merged["delta"] = merged["sctr"] - merged["sctr_past"]

    hi_cut = merged["delta"].quantile(1 - pct_threshold / 100)
    lo_cut = merged["delta"].quantile(pct_threshold / 100)
    merged["is_top"] = merged["delta"] >= hi_cut
    merged["is_bottom"] = merged["delta"] <= lo_cut

    def _capw_mean(g: pd.DataFrame) -> "tuple[float, bool]":
        """Cap-weighted mean delta. Falls back to equal-weight (flagged) when the
        group has no usable market-cap data, instead of returning NaN and silently
        dropping the industry out of divergence-based sorts."""
        w = g["market_cap"].fillna(0)
        if w.sum() <= 0:
            return float(g["delta"].mean()), True
        return float((g["delta"] * w).sum() / w.sum()), False

    rows = []
    for (industry, sector), g in merged.groupby(["industry", "sector"]):
        capw_val, capw_fallback = _capw_mean(g)
        rows.append({
            "industry": industry,
            "sector": sector,
            "n_stocks": len(g),
            "n_top": int(g["is_top"].sum()),
            "pct_top": round(g["is_top"].mean() * 100, 1),
            "n_bottom": int(g["is_bottom"].sum()),
            "pct_bottom": round(g["is_bottom"].mean() * 100, 1),
            "avg_delta_eq": round(g["delta"].mean(), 2),
            "avg_delta_capw": round(capw_val, 2),
            "capw_fallback": capw_fallback,
        })
    out = pd.DataFrame(rows)
    out["divergence"] = round(out["avg_delta_eq"] - out["avg_delta_capw"], 2)
    return out, ref_date, n_snapshots


# ── industry leaders ─────────────────────────────────────────────────────────

# sctr_rankings.industry uses shorter/differently-punctuated names than
# industry_summary.name for the same DJ industry in ~13 cases. Map the
# sctr_rankings spelling to the industry_summary spelling so the "official"
# industry SCTR still resolves for these.
_INDUSTRY_ALIAS = {
    "Entertainment":              "Broadcasting & Entertainment",
    "Building Materials":         "Building Materials & Fixtures",
    "Business Training Agencies": "Business Training & Employment Agencies",
    "Commercial Vehicles":        "Commercial Vehicles & Trucks",
    "Electrical Components":      "Electrical Components & Equipment",
    "Fixed Telecommunications":   "Fixed Line Telecommunications",
    "Food Retailers":             "Food Retailers & Wholesalers",
    "General Mining":             "Mining",
    "Nondurable Home Products":   "Nondurable Household Products",
    "Property-Casualty Insurance": "Property & Casualty Insurance",
    "Real Estate Development":    "Real Estate Holding & Development",
    "Special Consumer Services":  "Specialized Consumer Services",
    "Telecom Equipment":          "Telecommunications Equipment",
}


def _search_filter(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Case-insensitive substring filter across every column's string form."""
    q = query.strip().lower()
    if not q:
        return df
    mask = pd.Series(False, index=df.index)
    for col in df.columns:
        mask |= df[col].astype(str).str.lower().str.contains(q, regex=False, na=False)
    return df[mask]


def compute_industry_leaders(
    stock_df: pd.DataFrame,
    industry_sctr: pd.Series,
    top_n: int = 3,
) -> pd.DataFrame:
    """One row per industry: official SCTR, and top-N stocks by SCTR per cap bucket.

    stock_df       — per-stock rows for one snapshot date: symbol, industry, sctr, group_name.
    industry_sctr  — Series indexed by industry_summary.name → its official SCTR.
    Sorted by "gap" (best individual stock's SCTR minus the industry's official
    SCTR) descending — industries where a stock is running well ahead of a
    lagging aggregate float to the top.
    """
    def _bucket_top(g: pd.DataFrame) -> str:
        top = g.nlargest(top_n, "sctr")
        return ", ".join(f"{r.symbol} {r.sctr:.0f}" for r in top.itertuples())

    rows = []
    for industry, i_df in stock_df.dropna(subset=["industry"]).groupby("industry"):
        official_name = _INDUSTRY_ALIAS.get(industry, industry)
        official = industry_sctr.get(official_name, float("nan"))
        best_stock = i_df["sctr"].max()
        buckets = {
            grp: _bucket_top(i_df[i_df["group_name"] == grp])
            for grp in ("large", "mid", "small")
        }
        rows.append({
            "industry": industry,
            "official_sctr": official,
            "gap": (best_stock - official) if pd.notna(official) else float("nan"),
            "large": buckets["large"],
            "mid": buckets["mid"],
            "small": buckets["small"],
        })
    out = pd.DataFrame(rows)
    return out.sort_values("gap", ascending=False, na_position="last")


# ── heatmap builder ───────────────────────────────────────────────────────────

def build_heatmap_chart(df: pd.DataFrame, color_col: str, size_col: str) -> go.Figure:
    """Squarified treemap: Sector → Industry → Symbol."""
    df = df.dropna(subset=[size_col]).copy()
    df = df[df[size_col] > 0]
    if df.empty:
        return go.Figure()

    ids: list         = []
    labels: list      = []
    parents: list     = []
    values: list      = []
    colors: list      = []
    texts: list       = []
    hover_texts: list = []

    _nan = float("nan")

    def _fmt(v) -> str:
        if pd.isna(v):
            return ""
        if color_col in ("pct_1d", "pct_1w", "pct_1m", "pct_3m"):
            return f"{v:+.2f}%"
        if color_col == "sctr_delta":
            return f"{v:+.1f}"
        return f"{v:.1f}"

    def _hnum(v, fmt: str) -> str:
        return format(v, fmt) if pd.notna(v) else "—"

    def _top_line(g: pd.DataFrame, n: int = 3) -> str:
        """'Top: SYM 98, SYM 96, SYM 94' for the group's highest-SCTR stocks."""
        if "sctr" not in g or g["sctr"].dropna().empty:
            return ""
        top = g.nlargest(n, "sctr")
        return "Top: " + ", ".join(
            f"{r.symbol} {r.sctr:.0f}" for r in top.itertuples()
        )

    def _group_hover(label: str, g: pd.DataFrame) -> str:
        """Aggregate hover text for a sector/industry (non-leaf) node."""
        n = len(g)
        sctr_avg  = g["sctr"].mean() if "sctr" in g else _nan
        delta_avg = g["sctr_delta"].mean() if "sctr_delta" in g else _nan
        mcap_sum  = g["market_cap"].sum() if "market_cap" in g and g["market_cap"].notna().any() else _nan
        return (
            f"<b>{label}</b>  ·  {n} stock{'s' if n != 1 else ''}<br>"
            f"Avg SCTR: {_hnum(sctr_avg, '.1f')}   Avg ΔSCTR: {_hnum(delta_avg, '+.1f')}<br>"
            f"Total Mkt Cap: ${_hnum(mcap_sum, ',.0f')}"
        )

    # root node
    ids.append("__root__"); labels.append("Leaders"); parents.append("")
    values.append(df[size_col].sum()); colors.append(_nan)
    texts.append(""); hover_texts.append(_group_hover("All leaders", df))

    for sector, s_df in df.groupby("sector"):
        s_id = f"__s__{sector}"
        s_col = s_df[color_col].mean() if color_col in s_df else _nan
        s_label = str(sector).replace(" Sector", "")
        s_top = _top_line(s_df)
        ids.append(s_id); labels.append(s_label); parents.append("__root__")
        values.append(s_df[size_col].sum()); colors.append(s_col)
        texts.append(f"{s_label}<br>{s_top}" if s_top else s_label)
        hover_texts.append(_group_hover(s_label, s_df))

        for industry, i_df in s_df.groupby("industry"):
            i_id = f"__i__{sector}__{industry}"
            i_col = i_df[color_col].mean() if color_col in i_df else _nan
            i_label = str(industry)
            i_top = _top_line(i_df)
            ids.append(i_id); labels.append(i_label); parents.append(s_id)
            values.append(i_df[size_col].sum()); colors.append(i_col)
            texts.append(f"{i_label}<br>{i_top}" if i_top else i_label)
            hover_texts.append(_group_hover(i_label, i_df))

            for _, row in i_df.iterrows():
                sym = str(row["symbol"])
                cv  = row.get(color_col, _nan)
                ids.append(f"__stk__{sym}"); labels.append(sym); parents.append(i_id)
                values.append(row[size_col]); colors.append(cv)
                texts.append(f"{sym}<br>{_fmt(cv)}")
                mcap = row.get("market_cap", _nan)
                mcap = mcap if pd.notna(mcap) else row.get("volume", _nan)
                hover_texts.append(
                    f"<b>{sym}</b>  {row.get('name', '')}<br>"
                    f"SCTR: {_hnum(row.get('sctr', _nan), '.1f')}   "
                    f"ΔSCTR: {_hnum(row.get('sctr_delta', _nan), '+.1f')}<br>"
                    f"Close: ${_hnum(row.get('close', _nan), '.2f')}<br>"
                    f"Mkt Cap: ${_hnum(mcap, ',.0f')}"
                )

    cmid   = 92.0 if color_col == "sctr" else 0.0
    clabel = {
        "pct_1d": "1D %", "pct_1w": "1W %", "pct_1m": "1M %", "pct_3m": "3M %",
        "sctr_delta": "ΔSCTR", "sctr": "SCTR",
    }.get(color_col, color_col)

    fig = go.Figure(go.Treemap(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        text=texts,
        texttemplate="%{text}",
        textfont=dict(size=14, color="#000000"),
        insidetextfont=dict(size=14, color="#000000"),
        marker=dict(
            colors=colors,
            colorscale="RdYlGn",
            cmid=cmid,
            showscale=True,
            colorbar=dict(
                title=dict(text=clabel, font=dict(color=_FG, size=11)),
                tickfont=dict(color=_DIM, size=9),
                bgcolor=_BG_CARD,
                bordercolor=_BORDER,
                len=0.6,
            ),
            line=dict(color=_BG, width=1.5),
        ),
        hovertext=hover_texts,
        hoverinfo="text",
        branchvalues="total",
        maxdepth=3,
        tiling=dict(packing="squarify", squarifyratio=1),
    ))

    fig.update_layout(
        paper_bgcolor=_BG,
        font=dict(color=_FG),
        margin=dict(l=5, r=5, t=30, b=5),
        height=720,
        uniformtext=dict(minsize=9, mode="hide"),
    )
    return fig


# ── sector data helpers ────────────────────────────────────────────────────────

# ETF symbol → full sector name matching _SECTOR_COLORS keys
_SECTOR_ETF_MAP = {
    "XLB":  "Materials Sector",
    "XLC":  "Communication Services Sector",
    "XLE":  "Energy Sector",
    "XLF":  "Financial Sector",
    "XLI":  "Industrial Sector",
    "XLK":  "Technology Sector",
    "XLP":  "Consumer Staples Sector",
    "XLRE": "Real Estate Sector",
    "XLU":  "Utilities Sector",
    "XLV":  "Health Care Sector",
    "XLY":  "Consumer Discretionary Sector",
}


@st.cache_data(ttl=300)
def available_dates_sector() -> list[str]:
    with sqlite3.connect(DB_PATH) as c:
        try:
            return [r[0] for r in c.execute(
                "SELECT DISTINCT snapshot_date FROM sector_summary ORDER BY snapshot_date DESC"
            ).fetchall()]
        except sqlite3.OperationalError:
            return []


@st.cache_data(ttl=300)
def load_all_sector() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as c:
        try:
            df = pd.read_sql_query(
                "SELECT * FROM sector_summary ORDER BY snapshot_date DESC, sctr DESC",
                c,
            )
        except sqlite3.OperationalError:
            return pd.DataFrame()
    # add sector column so shared rank/chart functions work
    df["sector"] = df["symbol"].map(_SECTOR_ETF_MAP).fillna(df["name"])
    return df



# ── sidebar ───────────────────────────────────────────────────────────────────

dates_ind = available_dates_industry()

with st.sidebar:
    st.markdown("## StockCharts")
    st.caption("EOD data · read-only")
    st.divider()

    selected_date = st.selectbox("Snapshot date", dates_ind)

    all_sectors = sorted(
        load_industry(dates_ind[0])["sector"].unique().tolist()
    )
    sectors_sel = st.multiselect("Sectors", all_sectors, default=all_sectors)

    st.divider()
    st.caption(f"Available: {len(dates_ind)} snapshot(s)")
    for d in dates_ind:
        st.caption(f"  • {d}")


# ── industry drilldown dialog ─────────────────────────────────────────────────

@st.dialog("Industry drill-down", width="large")
def show_industry_drilldown(
    industry_name: str,
    snapshot_date: str,
    perf: dict,
    all_dates: tuple[str, ...],
) -> None:
    st.markdown("""
        <style>
        div[data-testid="stDialog"] > div > div[data-testid="stModalDialogContent"] {
            max-width: 92vw !important;
            width: 92vw !important;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown(f"### {industry_name}  ·  {_fmt_date(snapshot_date)}")

    # ── industry performance bar ──────────────────────────────────────────────
    _perf_labels = [
        ("SCTR",  "SCTR",  "{:.1f}"),
        ("1D%",   "1D %",  "{:+.2f}%"),
        ("1W%",   "1W %",  "{:+.2f}%"),
        ("1M%",   "1M %",  "{:+.2f}%"),
        ("3M%",   "3M %",  "{:+.2f}%"),
        ("YTD%",  "YTD %", "{:+.2f}%"),
    ]
    avail = [(col, lbl, fmt) for col, lbl, fmt in _perf_labels if col in perf and pd.notna(perf.get(col))]
    if avail:
        cols = st.columns(len(avail))
        for i, (col, lbl, fmt) in enumerate(avail):
            v = perf[col]
            delta_str = None
            if col != "SCTR":
                delta_str = fmt.format(v)
            cols[i].metric(lbl, fmt.format(v))

    st.divider()

    # ── stocks table ──────────────────────────────────────────────────────────
    stocks = load_industry_stocks(industry_name, snapshot_date, all_dates)
    if stocks.empty:
        st.info("No stock-level data found for this industry on the selected date.")
        return

    def _fmt_delta(v):
        if pd.isna(v):
            return "—"
        return f"↑{v:.1f}" if v > 0 else (f"↓{abs(v):.1f}" if v < 0 else "—")

    stocks["SCTR Δ 1D"] = stocks["sctr_delta"].apply(_fmt_delta)
    stocks["Cap"]    = stocks["group_name"].str.capitalize()

    rename = {
        "symbol": "Symbol", "name": "Name", "sctr": "SCTR",
        "close": "Close", "volume": "Volume", "market_cap": "Mkt Cap",
        "pct_1d": "1D%", "pct_1w": "1W%", "pct_1m": "1M%",
    }
    stocks = stocks.rename(columns=rename)

    base_cols  = ["Symbol", "Name", "Cap", "SCTR", "SCTR Δ 1D"]
    pct_cols   = [c for c in ["1D%", "1W%", "1M%"] if c in stocks.columns]
    price_cols = [c for c in ["Close", "Volume", "Mkt Cap"] if c in stocks.columns]
    display = stocks[base_cols + pct_cols + price_cols]

    _pct_cfg = {"format": "%+.2f%%"}
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        height=480,
        column_config={
            "SCTR":    st.column_config.NumberColumn("SCTR",    format="%.1f"),
            "1D%":     st.column_config.NumberColumn("1D%",     **_pct_cfg),
            "1W%":     st.column_config.NumberColumn("1W%",     **_pct_cfg),
            "1M%":     st.column_config.NumberColumn("1M%",     **_pct_cfg),
            "Close":   st.column_config.NumberColumn("Close",   format="$%.2f"),
            "Volume":  st.column_config.NumberColumn("Volume",  format="%,.0f"),
            "Mkt Cap": st.column_config.NumberColumn("Mkt Cap", format="$%,.0f"),
        },
    )
    st.caption(f"{len(stocks)} stocks · sorted by SCTR descending"
               + (" · 1W%/1M% approximated from snapshot closes" if "1W%" in pct_cols else ""))


@st.dialog("Sector drill-down", width="large")
def show_sector_drilldown(
    sector_name: str,
    as_of: str,
    rank_by: str,
    window_dates: list[str],
    bench_ser: "pd.Series | None",
    compare_date: str | None = None,
) -> None:
    st.markdown("""
        <style>
        div[data-testid="stDialog"] > div > div[data-testid="stModalDialogContent"] {
            max-width: 92vw !important;
            width: 92vw !important;
        }
        </style>
    """, unsafe_allow_html=True)

    mode_label = f"vs {_fmt_date(compare_date)}" if compare_date else f"rolling {len(window_dates)+1} snapshots"
    st.markdown(f"### {sector_name}  ·  {_fmt_date(as_of)}  ·  {mode_label}")

    all_ind = load_all_industry()
    sec_ind = all_ind[all_ind["sector"] == sector_name]

    if sec_ind.empty:
        st.info(f"No industry data found for **{sector_name}** on {_fmt_date(as_of)}.")
        return

    n_industries = sec_ind[sec_ind["snapshot_date"] == as_of]["symbol"].nunique()
    st.caption(f"{n_industries} industries in {sector_name}")
    st.divider()

    _pct_fmt = "%.2f%%"
    _ind_col_cfg = {
        "Rank": st.column_config.NumberColumn("Rank", format="%d"),
        "Name": st.column_config.TextColumn("Industry"),
        "SCTR": st.column_config.NumberColumn("SCTR", format="%.1f"),
        **{c: st.column_config.NumberColumn(c, format=_pct_fmt)
           for c in ["1D%", "1W%", "1M%", "3M%", "6M%", "1Y%", "YTD%"]},
    }

    if compare_date:
        # ── compare-with-date table ───────────────────────────────────────────
        ind_ranks = build_ranks_df(sec_ind, rank_by, as_of, [compare_date],
                                   bench_ser=bench_ser)
        ind_ranks = ind_ranks.drop(columns=["Sector"], errors="ignore")

        _rlbl    = _fmt_date(compare_date)
        _vs_col  = f"Δ Price% {_rlbl}"
        _ind_col_cfg["Δ Rank"] = st.column_config.NumberColumn(
            f"Δ Rank (vs {_rlbl})", format="%+d"
        )
        if _vs_col in ind_ranks.columns:
            _ind_col_cfg[_vs_col] = st.column_config.NumberColumn(_vs_col, format="%+.2f%%")
        for dc in [_fmt_date(d) for d in [as_of, compare_date]]:
            _ind_col_cfg[dc] = st.column_config.NumberColumn(dc, format="%d")

        _bench_note = f" · Price % relative to benchmark" if bench_ser is not None else ""
        st.caption(f"Δ Rank = rank change vs {_rlbl}{_bench_note}")
        st.dataframe(ind_ranks.reset_index(drop=True),
                     use_container_width=True, hide_index=True,
                     height=min(480, 40 + n_industries * 38),
                     column_config=_ind_col_cfg)

    else:
        # ── rolling window table ──────────────────────────────────────────────
        ind_compact = build_compact_ranks_df(sec_ind, rank_by, as_of, window_dates,
                                             bench_ser=bench_ser)
        ind_compact = ind_compact.drop(columns=["Sector"], errors="ignore")

        _delta_cols = [c for c in ind_compact.columns if c.startswith("Δ")]
        _metrics    = [c for c in ind_compact.columns
                       if c not in ("Rank", "Name") and not c.startswith("Δ")]
        _def_cols   = [c for c in ["SCTR", "1D%", "1W%", "1M%", "YTD%"] if c in _metrics]

        show_cols = ["Rank", "Name"] + _metrics + _delta_cols
        for dc in _delta_cols:
            _ind_col_cfg[dc] = st.column_config.TextColumn(dc)

        _bench_note = f" · Price % relative to benchmark" if bench_ser is not None else ""
        st.caption(f"Ranked by {rank_by} · {len(window_dates)+1} snapshots{_bench_note}")
        st.dataframe(
            ind_compact[[c for c in show_cols if c in ind_compact.columns]].reset_index(drop=True),
            use_container_width=True, hide_index=True,
            height=min(480, 40 + n_industries * 38),
            column_config=_ind_col_cfg,
        )


# ── tabs ──────────────────────────────────────────────────────────────────────

tab_sector, tab_ranks, tab_sctr, tab_theme, tab_heatmap, tab_rotation, tab_leaders = st.tabs([
    "🌐 Sector Ranks",
    "🏆 Industry Ranks",
    "⚡ SCTR",
    "📊 Visual Tracker",
    "🔥 Leaders Heatmap",
    "🔄 Rotation Radar",
    "🔍 Industry Leaders",
])


# ── Visual Tracker ────────────────────────────────────────────────────────────

with tab_theme:
    df_today = load_industry(selected_date)
    if sectors_sel:
        df_today = df_today[df_today["sector"].isin(sectors_sel)]

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        period_label = st.selectbox(
            "Period", list(PERIOD_MAP.keys()), index=2, key="tt_period"
        )
    with c2:
        sort_dir = st.radio("Sort", ["↓ Desc", "↑ Asc"], horizontal=True, key="tt_sort")
    with c3:
        show_mode = st.selectbox(
            "Show", ["All", "Top 20", "Bottom 20", "Top/Bottom 20", "Top/Bottom N"],
            key="tt_show_mode",
        )
    with c4:
        custom_days = st.number_input(
            "N days (Custom)", min_value=1, max_value=365, value=5,
            key="tt_ndays", disabled=(period_label != "Custom"),
        )

    show_custom_n = show_mode == "Top/Bottom N"
    if show_custom_n:
        show_n = st.number_input(
            "N (per side)", min_value=1, max_value=200, value=20, key="tt_show_n",
        )
    else:
        show_n = 20

    ascending = (sort_dir == "↑ Asc")

    # resolve which column / data to plot
    if period_label == "Custom":
        all_ind = load_all_industry()
        if sectors_sel:
            all_ind = all_ind[all_ind["sector"].isin(sectors_sel)]
        result = compute_nday_pct(selected_date, int(custom_days), all_ind)
        if result is None:
            st.warning(f"Not enough history for {custom_days}-day custom range. "
                       "Select a smaller N or wait for more daily snapshots.")
            st.stop()
        pct_col   = "pct_custom"
        past_date = result.attrs.get("past_date", "?")
        plot_df   = result.rename(columns={"pct_custom": "pct_custom"})
        st.caption(f"Comparing {selected_date} vs {past_date} · using EOD `last` prices")
    else:
        pct_col = PERIOD_MAP[period_label]
        plot_df = df_today.copy()

    if show_mode == "Top 20":
        plot_df = plot_df.nlargest(20, pct_col)
        st.plotly_chart(
            build_theme_chart(plot_df, pct_col, ascending),
            use_container_width=True,
        )
    elif show_mode == "Bottom 20":
        plot_df = plot_df.nsmallest(20, pct_col)
        st.plotly_chart(
            build_theme_chart(plot_df, pct_col, ascending),
            use_container_width=True,
        )
    elif show_mode in ("Top/Bottom 20", "Top/Bottom N"):
        top = plot_df.nlargest(int(show_n), pct_col)
        bot = plot_df.nsmallest(int(show_n), pct_col)
        combined = pd.concat([top, bot]).drop_duplicates()

        st.plotly_chart(
            build_theme_diverging_chart(combined, pct_col),
            use_container_width=True,
        )
    else:
        gainers = plot_df[plot_df[pct_col] >= 0]
        losers  = plot_df[plot_df[pct_col] < 0]

        col_gain, col_lose = st.columns(2)
        with col_gain:
            st.markdown(f"**Gainers** ({len(gainers)})")
            if gainers.empty:
                st.info("No gainers for this period.")
            else:
                st.plotly_chart(
                    build_theme_chart(gainers, pct_col, ascending),
                    use_container_width=True,
                )
        with col_lose:
            st.markdown(f"**Losers** ({len(losers)})")
            if losers.empty:
                st.info("No losers for this period.")
            else:
                st.plotly_chart(
                    build_theme_chart(losers, pct_col, ascending),
                    use_container_width=True,
                )

    with st.expander("Raw data table"):
        raw_cols = ["symbol", "name", "sector", "last", "sctr",
                    "pct_1d", "pct_1w", "pct_1m", "pct_3m", "pct_6m", "pct_1y", "pct_ytd"]
        raw_cols = [c for c in raw_cols if c in df_today.columns]
        st.dataframe(
            df_today[raw_cols].sort_values(
                pct_col if pct_col in df_today.columns else "sctr",
                ascending=False,
            ).reset_index(drop=True),
            use_container_width=True,
            height=300,
        )


# ── Tab 2 · Industry Ranks ────────────────────────────────────────────────────

with tab_ranks:
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        rank_by_label = st.selectbox("Rank by", list(RANK_BY_MAP.keys()), key="rk_by")
    with c2:
        rk_as_of = st.selectbox("As of", dates_ind, key="rk_as_of")
    with c3:
        _bench_opts = ["None"] + available_benchmarks()
        rk_bench = st.selectbox("vs. Benchmark", _bench_opts, key="rk_bench")

    rank_by = RANK_BY_MAP[rank_by_label]
    all_ind = load_all_industry()
    if sectors_sel:
        all_ind = all_ind[all_ind["sector"].isin(sectors_sel)]

    # pre-load benchmark series if selected
    _rk_bench_ser: "pd.Series | None" = None
    if rk_bench != "None":
        _b = load_benchmark_series(rk_bench)
        if not _b.empty:
            _rk_bench_ser = _b.set_index("snapshot_date")["close"]

    # dates before the selected anchor
    compare_opts = [d for d in dates_ind if d < rk_as_of]

    # ── view + mode-specific controls ───────────────────────────────────────
    vc1, vc2, vc3, vc4 = st.columns([2, 2, 2, 2])
    with vc1:
        rk_view = st.radio(
            "View", ["Rolling window", "Compare with date"],
            horizontal=True, key="rk_view",
        )

    # ── Rolling window ────────────────────────────────────────────────────────
    if rk_view == "Rolling window":
        with vc2:
            rk_window = st.selectbox(
                "Last N snapshots", [5, 10, 20, "All"],
                key="rk_window",
            )
        with vc3:
            rk_chart_mode = st.selectbox(
                "Chart shows", ["Top N", "Bottom N", "Top/Bottom N"],
                key="rk_chart_mode",
            )
        with vc4:
            rk_top_n = st.number_input(
                "N per side", min_value=1, max_value=100, value=20,
                key="rk_top_n",
            )

        window_dates = compare_opts if rk_window == "All" else compare_opts[:int(rk_window) - 1]

        _n = int(rk_top_n)
        if rk_chart_mode == "Top N":
            _chart_top, _chart_bot = _n, 0
        elif rk_chart_mode == "Bottom N":
            _chart_top, _chart_bot = 0, _n
        else:
            _chart_top, _chart_bot = _n, _n

        if not window_dates:
            st.info("Only one snapshot available — need at least two to compare.")
        else:
            with st.expander("Bump chart", expanded=False):
                st.plotly_chart(
                    build_bump_chart(
                        all_ind, rank_by, rk_as_of, window_dates,
                        top_n=_chart_top, bottom_n=_chart_bot,
                    ),
                    use_container_width=True,
                )

            compact_df = build_compact_ranks_df(all_ind, rank_by, rk_as_of, window_dates,
                                                bench_ser=_rk_bench_ser)

            _all_metrics = [c for c in compact_df.columns
                            if c not in ("Rank", "Name", "Sector") and not c.startswith("Δ")]
            _delta_cols  = [c for c in compact_df.columns if c.startswith("Δ")]
            _defaults    = [c for c in ["SCTR", "1D%", "1W%", "1M%", "YTD%"] if c in _all_metrics]

            selected_metrics = st.multiselect(
                "Columns to show",
                options=_all_metrics,
                default=_defaults,
                key="rk_cols",
            )
            if not selected_metrics:
                selected_metrics = _defaults

            show_cols = ["Rank", "Name", "Sector"] + selected_metrics + _delta_cols
            display_df = compact_df[[c for c in show_cols if c in compact_df.columns]]

            rk_search = st.text_input(
                "Search", key="rk_search",
                placeholder="Filter rows — e.g. an industry or sector name",
            )
            display_df = _search_filter(display_df, rk_search)

            _pct_fmt = "%.2f%%"
            col_cfg = {
                "SCTR": st.column_config.NumberColumn("SCTR", format="%.1f"),
                "1D%":  st.column_config.NumberColumn("1D%",  format=_pct_fmt),
                "1W%":  st.column_config.NumberColumn("1W%",  format=_pct_fmt),
                "1M%":  st.column_config.NumberColumn("1M%",  format=_pct_fmt),
                "3M%":  st.column_config.NumberColumn("3M%",  format=_pct_fmt),
                "6M%":  st.column_config.NumberColumn("6M%",  format=_pct_fmt),
                "1Y%":  st.column_config.NumberColumn("1Y%",  format=_pct_fmt),
                "YTD%": st.column_config.NumberColumn("YTD%", format=_pct_fmt),
            }
            for dc in _delta_cols:
                col_cfg[dc] = st.column_config.TextColumn(dc)

            _rk_search_note = f" · {len(display_df)} match \"{rk_search}\"" if rk_search.strip() else ""
            if _rk_bench_ser is not None:
                st.caption(f"Price % columns adjusted relative to **{rk_bench}**. SCTR unchanged. Click a row to drill down.{_rk_search_note}")
            else:
                st.caption(f"Click a row to drill down into its stocks.{_rk_search_note}")
            sel = st.dataframe(
                display_df,
                use_container_width=True,
                height=600,
                column_config=col_cfg,
                selection_mode="single-row",
                on_select="rerun",
                key="rk_table_sel",
            )
            selected_rows = sel.selection.rows
            if selected_rows:
                row_idx = selected_rows[0]
                _row = display_df.iloc[row_idx]
                industry_name = _row["Name"]
                _perf_keys = ["SCTR", "1D%", "1W%", "1M%", "3M%", "6M%", "1Y%", "YTD%"]
                perf = {k: _row[k] for k in _perf_keys if k in display_df.columns}
                show_industry_drilldown(industry_name, rk_as_of, perf, tuple(dates_ind))

    # ── Compare with date ─────────────────────────────────────────────────────
    else:
        import datetime as _dt

        _min_date = _dt.date.fromisoformat(compare_opts[-1]) if compare_opts else None
        _max_date = _dt.date.fromisoformat(rk_as_of) - _dt.timedelta(days=1)

        if _min_date is None:
            st.info("No historical snapshots available to compare with.")
        else:
            with vc2:
                picked = st.date_input(
                    "Compare with",
                    value=_max_date,
                    min_value=_min_date,
                    max_value=_max_date,
                    key="rk_compare_date",
                    help="Pick any date — weekends and holidays are automatically snapped to the nearest earlier trading day.",
                )

            resolved = _nearest_earlier_date(picked, compare_opts)

            if resolved is None:
                st.warning(
                    f"No snapshots available before **{picked.strftime('%b %-d')}**. "
                    f"Earliest available: **{_fmt_date(compare_opts[-1])}**."
                )
            else:
                if resolved != picked.isoformat():
                    st.info(
                        f"No snapshot for **{picked.strftime('%b %-d')}** "
                        f"(weekend / holiday / missing download). "
                        f"Using **{_fmt_date(resolved)}** instead."
                    )

                ranks_df = build_ranks_df(all_ind, rank_by, rk_as_of, [resolved],
                                          bench_ser=_rk_bench_ser)

                _rlbl = _fmt_date(resolved)
                _vs_price_col_r = f"Δ Price% {_rlbl}"
                date_cols = [_fmt_date(d) for d in [rk_as_of, resolved]]
                col_cfg_r = {
                    "SCTR":   st.column_config.NumberColumn("SCTR", format="%.1f"),
                    "1M%":    st.column_config.NumberColumn("1M%",  format="%.2f%%"),
                    "3M%":    st.column_config.NumberColumn("3M%",  format="%.2f%%"),
                    "YTD%":   st.column_config.NumberColumn("YTD%", format="%.2f%%"),
                    "Δ Rank": st.column_config.NumberColumn(
                        f"Δ Rank (vs {_rlbl})", format="%+d"
                    ),
                    _vs_price_col_r: st.column_config.NumberColumn(
                        _vs_price_col_r, format="%+.2f%%"
                    ),
                }
                for dc in date_cols:
                    col_cfg_r[dc] = st.column_config.NumberColumn(dc, format="%d")

                rk_cwd_search = st.text_input(
                    "Search", key="rk_cwd_search",
                    placeholder="Filter rows — e.g. an industry or sector name",
                )
                n_before = len(ranks_df)
                ranks_df = _search_filter(ranks_df, rk_cwd_search)
                _cwd_search_note = f" · {len(ranks_df)}/{n_before} match \"{rk_cwd_search}\"" if rk_cwd_search.strip() else ""

                _bench_note = f" · Price % relative to **{rk_bench}**" if _rk_bench_ser is not None else ""
                st.caption(f"Δ Rank = rank change vs {_rlbl} · Δ Price% = price change vs {_rlbl}{_bench_note}{_cwd_search_note}")
                st.dataframe(
                    ranks_df,
                    use_container_width=True,
                    height=600,
                    column_config=col_cfg_r,
                )


# ── Tab 3 · SCTR ─────────────────────────────────────────────────────────────

with tab_sctr:
    groups = available_sctr_groups()
    if not groups:
        st.info("No SCTR data in database yet.")
    else:
        # ── controls ──────────────────────────────────────────────────────────
        _COMBO_LABEL = "large+mid+small (All US stocks)"
        _group_display = groups + (
            [_COMBO_LABEL] if {"large", "mid", "small"} <= set(groups) else []
        )

        c1, c2, c3, c4 = st.columns([2, 2, 1, 2])
        with c1:
            group_choice = st.selectbox("Group", _group_display, key="sc_group")
            group_sel = ("large", "mid", "small") if group_choice == _COMBO_LABEL else group_choice
        with c2:
            sctr_dates = available_dates_sctr(group_sel)
            sctr_date  = st.selectbox("As of", sctr_dates, key="sc_date")
        with c3:
            top_n_sctr = st.selectbox("Show", ["Top 50", "Top 100", "All"], key="sc_n")
        with c4:
            _sc_bench_opts = ["None"] + available_benchmarks()
            sc_bench = st.selectbox("vs. Benchmark", _sc_bench_opts, key="sc_bench")

        # ── view + mode-specific controls ───────────────────────────────────────
        svc1, svc2 = st.columns([2, 2])
        with svc1:
            sc_view = st.radio(
                "View", ["Snapshot", "Rolling window", "Compare with date"],
                horizontal=True, key="sc_view",
            )

        import datetime as _dt
        sc_past_opts = [d for d in sctr_dates if d < sctr_date]

        # pre-load benchmark series if selected
        _sc_bench_ser: "pd.Series | None" = None
        if sc_bench != "None":
            _b = load_benchmark_series(sc_bench)
            if not _b.empty:
                _sc_bench_ser = _b.set_index("snapshot_date")["close"]

        _pf = "%+.2f%%"
        _SC_COL_CFG = {
            "Rank":        st.column_config.NumberColumn("Rank",      format="%d"),
            "symbol":      st.column_config.TextColumn("Symbol"),
            "name":        st.column_config.TextColumn("Name"),
            "sector":      st.column_config.TextColumn("Sector"),
            "industry":    st.column_config.TextColumn("Industry"),
            "group_name":  st.column_config.TextColumn("Cap"),
            "sctr":        st.column_config.NumberColumn("SCTR",       format="%.1f"),
            "sctr_delta":  st.column_config.NumberColumn("ΔSCTR 1D",   format="%+.1f"),
            "Δ SCTR 1W":   st.column_config.NumberColumn("Δ SCTR 1W",  format="%+.1f"),
            "Δ SCTR 1M":   st.column_config.NumberColumn("Δ SCTR 1M",  format="%+.1f"),
            "1D%":         st.column_config.NumberColumn("1D%",   format=_pf),
            "1W%":         st.column_config.NumberColumn("1W%",   format=_pf),
            "1M%":         st.column_config.NumberColumn("1M%",   format=_pf),
            "close":       st.column_config.NumberColumn("Close",     format="$%.2f"),
            "volume":      st.column_config.NumberColumn("Volume",    format="%,.0f"),
            "market_cap":  st.column_config.NumberColumn("Mkt Cap",   format="$%,.0f"),
            "Δ Rank":      st.column_config.NumberColumn("Δ Rank",    format="%+d"),
        }
        _fixed_cols = ["Rank", "symbol", "name", "sector", "industry"] + (
            ["group_name"] if isinstance(group_sel, tuple) else []
        )

        def _sctr_watchlist_button(df: pd.DataFrame, key: str) -> None:
            """Download the table's symbols as a comma-separated list — importable
            directly into a TradingView watchlist for chart-by-chart follow-up."""
            symbols = df["symbol"].dropna().tolist()
            st.download_button(
                f"⬇ Download {len(symbols)} symbols as TradingView watchlist",
                data=",".join(symbols),
                file_name=f"sctr_watchlist_{sctr_date}.txt",
                mime="text/plain",
                key=key,
                disabled=not symbols,
            )

        # ── Rolling window ────────────────────────────────────────────────────
        if sc_view == "Rolling window":
            with svc2:
                sc_window = st.selectbox(
                    "Last N snapshots", [5, 10, 20, "All"], key="sc_window",
                )
            window_dates = sc_past_opts if sc_window == "All" else sc_past_opts[:int(sc_window) - 1]

            if not window_dates:
                st.info("Only one snapshot available — need at least two to compare.")
            else:
                all_sctr = load_all_sctr_for_group(group_sel)
                compact  = build_compact_sctr_df(all_sctr, sctr_date, window_dates,
                                                 bench_ser=_sc_bench_ser)

                if top_n_sctr == "Top 50":
                    compact = compact.head(50)
                elif top_n_sctr == "Top 100":
                    compact = compact.head(100)

                sc_rw_min_sctr = st.slider(
                    "Min SCTR — concentrate on top performers", min_value=0, max_value=100,
                    value=0, key="sc_rw_min_sctr",
                )
                if sc_rw_min_sctr > 0:
                    compact = compact[compact["sctr"] >= sc_rw_min_sctr]

                _fixed   = [c for c in _fixed_cols if c in compact.columns]
                _opt_all = [c for c in compact.columns if c not in _fixed]
                _opt_def = [c for c in ["sctr", "sctr_delta", "Δ SCTR 1W", "1D%", "1W%",
                                        "Δ Rank 1d", "Δ Rank 1w"] if c in _opt_all]
                # always show the "Δ Rank vs [oldest]" column
                _vs_col  = [c for c in compact.columns if c.startswith("Δ Rank vs")]

                sc_sel = st.multiselect(
                    "Columns to show", options=_opt_all, default=_opt_def, key="sc_rw_cols",
                )
                if not sc_sel:
                    sc_sel = _opt_def

                show = compact[_fixed + [c for c in sc_sel if c in compact.columns] +
                               [c for c in _vs_col if c not in sc_sel]]

                # build col_cfg for text delta columns
                rw_cfg = {**_SC_COL_CFG}
                for c in show.columns:
                    if c.startswith("Δ Rank"):
                        rw_cfg[c] = st.column_config.TextColumn(c)

                sc_rw_search = st.text_input(
                    "Search", key="sc_rw_search",
                    placeholder="Filter rows — e.g. a symbol, name, sector, or industry",
                )
                n_before = len(show)
                show = _search_filter(show, sc_rw_search)
                _sc_rw_note = f" · {len(show)}/{n_before} match \"{sc_rw_search}\"" if sc_rw_search.strip() else ""

                if _sc_bench_ser is not None:
                    st.caption(f"Price % columns adjusted relative to **{sc_bench}**. SCTR unchanged.{_sc_rw_note}")
                elif _sc_rw_note:
                    st.caption(_sc_rw_note.lstrip(" ·"))
                st.dataframe(show.reset_index(drop=True),
                             use_container_width=True, height=620, column_config=rw_cfg)
                _sctr_watchlist_button(show, key="sc_rw_wl_dl")

        # ── Compare with date ─────────────────────────────────────────────────
        elif sc_view == "Compare with date":
            if not sc_past_opts:
                st.info("No earlier snapshots available for this group.")
            else:
                _sc_min = _dt.date.fromisoformat(sc_past_opts[-1])
                _sc_max = _dt.date.fromisoformat(sctr_date) - _dt.timedelta(days=1)
                with svc2:
                    sc_picked = st.date_input(
                        "Compare with date", value=_sc_max,
                        min_value=_sc_min, max_value=_sc_max, key="sc_compare_date",
                        help="Weekends and holidays snap to the nearest earlier trading day.",
                    )
                sc_compare_date = _nearest_earlier_date(sc_picked, sc_past_opts)
                if sc_compare_date is None:
                    st.warning(f"No snapshots before {sc_picked}. Earliest: {_fmt_date(sc_past_opts[-1])}")
                    st.stop()
                if sc_compare_date != sc_picked.isoformat():
                    st.info(f"No snapshot for **{sc_picked.strftime('%b %-d')}** — "
                            f"using **{_fmt_date(sc_compare_date)}** instead.")

                sctr_df = load_sctr_enriched(
                    sctr_date, group_sel, tuple(sctr_dates),
                    compare_date=sc_compare_date,
                    bench_symbol=sc_bench if sc_bench != "None" else None,
                )
                if top_n_sctr == "Top 50":
                    sctr_df = sctr_df.head(50)
                elif top_n_sctr == "Top 100":
                    sctr_df = sctr_df.head(100)

                sc_cwd_min_sctr = st.slider(
                    "Min SCTR — concentrate on top performers", min_value=0, max_value=100,
                    value=0, key="sc_cwd_min_sctr",
                )
                if sc_cwd_min_sctr > 0:
                    sctr_df = sctr_df[sctr_df["sctr"] >= sc_cwd_min_sctr]

                # dynamic column names that carry the compare date label
                _lbl = _fmt_date(sc_compare_date)
                _vs_price_col = f"Δ Price% {_lbl}"
                _vs_sctr_col  = f"Δ SCTR {_lbl}"

                _fixed_period = ["1D%", "1W%", "1M%", "3M%", "YTD%"]
                _opt = [c for c in
                        ["Δ Rank", _vs_price_col, _vs_sctr_col,
                         "sctr", "sctr_delta", "Δ SCTR 1W", "Δ SCTR 1M"]
                        + _fixed_period
                        + ["close", "volume", "market_cap"]
                        if c in sctr_df.columns]
                _def = [c for c in
                        ["Δ Rank", _vs_price_col, _vs_sctr_col, "sctr", "1D%", "1W%", "close"]
                        if c in sctr_df.columns]
                sc_sel = st.multiselect("Columns to show", _opt, default=_def, key="sc_cwd_cols")
                if not sc_sel:
                    sc_sel = _def

                # build col config — add dynamic columns
                cwd_cfg = {**_SC_COL_CFG,
                           _vs_price_col: st.column_config.NumberColumn(_vs_price_col, format="%+.2f%%"),
                           _vs_sctr_col:  st.column_config.NumberColumn(_vs_sctr_col,  format="%+.1f"),
                }
                _fixed = [c for c in _fixed_cols if c in sctr_df.columns]
                cwd_show = sctr_df[_fixed + [c for c in sc_sel if c in sctr_df.columns]]

                sc_cwd_search = st.text_input(
                    "Search", key="sc_cwd_search",
                    placeholder="Filter rows — e.g. a symbol, name, sector, or industry",
                )
                n_before = len(cwd_show)
                cwd_show = _search_filter(cwd_show, sc_cwd_search)
                _sc_cwd_note = f" · {len(cwd_show)}/{n_before} match \"{sc_cwd_search}\"" if sc_cwd_search.strip() else ""

                _sc_bench_note = f" · Price % relative to **{sc_bench}**" if _sc_bench_ser is not None else ""
                st.caption(f"Δ Rank = rank change vs {_lbl} · Δ Price% = price change vs {_lbl}{_sc_bench_note}{_sc_cwd_note}")
                st.dataframe(
                    cwd_show.reset_index(drop=True),
                    use_container_width=True, height=620, column_config=cwd_cfg,
                )
                _sctr_watchlist_button(cwd_show, key="sc_cwd_wl_dl")

        # ── Snapshot ──────────────────────────────────────────────────────────
        else:
            sctr_df = load_sctr_enriched(
                sctr_date, group_sel, tuple(sctr_dates),
                bench_symbol=sc_bench if sc_bench != "None" else None,
            )
            if top_n_sctr == "Top 50":
                sctr_df = sctr_df.head(50)
            elif top_n_sctr == "Top 100":
                sctr_df = sctr_df.head(100)

            sc_snap_min_sctr = st.slider(
                "Min SCTR — concentrate on top performers", min_value=0, max_value=100,
                value=0, key="sc_snap_min_sctr",
            )
            if sc_snap_min_sctr > 0:
                sctr_df = sctr_df[sctr_df["sctr"] >= sc_snap_min_sctr]

            _opt = [c for c in ["sctr", "sctr_delta", "Δ SCTR 1W", "Δ SCTR 1M",
                                 "1D%", "1W%", "1M%", "close", "volume", "market_cap"]
                    if c in sctr_df.columns]
            _def = [c for c in ["sctr", "sctr_delta", "1D%", "1W%", "close"] if c in _opt]
            sc_sel = st.multiselect("Columns to show", _opt, default=_def, key="sc_snap_cols")
            if not sc_sel:
                sc_sel = _def

            _fixed = [c for c in _fixed_cols if c in sctr_df.columns]
            snap_show = sctr_df[_fixed + [c for c in sc_sel if c in sctr_df.columns]]

            sc_snap_search = st.text_input(
                "Search", key="sc_snap_search",
                placeholder="Filter rows — e.g. a symbol, name, sector, or industry",
            )
            n_before = len(snap_show)
            snap_show = _search_filter(snap_show, sc_snap_search)
            _sc_snap_note = f" · {len(snap_show)}/{n_before} match \"{sc_snap_search}\"" if sc_snap_search.strip() else ""

            if _sc_bench_ser is not None:
                st.caption(f"Price % columns adjusted relative to **{sc_bench}**. SCTR unchanged.{_sc_snap_note}")
            elif _sc_snap_note:
                st.caption(_sc_snap_note.lstrip(" ·"))
            st.dataframe(
                snap_show.reset_index(drop=True),
                use_container_width=True, height=620, column_config=_SC_COL_CFG,
            )
            _sctr_watchlist_button(snap_show, key="sc_snap_wl_dl")


# ── Tab 4 · Leaders Heatmap ───────────────────────────────────────────────────

with tab_heatmap:
    _GROUP_MAP = {
        "Large Cap":           ("large",),
        "Mid Cap":             ("mid",),
        "Small Cap":           ("small",),
        "ETF":                 ("etf",),
        "Large + Mid":         ("large", "mid"),
        "Large + Mid + Small": ("large", "mid", "small"),
    }
    _COLOR_MAP = {
        "1D % Change":      "pct_1d",
        "1W % Change":      "pct_1w",
        "1M % Change":      "pct_1m",
        "3M % Change":      "pct_3m",
        "ΔSCTR (momentum)": "sctr_delta",
        "SCTR (absolute)":  "sctr",
    }

    c1, c2, c3, c4, c5 = st.columns([2, 2, 1, 2, 2])
    with c1:
        hm_group_label = st.selectbox(
            "Group", list(_GROUP_MAP.keys()), index=0, key="hm_group"
        )
    with c2:
        hm_color_label = st.selectbox(
            "Color by", list(_COLOR_MAP.keys()), key="hm_color",
        )
    with c3:
        hm_size_label = st.radio(
            "Size by", ["Mkt Cap", "Volume", "Equal weight"],
            horizontal=True, key="hm_size",
            help="Equal weight gives every stock the same tile size, regardless of "
                 "market cap or volume — useful for seeing small industries clearly.",
        )
    with c4:
        hm_threshold = st.slider(
            "Min SCTR", min_value=75, max_value=99, value=90, step=5,
            key="hm_thresh",
        )

    hm_groups = _GROUP_MAP[hm_group_label]
    hm_color  = _COLOR_MAP[hm_color_label]
    hm_size   = {
        "Mkt Cap": "market_cap", "Volume": "volume", "Equal weight": "_equal",
    }[hm_size_label]

    # ── persistence filter ────────────────────────────────────────────────────
    _hm_avail = _sctr_avail_dates(hm_groups[0])
    _hm_avail_on = [d for d in _hm_avail if d <= selected_date]
    _hm_max_persist = max(1, len(_hm_avail_on))

    with c5:
        hm_persist = st.slider(
            "Held ≥ SCTR for N snapshots", min_value=1, max_value=_hm_max_persist,
            value=1, key="hm_persist",
            help="Stock must have been above the Min SCTR threshold for this many consecutive snapshots (including today).",
        )

    # ── reference dates for period pct columns ────────────────────────────────
    import datetime as _hmdt
    _hm_older = [d for d in _hm_avail if d < selected_date]

    def _hm_ref(days: int) -> str | None:
        tgt = (_hmdt.date.fromisoformat(selected_date) - _hmdt.timedelta(days=days)).isoformat()
        return next((d for d in _hm_older if d <= tgt), None)

    hm_ref_1w = _hm_ref(7)
    hm_ref_1m = _hm_ref(30)
    hm_ref_3m = _hm_ref(90)

    hm_df = load_sctr_leaders(
        selected_date, hm_groups, float(hm_threshold),
        persist_n=hm_persist,
        ref_1w=hm_ref_1w, ref_1m=hm_ref_1m, ref_3m=hm_ref_3m,
    )
    hm_df["_equal"] = 1.0
    if sectors_sel:
        # sctr_rankings uses "Technology"; industry_summary uses "Technology Sector"
        normalized = {s.replace(" Sector", "") for s in sectors_sel}
        hm_df = hm_df[hm_df["sector"].isin(normalized)]

    if hm_df.empty:
        _persist_why = f" held ≥ {hm_threshold} SCTR for {hm_persist} snapshots" if hm_persist > 1 else ""
        st.warning(f"No stocks found with SCTR ≥ {hm_threshold}{_persist_why} for the selected group/date.")
    else:
        # ── caption ───────────────────────────────────────────────────────────
        _persist_note = f" · held ≥ {hm_threshold} for {hm_persist} snapshots" if hm_persist > 1 else ""
        _color_ref = {"pct_1w": hm_ref_1w, "pct_1m": hm_ref_1m, "pct_3m": hm_ref_3m}.get(hm_color)
        if _color_ref:
            _color_note = f" · color vs {_fmt_date(_color_ref)}"
        elif hm_color in ("pct_1w", "pct_1m", "pct_3m"):
            _color_note = " · (no history available for this period — color shows no data)"
        else:
            _color_note = ""
        st.caption(
            f"{len(hm_df)} leaders · SCTR ≥ {hm_threshold}{_persist_note} · "
            f"{selected_date}{_color_note} · click a sector to zoom in"
        )

        st.plotly_chart(
            build_heatmap_chart(hm_df, hm_color, hm_size),
            use_container_width=True,
        )

        with st.expander("Leaders table"):
            tbl_cols = [c for c in
                        ["symbol", "name", "sector", "industry",
                         "sctr", "sctr_delta", "close", "market_cap", "volume",
                         "pct_1d", "pct_1w", "pct_1m", "pct_3m"]
                        if c in hm_df.columns]
            st.dataframe(
                hm_df[tbl_cols].reset_index(drop=True),
                use_container_width=True,
                height=320,
                column_config={
                    "sctr":       st.column_config.NumberColumn("SCTR",   format="%.1f"),
                    "sctr_delta": st.column_config.NumberColumn("ΔSCTR",  format="%+.1f"),
                    "close":      st.column_config.NumberColumn("Close",  format="$%.2f"),
                    "market_cap": st.column_config.NumberColumn("Mkt Cap",format="$%,.0f"),
                    "volume":     st.column_config.NumberColumn("Volume", format="%,.0f"),
                    "pct_1d":    st.column_config.NumberColumn("1D %",   format="%+.2f%%"),
                    "pct_1w":    st.column_config.NumberColumn("1W %",   format="%+.2f%%"),
                    "pct_1m":    st.column_config.NumberColumn("1M %",   format="%+.2f%%"),
                    "pct_3m":    st.column_config.NumberColumn("3M %",   format="%+.2f%%"),
                },
            )


# ── Tab 5 · Rotation Radar ────────────────────────────────────────────────────

with tab_rotation:
    st.caption(
        "Which industries have the widest breadth of stocks improving/deteriorating in SCTR "
        "over a trailing window — a lead indicator ahead of the cap-weighted aggregate moving."
    )

    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    with c1:
        rot_days = st.number_input(
            "Lookback window (days)", min_value=2, max_value=365, value=20, key="rot_days",
        )
    with c2:
        rot_pct = st.slider(
            "Top/Bottom %", min_value=5, max_value=25, value=10, step=5, key="rot_pct",
            help="Stocks in the top/bottom X% of ΔSCTR across the whole universe count as movers.",
        )
    with c3:
        rot_min_n = st.number_input(
            "Min stocks per industry", min_value=1, max_value=50, value=5, key="rot_min_n",
        )
    with c4:
        rot_sort_label = st.selectbox(
            "Sort by",
            ["% Improving", "% Deteriorating", "Divergence"],
            key="rot_sort",
        )
    rot_search = st.text_input(
        "Search", key="rot_search",
        placeholder="Filter rows — e.g. an industry or sector name",
    )

    all_rot = load_all_sctr_multi(("large", "mid", "small"))
    if sectors_sel:
        normalized = {s.replace(" Sector", "") for s in sectors_sel}
        all_rot = all_rot[all_rot["sector"].isin(normalized)]

    rot_df, rot_ref, rot_n_snap = compute_rotation_radar(
        all_rot, selected_date, int(rot_days), float(rot_pct)
    )

    if rot_df is None:
        if rot_ref is None:
            st.warning(f"Not enough history for a {int(rot_days)}-day lookback. Try a smaller window.")
        else:
            st.info("No overlapping stocks between the two snapshots.")
    else:
        rot_df = rot_df[rot_df["n_stocks"] >= rot_min_n]
        sort_col = {
            "% Improving": "pct_top",
            "% Deteriorating": "pct_bottom",
            "Divergence": "divergence",
        }[rot_sort_label]
        rot_df = rot_df.sort_values(sort_col, ascending=False)

        n_fallback = int(rot_df["capw_fallback"].sum())
        display = rot_df.rename(columns={
            "industry": "Industry", "sector": "Sector", "n_stocks": "#Stocks",
            "n_top": "#Improving", "pct_top": "% Improving",
            "n_bottom": "#Deteriorating", "pct_bottom": "% Deteriorating",
            "avg_delta_eq": "Avg ΔSCTR (equal-wt)", "avg_delta_capw": "Avg ΔSCTR (cap-wt)",
            "divergence": "Divergence", "capw_fallback": "No cap data",
        })
        n_before = len(display)
        display = _search_filter(display, rot_search)
        _search_note = f" · {len(display)} match \"{rot_search}\"" if rot_search.strip() else ""

        span_note = (
            f"{rot_n_snap} trading snapshots" if rot_n_snap == int(rot_days)
            else f"{rot_n_snap} trading snapshots, {int(rot_days)}d requested"
        )
        st.caption(
            f"{selected_date} vs {_fmt_date(rot_ref)} ({span_note}) · "
            f"movers = top/bottom {int(rot_pct)}% of ΔSCTR across {len(all_rot['symbol'].unique())} stocks"
            f" · {n_before} industries{_search_note}"
        )

        st.dataframe(
            display.reset_index(drop=True),
            use_container_width=True,
            height=600,
            column_config={
                "% Improving":           st.column_config.NumberColumn("% Improving", format="%.1f%%"),
                "% Deteriorating":       st.column_config.NumberColumn("% Deteriorating", format="%.1f%%"),
                "Avg ΔSCTR (equal-wt)":  st.column_config.NumberColumn("Avg ΔSCTR (equal-wt)", format="%+.2f"),
                "Avg ΔSCTR (cap-wt)":    st.column_config.NumberColumn("Avg ΔSCTR (cap-wt)", format="%+.2f"),
                "Divergence":            st.column_config.NumberColumn("Divergence", format="%+.2f"),
                "No cap data":           st.column_config.CheckboxColumn("No cap data", disabled=True),
            },
        )
        st.caption(
            "Divergence = equal-weight avg ΔSCTR − cap-weight avg ΔSCTR. "
            "High % Improving with positive divergence → smaller names leading, early accumulation. "
            "High % Deteriorating with negative divergence → smaller names cracking first, early warning "
            "— often before the industry's big-cap names show it."
            + (f" · {n_fallback} industries have no market-cap data — cap-weighted column falls back to "
               "equal-weight for them (divergence = 0)." if n_fallback else "")
        )


# ── Tab 1 · Sector Ranks ──────────────────────────────────────────────────────

with tab_sector:
    dates_sec = available_dates_sector()
    if not dates_sec:
        st.info("No sector data in database yet. Run `python main.py` to download.")
    else:
        all_sec = load_all_sector()

        c1, c2, c3 = st.columns([2, 2, 2])
        with c1:
            sr_rank_by_label = st.selectbox("Rank by", list(RANK_BY_MAP.keys()), key="sr_by")
        with c2:
            sr_as_of = st.selectbox("As of", dates_sec, key="sr_as_of")
        with c3:
            _sr_bench_opts = ["None"] + available_benchmarks()
            sr_bench = st.selectbox("vs. Benchmark", _sr_bench_opts, key="sr_bench")

        sr_rank_by = RANK_BY_MAP[sr_rank_by_label]
        sr_compare_opts = [d for d in dates_sec if d < sr_as_of]

        _sr_bench_ser: "pd.Series | None" = None
        if sr_bench != "None":
            _b = load_benchmark_series(sr_bench)
            if not _b.empty:
                _sr_bench_ser = _b.set_index("snapshot_date")["close"]

        # ── view + mode-specific controls ─────────────────────────────────────
        svc1, svc2, svc3, svc4 = st.columns([2, 2, 2, 2])
        with svc1:
            sr_view = st.radio(
                "View", ["Rolling window", "Compare with date"],
                horizontal=True, key="sr_view",
            )

        # ── Rolling window ────────────────────────────────────────────────────
        if sr_view == "Rolling window":
            with svc2:
                sr_window = st.selectbox(
                    "Last N snapshots", [5, 10, 20, "All"], key="sr_window",
                )
            with svc3:
                sr_chart_mode = st.selectbox(
                    "Chart shows", ["Top N", "Bottom N", "Top/Bottom N"], key="sr_chart_mode",
                )
            with svc4:
                sr_top_n = st.number_input(
                    "N per side", min_value=1, max_value=11, value=5, key="sr_top_n",
                )

            sr_window_dates = (sr_compare_opts if sr_window == "All"
                               else sr_compare_opts[:int(sr_window) - 1])

            _sn = int(sr_top_n)
            if sr_chart_mode == "Top N":
                _sc_top, _sc_bot = _sn, 0
            elif sr_chart_mode == "Bottom N":
                _sc_top, _sc_bot = 0, _sn
            else:
                _sc_top, _sc_bot = _sn, _sn

            if not sr_window_dates:
                st.info("Only one snapshot available — need at least two to compare.")
            else:
                with st.expander("Bump chart", expanded=False):
                    st.plotly_chart(
                        build_bump_chart(
                            all_sec, sr_rank_by, sr_as_of, sr_window_dates,
                            top_n=_sc_top, bottom_n=_sc_bot,
                        ),
                        use_container_width=True,
                    )

                sr_compact = build_compact_ranks_df(
                    all_sec, sr_rank_by, sr_as_of, sr_window_dates,
                    bench_ser=_sr_bench_ser,
                )

                # drop "Sector" column — each row IS a sector
                sr_compact = sr_compact.drop(columns=["Sector"], errors="ignore")

                _sr_metrics = [c for c in sr_compact.columns
                               if c not in ("Rank", "Name") and not c.startswith("Δ")]
                _sr_deltas  = [c for c in sr_compact.columns if c.startswith("Δ")]
                _sr_def     = [c for c in ["SCTR", "1D%", "1W%", "1M%", "YTD%"] if c in _sr_metrics]

                sr_sel_metrics = st.multiselect(
                    "Columns to show", options=_sr_metrics, default=_sr_def, key="sr_cols",
                )
                if not sr_sel_metrics:
                    sr_sel_metrics = _sr_def

                sr_show = sr_compact[
                    ["Rank", "Name"] +
                    [c for c in sr_sel_metrics if c in sr_compact.columns] +
                    _sr_deltas
                ]

                _sr_pct_fmt = "%.2f%%"
                sr_col_cfg = {
                    "SCTR": st.column_config.NumberColumn("SCTR", format="%.1f"),
                    **{c: st.column_config.NumberColumn(c, format=_sr_pct_fmt)
                       for c in ["1D%", "1W%", "1M%", "3M%", "6M%", "1Y%", "YTD%"]},
                    **{c: st.column_config.TextColumn(c) for c in _sr_deltas},
                }

                if _sr_bench_ser is not None:
                    st.caption(f"Price % columns adjusted relative to **{sr_bench}**. SCTR unchanged. Click a row to see its industries.")
                else:
                    st.caption("Click a row to see its industries.")
                sr_rw_sel = st.dataframe(
                    sr_show.reset_index(drop=True),
                    use_container_width=True,
                    height=480,
                    column_config=sr_col_cfg,
                    selection_mode="single-row",
                    on_select="rerun",
                    key="sr_rw_sel",
                )
                if sr_rw_sel.selection.rows:
                    _sr_row = sr_show.iloc[sr_rw_sel.selection.rows[0]]
                    _sr_sector_name = _sr_row["Name"].rsplit(" Fund", 1)[0]
                    show_sector_drilldown(
                        _sr_sector_name, sr_as_of, sr_rank_by,
                        sr_window_dates, _sr_bench_ser,
                    )

        # ── Compare with date ─────────────────────────────────────────────────
        else:
            import datetime as _dt

            _sr_min = _dt.date.fromisoformat(sr_compare_opts[-1]) if sr_compare_opts else None
            _sr_max = _dt.date.fromisoformat(sr_as_of) - _dt.timedelta(days=1)

            if _sr_min is None:
                st.info("No historical snapshots available to compare with.")
            else:
                with svc2:
                    sr_picked = st.date_input(
                        "Compare with",
                        value=_sr_max,
                        min_value=_sr_min,
                        max_value=_sr_max,
                        key="sr_compare_date",
                        help="Weekends and holidays snap to the nearest earlier trading day.",
                    )
                sr_resolved = _nearest_earlier_date(sr_picked, sr_compare_opts)

                if sr_resolved is None:
                    st.warning(
                        f"No snapshots before **{sr_picked.strftime('%b %-d')}**. "
                        f"Earliest available: **{_fmt_date(sr_compare_opts[-1])}**."
                    )
                else:
                    if sr_resolved != sr_picked.isoformat():
                        st.info(
                            f"No snapshot for **{sr_picked.strftime('%b %-d')}** — "
                            f"using **{_fmt_date(sr_resolved)}** instead."
                        )

                    sr_ranks = build_ranks_df(
                        all_sec, sr_rank_by, sr_as_of, [sr_resolved],
                        bench_ser=_sr_bench_ser,
                    )
                    # drop Sector column — redundant at sector level
                    sr_ranks = sr_ranks.drop(columns=["Sector"], errors="ignore")

                    _sr_rlbl = _fmt_date(sr_resolved)
                    _sr_vs_price = f"Δ Price% {_sr_rlbl}"
                    sr_date_cols = [_fmt_date(d) for d in [sr_as_of, sr_resolved]]
                    sr_cmp_cfg = {
                        "SCTR":   st.column_config.NumberColumn("SCTR", format="%.1f"),
                        "1M%":    st.column_config.NumberColumn("1M%",  format="%.2f%%"),
                        "3M%":    st.column_config.NumberColumn("3M%",  format="%.2f%%"),
                        "YTD%":   st.column_config.NumberColumn("YTD%", format="%.2f%%"),
                        "Δ Rank": st.column_config.NumberColumn(
                            f"Δ Rank (vs {_sr_rlbl})", format="%+d"
                        ),
                        _sr_vs_price: st.column_config.NumberColumn(
                            _sr_vs_price, format="%+.2f%%"
                        ),
                    }
                    for dc in sr_date_cols:
                        sr_cmp_cfg[dc] = st.column_config.NumberColumn(dc, format="%d")

                    _sr_bench_note = f" · Price % relative to **{sr_bench}**" if _sr_bench_ser is not None else ""
                    st.caption(
                        f"Δ Rank = rank change vs {_sr_rlbl} · "
                        f"Δ Price% = price change vs {_sr_rlbl}{_sr_bench_note} · "
                        f"Click a row to see its industries."
                    )
                    sr_cwd_sel = st.dataframe(
                        sr_ranks.reset_index(drop=True),
                        use_container_width=True,
                        height=480,
                        column_config=sr_cmp_cfg,
                        selection_mode="single-row",
                        on_select="rerun",
                        key="sr_cwd_sel",
                    )
                    if sr_cwd_sel.selection.rows:
                        _sr_row = sr_ranks.iloc[sr_cwd_sel.selection.rows[0]]
                        _sr_sector_name = _sr_row["Name"].rsplit(" Fund", 1)[0]
                        show_sector_drilldown(
                            _sr_sector_name, sr_as_of, sr_rank_by,
                            sr_compare_opts, _sr_bench_ser,
                            compare_date=sr_resolved,
                        )


# ── Tab 7 · Industry Leaders ──────────────────────────────────────────────────

with tab_leaders:
    st.caption(
        "Industries ranked by how far their best individual stock's SCTR is running "
        "ahead of the industry's own aggregate score — surfaces a mid/small-cap leader "
        "hiding inside an industry whose overall number still looks weak."
    )

    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        il_top_n = st.selectbox("Top N per bucket", [3, 5, 10], key="il_top_n")
    with c2:
        il_min_gap = st.number_input(
            "Min gap to show", min_value=0, max_value=100, value=0, key="il_min_gap",
            help="Hide industries where the best stock isn't meaningfully ahead of "
                 "the industry's own aggregate SCTR. 0 = show all.",
        )
    with c3:
        il_search = st.text_input(
            "Search", key="il_search",
            placeholder="Filter rows — e.g. an industry name or a ticker like CAKE",
        )

    il_official = load_industry(selected_date).set_index("name")["sctr"]
    il_stocks = load_all_sctr_multi(("large", "mid", "small"))
    il_stocks = il_stocks[il_stocks["snapshot_date"] == selected_date]
    if sectors_sel:
        normalized = {s.replace(" Sector", "") for s in sectors_sel}
        il_stocks = il_stocks[il_stocks["sector"].isin(normalized)]

    if il_stocks.empty:
        st.info("No stock-level SCTR data for this date/filter.")
    else:
        il_df = compute_industry_leaders(il_stocks, il_official, top_n=int(il_top_n))
        if il_min_gap > 0:
            il_df = il_df[il_df["gap"] >= il_min_gap]

        display = il_df.rename(columns={
            "industry": "Industry", "official_sctr": "Industry SCTR", "gap": "Gap",
            "large": "Large (top N)", "mid": "Mid (top N)", "small": "Small (top N)",
        })
        n_before = len(display)
        display = _search_filter(display, il_search)

        _search_note = f" · {len(display)} match \"{il_search}\"" if il_search.strip() else ""
        st.caption(
            f"{n_before} industries · {selected_date} · "
            f"sorted by widest gap (best stock vs. industry aggregate){_search_note}"
        )

        st.dataframe(
            display.reset_index(drop=True),
            use_container_width=True,
            height=700,
            column_config={
                "Industry SCTR": st.column_config.NumberColumn("Industry SCTR", format="%.1f"),
                "Gap":           st.column_config.NumberColumn("Gap", format="%+.1f"),
            },
        )
        st.caption(
            "Gap = best individual stock's SCTR in that industry − the industry's own "
            "aggregate SCTR (same number shown in Industry Ranks). A large positive gap "
            "means a stock is running well ahead of what the industry-level number suggests."
        )


# ── 52W Highs/Lows ────────────────────────────────────────────────────────────

