"""
GICS Rotation Monitor — Streamlit dashboard.

Run:  streamlit run app.py
"""

import glob
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GICS Rotation Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

RESULTS = Path("results")

# ── Severity label colour map ──────────────────────────────────────────────

_SEV_COLOR = {
    "Both TF Confirmed + ATH":     "#00c853",
    "Both TF Confirmed":           "#00e676",
    "Strong Confirmed + ATH":      "#69f0ae",
    "Strong Confirmed":            "#b9f6ca",
    "Confirmed + ATH":             "#a5d6a7",
    "Confirmed":                   "#c8e6c9",
    "Early Signal":                "#fff9c4",
    "Pullback in Uptrend":         "#fff176",
    "Neutral / Watch":             "#e0e0e0",
    "Weakening":                   "#ffccbc",
    "Confirmed Exit":              "#ef9a9a",
}

_FABER_COLOR = {
    "STRONG BUY": "#00c853",
    "BUY":        "#69f0ae",
    "HOLD":       "#fff176",
    "WATCH":      "#90caf9",
    "CAUTION":    "#ffcc02",
    "EXIT":       "#ef9a9a",
}

_STAGE_COLOR = {
    "2B": "#00c853", "2A": "#69f0ae", "2C": "#fff176",
    "1":  "#e0e0e0", "3":  "#ffccbc", "4":  "#ef9a9a",
}


def _sev_clean(label: str) -> str:
    return re.sub(r'\s*[\(★].*$', '', str(label)).strip()


def _color_severity(val):
    c = _SEV_COLOR.get(_sev_clean(str(val)), "#ffffff")
    return f"background-color: {c}; color: #212121"

def _color_faber(val):
    c = _FABER_COLOR.get(str(val), "#ffffff")
    return f"background-color: {c}; color: #212121"

def _color_stage(val):
    c = _STAGE_COLOR.get(str(val), "#ffffff")
    return f"background-color: {c}; color: #212121"


# ── File helpers ───────────────────────────────────────────────────────────

def _available_dates(pattern: str) -> list[str]:
    files = sorted(glob.glob(str(RESULTS / pattern)))
    dates = []
    for f in files:
        m = re.search(r'(\d{4}-\d{2}-\d{2})', Path(f).stem)
        if m:
            dates.append(m.group(1))
    return sorted(set(dates), reverse=True)


@st.cache_data(ttl=300)
def _load(pattern: str, date_label: str) -> pd.DataFrame:
    p = RESULTS / pattern.replace("*", date_label)
    if not p.exists():
        files = sorted(glob.glob(str(RESULTS / pattern)))
        if not files:
            return pd.DataFrame()
        p = Path(files[-1])
    return pd.read_csv(p)


@st.cache_data(ttl=300)
def _load_breadth(date_label: str) -> dict:
    p = RESULTS / f"breadth_{date_label}.csv"
    if not p.exists():
        files = sorted(glob.glob(str(RESULTS / "breadth_*.csv")))
        if not files:
            return {}
        p = Path(files[-1])
    df = pd.read_csv(p)
    if 'metric' in df.columns and 'value' in df.columns:
        return df.set_index('metric')['value'].to_dict()
    return {}


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 GICS Rotation Monitor")
    dates = _available_dates("rotation_severity_*.csv")
    if not dates:
        st.error("No severity files found — run `python main.py --mode update`")
        st.stop()
    date_label = st.selectbox("As of date", dates)
    st.divider()
    st.caption("Tabs: Overview · Industries · Stage · Stocks · Delta")


# ── Load data ──────────────────────────────────────────────────────────────

sev     = _load("rotation_severity_*.csv",  date_label)
mom     = _load("momentum_screen_*.csv",    date_label)
stage   = _load("stage_analysis_*.csv",     date_label)
stocks  = _load("stock_screener_*.csv",     date_label)
delta   = _load("signal_delta_*.csv",       date_label)
br      = _load_breadth(date_label)
mc      = _load("market_clock_*.csv",       date_label)

# Merge faber + stage into severity
if not mom.empty and not sev.empty:
    sev = sev.merge(
        mom[['industry_key', 'faber_signal', 'stage', 'minervini_count',
             'momentum_score']].rename(columns={
                 'stage': 'stage_mom', 'minervini_count': 'mc_mom'}),
        on='industry_key', how='left'
    )
    for col in ('stage', 'minervini_count'):
        if col + '_mom' in sev.columns and col not in sev.columns:
            sev[col] = sev[col + '_mom']

# ── Tabs ───────────────────────────────────────────────────────────────────

tab_overview, tab_industries, tab_stage, tab_stocks, tab_delta = st.tabs(
    ["🏠 Overview", "📋 Industries", "📐 Stage Map", "🔍 Stocks", "🔔 Signal Delta"]
)

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════

with tab_overview:
    st.header(f"Market Overview — {date_label}")

    # ── Market clock row ──────────────────────────────────────────────────
    if not mc.empty:
        cols = st.columns(len(mc))
        for col, (_, row) in zip(cols, mc.iterrows()):
            state = str(row.get('state', ''))
            dds   = int(row.get('dd_count_25d', 0))
            color = ("#00c853" if "Uptrend" in state and "Stress" not in state
                     else "#ffcc02" if "Rally" in state or "Pressure" in state
                     else "#ef9a9a" if "Stress" in state or "Correction" in state
                     else "#e0e0e0")
            col.metric(
                label=str(row.get('ticker', '')),
                value=state,
                delta=f"{dds} distribution days",
                delta_color="inverse",
            )
    st.divider()

    # ── Breadth health score ──────────────────────────────────────────────
    if br:
        hs   = float(br.get('health_score', 0))
        hlbl = str(br.get('health_label', ''))
        c1, c2 = st.columns([1, 2])
        with c1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=hs,
                title={'text': f"Breadth Health<br><sub>{hlbl}</sub>"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': (
                        "#00c853" if hs >= 70 else
                        "#ffcc02" if hs >= 45 else "#ef9a9a"
                    )},
                    'steps': [
                        {'range': [0,  20], 'color': '#fce4ec'},
                        {'range': [20, 40], 'color': '#ffccbc'},
                        {'range': [40, 60], 'color': '#fff9c4'},
                        {'range': [60, 80], 'color': '#dcedc8'},
                        {'range': [80, 100],'color': '#c8e6c9'},
                    ],
                },
                number={'suffix': '/100'},
            ))
            fig.update_layout(height=260, margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            metrics = [
                ("Stage 2 (2A+2B)",   br.get('stage2_pct', 0),       "%"),
                ("Above 200-SMA",      br.get('above_sma200_pct', 0),  "%"),
                ("RS composite ≥ 50",  br.get('rs_positive_pct', 0),   "%"),
                ("Golden Cross",       br.get('golden_cross_pct', 0),  "%"),
                ("SCTR ≥ 60",         br.get('sctr60_pct', 0),        "%"),
                ("Minervini ≥ 5/7",   br.get('minervini5_pct', 0),    "%"),
                ("STRONG BUY+BUY",    br.get('faber_buy_pct', 0),     "%"),
                ("Within 5% of ATH",  br.get('near_ath_pct', 0),      "%"),
            ]
            r1, r2 = st.columns(2)
            for j, (lbl, val, sfx) in enumerate(metrics):
                (r1 if j % 2 == 0 else r2).metric(lbl, f"{float(val):.1f}{sfx}")
    else:
        st.info("No breadth data — run `--mode market-breadth`")

    st.divider()

    # ── Severity distribution bar ─────────────────────────────────────────
    if not sev.empty:
        st.subheader("Severity Distribution")
        sev_order = [
            "Both TF Confirmed + ATH", "Both TF Confirmed",
            "Strong Confirmed + ATH", "Strong Confirmed",
            "Confirmed + ATH", "Confirmed",
            "Early Signal", "Pullback in Uptrend",
            "Neutral / Watch", "Weakening", "Confirmed Exit",
        ]
        sev['sev_clean'] = sev['severity_label'].apply(_sev_clean)
        counts = sev['sev_clean'].value_counts().reset_index()
        counts.columns = ['label', 'count']
        counts['color'] = counts['label'].map(_SEV_COLOR).fillna("#e0e0e0")
        counts['order'] = counts['label'].apply(
            lambda x: sev_order.index(x) if x in sev_order else 99
        )
        counts = counts.sort_values('order')
        fig = px.bar(counts, x='label', y='count',
                     color='label', color_discrete_map=_SEV_COLOR,
                     labels={'label': '', 'count': 'Industries'})
        fig.update_layout(showlegend=False, height=280,
                          margin=dict(t=10, b=60, l=0, r=0),
                          xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

    # ── Sector health ─────────────────────────────────────────────────────
    sec_path_pattern = f"breadth_sector_{date_label}.csv"
    sec_files = sorted(glob.glob(str(RESULTS / "breadth_sector_*.csv")))
    if sec_files:
        sec_df = pd.read_csv(sec_files[-1])
        if not sec_df.empty and 'health_score' in sec_df.columns:
            st.subheader("Sector Health Scores")
            fig = px.bar(
                sec_df.sort_values('health_score'),
                x='health_score', y='sector_name',
                orientation='h',
                color='health_score',
                color_continuous_scale=['#ef9a9a', '#fff176', '#00c853'],
                range_color=[0, 100],
                labels={'health_score': 'Health Score', 'sector_name': ''},
            )
            fig.update_layout(height=340, margin=dict(t=10, b=20, l=0, r=0),
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — INDUSTRIES
# ══════════════════════════════════════════════════════════════════════════

with tab_industries:
    st.header("Industry Severity Table")

    if sev.empty:
        st.warning("No severity data.")
    else:
        # Filters
        fc1, fc2, fc3 = st.columns(3)
        sectors   = ["All"] + sorted(sev['sector_name'].dropna().unique().tolist())
        fabers    = ["All"] + ["STRONG BUY", "BUY", "HOLD", "WATCH", "CAUTION", "EXIT"]
        sev_lbls  = ["All"] + sorted(sev['severity_label'].dropna().unique().tolist())

        sel_sec   = fc1.selectbox("Sector",         sectors)
        sel_faber = fc2.selectbox("Faber Signal",   fabers)
        sel_sev   = fc3.selectbox("Severity Label", sev_lbls)

        filt = sev.copy()
        if sel_sec   != "All": filt = filt[filt['sector_name']    == sel_sec]
        if sel_faber != "All": filt = filt[filt.get('faber_signal', pd.Series()) == sel_faber]
        if sel_sev   != "All": filt = filt[filt['severity_label'] == sel_sev]

        show_cols = ['rank', 'industry_name', 'sector_name',
                     'severity_label', 'severity_score',
                     'faber_signal', 'stage', 'minervini_count',
                     'rs_pct_composite', 'rs_pct_3m', 'rs_new_high',
                     'momentum_score', 'sctr',
                     'quadrant', 'quadrant_m', 'ath_signal']
        show_cols = [c for c in show_cols if c in filt.columns]

        st.caption(f"{len(filt)} industries")

        styled = (
            filt[show_cols]
            .style
            .map(_color_severity, subset=['severity_label'])
            .map(_color_faber,    subset=['faber_signal']    if 'faber_signal' in show_cols else [])
            .map(_color_stage,    subset=['stage']           if 'stage' in show_cols else [])
            .format({
                'severity_score': '{:.1f}',
                'rs_pct_composite': '{:.0f}',
                'rs_pct_3m': '{:.0f}',
                'momentum_score': '{:.1f}',
                'sctr': '{:.1f}',
            }, na_rep='–')
        )
        st.dataframe(styled, use_container_width=True, height=600)


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — STAGE MAP
# ══════════════════════════════════════════════════════════════════════════

with tab_stage:
    st.header("Stage Distribution")

    if stage.empty:
        st.warning("No stage data — run `--mode stage`")
    else:
        c1, c2 = st.columns(2)

        # Donut
        with c1:
            stage_counts = stage['stage'].value_counts().reset_index()
            stage_counts.columns = ['stage', 'count']
            stage_counts['color'] = stage_counts['stage'].map(_STAGE_COLOR)
            fig = px.pie(stage_counts, values='count', names='stage',
                         color='stage', color_discrete_map=_STAGE_COLOR,
                         hole=0.45, title="All 143 Industries")
            fig.update_traces(textinfo='label+value+percent')
            fig.update_layout(height=360, margin=dict(t=40, b=10, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)

        # By sector
        with c2:
            if 'sector_name' in stage.columns:
                pivot = (
                    stage.groupby(['sector_name', 'stage'])
                    .size().reset_index(name='n')
                )
                fig2 = px.bar(pivot, x='n', y='sector_name', color='stage',
                              color_discrete_map=_STAGE_COLOR,
                              orientation='h',
                              title="Stage by Sector",
                              labels={'n': 'Industries', 'sector_name': ''})
                fig2.update_layout(height=360, margin=dict(t=40, b=10, l=0, r=0),
                                   legend=dict(orientation='h', y=1.08))
                st.plotly_chart(fig2, use_container_width=True)

        # Minervini heatmap
        st.subheader("Minervini Score Distribution")
        if 'minervini_count' in stage.columns and 'sector_name' in stage.columns:
            heat = (
                stage.groupby(['sector_name', 'minervini_count'])
                .size().reset_index(name='count')
            )
            pivot_heat = heat.pivot(index='sector_name', columns='minervini_count', values='count').fillna(0)
            fig3 = px.imshow(pivot_heat,
                             color_continuous_scale='RdYlGn',
                             labels={'color': 'Industries'},
                             title="Industries per Minervini Score by Sector")
            fig3.update_layout(height=380, margin=dict(t=40, b=10, l=0, r=0))
            st.plotly_chart(fig3, use_container_width=True)

        # Top Stage 2B table
        st.subheader("Stage 2B — Full Minervini (7/7)")
        s2b = stage[(stage['stage'] == '2B') & (stage['minervini_count'] == 7)]
        if not s2b.empty:
            cols2b = [c for c in ['industry_name', 'sector_name', 'minervini_count',
                                   'pct_vs_sma200', 'pct_vs_sma150', 'sma200_slope',
                                   'range_pos_52w', 'pct_from_52w_high'] if c in s2b.columns]
            st.dataframe(s2b[cols2b].sort_values('pct_vs_sma200', ascending=False),
                         use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — STOCKS
# ══════════════════════════════════════════════════════════════════════════

with tab_stocks:
    st.header("Stock Screener — STRONG BUY / BUY Industries")

    if stocks.empty:
        st.info("No stock screener data — run `--mode stock-screener`")
    else:
        sc1, sc2, sc3 = st.columns(3)
        min_composite = sc1.slider("Min composite score", 0, 100, 60)
        min_mc        = sc2.slider("Min Minervini count", 0, 7, 3)
        ind_choices   = ["All"] + sorted(stocks['industry_name'].dropna().unique().tolist())
        sel_ind       = sc3.selectbox("Industry", ind_choices)

        sf = stocks.copy()
        sf = sf[sf['composite'] >= min_composite]
        sf = sf[sf['minervini_count'] >= min_mc]
        if sel_ind != "All":
            sf = sf[sf['industry_name'] == sel_ind]

        st.caption(f"{len(sf)} stocks")

        # Scatter: RS% vs Stage score, sized by Minervini
        if len(sf) > 3:
            fig = px.scatter(
                sf,
                x='rs_pct', y='stage_score',
                size='minervini_count',
                color='ind_faber',
                color_discrete_map=_FABER_COLOR,
                hover_data=['ticker', 'industry_name', 'composite', 'cr_score'],
                labels={'rs_pct': 'RS Percentile', 'stage_score': 'Stage Score',
                        'ind_faber': 'Faber'},
                title="Stock Universe — RS vs Stage Quality",
            )
            fig.update_layout(height=380, margin=dict(t=40, b=10, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)

        show_s = [c for c in ['rank', 'ticker', 'industry_name', 'sector_name',
                               'ind_faber', 'composite', 'composite_pct',
                               'stage', 'minervini_count', 'rs_pct', 'cr_score',
                               'pct_vs_sma200', 'sma200_slope', 'range_pos_52w',
                               'pct_from_52w_high'] if c in sf.columns]
        styled_s = (
            sf[show_s]
            .style
            .map(_color_stage,  subset=['stage']     if 'stage' in show_s else [])
            .map(_color_faber,  subset=['ind_faber'] if 'ind_faber' in show_s else [])
            .background_gradient(subset=['composite'], cmap='RdYlGn', vmin=0, vmax=100)
            .format({'composite': '{:.1f}', 'pct_vs_sma200': '{:+.1f}%',
                     'sma200_slope': '{:+.2f}%', 'range_pos_52w': '{:.0f}%',
                     'pct_from_52w_high': '{:+.1f}%'}, na_rep='–')
        )
        st.dataframe(styled_s, use_container_width=True, height=600)


# ══════════════════════════════════════════════════════════════════════════
# TAB 5 — SIGNAL DELTA
# ══════════════════════════════════════════════════════════════════════════

with tab_delta:
    st.header("Signal Delta — What Changed")

    if delta.empty:
        st.info("No delta data yet — run `--mode severity` on two different dates, "
                "then `--mode signal-delta`.")
    else:
        prev_date = delta['prev_date'].iloc[0] if 'prev_date' in delta.columns else '?'
        curr_date = delta['curr_date'].iloc[0] if 'curr_date' in delta.columns else date_label
        ssd = 'severity_score_delta'

        d1, d2, d3 = st.columns(3)
        d1.metric("Comparing", f"{prev_date} → {curr_date}")
        d2.metric("Upgrades",   int((delta[ssd] > 0).sum()), delta_color="normal")
        d3.metric("Downgrades", int((delta[ssd] < 0).sum()), delta_color="inverse")

        st.subheader("Upgrades")
        ups = delta[delta[ssd] > 0].sort_values(ssd, ascending=False)
        if ups.empty:
            st.caption("No upgrades.")
        else:
            up_cols = [c for c in ['rank', 'industry_name', 'sector_name',
                                    'severity_score_delta', 'severity_label',
                                    'severity_label_prev', 'faber_signal',
                                    'faber_signal_prev', 'stage', 'stage_prev',
                                    'change_summary'] if c in ups.columns]
            st.dataframe(ups[up_cols].style.map(
                _color_severity, subset=['severity_label'] if 'severity_label' in up_cols else []
            ).format({ssd: '{:+.1f}'}, na_rep='–'),
                         use_container_width=True, hide_index=True)

        st.subheader("Downgrades")
        dns = delta[delta[ssd] < 0].sort_values(ssd)
        if dns.empty:
            st.caption("No downgrades.")
        else:
            dn_cols = [c for c in ['rank', 'industry_name', 'sector_name',
                                    'severity_score_delta', 'severity_label',
                                    'severity_label_prev', 'faber_signal',
                                    'faber_signal_prev', 'stage', 'stage_prev',
                                    'change_summary'] if c in dns.columns]
            st.dataframe(dns[dn_cols].style.map(
                _color_severity, subset=['severity_label'] if 'severity_label' in dn_cols else []
            ).format({ssd: '{:+.1f}'}, na_rep='–'),
                         use_container_width=True, hide_index=True)

        # Crossings
        st.subheader("Key Threshold Crossings")
        crossing_defs = [
            ("🟢 New STRONG BUY",
             (delta.get('faber_signal', pd.Series(dtype=str)) == 'STRONG BUY') &
             (delta.get('faber_signal_prev', pd.Series(dtype=str)) != 'STRONG BUY')),
            ("🔴 Lost STRONG BUY",
             (delta.get('faber_signal', pd.Series(dtype=str)) != 'STRONG BUY') &
             (delta.get('faber_signal_prev', pd.Series(dtype=str)) == 'STRONG BUY')),
            ("🟢 New Stage 2A/2B",
             delta.get('stage', pd.Series(dtype=str)).isin(['2A','2B']) &
             ~delta.get('stage_prev', pd.Series(dtype=str)).isin(['2A','2B'])),
            ("🔴 Dropped to Stage 3/4",
             delta.get('stage', pd.Series(dtype=str)).isin(['3','4']) &
             ~delta.get('stage_prev', pd.Series(dtype=str)).isin(['3','4'])),
        ]
        any_cross = False
        for lbl, mask in crossing_defs:
            try:
                sub = delta[mask]
            except Exception:
                continue
            if sub.empty:
                continue
            any_cross = True
            names = ', '.join(sub['industry_name'].tolist())
            st.markdown(f"**{lbl}** ({len(sub)}): {names}")
        if not any_cross:
            st.caption("No major threshold crossings.")
