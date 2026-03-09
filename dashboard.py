"""
Relative Valuation Dashboard
------------------------------
Run with:  streamlit run dashboard.py
"""

import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from config.paths import PREDICTED_PE_RATIO_RESULTS, STOCK_CLUSTERS_FOLDER

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Relative Valuation: S&P 500",
    page_icon="chart",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# THEME / STYLE
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0e1a;
    color: #c9d1e0;
}

/* ── Main background ── */
.stApp { background-color: #0a0e1a; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0d1220;
    border-right: 1px solid #1e2a40;
}
[data-testid="stSidebar"] * { color: #8a9bb5 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label { color: #4a90d9 !important; font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #0f1628;
    border: 1px solid #1e2a40;
    border-radius: 4px;
    padding: 16px 20px;
}
[data-testid="metric-container"] label { color: #4a6fa5 !important; font-size: 0.7rem !important; letter-spacing: 0.12em; text-transform: uppercase; font-family: 'IBM Plex Mono', monospace; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e8edf5 !important; font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem !important; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-family: 'IBM Plex Mono', monospace; }

/* ── Section headers ── */
h1 { font-family: 'IBM Plex Mono', monospace !important; color: #4a90d9 !important; letter-spacing: -0.02em; font-size: 1.4rem !important; }
h2 { font-family: 'IBM Plex Mono', monospace !important; color: #7aa3cc !important; font-size: 1rem !important; letter-spacing: 0.05em; text-transform: uppercase; border-bottom: 1px solid #1e2a40; padding-bottom: 6px; margin-top: 2rem !important; }
h3 { font-family: 'IBM Plex Mono', monospace !important; color: #5a8abf !important; font-size: 0.85rem !important; letter-spacing: 0.08em; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #1e2a40; border-radius: 4px; }
.dvn-scroller { background: #0a0e1a !important; }

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4a6fa5;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #4a90d9 !important;
    border-bottom: 2px solid #4a90d9;
}

/* ── Input fields ── */
.stTextInput input, .stSelectbox select {
    background: #0f1628 !important;
    border: 1px solid #1e2a40 !important;
    color: #c9d1e0 !important;
    font-family: 'IBM Plex Mono', monospace;
    border-radius: 3px;
}

/* ── Signal badges ── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 2px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    font-weight: 600;
}
.badge-strong-under  { background: #0d2e1a; color: #2ecc71; border: 1px solid #1a5c33; }
.badge-under         { background: #0d2419; color: #27ae60; border: 1px solid #145228; }
.badge-strong-over   { background: #2e0d0d; color: #e74c3c; border: 1px solid #5c1a1a; }
.badge-over          { background: #240d0d; color: #c0392b; border: 1px solid #521414; }
.badge-fair          { background: #0d1628; color: #4a90d9; border: 1px solid #1e3a5c; }
.badge-conflict      { background: #1e1a0d; color: #f39c12; border: 1px solid #5c4a14; }
.badge-insig         { background: #181828; color: #7a8aaa; border: 1px solid #2a3050; }

/* ── Divider ── */
hr { border-color: #1e2a40 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1e2a40; border-radius: 3px; }

/* ── Stock lookup card ── */
.lookup-card {
    background: #0f1628;
    border: 1px solid #1e2a40;
    border-radius: 4px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.lookup-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #4a6fa5;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 2px;
}
.lookup-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    color: #e8edf5;
}
.lookup-ticker {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    color: #4a90d9;
    font-weight: 600;
    letter-spacing: -0.02em;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "predicted_ratios", "predicted_pe_ratio_results"
)

SIGNAL_ORDER = [
    "Strong Undervalued",
    "Undervalued (Cluster only)",
    "Undervalued (Index only)",
    "Fairly Valued",
    "Conflicting",
    "Overvalued (Index only)",
    "Overvalued (Cluster only)",
    "Strong Overvalued",
    "Model Insignificant",
    "Insufficient Data",
]

SIGNAL_COLORS = {
    "Strong Undervalued":        "#2ecc71",
    "Undervalued (Cluster only)":"#27ae60",
    "Undervalued (Index only)":  "#1e8449",
    "Fairly Valued":             "#4a90d9",
    "Conflicting":               "#f39c12",
    "Overvalued (Index only)":   "#c0392b",
    "Overvalued (Cluster only)": "#e74c3c",
    "Strong Overvalued":         "#ff6b6b",
    "Model Insignificant":       "#4a6fa5",
    "Insufficient Data":         "#2a3050",
}


@st.cache_data(ttl=3600)
def load_loo_diagnostics(ticker: str, source_cluster: str):
    """
    Load the per-firm LOO regression diagnostics for a given ticker.
    Returns (cluster_diag, index_diag); either may be None if not found.
    """
    import json

    cluster_diag = None
    index_diag   = None

    # Cluster LOO
    cluster_json = os.path.join(RESULTS_DIR, f"loo_diagnostics_{source_cluster}.json")
    if os.path.exists(cluster_json):
        with open(cluster_json) as f:
            data = json.load(f)
        cluster_diag = data.get(ticker)

    # Index LOO
    index_json = os.path.join(RESULTS_DIR, "loo_diagnostics_index.json")
    if os.path.exists(index_json):
        with open(index_json) as f:
            data = json.load(f)
        index_diag = data.get(ticker)

    return cluster_diag, index_diag


@st.cache_data(ttl=3600)
def load_signal_history() -> pd.DataFrame:
    path = os.path.join(PREDICTED_PE_RATIO_RESULTS, "signal_history.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if 'Run Date' in df.columns:
        df['Run Date'] = pd.to_datetime(df['Run Date'])
    return df


@st.cache_data(ttl=3600)
def load_master() -> pd.DataFrame:
    path = os.path.join(RESULTS_DIR, "master_valuations.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Safely coerce numeric columns, only if they exist in this version of the CSV
    numeric_cols = [
        "PE Ratio (Current)",
        "Predicted PE (Cluster)", "Predicted PE (Index)",
        "PE Difference (Cluster)", "PE Difference (Index)",
        "Cluster R²", "Cluster Adj R²", "Cluster F p-value", "Cluster Residual SE",
        "Index R²", "Index Adj R²", "Index F p-value", "Index Residual SE",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Ensure PE Ratio (Current) exists even if missing from older CSV
    if "PE Ratio (Current)" not in df.columns:
        df["PE Ratio (Current)"] = float("nan")
    return df



SIGNAL_CSS = {
    "Strong Undervalued":        "#2ecc71",
    "Undervalued (Cluster only)":"#27ae60",
    "Undervalued (Index only)":  "#1e8449",
    "Fairly Valued":             "#4a90d9",
    "Conflicting":               "#f39c12",
    "Overvalued (Index only)":   "#c0392b",
    "Overvalued (Cluster only)": "#e74c3c",
    "Strong Overvalued":         "#ff6b6b",
    "Model Insignificant":       "#4a6fa5",
}

def diff_col(v):
    try:
        f = float(v)
        if f < -1: return "#2ecc71"
        if f > 1:  return "#e74c3c"
    except Exception:
        pass
    return "#c9d1e0"

def fmt_num(v, signed=False):
    import math
    try:
        f = float(v)
        if math.isnan(f): return "—"
        return (("+" if f >= 0 else "") + f"{f:.2f}") if signed else f"{f:.2f}"
    except Exception:
        return "—"

def build_html_table(df_in, col_defs, max_height=None):
    hdr = "".join(
        f"<th style='padding:7px 10px;text-align:left;color:#4a6fa5;font-size:0.65rem;"
        f"letter-spacing:0.08em;text-transform:uppercase;border-bottom:1px solid #1e2a40;"
        f"white-space:nowrap'>{label}</th>"
        for label, _ in col_defs
    )
    rows = ""
    for _, row in df_in.iterrows():
        cells = ""
        for label, render_fn in col_defs:
            cells += render_fn(row)
        rows += f"<tr style='border-bottom:1px solid #0f1420'>{cells}</tr>"
    scroll = f"max-height:{max_height};overflow-y:auto;" if max_height else ""
    return (
        f"<div style='overflow-x:auto;{scroll}border:1px solid #1e2a40;border-radius:4px'>"
        f"<table style='width:100%;border-collapse:collapse;font-family:IBM Plex Mono,monospace;"
        f"font-size:0.78rem;background:#0f1628;'>"
        f"<thead style='position:sticky;top:0;background:#0d1220;z-index:1'><tr>{hdr}</tr></thead>"
        f"<tbody>{rows}</tbody>"
        f"</table></div>"
    )

def render_loo_table(diag: dict, label: str) -> str:
    """Build an HTML table showing one firm's individual LOO regression output."""
    if not diag:
        return ""

    def p_color(p):
        if p < 0.01: return "#2ecc71"
        if p < 0.05: return "#27ae60"
        if p < 0.10: return "#f39c12"
        return "#e74c3c"

    def p_stars(p):
        if p < 0.01: return "***"
        if p < 0.05: return "**"
        if p < 0.10: return "*"
        return "ns"

    fp      = diag.get("f_pvalue", 1.0)
    r2      = diag.get("r_squared", float("nan"))
    adj_r2  = diag.get("adj_r_squared", float("nan"))
    se      = diag.get("residual_se", float("nan"))
    n       = diag.get("n_obs", "—")
    fstat   = diag.get("f_statistic", float("nan"))

    summary = (
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.72rem;"
        f"color:#4a6fa5;margin-bottom:8px;line-height:1.8'>"
        f"<span style='color:#7aa3cc'>{label}</span> &nbsp;|&nbsp; "
        f"n={n} (peers used in this regression) &nbsp;|&nbsp; "
        f"R²=<span style='color:{p_color(1-r2) if r2 < 0.1 else "#f39c12" if r2 < 0.3 else "#2ecc71"}'>{r2:.4f}</span> &nbsp;|&nbsp; "
        f"Adj R²={adj_r2:.4f} &nbsp;|&nbsp; "
        f"F={fstat:.4f} &nbsp;|&nbsp; "
        f"F p=<span style='color:{p_color(fp)}'>{fp:.4f} {p_stars(fp)}</span> &nbsp;|&nbsp; "
        f"Resid SE={se:.4f}"
        f"</div>"
    )

    hdr = (
        "<tr style='border-bottom:1px solid #1e2a40;color:#4a6fa5;font-size:0.65rem;"
        "letter-spacing:0.08em;text-transform:uppercase'>"
        "<th style='padding:7px 12px;text-align:left'>Variable</th>"
        "<th style='padding:7px 12px;text-align:left'>Coefficient</th>"
        "<th style='padding:7px 12px;text-align:left'>Std Error</th>"
        "<th style='padding:7px 12px;text-align:left'>p-value</th>"
        "<th style='padding:7px 12px;text-align:left'>Significance</th>"
        "<th style='padding:7px 12px;text-align:left'>Interpretation</th>"
        "</tr>"
    )

    def interp(feat, coef, pval):
        sig = pval < 0.10
        direction = "higher" if coef > 0 else "lower"
        if not sig:
            return f"<span style='color:#4a6fa5'>Not significant: {feat} does not reliably explain PE among this firm's peers</span>"
        if feat == "EPS Growth (ROE x Retention)":
            return f"<span style='color:#8a9bb5'>Significant: peers with higher earnings growth trade at {direction} PE multiples</span>"
        if feat == "Beta":
            return f"<span style='color:#8a9bb5'>Significant: higher systematic risk is associated with {direction} PE in this peer group</span>"
        if feat == "Payout Ratio":
            return f"<span style='color:#8a9bb5'>Significant: firms returning more cash to shareholders trade at {direction} PE multiples here</span>"
        return ""

    rows = ""
    const = diag.get("const", float("nan"))
    rows += (
        "<tr style='border-bottom:1px solid #0f1420'>"
        f"<td style='padding:7px 12px;color:#c9d1e0'>Intercept</td>"
        f"<td style='padding:7px 12px;color:#e8edf5;font-weight:600'>{const:+.4f}</td>"
        f"<td style='padding:7px 12px;color:#8a9bb5'>—</td>"
        f"<td style='padding:7px 12px;color:#4a6fa5'>—</td>"
        f"<td style='padding:7px 12px;color:#4a6fa5'>—</td>"
        f"<td style='padding:7px 12px;color:#4a6fa5'>Baseline PE when all predictors are zero</td>"
        "</tr>"
    )
    for c in diag.get("coefficients", []):
        p = c["pvalue"]
        rows += (
            "<tr style='border-bottom:1px solid #0f1420'>"
            f"<td style='padding:7px 12px;color:#c9d1e0'>{c['variable']}</td>"
            f"<td style='padding:7px 12px;color:#e8edf5;font-weight:600'>{c['coef']:+.4f}</td>"
            f"<td style='padding:7px 12px;color:#8a9bb5'>{c['std_err']:.4f}</td>"
            f"<td style='padding:7px 12px;color:{p_color(p)}'>{p:.4f}</td>"
            f"<td style='padding:7px 12px;color:{p_color(p)};font-weight:600'>{p_stars(p)}</td>"
            f"<td style='padding:7px 12px'>{interp(c['variable'], c['coef'], p)}</td>"
            "</tr>"
        )

    table = (
        "<div style='overflow-x:auto'>"
        "<table style='width:100%;border-collapse:collapse;font-family:IBM Plex Mono,monospace;"
        "font-size:0.78rem;background:#0f1628;'>"
        f"<thead><tr>{hdr}</tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table></div>"
    )
    return summary + table


def signal_badge(signal: str) -> str:
    classes = {
        "Strong Undervalued":        "badge-strong-under",
        "Undervalued (Cluster only)":"badge-under",
        "Undervalued (Index only)":  "badge-under",
        "Fairly Valued":             "badge-fair",
        "Conflicting":               "badge-conflict",
        "Overvalued (Index only)":   "badge-over",
        "Overvalued (Cluster only)": "badge-over",
        "Strong Overvalued":         "badge-strong-over",
        "Model Insignificant":       "badge-insig",
    }
    cls = classes.get(signal, "badge-insig")
    return f'<span class="badge {cls}">{signal}</span>'


def color_pe_diff(val):
    if pd.isna(val):
        return "color: #4a6fa5"
    if val < -1:
        return "color: #2ecc71"
    if val > 1:
        return "color: #e74c3c"
    return "color: #c9d1e0"


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        "<div style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;"
        "letter-spacing:0.15em;color:#2a4a6a;text-transform:uppercase;"
        "margin-bottom:4px'>Relative Valuation</div>"
        "<div style='font-family:IBM Plex Mono,monospace;font-size:1.1rem;"
        "color:#4a90d9;font-weight:600;margin-bottom:24px'>S&P 500 Screener</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Navigation")
    page = st.radio(
        "",
        ["Overview", "Strong Signals", "All Firms", "Clusters", "Signal History", "Stock Lookup"],
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Filters")

    df_full = load_master()

    if not df_full.empty and "Source Cluster" in df_full.columns:
        cluster_options = ["All"] + sorted(df_full["Source Cluster"].dropna().unique().tolist())
    else:
        cluster_options = ["All"]
    selected_cluster = st.selectbox("Cluster", cluster_options)

    if not df_full.empty and "Combined Signal" in df_full.columns:
        signal_options = ["All"] + [s for s in SIGNAL_ORDER if s in df_full["Combined Signal"].unique()]
    else:
        signal_options = ["All"]
    selected_signal = st.selectbox("Signal Filter", signal_options)

    if not df_full.empty and "Sector" in df_full.columns:
        sector_options = ["All"] + sorted(df_full["Sector"].dropna().unique().tolist())
    else:
        sector_options = ["All"]
    selected_sector = st.selectbox("Sector", sector_options)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-family:IBM Plex Mono,monospace;font-size:0.62rem;"
        "color:#2a4060;line-height:1.6'>Model: OLS Regression<br>"
        "Clusters: KMeans (K=5)<br>"
        "Predictors: EPS Growth,<br>&nbsp;&nbsp;Beta, Payout Ratio<br>"
        "Method: Leave-One-Out</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# FILTER DATA
# ---------------------------------------------------------------------------

df = df_full.copy()
if selected_cluster != "All":
    df = df[df["Source Cluster"] == selected_cluster]
if selected_signal != "All":
    df = df[df["Combined Signal"] == selected_signal]
if selected_sector != "All" and "Sector" in df.columns:
    df = df[df["Sector"] == selected_sector]

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------

st.markdown(
    "<div style='display:flex;align-items:baseline;gap:12px;margin-bottom:8px'>"
    "<h1 style='margin:0'>S&P 500 Relative Valuation</h1>"
    "<span style='font-family:IBM Plex Mono,monospace;font-size:0.7rem;"
    "color:#2a4a6a;letter-spacing:0.1em'>CLUSTER + INDEX MODEL</span>"
    "</div>",
    unsafe_allow_html=True,
)

# ── Data freshness banner ────────────────────────────────────────────────
import datetime
master_csv_path = os.path.join(PREDICTED_PE_RATIO_RESULTS, "master_valuations.csv")
try:
    mtime = os.path.getmtime(master_csv_path)
    last_run = datetime.datetime.fromtimestamp(mtime).strftime("%B %d, %Y")
    freshness_msg = (
        f"<span style='color:#8a9bb5'>Data as of <strong style='color:#c9d1e0'>{last_run}</strong>. "
        f"This dashboard displays a static snapshot generated by the last local pipeline run. "
        f"Valuations are not live and will not update automatically.</span>"
    )
except Exception:
    freshness_msg = (
        "<span style='color:#8a9bb5'>Data freshness unknown. "
        "This dashboard displays a static snapshot; valuations are not live.</span>"
    )

st.markdown(
    f"<div style='background:#0d1220;border:1px solid #1e2a40;border-left:3px solid #4a6fa5;"
    f"border-radius:4px;padding:10px 16px;margin-bottom:1.5rem;"
    f"font-family:IBM Plex Mono,monospace;font-size:0.75rem'>"
    f"{freshness_msg}"
    f"</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# ── PAGE: OVERVIEW ──
# ---------------------------------------------------------------------------

if page == "Overview":

    # KPI row
    total    = len(df_full)
    strong_u = len(df_full[df_full["Combined Signal"] == "Strong Undervalued"])
    strong_o = len(df_full[df_full["Combined Signal"] == "Strong Overvalued"])
    conflict = len(df_full[df_full["Combined Signal"] == "Conflicting"])
    fair     = len(df_full[df_full["Combined Signal"] == "Fairly Valued"])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Firms Evaluated", f"{total}")
    c2.metric("Strong Undervalued", f"{strong_u}", delta="High conviction")
    c3.metric("Strong Overvalued",  f"{strong_o}", delta="High conviction")
    c4.metric("Fairly Valued",      f"{fair}")
    c5.metric("Conflicting",        f"{conflict}")

    st.markdown("## Signal Distribution")

    signal_counts = (
        df_full["Combined Signal"]
        .value_counts()
        .reindex([s for s in SIGNAL_ORDER if s in df_full["Combined Signal"].unique()])
        .dropna()
    )

    fig_bar = go.Figure(go.Bar(
        x=signal_counts.index.tolist(),
        y=signal_counts.values.tolist(),
        marker_color=[SIGNAL_COLORS.get(s, "#4a6fa5") for s in signal_counts.index],
        marker_line_width=0,
        text=signal_counts.values.tolist(),
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=11, color="#8a9bb5"),
    ))
    fig_bar.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0a0e1a",
        font=dict(family="IBM Plex Mono", color="#8a9bb5", size=11),
        margin=dict(l=20, r=20, t=20, b=40),
        height=280,
        xaxis=dict(showgrid=False, tickangle=-20, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="#141e30", zeroline=False),
        bargap=0.35,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Cluster diagnostics
    st.markdown("## Cluster Regression Diagnostics")

    cluster_diag = (
        df_full.groupby("Source Cluster")
        .agg(
            Firms=("Ticker", "count"),
            R2=("Cluster R²", "first"),
            Adj_R2=("Cluster Adj R²", "first"),
            F_pvalue=("Cluster F p-value", "first"),
            Residual_SE=("Cluster Residual SE", "first"),
        )
        .reset_index()
        .rename(columns={
            "Source Cluster": "Cluster",
            "R2": "R²",
            "Adj_R2": "Adj R²",
            "F_pvalue": "F p-value",
            "Residual_SE": "Residual SE",
        })
    )

    def r2_color(v):
        if v >= 0.3: return "#2ecc71"
        if v >= 0.15: return "#f39c12"
        return "#e74c3c"

    def fp_color(v):
        if v <= 0.05: return "#2ecc71"
        if v <= 0.10: return "#f39c12"
        return "#e74c3c"

    rows_html = ""
    for _, r in cluster_diag.iterrows():
        rows_html += (
            "<tr>"
            f"<td style='padding:10px 14px'>{r['Cluster']}</td>"
            f"<td style='padding:10px 14px'>{int(r['Firms'])}</td>"
            f"<td style='padding:10px 14px;color:{r2_color(r[chr(82)+chr(178)])}'>{r[chr(82)+chr(178)]:.4f}</td>"
            f"<td style='padding:10px 14px;color:{r2_color(r['Adj R' + chr(178)])}'>{r['Adj R' + chr(178)]:.4f}</td>"
            f"<td style='padding:10px 14px;color:{fp_color(r['F p-value'])}'>{r['F p-value']:.4f}</td>"
            f"<td style='padding:10px 14px'>{r['Residual SE']:.4f}</td>"
            "</tr>"
        )

    st.markdown(
        "<table style='width:100%;border-collapse:collapse;font-family:IBM Plex Mono,monospace;"
        "font-size:0.8rem;background:#0f1628;'>"
        "<thead><tr style='border-bottom:1px solid #1e2a40;color:#4a6fa5;font-size:0.7rem;"
        "letter-spacing:0.08em;text-transform:uppercase;'>"
        "<th style='padding:10px 14px;text-align:left'>Cluster</th>"
        "<th style='padding:10px 14px;text-align:left'>Firms</th>"
        "<th style='padding:10px 14px;text-align:left'>R²</th>"
        "<th style='padding:10px 14px;text-align:left'>Adj R²</th>"
        "<th style='padding:10px 14px;text-align:left'>F p-value</th>"
        "<th style='padding:10px 14px;text-align:left'>Residual SE</th>"
        "</tr></thead>"
        f"<tbody style='color:#c9d1e0;'>{rows_html}</tbody>"
        "</table>"
        "<p style='font-size:0.68rem;color:#2a4060;font-family:IBM Plex Mono,monospace;margin-top:6px'>"
        "Color: <span style='color:#2ecc71'>■</span> Strong &nbsp;"
        "<span style='color:#f39c12'>■</span> Borderline &nbsp;"
        "<span style='color:#e74c3c'>■</span> Weak</p>",
        unsafe_allow_html=True,
    )

    # Whole-index diagnostics
    st.markdown("## Whole-Index Regression")
    wi1, wi2, wi3, wi4 = st.columns(4)
    wi_r2   = df_full["Index R²"].iloc[0] if not df_full.empty else 0
    wi_adjr2 = df_full["Index Adj R²"].iloc[0] if not df_full.empty else 0
    wi_fp   = df_full["Index F p-value"].iloc[0] if not df_full.empty else 0
    wi_se   = df_full["Index Residual SE"].iloc[0] if not df_full.empty else 0
    wi1.metric("Index R²",        f"{wi_r2:.4f}")
    wi2.metric("Index Adj R²",    f"{wi_adjr2:.4f}")
    wi3.metric("Index F p-value", f"{wi_fp:.4f}")
    wi4.metric("Index Residual SE", f"{wi_se:.4f}")

    # ── Model Interpretation Expander ─────────────────────────────────────
    with st.expander("How to interpret these results", expanded=False):
        st.markdown("""
<div style='font-family:IBM Plex Mono,monospace;font-size:0.8rem;line-height:1.8;color:#c9d1e0'>

<div style='color:#4a90d9;font-size:0.85rem;font-weight:600;margin-bottom:12px;
letter-spacing:0.05em;text-transform:uppercase'>R²: Goodness of Fit</div>

<p>R² measures what fraction of the cross-sectional variation in PE ratios the model explains
using only EPS growth, Beta, and Payout Ratio.</p>

<table style='width:100%;border-collapse:collapse;margin-bottom:16px'>
  <tr style='border-bottom:1px solid #1e2a40'>
    <td style='padding:6px 12px;color:#2ecc71;width:120px'>R² ≥ 0.30</td>
    <td style='padding:6px 12px'>Strong fit. The three predictors explain a meaningful share of PE dispersion within this peer group.</td>
  </tr>
  <tr style='border-bottom:1px solid #1e2a40'>
    <td style='padding:6px 12px;color:#f39c12'>R² 0.10–0.30</td>
    <td style='padding:6px 12px'>Moderate fit. Signals are usable but should be treated with more caution.</td>
  </tr>
  <tr style='border-bottom:1px solid #1e2a40'>
    <td style='padding:6px 12px;color:#e74c3c'>R² &lt; 0.10</td>
    <td style='padding:6px 12px'>Weak fit. The predictors explain little PE variation; other unmodelled factors dominate. Signals flagged as Model Insignificant.</td>
  </tr>
</table>

<p style='color:#4a6fa5;font-size:0.75rem'>Note: Low R² is expected and not alarming in cross-sectional PE regressions.
At the whole-index level, sector premiums, profitability quality, and competitive moats all drive PE but are absent from the model.
Clustering partially controls for this by grouping comparable firms, which is why cluster R² is typically higher than index R².</p>

<div style='color:#4a90d9;font-size:0.85rem;font-weight:600;margin:20px 0 12px;
letter-spacing:0.05em;text-transform:uppercase'>F p-value: Overall Model Significance</div>

<p>The F-test asks: do the predictors jointly explain PE better than a model with no predictors at all?
A low p-value means yes: the model is statistically meaningful.</p>

<table style='width:100%;border-collapse:collapse;margin-bottom:16px'>
  <tr style='border-bottom:1px solid #1e2a40'>
    <td style='padding:6px 12px;color:#2ecc71;width:120px'>p ≤ 0.05</td>
    <td style='padding:6px 12px'>Significant. High confidence the predictors jointly explain PE variation. Signals are reliable.</td>
  </tr>
  <tr style='border-bottom:1px solid #1e2a40'>
    <td style='padding:6px 12px;color:#f39c12'>p 0.05–0.10</td>
    <td style='padding:6px 12px'>Borderline significant. Signals are generated but should be treated with caution.</td>
  </tr>
  <tr style='border-bottom:1px solid #1e2a40'>
    <td style='padding:6px 12px;color:#e74c3c'>p &gt; 0.10</td>
    <td style='padding:6px 12px'>Not significant. The model cannot reliably distinguish undervalued from overvalued firms in this cluster. All signals are marked Model Insignificant and excluded from combined signals.</td>
  </tr>
</table>

<div style='color:#4a90d9;font-size:0.85rem;font-weight:600;margin:20px 0 12px;
letter-spacing:0.05em;text-transform:uppercase'>Residual Standard Error: Signal Threshold</div>

<p>The Residual SE is the average prediction error of the regression in PE units.
It is used directly as the valuation threshold: a firm must have an actual PE that differs
from its predicted PE by more than ±1 Residual SE to be flagged as under or overvalued.</p>

<table style='width:100%;border-collapse:collapse;margin-bottom:16px'>
  <tr style='border-bottom:1px solid #1e2a40'>
    <td style='padding:6px 12px;color:#2ecc71;width:120px'>SE &lt; 6</td>
    <td style='padding:6px 12px'>Tight band. The model predicts PE precisely; signals require a smaller absolute PE gap to trigger, making them more selective.</td>
  </tr>
  <tr style='border-bottom:1px solid #1e2a40'>
    <td style='padding:6px 12px;color:#f39c12'>SE 6–15</td>
    <td style='padding:6px 12px'>Moderate band. Reasonable precision. A firm needs to be roughly 6–15 PE points away from predicted to be flagged.</td>
  </tr>
  <tr style='border-bottom:1px solid #1e2a40'>
    <td style='padding:6px 12px;color:#e74c3c'>SE &gt; 15</td>
    <td style='padding:6px 12px'>Wide band. High unexplained variation; only extreme deviations trigger a signal. Clusters 1 and 3 fall here, reflecting heterogeneous peer groups.</td>
  </tr>
</table>

<div style='color:#4a90d9;font-size:0.85rem;font-weight:600;margin:20px 0 12px;
letter-spacing:0.05em;text-transform:uppercase'>How Under/Overvalued Signals Are Determined</div>

<p>Each firm's valuation signal is based on the difference between its <em style='color:#8a9bb5'>actual</em>
PE ratio and its <em style='color:#8a9bb5'>predicted</em> PE ratio:</p>

<div style='background:#0d1628;border:1px solid #1e2a40;border-radius:4px;padding:14px 18px;
margin:10px 0 16px;font-family:IBM Plex Mono,monospace;font-size:0.8rem;color:#c9d1e0'>
PE Difference = TTM PE &minus; Predicted PE
</div>

<table style='width:100%;border-collapse:collapse;margin-bottom:16px'>
  <tr style='border-bottom:1px solid #1e2a40'>
    <td style='padding:6px 12px;color:#2ecc71;width:180px'>Negative difference</td>
    <td style='padding:6px 12px'>The market is pricing the firm at a <em style='color:#8a9bb5'>cheaper</em> multiple
    than its fundamentals predict relative to peers. The stock appears undervalued —
    the market is underpricing it compared to comparable firms with similar growth, risk, and payout characteristics.</td>
  </tr>
  <tr style='border-bottom:1px solid #1e2a40'>
    <td style='padding:6px 12px;color:#e74c3c'>Positive difference</td>
    <td style='padding:6px 12px'>The market is paying a <em style='color:#8a9bb5'>premium</em> multiple above what
    the firm's fundamentals justify relative to peers. The stock appears overvalued —
    investors are paying more than comparable firms would suggest.</td>
  </tr>
</table>

<p>However, not every difference triggers a signal. The gap must exceed <strong style='color:#e8edf5'>±1 Residual SE</strong>
(the average prediction error of the regression) to filter out noise:</p>

<div style='background:#0d1628;border:1px solid #1e2a40;border-radius:4px;padding:14px 18px;
margin:10px 0 16px;font-family:IBM Plex Mono,monospace;font-size:0.78rem;line-height:2;color:#c9d1e0'>
TTM PE &lt; Predicted PE &minus; (1 &times; Residual SE) &nbsp;&rarr;&nbsp; <span style='color:#2ecc71'>Undervalued</span><br>
TTM PE &gt; Predicted PE + (1 &times; Residual SE) &nbsp;&rarr;&nbsp; <span style='color:#e74c3c'>Overvalued</span><br>
Otherwise &nbsp;&rarr;&nbsp; <span style='color:#4a90d9'>Fairly Valued</span>
</div>

<p style='color:#4a6fa5;font-size:0.75rem'>Important: these signals are <em>relative</em>, not absolute.
A firm flagged as Undervalued is cheaper than its peers given its fundamentals —
not necessarily cheap in any absolute sense. The market may have legitimate reasons
for the discount that the model does not capture, such as deteriorating earnings quality,
balance sheet risk, or sector-specific headwinds. Signals should always be investigated
further before drawing conclusions.</p>

<div style='color:#4a90d9;font-size:0.85rem;font-weight:600;margin:20px 0 12px;
letter-spacing:0.05em;text-transform:uppercase'>Combined Signal Logic</div>

<p>Each firm receives two independent valuations, one from its cluster regression and one from
the whole-index regression. The combined signal reflects whether they agree:</p>

<table style='width:100%;border-collapse:collapse;margin-bottom:16px'>
  <tr style='border-bottom:1px solid #1e2a40'>
    <td style='padding:6px 12px;color:#2ecc71;width:200px'>Strong Undervalued</td>
    <td style='padding:6px 12px'>Both models independently flag the stock as undervalued. Highest conviction signal.</td>
  </tr>
  <tr style='border-bottom:1px solid #1e2a40'>
    <td style='padding:6px 12px;color:#ff6b6b'>Strong Overvalued</td>
    <td style='padding:6px 12px'>Both models independently flag the stock as overvalued. Highest conviction signal.</td>
  </tr>
  <tr style='border-bottom:1px solid #1e2a40'>
    <td style='padding:6px 12px;color:#4a90d9'>Fairly Valued</td>
    <td style='padding:6px 12px'>Both models agree the stock is within the prediction band.</td>
  </tr>
  <tr style='border-bottom:1px solid #1e2a40'>
    <td style='padding:6px 12px;color:#f39c12'>Conflicting</td>
    <td style='padding:6px 12px'>The two models disagree. Could reflect genuine ambiguity or a structural difference between peer-group and market-wide pricing.</td>
  </tr>
  <tr style='border-bottom:1px solid #1e2a40'>
    <td style='padding:6px 12px;color:#27ae60'>Undervalued (Cluster only)</td>
    <td style='padding:6px 12px'>Only the cluster model flags undervaluation; the index model is insignificant. Moderate conviction.</td>
  </tr>
  <tr style='border-bottom:1px solid #1e2a40'>
    <td style='padding:6px 12px;color:#1e8449'>Undervalued (Index only)</td>
    <td style='padding:6px 12px'>Only the index model flags undervaluation; the cluster model is insignificant. Moderate conviction.</td>
  </tr>
  <tr style='border-bottom:1px solid #1e2a40'>
    <td style='padding:6px 12px;color:#4a6fa5'>Model Insignificant</td>
    <td style='padding:6px 12px'>Both models are statistically insignificant for this firm. No reliable signal can be generated.</td>
  </tr>
</table>

<div style='color:#4a90d9;font-size:0.85rem;font-weight:600;margin:20px 0 12px;
letter-spacing:0.05em;text-transform:uppercase'>Important Limitations</div>

<p style='color:#4a6fa5;font-size:0.78rem;line-height:1.8'>
This model uses only three predictors (EPS growth via ROE×Retention, Beta, and Payout Ratio)
and is intended as a <em style='color:#8a9bb5'>relative</em> valuation tool; it identifies firms that appear mispriced
<em style='color:#8a9bb5'>relative to their peers</em>, not in absolute terms. It does not account for sector premiums,
balance sheet quality, management quality, competitive moats, or macro conditions.
Signals should be treated as a starting point for deeper fundamental analysis, not as
standalone buy or sell recommendations.
</p>

</div>
""", unsafe_allow_html=True)

    # Scatter: actual vs predicted PE (cluster model)
    st.markdown("## Actual vs Predicted PE: Cluster Model")
    scatter_df = df_full.dropna(subset=["PE Ratio (Current)", "Predicted PE (Cluster)"])
    scatter_df = scatter_df[scatter_df["Valuation Signal (Cluster)"] != "Model Insignificant"]

    fig_scatter = px.scatter(
        scatter_df,
        x="Predicted PE (Cluster)",
        y="PE Ratio (Current)",
        color="Combined Signal",
        color_discrete_map=SIGNAL_COLORS,
        hover_data=["Ticker", "PE Difference (Cluster)", "Source Cluster"],
        labels={"Predicted PE (Cluster)": "Predicted PE", "PE Ratio (Current)": "TTM PE"},
    )

    # 45-degree reference line
    mn = scatter_df[["PE Ratio (Current)", "Predicted PE (Cluster)"]].min().min()
    mx = scatter_df[["PE Ratio (Current)", "Predicted PE (Cluster)"]].max().max()
    fig_scatter.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                          line=dict(color="#2a3a5a", width=1, dash="dot"))

    fig_scatter.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        font=dict(family="IBM Plex Mono", color="#8a9bb5", size=11),
        margin=dict(l=20, r=20, t=20, b=20),
        height=380,
        legend=dict(bgcolor="#0f1628", bordercolor="#1e2a40", borderwidth=1,
                    font=dict(size=10)),
        xaxis=dict(showgrid=True, gridcolor="#141e30", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#141e30", zeroline=False),
    )
    fig_scatter.update_traces(marker=dict(size=7, opacity=0.8, line=dict(width=0)))
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Sector Breakdown ────────────────────────────────────────────────────
    if "Sector" in df_full.columns:
        st.markdown("## Sector Breakdown")

        # Count signals by sector
        sector_signal_df = (
            df_full[df_full["Combined Signal"].isin([
                "Strong Undervalued", "Strong Overvalued",
                "Undervalued (Cluster only)", "Overvalued (Cluster only)",
                "Undervalued (Index only)", "Overvalued (Index only)",
                "Conflicting", "Fairly Valued", "Model Insignificant",
            ])]
            .groupby(["Sector", "Combined Signal"])
            .size()
            .reset_index(name="Count")
        )

        # Stacked bar: signal distribution per sector
        sector_order = (
            df_full.groupby("Sector")["Ticker"].count()
            .sort_values(ascending=True)
            .index.tolist()
        )

        fig_sector = go.Figure()
        for signal in [s for s in SIGNAL_ORDER if s in sector_signal_df["Combined Signal"].unique()]:
            sub = sector_signal_df[sector_signal_df["Combined Signal"] == signal]
            sector_map = dict(zip(sub["Sector"], sub["Count"]))
            fig_sector.add_trace(go.Bar(
                name=signal,
                y=sector_order,
                x=[sector_map.get(s, 0) for s in sector_order],
                orientation="h",
                marker_color=SIGNAL_COLORS.get(signal, "#4a6fa5"),
                marker_line_width=0,
            ))

        fig_sector.update_layout(
            barmode="stack",
            paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
            font=dict(family="IBM Plex Mono", color="#8a9bb5", size=11),
            margin=dict(l=20, r=20, t=20, b=20),
            height=420,
            legend=dict(bgcolor="#0f1628", bordercolor="#1e2a40", borderwidth=1,
                        font=dict(size=10), orientation="h", y=-0.15),
            xaxis=dict(showgrid=True, gridcolor="#141e30", zeroline=False, title="Firms"),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_sector, use_container_width=True)

        # Strong signals by sector
        st.markdown("## Strong Signals by Sector")
        strong_sector = (
            df_full[df_full["Combined Signal"].isin(["Strong Undervalued", "Strong Overvalued"])]
            .groupby(["Sector", "Combined Signal"])
            .size()
            .reset_index(name="Count")
        )

        if not strong_sector.empty:
            fig_strong_sector = go.Figure()
            for signal, color in [("Strong Undervalued", "#2ecc71"), ("Strong Overvalued", "#e74c3c")]:
                sub = strong_sector[strong_sector["Combined Signal"] == signal]
                if sub.empty:
                    continue
                fig_strong_sector.add_trace(go.Bar(
                    name=signal,
                    x=sub["Sector"],
                    y=sub["Count"],
                    marker_color=color,
                    marker_line_width=0,
                    text=sub["Count"],
                    textposition="outside",
                    textfont=dict(family="IBM Plex Mono", size=10, color="#8a9bb5"),
                ))
            fig_strong_sector.update_layout(
                barmode="group",
                paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
                font=dict(family="IBM Plex Mono", color="#8a9bb5", size=11),
                margin=dict(l=20, r=20, t=20, b=80),
                height=320,
                legend=dict(bgcolor="#0f1628", bordercolor="#1e2a40", borderwidth=1, font=dict(size=10)),
                xaxis=dict(showgrid=False, tickangle=-30),
                yaxis=dict(showgrid=True, gridcolor="#141e30", zeroline=False),
                bargap=0.3,
            )
            st.plotly_chart(fig_strong_sector, use_container_width=True)

            st.markdown(
                "<p style='font-size:0.75rem;color:#4a6fa5;font-family:IBM Plex Mono,monospace'>"
                "Sectors with disproportionate strong signals may indicate the model is systematically "
                "mispricing firms in that sector, often because sector-specific PE premiums are not "
                "captured by the three predictors. Treat concentrated sector signals with extra caution."
                "</p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<p style='color:#4a6fa5;font-family:IBM Plex Mono,monospace;font-size:0.8rem'>"
                "No strong signals found in current data.</p>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# ── PAGE: STRONG SIGNALS ──
# ---------------------------------------------------------------------------

elif page == "Strong Signals":

    st.markdown("## High-Conviction Signals")
    st.markdown(
        "<p style='font-size:0.8rem;color:#4a6fa5;font-family:IBM Plex Mono,monospace'>"
        "Firms where both the cluster regression and whole-index regression independently "
        "agree on valuation direction. Strongest possible signal from this model.</p>",
        unsafe_allow_html=True,
    )

    for signal, label in [
        ("Strong Undervalued", " STRONG UNDERVALUED"),
        ("Strong Overvalued",  " STRONG OVERVALUED"),
    ]:
        sig_df = df_full[df_full["Combined Signal"] == signal].copy()
        if sig_df.empty:
            continue

        if "Undervalued" in signal:
            sig_df = sig_df.sort_values("PE Difference (Cluster)", ascending=True)
        else:
            sig_df = sig_df.sort_values("PE Difference (Cluster)", ascending=False)

        st.markdown(f"### {label}  ({len(sig_df)} firms)")

        display_cols = [
            "Ticker", "PE Ratio (Current)", "Predicted PE (Cluster)",
            "PE Difference (Cluster)", "Predicted PE (Index)",
            "PE Difference (Index)", "Source Cluster",
        ]
        display_df = sig_df[display_cols].copy()

        col_defs = [
            ("Ticker",           lambda r: f"<td style='padding:7px 10px;color:#e8edf5;font-weight:600'>{r['Ticker']}</td>"),
            ("TTM PE",            lambda r: f"<td style='padding:7px 10px'>{fmt_num(r['PE Ratio (Current)'])}</td>"),
            ("Pred PE (Cluster)", lambda r: f"<td style='padding:7px 10px'>{fmt_num(r['Predicted PE (Cluster)'])}</td>"),
            ("Diff (Cluster)",    lambda r: f"<td style='padding:7px 10px;color:{diff_col(r['PE Difference (Cluster)'])}'>{fmt_num(r['PE Difference (Cluster)'], signed=True)}</td>"),
            ("Pred PE (Index)",   lambda r: f"<td style='padding:7px 10px'>{fmt_num(r['Predicted PE (Index)'])}</td>"),
            ("Diff (Index)",      lambda r: f"<td style='padding:7px 10px;color:{diff_col(r["PE Difference (Index)"])}'>{fmt_num(r['PE Difference (Index)'], signed=True)}</td>"),
            ("Cluster",           lambda r: f"<td style='padding:7px 10px;color:#4a6fa5'>{r['Source Cluster']}</td>"),
        ]
        st.markdown(build_html_table(display_df, col_defs), unsafe_allow_html=True)
        st.markdown("")

    # PE Difference waterfall for strong signals
    strong_df = df_full[df_full["Combined Signal"].isin(["Strong Undervalued", "Strong Overvalued"])].copy()
    if not strong_df.empty:
        st.markdown("## PE Gap: Cluster Model")
        strong_df = strong_df.sort_values("PE Difference (Cluster)")
        colors = ["#2ecc71" if v < 0 else "#e74c3c" for v in strong_df["PE Difference (Cluster)"]]
        fig_wf = go.Figure(go.Bar(
            x=strong_df["Ticker"],
            y=strong_df["PE Difference (Cluster)"],
            marker_color=colors,
            marker_line_width=0,
            text=[f"{v:+.1f}" for v in strong_df["PE Difference (Cluster)"]],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=10, color="#8a9bb5"),
        ))
        fig_wf.update_layout(
            paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
            font=dict(family="IBM Plex Mono", color="#8a9bb5", size=11),
            margin=dict(l=20, r=20, t=20, b=40),
            height=300,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#141e30", zeroline=True,
                       zerolinecolor="#2a3a5a"),
            shapes=[dict(type="line", x0=-0.5, y0=0, x1=len(strong_df)-0.5, y1=0,
                         line=dict(color="#2a3a5a", width=1))],
        )
        st.plotly_chart(fig_wf, use_container_width=True)


# ---------------------------------------------------------------------------
# ── PAGE: ALL FIRMS ──
# ---------------------------------------------------------------------------

elif page == "All Firms":

    st.markdown(f"## All Firms  <span style='font-size:0.75rem;color:#2a4060'>({len(df)} of {len(df_full)})</span>", unsafe_allow_html=True)

    display_cols = [
        "Ticker", "Sector", "PE Ratio (Current)",
        "Predicted PE (Cluster)", "PE Difference (Cluster)",
        "Valuation Signal (Cluster)",
        "Predicted PE (Index)", "PE Difference (Index)",
        "Valuation Signal (Index)", "Combined Signal",
        "Source Cluster",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    display_df = df[display_cols].copy()

    def color_signal(val):
        colors = {
            "Strong Undervalued":        "color: #2ecc71; font-weight: 600",
            "Undervalued (Cluster only)":"color: #27ae60",
            "Undervalued (Index only)":  "color: #1e8449",
            "Fairly Valued":             "color: #4a90d9",
            "Conflicting":               "color: #f39c12",
            "Overvalued (Index only)":   "color: #c0392b",
            "Overvalued (Cluster only)": "color: #e74c3c",
            "Strong Overvalued":         "color: #ff6b6b; font-weight: 600",
            "Model Insignificant":       "color: #4a6fa5",
        }
        return colors.get(str(val), "color: #c9d1e0")

    has_sector = "Sector" in display_df.columns
    sector_col_def = [("Sector", lambda r: f"<td style='padding:7px 10px;color:#7aa3cc'>{r.get('Sector','—')}</td>")] if has_sector else []
    all_col_defs = [
        ("Ticker",            lambda r: f"<td style='padding:7px 10px;color:#e8edf5;font-weight:600'>{r['Ticker']}</td>"),
    ] + sector_col_def + [
        ("TTM PE",             lambda r: f"<td style='padding:7px 10px'>{fmt_num(r['PE Ratio (Current)'])}</td>"),
        ("Pred PE (C)",        lambda r: f"<td style='padding:7px 10px'>{fmt_num(r['Predicted PE (Cluster)'])}</td>"),
        ("Diff (C)",           lambda r: f"<td style='padding:7px 10px;color:{diff_col(r['PE Difference (Cluster)'])}'>{fmt_num(r['PE Difference (Cluster)'], signed=True)}</td>"),
        ("Signal (Cluster)",   lambda r: f"<td style='padding:7px 10px;color:{SIGNAL_CSS.get(str(r['Valuation Signal (Cluster)']), chr(35)+chr(99)+chr(57)+chr(100)+chr(49)+chr(101)+chr(48))};white-space:nowrap'>{r['Valuation Signal (Cluster)']}</td>"),
        ("Pred PE (I)",        lambda r: f"<td style='padding:7px 10px'>{fmt_num(r['Predicted PE (Index)'])}</td>"),
        ("Diff (I)",           lambda r: f"<td style='padding:7px 10px;color:{diff_col(r['PE Difference (Index)'])}'>{fmt_num(r['PE Difference (Index)'], signed=True)}</td>"),
        ("Signal (Index)",     lambda r: f"<td style='padding:7px 10px;color:{SIGNAL_CSS.get(str(r['Valuation Signal (Index)']), chr(35)+chr(99)+chr(57)+chr(100)+chr(49)+chr(101)+chr(48))};white-space:nowrap'>{r['Valuation Signal (Index)']}</td>"),
        ("Combined",           lambda r: f"<td style='padding:7px 10px;color:{SIGNAL_CSS.get(str(r['Combined Signal']), chr(35)+chr(99)+chr(57)+chr(100)+chr(49)+chr(101)+chr(48))};font-weight:600;white-space:nowrap'>{r['Combined Signal']}</td>"),
        ("Cluster",            lambda r: f"<td style='padding:7px 10px;color:#4a6fa5'>{r['Source Cluster']}</td>"),
    ]
    st.markdown(build_html_table(display_df, all_col_defs, max_height='580px'), unsafe_allow_html=True)

    # Download button
    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇  Export filtered results as CSV",
        data=csv,
        file_name="valuation_results.csv",
        mime="text/csv",
    )



# ---------------------------------------------------------------------------
# ── PAGE: CLUSTERS ──
# ---------------------------------------------------------------------------

elif page == "Clusters":

    st.markdown("## Cluster Profiles")
    st.markdown(
        "<p style='font-family:IBM Plex Mono,monospace;font-size:0.8rem;color:#4a6fa5;"
        "margin-bottom:1.5rem'>Each cluster groups firms with similar EPS growth, market risk (Beta), "
        "and dividend yield. Firms within the same cluster are treated as comparable peers "
        "in the regression model.</p>",
        unsafe_allow_html=True,
    )

    # Load cluster files

    cluster_files = sorted([
        f for f in os.listdir(STOCK_CLUSTERS_FOLDER)
        if f.startswith("cluster_") and f.endswith(".csv")
        and f != "all_clusters_combined.csv"
    ])

    # Build cluster summaries
    cluster_data = {}
    for fname in cluster_files:
        cid = fname.replace("cluster_", "").replace(".csv", "")
        try:
            cdf = pd.read_csv(os.path.join(STOCK_CLUSTERS_FOLDER, fname))
            cluster_data[cid] = cdf
        except Exception:
            pass

    # Auto-label clusters based on characteristics
    def auto_label(cdf):
        growth = cdf["EPS Growth (ROE x Retention)"].mean() if "EPS Growth (ROE x Retention)" in cdf.columns else 0
        beta   = cdf["Beta"].mean() if "Beta" in cdf.columns else 0
        div    = cdf["Dividend Yield"].mean() if "Dividend Yield" in cdf.columns else 0

        if beta > 2.0:
            return "High-Risk / Speculative"
        if div > 4.0:
            return "High-Yield Defensive"
        if div > 2.5:
            return "Income: Moderate Yield"
        if div > 1.2 and beta < 0.9:
            return "Low-Risk Income"
        if div > 1.0 and beta >= 0.9:
            return "Dividend + Market Risk"
        if growth > 0.6:
            return "High-Growth Compounders"
        if growth > 0.25:
            return "Growth: Moderate Risk"
        if growth > 0.10 and beta > 1.2:
            return "Growth: Elevated Risk"
        return "Moderate Growth / Blend"

    # ── Profile Cards ───────────────────────────────────────────────────────
    cols_per_row = 3
    cids = list(cluster_data.keys())
    for row_start in range(0, len(cids), cols_per_row):
        row_cids = cids[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, cid in zip(cols, row_cids):
            cdf = cluster_data[cid]
            label  = auto_label(cdf)
            n      = len(cdf)
            growth = cdf["EPS Growth (ROE x Retention)"].mean() if "EPS Growth (ROE x Retention)" in cdf.columns else float("nan")
            beta   = cdf["Beta"].mean() if "Beta" in cdf.columns else float("nan")
            div    = cdf["Dividend Yield"].mean() if "Dividend Yield" in cdf.columns else float("nan")

            # Regression diagnostics from master table
            if "Source Cluster" in df_full.columns:
                try:
                    sc = df_full["Source Cluster"].astype(str)
                    # Source Cluster is stored as "cluster_1", cid is "1"
                    mask = (sc == f"cluster_{cid}") | (sc == cid) | (sc == str(int(cid)))
                except Exception:
                    mask = pd.Series([False]*len(df_full))
            else:
                mask = pd.Series([False]*len(df_full))
            r2_val  = df_full.loc[mask, "Cluster R²"].iloc[0] if mask.any() and "Cluster R²" in df_full.columns else None
            fp_val  = df_full.loc[mask, "Cluster F p-value"].iloc[0] if mask.any() and "Cluster F p-value" in df_full.columns else None
            rse_val = df_full.loc[mask, "Cluster Residual SE"].iloc[0] if mask.any() and "Cluster Residual SE" in df_full.columns else None

            # Colour the model quality indicator
            if r2_val is None or pd.isna(r2_val):
                model_badge = "<span style='color:#4a6fa5'>No Model</span>"
            elif r2_val >= 0.30:
                model_badge = f"<span style='color:#2ecc71'>R²={r2_val:.2f}</span>"
            elif r2_val >= 0.10:
                model_badge = f"<span style='color:#f39c12'>R²={r2_val:.2f}</span>"
            else:
                model_badge = f"<span style='color:#e74c3c'>R²={r2_val:.2f}</span>"

            fp_str  = f"{fp_val:.4f}" if fp_val is not None and not pd.isna(fp_val) else "—"
            rse_str = f"{rse_val:.1f}" if rse_val is not None and not pd.isna(rse_val) else "—"

            col.markdown(
                f"""<div style='background:#0f1628;border:1px solid #1e2a40;border-radius:6px;
                padding:18px 20px;margin-bottom:12px;height:100%'>
                <div style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;
                color:#2a4a6a;letter-spacing:0.15em;text-transform:uppercase;
                margin-bottom:4px'>Cluster {cid}</div>
                <div style='font-family:IBM Plex Mono,monospace;font-size:0.95rem;
                color:#7aa3cc;font-weight:600;margin-bottom:14px'>{label}</div>
                <div style='display:grid;grid-template-columns:1fr 1fr;gap:6px 16px;
                font-family:IBM Plex Mono,monospace;font-size:0.75rem;margin-bottom:14px'>
                  <div><span style='color:#4a6fa5'>Firms</span><br>
                    <span style='color:#e8edf5'>{n}</span></div>
                  <div><span style='color:#4a6fa5'>Avg Beta</span><br>
                    <span style='color:#e8edf5'>{beta:.2f}</span></div>
                  <div><span style='color:#4a6fa5'>Avg EPS Growth</span><br>
                    <span style='color:#e8edf5'>{growth*100:.1f}%</span></div>
                  <div><span style='color:#4a6fa5'>Avg Div Yield</span><br>
                    <span style='color:#e8edf5'>{div:.2f}%</span></div>
                </div>
                <div style='border-top:1px solid #1e2a40;padding-top:10px;
                font-family:IBM Plex Mono,monospace;font-size:0.72rem;
                display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px'>
                  <div><span style='color:#4a6fa5'>Model Fit</span><br>{model_badge}</div>
                  <div><span style='color:#4a6fa5'>F p-val</span><br>
                    <span style='color:{"#2ecc71" if fp_val is not None and not pd.isna(fp_val) and fp_val<=0.05 else "#f39c12" if fp_val is not None and not pd.isna(fp_val) and fp_val<=0.10 else "#e74c3c"}'>{fp_str}</span></div>
                  <div><span style='color:#4a6fa5'>Resid SE</span><br>
                    <span style='color:#c9d1e0'>{rse_str}</span></div>
                </div>
                </div>""",
                unsafe_allow_html=True,
            )

    # ── Sector Heatmap ──────────────────────────────────────────────────────
    st.markdown("## Sector Composition by Cluster")
    st.markdown(
        "<p style='font-family:IBM Plex Mono,monospace;font-size:0.78rem;color:#4a6fa5;"
        "margin-bottom:1rem'>Shows what proportion of each cluster belongs to each GICS sector. "
        "Concentrated sector columns suggest the model may be capturing sector effects "
        "rather than pure fundamental similarity; signals from those clusters should be "
        "interpreted with extra caution.</p>",
        unsafe_allow_html=True,
    )

    has_sector_data = "Sector" in df_full.columns and df_full["Sector"].notna().any() and (df_full["Sector"] != "Unknown").any()
    if has_sector_data and "Source Cluster" in df_full.columns:
        # Build count matrix: rows=Sector, cols=Cluster
        heat_df = (
            df_full.groupby(["Sector", "Source Cluster"])
            .size()
            .reset_index(name="Count")
        )
        pivot = heat_df.pivot(index="Sector", columns="Source Cluster", values="Count").fillna(0)

        # Convert to % of cluster total (column-wise normalisation)
        pivot_pct = pivot.div(pivot.sum(axis=0), axis=1) * 100

        # Sort sectors by total count descending
        pivot_pct = pivot_pct.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

        fig_heat = go.Figure(go.Heatmap(
            z=pivot_pct.values,
            x=[f"Cluster {c}" for c in pivot_pct.columns],
            y=pivot_pct.index.tolist(),
            colorscale=[
                [0.0,  "#0a0e1a"],
                [0.15, "#0d1f3c"],
                [0.35, "#1a4a7a"],
                [0.60, "#2a7abf"],
                [0.80, "#4a90d9"],
                [1.0,  "#7ab8f5"],
            ],
            text=[[f"{v:.0f}%" for v in row] for row in pivot_pct.values],
            texttemplate="%{text}",
            textfont=dict(family="IBM Plex Mono", size=10, color="#e8edf5"),
            hovertemplate="<b>%{y}</b><br>%{x}<br>%{text} of cluster<extra></extra>",
            showscale=True,
            colorbar=dict(
                tickfont=dict(family="IBM Plex Mono", color="#8a9bb5", size=10),
                title=dict(text="%", font=dict(family="IBM Plex Mono", color="#8a9bb5", size=10)),
                bgcolor="#0f1628",
                bordercolor="#1e2a40",
                borderwidth=1,
                len=0.8,
            ),
        ))

        fig_heat.update_layout(
            paper_bgcolor="#0a0e1a",
            plot_bgcolor="#0a0e1a",
            font=dict(family="IBM Plex Mono", color="#8a9bb5", size=11),
            margin=dict(l=20, r=20, t=20, b=20),
            height=420,
            xaxis=dict(showgrid=False, side="top", tickfont=dict(color="#7aa3cc")),
            yaxis=dict(showgrid=False, tickfont=dict(color="#8a9bb5")),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # ── Sector concentration warning ────────────────────────────────
        # Flag any cluster where a single sector exceeds 40%
        warnings = []
        for col in pivot_pct.columns:
            top_sector = pivot_pct[col].idxmax()
            top_pct    = pivot_pct[col].max()
            if top_pct > 40:
                warnings.append((col, top_sector, top_pct))

        if warnings:
            st.markdown("## Sector Concentration Warnings")
            for cid_w, sector_w, pct_w in warnings:
                st.markdown(
                    f"<div style='background:#1a0f0f;border:1px solid #4a1a1a;border-radius:4px;"
                    f"padding:10px 16px;margin-bottom:8px;font-family:IBM Plex Mono,monospace;"
                    f"font-size:0.78rem'>"
                    f"<span style='color:#e74c3c'>Cluster {cid_w}</span>"
                    f"<span style='color:#8a9bb5'> ({pct_w:.0f}% of firms are in "
                    f"<span style='color:#f39c12'>{sector_w}</span>. "
                    f"Signals may reflect sector-level PE premiums rather than firm-specific "
                    f"mispricings. Interpret with caution.</span></div>",
                    unsafe_allow_html=True,
                )

    else:
        st.markdown(
            "<div style='background:#0f1628;border:1px solid #1e2a40;border-radius:4px;"
            "padding:14px 18px;font-family:IBM Plex Mono,monospace;font-size:0.78rem'>"
            "<span style='color:#f39c12'>Sector data not available.</span>"
            "<span style='color:#4a6fa5'> Re-run Stage 1 (Data Collection) to fetch GICS sector "
            "labels from Wikipedia. The heatmap will appear automatically after the next full pipeline run.</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Cluster firm list ────────────────────────────────────────────────────
    st.markdown("## Firms by Cluster")
    st.markdown(
        "<p style='font-family:IBM Plex Mono,monospace;font-size:0.78rem;color:#4a6fa5;"
        "margin-bottom:1rem'>Select a cluster to see all member firms and their key characteristics.</p>",
        unsafe_allow_html=True,
    )

    selected_view_cluster = st.selectbox(
        "Select Cluster",
        options=[f"Cluster {c}" for c in sorted(cluster_data.keys(), key=int)],
        key="cluster_firm_select",
    )
    view_cid = selected_view_cluster.replace("Cluster ", "")

    if view_cid in cluster_data:
        view_df = cluster_data[view_cid].copy()

        # Merge signal from master
        if "Ticker" in view_df.columns and "Combined Signal" in df_full.columns:
            sig_cols = ["Ticker", "Combined Signal", "PE Ratio (Current)",
                        "Predicted PE (Cluster)", "PE Difference (Cluster)"]
            sig_cols = [c for c in sig_cols if c in df_full.columns]
            view_df = view_df.merge(df_full[sig_cols], on="Ticker", how="left")

        # Merge sector
        if "Sector" not in view_df.columns and "Sector" in df_full.columns:
            view_df = view_df.merge(df_full[["Ticker", "Sector"]], on="Ticker", how="left")

        # Build display columns
        show_cols = ["Ticker"]
        for c in ["Sector", "EPS Growth (ROE x Retention)", "Beta", "Dividend Yield",
                  "PE Ratio (Current)", "Predicted PE (Cluster)",
                  "PE Difference (Cluster)", "Combined Signal"]:
            if c in view_df.columns:
                show_cols.append(c)

        display_view = view_df[show_cols].copy()

        def fmt_num_cl(v, pct=False, signed=False):
            if pd.isna(v): return "—"
            if pct: return f"{v*100:.1f}%"
            prefix = "+" if signed and v > 0 else ""
            return f"{prefix}{v:.2f}"

        col_defs = [
            ("Ticker",     lambda r: f"<td style='padding:7px 10px;color:#e8edf5;font-weight:600'>{r['Ticker']}</td>"),
        ]
        if "Sector" in display_view.columns:
            col_defs.append(("Sector", lambda r: f"<td style='padding:7px 10px;color:#7aa3cc'>{r.get('Sector','—')}</td>"))
        if "EPS Growth (ROE x Retention)" in display_view.columns:
            col_defs.append(("EPS Growth", lambda r: f"<td style='padding:7px 10px'>{fmt_num_cl(r['EPS Growth (ROE x Retention)'], pct=True)}</td>"))
        if "Beta" in display_view.columns:
            col_defs.append(("Beta", lambda r: f"<td style='padding:7px 10px'>{fmt_num_cl(r['Beta'])}</td>"))
        if "Dividend Yield" in display_view.columns:
            col_defs.append(("Div Yield", lambda r: f"<td style='padding:7px 10px'>{fmt_num_cl(r['Dividend Yield'])}</td>"))
        if "PE Ratio (Current)" in display_view.columns:
            col_defs.append(("TTM PE",    lambda r: f"<td style='padding:7px 10px'>{fmt_num_cl(r['PE Ratio (Current)'])}</td>"))
        if "Predicted PE (Cluster)" in display_view.columns:
            col_defs.append(("Pred PE", lambda r: f"<td style='padding:7px 10px'>{fmt_num_cl(r['Predicted PE (Cluster)'])}</td>"))
        if "PE Difference (Cluster)" in display_view.columns:
            col_defs.append(("Diff", lambda r: f"<td style='padding:7px 10px;color:{diff_col(r['PE Difference (Cluster)'])}'>{fmt_num_cl(r['PE Difference (Cluster)'], signed=True)}</td>"))
        if "Combined Signal" in display_view.columns:
            col_defs.append(("Signal", lambda r: f"<td style='padding:7px 10px;color:{SIGNAL_CSS.get(str(r.get("Combined Signal","")),"#c9d1e0")};white-space:nowrap'>{r.get("Combined Signal","—")}</td>"))

        st.markdown(build_html_table(display_view, col_defs, max_height="480px"), unsafe_allow_html=True)



# ---------------------------------------------------------------------------
# ── PAGE: SIGNAL HISTORY ──
# ---------------------------------------------------------------------------

elif page == "Signal History":

    st.markdown("## Signal History")
    st.markdown(
        "<p style='font-family:IBM Plex Mono,monospace;font-size:0.8rem;color:#4a6fa5;"
        "margin-bottom:1.5rem'>Tracks which firms have been flagged as strong signals "
        "across pipeline runs. Firms appearing consistently across multiple runs represent "
        "more persistent mispricings and warrant closer attention than one-off appearances.</p>",
        unsafe_allow_html=True,
    )

    history = load_signal_history()

    if history.empty:
        st.markdown(
            "<div style='background:#0f1628;border:1px solid #1e2a40;border-radius:4px;"
            "padding:14px 18px;font-family:IBM Plex Mono,monospace;font-size:0.78rem'>"
            "<span style='color:#f39c12'>No signal history yet.</span>"
            "<span style='color:#4a6fa5'> History is recorded each time Stage 3 is run. "
            "Re-run the pipeline to generate the first entry.</span>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        n_runs = history['Run Date'].nunique()
        first_run = history['Run Date'].min().strftime("%b %d, %Y")
        last_run  = history['Run Date'].max().strftime("%b %d, %Y")

        c1, c2, c3 = st.columns(3)
        c1.metric("Pipeline Runs Logged", f"{n_runs}")
        c2.metric("First Run", first_run)
        c3.metric("Latest Run", last_run)

        # ── Persistence Table ────────────────────────────────────────────
        st.markdown("## Most Persistent Signals")
        st.markdown(
            "<p style='font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:#4a6fa5'>"
            "Firms ranked by how many runs they have appeared in. "
            "A persistence score of 100% means flagged in every run.</p>",
            unsafe_allow_html=True,
        )

        # Count appearances per ticker
        persistence = (
            history.groupby('Ticker')
            .agg(
                Appearances=('Run Date', 'nunique'),
                Signal=('Combined Signal', lambda x: x.mode()[0] if len(x) > 0 else '—'),
                Cluster=('Source Cluster', lambda x: x.mode()[0] if len(x) > 0 else '—'),
                Latest_PE_Diff=('PE Difference (Cluster)', 'last'),
            )
            .reset_index()
        )
        if 'Sector' in history.columns:
            sector_map = history.groupby('Ticker')['Sector'].last()
            persistence = persistence.merge(sector_map, on='Ticker', how='left')

        persistence['Persistence (%)'] = (persistence['Appearances'] / n_runs * 100).round(0).astype(int)
        persistence = persistence.sort_values(['Appearances', 'Latest_PE_Diff'], ascending=[False, True])

        # Colour persistence score
        def persist_color(pct):
            if pct >= 80: return "#2ecc71"
            if pct >= 50: return "#f39c12"
            return "#4a6fa5"

        persist_col_defs = [
            ("Ticker",          lambda r: f"<td style='padding:7px 10px;color:#e8edf5;font-weight:600'>{r['Ticker']}</td>"),
            ("Appearances",     lambda r: f"<td style='padding:7px 10px;text-align:center;color:{persist_color(r['Persistence (%)'])};font-weight:600'>{r['Appearances']} / {n_runs}</td>"),
            ("Persistence",     lambda r: f"<td style='padding:7px 10px;text-align:center;color:{persist_color(r['Persistence (%)'])}'>{r['Persistence (%)']}%</td>"),
            ("Most Common Signal", lambda r: f"<td style='padding:7px 10px;color:{SIGNAL_CSS.get(str(r.get("Signal","")),"#c9d1e0")};white-space:nowrap'>{r.get('Signal','—')}</td>"),
            ("Cluster",         lambda r: f"<td style='padding:7px 10px;color:#4a6fa5'>{r.get('Cluster','—')}</td>"),
        ]
        if 'Sector' in persistence.columns:
            persist_col_defs.insert(4, ("Sector", lambda r: f"<td style='padding:7px 10px;color:#7aa3cc'>{r.get('Sector','—')}</td>"))
        if 'Latest_PE_Diff' in persistence.columns:
            persist_col_defs.append(("Latest PE Diff", lambda r: f"<td style='padding:7px 10px;color:{diff_col(r.get("Latest_PE_Diff",0))}'>{fmt_num(r.get('Latest_PE_Diff',float('nan')), signed=True)}</td>"))

        st.markdown(build_html_table(persistence, persist_col_defs, max_height="520px"), unsafe_allow_html=True)

        # ── Signal trend chart ───────────────────────────────────────────
        st.markdown("## Signal Count Over Time")
        st.markdown(
            "<p style='font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:#4a6fa5'>"
            "Total strong signals flagged per run. A rising count may indicate "
            "broader market dislocation; a falling count suggests convergence.</p>",
            unsafe_allow_html=True,
        )

        run_counts = (
            history.groupby(['Run Date', 'Combined Signal'])
            .size()
            .reset_index(name='Count')
        )

        fig_trend = go.Figure()
        for signal, color in [
            ("Strong Undervalued",        "#2ecc71"),
            ("Strong Overvalued",         "#e74c3c"),
            ("Undervalued (Cluster only)","#27ae60"),
            ("Overvalued (Cluster only)", "#c0392b"),
            ("Undervalued (Index only)",  "#82e0aa"),
            ("Overvalued (Index only)",   "#f1948a"),
        ]:
            sub = run_counts[run_counts['Combined Signal'] == signal]
            if sub.empty:
                continue
            fig_trend.add_trace(go.Scatter(
                x=sub['Run Date'],
                y=sub['Count'],
                name=signal,
                mode='lines+markers',
                line=dict(color=color, width=2),
                marker=dict(size=7),
            ))

        fig_trend.update_layout(
            paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
            font=dict(family="IBM Plex Mono", color="#8a9bb5", size=11),
            margin=dict(l=20, r=20, t=20, b=20),
            height=320,
            legend=dict(bgcolor="#0f1628", bordercolor="#1e2a40", borderwidth=1,
                       font=dict(size=10), orientation="h", y=-0.25),
            xaxis=dict(showgrid=True, gridcolor="#141e30", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="#141e30", zeroline=False, title="Firms"),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # ── Raw history table ────────────────────────────────────────────
        st.markdown("## Full History Log")
        with st.expander("View raw log"):
            display_history = history.copy()
            display_history['Run Date'] = display_history['Run Date'].dt.strftime("%b %d, %Y")
            display_history = display_history.sort_values('Run Date', ascending=False).reset_index(drop=True)

            # Build columns dynamically from whatever columns exist
            raw_col_defs = []
            for col in display_history.columns:
                if col == 'Run Date':
                    raw_col_defs.append((col, lambda r, c=col: f"<td style='padding:7px 10px;color:#7aaac8;white-space:nowrap'>{r[c]}</td>"))
                elif col == 'Combined Signal':
                    raw_col_defs.append((col, lambda r, c=col: f"<td style='padding:7px 10px;color:{SIGNAL_CSS.get(str(r.get(c,'')),chr(35)+'c9d1e0')};white-space:nowrap'>{r.get(c,'—')}</td>"))
                elif col in ('PE Ratio (Current)', 'Predicted PE (Cluster)', 'PE Difference (Cluster)'):
                    raw_col_defs.append((col, lambda r, c=col: f"<td style='padding:7px 10px;color:#c9d1e0'>{fmt_num(r.get(c, float('nan')))}</td>"))
                else:
                    raw_col_defs.append((col, lambda r, c=col: f"<td style='padding:7px 10px;color:#8a9bb5'>{r.get(c,'—')}</td>"))

            st.markdown(build_html_table(display_history, raw_col_defs, max_height="400px"), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# ── PAGE: STOCK LOOKUP ──
# ---------------------------------------------------------------------------

elif page == "Stock Lookup":

    st.markdown("## Individual Stock Lookup")

    ticker_input = st.text_input(
        "Enter ticker symbol",
        placeholder="e.g. AAPL, MSFT, JPM",
        max_chars=10,
    ).upper().strip()

    if ticker_input:
        match = df_full[df_full["Ticker"] == ticker_input]

        if match.empty:
            st.warning(f"**{ticker_input}** not found in the valuation results. "
                       "It may have been excluded due to missing data.")
        else:
            row = match.iloc[0]
            combined = str(row.get("Combined Signal", "—"))

            # Header row
            col_t, col_b = st.columns([1, 3])
            with col_t:
                st.markdown(
                    f"<div class='lookup-card'>"
                    f"<div class='lookup-ticker'>{ticker_input}</div>"
                    f"<div style='margin-top:8px'>{signal_badge(combined)}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with col_b:
                r1, r2, r3 = st.columns(3)
                r1.metric("TTM PE",              f"{row['PE Ratio (Current)']:.2f}" if pd.notna(row['PE Ratio (Current)']) else "—")
                r2.metric("Predicted PE (Cluster)", f"{row['Predicted PE (Cluster)']:.2f}" if pd.notna(row['Predicted PE (Cluster)']) else "—",
                          delta=f"{row['PE Difference (Cluster)']:+.2f}" if pd.notna(row['PE Difference (Cluster)']) else None)
                r3.metric("Predicted PE (Index)", f"{row['Predicted PE (Index)']:.2f}" if pd.notna(row['Predicted PE (Index)']) else "—",
                          delta=f"{row['PE Difference (Index)']:+.2f}" if pd.notna(row['PE Difference (Index)']) else None)

            # Details grid
            st.markdown("### Fundamentals & Model Inputs")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("EPS Growth (ROE×Ret)",
                      f"{row['EPS Growth (ROE x Retention)']:.4f}" if pd.notna(row['EPS Growth (ROE x Retention)']) else "—")
            d2.metric("Beta",
                      f"{row['Beta']:.3f}" if pd.notna(row['Beta']) else "—")
            d3.metric("Payout Ratio",
                      f"{row['Payout Ratio']:.2f}%" if pd.notna(row['Payout Ratio']) else "—")
            d4.metric("Cluster",
                      str(row.get("Source Cluster", "—")))

            st.markdown("### Cluster Model Diagnostics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Cluster R²",       f"{row['Cluster R²']:.4f}"       if pd.notna(row['Cluster R²']) else "—")
            m2.metric("Cluster Adj R²",   f"{row['Cluster Adj R²']:.4f}"   if pd.notna(row['Cluster Adj R²']) else "—")
            m3.metric("Cluster F p-val",  f"{row['Cluster F p-value']:.4f}" if pd.notna(row['Cluster F p-value']) else "—")
            m4.metric("Cluster Resid SE", f"{row['Cluster Residual SE']:.4f}" if pd.notna(row['Cluster Residual SE']) else "—")

            # Per-firm LOO regression output
            firm_cluster = str(row.get("Source Cluster", ""))
            c_diag, i_diag = load_loo_diagnostics(ticker_input, firm_cluster)

            if c_diag or i_diag:
                st.markdown("### Individual LOO Regression Output")
                st.markdown(
                    "<p style='font-size:0.78rem;color:#4a6fa5;font-family:IBM Plex Mono,monospace'>​"
                    "These are the exact regression coefficients from the leave-one-out model "
                    "that priced this specific firm. The regression was fitted on all other firms "
                    "in the peer group with this firm excluded, so these coefficients are "
                    "independent of the firm being valued.</p>",
                    unsafe_allow_html=True,
                )
                if c_diag:
                    st.markdown(render_loo_table(c_diag, f"Cluster Model ({firm_cluster})"), unsafe_allow_html=True)
                    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                if i_diag:
                    st.markdown(render_loo_table(i_diag, "Index Model (whole S&P 500)"), unsafe_allow_html=True)

            st.markdown("### Signal Breakdown")
            sig_col1, sig_col2 = st.columns(2)
            with sig_col1:
                cluster_signal = str(row.get('Valuation Signal (Cluster)', '—'))
                cluster_diff   = f"{row['PE Difference (Cluster)']:+.2f}" if pd.notna(row['PE Difference (Cluster)']) else '—'
                st.markdown(
                    f"<div class='lookup-card'>"
                    f"<div class='lookup-label'>Cluster Signal</div>"
                    f"<div style='margin-top:6px'>{signal_badge(cluster_signal)}</div>"
                    f"<div class='lookup-label' style='margin-top:12px'>PE Difference</div>"
                    f"<div class='lookup-value'>{cluster_diff}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with sig_col2:
                index_signal = str(row.get('Valuation Signal (Index)', '—'))
                index_diff   = f"{row['PE Difference (Index)']:+.2f}" if pd.notna(row['PE Difference (Index)']) else '—'
                st.markdown(
                    f"<div class='lookup-card'>"
                    f"<div class='lookup-label'>Index Signal</div>"
                    f"<div style='margin-top:6px'>{signal_badge(index_signal)}</div>"
                    f"<div class='lookup-label' style='margin-top:12px'>PE Difference</div>"
                    f"<div class='lookup-value'>{index_diff}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Mini bar chart: actual vs both predicted
            pe_actual  = row["PE Ratio (Current)"]
            pe_cluster = row["Predicted PE (Cluster)"]
            pe_index   = row["Predicted PE (Index)"]

            if any(pd.notna(v) for v in [pe_actual, pe_cluster, pe_index]):
                st.markdown("### PE Comparison")
                labels = ["TTM PE", "Predicted (Cluster)", "Predicted (Index)"]
                values = [pe_actual, pe_cluster, pe_index]
                bar_colors = ["#4a90d9", "#2ecc71" if pe_cluster < pe_actual else "#e74c3c",
                              "#2ecc71" if pe_index < pe_actual else "#e74c3c"]

                fig_mini = go.Figure(go.Bar(
                    x=[v for v in values if pd.notna(v)],
                    y=[l for l, v in zip(labels, values) if pd.notna(v)],
                    orientation="h",
                    marker_color=[c for c, v in zip(bar_colors, values) if pd.notna(v)],
                    marker_line_width=0,
                    text=[f"{v:.2f}" for v in values if pd.notna(v)],
                    textposition="outside",
                    textfont=dict(family="IBM Plex Mono", size=12, color="#8a9bb5"),
                ))
                fig_mini.update_layout(
                    paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
                    font=dict(family="IBM Plex Mono", color="#8a9bb5", size=11),
                    margin=dict(l=20, r=60, t=10, b=10),
                    height=160,
                    xaxis=dict(showgrid=True, gridcolor="#141e30", zeroline=False),
                    yaxis=dict(showgrid=False),
                )
                st.plotly_chart(fig_mini, use_container_width=True)
    else:
        st.markdown(
            "<p style='color:#2a4060;font-family:IBM Plex Mono,monospace;font-size:0.85rem'>"
            "Enter a ticker symbol above to look up its valuation signals and model diagnostics.</p>",
            unsafe_allow_html=True,
        )