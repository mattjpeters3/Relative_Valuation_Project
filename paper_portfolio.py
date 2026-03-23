"""
Paper Portfolio Dashboard
Tracks hypothetical $1,000 notional positions opened on Strong Undervalued
signals and closed when the signal changes. Compares each position against
the S&P 500 over the same holding period.
"""

import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import datetime

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Paper Portfolio",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# PATHS — mirrors config/paths.py
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "predicted_ratios", "predicted_pe_ratio_results")
POSITIONS_PATH         = os.path.join(RESULTS_DIR, "paper_positions.csv")
POSITIONS_PATH_CLUSTER = os.path.join(RESULTS_DIR, "paper_positions_cluster.csv")

# ---------------------------------------------------------------------------
# THEME
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    background-color: #0a0e1a;
    color: #c9d1e0;
    font-family: 'IBM Plex Mono', monospace;
}
h1, h2, h3 { color: #c9d4e8; font-family: 'IBM Plex Mono', monospace; }
.stMetric { background: #0d1220; border: 1px solid #1e2a40; border-radius: 4px; padding: 12px; }
.stMetric label { color: #4a7fa5 !important; font-size: 0.72rem !important; letter-spacing: 0.08em; }
.stMetric [data-testid="stMetricValue"] { color: #c9d4e8 !important; font-size: 1.3rem !important; }
.stMetric [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }
div[data-testid="stExpander"] { background: #0d1220; border: 1px solid #1e2a40; border-radius: 4px; }
div[data-testid="stExpander"] summary { color: #4a7fa5; font-size: 0.8rem; letter-spacing: 0.05em; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

NOTIONAL = 1000.0

def fmt_pct(val, decimals=2):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "n/a"
    return f"{val:+.{decimals}f}%"

def fmt_dollar(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "n/a"
    sign = "+" if val >= 0 else ""
    return f"{sign}${val:,.2f}"

def color_val(val, positive_good=True):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "#8a9bb5"
    if positive_good:
        return "#2ecc71" if val > 0 else "#e74c3c" if val < 0 else "#8a9bb5"
    else:
        return "#e74c3c" if val > 0 else "#2ecc71" if val < 0 else "#8a9bb5"

def build_html_table(df, col_defs, max_height="500px"):
    header = "".join(
        f"<th style='padding:8px 10px;text-align:left;color:#4a7fa5;"
        f"font-size:0.7rem;letter-spacing:0.08em;border-bottom:1px solid #1e2a40;"
        f"white-space:nowrap'>{label}</th>"
        for label, _ in col_defs
    )
    rows = ""
    for _, row in df.iterrows():
        cells = "".join(fn(row) for _, fn in col_defs)
        rows += (
            f"<tr style='border-bottom:1px solid #0f1628;"
            f"font-size:0.8rem;font-family:IBM Plex Mono,monospace'>{cells}</tr>"
        )
    return (
        f"<div style='overflow-y:auto;max-height:{max_height};"
        f"border:1px solid #1e2a40;border-radius:4px'>"
        f"<table style='width:100%;border-collapse:collapse;"
        f"background:#0a0e1a'>"
        f"<thead style='position:sticky;top:0;background:#0d1220'>"
        f"<tr>{header}</tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_positions():
    if not os.path.exists(POSITIONS_PATH):
        return pd.DataFrame()
    df = pd.read_csv(POSITIONS_PATH)
    for col in ["Entry Date", "Exit Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["Entry Price", "Entry SP500", "Exit Price", "Exit SP500",
                "Stock Return", "SP500 Return", "Excess Return", "Dollar PnL"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@st.cache_data(ttl=3600)
def load_positions_cluster():
    if not os.path.exists(POSITIONS_PATH_CLUSTER):
        return pd.DataFrame()
    df = pd.read_csv(POSITIONS_PATH_CLUSTER)
    for col in ["Entry Date", "Exit Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["Entry Price", "Entry SP500", "Exit Price", "Exit SP500",
                "Stock Return", "SP500 Return", "Excess Return", "Dollar PnL"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fetch_live_prices(tickers):
    """Fetch current closing prices for a list of tickers."""
    prices = {}
    if not tickers:
        return prices
    try:
        sp500 = yf.Ticker("^GSPC").history(period="2d")
        prices["^GSPC"] = float(sp500["Close"].iloc[-1]) if not sp500.empty else None
    except Exception:
        prices["^GSPC"] = None
    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(period="2d")
            prices[ticker] = float(hist["Close"].iloc[-1]) if not hist.empty else None
        except Exception:
            prices[ticker] = None
    return prices

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------

st.markdown(
    "<div style='display:flex;align-items:baseline;gap:12px;margin-bottom:8px'>"
    "<h1 style='margin:0'>Paper Portfolio</h1>"
    "<span style='font-family:IBM Plex Mono,monospace;font-size:0.7rem;"
    "color:#2a4a6a;letter-spacing:0.1em'>SIGNAL EVALUATION TRACKER</span>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div style='background:#0d1220;border:1px solid #1e2a40;border-left:3px solid #4a7fa5;"
    "border-radius:4px;padding:10px 16px;margin-bottom:1.5rem;"
    "font-family:IBM Plex Mono,monospace;font-size:0.75rem'>"
    "<span style='color:#8a9bb5'>Tracks hypothetical $1,000 notional positions opened on "
    "<strong style='color:#2ecc71'>Strong Undervalued</strong> signals. Positions close when "
    "the signal changes to Fairly Valued, Conflicting, Overvalued, or when the firm drops from "
    "the model. All returns are compared against the S&P 500 over the same holding period. "
    "This is a signal evaluation exercise, not a portfolio.</span>"
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------

all_positions   = load_positions()
all_positions_c = load_positions_cluster()

if all_positions.empty and all_positions_c.empty:
    st.markdown(
        "<div style='background:#0f1628;border:1px solid #1e2a40;border-radius:4px;"
        "padding:14px 18px;font-family:IBM Plex Mono,monospace;font-size:0.78rem'>"
        "<span style='color:#f39c12'>No positions yet.</span>"
        "<span style='color:#4a6fa5'> Positions are opened automatically when the pipeline "
        "identifies Strong Undervalued signals. Run the full pipeline to generate the first entries.</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

open_pos    = all_positions[all_positions["Status"] == "Open"].copy()   if not all_positions.empty   else pd.DataFrame()
closed_pos  = all_positions[all_positions["Status"] == "Closed"].copy() if not all_positions.empty   else pd.DataFrame()
open_pos_c  = all_positions_c[all_positions_c["Status"] == "Open"].copy()   if not all_positions_c.empty else pd.DataFrame()
closed_pos_c= all_positions_c[all_positions_c["Status"] == "Closed"].copy() if not all_positions_c.empty else pd.DataFrame()

# ---------------------------------------------------------------------------
# SIDEBAR FILTER — exclude early testing period
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        "<p style='font-family:IBM Plex Mono,monospace;font-size:0.75rem;"
        "color:#4a7fa5;letter-spacing:0.06em;margin-bottom:0.5rem'>"
        "FILTER</p>",
        unsafe_allow_html=True,
    )

    all_dates = pd.concat([
        all_positions["Entry Date"].dropna() if not all_positions.empty else pd.Series(dtype="datetime64[ns]"),
        all_positions_c["Entry Date"].dropna() if not all_positions_c.empty else pd.Series(dtype="datetime64[ns]"),
    ])
    min_date = all_dates.min().date() if not all_dates.empty else datetime.date(2026, 1, 1)
    max_date = datetime.date.today()

    filter_from = st.date_input(
        "Exclude positions opened before",
        value=min_date,
        min_value=min_date,
        max_value=max_date,
        help="Set this to exclude early testing runs from the win rate breakdowns.",
    )
    st.markdown(
        "<p style='font-family:IBM Plex Mono,monospace;font-size:0.68rem;"
        "color:#2a4a6a;margin-top:0.25rem'>"
        "Use this to exclude early testing runs that used daily cadence "
        "rather than the normal weekly schedule.</p>",
        unsafe_allow_html=True,
    )

    filter_ts    = pd.Timestamp(filter_from)
    closed_pos   = closed_pos[closed_pos["Entry Date"]   >= filter_ts] if not closed_pos.empty   else closed_pos
    closed_pos_c = closed_pos_c[closed_pos_c["Entry Date"] >= filter_ts] if not closed_pos_c.empty else closed_pos_c
    st.markdown("---")
    st.markdown(
        f"<p style='font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#4a6fa5'>"
        f"Showing <strong style='color:#c9d1e0'>{len(closed_pos)}</strong> main / "
        f"<strong style='color:#c9d1e0'>{len(closed_pos_c)}</strong> cluster closed position(s) "
        f"from {filter_from.strftime('%b %d, %Y')} onward.</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# HELPER: compute unrealised metrics for open positions
# ---------------------------------------------------------------------------

def enrich_open(open_df):
    if open_df.empty:
        return open_df
    tickers    = open_df["Ticker"].tolist()
    prices     = fetch_live_prices(tickers)
    sp500_live = prices.get("^GSPC")

    def _unreal(row):
        cur = prices.get(row["Ticker"])
        ent = row["Entry Price"]
        return ((cur / ent) - 1) * 100 if cur and ent and pd.notna(ent) else None

    def _sp500_since(row):
        esp = row["Entry SP500"]
        return ((sp500_live / esp) - 1) * 100 if sp500_live and esp and pd.notna(esp) else None

    open_df = open_df.copy()
    open_df["Current Price"]     = open_df["Ticker"].map(prices)
    open_df["Unrealised Return"] = open_df.apply(_unreal, axis=1)
    open_df["SP500 Since Entry"] = open_df.apply(_sp500_since, axis=1)
    open_df["Excess (Live)"]     = open_df.apply(
        lambda r: r["Unrealised Return"] - r["SP500 Since Entry"]
        if pd.notna(r.get("Unrealised Return")) and pd.notna(r.get("SP500 Since Entry")) else None, axis=1)
    open_df["Days Held"] = open_df["Entry Date"].apply(
        lambda d: (datetime.date.today() - d.date()).days if pd.notna(d) else None)
    open_df["Unrealised PnL"] = open_df["Unrealised Return"].apply(
        lambda r: (NOTIONAL * r / 100) if pd.notna(r) else None)
    return open_df

open_pos   = enrich_open(open_pos)
open_pos_c = enrich_open(open_pos_c)

# ---------------------------------------------------------------------------
# HELPER: render a full tracker section (used by both tabs)
# ---------------------------------------------------------------------------

def render_tracker(open_df, closed_df, tracker_label):

    # ── KPIs ──────────────────────────────────────────────────────────────
    st.markdown("## Summary")
    n_open   = len(open_df)
    n_closed = len(closed_df)

    total_pnl     = closed_df["Dollar PnL"].sum()     if not closed_df.empty else 0.0
    avg_stock_ret = closed_df["Stock Return"].mean()  if not closed_df.empty else None
    avg_sp500_ret = closed_df["SP500 Return"].mean()  if not closed_df.empty else None
    avg_excess    = closed_df["Excess Return"].mean() if not closed_df.empty else None
    win_rate      = (
        (closed_df["Excess Return"] > 0).sum() / len(closed_df) * 100
        if not closed_df.empty else None
    )
    unreal_pnl = open_df["Unrealised PnL"].sum() if not open_df.empty and "Unrealised PnL" in open_df.columns else 0.0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Open Positions",         f"{n_open}")
    c2.metric("Closed Positions",       f"{n_closed}")
    c3.metric("Realised Profit/Loss",   fmt_dollar(total_pnl))
    c4.metric("Unrealised Profit/Loss", fmt_dollar(unreal_pnl))
    c5.metric("Avg Excess Return",      fmt_pct(avg_excess) if avg_excess is not None else "n/a")
    c6.metric("Win Rate",               f"{win_rate:.0f}%" if win_rate is not None else "n/a",
              help="% of closed positions that outperformed the S&P 500")

    # ── Open Positions ────────────────────────────────────────────────────
    st.markdown("## Open Positions")
    st.markdown(
        "<p style='font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:#4a6fa5;"
        "margin-bottom:1rem'>Live prices fetched at page load. Returns are unrealised "
        "and will be finalised when the signal changes on the next pipeline run.</p>",
        unsafe_allow_html=True,
    )
    if open_df.empty:
        st.markdown(
            "<p style='font-family:IBM Plex Mono,monospace;font-size:0.78rem;color:#4a6fa5'>"
            "No open positions at this time.</p>", unsafe_allow_html=True)
    else:
        def _open_unreal_ret(r):
            c = color_val(r.get("Unrealised Return"))
            return f"<td style='padding:7px 10px;text-align:right;color:{c}'>{fmt_pct(r.get('Unrealised Return'))}</td>"
        def _open_excess(r):
            c = color_val(r.get("Excess (Live)"))
            return f"<td style='padding:7px 10px;text-align:right;color:{c}'>{fmt_pct(r.get('Excess (Live)'))}</td>"
        def _open_pnl(r):
            c = color_val(r.get("Unrealised PnL"))
            return f"<td style='padding:7px 10px;text-align:right;color:{c}'>{fmt_dollar(r.get('Unrealised PnL'))}</td>"
        def _open_entry_price(r):
            return f"<td style='padding:7px 10px;text-align:right;color:#8a9bb5'>${r['Entry Price']:.2f}</td>" if pd.notna(r.get('Entry Price')) else "<td style='padding:7px 10px;color:#4a6a8a'>n/a</td>"
        def _open_curr_price(r):
            return f"<td style='padding:7px 10px;text-align:right;color:#c9d1e0'>${r['Current Price']:.2f}</td>" if pd.notna(r.get('Current Price')) else "<td style='padding:7px 10px;color:#4a6a8a'>n/a</td>"

        open_col_defs = [
            ("Ticker",      lambda r: f"<td style='padding:7px 10px;color:#e8edf5;font-weight:600'>{r['Ticker']}</td>"),
            ("Sector",      lambda r: f"<td style='padding:7px 10px;color:#7aaac8'>{r.get('Sector', 'n/a')}</td>"),
            ("Entry Date",  lambda r: f"<td style='padding:7px 10px;color:#8a9bb5;white-space:nowrap'>{r['Entry Date'].strftime('%b %d, %Y') if pd.notna(r['Entry Date']) else 'n/a'}</td>"),
            ("Days Held",   lambda r: f"<td style='padding:7px 10px;text-align:center;color:#8a9bb5'>{int(r['Days Held']) if pd.notna(r.get('Days Held')) else 'n/a'}</td>"),
            ("Entry Price", _open_entry_price),
            ("Current Price", _open_curr_price),
            ("Unrealised Return", _open_unreal_ret),
            ("S&P 500 Return", lambda r: f"<td style='padding:7px 10px;text-align:right;color:#8a9bb5'>{fmt_pct(r.get('SP500 Since Entry'))}</td>"),
            ("Excess Return", _open_excess),
            ("Unrealised Profit/Loss", _open_pnl),
        ]
        st.markdown(build_html_table(open_df, open_col_defs), unsafe_allow_html=True)

    # ── Closed Positions ──────────────────────────────────────────────────
    st.markdown("## Closed Positions")
    if closed_df.empty:
        st.markdown(
            "<p style='font-family:IBM Plex Mono,monospace;font-size:0.78rem;color:#4a6fa5'>"
            "No closed positions yet.</p>", unsafe_allow_html=True)
    else:
        def _cl_stock_ret(r):
            c = color_val(r.get("Stock Return"))
            return f"<td style='padding:7px 10px;text-align:right;color:{c}'>{fmt_pct(r.get('Stock Return'))}</td>"
        def _cl_excess(r):
            c = color_val(r.get("Excess Return"))
            return f"<td style='padding:7px 10px;text-align:right;color:{c}'>{fmt_pct(r.get('Excess Return'))}</td>"
        def _cl_pnl(r):
            c = color_val(r.get("Dollar PnL"))
            return f"<td style='padding:7px 10px;text-align:right;color:{c}'>{fmt_dollar(r.get('Dollar PnL'))}</td>"

        closed_col_defs = [
            ("Ticker",      lambda r: f"<td style='padding:7px 10px;color:#e8edf5;font-weight:600'>{r['Ticker']}</td>"),
            ("Sector",      lambda r: f"<td style='padding:7px 10px;color:#7aaac8'>{r.get('Sector', 'n/a')}</td>"),
            ("Entry Date",  lambda r: f"<td style='padding:7px 10px;color:#8a9bb5;white-space:nowrap'>{r['Entry Date'].strftime('%b %d, %Y') if pd.notna(r['Entry Date']) else 'n/a'}</td>"),
            ("Exit Date",   lambda r: f"<td style='padding:7px 10px;color:#8a9bb5;white-space:nowrap'>{r['Exit Date'].strftime('%b %d, %Y') if pd.notna(r['Exit Date']) else 'n/a'}</td>"),
            ("Days Held",   lambda r: f"<td style='padding:7px 10px;text-align:center;color:#8a9bb5'>{int(r['Holding Days']) if pd.notna(r.get('Holding Days')) else 'n/a'}</td>"),
            ("Exit Signal", lambda r: f"<td style='padding:7px 10px;color:#f39c12;white-space:nowrap'>{r.get('Exit Signal', 'n/a')}</td>"),
            ("Stock Return", _cl_stock_ret),
            ("S&P 500 Return", lambda r: f"<td style='padding:7px 10px;text-align:right;color:#8a9bb5'>{fmt_pct(r.get('SP500 Return'))}</td>"),
            ("Excess Return", _cl_excess),
            ("Profit/Loss", _cl_pnl),
        ]
        st.markdown(build_html_table(closed_df.sort_values("Exit Date", ascending=False), closed_col_defs), unsafe_allow_html=True)

        # ── Cumulative PnL chart ──────────────────────────────────────────
        st.markdown("## Cumulative Performance vs S&P 500")
        chart_df = closed_df.dropna(subset=["Exit Date", "Dollar PnL", "SP500 Return"]).copy()
        chart_df = chart_df.sort_values("Exit Date")
        chart_df["Cumulative PnL"]       = chart_df["Dollar PnL"].cumsum()
        chart_df["SP500 Dollar PnL"]     = NOTIONAL * chart_df["SP500 Return"] / 100
        chart_df["Cumulative SP500 PnL"] = chart_df["SP500 Dollar PnL"].cumsum()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=chart_df["Exit Date"], y=chart_df["Cumulative PnL"],
            name="Signal Strategy", mode="lines+markers",
            line=dict(color="#2ecc71", width=2), marker=dict(size=7)))
        fig.add_trace(go.Scatter(x=chart_df["Exit Date"], y=chart_df["Cumulative SP500 PnL"],
            name="S&P 500 Benchmark", mode="lines+markers",
            line=dict(color="#4a7fa5", width=2, dash="dash"), marker=dict(size=7)))
        fig.add_hline(y=0, line_color="#1e2a40", line_width=1)
        fig.update_layout(
            paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
            font=dict(family="IBM Plex Mono", color="#8a9bb5", size=11),
            margin=dict(l=20, r=20, t=20, b=20), height=340,
            legend=dict(bgcolor="#0f1628", bordercolor="#1e2a40", borderwidth=1,
                        font=dict(size=10), orientation="h", y=-0.2),
            xaxis=dict(showgrid=True, gridcolor="#141e30", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="#141e30", zeroline=False,
                       title="Cumulative PnL ($)", tickprefix="$"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Per-position excess return bar chart ──────────────────────────
        st.markdown("## Excess Return Per Position")
        bar_df = closed_df.dropna(subset=["Excess Return"]).sort_values("Excess Return")
        bar_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in bar_df["Excess Return"]]
        fig2 = go.Figure(go.Bar(
            x=bar_df["Ticker"], y=bar_df["Excess Return"],
            marker_color=bar_colors,
            text=[fmt_pct(v) for v in bar_df["Excess Return"]],
            textposition="outside", textfont=dict(size=10, color="#8a9bb5"),
        ))
        fig2.add_hline(y=0, line_color="#4a7fa5", line_width=1)
        fig2.update_layout(
            paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
            font=dict(family="IBM Plex Mono", color="#8a9bb5", size=11),
            margin=dict(l=20, r=20, t=20, b=20), height=320,
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="#141e30", zeroline=False,
                       title="Excess Return (%)", ticksuffix="%"),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── Closed position statistics ────────────────────────────────────
        st.markdown("## Closed Position Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Stock Return",   fmt_pct(avg_stock_ret))
        c2.metric("Avg S&P 500 Return", fmt_pct(avg_sp500_ret))
        c3.metric("Avg Excess Return",  fmt_pct(avg_excess),
                  delta=f"{'Outperformed' if avg_excess and avg_excess > 0 else 'Underperformed'}")
        avg_days = closed_df["Holding Days"].mean()
        c4.metric("Avg Holding Period", f"{avg_days:.0f} days" if pd.notna(avg_days) else "n/a")

        # ── Win rate by exit signal ───────────────────────────────────────
        st.markdown("## Win Rate by Exit Signal")
        if "Exit Signal" in closed_df.columns and not closed_df["Exit Signal"].isna().all():
            sig_groups = []
            for sig, grp in closed_df.groupby("Exit Signal"):
                n    = len(grp)
                wins = (grp["Excess Return"] > 0).sum()
                sig_groups.append({
                    "Exit Signal": sig, "Positions": n, "Wins": int(wins),
                    "Win Rate": wins / n * 100 if n > 0 else None,
                    "Avg Excess Return": grp["Excess Return"].mean(),
                    "Avg Holding Days":  grp["Holding Days"].mean() if "Holding Days" in grp.columns else None,
                })
            sig_df = pd.DataFrame(sig_groups).sort_values("Positions", ascending=False)
            sig_col_defs = [
                ("Exit Signal",       lambda r: f"<td style='padding:7px 10px;color:#f39c12;white-space:nowrap'>{r['Exit Signal']}</td>"),
                ("Positions",         lambda r: f"<td style='padding:7px 10px;text-align:center;color:#8a9bb5'>{r['Positions']}</td>"),
                ("Wins",              lambda r: f"<td style='padding:7px 10px;text-align:center;color:#2ecc71'>{r['Wins']}</td>"),
                ("Win Rate",          lambda r: f"<td style='padding:7px 10px;text-align:right;color:{'#2ecc71' if r['Win Rate'] and r['Win Rate'] >= 50 else '#e74c3c'}'>{fmt_pct(r['Win Rate'], 1)}</td>"),
                ("Avg Excess Return", lambda r: f"<td style='padding:7px 10px;text-align:right;color:{'#2ecc71' if pd.notna(r['Avg Excess Return']) and r['Avg Excess Return'] > 0 else '#e74c3c'}'>{fmt_pct(r['Avg Excess Return'])}</td>"),
                ("Avg Holding Days",  lambda r: (f"<td style='padding:7px 10px;text-align:center;color:#8a9bb5'>{r['Avg Holding Days']:.0f}d</td>" if pd.notna(r.get('Avg Holding Days')) else "<td style='padding:7px 10px;color:#4a6a8a'>n/a</td>")),
            ]
            st.markdown(build_html_table(sig_df, sig_col_defs, max_height="300px"), unsafe_allow_html=True)
        else:
            st.markdown("<p style='font-family:IBM Plex Mono,monospace;font-size:0.78rem;color:#4a6fa5'>No exit signal data yet.</p>", unsafe_allow_html=True)

        # ── Win rate by holding period ────────────────────────────────────
        st.markdown("## Win Rate by Holding Period")
        if "Holding Days" in closed_df.columns and not closed_df["Holding Days"].isna().all():
            def _bucket(d):
                if pd.isna(d): return "Unknown"
                if d <= 14:    return "0-14 days"
                if d <= 28:    return "15-28 days"
                if d <= 60:    return "29-60 days"
                return "60+ days"
            hdf = closed_df.copy()
            hdf["Bucket"] = hdf["Holding Days"].apply(_bucket)
            hold_groups = []
            for b in ["0-14 days", "15-28 days", "29-60 days", "60+ days", "Unknown"]:
                grp = hdf[hdf["Bucket"] == b]
                if grp.empty: continue
                n = len(grp); wins = (grp["Excess Return"] > 0).sum()
                hold_groups.append({
                    "Holding Period": b, "Positions": n, "Wins": int(wins),
                    "Win Rate": wins / n * 100 if n > 0 else None,
                    "Avg Excess Return": grp["Excess Return"].mean(),
                })
            hold_summary = pd.DataFrame(hold_groups)
            hold_col_defs = [
                ("Holding Period",    lambda r: f"<td style='padding:7px 10px;color:#7aaac8;white-space:nowrap'>{r['Holding Period']}</td>"),
                ("Positions",         lambda r: f"<td style='padding:7px 10px;text-align:center;color:#8a9bb5'>{r['Positions']}</td>"),
                ("Wins",              lambda r: f"<td style='padding:7px 10px;text-align:center;color:#2ecc71'>{r['Wins']}</td>"),
                ("Win Rate",          lambda r: f"<td style='padding:7px 10px;text-align:right;color:{'#2ecc71' if r['Win Rate'] and r['Win Rate'] >= 50 else '#e74c3c'}'>{fmt_pct(r['Win Rate'], 1)}</td>"),
                ("Avg Excess Return", lambda r: f"<td style='padding:7px 10px;text-align:right;color:{'#2ecc71' if pd.notna(r['Avg Excess Return']) and r['Avg Excess Return'] > 0 else '#e74c3c'}'>{fmt_pct(r['Avg Excess Return'])}</td>"),
            ]
            st.markdown(build_html_table(hold_summary, hold_col_defs, max_height="300px"), unsafe_allow_html=True)
        else:
            st.markdown("<p style='font-family:IBM Plex Mono,monospace;font-size:0.78rem;color:#4a6fa5'>No holding period data yet.</p>", unsafe_allow_html=True)

        # ── Win rate by cluster ───────────────────────────────────────────
        st.markdown("## Win Rate by Cluster")
        st.markdown(
            "<p style='font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:#4a6fa5;"
            "margin-bottom:1rem'>Clusters 0 and 1 have insignificant regressions and may show lower win rates over time.</p>",
            unsafe_allow_html=True,
        )
        def _fmt_cluster(val):
            try:    return str(int(float(val)))
            except: return str(val) if val is not None else "n/a"

        cluster_col = "Source Cluster" if "Source Cluster" in closed_df.columns else None
        if cluster_col and not closed_df[cluster_col].isna().all():
            clust_groups = []
            for clust, grp in closed_df.groupby(cluster_col):
                n = len(grp); wins = (grp["Excess Return"] > 0).sum()
                clust_groups.append({
                    "Cluster": clust, "Positions": n, "Wins": int(wins),
                    "Win Rate": wins / n * 100 if n > 0 else None,
                    "Avg Excess Return": grp["Excess Return"].mean(),
                })
            clust_df = pd.DataFrame(clust_groups).sort_values("Cluster")
            clust_col_defs = [
                ("Cluster",           lambda r: f"<td style='padding:7px 10px;color:#c9d1e0'>Cluster {_fmt_cluster(r['Cluster'])}</td>"),
                ("Positions",         lambda r: f"<td style='padding:7px 10px;text-align:center;color:#8a9bb5'>{r['Positions']}</td>"),
                ("Wins",              lambda r: f"<td style='padding:7px 10px;text-align:center;color:#2ecc71'>{r['Wins']}</td>"),
                ("Win Rate",          lambda r: f"<td style='padding:7px 10px;text-align:right;color:{'#2ecc71' if r['Win Rate'] and r['Win Rate'] >= 50 else '#e74c3c'}'>{fmt_pct(r['Win Rate'], 1)}</td>"),
                ("Avg Excess Return", lambda r: f"<td style='padding:7px 10px;text-align:right;color:{'#2ecc71' if pd.notna(r['Avg Excess Return']) and r['Avg Excess Return'] > 0 else '#e74c3c'}'>{fmt_pct(r['Avg Excess Return'])}</td>"),
            ]
            st.markdown(build_html_table(clust_df, clust_col_defs, max_height="300px"), unsafe_allow_html=True)
        else:
            st.markdown("<p style='font-family:IBM Plex Mono,monospace;font-size:0.78rem;color:#4a6fa5'>No cluster data yet.</p>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------

tab1, tab2 = st.tabs([
    "Main Tracker  (Open: cluster + index | Close: cluster + index)",
    "Cluster Tracker  (Open: cluster + index | Close: cluster only)",
])

with tab1:
    st.markdown(
        "<div style='background:#0d1220;border:1px solid #1e2a40;border-left:3px solid #4a7fa5;"
        "border-radius:4px;padding:8px 14px;margin-bottom:1rem;"
        "font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#8a9bb5'>"
        "Opens on <strong style='color:#2ecc71'>Strong Undervalued</strong> (cluster + index agree). "
        "Closes when either model changes signal.</div>",
        unsafe_allow_html=True,
    )
    render_tracker(open_pos, closed_pos, "main")

with tab2:
    st.markdown(
        "<div style='background:#0d1220;border:1px solid #1e2a40;border-left:3px solid #f39c12;"
        "border-radius:4px;padding:8px 14px;margin-bottom:1rem;"
        "font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#8a9bb5'>"
        "Opens on <strong style='color:#2ecc71'>Strong Undervalued</strong> (cluster + index agree). "
        "Closes only when the <strong style='color:#f39c12'>cluster model</strong> changes signal. "
        "Conflicting signals do not trigger a close.</div>",
        unsafe_allow_html=True,
    )
    render_tracker(open_pos_c, closed_pos_c, "cluster")