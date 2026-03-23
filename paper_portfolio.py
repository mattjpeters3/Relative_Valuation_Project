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
POSITIONS_PATH = os.path.join(RESULTS_DIR, "paper_positions.csv")

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

all_positions = load_positions()

if all_positions.empty:
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

open_pos   = all_positions[all_positions["Status"] == "Open"].copy()
closed_pos = all_positions[all_positions["Status"] == "Closed"].copy()

# Fetch live prices for open positions
live_prices = {}
if not open_pos.empty:
    live_prices = fetch_live_prices(open_pos["Ticker"].tolist())
    sp500_live  = live_prices.get("^GSPC")

    # Compute unrealised returns for open positions
    def unrealised(row):
        current = live_prices.get(row["Ticker"])
        entry   = row["Entry Price"]
        if current and entry and pd.notna(entry):
            return ((current / entry) - 1) * 100
        return None

    def unrealised_sp500(row):
        entry_sp = row["Entry SP500"]
        if sp500_live and entry_sp and pd.notna(entry_sp):
            return ((sp500_live / entry_sp) - 1) * 100
        return None

    open_pos["Current Price"]      = open_pos["Ticker"].map(live_prices)
    open_pos["Unrealised Return"]  = open_pos.apply(unrealised, axis=1)
    open_pos["SP500 Since Entry"]  = open_pos.apply(unrealised_sp500, axis=1)
    open_pos["Excess (Live)"]      = open_pos.apply(
        lambda r: r["Unrealised Return"] - r["SP500 Since Entry"]
        if pd.notna(r.get("Unrealised Return")) and pd.notna(r.get("SP500 Since Entry"))
        else None, axis=1
    )
    open_pos["Days Held"] = open_pos["Entry Date"].apply(
        lambda d: (datetime.date.today() - d.date()).days if pd.notna(d) else None
    )
    open_pos["Unrealised PnL"] = open_pos["Unrealised Return"].apply(
        lambda r: (NOTIONAL * r / 100) if pd.notna(r) else None
    )

# ---------------------------------------------------------------------------
# SUMMARY KPIs
# ---------------------------------------------------------------------------

st.markdown("## Summary")

n_open   = len(open_pos)
n_closed = len(closed_pos)
n_total  = n_open + n_closed

# Closed position stats
total_pnl       = closed_pos["Dollar PnL"].sum() if not closed_pos.empty else 0.0
avg_stock_ret   = closed_pos["Stock Return"].mean() if not closed_pos.empty else None
avg_sp500_ret   = closed_pos["SP500 Return"].mean() if not closed_pos.empty else None
avg_excess      = closed_pos["Excess Return"].mean() if not closed_pos.empty else None
win_rate        = (
    (closed_pos["Excess Return"] > 0).sum() / len(closed_pos) * 100
    if not closed_pos.empty else None
)

# Unrealised PnL from open positions
unreal_pnl = open_pos["Unrealised PnL"].sum() if not open_pos.empty and "Unrealised PnL" in open_pos.columns else 0.0

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Open Positions",    f"{n_open}")
c2.metric("Closed Positions",  f"{n_closed}")
c3.metric("Realised Profit/Loss",      fmt_dollar(total_pnl))
c4.metric("Unrealised Profit/Loss",    fmt_dollar(unreal_pnl))
c5.metric("Avg Excess Return", fmt_pct(avg_excess) if avg_excess is not None else "n/a")
c6.metric("Win Rate",          f"{win_rate:.0f}%" if win_rate is not None else "n/a",
          help="% of closed positions that outperformed the S&P 500")

# ---------------------------------------------------------------------------
# OPEN POSITIONS
# ---------------------------------------------------------------------------

st.markdown("## Open Positions")
st.markdown(
    "<p style='font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:#4a6fa5;"
    "margin-bottom:1rem'>Live prices fetched at page load. Returns are unrealised "
    "and will be finalised when the signal changes on the next pipeline run.</p>",
    unsafe_allow_html=True,
)

if open_pos.empty:
    st.markdown(
        "<p style='font-family:IBM Plex Mono,monospace;font-size:0.78rem;color:#4a6fa5'>"
        "No open positions at this time.</p>",
        unsafe_allow_html=True,
    )
else:
    def _open_unreal_ret(r):
        c = color_val(r.get("Unrealised Return"))
        v = fmt_pct(r.get("Unrealised Return"))
        return f"<td style='padding:7px 10px;text-align:right;color:{c}'>{v}</td>"
    def _open_excess(r):
        c = color_val(r.get("Excess (Live)"))
        v = fmt_pct(r.get("Excess (Live)"))
        return f"<td style='padding:7px 10px;text-align:right;color:{c}'>{v}</td>"
    def _open_pnl(r):
        c = color_val(r.get("Unrealised PnL"))
        v = fmt_dollar(r.get("Unrealised PnL"))
        return f"<td style='padding:7px 10px;text-align:right;color:{c}'>{v}</td>"
    def _open_entry_price(r):
        return f"<td style='padding:7px 10px;text-align:right;color:#8a9bb5'>${r['Entry Price']:.2f}</td>" if pd.notna(r.get('Entry Price')) else "<td style='padding:7px 10px;color:#4a6a8a'>n/a</td>"
    def _open_curr_price(r):
        return f"<td style='padding:7px 10px;text-align:right;color:#c9d1e0'>${r['Current Price']:.2f}</td>" if pd.notna(r.get('Current Price')) else "<td style='padding:7px 10px;color:#4a6a8a'>n/a</td>"

    open_col_defs = [
        ("Ticker",         lambda r: f"<td style='padding:7px 10px;color:#e8edf5;font-weight:600'>{r['Ticker']}</td>"),
        ("Sector",         lambda r: f"<td style='padding:7px 10px;color:#7aaac8'>{r.get('Sector', 'n/a')}</td>"),
        ("Entry Date",     lambda r: f"<td style='padding:7px 10px;color:#8a9bb5;white-space:nowrap'>{r['Entry Date'].strftime('%b %d, %Y') if pd.notna(r['Entry Date']) else 'n/a'}</td>"),
        ("Days Held",      lambda r: f"<td style='padding:7px 10px;text-align:center;color:#8a9bb5'>{int(r['Days Held']) if pd.notna(r.get('Days Held')) else 'n/a'}</td>"),
        ("Entry Price",    _open_entry_price),
        ("Current Price",  _open_curr_price),
        ("Unrealised Return", _open_unreal_ret),
        ("S&P 500 Return",    lambda r: f"<td style='padding:7px 10px;text-align:right;color:#8a9bb5'>{fmt_pct(r.get('SP500 Since Entry'))}</td>"),
        ("Excess Return",  _open_excess),
        ("Unrealised Profit/Loss", _open_pnl),
    ]
    st.markdown(build_html_table(open_pos, open_col_defs), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# CLOSED POSITIONS
# ---------------------------------------------------------------------------

st.markdown("## Closed Positions")

if closed_pos.empty:
    st.markdown(
        "<p style='font-family:IBM Plex Mono,monospace;font-size:0.78rem;color:#4a6fa5'>"
        "No closed positions yet. Positions close when the signal changes on a subsequent pipeline run.</p>",
        unsafe_allow_html=True,
    )
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
        ("Ticker",        lambda r: f"<td style='padding:7px 10px;color:#e8edf5;font-weight:600'>{r['Ticker']}</td>"),
        ("Sector",        lambda r: f"<td style='padding:7px 10px;color:#7aaac8'>{r.get('Sector', 'n/a')}</td>"),
        ("Entry Date",    lambda r: f"<td style='padding:7px 10px;color:#8a9bb5;white-space:nowrap'>{r['Entry Date'].strftime('%b %d, %Y') if pd.notna(r['Entry Date']) else 'n/a'}</td>"),
        ("Exit Date",     lambda r: f"<td style='padding:7px 10px;color:#8a9bb5;white-space:nowrap'>{r['Exit Date'].strftime('%b %d, %Y') if pd.notna(r['Exit Date']) else 'n/a'}</td>"),
        ("Days Held",     lambda r: f"<td style='padding:7px 10px;text-align:center;color:#8a9bb5'>{int(r['Holding Days']) if pd.notna(r.get('Holding Days')) else 'n/a'}</td>"),
        ("Exit Signal",   lambda r: f"<td style='padding:7px 10px;color:#f39c12;white-space:nowrap'>{r.get('Exit Signal', 'n/a')}</td>"),
        ("Stock Return",  _cl_stock_ret),
        ("S&P 500 Return",   lambda r: f"<td style='padding:7px 10px;text-align:right;color:#8a9bb5'>{fmt_pct(r.get('SP500 Return'))}</td>"),
        ("Excess Return", _cl_excess),
        ("Profit/Loss",    _cl_pnl),
    ]
    st.markdown(build_html_table(closed_pos.sort_values("Exit Date", ascending=False), closed_col_defs), unsafe_allow_html=True)

    # ── Cumulative PnL chart ──────────────────────────────────────────────
    st.markdown("## Cumulative Performance vs S&P 500")
    st.markdown(
        "<p style='font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:#4a6fa5;"
        "margin-bottom:1rem'>Cumulative dollar PnL across all closed positions "
        "vs an equivalent $1,000 investment in the S&P 500 per position over each holding period.</p>",
        unsafe_allow_html=True,
    )

    chart_df = closed_pos.dropna(subset=["Exit Date", "Dollar PnL", "SP500 Return"]).copy()
    chart_df = chart_df.sort_values("Exit Date")
    chart_df["Cumulative PnL"]       = chart_df["Dollar PnL"].cumsum()
    chart_df["SP500 Dollar PnL"]     = NOTIONAL * chart_df["SP500 Return"] / 100
    chart_df["Cumulative SP500 PnL"] = chart_df["SP500 Dollar PnL"].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_df["Exit Date"],
        y=chart_df["Cumulative PnL"],
        name="Signal Strategy",
        mode="lines+markers",
        line=dict(color="#2ecc71", width=2),
        marker=dict(size=7),
    ))
    fig.add_trace(go.Scatter(
        x=chart_df["Exit Date"],
        y=chart_df["Cumulative SP500 PnL"],
        name="S&P 500 Benchmark",
        mode="lines+markers",
        line=dict(color="#4a7fa5", width=2, dash="dash"),
        marker=dict(size=7),
    ))
    fig.add_hline(y=0, line_color="#1e2a40", line_width=1)
    fig.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        font=dict(family="IBM Plex Mono", color="#8a9bb5", size=11),
        margin=dict(l=20, r=20, t=20, b=20),
        height=340,
        legend=dict(bgcolor="#0f1628", bordercolor="#1e2a40", borderwidth=1,
                    font=dict(size=10), orientation="h", y=-0.2),
        xaxis=dict(showgrid=True, gridcolor="#141e30", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#141e30", zeroline=False,
                   title="Cumulative PnL ($)", tickprefix="$"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Per-position excess return bar chart ─────────────────────────────
    st.markdown("## Excess Return Per Position")
    bar_df = closed_pos.dropna(subset=["Excess Return"]).copy()
    bar_df = bar_df.sort_values("Excess Return", ascending=True)
    bar_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in bar_df["Excess Return"]]

    fig2 = go.Figure(go.Bar(
        x=bar_df["Ticker"],
        y=bar_df["Excess Return"],
        marker_color=bar_colors,
        text=[fmt_pct(v) for v in bar_df["Excess Return"]],
        textposition="outside",
        textfont=dict(size=10, color="#8a9bb5"),
    ))
    fig2.add_hline(y=0, line_color="#4a7fa5", line_width=1)
    fig2.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        font=dict(family="IBM Plex Mono", color="#8a9bb5", size=11),
        margin=dict(l=20, r=20, t=20, b=20),
        height=320,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#141e30", zeroline=False,
                   title="Excess Return (%)", ticksuffix="%"),
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Stats breakdown ───────────────────────────────────────────────────
    st.markdown("## Closed Position Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Stock Return",  fmt_pct(avg_stock_ret))
    c2.metric("Avg S&P 500 Return", fmt_pct(avg_sp500_ret))
    c3.metric("Avg Excess Return",  fmt_pct(avg_excess),
              delta=f"{'Outperformed' if avg_excess and avg_excess > 0 else 'Underperformed'}")
    avg_days = closed_pos["Holding Days"].mean()
    c4.metric("Avg Holding Period", f"{avg_days:.0f} days" if pd.notna(avg_days) else "n/a")

    # ── Win rate by exit signal ───────────────────────────────────────────
    st.markdown("## Win Rate by Exit Signal")
    st.markdown(
        "<p style='font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:#4a6fa5;"
        "margin-bottom:1rem'>Win rate and average excess return broken down by the signal "
        "that triggered the position close. Populates meaningfully with 50+ closed positions.</p>",
        unsafe_allow_html=True,
    )

    if "Exit Signal" in closed_pos.columns and not closed_pos["Exit Signal"].isna().all():
        signal_groups = []
        for sig, grp in closed_pos.groupby("Exit Signal"):
            n = len(grp)
            wins = (grp["Excess Return"] > 0).sum()
            wr = wins / n * 100 if n > 0 else None
            avg_exc = grp["Excess Return"].mean()
            avg_hold = grp["Holding Days"].mean() if "Holding Days" in grp.columns else None
            signal_groups.append({
                "Exit Signal": sig,
                "Positions": n,
                "Wins": int(wins),
                "Win Rate": wr,
                "Avg Excess Return": avg_exc,
                "Avg Holding Days": avg_hold,
            })
        sig_df = pd.DataFrame(signal_groups).sort_values("Positions", ascending=False)

        sig_col_defs = [
            ("Exit Signal",       lambda r: f"<td style='padding:7px 10px;color:#f39c12;white-space:nowrap'>{r['Exit Signal']}</td>"),
            ("Positions",         lambda r: f"<td style='padding:7px 10px;text-align:center;color:#8a9bb5'>{r['Positions']}</td>"),
            ("Wins",              lambda r: f"<td style='padding:7px 10px;text-align:center;color:#2ecc71'>{r['Wins']}</td>"),
            ("Win Rate",          lambda r: f"<td style='padding:7px 10px;text-align:right;color:{'#2ecc71' if r['Win Rate'] and r['Win Rate'] >= 50 else '#e74c3c'}'>{fmt_pct(r['Win Rate'], 1)}</td>"),
            ("Avg Excess Return", lambda r: f"<td style='padding:7px 10px;text-align:right;color:{'#2ecc71' if pd.notna(r['Avg Excess Return']) and r['Avg Excess Return'] > 0 else '#e74c3c'}'>{fmt_pct(r['Avg Excess Return'])}</td>"),
            ("Avg Holding Days",  lambda r: f"<td style='padding:7px 10px;text-align:center;color:#8a9bb5'>{r['Avg Holding Days']:.0f}d" if pd.notna(r.get('Avg Holding Days')) else "<td style='padding:7px 10px;color:#4a6a8a'>n/a</td>"),
        ]
        st.markdown(build_html_table(sig_df, sig_col_defs, max_height="300px"), unsafe_allow_html=True)
    else:
        st.markdown(
            "<p style='font-family:IBM Plex Mono,monospace;font-size:0.78rem;color:#4a6fa5'>"
            "No exit signal data available yet.</p>",
            unsafe_allow_html=True,
        )

    # ── Win rate by holding period ────────────────────────────────────────
    st.markdown("## Win Rate by Holding Period")
    st.markdown(
        "<p style='font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:#4a6fa5;"
        "margin-bottom:1rem'>Positions bucketed by how long they were held before closing.</p>",
        unsafe_allow_html=True,
    )

    if "Holding Days" in closed_pos.columns and not closed_pos["Holding Days"].isna().all():
        def holding_bucket(days):
            if pd.isna(days):
                return "Unknown"
            if days <= 14:
                return "0-14 days"
            elif days <= 28:
                return "15-28 days"
            elif days <= 60:
                return "29-60 days"
            else:
                return "60+ days"

        hold_df = closed_pos.copy()
        hold_df["Holding Bucket"] = hold_df["Holding Days"].apply(holding_bucket)
        bucket_order = ["0-14 days", "15-28 days", "29-60 days", "60+ days", "Unknown"]

        hold_groups = []
        for bucket in bucket_order:
            grp = hold_df[hold_df["Holding Bucket"] == bucket]
            if grp.empty:
                continue
            n = len(grp)
            wins = (grp["Excess Return"] > 0).sum()
            wr = wins / n * 100 if n > 0 else None
            avg_exc = grp["Excess Return"].mean()
            hold_groups.append({
                "Holding Period": bucket,
                "Positions": n,
                "Wins": int(wins),
                "Win Rate": wr,
                "Avg Excess Return": avg_exc,
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
        st.markdown(
            "<p style='font-family:IBM Plex Mono,monospace;font-size:0.78rem;color:#4a6fa5'>"
            "No holding period data available yet.</p>",
            unsafe_allow_html=True,
        )

    # ── Win rate by cluster ───────────────────────────────────────────────
    st.markdown("## Win Rate by Cluster")
    st.markdown(
        "<p style='font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:#4a6fa5;"
        "margin-bottom:1rem'>Which clusters are generating reliable signals. "
        "Clusters 0 and 1 have insignificant regressions and may show lower win rates over time.</p>",
        unsafe_allow_html=True,
    )

    cluster_col = "Source Cluster" if "Source Cluster" in closed_pos.columns else None
    if cluster_col and not closed_pos[cluster_col].isna().all():
        cluster_groups = []
        for clust, grp in closed_pos.groupby(cluster_col):
            n = len(grp)
            wins = (grp["Excess Return"] > 0).sum()
            wr = wins / n * 100 if n > 0 else None
            avg_exc = grp["Excess Return"].mean()
            cluster_groups.append({
                "Cluster": clust,
                "Positions": n,
                "Wins": int(wins),
                "Win Rate": wr,
                "Avg Excess Return": avg_exc,
            })
        clust_df = pd.DataFrame(cluster_groups).sort_values("Cluster")

        clust_col_defs = [
            ("Cluster",           lambda r: f"<td style='padding:7px 10px;color:#c9d1e0'>Cluster {int(float(r['Cluster'])) if pd.notna(r['Cluster']) else 'n/a'}</td>"),
            ("Positions",         lambda r: f"<td style='padding:7px 10px;text-align:center;color:#8a9bb5'>{r['Positions']}</td>"),
            ("Wins",              lambda r: f"<td style='padding:7px 10px;text-align:center;color:#2ecc71'>{r['Wins']}</td>"),
            ("Win Rate",          lambda r: f"<td style='padding:7px 10px;text-align:right;color:{'#2ecc71' if r['Win Rate'] and r['Win Rate'] >= 50 else '#e74c3c'}'>{fmt_pct(r['Win Rate'], 1)}</td>"),
            ("Avg Excess Return", lambda r: f"<td style='padding:7px 10px;text-align:right;color:{'#2ecc71' if pd.notna(r['Avg Excess Return']) and r['Avg Excess Return'] > 0 else '#e74c3c'}'>{fmt_pct(r['Avg Excess Return'])}</td>"),
        ]
        st.markdown(build_html_table(clust_df, clust_col_defs, max_height="300px"), unsafe_allow_html=True)
    else:
        st.markdown(
            "<p style='font-family:IBM Plex Mono,monospace;font-size:0.78rem;color:#4a6fa5'>"
            "No cluster data available yet.</p>",
            unsafe_allow_html=True,
        )