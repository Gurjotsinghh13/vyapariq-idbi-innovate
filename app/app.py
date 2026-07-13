"""VyaparIQ Streamlit dashboard."""

from html import escape
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model.business_analytics import build_benchmark, build_timeline
from model.recommendation_engine import build_recommendations, evaluate_loan_products
from model.score_engine import FEATURE_LABELS, HealthScoreEngine


st.set_page_config(page_title="VyaparIQ | MSME Financial Health", layout="wide")
st.markdown("""
<style>
    :root {
        --primary:#0F4C81;
        --secondary:#1E3A8A;
        --success:#16A34A;
        --warning:#D97706;
        --danger:#DC2626;
        --background:#F8FAFC;
        --surface:#FFFFFF;
        --border:#E5E7EB;
        --border-strong:#CBD5E1;
        --text-primary:#111827;
        --text-secondary:#6B7280;
        --text-muted:#4B5563;
        --shadow-sm:0 1px 2px rgba(15,23,42,.06);
        --shadow-md:0 10px 24px rgba(15,23,42,.08);
        --radius-lg:18px;
        --radius-md:14px;
        --radius-sm:10px;
    }

    html, body, [class*="css"] {
        color:var(--text-primary);
        font-family:"Inter", "Segoe UI", "Roboto", "Helvetica Neue", Arial, sans-serif;
    }

    .stApp {
        background:radial-gradient(circle at top left, rgba(15,76,129,.08), transparent 34rem), linear-gradient(180deg, #F8FAFC 0%, #EEF4FA 100%);
        color:var(--text-primary);
    }

    .block-container {
        max-width:1480px;
        padding:2rem 2rem 3.5rem;
    }

    h1, h2, h3 {color:var(--text-primary); letter-spacing:-.025em;}
    h1 {font-size:2.45rem !important; line-height:1.08 !important; font-weight:760 !important; margin-bottom:.35rem !important;}
    h2 {font-size:1.55rem !important; font-weight:720 !important;}
    h3 {font-size:1.18rem !important; font-weight:700 !important;}
    p, li, label, span {color:var(--text-primary);}

    [data-testid="stCaptionContainer"], .small-note, .card-caption {
        color:var(--text-muted) !important;
        font-size:.88rem;
        line-height:1.55;
    }

    [data-testid="stSidebar"] {
        background:#FFFFFF;
        border-right:1px solid var(--border);
        box-shadow:6px 0 24px rgba(15,23,42,.04);
    }

    [data-testid="stSidebar"] > div:first-child {padding:1.45rem 1.15rem 2rem;}
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {color:var(--text-primary) !important;}
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color:var(--text-muted) !important;}
    [data-testid="stSidebar"] .stButton {margin-bottom:.45rem;}

    .sidebar-brand {
        border:1px solid var(--border);
        border-radius:var(--radius-lg);
        background:linear-gradient(135deg, #0F4C81 0%, #1E3A8A 100%);
        padding:1rem 1.05rem;
        margin-bottom:1.1rem;
        box-shadow:var(--shadow-md);
    }

    .sidebar-brand-title {color:#FFFFFF !important; font-size:1.15rem; font-weight:760; letter-spacing:-.02em; margin:0;}
    .sidebar-brand-subtitle {color:#DBEAFE !important; font-size:.82rem; line-height:1.45; margin:.35rem 0 0;}
    .sidebar-section-title {color:var(--secondary) !important; font-size:.78rem; font-weight:760; letter-spacing:.08em; text-transform:uppercase; margin:1.15rem 0 .55rem;}
    .sidebar-panel {border:1px solid var(--border); background:#F8FAFC; border-radius:var(--radius-md); padding:.95rem; margin:.35rem 0 1rem;}

    .stButton > button,
    [data-testid="stDownloadButton"] > button {
        width:100%;
        min-height:2.65rem;
        border-radius:999px !important;
        border:1px solid var(--border-strong) !important;
        background:#FFFFFF !important;
        color:var(--text-primary) !important;
        font-weight:700 !important;
        letter-spacing:-.01em;
        box-shadow:var(--shadow-sm);
        transition:all .16s ease-in-out;
    }

    .stButton > button:hover,
    [data-testid="stDownloadButton"] > button:hover {
        border-color:var(--primary) !important;
        color:var(--primary) !important;
        transform:translateY(-1px);
        box-shadow:0 8px 18px rgba(15,76,129,.12);
    }

    div[role="radiogroup"] {gap:.35rem;}
    div[role="radiogroup"] label,
    [data-baseweb="select"] > div,
    [data-testid="stTextInput"] input {
        border-radius:var(--radius-sm) !important;
        border-color:var(--border-strong) !important;
        background:#FFFFFF !important;
    }

    [data-testid="stSlider"] {padding:.2rem 0 .7rem;}

    .hero {
        display:grid;
        grid-template-columns:minmax(0, 1.65fr) minmax(300px, .9fr);
        align-items:stretch;
        gap:1.1rem;
        border:1px solid var(--border);
        background:linear-gradient(135deg, #FFFFFF 0%, #F8FBFF 58%, #EEF6FF 100%);
        border-radius:20px;
        padding:1.15rem;
        margin-bottom:1.1rem;
        box-shadow:var(--shadow-md);
    }

    .hero-main {
        padding:.35rem .35rem .35rem .25rem;
        align-self:center;
    }

    .eyebrow {color:var(--primary) !important; font-size:.76rem; font-weight:800; letter-spacing:.11em; text-transform:uppercase; margin-bottom:.32rem;}
    .hero-title {color:var(--text-primary) !important; font-size:2.15rem; line-height:1.06; font-weight:800; letter-spacing:-.045em; margin:0;}
    .hero-subtitle {color:var(--text-muted) !important; max-width:780px; font-size:.98rem; line-height:1.5; margin:.48rem 0 0;}
    .hero-value {color:var(--text-primary) !important; max-width:820px; font-size:.92rem; line-height:1.5; font-weight:650; margin:.72rem 0 0;}
    .hero-info-panel {
        background:#FFFFFF;
        border:1px solid var(--border);
        border-radius:16px;
        padding:.95rem;
        box-shadow:var(--shadow-sm);
    }
    .hero-info-title {color:var(--text-primary) !important; font-size:.86rem; font-weight:780; letter-spacing:.02em; margin:0 0 .65rem;}
    .hero-info-grid {display:grid; gap:.58rem;}
    .hero-info-row {
        display:grid;
        grid-template-columns:112px 1fr;
        gap:.75rem;
        align-items:start;
        padding:.58rem 0;
        border-top:1px solid var(--border);
    }
    .hero-info-row:first-child {border-top:0; padding-top:0;}
    .hero-info-label {color:var(--text-secondary) !important; font-size:.72rem; font-weight:780; letter-spacing:.055em; text-transform:uppercase;}
    .hero-info-value {color:var(--text-primary) !important; font-size:.86rem; font-weight:700; line-height:1.35;}

    .summary-card, .panel-card, .health-card, .list-card {
        background:var(--surface);
        border:1px solid var(--border);
        border-radius:var(--radius-lg);
        box-shadow:var(--shadow-sm);
    }

    .summary-card {min-height:138px; padding:1.05rem 1.1rem; border-top:4px solid var(--primary); overflow-wrap:anywhere;}
    .summary-label {color:var(--text-secondary) !important; font-size:.76rem; font-weight:780; line-height:1.25; letter-spacing:.055em; text-transform:uppercase; margin-bottom:.8rem;}
    .summary-value {color:var(--text-primary) !important; font-size:1.7rem; line-height:1.12; font-weight:780; letter-spacing:-.035em;}
    .summary-help {color:var(--text-muted) !important; font-size:.82rem; line-height:1.35; margin-top:.45rem;}
    .panel-card {padding:1.15rem 1.2rem; margin-bottom:1rem;}
    .section-title {color:var(--text-primary) !important; font-size:1.22rem; font-weight:740; letter-spacing:-.025em; margin:0 0 .35rem;}
    .section-subtitle {color:var(--text-muted) !important; font-size:.9rem; line-height:1.55; margin:0 0 1rem;}

    .risk-badge {display:inline-flex; align-items:center; justify-content:center; border-radius:999px; color:#FFFFFF !important; padding:.55rem .9rem; font-size:.78rem; font-weight:780; letter-spacing:.055em; text-transform:uppercase; box-shadow:var(--shadow-sm);}
    .risk-low {background:var(--success);}
    .risk-medium {background:var(--warning);}
    .risk-high {background:var(--danger);}

    .signal-grid {display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:.8rem; margin-top:1rem;}
    .signal-item {background:#F8FAFC; border:1px solid var(--border); border-radius:var(--radius-md); padding:.85rem;}
    .signal-label {color:var(--text-secondary) !important; font-size:.74rem; font-weight:760; text-transform:uppercase; letter-spacing:.045em;}
    .signal-value {color:var(--text-primary) !important; font-size:1rem; font-weight:720; margin-top:.3rem;}

    .health-card {padding:1.35rem; border-top:5px solid var(--primary); box-shadow:var(--shadow-md);}
    .health-card-header {display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; border-bottom:1px solid var(--border); padding-bottom:1rem; margin-bottom:1rem;}
    .health-card-title {color:var(--text-primary) !important; font-size:1.45rem; font-weight:780; letter-spacing:-.035em; margin:0;}
    .health-card-meta {color:var(--text-muted) !important; font-size:.84rem; line-height:1.45; margin-top:.25rem;}

    .list-card {padding:1rem; min-height:100%; margin-bottom:1rem;}
    .list-card-title {color:var(--text-primary) !important; font-size:.98rem; font-weight:750; margin-bottom:.75rem;}
    .list-item {border-left:3px solid var(--primary); background:#F8FAFC; color:var(--text-primary) !important; border-radius:0 var(--radius-sm) var(--radius-sm) 0; padding:.72rem .8rem; margin-bottom:.55rem; line-height:1.45;}
    .list-item.success {border-left-color:var(--success);}
    .list-item.warning {border-left-color:var(--warning);}
    .list-item.danger {border-left-color:var(--danger);}

    [data-baseweb="tab-list"] {gap:.45rem; border-bottom:1px solid var(--border); margin-top:.5rem;}
    [data-baseweb="tab"] {background:#FFFFFF; border:1px solid var(--border); border-bottom:0; border-radius:14px 14px 0 0; padding:.85rem 1.15rem; color:var(--text-secondary) !important; font-weight:740;}
    [aria-selected="true"][data-baseweb="tab"] {color:var(--primary) !important; border-top:3px solid var(--primary); background:#FFFFFF;}

    div[data-testid="stExpander"] {background:#FFFFFF; border:1px solid var(--border); border-radius:var(--radius-lg); box-shadow:var(--shadow-sm); overflow:hidden;}
    div[data-testid="stExpander"] details summary p {color:var(--text-primary) !important; font-weight:720;}
    div[data-testid="stAlert"] {border-radius:var(--radius-md); border:1px solid var(--border);}
    [data-testid="stDataFrame"] {border:1px solid var(--border); border-radius:var(--radius-md); overflow:hidden;}
    hr {border-color:var(--border) !important; margin:1.5rem 0 !important;}

    @media (max-width: 1100px) {
        .hero {grid-template-columns:1fr;}
        .signal-grid {grid-template-columns:repeat(2, minmax(0,1fr));}
    }

    @media (max-width: 760px) {
        .block-container {padding:1.2rem .85rem 2.2rem;}
        .hero {padding:1.1rem; border-radius:18px;}
        .hero-title {font-size:1.75rem;}
        .hero-info-row {grid-template-columns:1fr; gap:.2rem;}
        .summary-card {min-height:auto;}
        .signal-grid {grid-template-columns:1fr;}
        .health-card-header {flex-direction:column;}
        [data-baseweb="tab-list"] {overflow-x:auto;}
        [data-baseweb="tab"] {white-space:nowrap;}
        [data-testid="column"] {width:100% !important; flex:1 1 100% !important;}
    }
</style>
""", unsafe_allow_html=True)

DATA_PATH = PROJECT_ROOT / "data" / "vendors_synthetic.csv"

CHART_COLORS = {
    "primary": "#0F4C81",
    "secondary": "#1E3A8A",
    "success": "#16A34A",
    "warning": "#D97706",
    "danger": "#DC2626",
    "grid": "#E5E7EB",
    "text": "#111827",
    "muted": "#6B7280",
}


def risk_class(risk_category: str) -> str:
    return {
        "Low Risk": "risk-low",
        "Medium Risk": "risk-medium",
        "High Risk": "risk-high",
    }[risk_category]


def render_section(title: str, subtitle: str | None = None) -> None:
    subtitle_html = f'<p class="section-subtitle">{escape(subtitle)}</p>' if subtitle else ""
    st.markdown(
        f'<div class="section-title">{escape(title)}</div>{subtitle_html}',
        unsafe_allow_html=True,
    )


def render_summary_card(label: str, value: str, helper: str) -> None:
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-label">{escape(label)}</div>
            <div class="summary-value">{escape(value)}</div>
            <div class="summary-help">{escape(helper)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_list_card(title: str, items: list[str], tone: str = "primary") -> None:
    content = "".join(
        f'<div class="list-item {tone}">{escape(item)}</div>'
        for item in items
    )
    st.markdown(
        f"""
        <div class="list-card">
            <div class="list-card-title">{escape(title)}</div>
            {content}
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_currency(value: float) -> str:
    return f"₹{value:,.0f}"


@st.cache_resource(show_spinner="Loading financial health model...")
def load_engine() -> HealthScoreEngine:
    return HealthScoreEngine()


@st.cache_data(show_spinner=False)
def load_vendor_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


try:
    engine = load_engine()
    df = load_vendor_data(DATA_PATH)
except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
    st.error(f"VyaparIQ could not load its model or vendor data: {exc}")
    st.stop()

FEATURES = engine.features

DEMO_PROFILES = {
    "Healthy MSME": {
        "business_name": "Shree Ganesh Kirana",
        "signals": {
            "avg_monthly_upi_inflow": 82_000, "upi_inflow_trend_pct": 1.2, "upi_inflow_volatility": 0.09,
            "upi_txn_count_monthly": 340, "avg_ticket_size": 241, "gig_order_volume_monthly": 270,
            "gig_order_trend_pct": 1.5, "business_vintage_months": 32, "bank_account_tenure_months": 30,
            "cash_recycling_score": 0.07, "existing_credit_lines": 0, "seasonal_dependency_index": 0.22,
            "gst_registered": 0, "epfo_active": 0,
        },
    },
    "High Risk MSME": {
        "business_name": "Metro Mobile & Electronics",
        "signals": {
            "avg_monthly_upi_inflow": 82_000, "upi_inflow_trend_pct": -4.5, "upi_inflow_volatility": 0.38,
            "upi_txn_count_monthly": 340, "avg_ticket_size": 241, "gig_order_volume_monthly": 270,
            "gig_order_trend_pct": -5.0, "business_vintage_months": 32, "bank_account_tenure_months": 30,
            "cash_recycling_score": 0.52, "existing_credit_lines": 2, "seasonal_dependency_index": 0.22,
            "gst_registered": 0, "epfo_active": 0,
        },
    },
}
SIMULATOR_KEYS = ("sim_upi_growth", "sim_volatility", "sim_invoice_delay", "sim_concentration", "sim_revenue_growth")


def clear_simulator() -> None:
    for key in SIMULATOR_KEYS:
        st.session_state.pop(key, None)


def load_demo(profile: str) -> None:
    st.session_state["assessment_source"] = "Contradiction demo"
    st.session_state["demo_vendor"] = profile
    st.session_state.pop("business_name", None)
    clear_simulator()


st.session_state.setdefault("assessment_source", "Contradiction demo")
st.session_state.setdefault("demo_vendor", "Healthy MSME")


def manual_input() -> dict:
    st.sidebar.markdown('<div class="sidebar-section-title">Business profile inputs</div>', unsafe_allow_html=True)
    values = {
        "avg_monthly_upi_inflow": st.sidebar.slider("Average monthly UPI inflow (₹)", 20_000, 250_000, 82_000, 1_000),
        "upi_inflow_trend_pct": st.sidebar.slider("UPI inflow trend (%/month)", -10.0, 10.0, 1.0, 0.1),
        "upi_inflow_volatility": st.sidebar.slider("UPI inflow volatility", 0.02, 0.9, 0.10, 0.01),
        "upi_txn_count_monthly": st.sidebar.slider("Monthly UPI transactions", 50, 900, 340, 10),
        "gig_order_volume_monthly": st.sidebar.slider("Monthly delivery orders", 20, 700, 260, 10),
        "gig_order_trend_pct": st.sidebar.slider("Revenue growth (%/month)", -10.0, 10.0, 1.0, 0.1),
        "business_vintage_months": st.sidebar.slider("Business vintage (months)", 2, 96, 30),
        "bank_account_tenure_months": st.sidebar.slider("Bank account tenure (months)", 1, 96, 28),
        "cash_recycling_score": st.sidebar.slider("Cash recycling indicator", 0.0, 0.98, 0.08, 0.01),
        "existing_credit_lines": st.sidebar.slider("Existing credit lines", 0, 6, 0),
        "seasonal_dependency_index": st.sidebar.slider("Customer concentration proxy", 0.0, 0.95, 0.25, 0.01),
        "gst_registered": int(st.sidebar.checkbox("GST Registered")),
        "epfo_active": int(st.sidebar.checkbox("EPFO Active")),
    }
    values["avg_ticket_size"] = values["avg_monthly_upi_inflow"] / max(values["upi_txn_count_monthly"], 1)
    return values


def selected_vendor() -> tuple[dict, str]:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <p class="sidebar-brand-title">VyaparIQ</p>
            <p class="sidebar-brand-subtitle">MSME financial-health intelligence for credit teams.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown('<div class="sidebar-section-title">Assessment workspace</div>', unsafe_allow_html=True)
    st.sidebar.markdown("**Demo scenarios**")
    healthy_col, risk_col = st.sidebar.columns(2)
    healthy_col.button("Healthy MSME", use_container_width=True, type="primary", on_click=load_demo, args=("Healthy MSME",))
    risk_col.button("High Risk MSME", use_container_width=True, on_click=load_demo, args=("High Risk MSME",))
    st.sidebar.button("Reset scenario", use_container_width=True, on_click=clear_simulator)
    st.sidebar.caption("Start with either profile, adjust the scenario, then reset to return to its baseline.")
    st.sidebar.divider()
    st.sidebar.markdown('<div class="sidebar-section-title">Assessment source</div>', unsafe_allow_html=True)

    mode = st.sidebar.radio(
        "Select input method", ["Contradiction demo", "Synthetic dataset", "Manual input"], key="assessment_source"
    )
    if mode == "Manual input":
        return manual_input(), "Manual assessment"
    if mode == "Synthetic dataset":
        vendor_id = st.sidebar.selectbox("Vendor ID", df["vendor_id"].tolist())
        row = df.loc[df["vendor_id"] == vendor_id].iloc[0]
        return {feature: row[feature] for feature in FEATURES}, str(vendor_id)

    choice = st.sidebar.selectbox("Demo business", list(DEMO_PROFILES), key="demo_vendor")
    profile = DEMO_PROFILES[choice]
    return dict(profile["signals"]), profile["business_name"]


def score_gauge(score: float) -> go.Figure:
    figure = go.Figure(go.Indicator(mode="gauge+number", value=score, gauge={
        "axis": {"range": [0, 100], "tickcolor": CHART_COLORS["muted"]},
        "bar": {"color": CHART_COLORS["primary"]},
        "bgcolor": "#FFFFFF",
        "borderwidth": 0,
        "steps": [{"range": [0, 45], "color": "#FEE2E2"}, {"range": [45, 70], "color": "#FEF3C7"},
                  {"range": [70, 100], "color": "#DCFCE7"}],
    }))
    figure.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=24, b=20),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color=CHART_COLORS["text"], family="Inter, Segoe UI, sans-serif"),
    )
    return figure


def style_chart(figure: go.Figure, height: int = 360, show_legend: bool = False) -> go.Figure:
    figure.update_layout(
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color=CHART_COLORS["text"], family="Inter, Segoe UI, sans-serif", size=12),
        margin=dict(l=18, r=18, t=46, b=24),
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    figure.update_xaxes(
        showgrid=True,
        gridcolor=CHART_COLORS["grid"],
        zerolinecolor=CHART_COLORS["grid"],
        linecolor=CHART_COLORS["grid"],
        tickfont=dict(color=CHART_COLORS["muted"]),
        title_font=dict(color=CHART_COLORS["text"]),
    )
    figure.update_yaxes(
        showgrid=True,
        gridcolor=CHART_COLORS["grid"],
        zerolinecolor=CHART_COLORS["grid"],
        linecolor=CHART_COLORS["grid"],
        tickfont=dict(color=CHART_COLORS["muted"]),
        title_font=dict(color=CHART_COLORS["text"]),
    )
    return figure


def apply_simulator(base: dict) -> dict:
    simulated = dict(base)
    with st.expander("Credit Scenario Analysis", expanded=True):
        st.caption("Model the impact of controllable operating changes on financial health and risk classification.")
        c1, c2, c3, c4, c5 = st.columns(5)
        simulated["upi_inflow_trend_pct"] = c1.slider("Monthly UPI growth (%)", -15.0, 10.0, float(base["upi_inflow_trend_pct"]), 0.1, key="sim_upi_growth")
        simulated["upi_inflow_volatility"] = c2.slider("Cash-flow volatility", 0.02, 0.90, float(base["upi_inflow_volatility"]), 0.01, key="sim_volatility")
        invoice_days = c3.slider("Invoice delay (days)", 0, 45, int(round(float(base["cash_recycling_score"]) * 45)), key="sim_invoice_delay")
        concentration = c4.slider("Customer concentration (%)", 0, 95, int(round(float(base["seasonal_dependency_index"]) * 100)), key="sim_concentration")
        simulated["gig_order_trend_pct"] = c5.slider("Revenue growth (%)", -15.0, 10.0, float(base["gig_order_trend_pct"]), 0.1, key="sim_revenue_growth")
        simulated["cash_recycling_score"] = invoice_days / 45
        simulated["seasonal_dependency_index"] = concentration / 100
        st.markdown('<div class="small-note">Invoice delay and concentration are mapped to the trained model’s working-capital stress and dependency signals.</div>', unsafe_allow_html=True)
    return simulated


def report_text(name: str, vendor: dict, result: dict, recommendations: dict,
                products: list[dict], percentile: int) -> str:
    eligible = [item["product"] for item in products if item["status"] == "Eligible"]
    lines = [
        "=" * 60, "VyaparIQ — FINANCIAL HEALTH CARD", "=" * 60,
        f"Business: {name}", f"Assessment timestamp: {datetime.now().astimezone().isoformat(timespec='minutes')}",
        f"Financial Health Score: {result['health_score']}/100", f"Risk Level: {result['risk_category']}",
        f"Industry Percentile: {percentile}th", "", "BUSINESS PROFILE",
        f"Monthly UPI inflow: ₹{float(vendor['avg_monthly_upi_inflow']):,.0f}",
        f"Business vintage: {int(vendor['business_vintage_months'])} months", "", "STRENGTHS",
    ]
    lines.extend(f"+ {item}" for item in recommendations["strengths"] or ["None identified"])
    lines.extend(["", "WEAKNESSES"])
    lines.extend(f"- {item}" for item in recommendations["weaknesses"] or ["None identified"])
    lines.extend(["", "RECOMMENDED ACTIONS"])
    lines.extend(f"• {item}" for item in recommendations["actions"])
    lines.extend(["", "ELIGIBLE LOAN PRODUCTS", *(eligible or ["Reassessment recommended"]), "", "EXPLAINABILITY", result["narrative"], "",
                  "Decision-support prototype using synthetic data; final credit decision remains subject to bank policy."])
    return "\n".join(lines)


base_vendor, default_name = selected_vendor()
vendor = apply_simulator(base_vendor)
result = engine.score_vendor(vendor)
recommendations = build_recommendations(vendor, result["health_score"])
products = evaluate_loan_products(vendor, result["health_score"])
benchmark, percentile = build_benchmark(vendor, df)

eligible_count = sum(item["status"] == "Eligible" for item in products)
assessment_time = datetime.now().astimezone().strftime("%d %b %Y, %I:%M %p")

st.markdown(
    f"""
    <section class="hero">
        <div class="hero-main">
            <div class="eyebrow">MSME Credit Decision Support</div>
            <h1 class="hero-title">VyaparIQ</h1>
            <p class="hero-subtitle">
                Relationship-manager workspace for assessing thin-file MSMEs using operating
                signals, risk drivers and preliminary product-fit indicators.
            </p>
            <p class="hero-value">
                Supports faster credit conversations by translating alternate business activity
                into a clear credit health view for review, monitoring and customer engagement.
            </p>
        </div>
        <div class="hero-info-panel">
            <div class="hero-info-title">Assessment Context</div>
            <div class="hero-info-grid">
                <div class="hero-info-row">
                    <div class="hero-info-label">Environment</div>
                    <div class="hero-info-value">Prototype portfolio simulation</div>
                </div>
                <div class="hero-info-row">
                    <div class="hero-info-label">Purpose</div>
                    <div class="hero-info-value">RM and credit officer decision support</div>
                </div>
                <div class="hero-info-row">
                    <div class="hero-info-label">Status</div>
                    <div class="hero-info-value">Indicative assessment, subject to bank policy</div>
                </div>
                <div class="hero-info-row">
                    <div class="hero-info-label">Updated</div>
                    <div class="hero-info-value">{escape(assessment_time)}</div>
                </div>
            </div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

summary1, summary2, summary3, summary4 = st.columns(4)
with summary1:
    render_summary_card("Financial Health Score", f"{result['health_score']}/100", "Composite operating health indicator")
with summary2:
    render_summary_card("Credit Risk Level", result["risk_category"], "Current model risk classification")
with summary3:
    render_summary_card("Peer Percentile", f"{percentile}th", "Relative standing against synthetic cohort")
with summary4:
    render_summary_card("Eligible Products", f"{eligible_count} products", "Indicative policy fit for discussion")

tab1, tab2, tab3, tab4 = st.tabs(["Executive Summary", "Advisory", "Performance & Peers", "Financial Health Card"])

with tab1:
    left, right = st.columns([1, 2])
    with left:
        render_section("Financial Health Gauge", "Current score and risk classification.")
        st.plotly_chart(score_gauge(result["health_score"]), use_container_width=True)
        st.markdown(
            f'<span class="risk-badge {risk_class(result["risk_category"])}">{escape(result["risk_category"])}</span>',
            unsafe_allow_html=True,
        )
    with right:
        render_section("Credit Decision Rationale", "Plain-language explanation for the current assessment.")
        st.markdown(f'<div class="panel-card">{escape(result["narrative"])}</div>', unsafe_allow_html=True)
        top = result["contributions"].head(8)
        chart = pd.DataFrame({"Signal": [FEATURE_LABELS.get(key, key) for key in top.index], "Impact on Risk": top.values}).sort_values("Impact on Risk")
        fig = px.bar(chart, x="Impact on Risk", y="Signal", orientation="h", color="Impact on Risk",
                     color_continuous_scale=[(0, CHART_COLORS["success"]), (0.5, "#E5E7EB"), (1, CHART_COLORS["danger"])])
        fig.update_layout(title="Primary Risk Drivers", coloraxis_showscale=False)
        fig.update_traces(marker_line_width=0, hovertemplate="%{y}<br>Impact on Risk: %{x:.4f}<extra></extra>")
        style_chart(fig, height=370)
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("View underlying assessment signals"):
        st.dataframe(pd.DataFrame({"Signal": [FEATURE_LABELS[f] for f in FEATURES], "Value": [vendor[f] for f in FEATURES]}),
                     hide_index=True, use_container_width=True)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        render_list_card("Financial Strengths", recommendations["strengths"] or ["No material strengths identified."], "success")
        render_list_card("Management Insights", recommendations["insights"], "primary")
    with c2:
        render_list_card("Risk Considerations", recommendations["weaknesses"] or ["No material weaknesses identified."], "warning")
        render_list_card("Priority Actions", [f"{number}. {item}" for number, item in enumerate(recommendations["actions"], 1)], "primary")
    render_section("Indicative Product Eligibility", "Preliminary product fit for discussion; subject to applicable credit policy and due diligence.")
    product_df = pd.DataFrame(products).rename(columns={"product": "Product", "status": "Status", "reason": "Policy rationale"})
    st.dataframe(product_df, hide_index=True, use_container_width=True)

with tab3:
    render_section("12-Month Financial Health Trend", "Illustrative backcast based on current signals; not presented as observed account history.")
    timeline = build_timeline(vendor, engine.score_vendor)
    timeline_fig = px.line(timeline, x="Month", y="Health Score", markers=True, color_discrete_sequence=[CHART_COLORS["primary"]])
    timeline_fig.update_traces(line=dict(width=3), marker=dict(size=8), hovertemplate="%{x}<br>Health Score: %{y:.1f}<extra></extra>")
    style_chart(timeline_fig, height=390)
    timeline_fig.update_yaxes(range=[0, 100], title="Financial Health Score")
    st.plotly_chart(timeline_fig, use_container_width=True)
    render_section("Peer Benchmark", "Comparison against synthetic industry baseline and top-quartile operating profile.")
    benchmark_fig = px.bar(benchmark, x="Metric", y="Index", color="Cohort", barmode="group",
                           color_discrete_map={"Current Business": CHART_COLORS["primary"], "Industry Average": "#94A3B8", "Top Quartile": CHART_COLORS["success"]})
    benchmark_fig.update_traces(marker_line_width=0, hovertemplate="%{x}<br>%{fullData.name}: %{y:.0f}<extra></extra>")
    benchmark_fig.update_layout(legend_title_text="Comparison Group")
    style_chart(benchmark_fig, height=410, show_legend=True)
    benchmark_fig.update_yaxes(range=[0, 100], title="Peer Percentile Index")
    st.plotly_chart(benchmark_fig, use_container_width=True)

with tab4:
    st.markdown(
        f"""
        <div class="health-card">
            <div class="health-card-header">
                <div>
                    <p class="health-card-title">Financial Health Card</p>
                    <p class="health-card-meta">Executive credit-review summary for relationship managers and sanctioning teams.</p>
                </div>
                <span class="risk-badge {risk_class(result["risk_category"])}">{escape(result["risk_category"])}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    business_name = st.text_input("Business name", value=default_name, key="business_name")
    a, b, c = st.columns(3)
    with a:
        render_summary_card("Score", f"{result['health_score']}/100", "Financial health indicator")
    with b:
        render_summary_card("Credit Risk Band", result["risk_category"], "Policy review classification")
    with c:
        render_summary_card("Peer Standing", f"{percentile}th percentile", "Relative cohort position")
    st.markdown(
        f"""
        <div class="signal-grid">
            <div class="signal-item">
                <div class="signal-label">Monthly digital inflow</div>
                <div class="signal-value">{escape(format_currency(float(vendor["avg_monthly_upi_inflow"])))}</div>
            </div>
            <div class="signal-item">
                <div class="signal-label">Business vintage</div>
                <div class="signal-value">{int(vendor["business_vintage_months"])} months</div>
            </div>
            <div class="signal-item">
                <div class="signal-label">Assessment timestamp</div>
                <div class="signal-value">{escape(assessment_time)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left_card, right_card = st.columns(2)
    with left_card:
        render_list_card("Strengths", recommendations["strengths"] or ["None identified"], "success")
        render_list_card("Recommendations", recommendations["actions"], "primary")
    with right_card:
        render_list_card("Weaknesses", recommendations["weaknesses"] or ["None identified"], "warning")
        eligible_products = [item["product"] for item in products if item["status"] == "Eligible"] or ["Reassessment recommended"]
        render_list_card("Loan Products", eligible_products, "primary")
    report = report_text(business_name, vendor, result, recommendations, products, percentile)
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", business_name.strip()).strip("_") or "Business"
    st.download_button("Download Financial Health Card", report,
                       file_name=f"VyaparIQ_HealthCard_{safe_name}.txt", mime="text/plain")

st.divider()
st.caption("VyaparIQ · Decision-support prototype for IDBI Innovate 2026 · Synthetic data only · Final credit decisions remain subject to IDBI Bank policy")
