"""VyaparIQ Streamlit dashboard."""

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


st.set_page_config(page_title="VyaparIQ | MSME Financial Health", page_icon="📊", layout="wide")
st.markdown("""
<style>
    :root {--bank-navy:#12345b; --bank-blue:#1d4f7a; --bank-gold:#b8892d; --ink:#182230; --muted:#667085;}
    .stApp {background:#f8fafc; color:var(--ink);}
    .block-container {max-width:1440px; padding-top:2rem; padding-bottom:3rem;}
    h1 {color:var(--bank-navy); font-size:2.15rem !important; letter-spacing:-0.02em; margin-bottom:.15rem !important;}
    h2, h3 {color:var(--bank-navy); letter-spacing:-0.01em;}
    [data-testid="stSidebar"] {background:#f1f5f9; border-right:1px solid #d8e0e8;}
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {color:var(--bank-navy);}
    [data-testid="stMetric"] {background:#ffffff; border:1px solid #dce3ea; border-top:3px solid var(--bank-blue); padding:1rem 1.1rem; border-radius:6px; box-shadow:0 1px 2px rgba(16,24,40,.04); min-height:112px;}
    [data-testid="stMetricLabel"] {color:#475467; font-size:.82rem; font-weight:650; text-transform:uppercase; letter-spacing:.035em;}
    [data-testid="stMetricValue"] {color:var(--bank-navy); font-weight:700;}
    [data-baseweb="tab-list"] {gap:.35rem; border-bottom:1px solid #dce3ea;}
    [data-baseweb="tab"] {padding:.75rem 1.1rem; color:#475467; font-weight:600;}
    [aria-selected="true"][data-baseweb="tab"] {color:var(--bank-navy);}
    .risk-low,.risk-medium,.risk-high {display:inline-block; color:white; padding:7px 16px; border-radius:4px; font-size:.84rem; font-weight:700; letter-spacing:.025em; text-transform:uppercase;}
    .risk-low {background:#18794e}.risk-medium {background:#a15c00}.risk-high {background:#b42318}
    .executive-kicker {color:var(--bank-gold); font-size:.76rem; font-weight:750; letter-spacing:.11em; text-transform:uppercase; margin-bottom:.25rem;}
    .executive-subtitle {color:#475467; font-size:1rem; margin-bottom:.1rem;}
    .small-note {color:var(--muted); font-size:.82rem; line-height:1.45; margin-top:.25rem;}
    div[data-testid="stExpander"] {background:#fff; border:1px solid #dce3ea; border-radius:6px;}
    div[data-testid="stAlert"] {border-radius:5px;}
    hr {border-color:#dce3ea !important;}
</style>
""", unsafe_allow_html=True)

DATA_PATH = PROJECT_ROOT / "data" / "vendors_synthetic.csv"


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
    st.sidebar.subheader("Business Profile Inputs")
    values = {
        "avg_monthly_upi_inflow": st.sidebar.slider("Avg monthly UPI inflow (₹)", 20_000, 250_000, 82_000, 1_000),
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
    st.sidebar.caption("ASSESSMENT WORKSPACE")
    st.sidebar.markdown("**Quick Demo**")
    healthy_col, risk_col = st.sidebar.columns(2)
    healthy_col.button("Healthy MSME", use_container_width=True, on_click=load_demo, args=("Healthy MSME",))
    risk_col.button("High Risk MSME", use_container_width=True, on_click=load_demo, args=("High Risk MSME",))
    st.sidebar.button("Reset current scenario", use_container_width=True, on_click=clear_simulator)
    st.sidebar.caption("Start with either profile, adjust the scenario, then reset to return to its baseline.")
    st.sidebar.divider()

    mode = st.sidebar.radio(
        "Assessment source", ["Contradiction demo", "Synthetic dataset", "Manual input"], key="assessment_source"
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
        "axis": {"range": [0, 100]}, "bar": {"color": "#143d6b"},
        "steps": [{"range": [0, 45], "color": "#fee4e2"}, {"range": [45, 70], "color": "#fef0c7"},
                  {"range": [70, 100], "color": "#d1fadf"}],
    }))
    figure.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20))
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


st.markdown('<div class="executive-kicker">IDBI Innovate 2026 · MSME Credit Intelligence</div>', unsafe_allow_html=True)
st.title("VyaparIQ")
st.markdown('<div class="executive-subtitle">Explainable financial-health assessment for new-to-credit and new-to-bank enterprises</div>', unsafe_allow_html=True)
st.caption("Decision support using alternate operating signals · Transparent, auditable and near real-time")

base_vendor, default_name = selected_vendor()
vendor = apply_simulator(base_vendor)
result = engine.score_vendor(vendor)
recommendations = build_recommendations(vendor, result["health_score"])
products = evaluate_loan_products(vendor, result["health_score"])
benchmark, percentile = build_benchmark(vendor, df)

summary1, summary2, summary3, summary4 = st.columns(4)
summary1.metric("Financial Health Score", f"{result['health_score']}/100")
summary2.metric("Credit Risk Band", result["risk_category"])
summary3.metric("Peer Percentile", f"{percentile}th")
summary4.metric("Indicative Product Fit", f"{sum(item['status'] == 'Eligible' for item in products)} products")

tab1, tab2, tab3, tab4 = st.tabs(["Executive Summary", "Advisory", "Performance & Peers", "Financial Health Card"])

with tab1:
    left, right = st.columns([1, 2])
    with left:
        st.plotly_chart(score_gauge(result["health_score"]), use_container_width=True)
        css = {"Low Risk": "risk-low", "Medium Risk": "risk-medium", "High Risk": "risk-high"}[result["risk_category"]]
        st.markdown(f'<span class="{css}">{result["risk_category"]}</span>', unsafe_allow_html=True)
    with right:
        st.subheader("Credit Decision Rationale")
        st.write(result["narrative"])
        top = result["contributions"].head(8)
        chart = pd.DataFrame({"Signal": [FEATURE_LABELS.get(key, key) for key in top.index], "Impact on Risk": top.values}).sort_values("Impact on Risk")
        fig = px.bar(chart, x="Impact on Risk", y="Signal", orientation="h", color="Impact on Risk",
                     color_continuous_scale=[(0, "#18794e"), (0.5, "#e5e7eb"), (1, "#b42318")])
        fig.update_layout(title="Primary risk drivers", height=350, coloraxis_showscale=False,
                          plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=dict(color="#344054"),
                          margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("View underlying assessment signals"):
        st.dataframe(pd.DataFrame({"Signal": [FEATURE_LABELS[f] for f in FEATURES], "Value": [vendor[f] for f in FEATURES]}),
                     hide_index=True, use_container_width=True)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Financial Strengths")
        for item in recommendations["strengths"] or ["No material strengths identified."]:
            st.success(item)
        st.subheader("Management Insights")
        for item in recommendations["insights"]:
            st.info(item)
    with c2:
        st.subheader("Risk Considerations")
        for item in recommendations["weaknesses"] or ["No material weaknesses identified."]:
            st.warning(item)
        st.subheader("Priority Actions")
        for number, item in enumerate(recommendations["actions"], 1):
            st.write(f"{number}. {item}")
    st.subheader("Indicative Product Eligibility")
    st.caption("Preliminary product fit for discussion; subject to applicable credit policy and due diligence.")
    product_df = pd.DataFrame(products).rename(columns={"product": "Product", "status": "Status", "reason": "Policy rationale"})
    st.dataframe(product_df, hide_index=True, use_container_width=True)

with tab3:
    st.subheader("12-Month Financial Health Trend")
    timeline = build_timeline(vendor, engine.score_vendor)
    timeline_fig = px.line(timeline, x="Month", y="Health Score", markers=True, color_discrete_sequence=["#143d6b"])
    timeline_fig.update_layout(plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=dict(color="#344054"))
    timeline_fig.update_yaxes(range=[0, 100], gridcolor="#e9eef3")
    st.plotly_chart(timeline_fig, use_container_width=True)
    st.caption("Illustrative backcast based on current signals; not presented as observed account history.")
    st.subheader("Peer Benchmark")
    benchmark_fig = px.bar(benchmark, x="Metric", y="Index", color="Cohort", barmode="group",
                           color_discrete_map={"Current Business": "#143d6b", "Industry Average": "#98a2b3", "Top Quartile": "#12b76a"})
    benchmark_fig.update_layout(plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=dict(color="#344054"), legend_title_text="Comparison Group")
    benchmark_fig.update_yaxes(range=[0, 100], title="Peer percentile index", gridcolor="#e9eef3")
    st.plotly_chart(benchmark_fig, use_container_width=True)

with tab4:
    st.subheader("Financial Health Card")
    st.caption("Executive summary for credit review and customer discussion.")
    business_name = st.text_input("Business name", value=default_name, key="business_name")
    a, b, c = st.columns(3)
    a.metric("Score", f"{result['health_score']}/100")
    b.metric("Credit Risk Band", result["risk_category"])
    c.metric("Peer Standing", f"{percentile}th percentile")
    st.markdown(f"**Profile:** ₹{float(vendor['avg_monthly_upi_inflow']):,.0f} monthly digital inflow • "
                f"{int(vendor['business_vintage_months'])} months vintage")
    report = report_text(business_name, vendor, result, recommendations, products, percentile)
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", business_name.strip()).strip("_") or "Business"
    st.download_button("Download Financial Health Card", report,
                       file_name=f"VyaparIQ_HealthCard_{safe_name}.txt", mime="text/plain")

st.divider()
st.caption("VyaparIQ · Decision-support prototype for IDBI Innovate 2026 · Synthetic data only · Final credit decisions remain subject to IDBI Bank policy")
