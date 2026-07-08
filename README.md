# VyaparIQ
### AI-Powered MSME Financial Health Intelligence Platform
**Built for IDBI Innovate 2026 · PS3 — Financial Health Score**
Team: RG Innovations (Gurjot, Ronit Gandhi)

## Problem

Traditional MSME credit assessment relies heavily on financial documents that
many New-to-Credit and New-to-Bank businesses do not possess. This is
especially true for **gig-economy-adjacent micro-retailers** — kirana stores
and small retailers who take orders directly from customers (UPI QR,
WhatsApp, phone) and use gig logistics (Porter/Dunzo/Swiggy-Genie-style
couriers) purely for last-mile delivery.

**VyaparIQ evaluates financial health using alternate business signals and
explainable AI to support faster, more inclusive lending decisions** — even
for businesses with no GST filing and no EPFO history, where that absence is
regulatory-consistent rather than a red flag.

## Why These Businesses Are Legitimately Thin-File, Not High-Risk

Under Section 24(ix) of the CGST Act, mandatory GST registration applies
specifically to sellers transacting *through an e-commerce operator that
collects TCS*. A vendor who takes orders directly and only uses gig logistics
for delivery does not trigger this mandatory-registration requirement, and
can legitimately sit below the GST threshold. Traditional credit scoring
reads "GST-thin" as "high risk." Often, it isn't — and VyaparIQ is built to
tell the difference.

## The Core Insight (and the demo)

Two vendors can have **near-identical GST status, EPFO status, and average
UPI inflow** — and still have wildly different creditworthiness, because the
real difference is in the *shape* of their cash flow (trend, volatility, cash
round-tripping), not in the averages.

In the included demo:
- Both vendors: GST-thin, EPFO-thin, ~₹82,000/month average UPI inflow
- Vendor A (stable trend, low volatility): **Health Score 93.9 — Low Risk**
- Vendor B (declining trend, high volatility, cash recycling): **Health Score 6.1 — High Risk**

The model explicitly assigns near-zero importance to GST/EPFO status
(confirmed via SHAP feature importance) — it is not penalizing thin-file
status, only genuine cash-flow risk signals.

## Features

- Explainable Financial Health Score (0–100) and risk assessment
- SHAP decision trail with a trained Random Forest model
- Live what-if simulator using the existing trained model
- Deterministic strengths, weaknesses, insights, and recommended actions
- Illustrative 12-month score timeline with explicit synthetic-data labelling
- Peer percentile, industry-average, and top-quartile benchmarks
- Auditable eligibility checks for five MSME lending products
- Downloadable Financial Health Card
- Synthetic banking dataset reflecting real regulatory patterns

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- SHAP
- Pandas / NumPy
- Plotly

## Screenshots

_Add dashboard screenshots here after deployment (see `/screenshots` folder)._

## What's Included

- **Synthetic data generator** (`data/generate_data.py`) — two vendor
  archetypes with overlapping surface stats but different underlying signal
  shape, plus realistic noise/label-overlap (no dataset is 100% clean)
- **Random Forest classifier** (`model/train_model.py`) — 93% accuracy,
  0.94 ROC-AUC on held-out data
- **Explainable scoring engine** (`model/score_engine.py`) — SHAP-based
  feature attribution, plain-language reasoning, strengths/weaknesses
  extraction, and a rule-based (auditable) loan product recommendation
- **Streamlit dashboard** (`app/app.py`) — score and SHAP assessment,
  what-if simulation, deterministic recommendations, timeline, peer
  benchmarks, product eligibility, and downloadable Health Card

## Architecture

```
Synthetic Data Generator
        │
        ▼
Feature Engineering (UPI trend/volatility, cash-recycling, gig-order signals)
        │
        ▼
Random Forest Classifier ──► Health Score (0-100) + Risk Category
        │
        ▼
SHAP Explainability Engine
        │
        ├──► Plain-language narrative (with regulatory context)
        ├──► Strengths / Weaknesses extraction
        └──► Rule-based Loan Recommendation
                │
                ▼
        Financial Health Card (Streamlit UI)
```

## Running Locally

Deploy with Python 3.12, which is compatible with the pinned model and SHAP
dependencies. In Streamlit Community Cloud, select Python 3.12 under Advanced
settings and use `msme-health-score/app/app.py` as the entrypoint.

Run local commands from the Git repository root:

```bash
pip install -r requirements.txt
streamlit run msme-health-score/app/app.py
```

The repository already includes the synthetic dataset and trained model.
Regenerate or retrain only when intentionally changing those artifacts:

```bash
python msme-health-score/data/generate_data.py
python msme-health-score/model/train_model.py
```

## Regulatory Grounding

This project's core segmentation logic is grounded in Section 24(ix) of the
CGST Act (mandatory GST registration for e-commerce-operator-facilitated
supply) and publicly documented seller-onboarding requirements of major
quick-commerce platforms. All vendor data in this prototype is **synthetic**
— generated to reflect plausible real-world patterns, not derived from real
GST/UPI/EPFO/Account Aggregator records. No live integration with GST, UPI,
Account Aggregator, EPFO, ULI, or OCEN systems exists in this prototype;
those connectors are represented as architectural placeholders for a
production build.

## Future Development

- Live integration with Account Aggregator framework for consented UPI/bank data
- OCEN-compliant lending disbursal flow
- ULI integration for last-mile credit delivery
- Real GST/EPFO API integration where applicable
- Multi-segment support beyond gig-economy-adjacent kirana retail
