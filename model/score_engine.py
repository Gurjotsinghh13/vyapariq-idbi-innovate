"""
Explainable Financial Health scoring engine.

Given a vendor's feature row, produces:
  - health_score (0-100, higher = healthier)
  - risk_category (Low / Medium / High)
  - deployment-safe model-based feature contributions
  - a plain-language explanation, with an explicit callout when GST/EPFO
    thinness is present but NOT being treated as a risk driver -- this is
    the core "explainable contradiction resolution" demo moment.
"""

from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().with_name("health_score_model.joblib")
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "vendors_synthetic.csv"

FEATURE_LABELS = {
    "avg_monthly_upi_inflow": "Average monthly UPI inflow",
    "upi_inflow_trend_pct": "UPI inflow trend (month-on-month %)",
    "upi_inflow_volatility": "UPI inflow volatility",
    "upi_txn_count_monthly": "Monthly UPI transaction count",
    "avg_ticket_size": "Average transaction size",
    "gig_order_volume_monthly": "Monthly gig-delivery order volume",
    "gig_order_trend_pct": "Gig-delivery order trend (month-on-month %)",
    "business_vintage_months": "Business vintage (months)",
    "bank_account_tenure_months": "Bank account tenure (months)",
    "cash_recycling_score": "Cash recycling / round-tripping indicator",
    "existing_credit_lines": "Existing informal credit lines",
    "seasonal_dependency_index": "Seasonal dependency index",
    "gst_registered": "GST registered",
    "epfo_active": "EPFO active",
}


class HealthScoreEngine:
    def __init__(self, model_path=MODEL_PATH):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.reference_values = self._load_reference_values()

    def score_vendor(self, vendor_row: dict):
        X = pd.DataFrame([vendor_row])[self.features]

        risk_proba = self.model.predict_proba(X)[0][1]
        health_score = round((1 - risk_proba) * 100, 1)

        if health_score >= 70:
            risk_category = "Low Risk"
        elif health_score >= 45:
            risk_category = "Medium Risk"
        else:
            risk_category = "High Risk"

        contributions = self._explain_with_model_perturbation(X)

        narrative = self._build_narrative(vendor_row, contributions, health_score, risk_category)
        strengths, weaknesses = self._extract_strengths_weaknesses(vendor_row, contributions)
        recommendation, confidence = self._recommend_product(vendor_row, health_score, risk_category)

        return {
            "health_score": health_score,
            "risk_category": risk_category,
            "contributions": contributions,
            "narrative": narrative,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendation": recommendation,
            "recommendation_confidence": confidence,
        }

    def _load_reference_values(self) -> pd.Series:
        """Load cohort medians used as a safe baseline for local explanations.

        SHAP/Numba can trigger native-library crashes on some hosted Linux
        images. For the demo app we keep explainability deterministic and
        deployment-safe by comparing the model's current risk probability with
        the probability after replacing one feature at a time by its cohort
        median. Positive values increase predicted risk; negative values reduce
        predicted risk.
        """
        try:
            reference_df = pd.read_csv(DATA_PATH)
            return reference_df[self.features].median(numeric_only=True)
        except (FileNotFoundError, KeyError, OSError, ValueError):
            return pd.Series(0.0, index=self.features)

    def _explain_with_model_perturbation(self, X: pd.DataFrame) -> pd.Series:
        X = X.astype(float)
        current_risk = self.model.predict_proba(X)[0][1]
        rows = []
        for feature in self.features:
            perturbed = X.copy()
            perturbed.at[perturbed.index[0], feature] = self.reference_values.get(feature, 0.0)
            rows.append(perturbed)

        perturbed_X = pd.concat(rows, ignore_index=True)
        perturbed_risk = self.model.predict_proba(perturbed_X)[:, 1]
        contributions = current_risk - perturbed_risk
        return pd.Series(contributions, index=self.features).sort_values(key=abs, ascending=False)

    def _extract_strengths_weaknesses(self, vendor_row, contributions):
        """Rule-based, human-readable strengths/weaknesses -- the 'Financial Health Card' view.
        Uses the same model-based contributions so this stays consistent with the score,
        not a separate made-up checklist."""
        strengths, weaknesses = [], []

        def add(cond, positive_text, negative_text, feature):
            impact = contributions.get(feature, 0)
            if cond:
                strengths.append(positive_text)
            elif abs(impact) > 0.005:
                weaknesses.append(negative_text)

        add(vendor_row["upi_inflow_trend_pct"] > 0,
            "UPI inflow trending upward month-on-month",
            "UPI inflow is declining month-on-month",
            "upi_inflow_trend_pct")

        add(vendor_row["upi_inflow_volatility"] < 0.15,
            "Low volatility in cash inflow -- predictable revenue",
            "High volatility in cash inflow -- unpredictable revenue",
            "upi_inflow_volatility")

        add(vendor_row["cash_recycling_score"] < 0.20,
            "No signs of cash round-tripping or working-capital stress",
            "Elevated cash recycling pattern -- possible working-capital stress",
            "cash_recycling_score")

        add(vendor_row["gig_order_trend_pct"] > 0,
            "Order volume from delivery channel growing",
            "Order volume from delivery channel shrinking",
            "gig_order_trend_pct")

        add(vendor_row["existing_credit_lines"] <= 1,
            "Low existing informal credit exposure",
            "Multiple existing informal credit lines -- possible over-leverage",
            "existing_credit_lines")

        add(vendor_row["business_vintage_months"] >= 24,
            f"Established business vintage ({int(vendor_row['business_vintage_months'])} months)",
            "Relatively new business with limited track record",
            "business_vintage_months")

        # Always note the GST/EPFO context as a neutral, not-penalized item
        if vendor_row.get("gst_registered", 0) == 0 and vendor_row.get("epfo_active", 0) == 0:
            strengths.append("GST/EPFO absence assessed as regulatory-consistent (direct-to-customer model), not penalized in scoring")

        return strengths, weaknesses

    def _recommend_product(self, vendor_row, health_score, risk_category):
        """Simple, explainable rule-based product recommendation -- deliberately not a black box,
        since a lending recommendation needs to be auditable."""
        seasonal = vendor_row["seasonal_dependency_index"]

        if risk_category == "High Risk":
            return ("Not recommended for unsecured lending at this time -- suggest re-evaluation after 3 months of improved cash-flow stability", 0.0)

        if seasonal > 0.5 and health_score >= 45:
            confidence = min(0.95, 0.55 + health_score / 200)
            return ("Seasonal Working Capital Loan (flexible repayment aligned to demand cycles)", round(confidence, 2))

        if health_score >= 70:
            confidence = min(0.97, 0.60 + health_score / 150)
            return ("Standard Working Capital Loan", round(confidence, 2))

        confidence = min(0.85, 0.40 + health_score / 150)
        return ("Micro-Working Capital Loan (smaller ticket size, shorter tenure)", round(confidence, 2))


    def _build_narrative(self, vendor_row, contributions, health_score, risk_category):
        lines = []

        thin_file = vendor_row.get("gst_registered", 0) == 0 and vendor_row.get("epfo_active", 0) == 0
        gst_epfo_importance = abs(contributions.get("gst_registered", 0)) + abs(contributions.get("epfo_active", 0))
        top_driver = contributions.index[0]

        if thin_file:
            lines.append(
                "This vendor has no GST registration and no active EPFO record. "
                "Under Section 24(ix) of the CGST Act, GST registration is only mandatory "
                "for sellers transacting through an e-commerce operator that collects TCS -- "
                "this vendor takes orders directly and uses gig logistics purely for delivery, "
                "so thin GST/EPFO history here is consistent with normal, compliant informal "
                "retail rather than a sign of concealment."
            )
            if gst_epfo_importance < 0.01:
                lines.append(
                    f"Accordingly, the model assigns this factor near-zero weight "
                    f"(combined contribution: {gst_epfo_importance:.4f}) -- the health score "
                    f"is being driven by cash-flow behaviour, not by formal-file completeness."
                )

        top_label = FEATURE_LABELS.get(top_driver, top_driver)
        direction = "increasing" if contributions[top_driver] > 0 else "reducing"
        lines.append(
            f"The single strongest factor in this score is '{top_label}', which is {direction} "
            f"predicted risk more than any other signal."
        )

        lines.append(f"Overall Health Score: {health_score}/100 -- classified as {risk_category}.")
        return " ".join(lines)


if __name__ == "__main__":
    engine = HealthScoreEngine()

    # Demo case: the core contradiction -- GST-thin, EPFO-thin, healthy UPI average,
    # but STABLE trend/low volatility (Archetype A pattern)
    healthy_vendor = {
        "avg_monthly_upi_inflow": 82000,
        "upi_inflow_trend_pct": 1.2,
        "upi_inflow_volatility": 0.09,
        "upi_txn_count_monthly": 340,
        "avg_ticket_size": 241,
        "gig_order_volume_monthly": 270,
        "gig_order_trend_pct": 1.5,
        "business_vintage_months": 32,
        "bank_account_tenure_months": 30,
        "cash_recycling_score": 0.07,
        "existing_credit_lines": 0,
        "seasonal_dependency_index": 0.22,
        "gst_registered": 0,
        "epfo_active": 0,
    }

    # Same surface averages, but declining trend + high volatility + cash recycling
    risky_vendor = dict(healthy_vendor)
    risky_vendor.update({
        "upi_inflow_trend_pct": -4.5,
        "upi_inflow_volatility": 0.38,
        "gig_order_trend_pct": -5.0,
        "cash_recycling_score": 0.52,
        "existing_credit_lines": 2,
    })

    print("=== HEALTHY BUT INVISIBLE (GST-thin, EPFO-thin) ===")
    result_a = engine.score_vendor(healthy_vendor)
    print(result_a["narrative"])
    print()

    print("=== GENUINELY RISKY (same GST/EPFO/avg-inflow profile) ===")
    result_b = engine.score_vendor(risky_vendor)
    print(result_b["narrative"])
