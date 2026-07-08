"""Auditable recommendations and loan eligibility for a scored MSME."""

from __future__ import annotations

from typing import Any


def build_recommendations(vendor: dict[str, Any], score: float) -> dict[str, list[str]]:
    """Return deterministic insights and actions linked to observable signals."""
    strengths: list[str] = []
    weaknesses: list[str] = []
    insights: list[str] = []
    actions: list[str] = []

    growth = float(vendor["upi_inflow_trend_pct"])
    volatility = float(vendor["upi_inflow_volatility"])
    order_growth = float(vendor["gig_order_trend_pct"])
    recycling = float(vendor["cash_recycling_score"])
    concentration = float(vendor["seasonal_dependency_index"])

    if growth >= 1:
        strengths.append(f"Digital collections are growing {growth:.1f}% month-on-month.")
    else:
        weaknesses.append(f"Digital collections growth is {growth:.1f}% month-on-month.")
        actions.append("Restore monthly UPI growth above 1% through repeat-customer offers and QR adoption.")

    if volatility <= 0.15:
        strengths.append("Cash-flow volatility is within a stable operating range.")
    else:
        weaknesses.append(f"Cash-flow volatility is elevated at {volatility:.0%}.")
        actions.append("Build a weekly cash buffer and align supplier payments to collection cycles.")

    if order_growth >= 0:
        strengths.append(f"Delivery-channel orders are trending upward by {order_growth:.1f}%.")
    else:
        weaknesses.append(f"Delivery-channel orders are declining by {abs(order_growth):.1f}%.")
        actions.append("Review delivery-channel pricing and reactivate dormant customers.")

    if recycling > 0.25:
        weaknesses.append("Cash recycling patterns indicate possible working-capital stress.")
        actions.append("Reduce circular transfers and document the source and purpose of large credits.")
    if concentration > 0.5:
        weaknesses.append("Revenue is highly dependent on a narrow seasonal demand window.")
        actions.append("Diversify products or customers to reduce seasonal revenue concentration below 50%.")

    insights.append(
        f"The business is currently assessed at {score:.1f}/100; "
        f"cash-flow stability and growth are the fastest controllable levers."
    )
    if vendor.get("gst_registered", 0) == 0:
        insights.append("GST-thin status is treated as context, not an automatic credit penalty.")
    if not actions:
        actions.append("Maintain current collection discipline and review limits after three stable months.")

    return {
        "strengths": strengths[:4],
        "weaknesses": weaknesses[:4],
        "insights": insights,
        "actions": actions[:4],
    }


def evaluate_loan_products(vendor: dict[str, Any], score: float) -> list[dict[str, str]]:
    """Evaluate a fixed banking product set using transparent policy rules."""
    vintage = int(vendor["business_vintage_months"])
    inflow = float(vendor["avg_monthly_upi_inflow"])
    seasonal = float(vendor["seasonal_dependency_index"])
    products = [
        ("Working Capital Loan", score >= 60 and vintage >= 12, "Stable operating cash-flow and 12+ months vintage"),
        ("Invoice Financing", score >= 50 and inflow >= 50_000, "Sufficient digital turnover for short-tenure financing"),
        ("Business Loan", score >= 72 and vintage >= 24, "Strong score and established operating history"),
        ("Machinery Loan", score >= 65 and vintage >= 18, "Adequate repayment capacity for productive assets"),
        ("Business Credit Card", score >= 55, "Meets minimum financial-health threshold"),
    ]
    results = []
    for name, eligible, reason in products:
        results.append({
            "product": name,
            "status": "Eligible" if eligible else "Not eligible",
            "reason": reason if eligible else f"Review after score improves (current: {score:.1f})",
        })
    if seasonal > 0.55 and score >= 60:
        results[0]["reason"] = "Seasonal working-capital structure recommended"
    return results
