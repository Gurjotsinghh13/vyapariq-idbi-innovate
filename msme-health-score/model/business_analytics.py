"""Deterministic demo analytics derived from current business signals."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd


def build_timeline(vendor: dict[str, Any], scorer: Callable[[dict[str, Any]], dict]) -> pd.DataFrame:
    """Reconstruct a stable 12-month score trend without claiming observed history."""
    rows = []
    current_growth = float(vendor["upi_inflow_trend_pct"])
    for months_ago in range(11, -1, -1):
        snapshot = dict(vendor)
        snapshot["upi_inflow_trend_pct"] = current_growth - (months_ago * 0.12)
        snapshot["upi_inflow_volatility"] = min(0.9, max(0.02, float(vendor["upi_inflow_volatility"]) + months_ago * 0.006))
        result = scorer(snapshot)
        rows.append({
            "Month": (pd.Timestamp.today().normalize() - pd.DateOffset(months=months_ago)).strftime("%b %Y"),
            "Health Score": result["health_score"],
            "Risk": result["risk_category"],
        })
    return pd.DataFrame(rows)


def build_benchmark(vendor: dict[str, Any], dataset: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Compare normalized operating metrics with the synthetic peer cohort."""
    specs = [
        ("UPI growth", "upi_inflow_trend_pct", False),
        ("Cash-flow stability", "upi_inflow_volatility", True),
        ("Revenue growth", "gig_order_trend_pct", False),
        ("Customer diversification", "seasonal_dependency_index", True),
    ]
    rows = []
    percentiles = []
    for label, feature, inverse in specs:
        series = dataset[feature].astype(float)
        value = float(vendor[feature])
        percentile = float((series <= value).mean() * 100)
        if inverse:
            percentile = 100 - percentile
        percentiles.append(percentile)
        rows.extend([
            {"Metric": label, "Cohort": "Current Business", "Index": round(percentile, 1)},
            {"Metric": label, "Cohort": "Industry Average", "Index": 50.0},
            {"Metric": label, "Cohort": "Top Quartile", "Index": 75.0},
        ])
    return pd.DataFrame(rows), int(round(float(np.mean(percentiles))))
