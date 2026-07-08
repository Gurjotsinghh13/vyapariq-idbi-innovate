"""
Trains the MSME Financial Health scoring model and saves it for the app.

Key design point for the demo: two vendors with near-identical GST/EPFO/UPI-average
profiles should get very different scores and different plain-language explanations,
because the model reasons over trend/volatility/cash-cycling shape, not just averages.
"""

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

FEATURES = [
    "avg_monthly_upi_inflow",
    "upi_inflow_trend_pct",
    "upi_inflow_volatility",
    "upi_txn_count_monthly",
    "avg_ticket_size",
    "gig_order_volume_monthly",
    "gig_order_trend_pct",
    "business_vintage_months",
    "bank_account_tenure_months",
    "cash_recycling_score",
    "existing_credit_lines",
    "seasonal_dependency_index",
    "gst_registered",
    "epfo_active",
]

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "vendors_synthetic.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "health_score_model.joblib")


def train():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=8,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, preds, target_names=["Healthy", "Risky"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}")

    importances = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nFeature importances:")
    print(importances.round(4))

    joblib.dump({"model": clf, "features": FEATURES}, MODEL_PATH)
    print(f"\nSaved model -> {MODEL_PATH}")

    return clf, X_test, y_test


if __name__ == "__main__":
    train()
