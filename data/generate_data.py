"""
Synthetic data generator for MSME Financial Health Score.

Segment: Independent kirana / local retailers who take orders directly
(UPI QR, WhatsApp, phone) and use gig logistics (Porter/Dunzo/Swiggy Genie
style couriers) purely for last-mile delivery. They sit below the GST
registration threshold and have thin/no EPFO trail -- legitimately, because
they are not "supplying through an e-commerce operator" under Sec 24(ix) CGST.

Two archetypes are generated with SIMILAR surface stats (GST-thin, EPFO-thin,
similar average UPI volume) but different underlying signal patterns:

  Archetype A - "Healthy but invisible": stable/growing UPI settlement,
      low volatility, no unusual cash cycling. Traditional scoring would
      flag them as high-risk (thin file). They are not.

  Archetype B - "Genuinely risky": similar surface averages, but declining
      trend, high volatility, and cash round-tripping patterns consistent
      with working-capital stress or borrowed-capital cycling.

The model's job: separate A from B using signal shape, not just averages.
"""

import os

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_PER_CLASS = 1200


def _clip(arr, lo, hi):
    return np.clip(arr, lo, hi)


def generate_archetype_a(n):
    """Healthy but invisible: platform-adjacent kirana, stable UPI, thin file by design not by risk."""
    months = 12
    base_inflow = RNG.normal(85000, 22000, n)          # avg monthly UPI inflow (INR)
    base_inflow = _clip(base_inflow, 25000, 250000)

    # low volatility, mild positive or flat trend
    monthly_growth = RNG.normal(0.01, 0.015, n)         # ~1%/month average growth
    volatility = RNG.normal(0.10, 0.03, n)              # coefficient of variation
    volatility = _clip(volatility, 0.04, 0.22)

    upi_txn_count = RNG.normal(340, 90, n)              # monthly transaction count
    upi_txn_count = _clip(upi_txn_count, 80, 900)

    gig_order_volume = RNG.normal(260, 70, n)           # monthly gig-logistics orders used
    gig_order_volume = _clip(gig_order_volume, 40, 700)
    gig_order_trend = RNG.normal(0.01, 0.02, n)

    business_vintage_months = RNG.normal(30, 14, n)
    business_vintage_months = _clip(business_vintage_months, 4, 96)

    bank_tenure_months = _clip(business_vintage_months - RNG.normal(2, 3, n), 2, 96)

    cash_recycling_score = RNG.normal(0.08, 0.05, n)    # low = little same-day in/out pairing
    cash_recycling_score = _clip(cash_recycling_score, 0.0, 0.6)

    existing_credit_lines = RNG.poisson(0.4, n)
    seasonal_dependency = RNG.normal(0.25, 0.12, n)     # moderate seasonality is normal
    seasonal_dependency = _clip(seasonal_dependency, 0.0, 0.9)

    gst_registered = (RNG.random(n) < 0.12).astype(int)   # a few crossed threshold
    epfo_active = (RNG.random(n) < 0.05).astype(int)

    label = np.zeros(n, dtype=int)  # 0 = healthy

    return _assemble(
        base_inflow, monthly_growth, volatility, upi_txn_count,
        gig_order_volume, gig_order_trend, business_vintage_months,
        bank_tenure_months, cash_recycling_score, existing_credit_lines,
        seasonal_dependency, gst_registered, epfo_active, label
    )


def generate_archetype_b(n):
    """Genuinely risky: similar averages, but declining trend + volatility + cash cycling."""
    base_inflow = RNG.normal(80000, 24000, n)
    base_inflow = _clip(base_inflow, 20000, 240000)

    monthly_growth = RNG.normal(-0.035, 0.03, n)        # declining
    volatility = RNG.normal(0.32, 0.09, n)              # much higher volatility
    volatility = _clip(volatility, 0.15, 0.75)

    upi_txn_count = RNG.normal(310, 100, n)
    upi_txn_count = _clip(upi_txn_count, 60, 900)

    gig_order_volume = RNG.normal(230, 80, n)
    gig_order_volume = _clip(gig_order_volume, 20, 650)
    gig_order_trend = RNG.normal(-0.04, 0.03, n)

    business_vintage_months = RNG.normal(22, 15, n)
    business_vintage_months = _clip(business_vintage_months, 2, 90)

    bank_tenure_months = _clip(business_vintage_months - RNG.normal(1, 3, n), 1, 90)

    cash_recycling_score = RNG.normal(0.45, 0.15, n)    # elevated round-tripping
    cash_recycling_score = _clip(cash_recycling_score, 0.05, 0.95)

    existing_credit_lines = RNG.poisson(1.8, n)         # more informal borrowing
    seasonal_dependency = RNG.normal(0.35, 0.18, n)
    seasonal_dependency = _clip(seasonal_dependency, 0.0, 0.95)

    gst_registered = (RNG.random(n) < 0.10).astype(int)
    epfo_active = (RNG.random(n) < 0.04).astype(int)

    label = np.ones(n, dtype=int)  # 1 = risky

    return _assemble(
        base_inflow, monthly_growth, volatility, upi_txn_count,
        gig_order_volume, gig_order_trend, business_vintage_months,
        bank_tenure_months, cash_recycling_score, existing_credit_lines,
        seasonal_dependency, gst_registered, epfo_active, label
    )


def _assemble(base_inflow, monthly_growth, volatility, upi_txn_count,
              gig_order_volume, gig_order_trend, business_vintage_months,
              bank_tenure_months, cash_recycling_score, existing_credit_lines,
              seasonal_dependency, gst_registered, epfo_active, label):
    n = len(base_inflow)
    avg_ticket_size = base_inflow / _clip(upi_txn_count, 1, None)
    return pd.DataFrame({
        "avg_monthly_upi_inflow": base_inflow.round(0),
        "upi_inflow_trend_pct": (monthly_growth * 100).round(2),
        "upi_inflow_volatility": volatility.round(3),
        "upi_txn_count_monthly": upi_txn_count.round(0),
        "avg_ticket_size": avg_ticket_size.round(0),
        "gig_order_volume_monthly": gig_order_volume.round(0),
        "gig_order_trend_pct": (gig_order_trend * 100).round(2),
        "business_vintage_months": business_vintage_months.round(0),
        "bank_account_tenure_months": bank_tenure_months.round(0),
        "cash_recycling_score": cash_recycling_score.round(3),
        "existing_credit_lines": existing_credit_lines,
        "seasonal_dependency_index": seasonal_dependency.round(3),
        "gst_registered": gst_registered,
        "epfo_active": epfo_active,
        "label": label,
    })


def generate_dataset():
    df_a = generate_archetype_a(N_PER_CLASS)
    df_b = generate_archetype_b(N_PER_CLASS)
    df = pd.concat([df_a, df_b], ignore_index=True)

    # Realism pass: real banking data is never cleanly separable.
    # 1) Add measurement noise to the key discriminating signals.
    noisy_cols = ["cash_recycling_score", "upi_inflow_volatility", "upi_inflow_trend_pct", "gig_order_trend_pct"]
    for col in noisy_cols:
        noise = RNG.normal(0, df[col].std() * 0.35, len(df))
        df[col] = df[col] + noise

    df["cash_recycling_score"] = _clip(df["cash_recycling_score"], 0.0, 0.98)
    df["upi_inflow_volatility"] = _clip(df["upi_inflow_volatility"], 0.02, 0.9)

    # 2) Flip a small fraction of labels to simulate real-world ambiguous/
    #    mislabeled cases (e.g. a healthy vendor going through a rough patch,
    #    or a risky vendor mid-recovery). No real dataset is 100% clean.
    flip_frac = 0.07
    flip_idx = RNG.choice(df.index, size=int(len(df) * flip_frac), replace=False)
    df.loc[flip_idx, "label"] = 1 - df.loc[flip_idx, "label"]

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.insert(0, "vendor_id", [f"V{i:05d}" for i in range(len(df))])
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = os.path.join(os.path.dirname(__file__), "vendors_synthetic.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df.groupby("label").mean(numeric_only=True).round(2).T)
