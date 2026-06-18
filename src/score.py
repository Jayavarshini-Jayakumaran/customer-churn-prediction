"""
score.py — Assign churn probability and risk tier to every customer.
"""

import pandas as pd
import numpy as np

THRESHOLDS = {"low": 0.30, "high": 0.60}
TIERS      = ["Low (<30%)", "Medium (30-60%)", "High (>60%)"]


# ── Public API ────────────────────────────────────────────────────────

def score_all_customers(
    df_raw      : pd.DataFrame,
    model,
    scaler,
    imputer,
    df_encoded  : pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the full dataset through the fitted model and return a scored frame.

    Parameters
    ----------
    df_raw    : original dataframe (for readable columns like customerID)
    model     : fitted sklearn / xgboost model
    scaler    : fitted StandardScaler
    imputer   : fitted SimpleImputer
    df_encoded: numerically encoded feature matrix (no customerID / Churn)

    Returns
    -------
    DataFrame sorted by churn_probability descending
    """
    X_sc   = scaler.transform(imputer.transform(df_encoded))
    proba  = model.predict_proba(X_sc)[:, 1]

    scored = df_raw[["customerID", "tenure", "Contract", "MonthlyCharges"]].copy()
    scored["churn_probability"] = proba
    scored["risk_tier"] = pd.cut(
        proba,
        bins=[0, THRESHOLDS["low"], THRESHOLDS["high"], 1.0],
        labels=TIERS,
    )
    return scored.sort_values("churn_probability", ascending=False).reset_index(drop=True)


def revenue_summary(scored: pd.DataFrame, roc_auc: float) -> dict:
    """Aggregate business-level metrics from the scored frame."""
    total_rev    = scored["MonthlyCharges"].sum()
    high_risk    = scored[scored["risk_tier"] == "High (>60%)"]
    rev_at_risk  = high_risk["MonthlyCharges"].sum()

    dist = (
        scored["risk_tier"]
        .value_counts(normalize=True)
        .reindex(TIERS)
        .mul(100)
    )

    return {
        "roc_auc"        : roc_auc,
        "total_customers": len(scored),
        "total_rev"      : total_rev,
        "rev_at_risk"    : rev_at_risk,
        "annual_at_risk" : rev_at_risk * 12,
        "high_risk_count": len(high_risk),
        "dist"           : dist,
    }
