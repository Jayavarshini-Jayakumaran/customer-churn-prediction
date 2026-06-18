import numpy as np
import pandas as pd


def churn_risk_distribution(y_prob):
    """Compute churn risk tier distribution as percentages."""
    bins = {
        "Low Risk (<30%)":      (0.0, 0.3),
        "Medium Risk (30–60%)": (0.3, 0.6),
        "High Risk (>60%)":     (0.6, 1.0),
    }

    total = len(y_prob)
    dist  = {}
    for label, (lo, hi) in bins.items():
        count      = ((y_prob >= lo) & (y_prob < hi)).sum()
        dist[label] = round((count / total) * 100, 2)

    return dist


def revenue_at_risk(df, y_prob, threshold=0.6):
    """Estimate total revenue exposed to high churn risk (probability ≥ threshold)."""
    mask     = y_prob >= threshold
    churners = df[mask]
    return churners["Total_Revenue"].sum()


def business_summary(roc_auc, risk_dist, revenue_risk):
    """Return a structured dict suitable for printing in main.py."""
    return {
        "roc_auc":          round(roc_auc, 3),
        "risk_distribution": risk_dist,
        "revenue_at_risk":  revenue_risk,
    }


def interpret_auc(roc_auc):
    """Translate a raw ROC-AUC score into a qualitative discrimination label.

    FIX (hardcoded interpretation): main.py used to always print "Model shows
    strong churn discrimination" regardless of the actual ROC-AUC value. This
    function ties the label to the real score instead, using the common
    rule-of-thumb scale for binary classifiers:
        >= 0.90  Excellent
        >= 0.80  Strong
        >= 0.70  Moderate / acceptable
        >= 0.60  Weak (only marginally better than chance)
        <  0.60  Negligible / close to random guessing

    Returns
    -------
    label        : str  — human-readable discrimination level
    is_reliable  : bool — whether the score is high enough (>= 0.7) to
                   responsibly act on the risk tiers / revenue-at-risk figures
    """
    if roc_auc >= 0.90:
        label = "excellent"
    elif roc_auc >= 0.80:
        label = "strong"
    elif roc_auc >= 0.70:
        label = "moderate"
    elif roc_auc >= 0.60:
        label = "weak — only marginally better than chance"
    else:
        label = "negligible — close to random guessing"

    is_reliable = roc_auc >= 0.70
    return label, is_reliable
