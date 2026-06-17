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
