import pandas as pd


def add_features(df):
    """Create business-relevant behavioral and value-based features.

    All engineered columns use the original raw column names so that the output
    DataFrame is consistent with what data_preprocessing.py expects as input.

    FIX (redundant raw columns): TotalRevenue, AvgMobileRevenue, and
    AvgFIXRevenue are now dropped once they've been used to build the
    engineered ratio/flag features below. Previously these raw columns were
    left in the frame alongside both Total_Revenue (an exact duplicate of
    TotalRevenue, kept here only because evaluate_model.py / utils.py need a
    revenue column for business reporting after the raw columns are gone)
    and the engineered ratio features that were meant to replace them —
    feeding the model duplicate/multicollinear signal for no benefit. This
    mirrors the intent of notebook step 8.11, which dropped the same columns.
    """

    df = df.copy()

    # Revenue base (kept as Total_Revenue so utils.revenue_at_risk can still
    # reference a revenue column after TotalRevenue itself is dropped below)
    df["Total_Revenue"] = df["TotalRevenue"]

    # Revenue per subscriber (+ 1 avoids division-by-zero)
    df["Revenue_per_Sub"] = df["Total_Revenue"] / (df["Total_SUBs"] + 1)

    # Engagement score: fraction of subscriptions that are active
    df["Engagement_Score"] = df["Active_subscribers"] / (df["Total_SUBs"] + 1)

    # Flag customers with any inactive subscriptions
    df["High_Inactive_Flag"] = (df["Not_Active_subscribers"] > 0).astype(int)

    # Mobile revenue share of total revenue
    df["Mobile_Revenue_Ratio"] = df["AvgMobileRevenue"] / (df["Total_Revenue"] + 1)

    # Fixed-line service indicator (FIX users showed higher loyalty in EDA)
    df["FIX_User"] = (df["AvgFIXRevenue"] > 0).astype(int)

    # Multi-service flag: customer uses both mobile and FIX services
    # (deliberately revenue/service-mix based rather than subscription-count
    # based, so the feature name matches what it actually measures)
    df["Multi_Service"] = (
        (df["AvgMobileRevenue"] > 0) & (df["AvgFIXRevenue"] > 0)
    ).astype(int)

    # High-value customer: top 10 % by total revenue
    revenue_threshold = df["Total_Revenue"].quantile(0.9)
    df["High_Value_Customer"] = (df["Total_Revenue"] >= revenue_threshold).astype(int)

    # Interaction feature: value x engagement
    df["Value_Engagement_Interaction"] = df["Revenue_per_Sub"] * df["Engagement_Score"]

    # Drop raw revenue columns now that they're represented by the engineered
    # ratio/flag features above (and by Total_Revenue, which is kept for
    # business reporting). Prevents duplicate/multicollinear signal reaching
    # the model.
    df = df.drop(columns=["TotalRevenue", "AvgMobileRevenue", "AvgFIXRevenue"])

    return df
