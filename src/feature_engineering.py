import pandas as pd


def add_features(df):
    """Create business-relevant behavioral and value-based features.

    Raw revenue columns (TotalRevenue, AvgMobileRevenue, AvgFIXRevenue) are
    dropped after the engineered features are derived from them, so the model
    receives ratio/flag features instead of duplicate raw signal.
    Total_Revenue is kept as a reporting column for revenue_at_risk() in utils.
    """

    df = df.copy()

    # Keep a copy for business reporting (utils.revenue_at_risk needs it)
    df["Total_Revenue"] = df["TotalRevenue"]

    # Revenue per subscriber — normalises spend by subscription count
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
    df["Multi_Service"] = (
        (df["AvgMobileRevenue"] > 0) & (df["AvgFIXRevenue"] > 0)
    ).astype(int)

    # High-value customer: top 10% by total revenue
    revenue_threshold = df["Total_Revenue"].quantile(0.9)
    df["High_Value_Customer"] = (df["Total_Revenue"] >= revenue_threshold).astype(int)

    # Interaction: high revenue × low engagement = high-risk profile
    df["Value_Engagement_Interaction"] = df["Revenue_per_Sub"] * df["Engagement_Score"]

    # Drop raw revenue columns — they are now represented by the engineered
    # features above. Leaving them in feeds the model duplicate/multicollinear
    # signal and crowds out the ratio features.
    df = df.drop(columns=["TotalRevenue", "AvgMobileRevenue", "AvgFIXRevenue"])

    return df
