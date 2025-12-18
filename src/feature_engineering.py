import pandas as pd

def add_features(df):
    """Create business-relevant behavioral and value-based features."""

    # Total Revenue
    df["Total_Revenue"] = df["TotalRevenue"]

    # Revenue per Subscriber
    df["Revenue_per_Sub"] = df["Total_Revenue"] / (df["Total_SUBs"] + 1)

    # Engagement Score
    df["Engagement_Score"] = df["Active_subscribers"] / (df["Total_SUBs"] + 1)

    # Inactive users flag
    df["High_Inactive_Flag"] = (df["Not_Active_subscribers"] > 0).astype(int)

    # Mobile Revenue Ratio
    df["Mobile_Revenue_Ratio"] = df["AvgMobileRevenue"] / (df["Total_Revenue"] + 1)

    # FIX service usage
    df["FIX_User"] = (df["AvgFIXRevenue"] > 0).astype(int)

    # Multi-service users
    df["Multi_Service"] = (
        (df["AvgMobileRevenue"] > 0) &
        (df["AvgFIXRevenue"] > 0)
    ).astype(int)

    # High value customers (top 10%)
    revenue_threshold = df["Total_Revenue"].quantile(0.9)
    df["High_Value_Customer"] = (
        df["Total_Revenue"] >= revenue_threshold
    ).astype(int)

    # Interaction feature
    df["Value_Engagement_Interaction"] = (
        df["Revenue_per_Sub"] * df["Engagement_Score"]
    )

    return df