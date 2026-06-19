import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


# Public API 
def load_and_clean(path: str) -> pd.DataFrame:
    """Read CSV, fix TotalCharges, binarise Churn."""
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Spend rate (handles tenure=0 customers)
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

    # Count of active add-on services
    add_ons = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["num_services"] = sum((df[s] == "Yes").astype(int) for s in add_ons)

    # High-value customer flag (top quartile by monthly charge)
    df["high_value"] = (
        df["MonthlyCharges"] > df["MonthlyCharges"].quantile(0.75)
    ).astype(int)

    # Tenure bucket
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"],
    )

    return df


def encode_and_split(df: pd.DataFrame):
    """
    Label-encode categoricals, impute nulls, scale, split 80/20.

    Returns
    -------
    X_train, X_test, y_train, y_test   — raw (un-scaled) splits
    X_train_sc, X_test_sc              — scaled arrays for models
    scaler, imputer                    — fitted transformers
    feature_names                      — list of column names
    """
    df = df.copy()
    drop_cols = ["customerID", "Churn"]
    X_raw = df.drop(columns=drop_cols)
    y     = df["Churn"]

    # Encode all object / category columns
    le = LabelEncoder()
    for col in X_raw.select_dtypes(include=["object", "category"]).columns:
        X_raw[col] = le.fit_transform(X_raw[col].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()

    X_train_sc = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test_sc  = scaler.transform(imputer.transform(X_test))

    return (
        X_train, X_test, y_train, y_test,
        X_train_sc, X_test_sc,
        scaler, imputer,
        X_raw.columns.tolist(),
    )
