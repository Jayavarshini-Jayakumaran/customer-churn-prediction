import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def preprocess_data(df):
    """Clean raw data and apply feature scaling & encoding.

    Returns
    -------
    X_processed : np.ndarray  (dense)
    y           : pd.Series
    raw_features: pd.DataFrame  (pre-encoding feature frame, for scoring output)
    """

    df = df.copy()

    # Standardize column names
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    # Remove non-informative identifiers and high-cardinality columns.
    # Billing_ZIP encodes hundreds of unique values → one-hot explodes to ~400 columns,
    # consuming ~22 MB just for SMOTE's nearest-neighbour arrays and adding no churn signal
    # (confirmed near-zero correlation in EDA).  KA_name is a free-text account name with
    # the same problem.  Both are dropped before encoding.
    df = df.drop(
        columns=["PID", "Suspended_subscribers", "Billing_ZIP", "KA_name"],
        errors="ignore",
    )

    # Handle missing values
    df["Not_Active_subscribers"] = df["Not_Active_subscribers"].fillna(0)
    df = df.dropna(subset=["CRM_PID_Value_Segment", "ARPU"])

    # Fix typographical error
    df["CRM_PID_Value_Segment"] = df["CRM_PID_Value_Segment"].replace({"Sliver": "Silver"})

    # Merge rare categories (< 2 % of data) into OtherSegment
    for col in ["CRM_PID_Value_Segment", "EffectiveSegment"]:
        if col in df.columns:
            counts = df[col].value_counts(normalize=True)
            rare = counts[counts < 0.02].index
            df[col] = df[col].replace(rare, "OtherSegment")

    # Encode target variable
    df["CHURN"] = df["CHURN"].map({"Yes": 1, "No": 0})

    # Split features and target
    X = df.drop("CHURN", axis=1)
    y = df["CHURN"]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_cols),
    ])

    # FIX (sparse matrix): fit_transform → always dense ndarray from this point on.
    # sparse_output=False on OneHotEncoder means ColumnTransformer returns dense directly,
    # so no .toarray() call is needed anywhere downstream.
    X_processed = preprocessor.fit_transform(X)   # shape: (n, p), dtype: float64

    joblib.dump(preprocessor, "models/preprocessor.pkl")

    return X_processed, y, X
