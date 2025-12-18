import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(df):
    """Clean raw data and apply feature scaling & encoding."""

    # Standardize column names
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    # Remove non-informative identifiers
    df = df.drop(columns=["PID"], errors="ignore")
    df = df.drop(columns=["Suspended_subscribers"], errors="ignore")

    # Handle missing values
    df["Not_Active_subscribers"] = df["Not_Active_subscribers"].fillna(0)
    df = df.dropna(subset=["CRM_PID_Value_Segment", "Billing_ZIP", "ARPU"])

    # Handle datatypes and Fix typographical error
    df["CRM_PID_Value_Segment"] = df["CRM_PID_Value_Segment"].replace({"Sliver": "Silver"})
    df["Billing_ZIP"] = df["Billing_ZIP"].astype(str)

    # Encode target variable
    df["CHURN"] = df["CHURN"].map({"Yes": 1, "No": 0})

    # Split features and target
    X = df.drop("CHURN", axis=1)
    y = df["CHURN"]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns

    # Define preprocessing pipeline
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
    ])

    # Transform features
    X_processed = preprocessor.fit_transform(X)

    # Persist preprocessor for inference & explainability
    joblib.dump(preprocessor, "models/preprocessor.pkl")

    return X_processed, y, X