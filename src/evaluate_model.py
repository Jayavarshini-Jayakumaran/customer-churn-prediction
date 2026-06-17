import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

from src.utils import churn_risk_distribution, revenue_at_risk, business_summary


def evaluate(model, X_test, y_test, raw_features):
    """Evaluate model and generate business outputs.

    FIX (hidden file-system dependency): the trained model is now accepted as
    an explicit parameter instead of being silently loaded from disk inside
    this function.  This makes the dependency visible at the call site (main.py)
    and prevents stale-model surprises when the file has been overwritten.

    Parameters
    ----------
    model        : fitted RandomForestClassifier returned by train()
    X_test       : np.ndarray — held-out feature matrix (no synthetic rows)
    y_test       : pd.Series  — true labels for the test set
    raw_features : pd.DataFrame — pre-encoding feature frame from preprocess_data(),
                   used to attach original column values to the risk-score output

    Returns
    -------
    dict with keys: summary, top_features, output_path
    """

    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)

    risk_dist    = churn_risk_distribution(y_prob)
    revenue_risk = revenue_at_risk(raw_features.iloc[:len(y_prob)], y_prob)

    # Customer-level risk scores
    customer_scores = raw_features.iloc[:len(y_prob)].copy()
    customer_scores["Churn_Probability"] = y_prob

    output_path = "data/processed/churn_risk_scores.csv"
    customer_scores.to_csv(output_path, index=False)

    # Feature importance
    preprocessor  = joblib.load("models/preprocessor.pkl")
    feature_names  = preprocessor.get_feature_names_out()

    feature_importance = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": model.feature_importances_,
    }).sort_values(by="Importance", ascending=False)

    feature_importance.to_csv("data/processed/feature_importance.csv", index=False)

    summary = business_summary(roc_auc, risk_dist, revenue_risk)

    return {
        "summary":      summary,
        "top_features": feature_importance.head(5),
        "output_path":  output_path,
    }
