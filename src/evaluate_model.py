import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
from src.utils import churn_risk_distribution, revenue_at_risk, business_summary


def evaluate(X_test, y_test, raw_df):
    """Evaluate model and generate explainability outputs."""
    model = joblib.load("models/churn_random_forest.pkl")

    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)

    risk_dist = churn_risk_distribution(y_prob)
    revenue_risk = revenue_at_risk(
        raw_df.iloc[:len(y_prob)],
        y_prob
    )

    # Save customer-level risk scores
    customer_scores = raw_df.iloc[:len(y_prob)].copy()
    customer_scores["Churn_Probability"] = y_prob

    output_path = "data/processed/churn_risk_scores.csv"
    customer_scores.to_csv(output_path, index=False)

    # Feature importance
    preprocessor = joblib.load("models/preprocessor.pkl")
    feature_names = preprocessor.get_feature_names_out()

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    summary = business_summary(
        roc_auc,
        risk_dist,
        revenue_risk
    )

    return {
        "summary": summary,
        "top_features": feature_importance.head(5),
        "output_path": output_path
    }