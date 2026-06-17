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

    FIX (row misalignment): X_test / y_test come out of train_test_split, which
    shuffles rows, so they no longer line up with the first N rows of
    raw_features in original file order. raw_features is now sliced with
    `.loc[y_test.index]` so every churn probability is attached to the correct
    customer. Without this fix, churn_risk_scores.csv and the revenue-at-risk
    figure were silently computed against the wrong set of customers — the
    ROC-AUC score itself was unaffected (it only ever used y_test/y_prob, which
    were already correctly paired), but the two main business-facing outputs
    were not.

    Parameters
    ----------
    model        : fitted RandomForestClassifier returned by train()
    X_test       : np.ndarray — held-out feature matrix (no synthetic rows)
    y_test       : pd.Series  — true labels for the test set. Must retain the
                   original DataFrame index assigned in preprocess_data(), since
                   that index is what's used below to re-align raw_features.
    raw_features : pd.DataFrame — pre-encoding feature frame from preprocess_data(),
                   used to attach original column values to the risk-score output.
                   Shares the same index as the y returned by preprocess_data().

    Returns
    -------
    dict with keys: summary, top_features, output_path
    """

    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)

    # Align raw (pre-encoding) feature rows to the actual test-set customers
    # by index, instead of assuming the first len(y_prob) rows in file order
    # correspond to the test set (they don't, post train_test_split shuffle).
    test_rows = raw_features.loc[y_test.index].copy()

    risk_dist    = churn_risk_distribution(y_prob)
    revenue_risk = revenue_at_risk(test_rows, y_prob)

    # Customer-level risk scores
    customer_scores = test_rows.copy()
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
