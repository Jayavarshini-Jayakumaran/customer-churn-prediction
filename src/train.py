import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


# Public API
def train_all(X_train_sc, y_train, X_test_sc, y_test) -> tuple[dict, str]:
    X_res, y_res = SMOTE(random_state=42).fit_resample(X_train_sc, y_train)

    model_zoo = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=0.5, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=4,
            eval_metric="logloss", random_state=42, verbosity=0,
        ),
    }

    results = {}
    for name, model in model_zoo.items():
        model.fit(X_res, y_res)
        proba = model.predict_proba(X_test_sc)[:, 1]
        preds = model.predict(X_test_sc)
        results[name] = {
            "model"        : model,
            "proba"        : proba,
            "preds"        : preds,
            "roc_auc"      : roc_auc_score(y_test, proba),
            "avg_precision": average_precision_score(y_test, proba),
            "report"       : classification_report(y_test, preds, output_dict=True),
            "cm"           : confusion_matrix(y_test, preds),
        }

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    return results, best_name


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        scores = model.feature_importances_
    elif hasattr(model, "coef_"):
        scores = np.abs(model.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    fi = pd.DataFrame({"feature": feature_names, "importance": scores})
    return fi.sort_values("importance", ascending=False).reset_index(drop=True)
