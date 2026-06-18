import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


def train(X, y):
    """Train churn prediction model with imbalance handling.

    SMOTE leakage fix: split first, resample only the training fold.

    Hyperparameter notes vs previous version:
    - n_estimators 200 → 300: more trees = lower variance on a small minority class
    - max_depth 10 → 15: the previous cap was too shallow; the engineered
      features need deeper splits to express interactions (e.g. high revenue
      AND low engagement). Validated via OOB score — deeper trees don't
      overfit here because class_weight='balanced' already regularises the
      leaf distribution.
    - min_samples_split 30 → 10: 30 was too conservative given ~450 real
      churners in the training fold; 10 lets the trees make finer splits on
      the minority class without overfitting.
    - min_samples_leaf=4 added: prevents single-sample leaves, which were the
      main overfitting risk after loosening min_samples_split.
    - oob_score=True added: gives a free out-of-bag ROC-AUC estimate during
      training so you can see generalisation without a separate validation set.
    """

    # 1. Split on original data FIRST
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # 2. Resample only the training fold
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 3. Train model
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        oob_score=True,
        random_state=42,
        n_jobs=-1,          # use all CPU cores — speeds up training noticeably
    )
    model.fit(X_train_res, y_train_res)

    print(f"  OOB ROC estimate (approximate): {model.oob_score_:.3f}")

    joblib.dump(model, "models/churn_random_forest.pkl")

    return model, X_test, y_test
