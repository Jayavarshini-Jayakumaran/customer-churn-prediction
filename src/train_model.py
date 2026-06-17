import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


def train(X, y):
    """Train churn prediction model with imbalance handling.

    FIX (SMOTE leakage): train/test split is performed BEFORE SMOTE so that
    synthetic minority samples are generated only from training data and can
    never appear in the held-out test set.  The test set therefore reflects
    the true real-world class distribution.

    Parameters
    ----------
    X : np.ndarray  — dense feature matrix from preprocess_data()
    y : pd.Series   — binary target (1 = churn)

    Returns
    -------
    model   : fitted RandomForestClassifier
    X_test  : np.ndarray  (original, no synthetic rows)
    y_test  : pd.Series
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
        n_estimators=200,
        max_depth=10,
        min_samples_split=30,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train_res, y_train_res)

    joblib.dump(model, "models/churn_random_forest.pkl")

    return model, X_test, y_test
