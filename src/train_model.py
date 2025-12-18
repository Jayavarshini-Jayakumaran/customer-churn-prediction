import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

def train(X, y):
    """Train churn prediction model with imbalance handling."""

    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Train-validation split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res,
        test_size=0.2,
        stratify=y_res,
        random_state=42
    )

    # Train-validation split
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=30,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    # Persist trained model
    joblib.dump(model, "models/churn_random_forest.pkl")

    return model, X_test, y_test