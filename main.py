"""
main.py — Customer Churn Prediction Pipeline
----------------------------------------
Orchestrates the full ML pipeline by importing from src/:

    src/preprocess.py  — load, clean, feature-engineer, encode, split
    src/train.py       — train 4 models with SMOTE, select best
    src/score.py       — score all customers, compute business metrics
    src/display.py     — all terminal output (no plots, no file saves)

Run:
    python main.py
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

# Allow: python main.py from any working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocess import load_and_clean, engineer_features, encode_and_split
from train      import train_all, get_feature_importance
from score      import score_all_customers, revenue_summary
from display    import (
    print_header,
    print_dataset_overview,
    print_model_comparison,
    print_best_model_metrics,
    print_confusion_matrix,
    print_feature_importance,
    print_risk_distribution,
    print_revenue_summary,
    print_sample_scores,
    print_recommendations,
    print_footer,
)

# ── Config ─────────────────────────────────────────────────────────────
RAW_PATH = os.path.join(os.path.dirname(__file__), "data", "raw", "telco_churn.csv")


# ══════════════════════════════════════════════════════════════════════
#  PIPELINE
# ══════════════════════════════════════════════════════════════════════

print_header()

# 1. Load & preprocess
df_raw  = load_and_clean(RAW_PATH)
df_feat = engineer_features(df_raw)

(X_train, X_test, y_train, y_test,
 X_train_sc, X_test_sc,
 scaler, imputer,
 feature_names) = encode_and_split(df_feat)

print_dataset_overview(df_raw, X_train, X_test)

# 2. Train
results, best_name = train_all(X_train_sc, y_train, X_test_sc, y_test)
best = results[best_name]

print_model_comparison(results, best_name)
print_best_model_metrics(best_name, best)
print_confusion_matrix(best)

# 3. Feature importance
fi_df = get_feature_importance(best["model"], feature_names)
print_feature_importance(fi_df, top_n=12)

# 4. Score all customers
# Re-encode full dataset (same steps as encode_and_split, no split)
from sklearn.preprocessing import LabelEncoder
df_enc = engineer_features(load_and_clean(RAW_PATH)).drop(columns=["customerID", "Churn"])
le = LabelEncoder()
for col in df_enc.select_dtypes(include=["object", "category"]).columns:
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))

scored  = score_all_customers(df_raw, best["model"], scaler, imputer, df_enc)
summary = revenue_summary(scored, best["roc_auc"])

# 5. Display results
print_risk_distribution(summary)
print_revenue_summary(summary)
print_sample_scores(scored, n=10)
print_recommendations(summary, fi_df, best_name)
print_footer()
