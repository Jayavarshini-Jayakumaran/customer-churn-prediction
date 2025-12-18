import pandas as pd
from src.feature_engineering import add_features
from src.data_preprocessing import preprocess_data
from src.train_model import train
from src.evaluate_model import evaluate


# Load raw data
raw_df = pd.read_csv("data/raw/customer_churn_raw.csv")
raw_df.columns = raw_df.columns.str.strip()

# Feature engineering
raw_df = add_features(raw_df)

# Preprocessing
X_processed, y, raw_features = preprocess_data(raw_df)

# Train model
model, X_test, y_test = train(X_processed, y)

# Evaluate
results = evaluate(X_test, y_test, raw_features)

# OUTPUTS & REPORTING
print("\n📁 Detailed customer-level risk scores saved to:")
print(results["output_path"])

print("\n🔍 Feature Impact Explanation")
print("Top churn drivers:")
for i, row in enumerate(results["top_features"].itertuples(), 1):
    clean_name = (
        row.Feature
        .replace("num__", "")
        .replace("cat__", "")
        .replace("_", " ")
    )
    print(f"{i}. {clean_name}")

summary = results["summary"]

print("\n📊 BUSINESS SUMMARY")
print("-" * 50)
print(f"ROC–AUC Score            : {summary['roc_auc']}")

print("\nChurn Risk Distribution:")
for k, v in summary["risk_distribution"].items():
    print(f"  {k:<25}: {v}%")

print(
    f"\nEstimated Revenue at Risk: "
    f"{summary['revenue_at_risk']:,.2f} monetary units"
)

print("\nInterpretation:")
print(
    "- Model shows strong churn discrimination\n"
    "- High-risk customers should be prioritized for retention\n"
    "- Revenue-at-risk helps optimize retention budget allocation"
)