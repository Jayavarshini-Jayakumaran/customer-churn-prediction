import pandas as pd
from src.feature_engineering import add_features
from src.data_preprocessing import preprocess_data
from src.train_model import train
from src.evaluate_model import evaluate
from src.utils import interpret_auc


# Load raw data
raw_df = pd.read_csv("data/raw/customer_churn_raw.csv")

# FIX (column-name mismatch): every notebook strips whitespace AND replaces
# remaining spaces with underscores (e.g. "Total SUBs" -> "Total_SUBs"). This
# script previously only stripped, so if the raw CSV headers contain spaces
# instead of underscores, downstream column lookups (e.g. "Total_SUBs" in
# feature_engineering.py) would raise a KeyError. Aligned with the notebooks
# here so both paths parse the same raw file identically.
raw_df.columns = raw_df.columns.str.strip().str.replace(" ", "_")

# Feature engineering
raw_df = add_features(raw_df)

# Preprocessing (scaling + encoding)
X_processed, y, raw_features = preprocess_data(raw_df)

# Train (split FIRST, then SMOTE on train fold only)
model, X_test, y_test = train(X_processed, y)

# Evaluate (model passed explicitly — no hidden file-system load)
results = evaluate(model, X_test, y_test, raw_features)

# Outputs & reporting 
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
    print(f"  {i}. {clean_name}")

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

# FIX (hardcoded interpretation): this block used to always print "Model
# shows strong churn discrimination" no matter what the ROC-AUC actually
# was. interpret_auc() now derives the label — and whether the risk tiers /
# revenue-at-risk figures are trustworthy enough to act on — from the real
# score, so the printed narrative can't silently contradict the metrics
# above it.
label, is_reliable = interpret_auc(summary["roc_auc"])

print("\nInterpretation:")
print(f"- Model shows {label} churn discrimination (ROC–AUC = {summary['roc_auc']})")

if is_reliable:
    print("- High-risk customers should be prioritised for retention")
    print("- Revenue-at-risk helps optimise retention budget allocation")
else:
    print("- Discrimination is too low to act on the risk tiers or revenue-at-risk figures above")
    print("- Treat these outputs as exploratory only until feature engineering or model selection improves ROC–AUC")
