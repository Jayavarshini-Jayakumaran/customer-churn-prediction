"""
Run this from your project root:
    python diagnose.py

It prints exactly what each stage of the pipeline produces so we can
see where the index / alignment problem actually is.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

# ── 1. Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv("data/raw/customer_churn_raw.csv")
df.columns = df.columns.str.strip().str.replace(" ", "_")
print(f"Raw shape      : {df.shape}")
print(f"CHURN values   : {df['CHURN'].value_counts().to_dict()}")

# ── 2. Minimal clean (no src/ imports — standalone) ──────────────────────────
DROP = ["PID", "Suspended_subscribers", "Billing_ZIP", "KA_name",
        "TotalRevenue", "AvgMobileRevenue", "AvgFIXRevenue"]

df = df.drop(columns=DROP, errors="ignore")
df["Not_Active_subscribers"] = df["Not_Active_subscribers"].fillna(0)
df = df.dropna(subset=["CRM_PID_Value_Segment", "ARPU"])
df["CRM_PID_Value_Segment"] = df["CRM_PID_Value_Segment"].replace({"Sliver": "Silver"})
for col in ["CRM_PID_Value_Segment", "EffectiveSegment"]:
    if col in df.columns:
        rare = df[col].value_counts(normalize=True)
        df[col] = df[col].replace(rare[rare < 0.02].index, "Other")
df["CHURN"] = df["CHURN"].map({"Yes": 1, "No": 0})

# ── 3. Engineer features ──────────────────────────────────────────────────────
df["Engagement_Score"]           = df["Active_subscribers"] / (df["Total_SUBs"] + 1)
df["High_Inactive_Flag"]         = (df["Not_Active_subscribers"] > 0).astype(int)
df["FIX_User"]                   = 0   # AvgFIXRevenue already dropped
df["Multi_Service"]              = 0   # same

print(f"\nAfter clean    : {df.shape}")
print(f"CHURN balance  : {df['CHURN'].value_counts().to_dict()}")

# ── 4. Split X / y and RESET INDEX ──────────────────────────────────────────
X = df.drop("CHURN", axis=1).reset_index(drop=True)
y = df["CHURN"].reset_index(drop=True)
print(f"\ny index type   : {type(y.index)}")
print(f"y index sample : {y.index[:5].tolist()}")
print(f"y dtype        : {y.dtype}")

# ── 5. Encode ─────────────────────────────────────────────────────────────────
num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()
print(f"\nnum_cols ({len(num_cols)}): {num_cols}")
print(f"cat_cols ({len(cat_cols)}): {cat_cols}")

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_cols),
])
X_proc = pre.fit_transform(X)
print(f"\nX_proc shape   : {X_proc.shape}  dtype={X_proc.dtype}")

# ── 6. Split ──────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_proc, y, test_size=0.2, stratify=y, random_state=42)
print(f"\nTrain shape    : {X_train.shape}  churn={y_train.sum()}/{len(y_train)}")
print(f"Test shape     : {X_test.shape}   churn={y_test.sum()}/{len(y_test)}")
print(f"y_test index sample: {y_test.index[:5].tolist()}")

# ── 7. SMOTE ──────────────────────────────────────────────────────────────────
smote = SMOTE(random_state=42)
X_tr, y_tr = smote.fit_resample(X_train, y_train)
print(f"\nAfter SMOTE    : {X_tr.shape}  balance={pd.Series(y_tr).value_counts().to_dict()}")

# ── 8. Train ──────────────────────────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=300, max_depth=15, min_samples_split=10,
    min_samples_leaf=4, class_weight="balanced",
    oob_score=True, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
print(f"\nOOB score      : {rf.oob_score_:.4f}")

# ── 9. Evaluate — three ways to catch any index bug ──────────────────────────
y_prob = rf.predict_proba(X_test)[:, 1]

# a) Series vs array — the potentially buggy path
auc_a = roc_auc_score(y_test, y_prob)

# b) numpy values vs array — bypasses any pandas index alignment
auc_b = roc_auc_score(y_test.values, y_prob)

# c) reset index explicitly
auc_c = roc_auc_score(y_test.reset_index(drop=True), y_prob)

print(f"\nROC-AUC (Series vs array)         : {auc_a:.4f}")
print(f"ROC-AUC (numpy .values vs array)  : {auc_b:.4f}")
print(f"ROC-AUC (reset_index vs array)    : {auc_c:.4f}")

if abs(auc_a - auc_b) > 0.01:
    print("\n⚠  INDEX MISALIGNMENT CONFIRMED — a vs b differ")
else:
    print("\n✓  Index alignment is fine — AUC values match")

print("\nDone.")
