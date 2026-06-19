import pandas as pd

W = 60

def _rule():
    print("-" * W)

def _section(title: str):
    print()
    print(title.upper())
    _rule()

# Public print functions 
def print_header():
    _rule()
    print("  CUSTOMER CHURN PREDICTION  |  TELCO DATASET")
    _rule()


def print_dataset_overview(df_raw, X_train, X_test):
    _section("Dataset")
    churn_rate = df_raw["Churn"].mean() * 100
    print(f"  Customers     : {len(df_raw):,}")
    print(f"  Churn rate    : {churn_rate:.1f}%")
    print(f"  Train / Test  : {len(X_train):,} / {len(X_test):,}")
    print(f"  Features      : {X_train.shape[1]}")


def print_model_comparison(results: dict, best_name: str):
    _section("Model Comparison")
    print(f"  {'Model':<25}  {'ROC-AUC':>8}  {'F1 (Churn)':>10}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*10}")
    for name, r in results.items():
        f1  = r["report"]["1"]["f1-score"]
        tag = "  *" if name == best_name else ""
        print(f"  {name:<25}  {r['roc_auc']:>8.4f}  {f1:>10.4f}{tag}")
    print(f"\n  * Best model selected")


def print_best_model_metrics(best_name: str, best: dict):
    _section(f"Best Model  —  {best_name}")
    cr = best["report"]
    print(f"  ROC-AUC   : {best['roc_auc']:.4f}")
    print(f"  Accuracy  : {cr['accuracy']*100:.1f}%")
    print()
    print(f"  {'Metric':<12}  {'No Churn':>9}  {'Churn':>9}")
    print(f"  {'-'*12}  {'-'*9}  {'-'*9}")
    for metric in ["precision", "recall", "f1-score"]:
        v0 = cr["0"][metric]
        v1 = cr["1"][metric]
        print(f"  {metric.capitalize():<12}  {v0:>9.3f}  {v1:>9.3f}")
    tn, fp, fn, tp = best["cm"].ravel()
    print(f"\n  Churners caught  : {tp:,} / {tp+fn:,}  ({tp/(tp+fn)*100:.1f}% recall)")
    print(f"  False alarms     : {fp:,}")


def print_feature_importance(fi_df: pd.DataFrame, top_n: int = 10):
    _section(f"Top {top_n} Churn Drivers")
    for i, row in fi_df.head(top_n).iterrows():
        print(f"  {i+1:>2}. {row['feature']:<26}  {row['importance']:.4f}")


def print_risk_distribution(summary: dict):
    _section("Risk Segmentation")
    for tier, pct in summary["dist"].items():
        count = int(summary["total_customers"] * pct / 100)
        print(f"  {str(tier):<20}  {pct:5.1f}%  ({count:,} customers)")


def print_revenue_summary(summary: dict):
    _section("Revenue at Risk")
    tr  = summary["total_rev"]
    rar = summary["rev_at_risk"]
    ann = summary["annual_at_risk"]
    print(f"  Monthly revenue total   : {tr:>12,.0f}")
    print(f"  High-risk monthly       : {rar:>12,.0f}  ({rar/tr*100:.1f}%)")
    print(f"  Annualised at risk      : {ann:>12,.0f}")
    print(f"  High-risk customers     : {summary['high_risk_count']:,} of {summary['total_customers']:,}")


def print_sample_scores(scored: pd.DataFrame, n: int = 10):
    _section(f"Top {n} Highest-Risk Customers")
    print(f"  {'#':<3}  {'Customer ID':<12}  {'Tenure':>6}  {'Contract':<16}  {'Monthly':>8}  {'Churn%':>7}")
    print(f"  {'-'*3}  {'-'*12}  {'-'*6}  {'-'*16}  {'-'*8}  {'-'*7}")
    for idx, row in scored.head(n).iterrows():
        pct = row["churn_probability"] * 100
        print(
            f"  {idx+1:<3}  {row['customerID']:<12}  {int(row['tenure']):>6}  "
            f"{row['Contract']:<16}  {row['MonthlyCharges']:>8.2f}  {pct:>6.1f}%"
        )


def print_recommendations(summary: dict, fi_df: pd.DataFrame, best_name: str):
    _section("Recommendations")
    print(f"  Model         : {best_name} (ROC-AUC {summary['roc_auc']:.3f})")
    print(f"  Action needed : {summary['high_risk_count']:,} customers at high churn risk")
    print(f"  Top driver    : {fi_df['feature'].iloc[0]}")
    print(f"  Annual exposure: {summary['annual_at_risk']:,.0f} monetary units")
    print()
    print("  Suggested actions:")
    print("    - Offer long-term contracts to high-risk customers")
    print("    - Review pricing for customers with high monthly spend")
    print("    - Promote service bundles to increase stickiness")


def print_footer():
    print()
    _rule()
    print("  Pipeline complete.")
    _rule()
    print()
