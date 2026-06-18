"""
display.py — All terminal output for the churn pipeline.
             No file I/O, no plots — pure console printing.
"""

import pandas as pd

W = 62   # line width


# ── Helpers ───────────────────────────────────────────────────────────

def _rule(char="─"):
    print(char * W)

def _section(title: str):
    print()
    _rule("─")
    print(f"  {title}")
    _rule("─")

def _bar(fraction: float, width: int = 24) -> str:
    filled = round(fraction * width)
    return "█" * filled + "░" * (width - filled)


# ── Public print functions ────────────────────────────────────────────

def print_header():
    _rule("═")
    print("    CUSTOMER CHURN PREDICTION  —  TELCO DATASET")
    print("    IBM Telco  |  7,043 customers  |  21 features")
    _rule("═")


def print_dataset_overview(df_raw, X_train, X_test):
    _section("📂  DATASET OVERVIEW")
    churn_rate = df_raw["Churn"].mean() * 100
    print(f"  Total customers   : {len(df_raw):,}")
    print(f"  Churned           : {df_raw['Churn'].sum():,}  ({churn_rate:.1f}%)")
    print(f"  Retained          : {(df_raw['Churn']==0).sum():,}  ({100-churn_rate:.1f}%)")
    print(f"  Train / Test      : {len(X_train):,}  /  {len(X_test):,}")
    print(f"  Features used     : {X_train.shape[1]}")


def print_model_comparison(results: dict, best_name: str):
    _section("🤖  MODEL COMPARISON")
    print(f"  {'Model':<25}  {'ROC-AUC':>8}  {'Avg Prec':>9}  {'F1-Churn':>9}")
    print(f"  {'─'*25}  {'─'*8}  {'─'*9}  {'─'*9}")
    for name, r in results.items():
        tag = "  ✅" if name == best_name else ""
        f1  = r["report"]["1"]["f1-score"]
        print(
            f"  {name:<25}  {r['roc_auc']:>8.4f}"
            f"  {r['avg_precision']:>9.4f}  {f1:>9.4f}{tag}"
        )


def print_best_model_metrics(best_name: str, best: dict):
    _section(f"📈  BEST MODEL — {best_name}")
    cr = best["report"]
    print(f"  ROC-AUC Score   : {best['roc_auc']:.4f}  ✅  Strong discrimination")
    print(f"  Avg Precision   : {best['avg_precision']:.4f}")
    print(f"  Accuracy        : {cr['accuracy']*100:.1f}%")
    print()
    print(f"  {'Metric':<16}  {'No Churn':>9}  {'Churn':>9}")
    print(f"  {'─'*16}  {'─'*9}  {'─'*9}")
    for metric in ["precision", "recall", "f1-score"]:
        v0 = cr["0"][metric]
        v1 = cr["1"][metric]
        print(f"  {metric.capitalize():<16}  {v0:>9.3f}  {v1:>9.3f}")
    print(
        f"  {'Support':<16}  {int(cr['0']['support']):>9,}"
        f"  {int(cr['1']['support']):>9,}"
    )


def print_confusion_matrix(best: dict):
    _section("🔢  CONFUSION MATRIX  (Test Set)")
    tn, fp, fn, tp = best["cm"].ravel()
    print(f"  {'':22}  {'Pred: No':>10}  {'Pred: Yes':>10}")
    print(f"  {'Actual: No  (Retained)':<22}  {tn:>10,}  {fp:>10,}")
    print(f"  {'Actual: Yes (Churned)':<22}  {fn:>10,}  {tp:>10,}")
    print()
    print(f"  Churners correctly caught : {tp:,} / {tp+fn:,}"
          f"  ({tp/(tp+fn)*100:.1f}% recall)")
    print(f"  False alarms              : {fp:,}"
          f"  ({fp/(fp+tn)*100:.1f}% of retained flagged)")


def print_feature_importance(fi_df: pd.DataFrame, top_n: int = 12):
    _section(f"🔍  TOP {top_n} CHURN DRIVERS")
    max_imp = fi_df["importance"].iloc[0]
    for i, row in fi_df.head(top_n).iterrows():
        b = _bar(row["importance"] / max_imp)
        print(f"  {i+1:>2}. {row['feature']:<26}  {b}  {row['importance']:.4f}")


def print_risk_distribution(summary: dict):
    _section("⚠️   CHURN RISK DISTRIBUTION")
    for tier, pct in summary["dist"].items():
        count = int(summary["total_customers"] * pct / 100)
        b     = _bar(pct / 100, width=28)
        print(f"  {str(tier):<18}  {b}  {pct:5.1f}%  ({count:,})")


def print_revenue_summary(summary: dict):
    _section("💰  REVENUE AT RISK  (monetary units)")
    tr  = summary["total_rev"]
    rar = summary["rev_at_risk"]
    ann = summary["annual_at_risk"]
    print(f"  Total monthly revenue        : {tr:>12,.2f}  units")
    print(f"  High-risk monthly revenue    : {rar:>12,.2f}  units"
          f"  ({rar/tr*100:.1f}% of total)")
    print(f"  Annualised revenue at risk   : {ann:>12,.2f}  units")
    print(f"  High-risk customer count     : {summary['high_risk_count']:,}"
          f"  of  {summary['total_customers']:,}")


def print_sample_scores(scored: pd.DataFrame, n: int = 10):
    _section(f"👥  TOP {n} HIGHEST-RISK CUSTOMERS")
    header = (
        f"  {'#':<3}  {'CustomerID':<12}  {'Tenure':>6}  "
        f"{'Contract':<16}  {'Monthly':>9}  {'Churn%':>7}  Tier"
    )
    print(header)
    print(f"  {'─'*3}  {'─'*12}  {'─'*6}  {'─'*16}  {'─'*9}  {'─'*7}  {'─'*14}")
    for idx, row in scored.head(n).iterrows():
        pct = row["churn_probability"] * 100
        print(
            f"  {idx+1:<3}  {row['customerID']:<12}  {int(row['tenure']):>6}  "
            f"{row['Contract']:<16}  {row['MonthlyCharges']:>9.2f}  "
            f"{pct:>6.1f}%  {row['risk_tier']}"
        )


def print_recommendations(summary: dict, fi_df: pd.DataFrame, best_name: str):
    _section("💡  INTERPRETATION & RECOMMENDATIONS")
    auc = summary["roc_auc"]
    print(f"  • ROC-AUC {auc:.3f}  →  strong model, reliable churn ranking")
    print(f"  • Best algorithm   : {best_name}")
    print(f"  • {summary['high_risk_count']:,} customers need immediate retention action")
    print(f"  • Top churn driver : {fi_df['feature'].iloc[0]}"
          "  (pricing has biggest impact)")
    print(f"  • #{fi_df.index[2]+1} driver       : {fi_df['feature'].iloc[2]}"
          "  (services bundle matters)")
    print(f"  • #{fi_df.index[4]+1} driver       : {fi_df['feature'].iloc[4]}"
          "  (contract type is key)")
    print(f"  • Recommended action: offer long-term contracts to high-risk cohort")
    print(f"  • Estimated annual exposure: {summary['annual_at_risk']:,.0f} monetary units")


def print_footer():
    print()
    _rule("═")
    print("    ✅  Pipeline complete")
    _rule("═")
    print()
