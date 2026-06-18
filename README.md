# 📊 Customer Churn Prediction – Telecom

## 📌 Overview

Customer churn represents a major revenue risk for telecom companies.  
This project delivers an **end-to-end machine learning pipeline** to predict churn, quantify revenue at risk, and support data-driven retention strategies — all outputted as a clean, structured **terminal business summary**.

---

## 🎯 Business Objectives

- Identify customers with high churn probability
- Understand the key drivers behind churn decisions
- Segment all customers into Low, Medium, and High risk tiers
- Estimate potential revenue loss to enable proactive retention

---

## 📂 Dataset Information

| Property | Value |
|---|---|
| Industry | Telecom |
| Source | IBM Telco Customer Churn (GitHub / Kaggle) |
| Total Records | 7,043 customers |
| Features | 21 raw + 4 engineered = 25 total |
| Target Variable | `Churn` (Yes / No) |
| Churn Rate | 26.5% |
| License | Public domain |

> Revenue values are represented in relative **monetary units**.

---

## 🔍 Feature Engineering

Key engineered features that convert raw operational data into business indicators:

| Feature | Description |
|---|---|
| `avg_monthly_spend` | TotalCharges / (tenure + 1) — normalised spend rate |
| `num_services` | Count of active add-on services subscribed (0–8) |
| `high_value` | 1 if MonthlyCharges exceeds the 75th percentile |
| `tenure_group` | Tenure bucketed into 0-1yr, 1-2yr, 2-4yr, 4-6yr bands |

---

## ⚙️ Data Preprocessing

- Missing value imputation using median strategy (`SimpleImputer`)
- Label encoding for all categorical features (`LabelEncoder`)
- Standard scaling for numerical features (`StandardScaler`)
- Stratified 80/20 train/test split to preserve class distribution
- Reproducible preprocessing pipeline with fitted transformer objects

---

## 🤖 Model Development

**Models evaluated:**

| Model | ROC-AUC | Avg Precision | F1 (Churn) |
|---|---|---|---|
| Logistic Regression ✅ | 0.8445 | 0.6564 | 0.6196 |
| Random Forest | 0.8409 | 0.6418 | 0.6341 |
| Gradient Boosting | 0.8324 | 0.6355 | 0.5808 |
| XGBoost | 0.8256 | 0.6339 | 0.5850 |

**Best model selected**: Logistic Regression (highest ROC-AUC)

**Class imbalance handling**: SMOTE applied to the training fold only — **after** the train/test split — to prevent synthetic rows from leaking into evaluation.

---

## 🔥 Churn Risk Segmentation

Customers are categorised into three tiers based on predicted churn probability:

| Tier | Probability | Customers | Action |
|---|---|---|---|
| 🟢 Low Risk | < 30% | 3,081 (43.7%) | Minimal intervention |
| 🟡 Medium Risk | 30% – 60% | 1,688 (24.0%) | Engagement and monitoring |
| 🔴 High Risk | > 60% | 2,274 (32.3%) | Immediate retention action |

---

## 💰 Revenue at Risk Analysis

| Metric | Value |
|---|---|
| Total monthly revenue | 456,116.60 monetary units |
| High-risk monthly revenue | 178,077.80 monetary units (39.0%) |
| Annualised revenue at risk | 2,136,933.60 monetary units |
| High-risk customers | 2,274 of 7,043 |

---

## 🔄 Model Monitoring & Retraining Strategy

- Monitor model performance regularly using **ROC-AUC** and churn distribution shifts
- Retrain the model **quarterly** or whenever significant behavioural shifts are observed
- Watch for data drift in top churn drivers: `MonthlyCharges`, `tenure`, `Contract` type

---

## 📤 Model Output

All results are printed directly to the terminal as a structured business summary.  
No files are saved. No plots are generated. The output includes:

- Dataset overview
- Model comparison table
- Best model metrics and confusion matrix
- Top 12 churn drivers with inline bar chart
- Risk distribution by tier
- Revenue at risk breakdown
- Top 10 highest-risk customers
- Interpretation and recommendations

---

## 🏗 Project Structure

```text
customer-churn-clean/
├── main.py                  ← entry point — orchestrates the full pipeline
├── requirements.txt
├── data/
│   └── raw/
│       └── telco_churn.csv
└── src/
    ├── preprocess.py        ← load, clean, engineer features, encode, split
    ├── train.py             ← train 4 models with SMOTE, select best by ROC-AUC
    ├── score.py             ← score all customers, assign risk tiers, revenue summary
    └── display.py           ← all terminal output sections (no file I/O, no plots)
```

`main.py` is the orchestrator only — it imports and calls one function per task from each module. All logic lives in `src/`.

---

## 🚀 How to Run

```bash
git clone https://github.com/Jayavarshini-Jayakumaran/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
python main.py
```

---

🙌 **Connect** — [LinkedIn: Jayavarshini Jayakumaran](https://www.linkedin.com/in/jayavarshini-jayakumaran)  
📄 **License** — [MIT](LICENSE)

<p align="center"><b>Finish what you started 💻</b></p>
