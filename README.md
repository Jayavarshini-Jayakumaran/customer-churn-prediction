# 📊 Customer Churn Prediction – Telecom Business Clients

## 📌 Overview
Customer churn represents a major revenue risk for telecom companies, particularly for **SME and enterprise customers**.  
This project delivers an **end-to-end machine learning solution** to predict churn, quantify revenue at risk, and support data-driven retention strategies.

## 🎯 Business Objectives
- Identify customers with high churn probability
- Understand key drivers behind churn
- Segment customers by churn risk level
- Estimate potential revenue loss for proactive decision-making

## 📂 Dataset Information
- **Industry**: Telecom (SME & Enterprise clients)
- **Total Records**: 8,454
- **Target Variable**: `CHURN` (Yes / No)
- **Source**: Mendeley Data (CC BY 4.0)
- **Note**: Revenue values are represented in relative monetary units

## 🔍 Feature Engineering
Key engineered features include:
- Engagement Score
- Revenue per Subscriber
- Mobile Revenue Ratio
- Multi-Service Indicator
- High-Value Customer Flag
- Revenue × Engagement Interaction

These features convert raw operational data into meaningful business indicators.

## ⚙️ Data Preprocessing
- Missing value treatment
- One-Hot Encoding for categorical features
- Standard Scaling for numerical features
- Reproducible preprocessing pipeline

## 🤖 Model Development
- **Models evaluated**: Logistic Regression, Decision Tree, Random Forest
- **Final Model Selected**: Random Forest
- **Class imbalance handling**: SMOTE, applied to the training fold only — **after** the train/test split, to prevent synthetic rows from leaking into evaluation

## 🔥 Churn Risk Segmentation
Customers are categorized into:
- **High Risk** – Immediate retention actions recommended
- **Medium Risk** – Engagement and monitoring required
- **Low Risk** – Minimal intervention needed

## 💰 Revenue at Risk Analysis
The model estimates potential revenue loss from high-risk customers, enabling:
- Targeted retention campaigns
- Optimized budget allocation
- Business-aligned decision-making

## 🔄 Model Monitoring & Retraining Strategy
- Model performance should be monitored regularly using ROC–AUC and churn distribution.
- The model should be retrained **quarterly** or whenever significant shifts in customer churn behavior are observed.

## 📤 Model Output
Customer-level churn probabilities are saved as:

`data/processed/churn_risk_scores.csv`

## 🏗 Project Structure
```text
customer-churn-prediction/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_exploration.ipynb         # EDA
│   ├── 02_preprocessing.ipynb       # cleaning + encoding, via src/ functions
│   ├── 03_feature_engineering.ipynb # rationale & validation for engineered features
│   ├── 04_model_training.ipynb      # model comparison + production training
│   └── 05_evaluation.ipynb          # diagnostics + business summary
├── src/
├── models/
├── reports/
├── main.py
├── requirements.txt
└── README.md
```

## 🛠 Notes on Pipeline Fixes
A few issues were identified and corrected to keep the notebooks and `src/`
pipeline in sync and to make the saved outputs trustworthy:

- **Train/test leakage:** SMOTE is now applied only to the training fold,
  *after* the train/test split (`src/train_model.py`), instead of before
  splitting. As a result, current ROC-AUC / recall / accuracy figures will
  be lower than any numbers quoted in older reports — that's the expected,
  leakage-free result, not a regression.
- **Customer-level output alignment:** `src/evaluate_model.py` now matches
  each predicted churn probability to the correct customer via the test
  set's original index, rather than assuming row order. This was silently
  wrong before and affected both `data/processed/churn_risk_scores.csv`
  and the revenue-at-risk figure (the ROC-AUC score itself was unaffected).
- **Redundant features:** `src/feature_engineering.py` now drops the raw
  `TotalRevenue`, `AvgMobileRevenue`, and `AvgFIXRevenue` columns once
  they've been used to build the engineered ratio/flag features, avoiding
  duplicate/multicollinear signal going into the model.
- **Notebook ↔ src alignment:** notebooks `02`–`05` now import and call
  the real functions in `src/` directly instead of re-implementing the
  logic inline, so they can't silently drift out of sync with what
  `main.py` actually runs. `03_feature_engineering.ipynb` (previously an
  empty file) is now a feature-rationale/validation notebook, and the old
  `04_evaluation.ipynb` — which referenced a non-existent
  `03_modeling_training.ipynb` — has been renamed to `05_evaluation.ipynb`
  and corrected.

## 🚀 How to Run the Project
``` bash
git clone https://github.com/Jayavarshini-Jayakumaran/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

---

🙌 **Connect** — [LinkedIn: Jayavarshini Jayakumaran](https://www.linkedin.com/in/jayavarshini-jayakumaran)

📄 **License** — [MIT](LICENSE)

<p align="center"><b>Finish what you started 💻 </b></p>
