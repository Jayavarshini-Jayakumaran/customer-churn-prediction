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
- **Class imbalance handling**: SMOTE

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
├── src/
├── models/
├── reports/
├── main.py
├── requirements.txt
└── README.md
```

## 🚀 How to Run the Project
``` bash
git clone https://github.com/Jayavarshini-Jayakumaran/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## 📄 License
This project is licensed under the [MIT License](LICENSE).

<p align="center"><b>Finish what you started 💻 </b></p>
