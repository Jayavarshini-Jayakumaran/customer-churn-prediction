# Customer Churn Prediction - Telco

End-to-end ML pipeline to predict customer churn, segment by risk tier, and estimate revenue at risk.

---

## Dataset

IBM Telco Customer Churn — 7,043 customers, 21 features, 26.5% churn rate.  
Source: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Project Structure

```
customer-churn-prediction/
├── main.py
├── requirements.txt
├── data/
│   └── raw/
│       └── telco_churn.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modelling.ipynb
└── src/
    ├── preprocess.py   # load, clean, feature engineering, encode, split
    ├── train.py        # train 4 models with SMOTE, select best by ROC-AUC
    ├── score.py        # score all customers, risk tiers, revenue summary
    └── display.py      # terminal output
```

The `notebooks/` folder documents the experimental work that preceded the pipeline.  
Each notebook is self-contained and does not import from `src/`.

---

## How to Run

```bash
git clone https://github.com/Jayavarshini-Jayakumaran/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv venv && venv\Scripts\activate   # Windows
pip install -r requirements.txt
python main.py
```

---

## Results

| Model | ROC-AUC | F1 (Churn) |
|---|---|---|
| Logistic Regression ✅ | 0.8445 | 0.619 |
| Random Forest | 0.8407 | 0.632 |
| Gradient Boosting | 0.8326 | 0.588 |
| XGBoost | 0.8242 | 0.573 |

- **2,271 high-risk customers** (>60% churn probability)
- **Annualised revenue at risk: ~2.1M monetary units**
- Top churn driver: `avg_monthly_spend`

---
📧 **Email** - [jayavarshinijayakumaran11@gmail.com](mailto:jayavarshinijayakumaran11@gmail.com)

🙌 **Connect** - [LinkedIn: Jayavarshini Jayakumaran](https://www.linkedin.com/in/jayavarshini-jayakumaran)

📄 **License** - [MIT](LICENSE)

<p align="center"><b>Finish what you started 💻</b></p>
