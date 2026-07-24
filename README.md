# 📊 Telco Customer Churn: Predictive Analysis & Strategic Insights

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview

This project provides a professional-grade end-to-end data science solution for the **Customer Churn** problem in the telecommunications industry. Using a dataset of over 7,000 customers, we develop a predictive framework to identify high-risk individuals and provide actionable strategic recommendations to increase retention and lifetime value.

### 📊 Data Source
The dataset used in this project is the **Telco Customer Churn** dataset, originally provided by IBM and hosted on **Kaggle**. You can find the original data and documentation here:
👉 [Kaggle - Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The analysis covers the entire data lifecycle: from **Exploratory Data Analysis (EDA)** and **Feature Engineering** to **Advanced Predictive Modeling** and **Model Interpretability**.

## 🚀 Key Methodologies

To ensure a robust and professional solution, this project implements two critical data science techniques:

### ⚖️ Handling Imbalance with SMOTE
Real-world churn datasets are often highly imbalanced (fewer churners than non-churners). A model trained on such data might achieve high accuracy by simply predicting "No Churn" for everyone, which is useless for business.
- **Technique:** **SMOTE (Synthetic Minority Over-sampling Technique)**.
- **Implementation:** Instead of duplicating data, SMOTE creates synthetic, realistic examples of the minority class (churners) by interpolating between existing points.
- **Result:** This balances the training set, allowing the model to learn the actual patterns of churn and significantly improving **Recall** (the ability to find actual churners).

### 🧠 Model Interpretability with SHAP
High-performance models like Gradient Boosting are often seen as "black boxes." For a business to act, it needs to know *why* a customer is at risk.
- **Technique:** **SHAP (SHapley Additive exPlanations)**.
- **Implementation:** Based on Game Theory, SHAP calculates the exact contribution of each feature to every single prediction.
- **Result:** We transform complex models into **transparent advisory tools**. We can identify global drivers (e.g., "Monthly contracts increase churn risk") and local reasons (e.g., "This specific customer is at risk due to high fiber optic costs").

## 📂 Project Structure

```text
telco_churn_project/
├── .github/workflows/      # CI/CD Automation (GitHub Actions)
├── data/                   # Raw and processed datasets
├── notebooks/              # Comprehensive Jupyter Notebooks
├── visualizations/         # High-resolution SHAP and EDA plots
├── reports/                # Performance summaries and logs
├── scripts/                # Modular Python automation scripts
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## 🛠️ Technologies Used

- **Language:** Python 3.11+
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-Learn, XGBoost, LightGBM, Imbalanced-learn
- **Interpretability:** SHAP
- **Automation:** GitHub Actions

## 📈 Principal Insights

1.  **Contract Type:** Month-to-month contracts are the strongest predictor of churn.
2.  **Tenure:** New customers are significantly more likely to leave within the first 6 months.
3.  **Service Quality:** Fiber optic users show a higher churn rate, suggesting potential issues with service stability.
4.  **Support:** Lack of technical support is highly correlated with customer departure.

## 🎯 Strategic Recommendations

- **Retention Incentives:** Targeted loyalty programs for month-to-month subscribers.
- **Onboarding Excellence:** Enhanced support for customers in their first 90 days.
- **Service Audit:** Quality investigation for the Fiber Optic segment.
- **Proactive Support:** Automated outreach for customers showing "at-risk" behavior patterns identified by SHAP.

## 🏁 How to Run

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/telco-churn-analysis.git
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the analysis:
    ```bash
    python scripts/06_generate_notebook.py
    ```

---
*Developed by **Manus AI** as a professional showcase for Data Analysis and Machine Learning.*
