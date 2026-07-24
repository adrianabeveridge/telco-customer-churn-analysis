
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import os

# Path to the project directory
project_dir = "/home/ubuntu/telco_churn_project_en"

# Create a new notebook
nb = new_notebook()

# ── Title and Introduction ──────────────────────────────────────────────────
nb.cells.append(new_markdown_cell("# Telco Customer Churn Analysis\n\nThis notebook presents a comprehensive analysis of the customer churn problem in a telecommunications company. The objective is to identify the factors that lead customers to cancel their services and to build predictive models to forecast churn.\n\n**Contents:**\n1.  Introduction and Data Loading\n2.  Initial Exploratory Data Analysis (EDA)\n3.  Data Cleaning, Preprocessing, and Feature Engineering\n4.  In-depth Exploratory Data Analysis (EDA) and Visualizations\n5.  Predictive Modeling\n6.  Model Evaluation, Feature Importance, and Interpretability\n7.  Conclusion and Recommendations"))

# ── 1. Introduction and Data Loading ────────────────────────────────────────
nb.cells.append(new_markdown_cell("## 1. Introduction and Data Loading\n\nLoading the `WA_Fn-UseC_-Telco-Customer-Churn.csv` dataset and an initial inspection of the data."))

code_initial_eda = open(os.path.join(project_dir, "scripts/01_initial_eda.py")).read()
output_initial_eda = open(os.path.join(project_dir, "reports/eda_output.txt")).read()

nb.cells.append(new_code_cell(code_initial_eda))
nb.cells.append(new_markdown_cell("### Initial EDA Script Output"))
nb.cells.append(new_code_cell(f"""print(""""""\n{output_initial_eda}"""""")"""))

# ── 2. Data Cleaning, Preprocessing, and Feature Engineering ────────────────
nb.cells.append(new_markdown_cell("## 2. Data Cleaning, Preprocessing, and Feature Engineering\n\nIn this section, we perform data cleaning, handle missing values, encode categorical variables, and scale numerical variables. We also create some new features to enrich the dataset."))

code_preprocessing = open(os.path.join(project_dir, "scripts/02_preprocessing_feature_engineering.py")).read()
output_preprocessing = open(os.path.join(project_dir, "reports/preprocessing_output.txt")).read()

nb.cells.append(new_code_cell(code_preprocessing))
nb.cells.append(new_markdown_cell("### Preprocessing Script Output"))
nb.cells.append(new_code_cell(f"""print(""""""\n{output_preprocessing}"""""")"""))

# ── 3. In-depth Exploratory Data Analysis (EDA) and Visualizations ────────
nb.cells.append(new_markdown_cell("## 3. In-depth Exploratory Data Analysis (EDA) and Visualizations\n\nDetailed exploration of the relationships between variables and the target variable (Churn), using various visualizations to extract insights."))

code_deep_eda = open(os.path.join(project_dir, "scripts/03_deep_eda_visualizations.py")).read()
output_deep_eda = open(os.path.join(project_dir, "reports/deep_eda_output.txt")).read()

nb.cells.append(new_code_cell(code_deep_eda))
nb.cells.append(new_markdown_cell("### In-depth EDA Script Output"))
nb.cells.append(new_code_cell(f"""print(""""""\n{output_deep_eda}"""""")"""))

# Add generated visualizations
visualizations_path = os.path.join(project_dir, "visualizations")
image_files = [
    "churn_distribution.png",
    "tenure_distribution_churn.png",
    "MonthlyCharges_distribution_churn.png",
    "TotalCharges_distribution_churn.png",
    "NumAdditionalServices_distribution_churn.png",
    "CostPerService_distribution_churn.png",
    "MultipleLines_Yes_churn_countplot.png",
    "InternetService_Fiber optic_churn_countplot.png",
    "InternetService_No_churn_countplot.png",
    "OnlineSecurity_Yes_churn_countplot.png",
    "OnlineBackup_Yes_churn_countplot.png",
    "DeviceProtection_Yes_churn_countplot.png",
    "TechSupport_Yes_churn_countplot.png",
    "StreamingTV_Yes_churn_countplot.png",
    "StreamingMovies_Yes_churn_countplot.png",
    "Contract_One year_churn_countplot.png",
    "Contract_Two year_churn_countplot.png",
    "PaymentMethod_Credit card (automatic)_churn_countplot.png",
    "PaymentMethod_Electronic check_churn_countplot.png",
    "PaymentMethod_Mailed check_churn_countplot.png",
    "correlation_matrix.png"
]

nb.cells.append(new_markdown_cell("### Generated Visualizations"))
for img_file in image_files:
    img_path = os.path.join(visualizations_path, img_file)
    if os.path.exists(img_path):
        nb.cells.append(new_markdown_cell(f"#### {img_file.replace("_", " ").replace(".png", "").title()}\n![{img_file}]({img_path})"))

# ── 4. Predictive Modeling ──────────────────────────────────────────────────
nb.cells.append(new_markdown_cell("## 4. Predictive Modeling\n\nIn this section, we train and evaluate various Machine Learning models to predict churn, including handling class imbalance with SMOTE."))

code_modeling = open(os.path.join(project_dir, "scripts/04_predictive_modeling.py")).read()
output_modeling = open(os.path.join(project_dir, "reports/predictive_modeling_output.txt")).read()

nb.cells.append(new_code_cell(code_modeling))
nb.cells.append(new_markdown_cell("### Predictive Modeling Script Output"))
nb.cells.append(new_code_cell(f"""print(""""""\n{output_modeling}"""""")"""))

# ── 5. Model Evaluation, Feature Importance, and Interpretability ───────────
nb.cells.append(new_markdown_cell("## 5. Model Evaluation, Feature Importance, and Interpretability\n\nIn-depth analysis of the best model, including ROC Curve, Confusion Matrix, Feature Importance, and Interpretability with SHAP."))

code_evaluation = open(os.path.join(project_dir, "scripts/05_model_evaluation_interpretability.py")).read()
output_evaluation = open(os.path.join(project_dir, "reports/model_evaluation_output.txt")).read()

nb.cells.append(new_code_cell(code_evaluation))
nb.cells.append(new_markdown_cell("### Model Evaluation and Interpretability Script Output"))
nb.cells.append(new_code_cell(f"""print(""""""\n{output_evaluation}"""""")"""))

# Add evaluation visualizations
eval_image_files = [
    "roc_curve.png",
    "confusion_matrix.png",
    "feature_importance.png",
    "shap_summary_bar.png",
    "shap_summary_beeswarm.png"
]

nb.cells.append(new_markdown_cell("### Evaluation and Interpretability Visualizations"))
for img_file in eval_image_files:
    img_path = os.path.join(visualizations_path, img_file)
    if os.path.exists(img_path):
        nb.cells.append(new_markdown_cell(f"#### {img_file.replace("_", " ").replace(".png", "").title()}\n![{img_file}]({img_path})"))

# ── 6. Conclusion and Recommendations ───────────────────────────────────────
nb.cells.append(new_markdown_cell("## 6. Conclusion and Recommendations\n\nBased on the analyses and models developed, we can draw the following conclusions and propose strategic recommendations for the telecommunications company:\n\n**Key Insights:**\n*   **Churn Factors:** The variables with the highest impact on churn include `Contract` (month-to-month contracts), `tenure` (new customers), `InternetService_Fiber optic` (fiber optic service), `MonthlyCharges` (higher monthly charges), and `TechSupport` (lack of technical support).\n*   **Imbalance:** The original dataset shows significant imbalance, with most customers not churning. Using techniques like SMOTE was crucial for training effective models.\n*   **Model Performance:** Tree-based models, such as Gradient Boosting and LightGBM, showed the best performance, especially in terms of AUC-ROC, indicating a good ability to distinguish between customers who will churn and those who will not.\n\n**Strategic Recommendations:**\n1.  **Loyalty Programs for Month-to-Month Contracts:** Customers with month-to-month contracts are more likely to churn. Offering incentives to migrate to longer-term contracts (annual or biennial) can reduce this rate.\n2.  **Attention to New Customers:** Customers with lower `tenure` (contract duration) are more prone to churn. Implementing robust welcome programs, proactive follow-ups, and special offers in the first few months can increase retention.\n3.  **Improve Fiber Optic Service:** The high churn rate among fiber optic users suggests issues with service quality or support. Investigating and resolving these issues is fundamental.\n4.  **Price and Package Optimization:** Customers with higher `MonthlyCharges` tend to churn. Reviewing the pricing structure and offering more competitive or personalized packages can be an effective strategy.\n5.  **Strengthen Technical Support:** The absence of `TechSupport` is a strong predictor of churn. Investing in quality technical support, with fast and effective service, is essential for customer satisfaction and retention.\n6.  **Personalized Retention Campaigns:** Use predictive models to identify high-risk churn customers and target personalized retention campaigns, offering discounts, service upgrades, or dedicated support."))

# Save the notebook
notebook_path = os.path.join(project_dir, "notebooks/telco_churn_analysis_en.ipynb")
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"\n✅ Jupyter Notebook generated successfully at: {notebook_path}")
