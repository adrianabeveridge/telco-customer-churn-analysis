"""
=============================================================================
PROJECT: CHURN ANALYSIS - TELCO CUSTOMER CHURN
Script 05: Model Evaluation, Feature Importance, and Interpretability
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import shap
import warnings
warnings.filterwarnings("ignore")

# ── Visualization Settings ──────────────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10

# ── Load Processed Data ─────────────────────────────────────────────────────
df = pd.read_csv("/home/ubuntu/telco_churn_project_en/data/processed_telco_churn.csv")

print("=" * 70)
print("  MODEL EVALUATION, FEATURE IMPORTANCE, AND INTERPRETABILITY")
print("=" * 70)

# ── Data Preparation for Modeling (re-split for consistency) ────────────────
X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ── Load the Best Model (e.g., Gradient Boosting, based on previous results) ──
# For this script, we will re-train Gradient Boosting to ensure consistency
# In a real scenario, the trained model would be saved and loaded.
print("\n🤖 TRAINING THE BEST MODEL (Gradient Boosting) FOR ANALYSIS...")
best_model = GradientBoostingClassifier(random_state=42)
best_model.fit(X_train, y_train)
print("   - Gradient Boosting model trained.")

# ── ROC Curve and AUC ───────────────────────────────────────────────────────
print("\n📈 ROC CURVE AND AUC")
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Gradient Boosting Classifier')
plt.legend(loc="lower right")
plt.savefig("/home/ubuntu/telco_churn_project_en/visualizations/roc_curve.png")
plt.close()
print("   - ROC curve saved.")

# ── Confusion Matrix ────────────────────────────────────────────────────────
print("\n📉 CONFUSION MATRIX")
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"]).plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Gradient Boosting Classifier")
plt.savefig("/home/ubuntu/telco_churn_project_en/visualizations/confusion_matrix.png")
plt.close()
print("   - Confusion Matrix saved.")

# ── Feature Importance (Model-Based) ────────────────────────────────────────
print("\n💡 FEATURE IMPORTANCE (MODEL-BASED)")
if hasattr(best_model, "feature_importances_"):
    feature_importances = pd.Series(best_model.feature_importances_, index=X.columns)
    feature_importances = feature_importances.sort_values(ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x=feature_importances, y=feature_importances.index, palette="viridis")
    plt.title("Feature Importance - Gradient Boosting Classifier")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("/home/ubuntu/telco_churn_project_en/visualizations/feature_importance.png")
    plt.close()
    print("   - Feature Importance plot saved.")
else:
    print("   - Model does not have feature_importances_ attribute.")

# ── Interpretability with SHAP (SHapley Additive exPlanations) ──────────────
print("\n🧠 INTERPRETABILITY WITH SHAP")
# Use a subset of the test data for SHAP to speed up calculation
sample_X_test = X_test.sample(n=100, random_state=42)

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(sample_X_test)

# Global summary of feature importance with SHAP
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, sample_X_test, plot_type="bar", show=False)
plt.title("Global Feature Importance (SHAP)")
plt.tight_layout()
plt.savefig("/home/ubuntu/telco_churn_project_en/visualizations/shap_summary_bar.png")
plt.close()
print("   - Global Feature Importance (SHAP) plot saved.")

# Beeswarm plot to understand impact and direction
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, sample_X_test, show=False)
plt.title("Feature Impact and Direction (SHAP)")
plt.tight_layout()
plt.savefig("/home/ubuntu/telco_churn_project_en/visualizations/shap_summary_beeswarm.png")
plt.close()
print("   - Feature Impact and Direction (SHAP) plot saved.")

print("\n✅ MODEL EVALUATION, FEATURE IMPORTANCE, AND INTERPRETABILITY COMPLETED.")
print("\n" + "=" * 70)
print("  MODEL EVALUATION AND INTERPRETABILITY SCRIPT COMPLETED SUCCESSFULLY")
print("=" * 70)
