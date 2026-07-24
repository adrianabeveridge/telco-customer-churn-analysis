"""
=============================================================================
PROJECT: CHURN ANALYSIS - TELCO CUSTOMER CHURN
Script 04: Predictive Modeling
=============================================================================
"""

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# ── Load Processed Data ─────────────────────────────────────────────────────
df = pd.read_csv("data/processed_telco_churn.csv")

print("=" * 70)
print("  PREDICTIVE MODELING - TELCO CUSTOMER CHURN")
print("=" * 70)

# ── Data Preparation for Modeling ───────────────────────────────────────────
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n📊 DATA DIMENSIONS FOR MODELING")
print(f"   X_train: {X_train.shape}")
print(f"   X_test : {X_test.shape}")
print(f"   y_train: {y_train.shape}")
print(f"   y_test : {y_test.shape}")

# ── Handling Imbalance with SMOTE ───────────────────────────────────────────
print("\n⚖️ HANDLING IMBALANCE WITH SMOTE")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"   X_train (SMOTE): {X_train_smote.shape}")
print(f"   y_train (SMOTE): {y_train_smote.shape}")
print(f"   Churn Distribution (original):\n{y_train.value_counts(normalize=True).mul(100).round(2)}")
print(f"   Churn Distribution (SMOTE):\n{y_train_smote.value_counts(normalize=True).mul(100).round(2)}")

# ── Model Definition ────────────────────────────────────────────────────────
print("\n🤖 DEFINING MACHINE LEARNING MODELS")
models = {
    "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear'),
    "Gaussian Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(random_state=42, probability=True),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(random_state=42)
}

results = {}

# ── Model Training and Evaluation ───────────────────────────────────────────
print("\n🚀 TRAINING AND EVALUATING MODELS...")
for name, model in models.items():
    print(f"\n--- {name} ---")
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    }

    print(f"   Accuracy : {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall   : {recall:.4f}")
    print(f"   F1-Score : {f1:.4f}")
    print(f"   ROC-AUC  : {roc_auc:.4f}")
    print("   Classification Report:")
    print(classification_report(y_test, y_pred))
    print("   Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

print("\n✅ MODEL TRAINING AND EVALUATION COMPLETED.")

# ── Results Summary ─────────────────────────────────────────────────────────
print("\n🏆 MODEL RESULTS SUMMARY")
results_df = pd.DataFrame(results).T.sort_values(by="ROC-AUC", ascending=False)
print(results_df.to_string())

# Save results for later use
results_df.to_csv("reports/model_performance_summary.csv")
print("   - Model performance summary saved to 'reports/model_performance_summary.csv'.")

print("\n" + "=" * 70)
print("  PREDICTIVE MODELING SCRIPT COMPLETED SUCCESSFULLY")
print("=" * 70)
