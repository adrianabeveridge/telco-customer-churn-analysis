"""
=============================================================================
PROJECT: CHURN ANALYSIS - TELCO CUSTOMER CHURN
Script 03: In-depth Exploratory Data Analysis (EDA) and Visualizations
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
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
print("  IN-DEPTH EXPLORATORY DATA ANALYSIS (EDA) AND VISUALIZATIONS")
print("=" * 70)

# ── Target Variable Analysis (Churn) ────────────────────────────────────────
print("\n📊 TARGET VARIABLE ANALYSIS (Churn)")
plt.figure(figsize=(6, 4))
sns.countplot(x="Churn", data=df, palette="viridis")
plt.title("Churn Distribution")
plt.xlabel("Churn (0=No, 1=Yes)")
plt.ylabel("Count")
plt.xticks([0, 1], ["No Churn", "Churn"])
plt.savefig("/home/ubuntu/telco_churn_project_en/visualizations/churn_distribution.png")
plt.close()
print("   - Churn distribution plot saved.")

# ── Numerical Variables Analysis ────────────────────────────────────────────
print("\n📈 NUMERICAL VARIABLES ANALYSIS")
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "NumAdditionalServices", "CostPerService"]

for col in numeric_cols:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True, bins=30, palette="viridis")
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    sns.boxplot(x="Churn", y=col, data=df, palette="viridis")
    plt.title(f"{col} vs Churn")
    plt.xlabel("Churn (0=No, 1=Yes)")
    plt.ylabel(col)
    plt.xticks([0, 1], ["No Churn", "Churn"])
    plt.tight_layout()
    plt.savefig(f"/home/ubuntu/telco_churn_project_en/visualizations/{col}_distribution_churn.png")
    plt.close()
    print(f"   - Distribution and boxplot for {col} saved.")

# ── Categorical Variables vs Churn Analysis ──────────────────────────────
print("\n📊 CATEGORICAL VARIABLES VS CHURN ANALYSIS")
# Exclude numerical columns and target variable
exclude_cols = numeric_cols + ["Churn"]

categorical_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype == "bool"]

for col in categorical_cols:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=col, hue="Churn", data=df, palette="viridis")
    plt.title(f"Churn by {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.legend(title="Churn", labels=["No", "Yes"])
    plt.tight_layout()
    plt.savefig(f"/home/ubuntu/telco_churn_project_en/visualizations/{col}_churn_countplot.png")
    plt.close()
    print(f"   - Count plot for {col} vs Churn saved.")

# ── Correlation Matrix ────────────────────────────────────────────────────
print("\n📈 CORRELATION MATRIX")
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numerical Variables")
plt.tight_layout()
plt.savefig("/home/ubuntu/telco_churn_project_en/visualizations/correlation_matrix.png")
plt.close()
print("   - Correlation Matrix saved.")

print("\n✅ IN-DEPTH EDA AND VISUALIZATIONS COMPLETED.")
print("\n" + "=" * 70)
print("  IN-DEPTH EDA SCRIPT COMPLETED SUCCESSFULLY")
print("=" * 70)
