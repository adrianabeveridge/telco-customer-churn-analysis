"""
=============================================================================
PROJECT: CHURN ANALYSIS - TELCO CUSTOMER CHURN
Script 01: Initial Data Exploration (EDA)
=============================================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Load Data ───────────────────────────────────────────────────────────────
df = pd.read_csv('/home/ubuntu/telco_churn_project_en/data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

print("=" * 70)
print("  INITIAL EXPLORATORY ANALYSIS - TELCO CUSTOMER CHURN")
print("=" * 70)

print(f"\n📊 DATASET DIMENSIONS")
print(f"   Rows     : {df.shape[0]:,}")
print(f"   Columns  : {df.shape[1]}")

print(f"\n📋 VARIABLE TYPES")
print(df.dtypes.to_string())

print(f"\n🔍 FIRST 5 ROWS")
print(df.head().to_string())

print(f"\n📈 DESCRIPTIVE STATISTICS - NUMERIC VARIABLES")
print(df.describe().to_string())

print(f"\n❓ MISSING VALUES")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing': missing, 'Percentage (%)': missing_pct})
print(missing_df[missing_df['Missing'] > 0].to_string())
if missing_df[missing_df["Missing"] > 0].empty:
    print("   No missing values detected (check TotalCharges)")

print(f"\n🎯 TARGET VARIABLE DISTRIBUTION (Churn)")
churn_counts = df['Churn'].value_counts()
churn_pct = df['Churn'].value_counts(normalize=True) * 100
print(f"   No Churn  : {churn_counts['No']:,} ({churn_pct['No']:.1f}%)")
print(f"   Churn     : {churn_counts['Yes']:,} ({churn_pct['Yes']:.1f}%)")
print(f"   Imbalance : {churn_pct['No']/churn_pct['Yes']:.1f}:1")

print(f"\n📌 CATEGORICAL VARIABLES - UNIQUE VALUES")
cat_cols = df.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    unique_vals = df[col].unique()
    print(f"   {col:30s}: {len(unique_vals)} unique → {list(unique_vals[:6])}")

print(f"\n💰 TotalCharges ANALYSIS")
print(f"   Original type: {df['TotalCharges'].dtype}")
# Check for blank spaces
spaces = (df['TotalCharges'] == ' ').sum()
print(f"   Values with blank spaces: {spaces}")

# Save summary
summary = {
    'n_rows': df.shape[0],
    'n_cols': df.shape[1],
    'churn_rate': round(churn_pct['Yes'], 2),
    'cat_cols': len(cat_cols),
    'num_cols': len(df.select_dtypes(include='number').columns)
}
print(f"\n✅ EXECUTIVE SUMMARY")
for k, v in summary.items():
    print(f"   {k}: {v}")

print("\n" + "=" * 70)
print("  INITIAL EDA COMPLETED SUCCESSFULLY")
print("=" * 70)
