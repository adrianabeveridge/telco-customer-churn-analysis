"""
=============================================================================
PROJECT: CHURN ANALYSIS - TELCO CUSTOMER CHURN
Script 02: Data Cleaning, Preprocessing, and Feature Engineering
=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ── Load Data ───────────────────────────────────────────────────────────────
df = pd.read_csv("/home/ubuntu/telco_churn_project_en/data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("=" * 70)
print("  DATA CLEANING, PREPROCESSING, AND FEATURE ENGINEERING")
print("=" * 70)

# ── Data Cleaning ───────────────────────────────────────────────────────────
print("\n🧹 STARTING DATA CLEANING...")

# 1. 'customerID' is not useful for modeling, but can be useful for tracking
df = df.drop("customerID", axis=1)
print("   - 'customerID' column removed.")

# 2. 'TotalCharges' is read as a string, needs to be converted to numeric
#    Blank values (' ') should be treated as NaN
df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
print("   - 'TotalCharges' converted to numeric, blank spaces treated as NaN.")

# 3. Impute missing values in 'TotalCharges' with the median
imputer_median = SimpleImputer(strategy="median")
df["TotalCharges"] = imputer_median.fit_transform(df[["TotalCharges"]])
print("   - Missing values in 'TotalCharges' imputed with the median.")

# ── Data Preprocessing ──────────────────────────────────────────────────────
print("\n⚙️ STARTING DATA PREPROCESSING...")

# 1. Encode target variable 'Churn'
#    \'Yes\' -> 1, \'No\' -> 0
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
print("   - Target variable 'Churn' encoded (Yes=1, No=0).")

# 2. Encode binary variables (Yes/No, Male/Female)
#    'No phone service' and 'No internet service' are treated as \'No\'
for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "gender"]:
    if col == "gender":
        df[col] = df[col].map({"Male": 1, "Female": 0})
    else:
        df[col] = df[col].map({"Yes": 1, "No": 0})
print("   - Binary variables encoded.")

# 3. Handle 'No internet service' and 'No phone service' for other columns
internet_service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                         "TechSupport", "StreamingTV", "StreamingMovies"]
for col in internet_service_cols:
    df[col] = df[col].replace("No internet service", "No")
print("   - 'No internet service' treated as 'No' in internet service columns.")

phone_service_cols = ["MultipleLines"]
for col in phone_service_cols:
    df[col] = df[col].replace("No phone service", "No")
print("   - 'No phone service' treated as 'No' in 'MultipleLines' column.")

# 4. One-Hot Encoding for remaining categorical variables
categorical_cols = df.select_dtypes(include="object").columns.tolist()
# Exclude 'Churn' if still in object (already handled)
if "Churn" in categorical_cols:
    categorical_cols.remove("Churn")

if categorical_cols:
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    print(f"   - Remaining categorical variables One-Hot encoded: {categorical_cols}")
else:
    print("   - No remaining categorical variables for One-Hot Encoding.")

# 5. Scale numerical variables
#    Identify numerical columns for scaling (excluding 'Churn' and already handled binary ones)
#    SeniorCitizen is already 0/1, no need to scale

# Numerical columns that need scaling
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print("   - Numerical variables scaled (StandardScaler).")

# ── Feature Engineering (Simple Examples) ───────────────────────────────────
print("\n🛠️ STARTING FEATURE ENGINEERING (Simple Examples)...")

# Example: Create an \'Additional Services\' feature
# Count how many additional services the customer has
additional_services = ["OnlineSecurity_Yes", "OnlineBackup_Yes", "DeviceProtection_Yes",
                       "TechSupport_Yes", "StreamingTV_Yes", "StreamingMovies_Yes"]
# Check if columns exist before summing
existing_additional_services = [col for col in additional_services if col in df.columns]
if existing_additional_services:
    df["NumAdditionalServices"] = df[existing_additional_services].sum(axis=1)
    print("   - 'NumAdditionalServices' feature created.")
else:
    print("   - Could not create 'NumAdditionalServices': service columns not found.")

# Example: Create a \'Cost per Month per Service\' feature
# Avoid division by zero
df["CostPerService"] = df.apply(lambda row: row["MonthlyCharges"] / row["NumAdditionalServices"] if row["NumAdditionalServices"] > 0 else 0,
                                axis=1)
print("   - 'CostPerService' feature created (handling division by zero).")

print("\n✅ PREPROCESSING AND FEATURE ENGINEERING COMPLETED.")
print(f"   Final dataset dimensions: {df.shape[0]} rows, {df.shape[1]} columns.")
print("   First 5 rows of the processed dataset:")
print(df.head().to_string())
print("\n" + "=" * 70)
print("  PREPROCESSING SCRIPT COMPLETED SUCCESSFULLY")
print("=" * 70)

# Save the processed DataFrame for later use
df.to_csv("/home/ubuntu/telco_churn_project_en/data/processed_telco_churn.csv", index=False)
print("   - Processed DataFrame saved to 'data/processed_telco_churn.csv'.")
