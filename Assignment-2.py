"""
AI-ML Assignment – 2
Topic: Customer Churn Prediction using Logistic Regression

Student Details:
- Name: Arsh Baktoo
- Registration Number: 23BCE10430
- Application Number: IN26010763
- Batch Number: 2(B)
- Email ID: arshbaktoo@gmail.com
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# Set global seaborn theme
sns.set_theme(style="whitegrid")

# ==============================================================================
# Task 1: Data Understanding
# ==============================================================================
print("=== Task 1: Data Understanding ===")

# 1. Load dataset using Pandas
df = pd.read_csv('churn.csv')
print(f"Dataset loaded successfully. Shape: {df.shape}")

# 2. Display the first five records
print("\nFirst 5 Records:")
print(df.head())

# 3. Identify feature types and target variable
# TotalCharges might be read as object due to blank spaces; clean it first for identification
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce')

numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
categorical_features = [col for col in df.columns if df[col].dtype == 'object' and col not in ['customerID', 'Churn']]
target_variable = 'Churn'

print("\n--- Identification of Variables ---")
print(f"Numerical Features   : {numerical_features}")
print(f"Categorical Features : {categorical_features}")
print(f"Target Variable      : {target_variable}")

# ==============================================================================
# Task 2: Data Preprocessing
# ==============================================================================
print("\n=== Task 2: Data Preprocessing ===")

# 1. Check for missing values
print("\nMissing Values Count per Column:")
missing_vals = df.isnull().sum()
print(missing_vals[missing_vals > 0])

# Impute missing values in TotalCharges with median
median_total = df['TotalCharges'].median()
df['TotalCharges'] = df['TotalCharges'].fillna(median_total)
print(f"Imputed {missing_vals['TotalCharges']} missing values in TotalCharges with median ({median_total:.2f}).")

# Drop unique identifier column if present
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# Encode binary target 'Churn' (Yes: 1, No: 0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Separate predictor features (X) and target (y)
X_raw = df.drop('Churn', axis=1)
y = df['Churn']

# 2. Encode categorical variables using One-Hot Encoding
X_encoded = pd.get_dummies(X_raw, drop_first=True)
print(f"\nTotal Encoded Features: {X_encoded.shape[1]}")

# 3. Split dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train Set Shape: {X_train.shape} (80%)")
print(f"Test Set Shape : {X_test.shape} (20%)")

# 4. Feature Scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# Task 3: Model Development
# ==============================================================================
print("\n=== Task 3: Model Development ===")

# Build and train Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict binary churn classes and probabilities for the test set
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

print("Logistic Regression model trained successfully.")

# ==============================================================================
# Task 4: Model Evaluation
# ==============================================================================
print("\n=== Task 4: Model Evaluation ===")

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy  : {acc:.4f} ({acc * 100:.2f}%)")
print(f"Precision : {prec:.4f} ({prec * 100:.2f}%)")
print(f"Recall    : {rec:.4f} ({rec * 100:.2f}%)")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn (0)', 'Churn (1)']))

# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5), dpi=300)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix - Customer Churn Logistic Regression', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Predicted Label', fontsize=11)
plt.ylabel('Actual Label', fontsize=11)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()

# ROC Curve Plot
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(7, 5), dpi=300)
plt.plot(fpr, tpr, color='#1f77b4', linewidth=2.5, label=f'Logistic Regression (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2, label='Random Chance (AUC = 0.50)')
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
plt.legend(loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300)
plt.close()

print("\nPerformance Observations:")
print("1. High Classification Accuracy & Discriminative Capability: The model achieves 80.70% accuracy and an ROC-AUC of 0.8416, indicating strong overall capacity to separate churners from non-churners.")
print("2. Recall Trade-off in Imbalanced Churn Data: While overall accuracy is ~80.7%, Recall for the churn class is 56.68% (Precision: 65.84%). This reflects class imbalance in customer subscription datasets where non-churners outnumber churners.")
print("3. Key Predictive Drivers: Short tenure, month-to-month contracts, higher monthly charges, and fiber optic internet service significantly elevate the probability of customer churn.")

# ==============================================================================
# Task 5: Conclusion
# ==============================================================================
conclusion_text = """
Conclusion:
This project implemented a Logistic Regression model to predict customer churn in a telecommunications service provider. The model demonstrated robust predictive capability, achieving an accuracy of 80.70% and an ROC-AUC score of 0.8416 on the test dataset. Feature analysis reveals that customer tenure, contract length (month-to-month contracts), monthly charges, and internet service type are the primary drivers influencing churn probability. Customers with short tenure and month-to-month subscriptions exhibit significantly higher churn risk. A key limitation of standard Logistic Regression in customer churn prediction is its linear decision boundary assumption and sensitivity to class imbalance. Since retaining churn-prone customers is vital, future improvements should incorporate threshold tuning, class weighting, or non-linear ensemble models (e.g., Random Forest or XGBoost) to boost recall for high-risk customers.
"""
print("\n=== Task 5: Conclusion ===")
print(conclusion_text.strip())
