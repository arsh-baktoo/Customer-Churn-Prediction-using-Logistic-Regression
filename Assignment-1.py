"""
AI-ML Assignment – 1
Topic: Medical Insurance Cost Prediction using Multiple Linear Regression

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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set style for plots
sns.set_theme(style="whitegrid")

# ==============================================================================
# Task 1: Data Understanding
# ==============================================================================
print("=== Task 1: Data Understanding ===")
# 1. Load the dataset using Pandas
df = pd.read_csv('insurance.csv')
print(f"Dataset loaded successfully. Shape: {df.shape}")

# 2. Display the first five records
print("\nFirst 5 Records:")
print(df.head())

# 3. Identify Numerical features, Categorical features, and Target variable
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.drop('charges').tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
target_variable = 'charges'

print("\n--- Feature Identification ---")
print(f"Numerical Features   : {numerical_features}")
print(f"Categorical Features : {categorical_features}")
print(f"Target Variable      : {target_variable}")

# ==============================================================================
# Task 2: Data Preprocessing
# ==============================================================================
print("\n=== Task 2: Data Preprocessing ===")
# 1. Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values Count:")
print(missing_values)

# 2. Encode categorical variables (sex, smoker, region) using One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
print("\nEncoded Columns:")
print(df_encoded.columns.tolist())

# 3. Split dataset into 80% training and 20% testing
X = df_encoded.drop(target_variable, axis=1)
y = df_encoded[target_variable]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain set shape: {X_train.shape} (80%)")
print(f"Test set shape : {X_test.shape} (20%)")

# ==============================================================================
# Task 3: Model Development
# ==============================================================================
print("\n=== Task 3: Model Development ===")
# Build and train Multiple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict charges for test dataset
y_pred = model.predict(X_test)

print("Multiple Linear Regression model trained successfully.")
print(f"Model Intercept: {model.intercept_:.2f}")
print("Feature Coefficients:")
for col, coef in zip(X.columns, model.coef_):
    print(f"  {col:<20}: {coef:.2f}")

# ==============================================================================
# Task 4: Model Evaluation
# ==============================================================================
print("\n=== Task 4: Model Evaluation ===")
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE) : ${mae:.2f}")
print(f"Mean Squared Error (MSE)  : {mse:.2f}")
print(f"Root Mean Squared Error   : ${rmse:.2f}")
print(f"R2 Score                  : {r2:.4f} ({r2 * 100:.2f}%)")

# Actual vs Predicted scatter plot
plt.figure(figsize=(8, 6), dpi=300)
plt.scatter(y_test, y_pred, alpha=0.7, color='#1f77b4', edgecolors='k', s=50, label='Predicted vs Actual')
max_val = max(y_test.max(), y_pred.max())
min_val = min(y_test.min(), y_pred.min())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit Line (y = x)')
plt.title('Actual vs Predicted Medical Insurance Charges', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Actual Charges ($)', fontsize=12)
plt.ylabel('Predicted Charges ($)', fontsize=12)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300)
plt.close()

print("\nPerformance Observations:")
print("1. High Explanatory Power: R2 Score of 0.7836 indicates ~78.36% variance explained.")
print("2. Dominance of Smoking: Smoking status (smoker_yes ~ +$23,651.13) is the single strongest factor driving charges.")
print("3. Higher Error at Extreme Charges: High-cost outliers (> $30,000) show larger residuals due to non-linear interaction between BMI and smoking.")

# ==============================================================================
# Task 5: Conclusion
# ==============================================================================
conclusion_text = """
Conclusion:
This project developed a Multiple Linear Regression model to estimate medical insurance charges based on customer demographic and health attributes. Key findings demonstrate that lifestyle and physical health metrics are primary drivers of insurance expenditure. Specifically, smoking status exerts the most dominant positive impact, increasing predicted charges by approximately $23,651, followed by age (+$257 per year) and BMI (+$337 per unit). The model achieved an R2 score of 0.7836 and a Mean Absolute Error of $4,181.19. A major limitation of Multiple Linear Regression in this context is its strict assumption of linear, additive relationships. In reality, health risk factors exhibit strong non-linear interactions—such as the compounding effect between high BMI and smoking—which leads to underestimation of extreme charges for high-risk individuals.
"""
print("\n=== Task 5: Conclusion ===")
print(conclusion_text.strip())
