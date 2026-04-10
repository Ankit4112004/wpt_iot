import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('evdata2.csv')

print("=" * 60)
print("BATTERY EFFICIENCY PREDICTION MODEL")
print("=" * 60)
print(f"\nOriginal data shape: {df.shape}")
print(f"Columns: {list(df.columns)}\n")

# Filter for Li-ion batteries only
df_lion = df[df['Battery Type'] == 'Li-ion'].copy()
print(f"Filtered to Li-ion batteries: {df_lion.shape[0]} records\n")

# Create features: abs(SOC), Voltage, Current, Battery Temp, Ambient Temp
df_lion['SOC_abs'] = np.abs(df_lion['SOC (%)'])

# Select features and target
features = ['SOC_abs', 'Voltage (V)', 'Current (A)', 'Battery Temp (°C)', 'Ambient Temp (°C)']
target = 'Efficiency (%)'

X = df_lion[features].copy()
y = df_lion[target].copy()

print(f"Features (X): {features}")
print(f"Target (y): {target}")
print(f"\nData shape - X: {X.shape}, y: {y.shape}\n")

# Check for missing values
print(f"Missing values in X:\n{X.isnull().sum()}")
print(f"Missing values in y: {y.isnull().sum()}\n")

# Remove any rows with NaN
mask = X.isnull().any(axis=1) | y.isnull()
X = X[~mask]
y = y[~mask]

print(f"After removing NaN - X: {X.shape}, y: {y.shape}\n")

# Display statistics
print("Feature Statistics:")
print(X.describe())
print(f"\nTarget Statistics:")
print(y.describe())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
print("\n" + "=" * 60)
print("RANDOM FOREST REGRESSOR")
print("=" * 60)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

y_pred_rf = rf_model.predict(X_test_scaled)

# Evaluation
print(f"\nModel Performance on Test Set:")
print(f"R² Score: {r2_score(y_test, y_pred_rf):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_rf):.4f}")

# Feature importance
print(f"\nFeature Importance:")
for feat, importance in zip(features, rf_model.feature_importances_):
    print(f"  {feat}: {importance:.4f}")

# Train Linear Regression for comparison
print("\n" + "=" * 60)
print("LINEAR REGRESSION (COMPARISON)")
print("=" * 60)

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

y_pred_lr = lr_model.predict(X_test_scaled)

print(f"\nModel Performance on Test Set:")
print(f"R² Score: {r2_score(y_test, y_pred_lr):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_lr):.4f}")

print(f"\nLinear Regression Coefficients:")
for feat, coef in zip(features, lr_model.coef_):
    print(f"  {feat}: {coef:.4f}")

# Sample predictions
print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS (First 10 test samples)")
print("=" * 60)
print(f"\n{'Actual':<10} {'RF Pred':<12} {'LR Pred':<12} {'Error RF':<12} {'Error LR':<12}")
print("-" * 60)

for i in range(min(10, len(y_test))):
    actual = y_test.iloc[i]
    pred_rf = y_pred_rf[i]
    pred_lr = y_pred_lr[i]
    error_rf = abs(actual - pred_rf)
    error_lr = abs(actual - pred_lr)
    print(f"{actual:<10.2f} {pred_rf:<12.2f} {pred_lr:<12.2f} {error_rf:<12.2f} {error_lr:<12.2f}")

# Save model summary
print("\n" + "=" * 60)
print("MODEL SUMMARY")
print("=" * 60)
print(f"""
Data:
  - Total Li-ion records: {df_lion.shape[0]}
  - Training samples: {X_train.shape[0]}
  - Testing samples: {X_test.shape[0]}

Input Features:
  1. SOC (%) - Absolute value
  2. Voltage (V)
  3. Current (A)
  4. Battery Temp (°C)
  5. Ambient Temp (°C)

Target: Efficiency (%)

Best Model: Random Forest (if R² > Linear Regression)
  - Tests: {X_test.shape[0]} samples
  - R² Score: {r2_score(y_test, y_pred_rf):.4f}
  - RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.4f}%
  - MAE: {mean_absolute_error(y_test, y_pred_rf):.4f}%
""")

print("Model trained successfully! ✓")
