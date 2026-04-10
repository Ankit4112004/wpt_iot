import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

# Load data
df = pd.read_csv('evdata2.csv')

# Filter for Li-ion batteries only
df_lion = df[df['Battery Type'] == 'Li-ion'].copy()

# Create features
df_lion['SOC_abs'] = np.abs(df_lion['SOC (%)'])

# Select features and target
features = ['SOC_abs', 'Voltage (V)', 'Current (A)', 'Battery Temp (°C)', 'Ambient Temp (°C)']
target = 'Efficiency (%)'

X = df_lion[features].copy()
y = df_lion[target].copy()

# Remove NaN
mask = X.isnull().any(axis=1) | y.isnull()
X = X[~mask]
y = y[~mask]

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

# Save model and scaler
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✓ Model saved as model.pkl")
print("✓ Scaler saved as scaler.pkl")
print(f"✓ Training R² Score: {model.score(X_train_scaled, y_train):.4f}")
print(f"✓ Testing R² Score: {model.score(X_test_scaled, y_test):.4f}")
