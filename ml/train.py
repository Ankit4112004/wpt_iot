import json
import os

import joblib
from sklearn.ensemble import IsolationForest, RandomForestRegressor

import features as F

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def main():
    print("Loading data ...")
    df = F.load_raw()
    print(f"  {len(df):,} raw measurements, batteries: {sorted(df['Battery'].unique())}")

    print("\n[1/2] Training temperature soft-sensor (regression) ...")
    X_inst, y_temp = F.build_instant_table(df)
    temp_model = RandomForestRegressor(
        n_estimators=40,
        max_depth=10,
        min_samples_leaf=30,
        random_state=42,
        n_jobs=-1,
    )
    temp_model.fit(X_inst.values, y_temp.values)
    joblib.dump(temp_model, os.path.join(MODELS_DIR, "temperature_regressor.pkl"))
    print(f"  features: {F.INSTANT_FEATURES} -> Temperature (C)")

    print("\n[2/2] Training anomaly detector (unsupervised) ...")
    anom_features = [
        "Voltage_measured",
        "Current_measured",
        "Power",
        "Temperature_measured",
    ]
    anom_model = IsolationForest(
        n_estimators=120,
        contamination=0.02,
        random_state=42,
        n_jobs=-1,
    )
    anom_model.fit(df[anom_features].values)
    joblib.dump(anom_model, os.path.join(MODELS_DIR, "anomaly_detector.pkl"))
    print(f"  features: {anom_features} -> anomaly score")

    meta = {
        "instant_features": F.INSTANT_FEATURES,
        "anomaly_features": anom_features,
        "dataset": "NASA Li-ion Battery Aging (B0005/06/07/18)",
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved 2 models + metadata.json to ml/models/")

if __name__ == "__main__":
    main()
