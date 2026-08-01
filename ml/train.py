"""
train.py — train the three ML models and save them to ml/models/.

Run:  python ml/train.py

The three models (one of each main ML type, so the project shows breadth):
  1. Temperature soft-sensor    -> RandomForestRegressor   (supervised regression)
  2. Battery-health classifier  -> RandomForestClassifier  (supervised classification)
  3. Anomaly detector           -> IsolationForest         (unsupervised)

Each is a single standard scikit-learn model — simple on purpose. The value of the
project is the working end-to-end system, not model complexity.

How to explain this file in an interview:
  - "I train three models from the same real dataset and save them as .pkl files that
     the backend loads at runtime."
  - "Random Forest = many decision trees that each vote; their average is the prediction."
  - "Isolation Forest learns what 'normal' operation looks like and flags outliers — it
     needs no labels, which is why it's called unsupervised."
"""

import os
import json
import joblib
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    IsolationForest,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import features as F  # run from inside ml/ , or `python ml/train.py` from repo root

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def main():
    print("Loading data ...")
    df = F.load_raw()
    print(f"  {len(df):,} raw measurements, batteries: {sorted(df['Battery'].unique())}")

    # ---------- Model 1: Temperature soft-sensor (regression) ----------
    print("\n[1/4] Training temperature soft-sensor (regression) ...")
    X_inst, y_temp = F.build_instant_table(df)
    # max_depth / min_samples_leaf bound the tree size: keeps the saved model a few MB
    # (deployable) instead of ~2 GB, with no real accuracy loss since temperature is smooth.
    temp_model = RandomForestRegressor(
        n_estimators=40, max_depth=10, min_samples_leaf=30, random_state=42, n_jobs=-1
    )
    # .values -> fit on a plain array so the model stores no column names; this lets the
    # backend predict with plain arrays without scikit-learn emitting feature-name warnings.
    temp_model.fit(X_inst.values, y_temp.values)
    joblib.dump(temp_model, os.path.join(MODELS_DIR, "temperature_regressor.pkl"))
    print(f"  features: {F.INSTANT_FEATURES}  ->  Temperature (C)")

    # ---------- Model 2: Battery-health classifier (COMMENTED OUT — downgraded) ----------
    # print("\n[2/4] Training battery-health classifier (classification) ...")
    # cyc = F.build_cycle_table(df)
    # health_model = DecisionTreeClassifier(
    #     random_state=42, class_weight="balanced"
    # )
    # health_model.fit(cyc[F.CYCLE_FEATURES].values, cyc["degraded"].values)
    # joblib.dump(health_model, os.path.join(MODELS_DIR, "health_classifier.pkl"))
    # print(f"  features: {F.CYCLE_FEATURES}  ->  degraded (0/1)")
    # print(f"  health capacity threshold: {cyc.attrs['health_threshold']:.4f} Ah")

    # ---------- Model 4: Remaining Useful Life (COMMENTED OUT — downgraded) ----------
    # print("\n[4/4] Training remaining-useful-life regressor (regression) ...")
    # rul = F.build_rul_table(df)
    # rul_model = DecisionTreeRegressor(
    #     max_depth=12, min_samples_leaf=5, random_state=42
    # )
    # rul_model.fit(rul[F.CYCLE_FEATURES].values, rul["RUL"].values)
    # joblib.dump(rul_model, os.path.join(MODELS_DIR, "rul_regressor.pkl"))
    # print(f"  features: {F.CYCLE_FEATURES}  ->  remaining cycles (EOL @ {F.EOL_CAPACITY} Ah)")

    # ---------- Model 3: Anomaly detector (unsupervised) ----------
    print("\n[2/2] Training anomaly detector (unsupervised) ...")
    # Learns the normal operating envelope of voltage/current/power/temperature.
    anom_features = ["Voltage_measured", "Current_measured", "Power", "Temperature_measured"]
    anom_model = IsolationForest(
        n_estimators=120, contamination=0.02, random_state=42, n_jobs=-1
    )
    anom_model.fit(df[anom_features].values)
    joblib.dump(anom_model, os.path.join(MODELS_DIR, "anomaly_detector.pkl"))
    print(f"  features: {anom_features}  ->  anomaly score")

    # ---------- Save metadata the backend/predict layer needs ----------
    meta = {
        "instant_features": F.INSTANT_FEATURES,
        # "cycle_features": F.CYCLE_FEATURES,
        "anomaly_features": anom_features,
        # "health_threshold": cyc.attrs["health_threshold"],
        # "eol_capacity": F.EOL_CAPACITY,
        "dataset": "NASA Li-ion Battery Aging (B0005/06/07/18)",
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved 2 models + metadata.json to ml/models/")


if __name__ == "__main__":
    main()
