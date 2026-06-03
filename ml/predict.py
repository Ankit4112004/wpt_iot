"""
predict.py — load the trained models once and expose simple predict functions.

The backend imports this module and calls these functions; it never touches scikit-learn
directly. Each function takes plain numbers/dicts and returns plain numbers/dicts.

How to explain this file in an interview:
  - "This is the boundary between the ML side and the web side: the backend just calls
     predict_temperature(...) and gets a number back."
  - "Models are loaded once when the process starts, not on every request."
"""

import os
import json
import joblib
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def _load(name):
    return joblib.load(os.path.join(MODELS_DIR, name))


# Load once at import. If models are missing, the backend can check MODELS_LOADED.
try:
    _temp = _load("temperature_regressor.pkl")
    _health = _load("health_classifier.pkl")
    _anom = _load("anomaly_detector.pkl")
    with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
        META = json.load(f)
    MODELS_LOADED = True
except FileNotFoundError:
    _temp = _health = _anom = None
    META = {}
    MODELS_LOADED = False


def predict_temperature(voltage, current, time_in_cycle=0.0):
    """Estimate battery temperature (deg C) from electrical readings (the soft-sensor)."""
    power = voltage * current
    X = np.array([[voltage, current, power, time_in_cycle]])  # order = META['instant_features']
    return round(float(_temp.predict(X)[0]), 3)


def detect_anomaly(voltage, current, temperature):
    """Return whether this operating point looks abnormal, plus a raw score."""
    power = voltage * current
    X = np.array([[voltage, current, power, temperature]])
    label = int(_anom.predict(X)[0])           # -1 = anomaly, 1 = normal
    score = float(_anom.score_samples(X)[0])   # lower = more anomalous
    return {"is_anomaly": label == -1, "score": round(score, 4)}


def predict_health(v_mean, v_min, i_mean, duration, temp_max, temp_mean):
    """Classify the battery for one discharge cycle as healthy / degraded."""
    X = np.array([[v_mean, v_min, i_mean, duration, temp_max, temp_mean]])
    degraded = int(_health.predict(X)[0])
    proba = float(_health.predict_proba(X)[0][1])  # probability of 'degraded'
    return {"degraded": bool(degraded),
            "label": "Degraded" if degraded else "Healthy",
            "degraded_probability": round(proba, 3)}
