"""Run the two active ML models on each telemetry reading."""

import os
import sys

# Make the ml/ package importable; it lives at repo_root/ml.
ML_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ml"))
sys.path.insert(0, ML_DIR)
import predict as M  # noqa: E402


def run_inference(voltage, current, time_in_cycle):
    """Return temperature and anomaly outputs for one reading."""
    temp = M.predict_temperature(voltage, current, time_in_cycle)
    anom = M.detect_anomaly(voltage, current, temp)
    return {
        "predicted_temp": temp,
        "is_anomaly": anom["is_anomaly"],
        "anomaly_score": anom["score"],
    }


def models_ready():
    return M.MODELS_LOADED
