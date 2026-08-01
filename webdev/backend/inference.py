"""
inference.py — run the 3 ML models on a reading.

This is the chain that turns raw electrical readings into the dashboard's intelligence:
  1. temperature soft-sensor  -> predicted temperature (no temp sensor needed)
  2. anomaly detector         -> fed the PREDICTED temp (consistent with 'no temp sensor')
  3. health classifier        -> needs per-cycle stats, so we keep a rolling window of
                                 recent readings and summarise it each tick

How to explain this file in an interview:
  - "The soft-sensor predicts temperature, and that prediction is then fed into the anomaly
     and health models — so the whole chain works from electrical signals alone."
  - "The health model is per-cycle, so I summarise a rolling window of recent readings
     into the cycle-shape features it expects."
"""

import os
import sys
from collections import deque

# Make the ml/ package importable (it lives at repo_root/ml).
ML_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ml"))
sys.path.insert(0, ML_DIR)
import predict as M  # noqa: E402  (the ml/predict.py module)

# Rolling window of recent (predicted) readings used to build per-cycle health features.
_window = deque(maxlen=60)


def run_inference(voltage, current, time_in_cycle):
    """Return the full ML output dict for one reading."""
    # 1) temperature soft-sensor
    temp = M.predict_temperature(voltage, current, time_in_cycle)

    # 2) anomaly detection on the (predicted) operating point
    anom = M.detect_anomaly(voltage, current, temp)

    # 3) battery-health from a rolling summary of recent readings
    #    (COMMENTED OUT — health classifier and RUL regressor downgraded)
    _window.append({"v": voltage, "i": current, "t": temp, "time": time_in_cycle})
    # vs = [w["v"] for w in _window]
    # cs = [w["i"] for w in _window]
    # ts = [w["t"] for w in _window]
    # times = [w["time"] for w in _window]
    # # The same per-cycle summary feeds BOTH the health classifier and the RUL regressor.
    # v_mean, v_min = sum(vs) / len(vs), min(vs)
    # i_mean = sum(cs) / len(cs)
    # duration = max(times) - min(times) if len(times) > 1 else times[0]
    # temp_max, temp_mean = max(ts), sum(ts) / len(ts)

    # health = M.predict_health(v_mean, v_min, i_mean, duration, temp_max, temp_mean)
    # rul = M.predict_rul(v_mean, v_min, i_mean, duration, temp_max, temp_mean)

    return {
        "predicted_temp": temp,
        "is_anomaly": anom["is_anomaly"],
        "anomaly_score": anom["score"],
        "health_label": "N/A",
        "degraded_probability": 0.0,
        "predicted_rul": 0,
    }


def what_if(voltage, current, time_in_cycle=1800.0):
    """Interactive on-demand prediction (used by the /api/whatif endpoint)."""
    return M.whatif(voltage, current, time_in_cycle)


def models_ready():
    return M.MODELS_LOADED
