"""
features.py — data loading and feature engineering for the EV battery ML models.

This is the single place where we turn the raw NASA battery measurements into the
feature tables the models train on. Keeping it in one module means train.py,
evaluate.py and predict.py all build features the exact same way (no drift).

Data source: NASA Li-ion Battery Aging dataset (real measured discharge cycles for
cells B0005/B0006/B0007/B0018). Each row is one instantaneous measurement.

How to explain this file in an interview:
  - "Raw data is per-timestep voltage/current/temperature for several real battery cells."
  - "I build TWO views: an instantaneous view (one row = one reading) for the temperature
     and anomaly models, and a per-cycle view (one row = one discharge cycle) for the
     battery-health model."
  - "Power = Voltage x Current is a derived feature; the rest are direct measurements."
"""

import os
import pandas as pd

# ---- paths ----
HERE = os.path.dirname(__file__)
RAW_CSV = os.path.join(HERE, "data", "raw", "nasa_discharge.csv")

# ---- feature definitions (kept as constants so every script agrees) ----

# Instantaneous features: what a live battery monitor can observe at one moment.
# Used by the temperature soft-sensor and the anomaly detector.
INSTANT_FEATURES = ["Voltage_measured", "Current_measured", "Power", "Time"]

# Per-cycle features: summary of one full discharge cycle.
# Used by BOTH the health classifier and the RUL (remaining-useful-life) regressor.
CYCLE_FEATURES = ["v_mean", "v_min", "i_mean", "duration", "temp_max", "temp_mean"]

# End-of-life threshold (Ah). The NASA-standard failure criterion: a cell is "dead" once
# its capacity fades to this value. RUL = how many cycles remain until it hits this.
EOL_CAPACITY = 1.4


def load_raw():
    """Load the raw NASA discharge measurements and add the derived Power column."""
    df = pd.read_csv(RAW_CSV)
    # Power (Watts) = terminal voltage x current. Negative current = discharging.
    df["Power"] = df["Voltage_measured"] * df["Current_measured"]
    return df


def build_instant_table(df):
    """
    Instantaneous view: one row per measurement.

    Returns X (features), y_temp (temperature target), and the raw frame so the
    anomaly detector can reuse the same rows.
    """
    X = df[INSTANT_FEATURES].copy()
    y_temp = df["Temperature_measured"].copy()
    return X, y_temp


def build_cycle_table(df):
    """
    Per-cycle view: collapse each (battery, cycle) into one summary row.

    - Features describe the shape of the discharge (mean/min voltage, duration, temps).
    - 'capacity' is the measured cell capacity for that cycle = the State-of-Health signal.
    - 'degraded' is the classification label: 1 if capacity is in the lower half (worn),
      0 if healthy. We also keep 'battery' so evaluation can split BY cell (a model must
      generalise to a cell it never trained on — that prevents leakage).
    """
    g = df.groupby(["Battery", "id_cycle"])
    cyc = g.agg(
        v_mean=("Voltage_measured", "mean"),
        v_min=("Voltage_measured", "min"),
        i_mean=("Current_measured", "mean"),
        duration=("Time", "max"),
        temp_max=("Temperature_measured", "max"),
        temp_mean=("Temperature_measured", "mean"),
        capacity=("Capacity", "first"),
    ).reset_index()

    # Health label: lower-half capacity = "degraded". Median split keeps classes balanced.
    threshold = cyc["capacity"].median()
    cyc["degraded"] = (cyc["capacity"] < threshold).astype(int)
    cyc.attrs["health_threshold"] = float(threshold)
    return cyc


def build_rul_table(df):
    """
    Per-cycle table labelled with Remaining Useful Life (RUL) = cycles left until the cell
    reaches end-of-life (capacity <= EOL_CAPACITY).

    For each battery we find the cycle where capacity first drops to EOL, then label every
    earlier cycle with how many cycles remain. Batteries that never reach EOL are skipped
    (their RUL is unknown / censored). Keeps 'Battery' so evaluation can split by cell.
    """
    cyc = build_cycle_table(df)
    parts = []
    for battery, g in cyc.groupby("Battery"):
        g = g.sort_values("id_cycle").reset_index(drop=True)
        below = g[g["capacity"] <= EOL_CAPACITY]
        if below.empty:
            continue
        eol_cycle = below["id_cycle"].iloc[0]
        g = g[g["id_cycle"] <= eol_cycle].copy()
        g["RUL"] = eol_cycle - g["id_cycle"]
        parts.append(g)
    return pd.concat(parts, ignore_index=True)
