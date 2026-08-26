"""Data loading and feature engineering for the production ML models.

The active models use one instantaneous feature table: voltage, current, power,
and time. Power is derived from the measured voltage and current.
"""

import os

import pandas as pd

HERE = os.path.dirname(__file__)
RAW_CSV = os.path.join(HERE, "data", "raw", "nasa_discharge.csv")

INSTANT_FEATURES = ["Voltage_measured", "Current_measured", "Power", "Time"]


def load_raw():
    """Load the raw NASA discharge measurements and add the Power column."""
    df = pd.read_csv(RAW_CSV)
    df["Power"] = df["Voltage_measured"] * df["Current_measured"]
    return df


def build_instant_table(df):
    """Return model features and temperature targets for each measurement."""
    X = df[INSTANT_FEATURES].copy()
    y_temp = df["Temperature_measured"].copy()
    return X, y_temp
