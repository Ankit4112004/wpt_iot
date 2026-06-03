"""
replay.py — streams the compact demo dataset, looping forever.

This is the demo safety-net: when no live ThingSpeak data is arriving, the ingestion
worker pulls the next row from here, so the dashboard always animates with realistic
data — no MATLAB, no ThingSpeak, no internet needed.

How to explain this file in an interview:
  - "It replays real recorded battery cycles in a loop, so the demo is fully deterministic
     and works offline."
"""

import csv
import config

# Load the demo rows once into memory.
with open(config.DEMO_CSV, newline="") as f:
    _ROWS = list(csv.DictReader(f))

_index = 0


def next_reading():
    """Return the next demo reading (loops back to the start at the end)."""
    global _index
    row = _ROWS[_index % len(_ROWS)]
    _index += 1
    return {
        "voltage": float(row["voltage"]),
        "current": float(row["current"]),
        "power": float(row["power"]),
        "soc": float(row["soc"]),
        "time_in_cycle": float(row["time_in_cycle"]),
    }
