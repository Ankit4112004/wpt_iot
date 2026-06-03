"""
config.py — all backend settings in one place, read from environment variables.

Locally everything has sensible defaults (SQLite, replay mode) so the app runs with zero
setup. In production you set DATABASE_URL to the Neon Postgres connection string.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # loads a .env file if present (ignored in git)

BASE_DIR = os.path.dirname(__file__)

# Database: SQLite file by default (no install needed); set to a Postgres URL in prod.
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'wpt.db')}")

# How often the ingestion worker produces a new reading (seconds).
TICK_SECONDS = int(os.getenv("TICK_SECONDS", "3"))

# If ThingSpeak gives no new data for this long, we auto-fall back to replay.
STALE_SECONDS = int(os.getenv("STALE_SECONDS", "60"))

# Safety threshold: predicted battery temperature above this raises an over-temp alert.
OVER_TEMP_C = float(os.getenv("OVER_TEMP_C", "38"))

# Keep only the most recent N readings so the DB doesn't grow forever (replay loops).
MAX_READINGS = int(os.getenv("MAX_READINGS", "5000"))

# Optional real IoT source. If unset, the app runs purely on replay (great for demos).
THINGSPEAK_CHANNEL = os.getenv("THINGSPEAK_CHANNEL", "")
THINGSPEAK_READ_KEY = os.getenv("THINGSPEAK_READ_KEY", "")

# Pack -> cell conversion. The real ThingSpeak telemetry is PACK level (~390 V, ~-10 A),
# but the ML models were trained on single-CELL data (~3.7 V, ~-2 A). Before inference we
# divide the live pack readings down to per-cell values so they land in the model's
# trained range. (390 V / 105 in series ≈ 3.7 V; -10 A / 5 parallel ≈ -2 A.)
# Replay data is already cell-level, so this only applies to live data.
CELLS_IN_SERIES = int(os.getenv("CELLS_IN_SERIES", "105"))
CELLS_IN_PARALLEL = int(os.getenv("CELLS_IN_PARALLEL", "5"))

# Compact demo dataset that the replay engine streams.
DEMO_CSV = os.path.join(BASE_DIR, "data", "demo_cycles.csv")
