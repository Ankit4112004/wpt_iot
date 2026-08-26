# EV WPT Monitor backend

The FastAPI backend is the source of truth for telemetry. It ingests optional ThingSpeak data, runs the two active ML models on each reading, stores telemetry and predictions, and serves the React dashboard. The frontend never connects to ThingSpeak directly.

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload          # http://localhost:8000
```

The default configuration uses a local SQLite database and the recorded replay stream in `data/demo_cycles.csv`, so local development does not require PostgreSQL or ThingSpeak.

## Data flow

```text
ThingSpeak (optional)
        |
        v
scheduled ingestion worker (every 3 seconds)
        |
        +--> live reading, when new data is available
        +--> replay reading, when live data is unavailable
        +--> last-known value, when a configured live source is stale
        |
        v
SQLite / PostgreSQL
        |
        +--> temperature soft-sensor
        +--> anomaly detector
        |
        v
FastAPI REST API --> React dashboard
```

The ingestion worker also creates over-temperature and anomaly alerts and prunes old readings according to the runtime configuration.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Reports service and model readiness |
| `GET /api/readings/latest` | Returns the newest reading, active prediction, source, and staleness |
| `GET /api/readings?limit=` | Returns recent telemetry and predicted temperature for charts |
| `GET /api/alerts?limit=` | Returns recent over-temperature and anomaly alerts |
| `GET /api/analytics/summary` | Returns dashboard headline metrics |

## Files

| File | Purpose |
|---|---|
| `config.py` | Environment settings and replay/database defaults |
| `db.py` | SQLAlchemy connection and telemetry/prediction/alert tables |
| `inference.py` | Adapter for the temperature and anomaly models |
| `replay.py` | Loops through the recorded demo dataset |
| `ingest.py` | Scheduled source resolution, persistence, inference, alerts, and pruning |
| `main.py` | FastAPI application, lifecycle, and REST endpoints |

## Production

Set `DATABASE_URL` to a Neon PostgreSQL connection string. Set `THINGSPEAK_CHANNEL` and `THINGSPEAK_READ_KEY` only when a live ThingSpeak source is available. If the ThingSpeak variables are blank, the backend intentionally runs in replay mode.
