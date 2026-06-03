# Backend — FastAPI + Postgres

The system's source of truth. It ingests battery telemetry, runs the 3 ML models on every
reading, stores everything, and serves the dashboard. The frontend talks **only** to this
API — never to ThingSpeak directly — which is what makes the dashboard resilient.

## Run locally (zero setup — uses SQLite + replay)

```bash
pip install -r requirements.txt
uvicorn main:app --reload          # http://localhost:8000
```
Interactive API docs: **http://localhost:8000/docs**

No database to install and no ThingSpeak needed: it defaults to a local SQLite file and
streams a recorded battery cycle on a loop.

## Architecture

```
ThingSpeak (optional) ─┐
                       ▼
            ingestion worker (every 3s, APScheduler)
                       │  auto-picks source:  LIVE → REPLAY → last-known
                       ▼
   Postgres  ◄── reading + ML prediction + alerts written each tick
       ▲
       │  REST
   FastAPI  ──►  React dashboard
```

## The 3-tier resolver (the resilience trick)
Each tick the worker tries ThingSpeak; if there's no fresh live data it automatically
streams the next recorded reading instead. The decision is automatic and invisible — the
UI only shows **"last updated X ago"**. So the demo always animates, even with no live
source, and an upstream outage just degrades to last-known-value.

## Endpoints
- `GET /health` — status + whether models loaded
- `GET /api/readings/latest` — newest reading + prediction + staleness
- `GET /api/readings?limit=` — recent series for the charts
- `GET /api/alerts?limit=` — over-temp / anomaly events
- `GET /api/analytics/summary` — headline KPIs

## Files
- `config.py` — settings from env (SQLite/replay defaults).
- `db.py` — SQLAlchemy connection + 4 tables (channels, readings, predictions, alerts).
- `inference.py` — runs the 3 ML models on a reading (imports `ml/predict.py`).
- `replay.py` — streams the recorded demo cycles in a loop.
- `ingest.py` — the scheduled worker + 3-tier source resolver + alert logic + pruning.
- `main.py` — FastAPI app, endpoints, startup wiring.

## Production
Set `DATABASE_URL` to a Neon Postgres connection string; everything else is unchanged.
The worker is a simple in-process scheduled task (not a message queue) — correct for a
single low-rate source.
