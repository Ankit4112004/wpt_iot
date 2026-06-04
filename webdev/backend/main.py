"""
main.py — the FastAPI application: endpoints + startup wiring.

Run locally:  uvicorn main:app --reload   (from webdev/backend/)
Interactive API docs are auto-generated at  http://localhost:8000/docs

The frontend talks ONLY to these endpoints — never to ThingSpeak directly. That's what
makes the dashboard resilient: it always reads from our database.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import select, desc
from sqlalchemy.orm import joinedload

import config
from db import SessionLocal, init_db, Channel, Reading, Prediction, Alert
from ingest import tick
from inference import models_ready, what_if

scheduler = BackgroundScheduler(daemon=True)


def _seconds_ago(ts: datetime) -> int:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return int((datetime.now(timezone.utc) - ts).total_seconds())


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    init_db()
    with SessionLocal() as s:
        if not s.scalar(select(Channel).limit(1)):
            s.add(Channel(name="Simulink WPT Channel",
                          thingspeak_channel=config.THINGSPEAK_CHANNEL,
                          read_key=config.THINGSPEAK_READ_KEY))
            s.commit()
    tick()  # produce one reading immediately so the DB is never empty on first request
    scheduler.add_job(tick, "interval", seconds=config.TICK_SECONDS, id="ingest",
                      max_instances=1, coalesce=True)
    scheduler.start()
    yield
    # --- shutdown ---
    scheduler.shutdown(wait=False)


app = FastAPI(title="EV WPT Monitor API", version="2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": models_ready(),
            "tick_seconds": config.TICK_SECONDS}


@app.get("/api/readings/latest")
def latest_reading():
    """Newest reading + its ML prediction + how stale it is (for the 'last updated' label)."""
    with SessionLocal() as s:
        r = s.scalar(select(Reading).options(joinedload(Reading.prediction)).order_by(desc(Reading.id)).limit(1))
        if not r:
            return {"available": False}
        p = r.prediction
        return {
            "available": True,
            "ts": r.ts.isoformat(),
            "last_updated_seconds": _seconds_ago(r.ts),
            "source": r.source,
            "voltage": round(r.voltage, 3), "current": round(r.current, 3),
            "power": round(r.power, 2), "soc": round(r.soc, 2),
            "predicted_temp": p.predicted_temp if p else None,
            "is_anomaly": p.is_anomaly if p else None,
            "health_label": p.health_label if p else None,
            "degraded_probability": p.degraded_probability if p else None,
        }


@app.get("/api/readings")
def readings(limit: int = 100):
    """Recent readings (oldest -> newest) with predicted temperature, for the charts."""
    limit = max(1, min(limit, 500))
    with SessionLocal() as s:
        rows = s.scalars(select(Reading).options(joinedload(Reading.prediction)).order_by(desc(Reading.id)).limit(limit)).all()
        rows = list(reversed(rows))
        return [{
            "ts": r.ts.isoformat(),
            "voltage": round(r.voltage, 3), "current": round(r.current, 3),
            "power": round(r.power, 2), "soc": round(r.soc, 2),
            "predicted_temp": r.prediction.predicted_temp if r.prediction else None,
        } for r in rows]


@app.get("/api/whatif")
def whatif(voltage: float, current: float, time_in_cycle: float = 1800.0):
    """Interactive 'what-if': given electrical inputs, return the live ML outputs.

    Not stored in the DB — it's an on-demand prediction for the dashboard's slider panel.
    Inputs are per-cell values (the model's native range): voltage ~2.5-4.2 V,
    current negative for discharge.
    """
    return what_if(voltage, current, time_in_cycle)


@app.get("/api/alerts")
def alerts(limit: int = 20):
    with SessionLocal() as s:
        rows = s.scalars(select(Alert).order_by(desc(Alert.id)).limit(limit)).all()
        return [{
            "ts": a.ts.isoformat(), "seconds_ago": _seconds_ago(a.ts),
            "type": a.type, "severity": a.severity,
            "message": a.message, "value": round(a.value, 3),
        } for a in rows]


@app.get("/api/analytics/summary")
def summary():
    """Headline KPIs for the dashboard top bar."""
    with SessionLocal() as s:
        latest = s.scalar(select(Reading).options(joinedload(Reading.prediction)).order_by(desc(Reading.id)).limit(1))
        recent = s.scalars(select(Reading).options(joinedload(Reading.prediction)).order_by(desc(Reading.id)).limit(100)).all()
        n_alerts = s.scalar(select(Alert).order_by(desc(Alert.id)).limit(1))
        temps = [r.prediction.predicted_temp for r in recent if r.prediction]
        powers = [r.power for r in recent]
        return {
            "current_soc": round(latest.soc, 1) if latest else None,
            "predicted_temp": latest.prediction.predicted_temp if latest and latest.prediction else None,
            "health_label": latest.prediction.health_label if latest and latest.prediction else None,
            "peak_temp_recent": round(max(temps), 2) if temps else None,
            "avg_power_recent": round(sum(powers) / len(powers), 2) if powers else None,
            "readings_window": len(recent),
            "has_active_alert": n_alerts is not None and _seconds_ago(n_alerts.ts) < 60,
        }
