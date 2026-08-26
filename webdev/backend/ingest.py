"""
ingest.py — the ingestion worker. Runs every TICK_SECONDS on a background scheduler.

Each tick it:
  1. Picks a data source automatically (3-tier resolver):
       LIVE     - ThingSpeak has genuinely newer data  -> use it
       REPLAY   - no fresh live data                    -> stream the next demo row
       (last-known is implicit: if a tick produced nothing, the latest DB row simply stays)
     The decision is automatic and invisible — the UI only ever shows "last updated X ago".
  2. Stores the reading in Postgres.
  3. Runs the two active ML models on it and stores the prediction.
  4. Raises alerts (over-temp / anomaly) when warranted.
  5. Prunes old rows so the DB stays small.

This is a simple in-process scheduled task (APScheduler) — NOT a message queue. At one
reading every few seconds, a queue like Kafka would be massive over-engineering.
"""

import threading
import time
from datetime import datetime, timezone

import requests
from sqlalchemy import select, delete, func

import config
import replay
from db import SessionLocal, Reading, Prediction, Alert
from inference import run_inference

# Wall-clock time we last ingested genuinely-new live data (for the staleness fallback).
_last_live_ts = 0.0
# ThingSpeak's own timestamp of the last feed we ingested (to detect NEW data).
_last_feed_time = None
# Cooldown so a sustained condition doesn't spam identical alerts (seconds).
_alert_cooldown = {"over_temp": 0.0, "anomaly": 0.0}
_ALERT_GAP = 20.0


_charge_start_ts = None
_tick_lock = threading.Lock()
_refresh_lock = threading.Lock()
_last_refresh_request = 0.0

def _fetch_thingspeak():
    """
    Return (reading_dict, feed_timestamp) for the newest ThingSpeak feed, or None if
    unconfigured/unavailable. feed_timestamp is ThingSpeak's 'created_at' so the caller
    can tell whether this is genuinely NEW data or the same old value repeated.
    """
    global _charge_start_ts
    if not (config.THINGSPEAK_CHANNEL and config.THINGSPEAK_READ_KEY):
        return None
    try:
        url = (f"https://api.thingspeak.com/channels/{config.THINGSPEAK_CHANNEL}"
               f"/feeds.json?api_key={config.THINGSPEAK_READ_KEY}&results=1")
        feed = requests.get(url, timeout=8).json()["feeds"][-1]
        # field1=power, field2=current, field3=voltage, field4=soc (project convention)
        voltage = float(feed.get("field3") or 0)
        current = float(feed.get("field2") or 0)
        soc = float(feed.get("field4") or 0)
        
        # Simple charge cycle tracker: if current is 0, reset cycle
        if current >= 0:
            _charge_start_ts = None
            time_in_cycle = 0.0
        else:
            now = time.time()
            if _charge_start_ts is None:
                _charge_start_ts = now
            time_in_cycle = now - _charge_start_ts

        reading = {
            "voltage": voltage,
            "current": current,
            "power": float(feed.get("field1") or voltage * current),
            "soc": soc,
            "time_in_cycle": time_in_cycle,
        }
        return reading, feed.get("created_at")
    except Exception:
        return None  # any failure -> treated as "no live data"


def _resolve_source():
    """
    Auto-pick the data source for this tick. Returns (reading_dict, source) or None.

    3-tier priority:
      LIVE       - ThingSpeak has a genuinely newer feed   -> use it
      LAST-KNOWN - live configured but no new data yet (<STALE_SECONDS) -> None
                   (we insert nothing; the dashboard's "last updated" simply grows)
      REPLAY     - no live source, or live has been silent >= STALE_SECONDS -> demo data
    """
    global _last_live_ts, _last_feed_time
    fetched = _fetch_thingspeak()
    if fetched is not None:
        reading, feed_time = fetched
        if _last_feed_time is None or (feed_time and feed_time > _last_feed_time):
            _last_feed_time = feed_time
            _last_live_ts = time.time()
            return reading, "live"

    # No genuinely-new live data this tick.
    live_configured = bool(config.THINGSPEAK_CHANNEL and config.THINGSPEAK_READ_KEY)
    if live_configured and (time.time() - _last_live_ts) < config.STALE_SECONDS:
        return None  # hold on last-known value; don't inject replay yet
    return replay.next_reading(), "replay"


def _maybe_alert(session, reading, pred):
    """Insert over-temp / anomaly alerts, with a per-type cooldown to avoid spam."""
    now = time.time()
    if pred["predicted_temp"] > config.OVER_TEMP_C and now - _alert_cooldown["over_temp"] > _ALERT_GAP:
        session.add(Alert(type="over_temp", severity="critical",
                          message=f"Predicted battery temperature {pred['predicted_temp']:.1f} C "
                                  f"exceeds safe limit {config.OVER_TEMP_C:.0f} C",
                          value=pred["predicted_temp"]))
        _alert_cooldown["over_temp"] = now
    if pred["is_anomaly"] and now - _alert_cooldown["anomaly"] > _ALERT_GAP:
        session.add(Alert(type="anomaly", severity="warning",
                          message="Abnormal operating point detected by anomaly model",
                          value=pred["anomaly_score"]))
        _alert_cooldown["anomaly"] = now


def _tick_once():
    """One ingestion cycle."""
    resolved = _resolve_source()
    if resolved is None:
        return  # last-known mode: nothing new to store this tick
    reading_dict, source = resolved

    # The displayed reading is whatever the source gave (pack-level for live). But the ML
    # models expect CELL-level inputs, so for live data we scale pack -> per-cell first.
    if source == "live":
        infer_v = reading_dict["voltage"] / config.CELLS_IN_SERIES
        infer_i = reading_dict["current"] / config.CELLS_IN_PARALLEL
    else:
        infer_v, infer_i = reading_dict["voltage"], reading_dict["current"]

    session = SessionLocal()
    try:
        r = Reading(ts=datetime.now(timezone.utc),
                    voltage=reading_dict["voltage"], current=reading_dict["current"],
                    power=reading_dict["power"], soc=reading_dict["soc"], source=source)
        session.add(r)
        session.flush()  # assigns r.id

        pred = run_inference(infer_v, infer_i, reading_dict["time_in_cycle"])
        session.add(Prediction(reading_id=r.id, **pred))
        _maybe_alert(session, r, pred)

        # prune: keep only the newest MAX_READINGS readings (+ their predictions)
        count = session.scalar(select(func.count(Reading.id)))
        if count > config.MAX_READINGS:
            cutoff = session.scalar(
                select(Reading.id).order_by(Reading.id.desc())
                .offset(config.MAX_READINGS).limit(1))
            if cutoff:
                session.execute(delete(Prediction).where(Prediction.reading_id <= cutoff))
                session.execute(delete(Reading).where(Reading.id <= cutoff))

        session.commit()
    except Exception as e:
        session.rollback()
        print("ingest tick error:", e)
    finally:
        session.close()


def tick():
    """Run one ingestion cycle, serialized against API-triggered refreshes."""
    with _tick_lock:
        return _tick_once()


def ensure_fresh():
    """Advance the source when the last refresh request is older than one tick.

    The scheduler remains the primary mechanism. This lightweight guard makes the
    dashboard resilient on hosts that pause or delay background threads: a normal
    latest-reading request can restart replay progress without creating duplicate
    rows for every API call.
    """
    global _last_refresh_request
    now = time.monotonic()
    with _refresh_lock:
        if now - _last_refresh_request < max(1, config.TICK_SECONDS):
            return
        _last_refresh_request = now
    tick()
