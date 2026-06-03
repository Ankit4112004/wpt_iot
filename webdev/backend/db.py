"""
db.py — the database layer: connection + the 4 tables (SQLAlchemy ORM).

The same code works on SQLite (local dev) and Postgres (production) — only the
connection string in config.DATABASE_URL changes.

The 4 tables and why each exists:
  - Channel    : the data source config (ThingSpeak creds) — keeps keys out of the browser.
  - Reading    : every telemetry sample (V, I, power, SOC) — the source of truth.
  - Prediction : the ML output for each reading (predicted temp, anomaly, health).
  - Alert      : safety/anomaly events shown on the dashboard's alerts panel.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    create_engine, ForeignKey, String, Integer, Float, Boolean, DateTime, func
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, sessionmaker, relationship
)

import config

# SQLite needs this flag because the scheduler thread and web thread share the connection.
connect_args = {"check_same_thread": False} if config.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(config.DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def _now():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Channel(Base):
    __tablename__ = "channels"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    thingspeak_channel: Mapped[str] = mapped_column(String(40), default="")
    read_key: Mapped[str] = mapped_column(String(40), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Reading(Base):
    __tablename__ = "readings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    voltage: Mapped[float] = mapped_column(Float)
    current: Mapped[float] = mapped_column(Float)
    power: Mapped[float] = mapped_column(Float)
    soc: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(16))  # 'live' | 'replay'
    prediction: Mapped["Prediction"] = relationship(back_populates="reading", uselist=False)


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    reading_id: Mapped[int] = mapped_column(ForeignKey("readings.id"), index=True)
    predicted_temp: Mapped[float] = mapped_column(Float)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, default=False)
    anomaly_score: Mapped[float] = mapped_column(Float, default=0.0)
    health_label: Mapped[str] = mapped_column(String(16), default="Healthy")
    degraded_probability: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    reading: Mapped["Reading"] = relationship(back_populates="prediction")


class Alert(Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    type: Mapped[str] = mapped_column(String(24))      # 'over_temp' | 'anomaly'
    severity: Mapped[str] = mapped_column(String(16))  # 'warning' | 'critical'
    message: Mapped[str] = mapped_column(String(160))
    value: Mapped[float] = mapped_column(Float, default=0.0)


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(engine)
