"""Prediction log - one row per prediction the app serves.

Written to the same Postgres that MLflow uses (table app_predictions), so
Grafana (Step 4) can chart live traffic next to the registry's metrics:
prediction rate by model version, predicted-PNEUMONIA share, and the
brightness/contrast of incoming images - the drift signal that needs no labels.
"""

from __future__ import annotations

import os

from sqlalchemy import (Column, DateTime, Float, Integer, MetaData, String, Table,
                        create_engine, func, insert)

metadata = MetaData()
predictions = Table(
    "app_predictions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("source", String(20)),            # dataset | upload | simulation
    Column("model_version", Integer),
    Column("model_role", String(40)),
    Column("dataset_version", String(10)),   # null for uploads
    Column("image_ref", String(300)),
    Column("true_label", Integer),           # null when unknown (uploads)
    Column("prob_pneumonia", Float),
    Column("predicted", Integer),
    Column("threshold", Float),
    Column("mean_intensity", Float),
    Column("std_intensity", Float),
    Column("latency_ms", Float),
)

_engine = None
_error: str | None = None


def engine():
    """Connect once; if the database is unreachable the app keeps working unlogged."""
    global _engine, _error
    if _engine is None and _error is None:
        url = os.environ.get("MONITORING_DB_URL")
        if not url:
            _error = "MONITORING_DB_URL not set"
            return None
        try:
            _engine = create_engine(url, pool_pre_ping=True)
            metadata.create_all(_engine)
        except Exception as e:  # noqa: BLE001 - surface any DB problem in the UI
            _error = str(e).splitlines()[0]
    return _engine


def status() -> str | None:
    """None when logging works, otherwise the reason it doesn't."""
    engine()
    return _error


def log(**row) -> None:
    eng = engine()
    if eng is None:
        return
    with eng.begin() as conn:
        conn.execute(insert(predictions).values(**row))


def count() -> int:
    eng = engine()
    if eng is None:
        return 0
    with eng.connect() as conn:
        return conn.execute(func.count(predictions.c.id).select()).scalar_one()
