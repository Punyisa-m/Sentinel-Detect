"""
Sentinel-Detect | api.py
==========================
FastAPI backend — ML scoring separated from Streamlit UI.

Endpoints
─────────
GET  /health              System health, RAM, model status
POST /score/event         Score a single event → persist + create alert
POST /score/batch         Score a list of events (memory-efficient loop)
GET  /alerts              Paginated alert log with optional status filter
PATCH /alerts/{id}/review Whitelist / Confirm Fraud / Close an alert
GET  /stats/kpis          Live KPI summary
GET  /stats/daily         Historical daily stats (last N days)

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
"""

import os
import sys
import time
import sqlite3
from contextlib  import asynccontextmanager
from datetime    import datetime
from pathlib     import Path
from typing      import Optional

import joblib
import numpy  as np
import pandas as pd
from fastapi             import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic            import BaseModel, Field
from loguru              import logger

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from model_engine import (
    train_model, score_single_event, get_integrity_score,
    FEATURE_COLS, ANOMALY_THRESHOLD, MODEL_PATH, SCALER_PATH,
    explain_anomaly,
)
from data_generator import DB_PATH

# ── Loguru setup ──────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    level="INFO",
    colorize=True,
)
logger.add(
    "logs/sentinel_api.log",
    rotation="10 MB",
    retention="14 days",
    compression="gz",
    level="DEBUG",
    enqueue=True,   # non-blocking async write
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class EventIn(BaseModel):
    employee_id:        str
    department:         str
    action_type:        str
    timestamp:          str   = Field(default_factory=lambda: datetime.now().isoformat())
    duration_minutes:   float = Field(ge=0)
    security_level:     int   = Field(ge=1, le=4)
    transaction_amount: float = Field(default=0.0, ge=0)
    is_anomaly:         int   = Field(default=0)


class BatchIn(BaseModel):
    events: list[EventIn]


class ReviewIn(BaseModel):
    new_status:  str   # WHITELISTED | CONFIRMED_FRAUD | CLOSED | OPEN
    reviewed_by: str   = "analyst"
    review_note: str   = ""


# ── Alert helpers (direct SQLite — no database.py dependency) ─────────────────
def _get_alert_counts() -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        r = conn.execute("""
            SELECT
                COUNT(*)                          AS total,
                SUM(status='OPEN')                AS open,
                SUM(status='WHITELISTED')         AS whitelisted,
                SUM(status='CONFIRMED_FRAUD')     AS fraud,
                SUM(status='CLOSED')              AS closed
            FROM alert_logs
        """).fetchone()
        return dict(r) if r else {}


def _get_alerts_page(status=None, limit=100, offset=0) -> list[dict]:
    where = f"status='{status}'" if status else "1=1"
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT * FROM alert_logs WHERE {where} "
            f"ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]


def _create_alert(log_id, employee_id, department, action_type,
                  event_timestamp, anomaly_score, reason) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        # Prevent duplicate alert for same log_id
        ex = conn.execute(
            "SELECT id FROM alert_logs WHERE log_id=?", (log_id,)
        ).fetchone()
        if ex:
            return ex["id"]
        cur = conn.execute("""
            INSERT INTO alert_logs
                (log_id, employee_id, department, action_type,
                 event_timestamp, anomaly_score, reason, status)
            VALUES (?,?,?,?,?,?,?,'OPEN')
        """, (log_id, employee_id, department, action_type,
               event_timestamp, anomaly_score, reason))
        return cur.lastrowid


def _review_alert(alert_id, new_status, reviewed_by, review_note) -> bool:
    valid = {"WHITELISTED", "CONFIRMED_FRAUD", "CLOSED", "OPEN"}
    if new_status not in valid:
        raise ValueError(f"Invalid status: {new_status}")
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("""
            UPDATE alert_logs
               SET status=?, reviewed_by=?, review_note=?,
                   reviewed_at=datetime('now')
             WHERE id=?
        """, (new_status, reviewed_by, review_note, alert_id))
        return cur.rowcount > 0


def _ensure_alert_logs_table():
    """Create alert_logs if it doesn't exist yet."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alert_logs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                log_id          INTEGER,
                employee_id     TEXT,
                department      TEXT,
                action_type     TEXT,
                event_timestamp TEXT,
                anomaly_score   REAL,
                reason          TEXT,
                status          TEXT NOT NULL DEFAULT 'OPEN',
                reviewed_by     TEXT,
                review_note     TEXT,
                reviewed_at     TEXT,
                created_at      TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_al_status ON alert_logs(status)"
        )


def _upsert_daily_stats():
    today = datetime.now().date().isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        # Ensure table exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_stats (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                stat_date       TEXT NOT NULL UNIQUE,
                total_events    INTEGER DEFAULT 0,
                total_anomalies INTEGER DEFAULT 0,
                confirmed_fraud INTEGER DEFAULT 0,
                whitelisted     INTEGER DEFAULT 0,
                integrity_score REAL    DEFAULT 100.0,
                created_at      TEXT DEFAULT (datetime('now'))
            )
        """)
        t = conn.execute(
            "SELECT COUNT(*) AS total, SUM(is_anomaly_pred) AS anom "
            "FROM activity_logs WHERE DATE(ingested_at)=?", (today,)
        ).fetchone()
        f = conn.execute(
            "SELECT COUNT(*) AS cnt FROM alert_logs "
            "WHERE DATE(created_at)=? AND status='CONFIRMED_FRAUD'", (today,)
        ).fetchone()
        w = conn.execute(
            "SELECT COUNT(*) AS cnt FROM alert_logs "
            "WHERE DATE(created_at)=? AND status='WHITELISTED'", (today,)
        ).fetchone()

        total = int(t["total"] or 0)
        anom  = int(t["anom"]  or 0)
        score = round(max(0, 100 * (1 - anom / max(total, 1)) - (f["cnt"] or 0) * 0.5), 1)

        conn.execute("""
            INSERT INTO daily_stats
                (stat_date, total_events, total_anomalies,
                 confirmed_fraud, whitelisted, integrity_score)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(stat_date) DO UPDATE SET
                total_events    = excluded.total_events,
                total_anomalies = excluded.total_anomalies,
                confirmed_fraud = excluded.confirmed_fraud,
                whitelisted     = excluded.whitelisted,
                integrity_score = excluded.integrity_score
        """, (today, total, anom, f["cnt"] or 0, w["cnt"] or 0, score))


# ── Cached model artefacts ────────────────────────────────────────────────────
_model = _scaler = _feature_stats = None


def load_artifacts():
    global _model, _scaler, _feature_stats
    if _model is None:
        if not Path(MODEL_PATH).exists():
            logger.warning("Model not found — training now …")
            train_model(DB_PATH)
        _model         = joblib.load(MODEL_PATH)
        _scaler        = joblib.load(SCALER_PATH)
        _feature_stats = joblib.load("sentinel_feature_stats.joblib")
        logger.success("Model artefacts loaded into memory ✓")
    return _model, _scaler, _feature_stats


# ── App lifespan ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Sentinel-Detect API starting …")
    _ensure_alert_logs_table()
    load_artifacts()
    yield
    logger.info("🛑 API shutting down")


app = FastAPI(
    title       = "Sentinel-Detect API",
    version     = "2.0.0",
    description = "AI-powered employee behaviour anomaly detection",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── GET /health ───────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    """
    Returns system health — memory, model status, live KPIs.
    Used by Docker HEALTHCHECK and external monitors.
    """
    try:
        try:
            import psutil
            mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
        except Exception:
            mem_mb = -1

        model_ok = Path(MODEL_PATH).exists()
        db_ok    = Path(DB_PATH).exists()
        status   = "healthy" if (model_ok and db_ok) else "degraded"

        # Quick KPI from DB
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            r = conn.execute(
                "SELECT COUNT(*) AS total, SUM(is_anomaly_pred) AS flagged "
                "FROM activity_logs"
            ).fetchone()
        total   = int(r["total"]   or 0)
        flagged = int(r["flagged"] or 0)

        logger.debug("Health check → {}", status)
        return {
            "status":       status,
            "timestamp":    datetime.now().isoformat(),
            "model_loaded": _model is not None,
            "model_file":   model_ok,
            "db_file":      db_ok,
            "memory_mb":    round(mem_mb, 1),
            "kpis": {
                "total":   total,
                "flagged": flagged,
                "score":   round(max(0, 100 * (1 - flagged / max(total, 1))), 1),
            },
        }
    except Exception as e:
        logger.error("Health check error: {}", e)
        return {"status": "error", "detail": str(e)}


# ── POST /score/event ─────────────────────────────────────────────────────────
@app.post("/score/event", tags=["Scoring"])
def score_event(event: EventIn):
    """
    Score a single activity log event.
    Persists to DB and creates an alert_log entry if anomalous.
    """
    t0 = time.perf_counter()
    result = score_single_event(event.model_dump(), DB_PATH)

    alert_id = None
    if result["is_anomaly_pred"] and result["anomaly_score"] < ANOMALY_THRESHOLD:
        # score_single_event already inserts activity_log → get last log_id
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                "SELECT id FROM activity_logs ORDER BY id DESC LIMIT 1"
            ).fetchone()
            log_id = row[0] if row else 0

        alert_id = _create_alert(
            log_id          = log_id,
            employee_id     = event.employee_id,
            department      = event.department,
            action_type     = event.action_type,
            event_timestamp = event.timestamp,
            anomaly_score   = result["anomaly_score"],
            reason          = result.get("anomaly_reason") or "",
        )
        _upsert_daily_stats()

    ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info(
        "Score | {} | {} | score={:.4f} | pred={} | {}ms",
        event.employee_id, event.action_type,
        result["anomaly_score"], result["is_anomaly_pred"], ms,
    )
    return {**result, "alert_id": alert_id, "latency_ms": ms}


# ── POST /score/batch ─────────────────────────────────────────────────────────
@app.post("/score/batch", tags=["Scoring"])
def score_batch(batch: BatchIn):
    """
    Score a list of events one at a time — constant memory regardless of batch size.
    """
    results = []
    t0      = time.perf_counter()

    for ev in batch.events:
        result = score_single_event(ev.model_dump(), DB_PATH)
        aid    = None
        if result["is_anomaly_pred"] and result["anomaly_score"] < ANOMALY_THRESHOLD:
            with sqlite3.connect(DB_PATH) as conn:
                row = conn.execute(
                    "SELECT id FROM activity_logs ORDER BY id DESC LIMIT 1"
                ).fetchone()
                log_id = row[0] if row else 0
            aid = _create_alert(
                log_id=log_id, employee_id=ev.employee_id,
                department=ev.department, action_type=ev.action_type,
                event_timestamp=ev.timestamp, anomaly_score=result["anomaly_score"],
                reason=result.get("anomaly_reason") or "",
            )
        results.append({**result, "alert_id": aid})

    _upsert_daily_stats()
    flagged = sum(1 for r in results if r["is_anomaly_pred"])
    ms      = round((time.perf_counter() - t0) * 1000, 2)
    logger.info("Batch | {} events | {} flagged | {}ms", len(results), flagged, ms)
    return {
        "processed":  len(results),
        "flagged":    flagged,
        "latency_ms": ms,
        "results":    results,
    }


# ── GET /alerts ───────────────────────────────────────────────────────────────
@app.get("/alerts", tags=["Alerts"])
def get_alerts(
    status: Optional[str] = Query(None),
    limit:  int           = Query(100, ge=1, le=500),
    offset: int           = Query(0, ge=0),
):
    return {
        "counts": _get_alert_counts(),
        "alerts": _get_alerts_page(status, limit, offset),
    }


# ── PATCH /alerts/{id}/review ─────────────────────────────────────────────────
@app.patch("/alerts/{alert_id}/review", tags=["Alerts"])
def review(alert_id: int, body: ReviewIn):
    """
    Update alert status.
    WHITELISTED     → false positive, suppress
    CONFIRMED_FRAUD → escalate to HR / Legal
    CLOSED          → reviewed and resolved
    """
    ok = _review_alert(alert_id, body.new_status, body.reviewed_by, body.review_note)
    if not ok:
        raise HTTPException(status_code=404, detail="Alert not found")
    _upsert_daily_stats()
    logger.info("Alert {} → {} by {}", alert_id, body.new_status, body.reviewed_by)
    return {"ok": True, "alert_id": alert_id, "new_status": body.new_status}


# ── GET /stats/kpis ───────────────────────────────────────────────────────────
@app.get("/stats/kpis", tags=["Stats"])
def kpis():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        r = conn.execute(
            "SELECT COUNT(*) AS total, SUM(is_anomaly_pred) AS flagged "
            "FROM activity_logs"
        ).fetchone()
    total   = int(r["total"]   or 0)
    flagged = int(r["flagged"] or 0)
    return {
        "total":   total,
        "flagged": flagged,
        "rate":    round(flagged / max(total, 1), 4),
        "score":   round(max(0, 100 * (1 - flagged / max(total, 1))), 1),
    }


# ── GET /stats/daily ──────────────────────────────────────────────────────────
@app.get("/stats/daily", tags=["Stats"])
def daily_stats(days: int = Query(30, ge=1, le=365)):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM daily_stats ORDER BY stat_date DESC LIMIT ?", (days,)
        ).fetchall()
    return [dict(r) for r in rows]