"""
Sentinel-Detect | model_engine.py
=====================================
Isolation Forest anomaly detection engine with built-in explainability (XAI).
Scores every activity log, writes results back to SQLite, and triggers
Audit Tasks when the anomaly score crosses the configured threshold.
"""

import sqlite3
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
DB_PATH        = "sentinel.db"
MODEL_PATH     = "sentinel_model.joblib"
SCALER_PATH    = "sentinel_scaler.joblib"
ANOMALY_THRESHOLD = -0.10          # Isolation Forest score below this → anomaly
CONTAMINATION     = 0.08           # Expected fraction of anomalies in training data

FEATURE_COLS = [
    "freq_score",
    "duration_minutes",
    "security_level",
    "transaction_amount",
    "hour_of_day",
    "action_encoded",
]

# Human-readable feature labels for XAI
FEATURE_LABELS = {
    "freq_score":         "Event Frequency (last 60 min)",
    "duration_minutes":   "Task Duration (minutes)",
    "security_level":     "Document Security Level",
    "transaction_amount": "Transaction Amount ($)",
    "hour_of_day":        "Hour of Day",
    "action_encoded":     "Action Type",
}


# ─────────────────────────────────────────────
#  MODEL TRAINING
# ─────────────────────────────────────────────
def train_model(db_path: str = DB_PATH) -> tuple:
    """
    Loads activity logs from SQLite, trains an Isolation Forest model,
    and persists both the model and the feature scaler.

    Returns (model, scaler, feature_stats)
    where feature_stats contains per-feature mean/std for XAI.
    """
    conn = sqlite3.connect(db_path)
    df   = pd.read_sql("SELECT * FROM activity_logs", conn)
    conn.close()

    if df.empty:
        raise ValueError("No logs found in DB. Run data_generator.py first.")

    df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)

    X = df[FEATURE_COLS].values

    # Standardise features so no single feature dominates
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Per-feature stats used by XAI explainer
    feature_stats = {
        col: {
            "mean": float(scaler.mean_[i]),
            "std":  float(np.sqrt(scaler.var_[i])),
        }
        for i, col in enumerate(FEATURE_COLS)
    }

    # Train Isolation Forest
    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # Persist artefacts
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_stats, "sentinel_feature_stats.joblib")

    print(f"[model_engine] Model trained on {len(df)} records.")
    print(f"[model_engine] Artefacts saved → {MODEL_PATH}, {SCALER_PATH}")
    return model, scaler, feature_stats


# ─────────────────────────────────────────────
#  XAI EXPLAINER
# ─────────────────────────────────────────────
def explain_anomaly(row: pd.Series, feature_stats: dict) -> str:
    """
    Produces a human-readable explanation for a flagged anomaly by
    identifying which features deviate most from the training distribution.

    Approach: z-score each feature value; report top deviating features
    with the magnitude and direction of the deviation.
    """
    deviations = []
    for col in FEATURE_COLS:
        val   = row[col]
        mean  = feature_stats[col]["mean"]
        std   = feature_stats[col]["std"]
        if std < 1e-9:
            continue
        z = (val - mean) / std
        if abs(z) >= 1.5:          # only meaningful deviations
            direction = "above" if z > 0 else "below"
            deviations.append((abs(z), col, val, z, direction))

    if not deviations:
        return "Combination of features collectively deviates from normal patterns."

    deviations.sort(reverse=True)
    parts = []
    for _, col, val, z, direction in deviations[:3]:
        label = FEATURE_LABELS[col]
        if col == "transaction_amount":
            display_val = f"${val:,.2f}"
        elif col == "duration_minutes":
            display_val = f"{val:.2f} min"
        elif col == "hour_of_day":
            display_val = f"{int(val)}:00"
        else:
            display_val = f"{val:.2f}"
        parts.append(
            f"'{label}' = {display_val} "
            f"({abs(z):.1f}σ {direction} mean)"
        )

    return "; ".join(parts)


# ─────────────────────────────────────────────
#  BATCH SCORING
# ─────────────────────────────────────────────
def score_logs(db_path: str = DB_PATH, retrain: bool = False):
    """
    Scores all unscored activity_logs rows, writes anomaly_score +
    is_anomaly_pred back, and creates Audit Tasks for flagged records.
    """
    # Load or train artefacts
    if retrain or not Path(MODEL_PATH).exists():
        model, scaler, feature_stats = train_model(db_path)
    else:
        model         = joblib.load(MODEL_PATH)
        scaler        = joblib.load(SCALER_PATH)
        feature_stats = joblib.load("sentinel_feature_stats.joblib")

    conn = sqlite3.connect(db_path)
    df   = pd.read_sql(
        "SELECT * FROM activity_logs WHERE anomaly_score IS NULL", conn
    )

    if df.empty:
        print("[model_engine] No unscored records.")
        conn.close()
        return pd.DataFrame()

    df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
    X_scaled = scaler.transform(df[FEATURE_COLS].values)

    # score_samples returns negative values; more negative = more anomalous
    raw_scores       = model.score_samples(X_scaled)
    predictions      = model.predict(X_scaled)          # -1 = anomaly, 1 = normal
    df["anomaly_score"]   = raw_scores.round(4)
    df["is_anomaly_pred"] = (predictions == -1).astype(int)

    # Generate XAI reasons for flagged rows
    df["xai_reason"] = ""
    flagged = df[df["is_anomaly_pred"] == 1]
    for idx, row in flagged.iterrows():
        df.at[idx, "xai_reason"] = explain_anomaly(row, feature_stats)

    # Write scores back to DB
    cur = conn.cursor()
    for _, row in df.iterrows():
        cur.execute("""
            UPDATE activity_logs
               SET anomaly_score    = ?,
                   is_anomaly_pred  = ?,
                   anomaly_reason   = COALESCE(anomaly_reason, ?)
             WHERE id = ?
        """, (
            row["anomaly_score"],
            row["is_anomaly_pred"],
            row["xai_reason"] if row["is_anomaly_pred"] == 1 else None,
            row["id"],
        ))

    # ── Trigger Audit Tasks ──────────────────
    audit_created = 0
    for _, row in flagged.iterrows():
        if row["anomaly_score"] < ANOMALY_THRESHOLD:
            cur.execute("""
                INSERT INTO audit_tasks
                    (log_id, employee_id, department, action_type,
                     timestamp, anomaly_score, reason, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN')
            """, (
                int(row["id"]),
                row["employee_id"],
                row["department"],
                row["action_type"],
                str(row["timestamp"]),
                float(row["anomaly_score"]),
                row["xai_reason"],
            ))
            audit_created += 1

    conn.commit()
    conn.close()

    total_flagged = int(df["is_anomaly_pred"].sum())
    print(f"[model_engine] Scored {len(df)} records → "
          f"{total_flagged} anomalies detected, {audit_created} audit tasks created.")
    return df[df["is_anomaly_pred"] == 1]


# ─────────────────────────────────────────────
#  REAL-TIME SINGLE-EVENT SCORING
# ─────────────────────────────────────────────
def score_single_event(event: dict, db_path: str = DB_PATH) -> dict:
    """
    Scores a single incoming event dict in real-time.
    Inserts into activity_logs and, if anomalous, creates an Audit Task.

    Returns the event dict enriched with anomaly_score, is_anomaly_pred,
    and anomaly_reason.
    """
    if not Path(MODEL_PATH).exists():
        train_model(db_path)

    model         = joblib.load(MODEL_PATH)
    scaler        = joblib.load(SCALER_PATH)
    feature_stats = joblib.load("sentinel_feature_stats.joblib")

    # Compute freq_score (simple: recent events in DB for this employee)
    conn     = sqlite3.connect(db_path)
    emp_id   = event.get("employee_id", "UNKNOWN")
    one_hour_ago = (datetime.now().replace(second=0, microsecond=0)).__str__()

    freq_df = pd.read_sql(
        "SELECT COUNT(*) AS cnt FROM activity_logs "
        "WHERE employee_id = ? AND timestamp >= ?",
        conn,
        params=(emp_id, one_hour_ago),
    )
    freq_score = int(freq_df["cnt"].iloc[0]) + 1

    row = {
        "freq_score":         freq_score,
        "duration_minutes":   float(event.get("duration_minutes", 0)),
        "security_level":     int(event.get("security_level", 1)),
        "transaction_amount": float(event.get("transaction_amount", 0) or 0),
        "hour_of_day":        datetime.now().hour,
        "action_encoded":     {"Access Doc": 0, "Submit Task": 1,
                               "Financial Claim": 2}.get(
                                   event.get("action_type"), 0),
    }

    X = np.array([[row[c] for c in FEATURE_COLS]])
    X_scaled     = scaler.transform(X)
    score        = float(model.score_samples(X_scaled)[0])
    prediction   = int(model.predict(X_scaled)[0] == -1)
    row_series   = pd.Series(row)
    xai_reason   = explain_anomaly(row_series, feature_stats) if prediction else None

    # Persist to DB
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO activity_logs
            (employee_id, department, action_type, timestamp,
             duration_minutes, security_level, transaction_amount,
             freq_score, hour_of_day, action_encoded,
             anomaly_score, is_anomaly_pred, is_anomaly_actual, anomaly_reason)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        event.get("employee_id"),
        event.get("department"),
        event.get("action_type"),
        str(event.get("timestamp", datetime.now())),
        row["duration_minutes"],
        row["security_level"],
        row["transaction_amount"],
        freq_score,
        row["hour_of_day"],
        row["action_encoded"],
        round(score, 4),
        prediction,
        int(event.get("is_anomaly", 0)),
        xai_reason,
    ))
    log_id = cur.lastrowid

    # Trigger audit task
    if prediction and score < ANOMALY_THRESHOLD:
        cur.execute("""
            INSERT INTO audit_tasks
                (log_id, employee_id, department, action_type,
                 timestamp, anomaly_score, reason, status)
            VALUES (?,?,?,?,?,?,?,'OPEN')
        """, (
            log_id,
            event.get("employee_id"),
            event.get("department"),
            event.get("action_type"),
            str(event.get("timestamp", datetime.now())),
            round(score, 4),
            xai_reason,
        ))

    conn.commit()
    conn.close()

    return {
        **event,
        "anomaly_score":   round(score, 4),
        "is_anomaly_pred": prediction,
        "anomaly_reason":  xai_reason,
        "freq_score":      freq_score,
    }


# ─────────────────────────────────────────────
#  ANALYTICS HELPERS  (used by dashboard)
# ─────────────────────────────────────────────
def get_integrity_score(db_path: str = DB_PATH) -> float:
    """
    Company-wide Integrity Score (0–100).
    Penalises open audit tasks and high anomaly rates.
    """
    conn = sqlite3.connect(db_path)
    totals = pd.read_sql(
        "SELECT COUNT(*) AS total, SUM(is_anomaly_pred) AS flagged "
        "FROM activity_logs", conn
    )
    open_tasks = pd.read_sql(
        "SELECT COUNT(*) AS cnt FROM audit_tasks WHERE status = 'OPEN'", conn
    )
    conn.close()

    total   = int(totals["total"].iloc[0]) or 1
    flagged = int(totals["flagged"].iloc[0] or 0)
    tasks   = int(open_tasks["cnt"].iloc[0])

    anomaly_rate  = flagged / total
    task_penalty  = min(tasks * 0.5, 20)          # max 20 pts penalty
    raw_score     = 100 * (1 - anomaly_rate) - task_penalty
    return round(max(0.0, min(100.0, raw_score)), 1)


def get_flagged_activities(db_path: str = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df   = pd.read_sql("""
        SELECT id, employee_id, department, action_type, timestamp,
               duration_minutes, security_level, transaction_amount,
               anomaly_score, anomaly_reason
          FROM activity_logs
         WHERE is_anomaly_pred = 1
         ORDER BY anomaly_score ASC
    """, conn)
    conn.close()
    return df


def get_scatter_data(db_path: str = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df   = pd.read_sql("""
        SELECT duration_minutes, security_level, transaction_amount,
               freq_score, anomaly_score, is_anomaly_pred, action_type, department,
               hour_of_day
          FROM activity_logs
         WHERE anomaly_score IS NOT NULL
    """, conn)
    conn.close()
    return df


def get_audit_tasks(db_path: str = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df   = pd.read_sql("""
        SELECT * FROM audit_tasks ORDER BY created_at DESC LIMIT 100
    """, conn)
    conn.close()
    return df


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🤖 Training Isolation Forest model …")
    train_model()

    print("\n🔍 Scoring all activity logs …")
    flagged = score_logs(retrain=False)
    if not flagged.empty:
        print(f"\n⚠️  Top flagged records:\n")
        print(flagged[["employee_id", "department", "action_type",
                        "anomaly_score", "xai_reason"]].head(10).to_string(index=False))

    print(f"\n📊 Integrity Score: {get_integrity_score()}/100")
    