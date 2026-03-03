"""
Sentinel-Detect | data_generator.py
=====================================
Generates synthetic enterprise activity logs with realistic patterns
and injected anomalies for training and real-time simulation.
"""

import pandas as pd
import numpy as np
import sqlite3
import random
from datetime import datetime, timedelta
from pathlib import Path

# ─────────────────────────────────────────────
#  CONSTANTS & CONFIGURATION
# ─────────────────────────────────────────────
DB_PATH = "sentinel.db"

DEPARTMENTS = ["Engineering", "Finance", "HR", "Legal", "Marketing", "Operations"]
ACTION_TYPES = ["Access Doc", "Submit Task", "Financial Claim"]
SECURITY_LEVELS = {"Public": 1, "Internal": 2, "Confidential": 3, "Restricted": 4}

# Expected duration ranges (minutes) per action (mean, std)
DURATION_PROFILE = {
    "Access Doc":      {"mean": 15,  "std": 5},
    "Submit Task":     {"mean": 30,  "std": 8},
    "Financial Claim": {"mean": 20,  "std": 6},
}

# Department → allowed max security level
DEPT_CLEARANCE = {
    "Engineering": 3,
    "Finance":     4,
    "HR":          4,
    "Legal":       4,
    "Marketing":   2,
    "Operations":  3,
}

EMPLOYEE_POOL = [f"EMP-{str(i).zfill(4)}" for i in range(1, 51)]


# ─────────────────────────────────────────────
#  NORMAL LOG GENERATOR
# ─────────────────────────────────────────────
def generate_normal_log(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generates baseline normal employee activity logs."""
    rng = np.random.default_rng(seed)
    records = []

    base_time = datetime.now() - timedelta(days=7)

    for _ in range(n):
        emp_id     = random.choice(EMPLOYEE_POOL)
        dept       = random.choice(DEPARTMENTS)
        action     = random.choice(ACTION_TYPES)
        clearance  = DEPT_CLEARANCE[dept]
        sec_level  = rng.integers(1, min(clearance, 3) + 1)  # stay within clearance
        profile    = DURATION_PROFILE[action]
        duration   = max(1, rng.normal(profile["mean"], profile["std"]))
        timestamp  = base_time + timedelta(
            seconds=int(rng.uniform(0, 7 * 24 * 3600))
        )
        amount     = (
            float(rng.normal(500, 150)) if action == "Financial Claim" else None
        )

        records.append({
            "employee_id":       emp_id,
            "department":        dept,
            "action_type":       action,
            "timestamp":         timestamp,
            "duration_minutes":  round(duration, 2),
            "security_level":    sec_level,
            "transaction_amount": round(amount, 2) if amount else None,
            "is_anomaly":        0,
            "anomaly_reason":    None,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
#  ANOMALY INJECTOR
# ─────────────────────────────────────────────
def inject_anomalies(df: pd.DataFrame, n_anomalies: int = 40) -> pd.DataFrame:
    """
    Injects three distinct anomaly patterns into the dataset:

      1. HIGH-FREQUENCY SENSITIVE ACCESS  – employee rapidly accesses
         Restricted docs in a dept they shouldn't.
      2. DURATION OUTLIER                 – task completes in 2 s or drags
         20+ hours, flagging automation or abandonment.
      3. FINANCIAL OUTLIER               – transaction amount is 10-30x
         the normal range, indicating fraud or error.
    """
    anomalies = []
    base_time = datetime.now() - timedelta(hours=2)

    # ── Pattern 1: Rapid sensitive-doc access ──
    rogue_emp = random.choice(EMPLOYEE_POOL[:10])
    for i in range(15):
        anomalies.append({
            "employee_id":        rogue_emp,
            "department":         "Marketing",        # low clearance dept
            "action_type":        "Access Doc",
            "timestamp":          base_time + timedelta(minutes=i * 0.5),
            "duration_minutes":   round(random.uniform(0.1, 0.8), 2),
            "security_level":     4,                  # Restricted – above clearance
            "transaction_amount": None,
            "is_anomaly":         1,
            "anomaly_reason":     "High-frequency access to Restricted docs by low-clearance dept",
        })

    # ── Pattern 2: Duration outliers ──
    for _ in range(15):
        action = random.choice(ACTION_TYPES)
        extreme = random.choice(["fast", "slow"])
        duration = round(random.uniform(0.01, 0.05), 3) if extreme == "fast" \
                   else round(random.uniform(900, 1200), 1)
        anomalies.append({
            "employee_id":        random.choice(EMPLOYEE_POOL[10:30]),
            "department":         random.choice(DEPARTMENTS),
            "action_type":        action,
            "timestamp":          base_time + timedelta(minutes=random.randint(0, 120)),
            "duration_minutes":   duration,
            "security_level":     random.randint(1, 3),
            "transaction_amount": None,
            "is_anomaly":         1,
            "anomaly_reason":     f"Duration is an extreme outlier ({duration} min) for '{action}'",
        })

    # ── Pattern 3: Financial amount outliers ──
    for _ in range(10):
        amount = round(random.uniform(8000, 25000), 2)
        anomalies.append({
            "employee_id":        random.choice(EMPLOYEE_POOL[30:]),
            "department":         random.choice(["Finance", "Operations"]),
            "action_type":        "Financial Claim",
            "timestamp":          base_time + timedelta(minutes=random.randint(0, 180)),
            "duration_minutes":   round(random.uniform(5, 25), 2),
            "security_level":     random.randint(2, 4),
            "transaction_amount": amount,
            "is_anomaly":         1,
            "anomaly_reason":     f"Transaction amount ${amount:,.2f} is far above normal range",
        })

    anomaly_df = pd.DataFrame(anomalies)
    combined   = pd.concat([df, anomaly_df], ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    return combined.sort_values("timestamp").reset_index(drop=True)


# ─────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the ML feature matrix from raw logs.

    Features created
    ────────────────
    • freq_score        – rolling 1-hour event count per employee
                          (captures burst/high-frequency behaviour)
    • duration_minutes  – raw task duration
                          (captures impossibly fast / suspiciously long tasks)
    • security_level    – numeric encoding of document sensitivity (1-4)
                          (captures unauthorised-access risk)
    • transaction_amount– financial claim value (0 for non-financial events)
                          (captures fraud-sized outliers)
    • hour_of_day       – integer hour 0-23
                          (captures after-hours activity)
    • action_encoded    – ordinal encoding of action type
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Frequency score: events per employee in last 60 minutes
    df["freq_score"] = 0.0
    for emp, grp in df.groupby("employee_id"):
        times = grp["timestamp"].values
        for idx, t in zip(grp.index, times):
            window_start = t - np.timedelta64(60, "m")
            count = np.sum((times >= window_start) & (times <= t))
            df.at[idx, "freq_score"] = float(count)

    df["transaction_amount"] = df["transaction_amount"].fillna(0.0)
    df["hour_of_day"]        = df["timestamp"].dt.hour
    df["action_encoded"]     = df["action_type"].map(
        {"Access Doc": 0, "Submit Task": 1, "Financial Claim": 2}
    )

    return df


# ─────────────────────────────────────────────
#  DATABASE INITIALISATION & STORAGE
# ─────────────────────────────────────────────
def init_db(db_path: str = DB_PATH):
    """Creates SQLite tables for activity logs and audit tasks."""
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS activity_logs (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id         TEXT,
            department          TEXT,
            action_type         TEXT,
            timestamp           TEXT,
            duration_minutes    REAL,
            security_level      INTEGER,
            transaction_amount  REAL,
            freq_score          REAL,
            hour_of_day         INTEGER,
            action_encoded      INTEGER,
            anomaly_score       REAL,
            is_anomaly_pred     INTEGER DEFAULT 0,
            is_anomaly_actual   INTEGER DEFAULT 0,
            anomaly_reason      TEXT,
            ingested_at         TEXT DEFAULT (datetime('now'))
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_tasks (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            log_id          INTEGER,
            employee_id     TEXT,
            department      TEXT,
            action_type     TEXT,
            timestamp       TEXT,
            anomaly_score   REAL,
            reason          TEXT,
            status          TEXT DEFAULT 'OPEN',
            created_at      TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(log_id) REFERENCES activity_logs(id)
        )
    """)

    conn.commit()
    conn.close()


def store_logs(df: pd.DataFrame, db_path: str = DB_PATH):
    """Persists feature-engineered logs into the SQLite activity_logs table."""
    cols = [
        "employee_id", "department", "action_type", "timestamp",
        "duration_minutes", "security_level", "transaction_amount",
        "freq_score", "hour_of_day", "action_encoded",
        "is_anomaly",
    ]
    subset = df[cols].copy()
    subset.rename(columns={"is_anomaly": "is_anomaly_actual"}, inplace=True)
    subset["timestamp"] = subset["timestamp"].astype(str)
    subset["anomaly_score"]    = None
    subset["is_anomaly_pred"]  = 0
    subset["anomaly_reason"]   = df.get("anomaly_reason", None)

    conn = sqlite3.connect(db_path)
    subset.to_sql("activity_logs", conn, if_exists="append", index=False)
    conn.close()
    print(f"[data_generator] Stored {len(subset)} records → {db_path}")


# ─────────────────────────────────────────────
#  REAL-TIME SIMULATION HELPER
# ─────────────────────────────────────────────
def generate_realtime_event() -> dict:
    """
    Produces a single synthetic event for real-time streaming simulation.
    5 % chance of injecting an anomalous event.
    """
    rng = np.random.default_rng()
    action   = random.choice(ACTION_TYPES)
    dept     = random.choice(DEPARTMENTS)
    is_anom  = rng.random() < 0.05

    if is_anom:
        anomaly_kind = random.choice(["sensitive", "duration", "financial"])
        if anomaly_kind == "sensitive":
            return {
                "employee_id":        random.choice(EMPLOYEE_POOL),
                "department":         "Marketing",
                "action_type":        "Access Doc",
                "timestamp":          datetime.now(),
                "duration_minutes":   round(rng.uniform(0.05, 0.3), 3),
                "security_level":     4,
                "transaction_amount": 0.0,
                "is_anomaly":         1,
                "anomaly_reason":     "Restricted doc access by low-clearance department",
            }
        elif anomaly_kind == "duration":
            dur = round(rng.choice([rng.uniform(0.01, 0.05),
                                    rng.uniform(900, 1200)]), 3)
            return {
                "employee_id":        random.choice(EMPLOYEE_POOL),
                "department":         dept,
                "action_type":        action,
                "timestamp":          datetime.now(),
                "duration_minutes":   dur,
                "security_level":     random.randint(1, 3),
                "transaction_amount": 0.0,
                "is_anomaly":         1,
                "anomaly_reason":     f"Duration outlier: {dur} min",
            }
        else:
            amt = round(rng.uniform(8000, 20000), 2)
            return {
                "employee_id":        random.choice(EMPLOYEE_POOL),
                "department":         "Finance",
                "action_type":        "Financial Claim",
                "timestamp":          datetime.now(),
                "duration_minutes":   round(rng.uniform(5, 25), 2),
                "security_level":     random.randint(2, 4),
                "transaction_amount": amt,
                "is_anomaly":         1,
                "anomaly_reason":     f"Anomalous transaction: ${amt:,.2f}",
            }
    else:
        profile  = DURATION_PROFILE[action]
        duration = max(1, rng.normal(profile["mean"], profile["std"]))
        amount   = float(rng.normal(500, 150)) if action == "Financial Claim" else 0.0
        return {
            "employee_id":        random.choice(EMPLOYEE_POOL),
            "department":         dept,
            "action_type":        action,
            "timestamp":          datetime.now(),
            "duration_minutes":   round(duration, 2),
            "security_level":     random.randint(1, 3),
            "transaction_amount": round(max(0, amount), 2),
            "is_anomaly":         0,
            "anomaly_reason":     None,
        }


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🔧 Initialising Sentinel-Detect database …")
    init_db()

    print("📊 Generating synthetic activity logs …")
    normal_df  = generate_normal_log(n=500)
    full_df    = inject_anomalies(normal_df, n_anomalies=40)
    featured   = engineer_features(full_df)

    store_logs(featured)
    print(f"✅ Dataset ready — {len(featured)} total records "
          f"({featured['is_anomaly'].sum()} ground-truth anomalies)")
    print(featured[["employee_id", "action_type", "duration_minutes",
                     "security_level", "freq_score", "is_anomaly"]].head(10))
