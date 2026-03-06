"""
Sentinel-Detect | auth.py
=========================
User authentication & role-based access control.
Stores users in SQLite with PBKDF2-HMAC-SHA256 hashed passwords.

Roles
-----
  admin    – sees all departments, all logs, full analytics
  manager  – sees ONLY their own department's data
"""

import sqlite3
import hashlib
import secrets
import os
from datetime import datetime

DB_PATH = "sentinel.db"

DEPARTMENTS = [
    "Engineering", "Finance", "HR",
    "Legal", "Marketing", "Operations",
]

# ── Password helpers ──────────────────────────────────────────────────────────

def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Returns (hashed_hex, salt_hex)."""
    if salt is None:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations=260_000,
    )
    return dk.hex(), salt


def verify_password(password: str, hashed: str, salt: str) -> bool:
    computed, _ = _hash_password(password, salt)
    return secrets.compare_digest(computed, hashed)


# ── DB init ───────────────────────────────────────────────────────────────────

def init_auth_tables(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    NOT NULL UNIQUE,
            password_hash TEXT  NOT NULL,
            salt        TEXT    NOT NULL,
            role        TEXT    NOT NULL CHECK(role IN ('admin','manager')),
            department  TEXT,          -- NULL for admin
            display_name TEXT,
            created_at  TEXT    DEFAULT (datetime('now')),
            last_login  TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS login_audit (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT,
            success     INTEGER,
            ip_hint     TEXT,
            ts          TEXT DEFAULT (datetime('now'))
        )
    """)

    conn.commit()
    conn.close()


# ── Seed default users ────────────────────────────────────────────────────────

def seed_default_users(db_path: str = DB_PATH):
    """
    Creates default accounts if they don't exist yet.

    Default accounts
    ----------------
    admin    / admin123       → admin (all departments)
    eng_mgr  / manager123     → manager, Engineering
    fin_mgr  / manager123     → manager, Finance
    hr_mgr   / manager123     → manager, HR
    legal_mgr/ manager123     → manager, Legal
    mkt_mgr  / manager123     → manager, Marketing
    ops_mgr  / manager123     → manager, Operations
    """
    defaults = [
        ("admin",     "admin123",   "admin",   None,          "Security Admin"),
        ("eng_mgr",   "manager123", "manager", "Engineering", "Engineering Manager"),
        ("fin_mgr",   "manager123", "manager", "Finance",     "Finance Manager"),
        ("hr_mgr",    "manager123", "manager", "HR",          "HR Manager"),
        ("legal_mgr", "manager123", "manager", "Legal",       "Legal Manager"),
        ("mkt_mgr",   "manager123", "manager", "Marketing",   "Marketing Manager"),
        ("ops_mgr",   "manager123", "manager", "Operations",  "Operations Manager"),
    ]

    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    for username, password, role, dept, display in defaults:
        exists = cur.execute(
            "SELECT 1 FROM users WHERE username = ?", (username,)
        ).fetchone()
        if not exists:
            hashed, salt = _hash_password(password)
            cur.execute("""
                INSERT INTO users (username, password_hash, salt, role, department, display_name)
                VALUES (?,?,?,?,?,?)
            """, (username, hashed, salt, role, dept, display))
    conn.commit()
    conn.close()


# ── Login / user ops ──────────────────────────────────────────────────────────

def login(username: str, password: str, db_path: str = DB_PATH) -> dict | None:
    """
    Returns user dict on success, None on failure.
    Logs every attempt to login_audit.
    """
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    row = cur.execute(
        "SELECT id, username, password_hash, salt, role, department, display_name "
        "FROM users WHERE username = ?",
        (username.strip(),),
    ).fetchone()

    success = False
    user    = None

    if row:
        uid, uname, pw_hash, salt, role, dept, display = row
        if verify_password(password, pw_hash, salt):
            success = True
            cur.execute(
                "UPDATE users SET last_login = ? WHERE id = ?",
                (datetime.now().isoformat(timespec="seconds"), uid),
            )
            user = {
                "id":         uid,
                "username":   uname,
                "role":       role,
                "department": dept,
                "display_name": display or uname,
            }

    cur.execute(
        "INSERT INTO login_audit (username, success) VALUES (?,?)",
        (username.strip(), int(success)),
    )
    conn.commit()
    conn.close()
    return user


def get_all_users(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    import pandas as pd
    df = pd.read_sql(
        "SELECT id, username, role, department, display_name, created_at, last_login "
        "FROM users ORDER BY role, department",
        conn,
    )
    conn.close()
    return df


def create_user(
    username: str, password: str, role: str,
    department: str | None, display_name: str,
    db_path: str = DB_PATH,
) -> tuple[bool, str]:
    """Returns (ok, message)."""
    if role == "manager" and not department:
        return False, "Manager must be assigned a department."
    if role not in ("admin", "manager"):
        return False, "Invalid role."

    hashed, salt = _hash_password(password)
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("""
            INSERT INTO users (username, password_hash, salt, role, department, display_name)
            VALUES (?,?,?,?,?,?)
        """, (username.strip(), hashed, salt, role, department, display_name.strip()))
        conn.commit()
        conn.close()
        return True, f"User '{username}' created."
    except sqlite3.IntegrityError:
        return False, f"Username '{username}' already exists."


def change_password(
    username: str, new_password: str, db_path: str = DB_PATH
) -> tuple[bool, str]:
    hashed, salt = _hash_password(new_password)
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute(
        "UPDATE users SET password_hash = ?, salt = ? WHERE username = ?",
        (hashed, salt, username),
    )
    ok = cur.rowcount > 0
    conn.commit()
    conn.close()
    return (True, "Password updated.") if ok else (False, "User not found.")


def delete_user(username: str, db_path: str = DB_PATH) -> tuple[bool, str]:
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("DELETE FROM users WHERE username = ?", (username,))
    ok = cur.rowcount > 0
    conn.commit()
    conn.close()
    return (True, f"User '{username}' deleted.") if ok else (False, "User not found.")