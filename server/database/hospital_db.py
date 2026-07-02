"""
hospital_db.py — SQLite database builder and query helper.

All paths are config-driven. Connection pooling via a module-level singleton.

Tables:
  hospitals  (hospital_name, code, address, status, phone_number, city)
  doctors    (doctor_name, register_number, code, address, status, phone_number)

FTS5 virtual tables:
  hospitals_fts (hospital_name, city, address, code)
  doctors_fts   (doctor_name, address, code)

JOIN key: hospitals.code = doctors.code
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
from typing import Optional

import pandas as pd

from utils.config import HOSPITAL_DB, HOSPITALS_CSV, DOCTORS_CSV
from utils.logger import get_logger

logger = get_logger(__name__)

# Thread-safe connection pool (one write connection, per-thread read)
_write_lock   = threading.Lock()
_thread_local = threading.local()

# Column normalisation helpers

def _clean_col(c: str) -> str:
    c = c.strip().lower()
    c = re.sub(r"[^a-z0-9_]", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [_clean_col(c) for c in df.columns]
    return df

def _drop_junk(df: pd.DataFrame) -> pd.DataFrame:
    df = df[[c for c in df.columns if not c.startswith("unnamed")]]
    df = df.dropna(axis=1, how="all")
    df.columns = [c.rstrip("_") for c in df.columns]
    return df

# Main builder

def build_hospital_db(
    hospitals_csv: str = HOSPITALS_CSV,
    doctors_csv:   str = DOCTORS_CSV,
    db_path:       str = HOSPITAL_DB,
) -> None:
    """
    Load hospitals.csv and doctors.csv into SQLite.
    Creates indexes and FTS5 virtual tables after loading.
    Safe to call on every startup — existing data is replaced.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    # Hospitals
    h_df = pd.read_csv(hospitals_csv, encoding="utf-8", on_bad_lines="skip")
    h_df = _normalize_columns(h_df)
    h_df = _drop_junk(h_df)
    h_df = h_df.rename(columns={"hospital_name_": "hospital_name"})

    if "address" in h_df.columns:
        h_df["city"] = h_df["address"].apply(
            lambda a: [p.strip() for p in str(a).split(",")][-1] if pd.notna(a) else ""
        )

    for col in ("hospital_name", "code", "address", "status", "phone_number", "city"):
        if col not in h_df.columns:
            h_df[col] = ""

    for col in h_df.select_dtypes(include="object").columns:
        h_df[col] = h_df[col].astype(str).str.strip()

    # Add normalised lowercase columns for fast LIKE matching
    h_df["_name_lower"]    = h_df["hospital_name"].str.lower()
    h_df["_city_lower"]    = h_df["city"].str.lower()
    h_df["_address_lower"] = h_df["address"].str.lower()

    h_df.to_sql("hospitals", conn, index=False, if_exists="replace")
    logger.info("hospitals: %d rows  columns=%s", len(h_df), list(h_df.columns))

    # Doctors
    d_df = pd.read_csv(doctors_csv, encoding="utf-8", on_bad_lines="skip")
    d_df = _normalize_columns(d_df)
    d_df = _drop_junk(d_df)
    d_df = d_df.rename(columns={"doctors_name": "doctor_name"})

    for col in ("doctor_name", "register_number", "code", "address", "status", "phone_number"):
        if col not in d_df.columns:
            d_df[col] = ""

    for col in d_df.select_dtypes(include="object").columns:
        d_df[col] = d_df[col].astype(str).str.strip()

    # Normalised lowercase columns for fast LIKE matching
    d_df["_name_lower"]    = d_df["doctor_name"].str.lower()
    d_df["_address_lower"] = d_df["address"].str.lower()

    d_df.to_sql("doctors", conn, index=False, if_exists="replace")
    logger.info("doctors: %d rows  columns=%s", len(d_df), list(d_df.columns))

    # Indexes
    idx_sql = [
        "CREATE INDEX IF NOT EXISTS idx_hospital_name    ON hospitals(hospital_name)",
        "CREATE INDEX IF NOT EXISTS idx_hospital_city    ON hospitals(city)",
        "CREATE INDEX IF NOT EXISTS idx_hospital_code    ON hospitals(code)",
        "CREATE INDEX IF NOT EXISTS idx_hospital_status  ON hospitals(status)",
        "CREATE INDEX IF NOT EXISTS idx_hospital_nl      ON hospitals(_name_lower)",
        "CREATE INDEX IF NOT EXISTS idx_hospital_cl      ON hospitals(_city_lower)",
        "CREATE INDEX IF NOT EXISTS idx_doctor_name      ON doctors(doctor_name)",
        "CREATE INDEX IF NOT EXISTS idx_doctor_code      ON doctors(code)",
        "CREATE INDEX IF NOT EXISTS idx_doctor_status    ON doctors(status)",
        "CREATE INDEX IF NOT EXISTS idx_doctor_nl        ON doctors(_name_lower)",
    ]
    for sql in idx_sql:
        conn.execute(sql)
    logger.info("Indexes created")

    # FTS5 virtual tables
    conn.execute("DROP TABLE IF EXISTS hospitals_fts")
    conn.execute("DROP TABLE IF EXISTS doctors_fts")

    conn.execute("""
        CREATE VIRTUAL TABLE hospitals_fts USING fts5(
            hospital_name, city, address, code,
            content='hospitals',
            content_rowid='rowid'
        )
    """)
    conn.execute("""
        INSERT INTO hospitals_fts(rowid, hospital_name, city, address, code)
        SELECT rowid, hospital_name, city, address, code FROM hospitals
    """)

    conn.execute("""
        CREATE VIRTUAL TABLE doctors_fts USING fts5(
            doctor_name, address, code,
            content='doctors',
            content_rowid='rowid'
        )
    """)
    conn.execute("""
        INSERT INTO doctors_fts(rowid, doctor_name, address, code)
        SELECT rowid, doctor_name, address, code FROM doctors
    """)

    conn.commit()
    conn.close()
    logger.info("FTS5 tables built — hospital_db ready")

# Connection helper

def _get_thread_connection() -> sqlite3.Connection:
    """Per-thread read connection (WAL mode allows concurrent reads)."""
    if not hasattr(_thread_local, "conn") or _thread_local.conn is None:
        conn = sqlite3.connect(HOSPITAL_DB, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=4000")
        conn.execute("PRAGMA temp_store=MEMORY")
        _thread_local.conn = conn
    return _thread_local.conn

def run_sql(sql: str, params: tuple = ()) -> list[dict]:
    """Execute a parameterised SELECT and return rows as list of dicts."""
    conn = _get_thread_connection()
    try:
        cur  = conn.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]
    except sqlite3.Error as exc:
        logger.error("SQL error: %s | sql=%s | params=%s", exc, sql[:200], params)
        raise

def is_db_ready() -> bool:
    """Return True if the SQLite database file exists and has hospitals table."""
    if not os.path.isfile(HOSPITAL_DB):
        return False
    try:
        conn = sqlite3.connect(HOSPITAL_DB)
        conn.execute("SELECT 1 FROM hospitals LIMIT 1")
        conn.close()
        return True
    except Exception:
        return False

def get_compact_schema() -> str:
    """Return one-line-per-table schema string for LLM prompts."""
    conn  = _get_thread_connection()
    lines = []
    for table in ("hospitals", "doctors"):
        cur  = conn.execute(f"PRAGMA table_info({table})")
        cols = ", ".join(
            r[1] for r in cur.fetchall()
            if not str(r[1]).startswith("_")   # hide internal _*_lower cols
        )
        lines.append(f"{table}({cols})")
    return "\n".join(lines)

def get_all_hospitals() -> list[dict]:
    """Return all hospital rows for the summary table."""
    return run_sql(
        "SELECT hospital_name, code, address, city, status, phone_number "
        "FROM hospitals ORDER BY hospital_name"
    )
