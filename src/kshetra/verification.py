"""Local human-verification log. Store district-level notes only; never record personal data."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "field_verifications.sqlite"
STATUSES = {"Pending", "Verified concern", "No concern found", "Needs follow-up"}


def _connect(path: Path = DEFAULT_DB) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS verifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            state TEXT NOT NULL,
            district TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            status TEXT NOT NULL,
            domain TEXT NOT NULL,
            note TEXT NOT NULL,
            source_or_method TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    con.commit()
    return con


def add_verification(state: str, district: str, status: str, domain: str, note: str,
                     source_or_method: str, path: Path = DEFAULT_DB) -> None:
    """Append a district-level review record after basic validation."""
    if status not in STATUSES:
        raise ValueError(f"status must be one of {sorted(STATUSES)}")
    for label, value in [("state", state), ("district", district), ("domain", domain),
                         ("note", note), ("source_or_method", source_or_method)]:
        if not str(value).strip():
            raise ValueError(f"{label} must not be blank")
    if len(note) > 2000:
        raise ValueError("note must be 2000 characters or fewer")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect(path) as con:
        con.execute(
            "INSERT INTO verifications (state,district,observed_at,status,domain,note,source_or_method,created_at) VALUES (?,?,?,?,?,?,?,?)",
            (state.strip(), district.strip(), now, status, domain.strip(), note.strip(), source_or_method.strip(), now),
        )


def list_verifications(path: Path = DEFAULT_DB) -> pd.DataFrame:
    """Return the review log, newest first; empty frame if there are no records."""
    with _connect(path) as con:
        return pd.read_sql_query("SELECT * FROM verifications ORDER BY created_at DESC, id DESC", con)
