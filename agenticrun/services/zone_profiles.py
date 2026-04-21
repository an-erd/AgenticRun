from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from typing import Any


def _agenticrun_debug() -> bool:
    return os.getenv("AGENTICRUN_DEBUG", "").lower() in {"1", "true", "yes", "on"}

ZONE_PROFILES_TABLE = """
CREATE TABLE IF NOT EXISTS zone_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    effective_from TEXT NOT NULL,
    source_run_id TEXT,
    source_type TEXT,
    hr_zone_boundaries TEXT,
    power_zone_boundaries TEXT,
    threshold_heart_rate REAL,
    functional_threshold_power REAL,
    max_heart_rate REAL,
    hr_calc_type TEXT,
    power_calc_type TEXT,
    zone_source TEXT NOT NULL,
    profile_fingerprint TEXT NOT NULL,
    hr_zone_time_sec TEXT,
    power_zone_time_sec TEXT,
    hr_zone_boundaries_high_bpm TEXT,
    power_zone_boundaries_high_w TEXT
);
CREATE INDEX IF NOT EXISTS idx_zone_profiles_effective_from ON zone_profiles (effective_from);
"""


def _migrate_zone_profiles_columns(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(zone_profiles)").fetchall()}
    if "hr_zone_time_sec" not in cols:
        conn.execute("ALTER TABLE zone_profiles ADD COLUMN hr_zone_time_sec TEXT")
    if "power_zone_time_sec" not in cols:
        conn.execute("ALTER TABLE zone_profiles ADD COLUMN power_zone_time_sec TEXT")
    if "hr_zone_boundaries_high_bpm" not in cols:
        conn.execute("ALTER TABLE zone_profiles ADD COLUMN hr_zone_boundaries_high_bpm TEXT")
    if "power_zone_boundaries_high_w" not in cols:
        conn.execute("ALTER TABLE zone_profiles ADD COLUMN power_zone_boundaries_high_w TEXT")


def ensure_zone_profiles_table(conn: sqlite3.Connection) -> None:
    conn.executescript(ZONE_PROFILES_TABLE)
    _migrate_zone_profiles_columns(conn)


def zone_profile_fingerprint(snapshot: dict[str, Any]) -> str:
    canon = {
        "hr": snapshot.get("hr_zone_boundaries") or [],
        "pw": snapshot.get("power_zone_boundaries") or [],
        "hr_hi": snapshot.get("hr_zone_boundaries_high_bpm") or [],
        "pw_hi": snapshot.get("power_zone_boundaries_high_w") or [],
        "thr": snapshot.get("threshold_heart_rate"),
        "ftp": snapshot.get("functional_threshold_power"),
        "mx": snapshot.get("max_heart_rate"),
        "hrct": snapshot.get("hr_calc_type"),
        "pwct": snapshot.get("power_calc_type"),
    }
    raw = json.dumps(canon, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _snapshot_from_row(row: sqlite3.Row) -> dict[str, Any]:
    def _loads(val: Any) -> list[float]:
        if not val:
            return []
        try:
            data = json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return []
        if not isinstance(data, list):
            return []
        out: list[float] = []
        for x in data:
            try:
                out.append(float(x))
            except (TypeError, ValueError):
                continue
        return out

    snap: dict[str, Any] = {
        "hr_zone_boundaries": _loads(row["hr_zone_boundaries"]),
        "power_zone_boundaries": _loads(row["power_zone_boundaries"]),
        "threshold_heart_rate": row["threshold_heart_rate"],
        "functional_threshold_power": row["functional_threshold_power"],
        "max_heart_rate": row["max_heart_rate"],
        "hr_calc_type": row["hr_calc_type"],
        "power_calc_type": row["power_calc_type"],
    }
    hr_t = row["hr_zone_time_sec"]
    pw_t = row["power_zone_time_sec"]
    if hr_t:
        loaded = _loads(hr_t)
        if loaded:
            snap["hr_zone_time_sec"] = loaded
    if pw_t:
        loaded = _loads(pw_t)
        if loaded:
            snap["power_zone_time_sec"] = loaded
    _rk = row.keys()
    hr_hi = row["hr_zone_boundaries_high_bpm"] if "hr_zone_boundaries_high_bpm" in _rk else None
    pw_hi = row["power_zone_boundaries_high_w"] if "power_zone_boundaries_high_w" in _rk else None
    if hr_hi:
        loaded = _loads(hr_hi)
        if loaded:
            snap["hr_zone_boundaries_high_bpm"] = loaded
    if pw_hi:
        loaded = _loads(pw_hi)
        if loaded:
            snap["power_zone_boundaries_high_w"] = loaded
    return snap


def fetch_latest_zone_profile_at_or_before(conn: sqlite3.Connection, at_iso: str) -> dict[str, Any] | None:
    ensure_zone_profiles_table(conn)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT * FROM zone_profiles
        WHERE effective_from <= ?
        ORDER BY effective_from DESC, id DESC
        LIMIT 1
        """,
        (at_iso,),
    ).fetchone()
    if not row:
        return None
    snap = _snapshot_from_row(row)
    return {
        "snapshot": snap,
        "effective_from": row["effective_from"],
        "source_run_id": row["source_run_id"],
        "source_type": row["source_type"],
        "zone_source": row["zone_source"],
        "profile_fingerprint": row["profile_fingerprint"],
        "row_id": row["id"],
    }


def insert_zone_profile_if_new(
    conn: sqlite3.Connection,
    *,
    effective_from: str,
    source_run_id: str | None,
    source_type: str | None,
    snapshot: dict[str, Any],
    zone_source: str = "fit_garmin",
) -> bool:
    ensure_zone_profiles_table(conn)
    fp = zone_profile_fingerprint(snapshot)
    prev = conn.execute(
        "SELECT profile_fingerprint FROM zone_profiles ORDER BY effective_from DESC, id DESC LIMIT 1"
    ).fetchone()
    if prev and prev[0] == fp:
        if _agenticrun_debug():
            print(
                "zone_profile_store: no (duplicate; fingerprint matches latest row, INSERT skipped)",
                flush=True,
            )
            print(f"  effective_from: {effective_from}", flush=True)
            print(f"  source_run_id: {source_run_id}", flush=True)
            print(f"  zone_source: {zone_source}", flush=True)
        return False

    hr_b = snapshot.get("hr_zone_boundaries") or []
    pw_b = snapshot.get("power_zone_boundaries") or []
    hr_t = snapshot.get("hr_zone_time_sec") or []
    pw_t = snapshot.get("power_zone_time_sec") or []
    hr_hi = snapshot.get("hr_zone_boundaries_high_bpm") or []
    pw_hi = snapshot.get("power_zone_boundaries_high_w") or []
    cur = conn.execute(
        """
        INSERT INTO zone_profiles (
            effective_from, source_run_id, source_type,
            hr_zone_boundaries, power_zone_boundaries,
            threshold_heart_rate, functional_threshold_power, max_heart_rate,
            hr_calc_type, power_calc_type,
            zone_source, profile_fingerprint,
            hr_zone_time_sec, power_zone_time_sec,
            hr_zone_boundaries_high_bpm, power_zone_boundaries_high_w
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            effective_from,
            source_run_id,
            source_type,
            json.dumps(hr_b),
            json.dumps(pw_b),
            snapshot.get("threshold_heart_rate"),
            snapshot.get("functional_threshold_power"),
            snapshot.get("max_heart_rate"),
            snapshot.get("hr_calc_type"),
            snapshot.get("power_calc_type"),
            zone_source,
            fp,
            json.dumps(hr_t if isinstance(hr_t, list) else []),
            json.dumps(pw_t if isinstance(pw_t, list) else []),
            json.dumps(hr_hi if isinstance(hr_hi, list) else []),
            json.dumps(pw_hi if isinstance(pw_hi, list) else []),
        ),
    )
    conn.commit()
    if _agenticrun_debug():
        print("zone_profile_store: yes (new row INSERT)", flush=True)
        print(f"  effective_from: {effective_from}", flush=True)
        print(f"  source_run_id: {source_run_id}", flush=True)
        print(f"  zone_source: {zone_source}", flush=True)
        print(f"  new_row_id: {cur.lastrowid}", flush=True)
    return True
