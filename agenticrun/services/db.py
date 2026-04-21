from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from agenticrun.core.models import (
    Recommendation,
    RunAnalysis,
    RunRecord,
    RunState,
    TrendAssessment,
)
from agenticrun.services.zone_profiles import ensure_zone_profiles_table


def _agenticrun_debug() -> bool:
    return os.getenv("AGENTICRUN_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _fmt_seg_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _unit_verify_line(s: dict[str, Any]) -> str:
    """One compact chunk: fitparse-decoded lap fields vs persisted segment metrics + unit note."""
    return (
        f"idx={s.get('idx')}|map={s.get('mapped_type')}|"
        f"FIT_timer_s={_fmt_seg_metric(s.get('fit_total_timer_time_s'))}|"
        f"FIT_dist_m={_fmt_seg_metric(s.get('fit_total_distance_m'))}|"
        f"FIT_spd_m_s={_fmt_seg_metric(s.get('fit_avg_speed_m_s'))}|"
        f"stored dur_s={_fmt_seg_metric(s.get('persisted_duration_sec'))}|"
        f"dist_m={_fmt_seg_metric(s.get('persisted_distance_m'))}|"
        f"spd_m_s={_fmt_seg_metric(s.get('persisted_avg_speed_m_s'))}|"
        f"pace_sec_per_km={_fmt_seg_metric(s.get('persisted_pace_sec_per_km'))}|"
        f"[fitparse→s,m,m/s; pace=duration÷(dist_km)]"
    )


def _segment_row_sample_str(r: dict[str, Any]) -> str:
    n = r.get("segment_index", r.get("idx"))
    t = r.get("segment_type_mapped", r.get("mapped_type"))
    return (
        f"n={_fmt_seg_metric(n)}|t={t}|d={_fmt_seg_metric(r.get('duration_sec'))}|"
        f"m={_fmt_seg_metric(r.get('distance_m'))}|"
        f"hr={_fmt_seg_metric(r.get('avg_hr'))}/{_fmt_seg_metric(r.get('max_hr'))}|"
        f"pwr={_fmt_seg_metric(r.get('avg_power'))}|v={_fmt_seg_metric(r.get('avg_speed_m_s'))}|"
        f"pace={_fmt_seg_metric(r.get('avg_pace_sec_per_km'))}"
    )


def _sample_segments_for_debug(segments: list[Any], limit: int = 5) -> list[dict[str, Any]]:
    rows = [s for s in segments if isinstance(s, dict)]
    out: list[dict[str, Any]] = []
    got_work = False
    got_recovery = False
    for s in rows:
        t = s.get("segment_type_mapped") or s.get("mapped_type")
        if t == "work" and not got_work:
            out.append(s)
            got_work = True
        elif t == "recovery" and not got_recovery:
            out.append(s)
            got_recovery = True
        if got_work and got_recovery:
            break
    for s in rows:
        if len(out) >= limit:
            break
        if s not in out:
            out.append(s)
    return out[:limit]


RUN_SEGMENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_segments (
    run_id TEXT NOT NULL,
    segment_idx INTEGER NOT NULL,
    fit_source TEXT NOT NULL,
    intensity_raw TEXT,
    lap_trigger_raw TEXT,
    wkt_step_index INTEGER,
    sport_event_raw TEXT,
    segment_status_raw TEXT,
    segment_name TEXT,
    mapped_type TEXT NOT NULL,
    total_timer_sec REAL,
    total_distance_m REAL,
    segment_index INTEGER,
    segment_type_mapped TEXT,
    duration_sec REAL,
    distance_m REAL,
    avg_hr REAL,
    max_hr REAL,
    avg_power REAL,
    avg_speed_m_s REAL,
    avg_pace_sec_per_km REAL,
    PRIMARY KEY (run_id, segment_idx)
);
"""

RUN_SEGMENTS_EXTRA_COLUMNS: list[tuple[str, str]] = [
    ("segment_index", "INTEGER"),
    ("segment_type_mapped", "TEXT"),
    ("duration_sec", "REAL"),
    ("distance_m", "REAL"),
    ("avg_hr", "REAL"),
    ("max_hr", "REAL"),
    ("avg_power", "REAL"),
    ("avg_speed_m_s", "REAL"),
    ("avg_pace_sec_per_km", "REAL"),
]


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    source_file TEXT,
    source_type TEXT,
    run_date TEXT,
    title TEXT,
    distance_km REAL,
    duration_sec REAL,
    avg_pace_sec_km REAL,
    avg_hr REAL,
    max_hr REAL,
    avg_power REAL,
    max_power REAL,
    avg_cadence REAL,
    calories REAL,
    avg_speed REAL,
    max_speed REAL,
    max_cadence REAL,
    moving_time_sec REAL,
    avg_moving_pace_sec_km REAL,
    power_zone_z1_sec REAL,
    power_zone_z2_sec REAL,
    power_zone_z3_sec REAL,
    power_zone_z4_sec REAL,
    power_zone_z5_sec REAL,
    hr_zone_z1_sec REAL,
    hr_zone_z2_sec REAL,
    hr_zone_z3_sec REAL,
    hr_zone_z4_sec REAL,
    hr_zone_z5_sec REAL,
    has_power INTEGER,
    has_hr INTEGER,
    has_cadence INTEGER,
    has_gps INTEGER,
    stopped_time_sec REAL,
    data_quality_score REAL,
    fit_parse_warnings TEXT,
    sport TEXT,
    sub_sport TEXT,
    device_name TEXT,
    lap_count INTEGER,
    power_available INTEGER,
    hr_available INTEGER,
    cadence_available INTEGER,
    record_count INTEGER,
    elevation_gain_m REAL,
    training_load REAL,
    notes TEXT,
    training_type TEXT,
    intensity_label TEXT,
    execution_quality TEXT,
    analysis_confidence REAL,
    history_count INTEGER,
    similar_count INTEGER,
    trend_label TEXT,
    fitness_signal TEXT,
    fatigue_signal TEXT,
    next_session TEXT,
    load_action TEXT,
    warning_flag INTEGER,
    analysis_summary TEXT,
    classification_trace TEXT,
    trend_summary TEXT,
    recommendation_summary TEXT,
    llm_summary TEXT,
    llm_summary_short TEXT,
    llm_context_progress TEXT,
    llm_context_progress_short TEXT,
    llm_what_next_short TEXT,
    recommendation_signals TEXT,
    zone_model_source TEXT,
    zone_model_effective_from TEXT,
    zone_model_source_run_id TEXT
);
"""

ADDITIONAL_COLUMNS: list[tuple[str, str]] = [
    ("calories", "REAL"),
    ("avg_speed", "REAL"),
    ("max_speed", "REAL"),
    ("max_cadence", "REAL"),
    ("moving_time_sec", "REAL"),
    ("avg_moving_pace_sec_km", "REAL"),
    ("power_zone_z1_sec", "REAL"),
    ("power_zone_z2_sec", "REAL"),
    ("power_zone_z3_sec", "REAL"),
    ("power_zone_z4_sec", "REAL"),
    ("power_zone_z5_sec", "REAL"),
    ("hr_zone_z1_sec", "REAL"),
    ("hr_zone_z2_sec", "REAL"),
    ("hr_zone_z3_sec", "REAL"),
    ("hr_zone_z4_sec", "REAL"),
    ("hr_zone_z5_sec", "REAL"),
    ("has_power", "INTEGER"),
    ("has_hr", "INTEGER"),
    ("has_cadence", "INTEGER"),
    ("has_gps", "INTEGER"),
    ("stopped_time_sec", "REAL"),
    ("data_quality_score", "REAL"),
    ("fit_parse_warnings", "TEXT"),
    ("sport", "TEXT"),
    ("sub_sport", "TEXT"),
    ("device_name", "TEXT"),
    ("lap_count", "INTEGER"),
    ("power_available", "INTEGER"),
    ("hr_available", "INTEGER"),
    ("cadence_available", "INTEGER"),
    ("record_count", "INTEGER"),
    ("classification_trace", "TEXT"),
    ("llm_summary_short", "TEXT"),
    ("llm_context_progress", "TEXT"),
    ("llm_context_progress_short", "TEXT"),
    ("llm_what_next_short", "TEXT"),
    ("recommendation_signals", "TEXT"),
    ("zone_model_source", "TEXT"),
    ("zone_model_effective_from", "TEXT"),
    ("zone_model_source_run_id", "TEXT"),
    ("fit_activity_key", "TEXT"),
]


def _migrate_run_segments_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(run_segments)").fetchall()}
    for col_name, col_type in RUN_SEGMENTS_EXTRA_COLUMNS:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE run_segments ADD COLUMN {col_name} {col_type}")


def ensure_run_segments_table(conn: sqlite3.Connection) -> None:
    conn.executescript(RUN_SEGMENTS_SCHEMA)
    _migrate_run_segments_columns(conn)


def replace_run_segments(conn: sqlite3.Connection, run_id: str, segments: list[dict[str, Any]]) -> int:
    """Replace all ``run_segments`` for ``run_id`` (delete-then-insert). Safe to call repeatedly."""
    conn.execute("DELETE FROM run_segments WHERE run_id = ?", (run_id,))
    payload: list[tuple[Any, ...]] = []
    for r in segments:
        if not isinstance(r, dict):
            continue
        if "idx" not in r or "fit_source" not in r or "mapped_type" not in r:
            continue
        seg_i = r.get("segment_index", r["idx"])
        seg_t = r.get("segment_type_mapped", r["mapped_type"])
        payload.append(
            (
                run_id,
                r["idx"],
                r["fit_source"],
                r.get("intensity_raw"),
                r.get("lap_trigger_raw"),
                r.get("wkt_step_index"),
                r.get("sport_event_raw"),
                r.get("segment_status_raw"),
                r.get("segment_name"),
                r["mapped_type"],
                r.get("total_timer_sec"),
                r.get("total_distance_m"),
                seg_i,
                seg_t,
                r.get("duration_sec"),
                r.get("distance_m"),
                r.get("avg_hr"),
                r.get("max_hr"),
                r.get("avg_power"),
                r.get("avg_speed_m_s"),
                r.get("avg_pace_sec_per_km"),
            )
        )
    if not payload:
        return 0
    conn.executemany(
        """
        INSERT INTO run_segments (
            run_id, segment_idx, fit_source, intensity_raw, lap_trigger_raw, wkt_step_index,
            sport_event_raw, segment_status_raw, segment_name, mapped_type,
            total_timer_sec, total_distance_m,
            segment_index, segment_type_mapped, duration_sec, distance_m,
            avg_hr, max_hr, avg_power, avg_speed_m_s, avg_pace_sec_per_km
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )
    return len(payload)


def lookup_run_id_by_fit_activity_key(
    conn: sqlite3.Connection, fit_activity_key: str
) -> str | None:
    """Return ``run_id`` of an existing row with this FIT duplicate key, if any."""
    if not fit_activity_key:
        return None
    row = conn.execute(
        "SELECT run_id FROM runs WHERE fit_activity_key = ? LIMIT 1",
        (fit_activity_key,),
    ).fetchone()
    return str(row[0]) if row else None


def _row_opt_float(row: sqlite3.Row, key: str) -> float | None:
    if key not in row.keys():
        return None
    v = row[key]
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _row_opt_int(row: sqlite3.Row, key: str) -> int | None:
    if key not in row.keys():
        return None
    v = row[key]
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def load_cached_run_state_from_db(
    conn: sqlite3.Connection,
    run_id: str,
    incoming_source_path: str,
) -> RunState | None:
    """Load persisted run row into a :class:`RunState` for duplicate-upload surfacing.

    Does not re-parse FIT or write to ``runs``. ``source_path`` is the incoming upload path;
    ``run_record.source_file`` is set to that file's basename so batch output reflects the
    file the user dropped.
    """
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM runs WHERE run_id = ? LIMIT 1",
        (str(run_id).strip(),),
    ).fetchone()
    if row is None:
        return None

    incoming = Path(incoming_source_path)
    fit_key = None
    if "fit_activity_key" in row.keys():
        fk = row["fit_activity_key"]
        fit_key = str(fk) if fk is not None and str(fk).strip() else None

    rr = RunRecord(
        run_id=str(row["run_id"]),
        source_file=incoming.name,
        source_type=str(row["source_type"] or "garmin_fit"),
        run_date=str(row["run_date"] or ""),
        title=str(row["title"] or ""),
        distance_km=_row_opt_float(row, "distance_km"),
        duration_sec=_row_opt_float(row, "duration_sec"),
        avg_pace_sec_km=_row_opt_float(row, "avg_pace_sec_km"),
        avg_hr=_row_opt_float(row, "avg_hr"),
        max_hr=_row_opt_float(row, "max_hr"),
        avg_power=_row_opt_float(row, "avg_power"),
        max_power=_row_opt_float(row, "max_power"),
        avg_cadence=_row_opt_float(row, "avg_cadence"),
        elevation_gain_m=_row_opt_float(row, "elevation_gain_m"),
        training_load=_row_opt_float(row, "training_load"),
        notes=str(row["notes"] or "") if row["notes"] is not None else "",
        fit_activity_key=fit_key,
        raw_summary={},
    )

    ac = _row_opt_float(row, "analysis_confidence")
    analysis = RunAnalysis(
        training_type=str(row["training_type"] or "unknown"),
        intensity_label=str(row["intensity_label"] or "unknown"),
        execution_quality=str(row["execution_quality"] or "unknown"),
        confidence=float(ac) if ac is not None else 0.5,
        session_flags=[],
        summary=str(row["analysis_summary"] or "") if row["analysis_summary"] is not None else "",
        classification_trace=str(row["classification_trace"] or "")
        if row["classification_trace"] is not None
        else "",
    )

    hc = _row_opt_int(row, "history_count")
    sc = _row_opt_int(row, "similar_count")
    wf = row["warning_flag"] if "warning_flag" in row.keys() else None
    warning_flag = bool(wf) if wf is not None else False

    trend = TrendAssessment(
        history_count=int(hc) if hc is not None else 0,
        similar_count=int(sc) if sc is not None else 0,
        trend_label=str(row["trend_label"] or "insufficient_history"),
        fitness_signal=str(row["fitness_signal"] or "unknown"),
        fatigue_signal=str(row["fatigue_signal"] or "unknown"),
        trend_summary=str(row["trend_summary"] or "") if row["trend_summary"] is not None else "",
    )

    rec_sig_raw = row["recommendation_signals"] if "recommendation_signals" in row.keys() else None
    rec_signals: dict[str, Any] = {}
    if rec_sig_raw:
        try:
            parsed = json.loads(str(rec_sig_raw))
            if isinstance(parsed, dict):
                rec_signals = parsed
        except json.JSONDecodeError:
            rec_signals = {}

    recommendation = Recommendation(
        next_session=str(row["next_session"] or "") if row["next_session"] is not None else "",
        load_action=str(row["load_action"] or "hold") if row["load_action"] is not None else "hold",
        warning_flag=warning_flag,
        recommendation_summary=str(row["recommendation_summary"] or "")
        if row["recommendation_summary"] is not None
        else "",
        recommendation_signals=rec_signals,
    )

    llm_text = row["llm_summary"] if "llm_summary" in row.keys() else None
    llm_summary = str(llm_text) if llm_text is not None else ""
    llm_short = ""
    if "llm_summary_short" in row.keys():
        _s = row["llm_summary_short"]
        llm_short = str(_s) if _s is not None else ""
    llm_context = ""
    if "llm_context_progress" in row.keys():
        _ctx = row["llm_context_progress"]
        llm_context = str(_ctx) if _ctx is not None else ""
    llm_context_short = ""
    if "llm_context_progress_short" in row.keys():
        _ctxs = row["llm_context_progress_short"]
        llm_context_short = str(_ctxs) if _ctxs is not None else ""
    llm_what_next = ""
    if "llm_what_next_short" in row.keys():
        _wn = row["llm_what_next_short"]
        llm_what_next = str(_wn) if _wn is not None else ""

    return RunState(
        source_path=str(incoming),
        status="initialized",
        run_record=rr,
        analysis=analysis,
        trend=trend,
        recommendation=recommendation,
        warnings=[],
        llm_summary=llm_summary,
        llm_summary_short=llm_short,
        llm_context_progress=llm_context,
        llm_context_progress_short=llm_context_short,
        llm_what_next_short=llm_what_next,
        fit_stream_bundle=None,
    )


def connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA)
    _migrate_runs_table(conn)
    ensure_zone_profiles_table(conn)
    ensure_run_segments_table(conn)
    conn.commit()
    return conn


def _migrate_runs_table(conn: sqlite3.Connection) -> None:
    existing_cols = {
        row[1]
        for row in conn.execute("PRAGMA table_info(runs)").fetchall()
    }
    for col_name, col_type in ADDITIONAL_COLUMNS:
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {col_name} {col_type}")


def upsert_state(conn: sqlite3.Connection, state: RunState) -> None:
    """Persist ``runs`` and, for Garmin FIT, replace ``run_segments`` for that ``run_id``.

    Ingest may call this twice (pre-LLM and post-LLM); ``INSERT OR REPLACE`` and
    :func:`replace_run_segments` keep a single logical run row and non-duplicated segments.
    """
    row = state.as_flat_dict()
    fit_metrics: dict[str, Any] = {}
    if state.run_record and isinstance(state.run_record.raw_summary, dict):
        candidate = state.run_record.raw_summary.get("fit_session_metrics")
        if isinstance(candidate, dict):
            fit_metrics = candidate

    row.update({
        "calories": fit_metrics.get("calories"),
        "avg_speed": fit_metrics.get("avg_speed"),
        "max_speed": fit_metrics.get("max_speed"),
        "max_cadence": fit_metrics.get("max_cadence"),
        "moving_time_sec": fit_metrics.get("moving_time_sec"),
        "avg_moving_pace_sec_km": fit_metrics.get("avg_moving_pace_sec_km"),
        "power_zone_z1_sec": fit_metrics.get("power_zone_z1_sec"),
        "power_zone_z2_sec": fit_metrics.get("power_zone_z2_sec"),
        "power_zone_z3_sec": fit_metrics.get("power_zone_z3_sec"),
        "power_zone_z4_sec": fit_metrics.get("power_zone_z4_sec"),
        "power_zone_z5_sec": fit_metrics.get("power_zone_z5_sec"),
        "hr_zone_z1_sec": fit_metrics.get("hr_zone_z1_sec"),
        "hr_zone_z2_sec": fit_metrics.get("hr_zone_z2_sec"),
        "hr_zone_z3_sec": fit_metrics.get("hr_zone_z3_sec"),
        "hr_zone_z4_sec": fit_metrics.get("hr_zone_z4_sec"),
        "hr_zone_z5_sec": fit_metrics.get("hr_zone_z5_sec"),
        "has_power": _to_sqlite_bool_int(fit_metrics.get("has_power")),
        "has_hr": _to_sqlite_bool_int(fit_metrics.get("has_hr")),
        "has_cadence": _to_sqlite_bool_int(fit_metrics.get("has_cadence")),
        "has_gps": _to_sqlite_bool_int(fit_metrics.get("has_gps")),
        "stopped_time_sec": fit_metrics.get("stopped_time_sec"),
        "data_quality_score": fit_metrics.get("data_quality_score"),
        "fit_parse_warnings": fit_metrics.get("fit_parse_warnings"),
        "sport": fit_metrics.get("sport"),
        "sub_sport": fit_metrics.get("sub_sport"),
        "device_name": fit_metrics.get("device_name"),
        "lap_count": fit_metrics.get("lap_count"),
        "power_available": _to_sqlite_bool_int(fit_metrics.get("power_available")),
        "hr_available": _to_sqlite_bool_int(fit_metrics.get("hr_available")),
        "cadence_available": _to_sqlite_bool_int(fit_metrics.get("cadence_available")),
        "record_count": fit_metrics.get("record_count"),
        "zone_model_source": fit_metrics.get("zone_model_source"),
        "zone_model_effective_from": fit_metrics.get("zone_model_effective_from"),
        "zone_model_source_run_id": fit_metrics.get("zone_model_source_run_id"),
    })
    conn.execute(
        """
        INSERT OR REPLACE INTO runs (
            run_id, source_file, source_type, run_date, title, distance_km, duration_sec,
            avg_pace_sec_km, avg_hr, max_hr, avg_power, max_power, avg_cadence,
            calories, avg_speed, max_speed, max_cadence, moving_time_sec, avg_moving_pace_sec_km,
            power_zone_z1_sec, power_zone_z2_sec, power_zone_z3_sec, power_zone_z4_sec, power_zone_z5_sec,
            hr_zone_z1_sec, hr_zone_z2_sec, hr_zone_z3_sec, hr_zone_z4_sec, hr_zone_z5_sec,
            has_power, has_hr, has_cadence, has_gps, stopped_time_sec, data_quality_score, fit_parse_warnings,
            sport, sub_sport, device_name,
            lap_count, power_available, hr_available, cadence_available, record_count,
            elevation_gain_m, training_load, notes, training_type, intensity_label,
            execution_quality, analysis_confidence, history_count, similar_count,
            trend_label, fitness_signal, fatigue_signal, next_session, load_action,
            warning_flag, analysis_summary, classification_trace, trend_summary, recommendation_summary,
            llm_summary, llm_summary_short, llm_what_next_short, llm_context_progress, llm_context_progress_short,
            recommendation_signals,
            zone_model_source, zone_model_effective_from, zone_model_source_run_id,
            fit_activity_key
        ) VALUES (
            :run_id, :source_file, :source_type, :run_date, :title, :distance_km, :duration_sec,
            :avg_pace_sec_km, :avg_hr, :max_hr, :avg_power, :max_power, :avg_cadence,
            :calories, :avg_speed, :max_speed, :max_cadence, :moving_time_sec, :avg_moving_pace_sec_km,
            :power_zone_z1_sec, :power_zone_z2_sec, :power_zone_z3_sec, :power_zone_z4_sec, :power_zone_z5_sec,
            :hr_zone_z1_sec, :hr_zone_z2_sec, :hr_zone_z3_sec, :hr_zone_z4_sec, :hr_zone_z5_sec,
            :has_power, :has_hr, :has_cadence, :has_gps, :stopped_time_sec, :data_quality_score, :fit_parse_warnings,
            :sport, :sub_sport, :device_name,
            :lap_count, :power_available, :hr_available, :cadence_available, :record_count,
            :elevation_gain_m, :training_load, :notes, :training_type, :intensity_label,
            :execution_quality, :analysis_confidence, :history_count, :similar_count,
            :trend_label, :fitness_signal, :fatigue_signal, :next_session, :load_action,
            :warning_flag, :analysis_summary, :classification_trace, :trend_summary, :recommendation_summary,
            :llm_summary, :llm_summary_short, :llm_what_next_short, :llm_context_progress, :llm_context_progress_short,
            :recommendation_signals,
            :zone_model_source, :zone_model_effective_from, :zone_model_source_run_id,
            :fit_activity_key
        )
        """,
        row,
    )
    rid = row.get("run_id")
    if rid and state.run_record and state.run_record.source_type == "garmin_fit":
        raw = state.run_record.raw_summary if isinstance(state.run_record.raw_summary, dict) else {}
        segments = raw.get("fit_run_segments") if isinstance(raw.get("fit_run_segments"), list) else []
        seg_meta = raw.get("fit_segment_extract_meta") if isinstance(raw.get("fit_segment_extract_meta"), dict) else {}
        inserted = replace_run_segments(conn, str(rid), segments)
        if _agenticrun_debug():
            summary: dict[str, int] = {}
            for s in segments:
                if isinstance(s, dict):
                    mt = s.get("mapped_type") or "other"
                    summary[mt] = summary.get(mt, 0) + 1
            map_s = (
                "{}"
                if not summary
                else "{" + ",".join(f"{k}:{summary[k]}" for k in sorted(summary)) + "}"
            )
            src = seg_meta.get("extract_source") or "FIT lap + segment_lap messages"
            lap_n = seg_meta.get("lap_message_count", 0)
            seg_n = seg_meta.get("segment_lap_message_count", 0)
            raw_c = seg_meta.get("raw_compact") or "-"
            print(
                f"segment_extract_start run_id={rid} source={src!r} "
                f"lap_msgs={lap_n} segment_lap_msgs={seg_n} rows={len(segments)} "
                f"raw={raw_c} inserted={inserted} map={map_s}",
                flush=True,
            )
            lap_map = seg_meta.get("lap_map_compact")
            if lap_map:
                print(f"segment_lap_map {lap_map}", flush=True)
            samples = _sample_segments_for_debug(segments, limit=5)
            if samples:
                parts = [_segment_row_sample_str(s) for s in samples]
                print(f"segment_row_samples {' ;; '.join(parts)}", flush=True)
            uv = seg_meta.get("unit_verify_samples")
            if isinstance(uv, list) and uv:
                chunks = [_unit_verify_line(s) for s in uv if isinstance(s, dict)]
                if chunks:
                    print(f"segment_units_verify {' ;; '.join(chunks)}", flush=True)
    conn.commit()


def persistence_audit_for_run(conn: sqlite3.Connection, run_id: str) -> dict[str, Any]:
    """SQLite counts for ingest validation: one ``runs`` row per ``run_id``, segment rows, LLM text present."""
    rid = str(run_id).strip()
    if not rid:
        return {
            "run_rowcount": 0,
            "segments_rowcount": 0,
            "llm_summary_stored": False,
            "llm_context_progress_stored": False,
        }
    n_run = int(
        conn.execute("SELECT COUNT(*) FROM runs WHERE run_id = ?", (rid,)).fetchone()[0]
    )
    n_seg = int(
        conn.execute("SELECT COUNT(*) FROM run_segments WHERE run_id = ?", (rid,)).fetchone()[0]
    )
    row = conn.execute(
        "SELECT llm_summary, llm_summary_short, llm_context_progress, llm_context_progress_short "
        "FROM runs WHERE run_id = ?",
        (rid,),
    ).fetchone()
    raw_full = row[0] if row else None
    raw_short = row[1] if row and len(row) > 1 else None
    raw_ctx_full = row[2] if row and len(row) > 2 else None
    raw_ctx_short = row[3] if row and len(row) > 3 else None
    llm_ok = bool(str(raw_full or "").strip() or str(raw_short or "").strip())
    llm_ctx_ok = bool(
        str(raw_ctx_full or "").strip() or str(raw_ctx_short or "").strip()
    )
    return {
        "run_rowcount": n_run,
        "segments_rowcount": n_seg,
        "llm_summary_stored": llm_ok,
        "llm_context_progress_stored": llm_ctx_ok,
    }


def _to_sqlite_bool_int(value: Any) -> int | None:
    if value is None:
        return None
    return 1 if bool(value) else 0


def load_history(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM runs ORDER BY run_date").fetchall()
    return [dict(r) for r in rows]


def fetch_work_recovery_segments_history(
    conn: sqlite3.Connection,
    *,
    limit: int | None = None,
    newest_first: bool = False,
) -> list[dict[str, Any]]:
    """Return compact work/recovery segment rows joined to runs, ordered by run_date.

    Uses segment_type_mapped when set, otherwise mapped_type. Distance prefers distance_m,
    then total_distance_m. Suitable for timeline-style reads from SQLite.
    """
    ensure_run_segments_table(conn)
    order_dir = "DESC" if newest_first else "ASC"
    lim_sql = f" LIMIT {int(limit)}" if limit is not None else ""
    sql = f"""
        SELECT
            r.run_date AS run_date,
            rs.run_id AS run_id,
            COALESCE(rs.segment_index, rs.segment_idx) AS segment_index,
            COALESCE(rs.segment_type_mapped, rs.mapped_type) AS segment_type_mapped,
            rs.duration_sec AS duration_sec,
            COALESCE(rs.distance_m, rs.total_distance_m) AS distance_m,
            rs.avg_hr AS avg_hr,
            rs.max_hr AS max_hr,
            rs.avg_power AS avg_power,
            rs.avg_speed_m_s AS avg_speed_m_s,
            rs.avg_pace_sec_per_km AS avg_pace_sec_per_km
        FROM run_segments rs
        INNER JOIN runs r ON r.run_id = rs.run_id
        WHERE COALESCE(rs.segment_type_mapped, rs.mapped_type) IN ('work', 'recovery')
        ORDER BY r.run_date {order_dir}, rs.run_id ASC, COALESCE(rs.segment_index, rs.segment_idx) ASC
        {lim_sql}
    """
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


def _normalize_work_recovery_agg_row(out: dict[str, Any]) -> None:
    for key in ("work_count", "recovery_count"):
        v = out.get(key)
        out[key] = 0 if v is None else int(v)


def _sql_work_recovery_session_agg_select(typ: str) -> str:
    """Shared SELECT + JOIN + GROUP BY for per-run work/recovery aggregates (same columns as single-run)."""
    return f"""
        SELECT
            r.run_date AS run_date,
            r.run_id AS run_id,
            SUM(CASE WHEN {typ} = 'work' THEN 1 ELSE 0 END) AS work_count,
            AVG(CASE WHEN {typ} = 'work' THEN rs.duration_sec END) AS work_mean_duration_sec,
            AVG(CASE WHEN {typ} = 'work' THEN rs.avg_power END) AS work_mean_power_w,
            AVG(CASE WHEN {typ} = 'work' THEN rs.avg_hr END) AS work_mean_hr_avg,
            AVG(CASE WHEN {typ} = 'work' THEN rs.avg_pace_sec_per_km END)
                AS work_mean_pace_sec_per_km,
            SUM(CASE WHEN {typ} = 'recovery' THEN 1 ELSE 0 END) AS recovery_count,
            AVG(CASE WHEN {typ} = 'recovery' THEN rs.duration_sec END)
                AS recovery_mean_duration_sec,
            AVG(CASE WHEN {typ} = 'recovery' THEN rs.avg_power END) AS recovery_mean_power_w,
            AVG(CASE WHEN {typ} = 'recovery' THEN rs.avg_hr END) AS recovery_mean_hr_avg,
            AVG(CASE WHEN {typ} = 'recovery' THEN rs.avg_pace_sec_per_km END)
                AS recovery_mean_pace_sec_per_km
        FROM runs r
        LEFT JOIN run_segments rs
            ON rs.run_id = r.run_id AND {typ} IN ('work', 'recovery')
        GROUP BY r.run_id, r.run_date
    """


def fetch_work_recovery_session_summaries(
    conn: sqlite3.Connection,
    *,
    newest_first: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Historical list of per-run work/recovery aggregates (all runs in ``runs``), ordered by ``run_date``."""
    ensure_run_segments_table(conn)
    typ = "COALESCE(rs.segment_type_mapped, rs.mapped_type)"
    order_dir = "DESC" if newest_first else "ASC"
    lim_sql = f" LIMIT {int(limit)}" if limit is not None else ""
    sql = (
        f"SELECT * FROM ({_sql_work_recovery_session_agg_select(typ)}) AS agg "
        f"ORDER BY agg.run_date {order_dir}, agg.run_id ASC{lim_sql}"
    )
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql).fetchall()
    out = [dict(r) for r in rows]
    for row in out:
        _normalize_work_recovery_agg_row(row)
    return out


def aggregate_work_recovery_segments_for_run(
    conn: sqlite3.Connection, run_id: str
) -> dict[str, Any] | None:
    """Aggregate persisted work/recovery segment rows for one run (means use SQL AVG, NULL if none).

    Returns None if ``run_id`` is missing from ``runs``. Counts are integers; means may be None.
    """
    ensure_run_segments_table(conn)
    typ = "COALESCE(rs.segment_type_mapped, rs.mapped_type)"
    sql = _sql_work_recovery_session_agg_select(typ).replace(
        "GROUP BY r.run_id, r.run_date",
        "WHERE r.run_id = ?\n        GROUP BY r.run_id, r.run_date",
        1,
    )
    conn.row_factory = sqlite3.Row
    row = conn.execute(sql, (run_id,)).fetchone()
    if row is None:
        return None
    out = dict(row)
    _normalize_work_recovery_agg_row(out)
    return out


def aggregate_work_only_session_for_run(
    conn: sqlite3.Connection, run_id: str
) -> dict[str, Any] | None:
    """Aggregate only ``work``-mapped segments (excludes recovery, warmup, cooldown, other).

    Pace is total work time divided by total work distance (km) when distance sums > 0;
    otherwise falls back to the simple mean of per-segment ``avg_pace_sec_per_km``.
    ``work_w_per_hr`` is mean power divided by mean HR when both are present and HR > 0.
    """
    ensure_run_segments_table(conn)
    if conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone() is None:
        return None
    typ = "COALESCE(rs.segment_type_mapped, rs.mapped_type)"
    sql = f"""
        SELECT
            r.run_date AS run_date,
            r.run_id AS run_id,
            COUNT(rs.segment_idx) AS work_block_count,
            COALESCE(SUM(rs.duration_sec), 0) AS work_total_time_sec,
            COALESCE(SUM(COALESCE(rs.distance_m, rs.total_distance_m)), 0)
                AS work_total_distance_m,
            AVG(rs.avg_power) AS work_mean_power_w,
            AVG(rs.avg_hr) AS work_mean_hr_avg,
            CASE
                WHEN COALESCE(SUM(COALESCE(rs.distance_m, rs.total_distance_m)), 0) > 0
                     AND COALESCE(SUM(rs.duration_sec), 0) > 0
                THEN SUM(rs.duration_sec)
                    / (SUM(COALESCE(rs.distance_m, rs.total_distance_m)) / 1000.0)
                ELSE AVG(rs.avg_pace_sec_per_km)
            END AS work_mean_pace_sec_per_km
        FROM runs r
        LEFT JOIN run_segments rs
            ON rs.run_id = r.run_id AND {typ} = 'work'
        WHERE r.run_id = ?
        GROUP BY r.run_id, r.run_date
    """
    conn.row_factory = sqlite3.Row
    row = conn.execute(sql, (run_id,)).fetchone()
    if row is None:
        return None
    out = dict(row)
    n = out.get("work_block_count")
    out["work_block_count"] = 0 if n is None else int(n)
    pwr = out.get("work_mean_power_w")
    hr = out.get("work_mean_hr_avg")
    try:
        if pwr is not None and hr is not None and float(hr) > 0:
            out["work_w_per_hr"] = float(pwr) / float(hr)
        else:
            out["work_w_per_hr"] = None
    except (TypeError, ValueError):
        out["work_w_per_hr"] = None
    return out


def analyze_structured_work_reps_for_run(
    conn: sqlite3.Connection, run_id: str
) -> dict[str, Any]:
    """Rep-by-rep work-segment analysis for one run (deterministic, no classification changes).

    Uses persisted ``run_segments`` rows mapped as ``work`` and returns per-rep rows, a compact
    summary, and simple intra-session execution interpretation for structured-workout UI.
    """
    ensure_run_segments_table(conn)
    if conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone() is None:
        return {
            "run_id": run_id,
            "available": False,
            "reason": "run_not_found",
            "reps": [],
            "summary": {},
            "interpretation": {},
        }

    typ = "COALESCE(rs.segment_type_mapped, rs.mapped_type)"
    sql = f"""
        SELECT
            COALESCE(rs.segment_index, rs.segment_idx) AS seg_idx,
            rs.duration_sec AS duration_sec,
            COALESCE(rs.distance_m, rs.total_distance_m) AS distance_m,
            rs.avg_pace_sec_per_km AS avg_pace_sec_per_km,
            rs.avg_power AS avg_power,
            rs.avg_hr AS avg_hr,
            rs.max_hr AS max_hr
        FROM run_segments rs
        WHERE rs.run_id = ? AND {typ} = 'work'
        ORDER BY COALESCE(rs.segment_index, rs.segment_idx) ASC
    """
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, (run_id,)).fetchall()
    if not rows:
        return {
            "run_id": run_id,
            "available": False,
            "reason": "no_work_reps",
            "reps": [],
            "summary": {},
            "interpretation": {},
        }

    def _as_float(v: Any) -> float | None:
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _mean(vals: list[float]) -> float | None:
        return (sum(vals) / len(vals)) if vals else None

    def _rep_label(dist_m: float | None, dur_s: float | None) -> str:
        if dist_m is not None and dist_m >= 25:
            return f"{int(round(dist_m / 10.0) * 10)} m"
        if dur_s is not None and dur_s >= 30:
            return f"{max(1, int(round(dur_s / 60.0)))} min"
        return "work rep"

    reps: list[dict[str, Any]] = []
    for i, r in enumerate(rows, start=1):
        dur = _as_float(r["duration_sec"])
        dist = _as_float(r["distance_m"])
        pace = _as_float(r["avg_pace_sec_per_km"])
        pwr = _as_float(r["avg_power"])
        hr = _as_float(r["avg_hr"])
        mx = _as_float(r["max_hr"])
        reps.append(
            {
                "rep_no": i,
                "segment_index": r["seg_idx"],
                "rep_label": _rep_label(dist, dur),
                "duration_sec": dur,
                "distance_m": dist,
                "pace_sec_per_km": pace,
                "power_w": pwr,
                "hr_avg": hr,
                "hr_max": mx,
                "cadence_spm": None,
                "stride_length_m": None,
                "ground_contact_ms": None,
            }
        )

    pace_vals = [float(x["pace_sec_per_km"]) for x in reps if x.get("pace_sec_per_km") is not None]
    pwr_vals = [float(x["power_w"]) for x in reps if x.get("power_w") is not None]
    hr_vals = [float(x["hr_avg"]) for x in reps if x.get("hr_avg") is not None]
    mx_vals = [float(x["hr_max"]) for x in reps if x.get("hr_max") is not None]
    wphr_vals = [float(x["power_w"]) / float(x["hr_avg"]) for x in reps if x.get("power_w") is not None and x.get("hr_avg") not in (None, 0)]
    dur_vals = [float(x["duration_sec"]) for x in reps if x.get("duration_sec") is not None]

    summary = {
        "rep_count": len(reps),
        "total_work_time_sec": sum(dur_vals) if dur_vals else None,
        "work_mean_pace_sec_per_km": _mean(pace_vals),
        "work_mean_power_w": _mean(pwr_vals),
        "work_mean_hr_avg": _mean(hr_vals),
        "work_max_hr_avg": _mean(mx_vals),
        "work_w_per_hr": _mean(wphr_vals),
    }

    n = len(reps)
    split = max(1, n // 2)
    first = reps[:split]
    last = reps[-split:]
    first_pace = _mean([float(x["pace_sec_per_km"]) for x in first if x.get("pace_sec_per_km") is not None])
    last_pace = _mean([float(x["pace_sec_per_km"]) for x in last if x.get("pace_sec_per_km") is not None])
    first_pwr = _mean([float(x["power_w"]) for x in first if x.get("power_w") is not None])
    last_pwr = _mean([float(x["power_w"]) for x in last if x.get("power_w") is not None])
    first_hr = _mean([float(x["hr_avg"]) for x in first if x.get("hr_avg") is not None])
    last_hr = _mean([float(x["hr_avg"]) for x in last if x.get("hr_avg") is not None])

    pace_delta = (last_pace - first_pace) if first_pace is not None and last_pace is not None else None
    power_delta = (last_pwr - first_pwr) if first_pwr is not None and last_pwr is not None else None
    hr_delta = (last_hr - first_hr) if first_hr is not None and last_hr is not None else None

    interval_stability = "stable across reps"
    if pace_delta is not None:
        if pace_delta >= 8:
            interval_stability = "clear late fade"
        elif pace_delta >= 4:
            interval_stability = "slight late fade"
    if power_delta is not None:
        if power_delta <= -10:
            interval_stability = "clear late fade"
        elif interval_stability == "stable across reps" and power_delta <= -5:
            interval_stability = "slight late fade"

    if hr_delta is not None and power_delta is not None and pace_delta is not None:
        if hr_delta >= 3 and abs(power_delta) <= 5:
            coupling = "HR rose normally while power stayed stable"
        elif hr_delta >= 4 and power_delta <= -6:
            coupling = "power dropped while HR rose"
        elif abs(pace_delta) <= 4 and abs(power_delta) <= 5:
            coupling = "pace and power stayed aligned"
        else:
            coupling = "no meaningful execution breakdown"
    else:
        coupling = "no meaningful execution breakdown"

    interpretation = {
        "interval_stability": interval_stability,
        "coupling": coupling,
        "pace_delta_late_vs_early_sec_per_km": pace_delta,
        "power_delta_late_vs_early_w": power_delta,
        "hr_delta_late_vs_early_bpm": hr_delta,
    }
    return {
        "run_id": run_id,
        "available": True,
        "reason": None,
        "reps": reps,
        "summary": summary,
        "interpretation": interpretation,
    }


def _work_label_round_minutes(duration_sec: Any) -> int | None:
    if duration_sec is None:
        return None
    try:
        s = float(duration_sec)
    except (TypeError, ValueError):
        return None
    if s < 30:
        return 1 if s > 5 else None
    return max(1, int(round(s / 60.0)))


def _work_label_round_distance_m(dist_m: Any) -> int | None:
    if dist_m is None:
        return None
    try:
        d = float(dist_m)
    except (TypeError, ValueError):
        return None
    if d < 25:
        return None
    return int(round(d / 10.0) * 10)


def _format_work_block_mixed_minutes(min_ctr: Counter[int]) -> str:
    parts = [
        f"{c}x{m} min"
        for m, c in sorted(min_ctr.items(), key=lambda t: (-t[1], -t[0]))
    ]
    return " + ".join(parts)


def _format_work_block_mixed_meters(mtr_ctr: Counter[int]) -> str:
    parts = [
        f"{c}x{d} m"
        for d, c in sorted(mtr_ctr.items(), key=lambda t: (-t[1], -t[0]))
    ]
    return " + ".join(parts)


# Rounded distances typical of track/road reps; used to prefer ``NxDDD m`` over time when blocks are short.
_STANDARD_WORK_REP_DISTANCES_M: frozenset[int] = frozenset(
    {200, 400, 600, 800, 1000, 1200, 1600, 2000, 2400, 3000, 3200, 4000, 5000}
)


def derive_work_block_label_for_run(
    conn: sqlite3.Connection, run_id: str
) -> dict[str, Any] | None:
    """Human-readable work-block label from ``work`` segments only (e.g. ``3x10 min``, ``5x800 m``).

    Rounding: duration ≥30 s → nearest minute (at least 1 min); distance ≥25 m → nearest 10 m.

    Uniform blocks: if both time and distance are uniform, uses **minutes** when mean segment
    duration ≥10 min, else **meters** when the rounded distance is a common rep length
    (see ``_STANDARD_WORK_REP_DISTANCES_M``), else minutes.

    Mixed blocks: if multiple distance buckets and every block has distance, joins as
    ``3x800 m + 3x400 m``; else multiple minute buckets as ``2x10 min + 1x5 min``; else
    ``mixed work (N blocks)``.
    """
    ensure_run_segments_table(conn)
    run_id = str(run_id)
    if conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone() is None:
        return None
    typ = "COALESCE(segment_type_mapped, mapped_type)"
    sql = f"""
        SELECT duration_sec,
               COALESCE(distance_m, total_distance_m) AS distance_m
        FROM run_segments
        WHERE run_id = ? AND {typ} = 'work'
        ORDER BY COALESCE(segment_index, segment_idx) ASC
    """
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, (run_id,)).fetchall()
    if not rows:
        return {"run_id": run_id, "work_block_label": "no work blocks"}

    n = len(rows)
    minutes = [_work_label_round_minutes(r["duration_sec"]) for r in rows]
    meters = [_work_label_round_distance_m(r["distance_m"]) for r in rows]

    if n == 1:
        m0, d0 = minutes[0], meters[0]
        if m0 is not None:
            return {"run_id": run_id, "work_block_label": f"1x{m0} min"}
        if d0 is not None:
            return {"run_id": run_id, "work_block_label": f"1x{d0} m"}
        return {"run_id": run_id, "work_block_label": "no work blocks"}

    min_ctr = Counter(m for m in minutes if m is not None)
    mtr_ctr = Counter(d for d in meters if d is not None)
    all_have_dist = all(d is not None for d in meters)
    all_have_min = all(m is not None for m in minutes)

    uniform_dist = all_have_dist and len(mtr_ctr) == 1
    uniform_min = all_have_min and len(min_ctr) == 1

    if uniform_dist and uniform_min:
        d0 = next(iter(mtr_ctr))
        m0 = next(iter(min_ctr))
        mean_sec = sum(float(r["duration_sec"] or 0) for r in rows) / float(n)
        if mean_sec >= 600:
            return {"run_id": run_id, "work_block_label": f"{n}x{m0} min"}
        if d0 in _STANDARD_WORK_REP_DISTANCES_M:
            return {"run_id": run_id, "work_block_label": f"{n}x{d0} m"}
        return {"run_id": run_id, "work_block_label": f"{n}x{m0} min"}
    if uniform_dist:
        d0 = next(iter(mtr_ctr))
        return {"run_id": run_id, "work_block_label": f"{n}x{d0} m"}
    if uniform_min:
        m0 = next(iter(min_ctr))
        return {"run_id": run_id, "work_block_label": f"{n}x{m0} min"}
    if len(mtr_ctr) > 1 and all_have_dist:
        return {
            "run_id": run_id,
            "work_block_label": _format_work_block_mixed_meters(mtr_ctr),
        }
    if len(min_ctr) > 1:
        return {
            "run_id": run_id,
            "work_block_label": _format_work_block_mixed_minutes(min_ctr),
        }
    if len(mtr_ctr) > 1:
        return {
            "run_id": run_id,
            "work_block_label": _format_work_block_mixed_meters(mtr_ctr),
        }
    return {"run_id": run_id, "work_block_label": f"mixed work ({n} blocks)"}


_THRESHOLD_FAMILY_MINUTE_LABELS_EXACT: frozenset[str] = frozenset(
    {
        "1x18 min",
        "1x20 min",
        "1x22 min",
        "1x25 min",
        "1x28 min",
        "1x30 min",
        "1x35 min",
        "1x40 min",
        "1x45 min",
        "1x50 min",
        "1x55 min",
        "1x60 min",
        "2x8 min",
        "2x10 min",
        "2x12 min",
        "2x15 min",
        "2x18 min",
        "2x20 min",
        "2x25 min",
        "3x6 min",
        "3x8 min",
        "3x10 min",
        "3x12 min",
        "3x15 min",
        "4x5 min",
        "4x6 min",
        "4x8 min",
        "4x10 min",
        "5x5 min",
        "5x6 min",
        "6x6 min",
    }
)

_VO2_FAMILY_MINUTE_LABELS_EXACT: frozenset[str] = frozenset(
    {
        "5x2 min",
        "6x2 min",
        "8x2 min",
        "10x2 min",
        "12x2 min",
        "5x3 min",
        "6x3 min",
        "8x3 min",
        "10x3 min",
        "5x4 min",
        "6x4 min",
        "8x4 min",
        "10x1 min",
        "12x1 min",
    }
)


def _work_family_vo2_from_meters_label(label: str) -> bool:
    m = re.match(r"^(\d+)x(\d+) m$", label.strip())
    if not m:
        return False
    n, d = int(m.group(1)), int(m.group(2))
    if d == 400:
        return n >= 6
    if d == 600:
        return n >= 5
    if d == 800:
        return n >= 5
    if d == 1000:
        return n >= 4
    if d == 1200:
        return n >= 4
    if d == 2000:
        return n >= 3
    return False


def _work_family_vo2_from_minutes_pattern(label: str) -> bool:
    m = re.match(r"^(\d+)x(\d+) min$", label.strip())
    if not m:
        return False
    n, mm = int(m.group(1)), int(m.group(2))
    if mm <= 2 and n >= 5:
        return True
    if mm == 3 and n >= 6:
        return True
    if mm == 4 and n >= 6:
        return True
    return False


def _work_family_threshold_from_minutes_pattern(label: str) -> bool:
    m = re.match(r"^1x(\d+) min$", label.strip())
    if m and int(m.group(1)) >= 18:
        return True
    m = re.match(r"^(\d+)x(\d+) min$", label.strip())
    if not m:
        return False
    n, mm = int(m.group(1)), int(m.group(2))
    if _work_family_vo2_from_minutes_pattern(label):
        return False
    if mm >= 8 and 2 <= n <= 6:
        return True
    if mm >= 6 and n == 2:
        return True
    if mm == 6 and 3 <= n <= 5:
        return True
    if mm == 5 and n == 4:
        return True
    return False


def _work_family_from_pace_power_time(label: str, wo: dict[str, Any]) -> str | None:
    """Optional metric nudge when label already looks interval-like but patterns missed."""
    if "x" not in label or "no work" in label:
        return None
    blocks = int(wo.get("work_block_count") or 0)
    if blocks < 1:
        return None
    try:
        tt = float(wo.get("work_total_time_sec") or 0)
    except (TypeError, ValueError):
        tt = 0.0
    pace = wo.get("work_mean_pace_sec_per_km")
    pwr = wo.get("work_mean_power_w")
    try:
        pace_f = float(pace) if pace is not None else None
    except (TypeError, ValueError):
        pace_f = None
    try:
        pwr_f = float(pwr) if pwr is not None else None
    except (TypeError, ValueError):
        pwr_f = None
    if blocks <= 2 and tt >= 2400 and "min" in label and "+" not in label:
        return "threshold_session"
    if blocks >= 5 and pace_f is not None and pace_f <= 280:
        if pwr_f is None or pwr_f >= 200:
            return "vo2max_session"
    return None


def _classify_work_session_family(label: str, wo: dict[str, Any]) -> str:
    label_s = label.strip()
    if not label_s or "no work blocks" in label_s:
        return "other_interval_session"

    if label_s in _VO2_FAMILY_MINUTE_LABELS_EXACT:
        return "vo2max_session"
    if _work_family_vo2_from_meters_label(label_s):
        return "vo2max_session"
    if _work_family_vo2_from_minutes_pattern(label_s):
        return "vo2max_session"

    if label_s in _THRESHOLD_FAMILY_MINUTE_LABELS_EXACT:
        return "threshold_session"
    if _work_family_threshold_from_minutes_pattern(label_s):
        return "threshold_session"

    metric_fam = _work_family_from_pace_power_time(label_s, wo)
    if metric_fam:
        return metric_fam
    return "other_interval_session"


def _nudge_work_session_family_with_hint(
    family: str,
    label: str,
    wo: dict[str, Any],
    training_type_hint: str | None,
) -> str:
    if family != "other_interval_session" or not training_type_hint:
        return family
    hint = training_type_hint.strip().lower()
    blocks = int(wo.get("work_block_count") or 0)
    if blocks < 1:
        return family
    if hint == "test_or_vo2_session":
        if blocks >= 4 or _work_family_vo2_from_meters_label(label):
            return "vo2max_session"
    elif hint == "threshold_run":
        try:
            tt = float(wo.get("work_total_time_sec") or 0)
        except (TypeError, ValueError):
            tt = 0.0
        if tt >= 1200 or label.strip().startswith("1x"):
            return "threshold_session"
    return family


_EASY_RECOVERY_TRAINING_TYPES: frozenset[str] = frozenset({"easy_run", "recovery_run"})

# Long / steady aerobic: recurring duration-style runs compared against a shared pool
# (steady, long, easy, recovery) — no structural interval fingerprint required.
_LONG_STEADY_AEROBIC_UI_TYPES: frozenset[str] = frozenset({"steady_run", "long_run"})
_LONG_STEADY_AEROBIC_POOL_TYPES: frozenset[str] = frozenset(
    {"steady_run", "long_run", "easy_run", "recovery_run"}
)
# Prefer a prior session in a similar duration band (fraction of current duration).
_LONG_STEADY_DURATION_MATCH_LO = 0.55
_LONG_STEADY_DURATION_MATCH_HI = 1.45
# "Stable" load context vs prior (±15% duration / ±12% distance).
_LONG_STEADY_LOAD_REL_EPS_DURATION = 0.15
_LONG_STEADY_LOAD_REL_EPS_DISTANCE = 0.12


def _strong_interval_evidence_for_easy_hint(
    family: str, label: str, wo: dict[str, Any]
) -> bool:
    """Very strict override path: easy/recovery can remain interval-family only with strong evidence."""
    label_s = label.strip()
    blocks = int(wo.get("work_block_count") or 0)
    try:
        total_time_sec = float(wo.get("work_total_time_sec") or 0)
    except (TypeError, ValueError):
        total_time_sec = 0.0
    try:
        pace_f = float(wo.get("work_mean_pace_sec_per_km") or 0)
    except (TypeError, ValueError):
        pace_f = 0.0
    try:
        pwr_f = float(wo.get("work_mean_power_w") or 0)
    except (TypeError, ValueError):
        pwr_f = 0.0

    if family == "vo2max_session":
        vo2_shape = (
            label_s in _VO2_FAMILY_MINUTE_LABELS_EXACT
            or _work_family_vo2_from_minutes_pattern(label_s)
            or _work_family_vo2_from_meters_label(label_s)
        )
        ch = _parse_work_block_mixed_meters_chunks(label_s)
        vo2_mixed = (
            ch is not None
            and _mixed_short_fast_vo2_reps_structural_ok(ch)
            and _work_only_vo2_like_for_mixed_short_reps(wo)
        )
        return (vo2_shape or vo2_mixed) and blocks >= 6 and (pace_f <= 270 or pwr_f >= 230)

    if family == "threshold_session":
        thr_shape = (
            label_s in _THRESHOLD_FAMILY_MINUTE_LABELS_EXACT
            or _work_family_threshold_from_minutes_pattern(label_s)
        )
        ch_min = _parse_work_block_mixed_minutes_chunks(label_s)
        ch_mtr = _parse_work_block_mixed_meters_chunks(label_s)
        thr_mixed = (
            (ch_min is not None and _mixed_threshold_minutes_structural_ok(ch_min))
            or (ch_mtr is not None and _mixed_threshold_meters_structural_ok(ch_mtr))
        )
        return (thr_shape or thr_mixed) and blocks <= 4 and total_time_sec >= 1800 and (
            pace_f <= 320 or pwr_f >= 200
        )

    return False


def _apply_easy_recovery_guardrail(
    family: str, label: str, wo: dict[str, Any], training_type_hint: str | None
) -> str:
    """Prevent easy/recovery sessions from being over-routed into threshold/VO2 families."""
    if family not in {"threshold_session", "vo2max_session"}:
        return family
    hint = str(training_type_hint or "").strip().lower()
    if hint not in _EASY_RECOVERY_TRAINING_TYPES:
        return family
    if _strong_interval_evidence_for_easy_hint(family, label, wo):
        return family
    return "other_interval_session"


def _parse_work_block_mixed_meters_chunks(label: str) -> list[tuple[int, int]] | None:
    """``NxD m + ...`` segments only; returns list of (count, distance_m) or None if not mixed-meters."""
    label_s = label.strip()
    if "min" in label_s or "+" not in label_s or " m" not in label_s:
        return None
    parts = [p.strip() for p in label_s.split("+")]
    if len(parts) < 2:
        return None
    out: list[tuple[int, int]] = []
    for p in parts:
        m = re.match(r"^(\d+)x(\d+) m$", p)
        if not m:
            return None
        out.append((int(m.group(1)), int(m.group(2))))
    return out


def _parse_work_block_mixed_minutes_chunks(label: str) -> list[tuple[int, int]] | None:
    """``NxM min + ...`` segments only; returns list of (count, minutes) or None."""
    label_s = label.strip()
    if " m" in label_s or "min" not in label_s or "+" not in label_s:
        return None
    parts = [p.strip() for p in label_s.split("+")]
    if len(parts) < 2:
        return None
    out: list[tuple[int, int]] = []
    for p in parts:
        m = re.match(r"^(\d+)x(\d+) min$", p)
        if not m:
            return None
        out.append((int(m.group(1)), int(m.group(2))))
    return out


def _mixed_short_fast_vo2_reps_structural_ok(chunks: list[tuple[int, int]]) -> bool:
    """Conservative: ≥5 work blocks, ≥2 distance buckets, every distance ~400 m or ~800 m, both bands used."""
    total = sum(c for c, _ in chunks)
    if total < 5 or len(chunks) < 2:
        return False
    has_400_band = False
    has_800_band = False
    for _, d in chunks:
        if 360 <= d <= 440:
            has_400_band = True
        elif 760 <= d <= 840:
            has_800_band = True
        else:
            return False
    return has_400_band and has_800_band


def _mixed_threshold_minutes_structural_ok(chunks: list[tuple[int, int]]) -> bool:
    """Conservative threshold mix: mostly longer reps and enough total threshold-like time."""
    total_blocks = sum(c for c, _ in chunks)
    total_min = sum(c * m for c, m in chunks)
    if total_blocks < 2 or total_blocks > 8:
        return False
    if total_min < 18:
        return False
    short_blocks = sum(c for c, m in chunks if m <= 3)
    if short_blocks > 0:
        return False
    threshold_like_blocks = sum(c for c, m in chunks if m >= 5)
    if threshold_like_blocks / float(total_blocks) < 0.60:
        return False
    return True


def _mixed_threshold_meters_structural_ok(chunks: list[tuple[int, int]]) -> bool:
    """Conservative threshold-like mixed meters: predominantly longer reps, not short VO2 ladders."""
    total_blocks = sum(c for c, _ in chunks)
    if total_blocks < 2 or total_blocks > 8:
        return False
    short_blocks = sum(c for c, d in chunks if d <= 900)
    if short_blocks > 0:
        return False
    threshold_like_blocks = sum(c for c, d in chunks if d >= 1000)
    if threshold_like_blocks / float(total_blocks) < 0.75:
        return False
    return True


def _work_only_vo2_like_for_mixed_short_reps(wo: dict[str, Any]) -> bool:
    """Same power rule as metric VO2; pace band slightly wider than ``_work_family_from_pace_power_time`` so blended 400/800 work-means are not rejected when uniform 5x800 would pass per-segment."""
    pace = wo.get("work_mean_pace_sec_per_km")
    pwr = wo.get("work_mean_power_w")
    try:
        pace_f = float(pace) if pace is not None else None
    except (TypeError, ValueError):
        pace_f = None
    try:
        pwr_f = float(pwr) if pwr is not None else None
    except (TypeError, ValueError):
        pwr_f = None
    if pace_f is None or pace_f > 295:
        return False
    if pwr_f is not None and pwr_f < 200:
        return False
    return True


def derive_work_session_family_for_run(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    training_type_hint: str | None = None,
) -> dict[str, Any] | None:
    """Deterministic work-session family from work-only aggregates, block label, optional classifier hint.

    Families: ``threshold_session``, ``vo2max_session``, ``other_interval_session``.
    Primary rules: exact minute labels, meter-repeat heuristics, minute-pattern thresholds vs VO2.
    Secondary: if still ``other_interval_session``, ``training_type_hint`` may nudge using
    ``test_or_vo2_session`` / ``threshold_run`` from session analysis.
    """
    run_id = str(run_id)
    if conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone() is None:
        return None
    wo = aggregate_work_only_session_for_run(conn, run_id)
    wbl = derive_work_block_label_for_run(conn, run_id)
    if wo is None or wbl is None:
        return None
    label = str(wbl.get("work_block_label") or "")
    fam = _classify_work_session_family(label, wo)
    if fam == "other_interval_session":
        ch_min = _parse_work_block_mixed_minutes_chunks(label)
        if ch_min is not None and _mixed_threshold_minutes_structural_ok(ch_min):
            fam = "threshold_session"
            if _agenticrun_debug():
                print(
                    "work_session_family: mixed_threshold_minutes "
                    f"run_id={run_id} blocks={sum(c for c, _ in ch_min)} "
                    f"total_min={sum(c * m for c, m in ch_min)} "
                    f"label={label!r}",
                    flush=True,
                )
    if fam == "other_interval_session":
        ch_mtr = _parse_work_block_mixed_meters_chunks(label)
        if ch_mtr is not None and _mixed_threshold_meters_structural_ok(ch_mtr):
            fam = "threshold_session"
            if _agenticrun_debug():
                print(
                    "work_session_family: mixed_threshold_meters "
                    f"run_id={run_id} blocks={sum(c for c, _ in ch_mtr)} "
                    f"label={label!r}",
                    flush=True,
                )
    if fam == "other_interval_session":
        ch = _parse_work_block_mixed_meters_chunks(label)
        if (
            ch is not None
            and _mixed_short_fast_vo2_reps_structural_ok(ch)
            and _work_only_vo2_like_for_mixed_short_reps(wo)
        ):
            fam = "vo2max_session"
            if _agenticrun_debug():
                print(
                    "work_session_family: mixed_short_fast_vo2 "
                    f"run_id={run_id} blocks={sum(c for c, _ in ch)} "
                    f"pace_s_km={_fmt_seg_metric(wo.get('work_mean_pace_sec_per_km'))} "
                    f"pwr_w={_fmt_seg_metric(wo.get('work_mean_power_w'))} "
                    f"label={label!r}",
                    flush=True,
                )
    fam = _nudge_work_session_family_with_hint(fam, label, wo, training_type_hint)
    fam = _apply_easy_recovery_guardrail(fam, label, wo, training_type_hint)
    out: dict[str, Any] = {
        "run_id": run_id,
        "work_block_label": label,
        "work_session_family": fam,
    }
    if training_type_hint:
        out["training_type_hint"] = training_type_hint
    return out


_WORK_SESSION_FAMILY_HISTORY_FAMILIES: frozenset[str] = frozenset(
    {"threshold_session", "vo2max_session"}
)

# Window of up to this many deduplicated family rows ending at the selected run (Streamlit).
_FAMILY_HISTORY_WINDOW_FOR_SELECTED_RUN_MAX: int = 5


def _family_history_window_ending_at_selected(
    hist: list[dict[str, Any]],
    idx: int,
    *,
    max_rows: int = _FAMILY_HISTORY_WINDOW_FOR_SELECTED_RUN_MAX,
) -> tuple[list[dict[str, Any]], str | None]:
    """Newest-first slice of ``hist`` ending at ``hist[idx]``; baseline id is the in-family prior."""
    start = max(0, idx - (max_rows - 1))
    window = list(reversed(hist[start : idx + 1]))
    baseline_rid = str(hist[idx - 1]["run_id"]) if idx >= 1 else None
    return window, baseline_rid


def fetch_work_family_session_history(
    conn: sqlite3.Connection,
    work_session_family: str,
) -> list[dict[str, Any]]:
    """Historical work-only rows for runs in a given ``work_session_family``.

    Uses persisted ``run_segments`` (work rows) and ``runs.training_type`` as the optional
    nudge hint, matching :func:`derive_work_session_family_for_run`. Only
    ``threshold_session`` and ``vo2max_session`` are supported; any other family string
    returns an empty list.

    Rows are ordered by ``run_date`` ascending, then ``run_id``. Each dict includes
    ``run_date``, ``run_id``, ``work_block_label``, ``work_total_time_sec``,
    ``work_mean_pace_sec_per_km``, ``work_mean_power_w``, ``work_mean_hr_avg``,
    ``work_w_per_hr``, and ``work_session_family``.
    """
    if work_session_family not in _WORK_SESSION_FAMILY_HISTORY_FAMILIES:
        return []
    ensure_run_segments_table(conn)
    typ = "COALESCE(rs.segment_type_mapped, rs.mapped_type)"
    conn.row_factory = sqlite3.Row
    sql = f"""
        SELECT DISTINCT r.run_id, r.run_date, r.training_type
        FROM runs r
        INNER JOIN run_segments rs ON rs.run_id = r.run_id AND {typ} = 'work'
        ORDER BY r.run_date ASC, r.run_id ASC
    """
    candidates = conn.execute(sql).fetchall()
    out: list[dict[str, Any]] = []
    for row in candidates:
        run_id = str(row["run_id"])
        tt = row["training_type"]
        hint = tt if isinstance(tt, str) and tt.strip() else None
        wsf = derive_work_session_family_for_run(
            conn, run_id, training_type_hint=hint
        )
        if wsf is None or wsf.get("work_session_family") != work_session_family:
            continue
        wo = aggregate_work_only_session_for_run(conn, run_id)
        if wo is None:
            continue
        out.append(
            {
                "run_date": wo.get("run_date"),
                "run_id": wo.get("run_id"),
                "work_block_label": wsf.get("work_block_label"),
                "work_total_time_sec": wo.get("work_total_time_sec"),
                "work_mean_pace_sec_per_km": wo.get("work_mean_pace_sec_per_km"),
                "work_mean_power_w": wo.get("work_mean_power_w"),
                "work_mean_hr_avg": wo.get("work_mean_hr_avg"),
                "work_w_per_hr": wo.get("work_w_per_hr"),
                "work_session_family": work_session_family,
            }
        )
    return out


def fetch_dedup_work_family_session_history(
    conn: sqlite3.Connection,
    work_session_family: str,
) -> list[dict[str, Any]]:
    """Work-family history rows with duplicate activities collapsed (same dedup as latest-vs-prior)."""
    return _dedup_vo2max_family_history_rows(
        fetch_work_family_session_history(conn, work_session_family)
    )


def work_family_membership_diagnostic(
    conn: sqlite3.Connection,
    work_session_family: str,
    *,
    included_recent_limit: int = 12,
) -> dict[str, Any]:
    """Deterministic counts explaining work-family membership (read-only; same rules as history fetch).

    Scans every run that has at least one persisted ``work`` row in ``run_segments``, then applies
    the same ``derive_work_session_family_for_run`` + aggregate path as
    :func:`fetch_work_family_session_history`. Returns how many runs land in the requested family
    before/after deduplication, and how many were skipped for other reasons.
    """
    if work_session_family not in _WORK_SESSION_FAMILY_HISTORY_FAMILIES:
        return {"ok": False, "error": "unsupported_work_session_family"}
    ensure_run_segments_table(conn)
    typ = "COALESCE(rs.segment_type_mapped, rs.mapped_type)"
    conn.row_factory = sqlite3.Row
    sql = f"""
        SELECT DISTINCT r.run_id, r.run_date, r.training_type
        FROM runs r
        INNER JOIN run_segments rs ON rs.run_id = r.run_id AND {typ} = 'work'
        ORDER BY r.run_date ASC, r.run_id ASC
    """
    candidates = conn.execute(sql).fetchall()
    n_candidates = len(candidates)
    n_not_classified = 0
    n_classified_other_family = 0
    n_aggregate_failed_after_family_match = 0
    raw_rows: list[dict[str, Any]] = []

    for row in candidates:
        run_id = str(row["run_id"])
        tt = row["training_type"]
        hint = tt if isinstance(tt, str) and tt.strip() else None
        wsf = derive_work_session_family_for_run(conn, run_id, training_type_hint=hint)
        if wsf is None:
            n_not_classified += 1
            continue
        if wsf.get("work_session_family") != work_session_family:
            n_classified_other_family += 1
            continue
        wo = aggregate_work_only_session_for_run(conn, run_id)
        if wo is None:
            n_aggregate_failed_after_family_match += 1
            continue
        raw_rows.append(
            {
                "run_date": wo.get("run_date"),
                "run_id": wo.get("run_id"),
                "work_block_label": wsf.get("work_block_label"),
                "work_total_time_sec": wo.get("work_total_time_sec"),
                "work_mean_pace_sec_per_km": wo.get("work_mean_pace_sec_per_km"),
                "work_mean_power_w": wo.get("work_mean_power_w"),
                "work_mean_hr_avg": wo.get("work_mean_hr_avg"),
                "work_w_per_hr": wo.get("work_w_per_hr"),
                "work_session_family": work_session_family,
            }
        )

    deduped = _dedup_vo2max_family_history_rows(raw_rows)
    n_raw = len(raw_rows)
    n_dedup = len(deduped)
    tail = list(reversed(deduped))[: max(0, int(included_recent_limit))]
    included_recent = [
        {
            "run_date": str(r.get("run_date") if r.get("run_date") is not None else "-"),
            "run_id": str(r.get("run_id") or "-"),
            "work_block_label": str(r.get("work_block_label") or "-"),
        }
        for r in tail
    ]
    return {
        "ok": True,
        "work_session_family": work_session_family,
        "n_runs_with_work_segments": n_candidates,
        "n_not_classified": n_not_classified,
        "n_classified_other_family": n_classified_other_family,
        "n_aggregate_failed_after_family_match": n_aggregate_failed_after_family_match,
        "n_raw_in_target_family": n_raw,
        "n_deduped_in_target_family": n_dedup,
        "n_collapsed_by_dedup": max(0, n_raw - n_dedup),
        "included_recent": included_recent,
    }


def work_segment_family_distribution_diagnostic(
    conn: sqlite3.Connection,
    *,
    other_recent_limit: int = 15,
) -> dict[str, Any]:
    """Count derived ``work_session_family`` for every run with ≥1 stored work segment (read-only).

    Uses the same ``derive_work_session_family_for_run`` path as work-family history; does not
    change classification. Helps explain how many work-segment runs land in threshold vs VO2 vs
    ``other_interval_session`` vs unclassified.
    """
    ensure_run_segments_table(conn)
    typ = "COALESCE(rs.segment_type_mapped, rs.mapped_type)"
    conn.row_factory = sqlite3.Row
    sql = f"""
        SELECT DISTINCT r.run_id, r.run_date, r.training_type
        FROM runs r
        INNER JOIN run_segments rs ON rs.run_id = r.run_id AND {typ} = 'work'
        ORDER BY r.run_date ASC, r.run_id ASC
    """
    candidates = conn.execute(sql).fetchall()
    by_family: dict[str, int] = {}
    n_not_derived = 0
    other_rows: list[dict[str, str]] = []

    for row in candidates:
        run_id = str(row["run_id"])
        tt = row["training_type"]
        hint = tt if isinstance(tt, str) and tt.strip() else None
        wsf = derive_work_session_family_for_run(conn, run_id, training_type_hint=hint)
        if wsf is None:
            n_not_derived += 1
            continue
        fam_raw = wsf.get("work_session_family")
        fam = str(fam_raw).strip() if fam_raw is not None else ""
        if not fam:
            fam = "unknown"
        by_family[fam] = by_family.get(fam, 0) + 1
        if fam == "other_interval_session":
            other_rows.append(
                {
                    "run_date": str(row["run_date"] if row["run_date"] is not None else "-"),
                    "run_id": run_id,
                    "training_type": str(row["training_type"] or "-"),
                    "work_block_label": str(wsf.get("work_block_label") or "-"),
                }
            )

    lim = max(0, int(other_recent_limit))
    other_recent: list[dict[str, str]] = sorted(
        other_rows,
        key=lambda r: (r["run_date"], r["run_id"]),
        reverse=True,
    )[:lim]

    return {
        "ok": True,
        "n_runs_with_work_segments": len(candidates),
        "n_not_derived": n_not_derived,
        "by_family": by_family,
        "other_interval_recent": other_recent,
    }


def _round_seg_metric(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return 0


def interval_structure_fingerprint_for_run(
    conn: sqlite3.Connection, run_id: str
) -> dict[str, Any] | None:
    """Ordered work/recovery interval structure from persisted segments (for historical comparison).

    ``work_distances_m`` are ``COALESCE(distance_m, total_distance_m)`` rounded to integer meters.
    ``recovery_durations_s`` are rounded seconds when ``duration_sec`` is present (>= 1 s);
    otherwise rounded distance in meters if distance is present, else 0 (same integer list shape
    as recovery segments).

    ``fingerprint`` is the first 16 hex chars of SHA-256 over a canonical JSON payload of the
    two lists (order-sensitive).
    """
    ensure_run_segments_table(conn)
    if conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone() is None:
        return None
    typ = "COALESCE(rs.segment_type_mapped, rs.mapped_type)"
    sql = f"""
        SELECT {typ} AS typ,
               COALESCE(rs.distance_m, rs.total_distance_m) AS distance_m,
               rs.duration_sec AS duration_sec
        FROM run_segments rs
        WHERE rs.run_id = ? AND {typ} IN ('work', 'recovery')
        ORDER BY COALESCE(rs.segment_index, rs.segment_idx) ASC
    """
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, (run_id,)).fetchall()
    work_distances_m: list[int] = []
    recovery_durations_s: list[int] = []
    for r in rows:
        t = (r["typ"] or "").strip().lower()
        dist = r["distance_m"]
        dur = r["duration_sec"]
        if t == "work":
            work_distances_m.append(_round_seg_metric(dist))
        elif t == "recovery":
            use_dur = False
            if dur is not None:
                try:
                    use_dur = float(dur) >= 1.0
                except (TypeError, ValueError):
                    pass
            if use_dur:
                recovery_durations_s.append(_round_seg_metric(dur))
            else:
                recovery_durations_s.append(_round_seg_metric(dist))
    canonical = json.dumps(
        {"wd": work_distances_m, "rd": recovery_durations_s},
        separators=(",", ":"),
        sort_keys=True,
    )
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return {
        "run_id": run_id,
        "work_count": len(work_distances_m),
        "recovery_count": len(recovery_durations_s),
        "work_distances_m": work_distances_m,
        "recovery_durations_s": recovery_durations_s,
        "fingerprint": fingerprint,
    }


def fetch_comparable_interval_sessions_by_fingerprint(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    newest_first: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Session summaries for runs whose interval structure fingerprint matches ``run_id``'s.

    ``match_count`` is the total number of matching runs (including the anchor when present).
    ``matches`` contains per-run work/recovery aggregates (same shape as
    :func:`aggregate_work_recovery_segments_for_run`), ordered by ``run_date`` ascending by
    default, optionally reversed and truncated by ``limit``.
    """
    ensure_run_segments_table(conn)
    anchor = interval_structure_fingerprint_for_run(conn, run_id)
    if anchor is None:
        return {
            "fingerprint": None,
            "anchor_run_id": run_id,
            "match_count": 0,
            "matches": [],
        }
    fp = anchor["fingerprint"]
    conn.row_factory = sqlite3.Row
    ordered_ids = [
        str(r[0])
        for r in conn.execute(
            "SELECT run_id FROM runs ORDER BY run_date ASC, run_id ASC"
        ).fetchall()
    ]
    summaries: list[dict[str, Any]] = []
    for rid in ordered_ids:
        struct = interval_structure_fingerprint_for_run(conn, rid)
        if struct is None or struct.get("fingerprint") != fp:
            continue
        row = aggregate_work_recovery_segments_for_run(conn, rid)
        if row is not None:
            summaries.append(row)
    match_count = len(summaries)
    if newest_first:
        summaries.reverse()
    if limit is not None:
        summaries = summaries[: int(limit)]
    return {
        "fingerprint": fp,
        "anchor_run_id": run_id,
        "match_count": match_count,
        "matches": summaries,
    }


# Absolute tolerances for interval delta "stable" (no trend label).
_INTERVAL_DELTA_EPS_PACE_SEC_KM = 3.0
_INTERVAL_DELTA_EPS_POWER_W = 3.0
_INTERVAL_DELTA_EPS_HR_BPM = 2.0
_WORK_FAMILY_VO2_EPS_W_PER_HR = 0.05

# Easy / recovery sessions: slightly looser pace band than work-interval deltas; conservative power eps.
_EASY_AEROBIC_EPS_PACE_SEC_KM = 5.0
_EASY_AEROBIC_EPS_POWER_W = 4.0
_EASY_AEROBIC_EPS_W_PER_HR = 0.04


def _as_float_metric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pace_delta_status(
    current: float | None, baseline: float | None, eps: float
) -> tuple[float | None, str]:
    """Lower sec/km = faster."""
    if current is None or baseline is None:
        return None, "unknown"
    delta = current - baseline
    if abs(delta) <= eps:
        return delta, "stable"
    return delta, "faster" if delta < 0 else "slower"


def _higher_is_better_delta_status(
    current: float | None, baseline: float | None, eps: float
) -> tuple[float | None, str]:
    if current is None or baseline is None:
        return None, "unknown"
    delta = current - baseline
    if abs(delta) <= eps:
        return delta, "stable"
    return delta, "higher" if delta > 0 else "lower"


def _lower_is_better_delta_status(
    current: float | None, baseline: float | None, eps: float
) -> tuple[float | None, str]:
    if current is None or baseline is None:
        return None, "unknown"
    delta = current - baseline
    if abs(delta) <= eps:
        return delta, "stable"
    return delta, "lower" if delta < 0 else "higher"


def _w_per_hr_delta_status_vo2_family(
    current: float | None, baseline: float | None, eps: float
) -> tuple[float | None, str]:
    """Higher W/HR (more watts per bpm on work) → ``better`` for VO2-style intervals."""
    if current is None or baseline is None:
        return None, "unknown"
    delta = current - baseline
    if abs(delta) <= eps:
        return delta, "stable"
    return delta, "better" if delta > 0 else "worse"


def _canonical_activity_id_from_run_id(run_id: str) -> str | None:
    """Garmin-style ``run_id`` prefix ``YYYY-MM-DD_<digits>…`` → leading activity id digits."""
    m = re.match(r"^\d{4}-\d{2}-\d{2}_(\d{8,})", str(run_id).strip())
    return m.group(1) if m else None


def _vo2max_family_history_dedup_key(row: dict[str, Any]) -> tuple[Any, ...]:
    rid = row.get("run_id")
    if rid is not None:
        aid = _canonical_activity_id_from_run_id(str(rid))
        if aid is not None:
            return ("aid", aid)
    return (
        "sig",
        row.get("run_date"),
        row.get("work_block_label"),
        row.get("work_total_time_sec"),
        row.get("work_mean_power_w"),
        row.get("work_mean_pace_sec_per_km"),
    )


def _dedup_vo2max_family_history_rows(
    hist: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One row per duplicate-equivalent activity; last row in input order wins per key."""
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in hist:
        merged[_vo2max_family_history_dedup_key(row)] = row
    return sorted(
        merged.values(),
        key=lambda r: (str(r.get("run_date") or ""), str(r.get("run_id") or "")),
    )


def compare_vo2max_family_latest_vs_prior(
    conn: sqlite3.Connection,
) -> dict[str, Any]:
    """Latest ``vo2max_session`` vs the immediately prior one from family history (date order).

    Uses :func:`fetch_work_family_session_history` (``run_date`` ASC). Duplicate imports of
    the same activity (same Garmin id prefix in ``run_id``, else same work signature) are
    collapsed before picking current vs baseline. Returns raw numeric deltas and compact
    status labels: pace faster/slower/stable, power higher/lower/stable, HR lower/higher/stable,
    W/HR better/worse/stable (same eps conventions as interval deltas where applicable).
    """
    hist = _dedup_vo2max_family_history_rows(
        fetch_work_family_session_history(conn, "vo2max_session")
    )
    if len(hist) < 2:
        return {
            "insufficient_history": True,
            "reason": "fewer_than_two_vo2max_family_runs",
            "current": None,
            "baseline": None,
            "work_block_label": {
                "current": None,
                "baseline": None,
                "status": "unknown",
            },
            "metrics": {},
        }

    baseline_row = hist[-2]
    current_row = hist[-1]

    def slice_row(r: dict[str, Any]) -> dict[str, Any]:
        return {
            "run_date": r.get("run_date"),
            "run_id": r.get("run_id"),
            "work_block_label": r.get("work_block_label"),
            "work_mean_pace_sec_per_km": r.get("work_mean_pace_sec_per_km"),
            "work_mean_power_w": r.get("work_mean_power_w"),
            "work_mean_hr_avg": r.get("work_mean_hr_avg"),
            "work_w_per_hr": r.get("work_w_per_hr"),
        }

    lc = current_row.get("work_block_label")
    lb = baseline_row.get("work_block_label")
    sc = str(lc) if lc is not None else ""
    sb = str(lb) if lb is not None else ""
    label_status = "same" if sc == sb else "different"

    cur_pace = _as_float_metric(current_row.get("work_mean_pace_sec_per_km"))
    base_pace = _as_float_metric(baseline_row.get("work_mean_pace_sec_per_km"))
    d_pace, st_pace = _pace_delta_status(
        cur_pace, base_pace, _INTERVAL_DELTA_EPS_PACE_SEC_KM
    )

    cur_pwr = _as_float_metric(current_row.get("work_mean_power_w"))
    base_pwr = _as_float_metric(baseline_row.get("work_mean_power_w"))
    d_pwr, st_pwr = _higher_is_better_delta_status(
        cur_pwr, base_pwr, _INTERVAL_DELTA_EPS_POWER_W
    )

    cur_hr = _as_float_metric(current_row.get("work_mean_hr_avg"))
    base_hr = _as_float_metric(baseline_row.get("work_mean_hr_avg"))
    d_hr, st_hr = _lower_is_better_delta_status(
        cur_hr, base_hr, _INTERVAL_DELTA_EPS_HR_BPM
    )

    cur_wh = _as_float_metric(current_row.get("work_w_per_hr"))
    base_wh = _as_float_metric(baseline_row.get("work_w_per_hr"))
    d_wh, st_wh = _w_per_hr_delta_status_vo2_family(
        cur_wh, base_wh, _WORK_FAMILY_VO2_EPS_W_PER_HR
    )

    metrics: dict[str, Any] = {
        "work_mean_pace_sec_per_km": {
            "current": cur_pace,
            "baseline": base_pace,
            "delta": d_pace,
            "status": st_pace,
        },
        "work_mean_power_w": {
            "current": cur_pwr,
            "baseline": base_pwr,
            "delta": d_pwr,
            "status": st_pwr,
        },
        "work_mean_hr_avg": {
            "current": cur_hr,
            "baseline": base_hr,
            "delta": d_hr,
            "status": st_hr,
        },
        "work_w_per_hr": {
            "current": cur_wh,
            "baseline": base_wh,
            "delta": d_wh,
            "status": st_wh,
        },
    }

    return {
        "insufficient_history": False,
        "reason": None,
        "current": slice_row(current_row),
        "baseline": slice_row(baseline_row),
        "work_block_label": {
            "current": lc,
            "baseline": lb,
            "status": label_status,
        },
        "metrics": metrics,
    }


def compare_threshold_session_family_latest_vs_prior(
    conn: sqlite3.Connection,
) -> dict[str, Any]:
    """Latest ``threshold_session`` vs the immediately prior one from family history (date order).

    Uses :func:`fetch_work_family_session_history` (``run_date`` ASC). Duplicate imports of
    the same activity are collapsed with the same keying as VO2 family history before
    picking current vs baseline. Returns raw numeric deltas and compact status labels:
    pace faster/slower/stable, power higher/lower/stable, HR lower/higher/stable,
    W/HR better/worse/stable (same eps conventions as VO2 family comparison).
    """
    hist = _dedup_vo2max_family_history_rows(
        fetch_work_family_session_history(conn, "threshold_session")
    )
    if len(hist) < 2:
        return {
            "insufficient_history": True,
            "reason": "fewer_than_two_threshold_family_runs",
            "current": None,
            "baseline": None,
            "work_block_label": {
                "current": None,
                "baseline": None,
                "status": "unknown",
            },
            "metrics": {},
        }

    baseline_row = hist[-2]
    current_row = hist[-1]

    def slice_row(r: dict[str, Any]) -> dict[str, Any]:
        return {
            "run_date": r.get("run_date"),
            "run_id": r.get("run_id"),
            "work_block_label": r.get("work_block_label"),
            "work_mean_pace_sec_per_km": r.get("work_mean_pace_sec_per_km"),
            "work_mean_power_w": r.get("work_mean_power_w"),
            "work_mean_hr_avg": r.get("work_mean_hr_avg"),
            "work_w_per_hr": r.get("work_w_per_hr"),
        }

    lc = current_row.get("work_block_label")
    lb = baseline_row.get("work_block_label")
    sc = str(lc) if lc is not None else ""
    sb = str(lb) if lb is not None else ""
    label_status = "same" if sc == sb else "different"

    cur_pace = _as_float_metric(current_row.get("work_mean_pace_sec_per_km"))
    base_pace = _as_float_metric(baseline_row.get("work_mean_pace_sec_per_km"))
    d_pace, st_pace = _pace_delta_status(
        cur_pace, base_pace, _INTERVAL_DELTA_EPS_PACE_SEC_KM
    )

    cur_pwr = _as_float_metric(current_row.get("work_mean_power_w"))
    base_pwr = _as_float_metric(baseline_row.get("work_mean_power_w"))
    d_pwr, st_pwr = _higher_is_better_delta_status(
        cur_pwr, base_pwr, _INTERVAL_DELTA_EPS_POWER_W
    )

    cur_hr = _as_float_metric(current_row.get("work_mean_hr_avg"))
    base_hr = _as_float_metric(baseline_row.get("work_mean_hr_avg"))
    d_hr, st_hr = _lower_is_better_delta_status(
        cur_hr, base_hr, _INTERVAL_DELTA_EPS_HR_BPM
    )

    cur_wh = _as_float_metric(current_row.get("work_w_per_hr"))
    base_wh = _as_float_metric(baseline_row.get("work_w_per_hr"))
    d_wh, st_wh = _w_per_hr_delta_status_vo2_family(
        cur_wh, base_wh, _WORK_FAMILY_VO2_EPS_W_PER_HR
    )

    metrics: dict[str, Any] = {
        "work_mean_pace_sec_per_km": {
            "current": cur_pace,
            "baseline": base_pace,
            "delta": d_pace,
            "status": st_pace,
        },
        "work_mean_power_w": {
            "current": cur_pwr,
            "baseline": base_pwr,
            "delta": d_pwr,
            "status": st_pwr,
        },
        "work_mean_hr_avg": {
            "current": cur_hr,
            "baseline": base_hr,
            "delta": d_hr,
            "status": st_hr,
        },
        "work_w_per_hr": {
            "current": cur_wh,
            "baseline": base_wh,
            "delta": d_wh,
            "status": st_wh,
        },
    }

    return {
        "insufficient_history": False,
        "reason": None,
        "current": slice_row(current_row),
        "baseline": slice_row(baseline_row),
        "work_block_label": {
            "current": lc,
            "baseline": lb,
            "status": label_status,
        },
        "metrics": metrics,
    }


def fetch_easy_aerobic_run_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Runs classified as easy or recovery (``training_type``), chronological order."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT run_id, run_date, training_type, avg_pace_sec_km, avg_hr, avg_power
        FROM runs
        WHERE training_type IN ('easy_run', 'recovery_run')
        ORDER BY run_date ASC, run_id ASC
        """
    ).fetchall()
    return [dict(r) for r in rows]


def _easy_aerobic_dedup_key(row: dict[str, Any]) -> tuple[Any, ...]:
    rid = row.get("run_id")
    if rid is not None:
        aid = _canonical_activity_id_from_run_id(str(rid))
        if aid is not None:
            return ("aid", aid)
    return (
        "sig",
        row.get("run_date"),
        row.get("avg_pace_sec_km"),
        row.get("avg_hr"),
        row.get("avg_power"),
    )


def _dedup_easy_aerobic_rows(hist: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per duplicate-equivalent activity; last in chronological input wins per key."""
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in hist:
        merged[_easy_aerobic_dedup_key(row)] = row
    return sorted(
        merged.values(),
        key=lambda r: (str(r.get("run_date") or ""), str(r.get("run_id") or "")),
    )


def fetch_dedup_easy_aerobic_run_history(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Easy and recovery ``runs`` rows with duplicate activities collapsed (same dedup as efficiency trend)."""
    return _dedup_easy_aerobic_rows(fetch_easy_aerobic_run_rows(conn))


def _w_per_hr_session(avg_power: Any, avg_hr: Any) -> float | None:
    p = _as_float_metric(avg_power)
    h = _as_float_metric(avg_hr)
    if p is None or h is None or h <= 0.0:
        return None
    return p / h


def _easy_aerobic_compare_two_sessions(
    baseline_row: dict[str, Any],
    current_row: dict[str, Any],
) -> dict[str, Any]:
    """Latest vs prior easy/recovery row: pace, HR, optional power (if pace stable), optional W/HR."""
    metrics: dict[str, Any] = {}

    cur_pace = _as_float_metric(current_row.get("avg_pace_sec_km"))
    base_pace = _as_float_metric(baseline_row.get("avg_pace_sec_km"))
    d_pace, st_pace = _pace_delta_status(
        cur_pace, base_pace, _EASY_AEROBIC_EPS_PACE_SEC_KM
    )
    if st_pace != "unknown":
        metrics["avg_pace_sec_km"] = {
            "current": cur_pace,
            "baseline": base_pace,
            "delta": d_pace,
            "status": st_pace,
        }

    cur_hr = _as_float_metric(current_row.get("avg_hr"))
    base_hr = _as_float_metric(baseline_row.get("avg_hr"))
    d_hr, st_hr = _lower_is_better_delta_status(
        cur_hr, base_hr, _INTERVAL_DELTA_EPS_HR_BPM
    )
    if st_hr != "unknown":
        metrics["avg_hr"] = {
            "current": cur_hr,
            "baseline": base_hr,
            "delta": d_hr,
            "status": st_hr,
        }

    cur_pwr = _as_float_metric(current_row.get("avg_power"))
    base_pwr = _as_float_metric(baseline_row.get("avg_power"))
    if st_pace == "stable":
        d_pwr, st_pwr = _lower_is_better_delta_status(
            cur_pwr, base_pwr, _EASY_AEROBIC_EPS_POWER_W
        )
        if st_pwr != "unknown":
            metrics["avg_power"] = {
                "current": cur_pwr,
                "baseline": base_pwr,
                "delta": d_pwr,
                "status": st_pwr,
            }

    cur_wh = _w_per_hr_session(current_row.get("avg_power"), current_row.get("avg_hr"))
    base_wh = _w_per_hr_session(baseline_row.get("avg_power"), baseline_row.get("avg_hr"))
    if cur_wh is not None and base_wh is not None:
        d_wh, st_wh = _w_per_hr_delta_status_vo2_family(
            cur_wh, base_wh, _EASY_AEROBIC_EPS_W_PER_HR
        )
        if st_wh != "unknown":
            metrics["w_per_hr"] = {
                "current": cur_wh,
                "baseline": base_wh,
                "delta": d_wh,
                "status": st_wh,
            }

    return metrics


def _easy_aerobic_signal_from_metrics(metrics: dict[str, Any]) -> str:
    """Map directional statuses to a single conservative label."""

    def _skip_status(st: Any) -> bool:
        if st is None:
            return True
        return isinstance(st, float) and bool(math.isnan(st))

    good = 0
    bad = 0
    for key, g, b in (
        ("avg_pace_sec_km", "faster", "slower"),
        ("avg_hr", "lower", "higher"),
        ("avg_power", "lower", "higher"),
        ("w_per_hr", "better", "worse"),
    ):
        st = (metrics.get(key) or {}).get("status")
        if _skip_status(st):
            continue
        st_s = str(st).strip()
        if st_s == g:
            good += 1
        elif st_s == b:
            bad += 1
    if good == 0 and bad == 0:
        return "Stable"
    if good > 0 and bad > 0:
        return "Mixed"
    return "Improving" if good > bad else "Declining"


def derive_easy_aerobic_efficiency_trend(
    conn: sqlite3.Connection,
) -> dict[str, Any]:
    """Deterministic easy / aerobic efficiency vs the prior easy or recovery run.

    Uses ``runs`` rows with ``training_type`` in ``easy_run`` or ``recovery_run`` only.
    Compares the latest session to the immediately prior one (after duplicate collapse),
    using pace, HR, power (only when pace change is within the stable band), and W/HR when
    power and HR exist. Does **not** alter threshold/VO2 work-family logic.

    Returns ``easy_aerobic_signal`` in:
    ``Improving`` | ``Stable`` | ``Declining`` | ``Mixed`` | ``Insufficient history``.
    """
    raw = fetch_easy_aerobic_run_rows(conn)
    hist = _dedup_easy_aerobic_rows(raw)
    reason = "fewer_than_two_easy_or_recovery_runs"
    if len(hist) < 2:
        dl = (
            f"easy_aerobic_efficiency: signal=Insufficient history n={len(hist)} "
            f"reason={reason}"
        )
        return {
            "insufficient_history": True,
            "reason": reason,
            "easy_aerobic_signal": "Insufficient history",
            "current": None,
            "baseline": None,
            "metrics": {},
            "debug_line": dl,
        }

    baseline_row = hist[-2]
    current_row = hist[-1]

    def _slice(r: dict[str, Any]) -> dict[str, Any]:
        ap = _as_float_metric(r.get("avg_power"))
        ah = _as_float_metric(r.get("avg_hr"))
        wh = _w_per_hr_session(r.get("avg_power"), r.get("avg_hr"))
        return {
            "run_id": r.get("run_id"),
            "run_date": r.get("run_date"),
            "training_type": r.get("training_type"),
            "avg_pace_sec_km": _as_float_metric(r.get("avg_pace_sec_km")),
            "avg_hr": ah,
            "avg_power": ap,
            "w_per_hr": wh,
        }

    metrics = _easy_aerobic_compare_two_sessions(baseline_row, current_row)

    if not metrics:
        reason = "no_comparable_metrics"
        sig = "Insufficient history"
        dl = (
            f"easy_aerobic_efficiency: signal={sig} n={len(hist)} reason={reason} "
            f"cur={current_row.get('run_date')!s}"
        )
        return {
            "insufficient_history": True,
            "reason": reason,
            "easy_aerobic_signal": sig,
            "current": _slice(current_row),
            "baseline": _slice(baseline_row),
            "metrics": {},
            "debug_line": dl,
        }

    sig = _easy_aerobic_signal_from_metrics(metrics)
    reason = None
    mp = metrics.get("avg_pace_sec_km") or {}
    mh = metrics.get("avg_hr") or {}
    mw = metrics.get("avg_power") or {}
    mr = metrics.get("w_per_hr") or {}

    def _st(m: dict[str, Any]) -> str:
        s = m.get("status")
        return str(s) if s is not None else "-"

    dl = (
        f"easy_aerobic_efficiency: signal={sig} n={len(hist)} "
        f"pace={_st(mp)} hr={_st(mh)} "
        f"pwr={_st(mw)} w/hr={_st(mr)} "
        f"cur={current_row.get('run_date')!s} prev={baseline_row.get('run_date')!s}"
    )

    return {
        "insufficient_history": False,
        "reason": reason,
        "easy_aerobic_signal": sig,
        "current": _slice(current_row),
        "baseline": _slice(baseline_row),
        "metrics": metrics,
        "debug_line": dl,
    }


def fetch_long_steady_aerobic_pool_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Chronological easy/recovery/steady/long rows used for long-aerobic comparison pool."""
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(_LONG_STEADY_AEROBIC_POOL_TYPES))
    q = (
        "SELECT run_id, run_date, training_type, avg_pace_sec_km, avg_hr, avg_power, "
        "distance_km, duration_sec "
        f"FROM runs WHERE training_type IN ({placeholders}) "
        "ORDER BY run_date ASC, run_id ASC"
    )
    rows = conn.execute(q, tuple(sorted(_LONG_STEADY_AEROBIC_POOL_TYPES))).fetchall()
    return [dict(r) for r in rows]


def fetch_dedup_long_steady_aerobic_pool_history(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Same duplicate collapse as easy/recovery history (one row per activity signature)."""
    return _dedup_easy_aerobic_rows(fetch_long_steady_aerobic_pool_rows(conn))


def _load_context_length_status(
    current: float | None,
    baseline: float | None,
    *,
    rel_eps: float,
) -> tuple[float | None, str]:
    """Duration/distance vs prior: stable within ±rel_eps, else longer/shorter (not 'higher' HR sense)."""
    if current is None or baseline is None or baseline <= 0:
        return None, "unknown"
    delta = current - baseline
    ratio = current / baseline
    if abs(ratio - 1.0) <= rel_eps:
        return delta, "stable"
    return delta, "longer" if current > baseline else "shorter"


def _long_steady_aerobic_compare_sessions(
    baseline_row: dict[str, Any],
    current_row: dict[str, Any],
) -> dict[str, Any]:
    """Pace/HR/power/W/HR (easy-style) plus duration and distance context."""
    metrics: dict[str, Any] = dict(
        _easy_aerobic_compare_two_sessions(baseline_row, current_row)
    )

    cur_dur = _as_float_metric(current_row.get("duration_sec"))
    base_dur = _as_float_metric(baseline_row.get("duration_sec"))
    d_dur, st_dur = _load_context_length_status(
        cur_dur, base_dur, rel_eps=_LONG_STEADY_LOAD_REL_EPS_DURATION
    )
    if st_dur != "unknown" and cur_dur is not None and base_dur is not None:
        metrics["duration_sec"] = {
            "current": cur_dur,
            "baseline": base_dur,
            "delta": d_dur,
            "status": st_dur,
        }

    cur_dist = _as_float_metric(current_row.get("distance_km"))
    base_dist = _as_float_metric(baseline_row.get("distance_km"))
    d_dist, st_dist = _load_context_length_status(
        cur_dist, base_dist, rel_eps=_LONG_STEADY_LOAD_REL_EPS_DISTANCE
    )
    if st_dist != "unknown" and cur_dist is not None and base_dist is not None:
        metrics["distance_km"] = {
            "current": cur_dist,
            "baseline": base_dist,
            "delta": d_dist,
            "status": st_dist,
        }

    return metrics


def _pick_long_steady_baseline_index(
    hist: list[dict[str, Any]], idx: int
) -> int | None:
    """Prefer the most recent prior run with similar duration; else chronological prior."""
    if idx <= 0:
        return None
    cur = hist[idx]
    dur_c = _as_float_metric(cur.get("duration_sec"))
    lo, hi = _LONG_STEADY_DURATION_MATCH_LO, _LONG_STEADY_DURATION_MATCH_HI
    if dur_c is not None and dur_c >= 120.0:
        for j in range(idx - 1, -1, -1):
            dur_p = _as_float_metric(hist[j].get("duration_sec"))
            if dur_p is None or dur_p < 120.0:
                continue
            r = dur_c / dur_p
            if lo <= r <= hi:
                return j
    return idx - 1


def derive_long_steady_aerobic_vs_prior(
    conn: sqlite3.Connection,
    run_id: str,
) -> dict[str, Any]:
    """Compare this steady/long session to a deterministic prior from the shared aerobic pool.

    Pool: ``steady_run``, ``long_run``, ``easy_run``, ``recovery_run`` (deduplicated).
    Baseline: most recent **prior** row in time; when duration data exists, prefer the newest
    prior whose duration is within ~0.55–1.45× the current run (recurring long-aerobic anchor).
    """
    run_id = str(run_id).strip()
    hist = fetch_dedup_long_steady_aerobic_pool_history(conn)
    reason = "fewer_than_two_distinct_pool_runs"
    if len(hist) < 2:
        dl = (
            f"long_steady_aerobic: signal=Insufficient history n={len(hist)} "
            f"reason={reason} run_id={run_id!s}"
        )
        return {
            "insufficient_history": True,
            "reason": reason,
            "long_steady_signal": "Insufficient history",
            "current": None,
            "baseline": None,
            "metrics": {},
            "debug_line": dl,
            "pool_run_count": len(hist),
            "pool_history_recent": hist[-5:],
        }

    idx = next((i for i, r in enumerate(hist) if str(r.get("run_id")) == run_id), None)
    if idx is None:
        reason = "current_run_not_in_pool"
        dl = f"long_steady_aerobic: insufficient reason={reason} run_id={run_id!s}"
        return {
            "insufficient_history": True,
            "reason": reason,
            "long_steady_signal": "Insufficient history",
            "current": None,
            "baseline": None,
            "metrics": {},
            "debug_line": dl,
            "pool_run_count": len(hist),
            "pool_history_recent": hist[-5:],
        }

    if idx < 1:
        reason = "no_prior_in_pool"
        dl = (
            f"long_steady_aerobic: insufficient reason={reason} "
            f"n={len(hist)} run_id={run_id!s}"
        )
        return {
            "insufficient_history": True,
            "reason": reason,
            "long_steady_signal": "Insufficient history",
            "current": None,
            "baseline": None,
            "metrics": {},
            "debug_line": dl,
            "pool_run_count": len(hist),
            "pool_history_recent": hist[-5:],
        }

    b_idx = _pick_long_steady_baseline_index(hist, idx)
    if b_idx is None:
        reason = "no_baseline_index"
        return {
            "insufficient_history": True,
            "reason": reason,
            "long_steady_signal": "Insufficient history",
            "current": None,
            "baseline": None,
            "metrics": {},
            "debug_line": f"long_steady_aerobic: {reason} run_id={run_id!s}",
            "pool_run_count": len(hist),
            "pool_history_recent": hist[-5:],
        }

    baseline_row = hist[b_idx]
    current_row = hist[idx]

    def _slice(r: dict[str, Any]) -> dict[str, Any]:
        ap = _as_float_metric(r.get("avg_power"))
        ah = _as_float_metric(r.get("avg_hr"))
        wh = _w_per_hr_session(r.get("avg_power"), r.get("avg_hr"))
        return {
            "run_id": r.get("run_id"),
            "run_date": r.get("run_date"),
            "training_type": r.get("training_type"),
            "avg_pace_sec_km": _as_float_metric(r.get("avg_pace_sec_km")),
            "avg_hr": ah,
            "avg_power": ap,
            "w_per_hr": wh,
            "distance_km": _as_float_metric(r.get("distance_km")),
            "duration_sec": _as_float_metric(r.get("duration_sec")),
        }

    metrics = _long_steady_aerobic_compare_sessions(baseline_row, current_row)

    if not metrics:
        reason = "no_comparable_metrics"
        dl = (
            f"long_steady_aerobic: signal=Insufficient history n={len(hist)} "
            f"reason={reason} cur={current_row.get('run_date')!s}"
        )
        return {
            "insufficient_history": True,
            "reason": reason,
            "long_steady_signal": "Insufficient history",
            "current": _slice(current_row),
            "baseline": _slice(baseline_row),
            "metrics": {},
            "debug_line": dl,
            "pool_run_count": len(hist),
            "pool_history_recent": hist[-5:],
            "pool_position": {"index_1_based": idx + 1, "total": len(hist)},
        }

    sig = _easy_aerobic_signal_from_metrics(metrics)
    mp = metrics.get("avg_pace_sec_km") or {}
    mh = metrics.get("avg_hr") or {}
    mw = metrics.get("avg_power") or {}
    mr = metrics.get("w_per_hr") or {}

    def _st(m: dict[str, Any]) -> str:
        s = m.get("status")
        return str(s) if s is not None else "-"

    dl = (
        f"long_steady_aerobic: signal={sig} n={len(hist)} "
        f"pace={_st(mp)} hr={_st(mh)} pwr={_st(mw)} w/hr={_st(mr)} "
        f"cur={current_row.get('run_date')!s} prev={baseline_row.get('run_date')!s}"
    )

    return {
        "insufficient_history": False,
        "reason": None,
        "long_steady_signal": sig,
        "current": _slice(current_row),
        "baseline": _slice(baseline_row),
        "metrics": metrics,
        "debug_line": dl,
        "pool_run_count": len(hist),
        "pool_history_recent": hist[-5:],
        "pool_position": {"index_1_based": idx + 1, "total": len(hist)},
        "baseline_selection": "similar_duration_prior"
        if b_idx != idx - 1
        else "immediate_prior_in_pool",
    }


def _work_family_two_row_comparison(
    baseline_row: dict[str, Any],
    current_row: dict[str, Any],
) -> dict[str, Any]:
    """Pairwise threshold/VO2 family comparison (same metrics as latest-vs-prior helpers)."""

    def slice_row(r: dict[str, Any]) -> dict[str, Any]:
        return {
            "run_date": r.get("run_date"),
            "run_id": r.get("run_id"),
            "work_block_label": r.get("work_block_label"),
            "work_mean_pace_sec_per_km": r.get("work_mean_pace_sec_per_km"),
            "work_mean_power_w": r.get("work_mean_power_w"),
            "work_mean_hr_avg": r.get("work_mean_hr_avg"),
            "work_w_per_hr": r.get("work_w_per_hr"),
        }

    lc = current_row.get("work_block_label")
    lb = baseline_row.get("work_block_label")
    sc = str(lc) if lc is not None else ""
    sb = str(lb) if lb is not None else ""
    label_status = "same" if sc == sb else "different"

    cur_pace = _as_float_metric(current_row.get("work_mean_pace_sec_per_km"))
    base_pace = _as_float_metric(baseline_row.get("work_mean_pace_sec_per_km"))
    d_pace, st_pace = _pace_delta_status(
        cur_pace, base_pace, _INTERVAL_DELTA_EPS_PACE_SEC_KM
    )

    cur_pwr = _as_float_metric(current_row.get("work_mean_power_w"))
    base_pwr = _as_float_metric(baseline_row.get("work_mean_power_w"))
    d_pwr, st_pwr = _higher_is_better_delta_status(
        cur_pwr, base_pwr, _INTERVAL_DELTA_EPS_POWER_W
    )

    cur_hr = _as_float_metric(current_row.get("work_mean_hr_avg"))
    base_hr = _as_float_metric(baseline_row.get("work_mean_hr_avg"))
    d_hr, st_hr = _lower_is_better_delta_status(
        cur_hr, base_hr, _INTERVAL_DELTA_EPS_HR_BPM
    )

    cur_wh = _as_float_metric(current_row.get("work_w_per_hr"))
    base_wh = _as_float_metric(baseline_row.get("work_w_per_hr"))
    d_wh, st_wh = _w_per_hr_delta_status_vo2_family(
        cur_wh, base_wh, _WORK_FAMILY_VO2_EPS_W_PER_HR
    )

    metrics: dict[str, Any] = {
        "work_mean_pace_sec_per_km": {
            "current": cur_pace,
            "baseline": base_pace,
            "delta": d_pace,
            "status": st_pace,
        },
        "work_mean_power_w": {
            "current": cur_pwr,
            "baseline": base_pwr,
            "delta": d_pwr,
            "status": st_pwr,
        },
        "work_mean_hr_avg": {
            "current": cur_hr,
            "baseline": base_hr,
            "delta": d_hr,
            "status": st_hr,
        },
        "work_w_per_hr": {
            "current": cur_wh,
            "baseline": base_wh,
            "delta": d_wh,
            "status": st_wh,
        },
    }

    return {
        "insufficient_history": False,
        "reason": None,
        "current": slice_row(current_row),
        "baseline": slice_row(baseline_row),
        "work_block_label": {
            "current": lc,
            "baseline": lb,
            "status": label_status,
        },
        "metrics": metrics,
    }


def compare_selected_run_work_family_vs_prior(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    training_type_hint: str | None = None,
) -> dict[str, Any]:
    """Compare **this** run to the chronologically prior row in its work session family.

    Uses the same deduplicated family history as :func:`compare_vo2max_family_latest_vs_prior`.
    If the run is not ``threshold_session`` / ``vo2max_session``, or has no prior in-family
    session, returns a structured result for :func:`format_selected_run_interval_family_insight`.

    When the run is in the deduplicated family history, ``family_history_window`` lists up to
    five rows ending at this run (newest first); ``baseline_run_id_for_comparison`` is the
    immediate in-family prior when present (same as the insight baseline when applicable).
    """
    run_id = str(run_id).strip()
    wsf = derive_work_session_family_for_run(
        conn, run_id, training_type_hint=training_type_hint
    )
    if wsf is None:
        return {
            "applicable": False,
            "reason": "cannot_classify_run",
            "work_session_family": None,
            "selected_run_id": run_id,
        }

    fam = wsf.get("work_session_family")
    if fam not in _WORK_SESSION_FAMILY_HISTORY_FAMILIES:
        return {
            "applicable": False,
            "reason": "not_interval_family",
            "work_session_family": fam,
            "selected_run_id": run_id,
        }

    hist = _dedup_vo2max_family_history_rows(
        fetch_work_family_session_history(conn, str(fam))
    )
    idx = next(
        (i for i, r in enumerate(hist) if str(r.get("run_id")) == run_id),
        None,
    )
    if idx is None:
        return {
            "applicable": False,
            "reason": "run_not_in_family_history",
            "work_session_family": fam,
            "selected_run_id": run_id,
            "family_history_window": None,
            "baseline_run_id_for_comparison": None,
        }
    win, baseline_rid = _family_history_window_ending_at_selected(hist, idx)
    if idx < 1:
        return {
            "applicable": True,
            "insufficient_history": True,
            "reason": "no_prior_family_session",
            "work_session_family": fam,
            "selected_run_id": run_id,
            "family_history_window": win,
            "baseline_run_id_for_comparison": baseline_rid,
        }

    pair = _work_family_two_row_comparison(hist[idx - 1], hist[idx])
    out: dict[str, Any] = {
        **pair,
        "applicable": True,
        "work_session_family": fam,
        "selected_run_id": run_id,
        "family_history_window": win,
        "baseline_run_id_for_comparison": baseline_rid,
    }
    return out


def build_interval_family_insight_summary(
    conn: sqlite3.Connection,
) -> dict[str, Any]:
    """Deterministic bundle of latest-vs-prior work-family comparisons for threshold and VO2max.

    Each value is the full return dict from the corresponding compare function; shapes are
    unchanged.
    """
    return {
        "threshold_latest_vs_prior": compare_threshold_session_family_latest_vs_prior(conn),
        "vo2max_latest_vs_prior": compare_vo2max_family_latest_vs_prior(conn),
    }


def _format_interval_family_section(title: str, cmp: dict[str, Any]) -> str:
    # Compare functions set ``insufficient_history`` explicitly to False when data exists.
    if cmp.get("insufficient_history") is not False:
        reason = cmp.get("reason") if cmp.get("insufficient_history") is True else None
        rtext = str(reason) if reason is not None else "insufficient_or_unavailable"
        return f"{title}\n  Insufficient history ({rtext})."

    cur = cmp.get("current") or {}
    base = cmp.get("baseline") or {}
    wbl = cmp.get("work_block_label") or {}
    mm = cmp.get("metrics") or {}

    def _d(value: Any) -> str:
        return str(value) if value is not None else "?"

    def _st(metric_key: str) -> str:
        st = (mm.get(metric_key) or {}).get("status")
        return str(st) if st is not None else "unknown"

    cdate = cur.get("run_date")
    bdate = base.get("run_date")
    wls = wbl.get("status")
    wls_s = str(wls) if wls is not None else "unknown"

    line1 = (
        f"  Current {_d(cdate)} vs baseline {_d(bdate)}. "
        f"Work block label: {wls_s}."
    )
    line2 = (
        f"  Pace {_st('work_mean_pace_sec_per_km')}, power {_st('work_mean_power_w')}, "
        f"HR {_st('work_mean_hr_avg')}, W/HR {_st('work_w_per_hr')}."
    )
    return f"{title}\n{line1}\n{line2}"


def format_interval_family_insight_summary(summary: dict[str, Any]) -> str:
    """Deterministic compact text for :func:`build_interval_family_insight_summary` output."""
    th = summary.get("threshold_latest_vs_prior") or {}
    v2 = summary.get("vo2max_latest_vs_prior") or {}
    return "\n\n".join(
        (
            _format_interval_family_section("Threshold", th),
            _format_interval_family_section("VO2max", v2),
        )
    )


def format_selected_run_interval_family_insight(payload: dict[str, Any]) -> str:
    """Compact text for :func:`compare_selected_run_work_family_vs_prior` output."""
    if not payload.get("applicable"):
        return "No interval-family comparison available for this run."

    fam = payload.get("work_session_family")
    title = (
        "Threshold (this run vs prior in family)"
        if fam == "threshold_session"
        else (
            "VO2max (this run vs prior in family)"
            if fam == "vo2max_session"
            else "Interval (this run vs prior in family)"
        )
    )

    if payload.get("insufficient_history"):
        if payload.get("reason") == "no_prior_family_session":
            return (
                f"{title}\n"
                "  No prior session in this family to compare against "
                "(this is the earliest threshold/VO2 family session in your history)."
            )
        return f"{title}\n  Insufficient history."

    cmp_keys = (
        "insufficient_history",
        "reason",
        "current",
        "baseline",
        "work_block_label",
        "metrics",
    )
    cmp_payload = {k: payload[k] for k in cmp_keys if k in payload}
    return _format_interval_family_section(title, cmp_payload)


# When bundle build fails or run_id is missing; trace + console use this shape.
LLM_CONTEXT_METADATA_UNAVAILABLE: dict[str, Any] = {
    "deterministic_run_takeaway_used": "no",
    "interval_insight_used": "no",
    "family_history_used": "no",
    "recommendation_summary_used": "no",
    "work_session_family": None,
}


def format_llm_context_applied_log_line(metadata: dict[str, Any] | None) -> str:
    """One-line console proof that the LLM prompt used the deterministic bundle flags."""
    if not metadata:
        return (
            "llm_context_applied: takeaway=no interval=no family_history=no "
            "recommendation=no family=-"
        )
    fam = metadata.get("work_session_family")
    fam_s = fam if isinstance(fam, str) and fam.strip() else "-"
    return (
        "llm_context_applied: "
        f"takeaway={metadata.get('deterministic_run_takeaway_used', 'no')} "
        f"interval={metadata.get('interval_insight_used', 'no')} "
        f"family_history={metadata.get('family_history_used', 'no')} "
        f"recommendation={metadata.get('recommendation_summary_used', 'no')} "
        f"family={fam_s}"
    )


def build_llm_prompt_deterministic_bundle(
    conn: sqlite3.Connection,
    state: RunState,
) -> dict[str, Any]:
    """Compact deterministic findings + flags for LLM prompt injection (no LLM calls)."""
    no_flags = (
        "llm_prompt_context: deterministic_run_takeaway=no interval_insight=no family_history=no"
    )
    if not state.run_record or not str(state.run_record.run_id or "").strip():
        return {
            "llm_prompt_context_line": no_flags,
            "findings_text": "",
            "llm_context_metadata": dict(LLM_CONTEXT_METADATA_UNAVAILABLE),
            "prompt_grounding_audit": {
                "structured_recommendation_signals_in_prompt": "no",
                "family_history_block_in_prompt": "no",
                "recommendation_candidates_in_prompt": "no",
            },
        }

    run_id = str(state.run_record.run_id).strip()
    tt: str | None = None
    if state.analysis and isinstance(state.analysis.training_type, str):
        tts = state.analysis.training_type.strip()
        if tts:
            tt = tts

    sel = compare_selected_run_work_family_vs_prior(conn, run_id, training_type_hint=tt)
    interval_text = format_selected_run_interval_family_insight(sel).strip()
    win = sel.get("family_history_window") or []
    fh_flag = "yes" if win else "no"
    iv_flag = "yes" if sel.get("applicable") else "no"

    if win:
        fam_lines: list[str] = []
        for r in win:
            fam_lines.append(
                f"  - {r.get('run_date')}: work_block_label={r.get('work_block_label')!s}, "
                f"work_total_time_sec={r.get('work_total_time_sec')}, "
                f"work_mean_pace_sec_per_km={r.get('work_mean_pace_sec_per_km')}, "
                f"work_mean_power_w={r.get('work_mean_power_w')}, "
                f"work_mean_hr_avg={r.get('work_mean_hr_avg')}, "
                f"work_w_per_hr={r.get('work_w_per_hr')}"
            )
        family_blob = "\n".join(fam_lines)
    else:
        family_blob = (
            "(No rows in the compact same-family window, or not in threshold/VO2 family history.)"
        )

    rec_sum = (state.recommendation.recommendation_summary or "").strip()
    rec_line = (
        f"Recommendation summary (deterministic): {rec_sum}\n\n" if rec_sum else ""
    )

    sig = getattr(state.recommendation, "recommendation_signals", None) or {}
    sig_json = json.dumps(sig, ensure_ascii=False, indent=2) if sig else "{}"
    sig_block = (
        "--- Structured recommendation signals (deterministic; authoritative facts & tiers) ---\n"
        f"{sig_json}\n\n"
        "Coaching prioritization: use `prioritization_hints.primary_for_llm_coaching` for the main story of THIS "
        "selected run. Entries in `caution_signals` with priority `secondary` (e.g. easy/recovery drift on steady/long) "
        "are supporting context only — they must not override a strong primary read of the current session unless the "
        "text still fits.\n"
        "When training_type is steady_run or long_run and `comparable_aerobic_signal.available` is true, "
        "the primary athlete-facing narrative should come from this run versus that comparable aerobic prior. "
        "Upper-zone clustering and easy/recovery drift remain supporting context for spacing, not the headline.\n"
        "The lines labeled default deterministic next-session / load below are candidate guidance derived from the "
        "dominant rule; phrase recommendations in your own words and choose emphasis using the signals bundle.\n\n"
    )

    findings = (
        f"Training type: {state.analysis.training_type}\n"
        f"Trend: {state.trend.trend_label}\n"
        f"Execution quality: {state.analysis.execution_quality}\n"
        f"Fatigue signal: {state.trend.fatigue_signal}\n"
        f"Fitness signal: {state.trend.fitness_signal}\n"
        f"{sig_block}"
        f"Default next-session line (deterministic candidate from dominant rule): {state.recommendation.next_session}\n"
        f"Default load action (deterministic candidate): {state.recommendation.load_action}\n"
        f"{rec_line}"
        f"Interval insight (segment-based vs prior in same work family):\n{interval_text}\n"
        f"\n"
        f"Family history (compact rows, same work family, newest first):\n{family_blob}"
    )
    findings_stripped = findings.strip()
    rec_used = "yes" if rec_sum else "no"
    takeaway_used = "yes" if findings_stripped else "no"
    ws_fam = sel.get("work_session_family")
    tt_norm = str(state.analysis.training_type or "").strip().lower()
    if tt_norm in _LONG_STEADY_AEROBIC_UI_TYPES:
        # Keep technical family info, but avoid misleading "interval family" wording for steady/long.
        ws_fam = "long_steady_aerobic_shared_pool"
    llm_context_metadata: dict[str, Any] = {
        "deterministic_run_takeaway_used": takeaway_used,
        "interval_insight_used": iv_flag,
        "family_history_used": fh_flag,
        "recommendation_summary_used": rec_used,
        "work_session_family": ws_fam if ws_fam is not None else None,
    }

    cand = sig.get("recommendation_candidates") if isinstance(sig, dict) else None
    has_candidates = isinstance(cand, list) and len(cand) > 0
    has_structured_signals = bool(
        isinstance(sig, dict)
        and (
            sig.get("dominant_rule_id")
            or sig.get("primary_run_read")
            or sig.get("schema_version") is not None
            or has_candidates
        )
    )
    prompt_grounding_audit: dict[str, str] = {
        "structured_recommendation_signals_in_prompt": "yes" if has_structured_signals else "no",
        "family_history_block_in_prompt": "yes" if fh_flag == "yes" else "no",
        "recommendation_candidates_in_prompt": "yes" if has_candidates else "no",
    }

    return {
        "llm_prompt_context_line": (
            "llm_prompt_context: deterministic_run_takeaway=yes "
            f"interval_insight={iv_flag} family_history={fh_flag}"
        ),
        "findings_text": findings_stripped,
        "llm_context_metadata": llm_context_metadata,
        "prompt_grounding_audit": prompt_grounding_audit,
    }


def _summarize_family_window_trend(window: list[dict[str, Any]]) -> dict[str, Any]:
    if len(window) < 2:
        return {
            "window_size": len(window),
            "transitions_analyzed": 0,
            "status_counts": {},
        }

    status_keys = (
        "work_mean_pace_sec_per_km",
        "work_mean_power_w",
        "work_mean_hr_avg",
        "work_w_per_hr",
    )
    counts: Counter[str] = Counter()
    for i in range(1, len(window)):
        pair = _work_family_two_row_comparison(window[i], window[i - 1])
        metrics = pair.get("metrics") or {}
        for key in status_keys:
            st = ((metrics.get(key) or {}).get("status") or "unknown").strip()
            if st:
                counts[st] += 1
    return {
        "window_size": len(window),
        "transitions_analyzed": max(0, len(window) - 1),
        "status_counts": dict(counts),
    }


def build_llm_context_progress_bundle(
    conn: sqlite3.Connection,
    state: RunState,
) -> dict[str, Any]:
    """Compact deterministic comparison bundle for context/progress LLM interpretation."""
    if not state.run_record or not str(state.run_record.run_id or "").strip():
        return {"available": False, "reason": "missing_run_id"}

    run_id = str(state.run_record.run_id).strip()
    tt = (state.analysis.training_type or "").strip()
    tt_hint = tt or None

    def _pick_metric(metrics: dict[str, Any], key: str) -> dict[str, Any]:
        m = metrics.get(key) or {}
        return {
            "current": m.get("current"),
            "prior": m.get("baseline"),
            "delta": m.get("delta"),
            "status": m.get("status"),
        }

    def _best_recent_indicator(status_counts: dict[str, Any]) -> str | None:
        if not status_counts:
            return None
        priority = (
            "better",
            "faster",
            "lower",
            "higher",
            "stable",
            "worse",
            "slower",
            "unknown",
        )
        best = max(
            (str(k), int(v))
            for k, v in status_counts.items()
            if isinstance(v, (int, float))
        )
        best_status = best[0]
        best_count = best[1]
        # If ties exist, prefer statuses that are easier to interpret as directional.
        tied = [str(k) for k, v in status_counts.items() if int(v) == best_count]
        for p in priority:
            if p in tied:
                best_status = p
                break
        return f"{best_status} ({best_count} recent status hits)"

    current_run: dict[str, Any] = {
        "run_id": run_id,
        "run_date": state.run_record.run_date,
        "training_type": state.analysis.training_type,
        "intensity_label": state.analysis.intensity_label,
        "execution_quality": state.analysis.execution_quality,
        "trend_label": state.trend.trend_label,
        "fitness_signal": state.trend.fitness_signal,
        "fatigue_signal": state.trend.fatigue_signal,
        "next_session": state.recommendation.next_session,
        "load_action": state.recommendation.load_action,
        "distance_km": state.run_record.distance_km,
        "duration_sec": state.run_record.duration_sec,
    }

    selected_family = compare_selected_run_work_family_vs_prior(
        conn, run_id, training_type_hint=tt_hint
    )
    interval_fp_cmp = compare_interval_session_vs_prior(conn, run_id)
    interval_fp_pack = fetch_comparable_interval_sessions_by_fingerprint(
        conn, run_id, newest_first=False, limit=8
    )
    easy_cmp = derive_easy_aerobic_efficiency_trend(conn)
    easy_hist = fetch_dedup_easy_aerobic_run_history(conn)
    easy_idx = next(
        (i for i, r in enumerate(easy_hist) if str(r.get("run_id")) == run_id),
        None,
    )

    comparable_run: dict[str, Any] = {
        "match_type": "none",
        "reason": "no_direct_comparable_context",
        "prior_run": None,
        "work_block_label": None,
        "metrics": {},
    }
    family_context: dict[str, Any] = {
        "session_family": "other",
        "recent_family_trend": None,
        "best_recent_indicator": None,
        "position_in_recent_family": None,
        "recent_family_runs": [],
        "weekly_load_context": {
            "trend_label": state.trend.trend_label,
            "load_action": state.recommendation.load_action,
        },
    }

    # Structured-workout family path (threshold / VO2)
    if selected_family.get("applicable"):
        fam = str(selected_family.get("work_session_family") or "other")
        family_window = list(selected_family.get("family_history_window") or [])
        trend = _summarize_family_window_trend(family_window)
        family_context["session_family"] = fam
        family_context["recent_family_trend"] = trend
        family_context["best_recent_indicator"] = _best_recent_indicator(
            trend.get("status_counts") or {}
        )
        if family_window:
            ids = [str(r.get("run_id")) for r in family_window]
            if run_id in ids:
                # family_window is newest-first; report 1-based position from oldest for readability.
                rev = list(reversed(ids))
                family_context["position_in_recent_family"] = {
                    "index_1_based": rev.index(run_id) + 1,
                    "total": len(ids),
                }
            family_context["recent_family_runs"] = [
                {
                    "run_id": r.get("run_id"),
                    "run_date": r.get("run_date"),
                    "work_block_label": r.get("work_block_label"),
                    "work_mean_pace_sec_per_km": r.get("work_mean_pace_sec_per_km"),
                    "work_mean_power_w": r.get("work_mean_power_w"),
                    "work_mean_hr_avg": r.get("work_mean_hr_avg"),
                    "work_w_per_hr": r.get("work_w_per_hr"),
                }
                for r in family_window[:5]
            ]

        if interval_fp_cmp.get("insufficient_history") is False:
            comparable_run = {
                "match_type": "exact_fingerprint_prior",
                "reason": "same_interval_structure_fingerprint",
                "prior_run": {
                    "run_id": interval_fp_cmp.get("baseline_run_id"),
                    "run_date": interval_fp_cmp.get("baseline_run_date"),
                },
                "work_block_label": None,
                "metrics": {
                    "work_pace": _pick_metric(
                        interval_fp_cmp.get("metrics") or {},
                        "work_mean_pace_sec_per_km",
                    ),
                    "work_power": _pick_metric(
                        interval_fp_cmp.get("metrics") or {},
                        "work_mean_power_w",
                    ),
                    "recovery_hr": _pick_metric(
                        interval_fp_cmp.get("metrics") or {},
                        "recovery_mean_hr_avg",
                    ),
                    "recovery_power": _pick_metric(
                        interval_fp_cmp.get("metrics") or {},
                        "recovery_mean_power_w",
                    ),
                },
            }
            current_run["work_only_metrics"] = {
                "fingerprint": interval_fp_cmp.get("fingerprint"),
                "fingerprint_match_count": interval_fp_pack.get("match_count"),
            }
        elif selected_family.get("insufficient_history") is not True:
            fm = selected_family.get("metrics") or {}
            comparable_run = {
                "match_type": (
                    "family_same_block"
                    if ((selected_family.get("work_block_label") or {}).get("status") == "same")
                    else "family_near_match"
                ),
                "reason": "same_work_session_family_fallback",
                "prior_run": {
                    "run_id": (selected_family.get("baseline") or {}).get("run_id"),
                    "run_date": (selected_family.get("baseline") or {}).get("run_date"),
                },
                "work_block_label": selected_family.get("work_block_label"),
                "metrics": {
                    "work_pace": _pick_metric(fm, "work_mean_pace_sec_per_km"),
                    "work_power": _pick_metric(fm, "work_mean_power_w"),
                    "work_hr": _pick_metric(fm, "work_mean_hr_avg"),
                    "work_w_per_hr": _pick_metric(fm, "work_w_per_hr"),
                },
            }
            current_run["work_only_metrics"] = {
                "work_block_label": (selected_family.get("current") or {}).get("work_block_label"),
                "work_mean_pace_sec_per_km": (selected_family.get("current") or {}).get("work_mean_pace_sec_per_km"),
                "work_mean_power_w": (selected_family.get("current") or {}).get("work_mean_power_w"),
                "work_mean_hr_avg": (selected_family.get("current") or {}).get("work_mean_hr_avg"),
                "work_w_per_hr": (selected_family.get("current") or {}).get("work_w_per_hr"),
            }
        else:
            comparable_run = {
                "match_type": "family_insufficient_history",
                "reason": str(selected_family.get("reason") or "no_prior_family_session"),
                "prior_run": None,
                "work_block_label": None,
                "metrics": {},
            }

    # Easy/recovery path for non-structured runs.
    elif tt in _EASY_RECOVERY_TRAINING_TYPES:
        family_context["session_family"] = "easy_aerobic"
        family_context["recent_family_trend"] = {
            "easy_aerobic_signal": easy_cmp.get("easy_aerobic_signal"),
            "insufficient_history": easy_cmp.get("insufficient_history"),
        }
        family_context["best_recent_indicator"] = easy_cmp.get("easy_aerobic_signal")
        if easy_idx is not None:
            family_context["position_in_recent_family"] = {
                "index_1_based": easy_idx + 1,
                "total": len(easy_hist),
            }
        family_context["recent_family_runs"] = [
            {
                "run_id": r.get("run_id"),
                "run_date": r.get("run_date"),
                "training_type": r.get("training_type"),
                "avg_pace_sec_km": r.get("avg_pace_sec_km"),
                "avg_hr": r.get("avg_hr"),
                "avg_power": r.get("avg_power"),
            }
            for r in easy_hist[-5:]
        ]
        if (
            easy_cmp.get("insufficient_history") is False
            and str((easy_cmp.get("current") or {}).get("run_id")) == run_id
        ):
            em = easy_cmp.get("metrics") or {}
            comparable_run = {
                "match_type": "easy_aerobic_prior",
                "reason": "prior_easy_or_recovery_run",
                "prior_run": {
                    "run_id": (easy_cmp.get("baseline") or {}).get("run_id"),
                    "run_date": (easy_cmp.get("baseline") or {}).get("run_date"),
                },
                "work_block_label": None,
                "metrics": {
                    "easy_pace": _pick_metric(em, "avg_pace_sec_km"),
                    "easy_hr": _pick_metric(em, "avg_hr"),
                    "easy_power": _pick_metric(em, "avg_power"),
                    "easy_w_per_hr": _pick_metric(em, "w_per_hr"),
                },
            }

    # Steady / long aerobic: compare within a shared pool (includes easy/recovery anchors).
    elif tt in _LONG_STEADY_AEROBIC_UI_TYPES:
        ls_cmp = derive_long_steady_aerobic_vs_prior(conn, run_id)
        pool_recent = list(ls_cmp.get("pool_history_recent") or [])
        family_context["session_family"] = "long_steady_aerobic"
        family_context["recent_family_trend"] = {
            "long_steady_signal": ls_cmp.get("long_steady_signal"),
            "insufficient_history": ls_cmp.get("insufficient_history"),
            "pool_run_count": ls_cmp.get("pool_run_count"),
            "baseline_mode": ls_cmp.get("baseline_selection"),
        }
        family_context["best_recent_indicator"] = ls_cmp.get("long_steady_signal")
        pos = ls_cmp.get("pool_position")
        if isinstance(pos, dict) and pos.get("total"):
            family_context["position_in_recent_family"] = pos
        family_context["recent_family_runs"] = [
            {
                "run_id": r.get("run_id"),
                "run_date": r.get("run_date"),
                "training_type": r.get("training_type"),
                "duration_sec": r.get("duration_sec"),
                "distance_km": r.get("distance_km"),
                "avg_pace_sec_km": r.get("avg_pace_sec_km"),
                "avg_hr": r.get("avg_hr"),
                "avg_power": r.get("avg_power"),
            }
            for r in pool_recent
        ]
        if (
            ls_cmp.get("insufficient_history") is False
            and str((ls_cmp.get("current") or {}).get("run_id")) == run_id
        ):
            em = ls_cmp.get("metrics") or {}
            comparable_run = {
                "match_type": "long_steady_aerobic_prior",
                "reason": "prior_session_in_shared_aerobic_pool",
                "prior_run": {
                    "run_id": (ls_cmp.get("baseline") or {}).get("run_id"),
                    "run_date": (ls_cmp.get("baseline") or {}).get("run_date"),
                    "training_type": (ls_cmp.get("baseline") or {}).get("training_type"),
                },
                "work_block_label": None,
                "metrics": {
                    "avg_pace_sec_km": _pick_metric(em, "avg_pace_sec_km"),
                    "avg_hr": _pick_metric(em, "avg_hr"),
                    "avg_power": _pick_metric(em, "avg_power"),
                    "w_per_hr": _pick_metric(em, "w_per_hr"),
                    "duration_sec": _pick_metric(em, "duration_sec"),
                    "distance_km": _pick_metric(em, "distance_km"),
                },
            }

    if _agenticrun_debug():
        print(
            "llm_context_progress_bundle: "
            f"run_id={run_id} "
            f"family={family_context.get('session_family')} "
            f"match_type={comparable_run.get('match_type')}",
            flush=True,
        )

    rec_sig = getattr(state.recommendation, "recommendation_signals", None) or {}

    return {
        "available": True,
        "run_id": run_id,
        "current_run": current_run,
        "comparable_run": comparable_run,
        "family_context": family_context,
        "recommendation_signals": rec_sig,
    }


def compare_interval_session_vs_prior(
    conn: sqlite3.Connection,
    run_id: str,
) -> dict[str, Any]:
    """Compare the current run to **prior** structurally identical sessions (same fingerprint).

    Baseline: the **immediately previous** matching run in chronological order
    (``run_date`` ASC, ``run_id`` ASC). The current run is excluded from that baseline;
    if there is no strictly earlier matching run, returns an insufficient-history result.

    Metrics: raw deltas and compact status labels (stable within epsilon).
    """
    run_id = str(run_id)
    pack = fetch_comparable_interval_sessions_by_fingerprint(
        conn, run_id, newest_first=False, limit=None
    )
    fp = pack.get("fingerprint")
    if fp is None:
        return {
            "fingerprint": None,
            "baseline_mode": None,
            "insufficient_history": True,
            "reason": "anchor_run_not_found",
            "prior_count": 0,
            "matches_used": 0,
            "baseline_run_id": None,
            "baseline_run_date": None,
            "current_run_id": run_id,
            "metrics": {},
        }

    matches: list[dict[str, Any]] = list(pack.get("matches") or [])
    idx = next(
        (i for i, m in enumerate(matches) if str(m.get("run_id")) == run_id),
        None,
    )
    if idx is None:
        return {
            "fingerprint": fp,
            "baseline_mode": None,
            "insufficient_history": True,
            "reason": "current_run_not_in_fingerprint_match_set",
            "prior_count": 0,
            "matches_used": 0,
            "baseline_run_id": None,
            "baseline_run_date": None,
            "current_run_id": run_id,
            "metrics": {},
        }

    prior_count = idx
    if prior_count < 1:
        return {
            "fingerprint": fp,
            "baseline_mode": None,
            "insufficient_history": True,
            "reason": "no_prior_structurally_identical_runs",
            "prior_count": 0,
            "matches_used": 0,
            "baseline_run_id": None,
            "baseline_run_date": None,
            "current_run_id": run_id,
            "metrics": {},
        }

    baseline_row = matches[idx - 1]
    current_row = matches[idx]
    b_rid = baseline_row.get("run_id")
    b_date = baseline_row.get("run_date")

    def one(
        key: str,
        kind: str,
    ) -> dict[str, Any]:
        cur_v = _as_float_metric(current_row.get(key))
        base_v = _as_float_metric(baseline_row.get(key))
        if kind == "pace":
            d, st = _pace_delta_status(
                cur_v, base_v, _INTERVAL_DELTA_EPS_PACE_SEC_KM
            )
        elif kind == "power":
            d, st = _higher_is_better_delta_status(
                cur_v, base_v, _INTERVAL_DELTA_EPS_POWER_W
            )
        elif kind == "rec_hr":
            d, st = _lower_is_better_delta_status(
                cur_v, base_v, _INTERVAL_DELTA_EPS_HR_BPM
            )
        else:
            d, st = None, "unknown"
        return {
            "current": cur_v,
            "baseline": base_v,
            "delta": d,
            "status": st,
        }

    metrics: dict[str, Any] = {
        "work_mean_pace_sec_per_km": one("work_mean_pace_sec_per_km", "pace"),
        "work_mean_power_w": one("work_mean_power_w", "power"),
        "recovery_mean_hr_avg": one("recovery_mean_hr_avg", "rec_hr"),
        "recovery_mean_power_w": one("recovery_mean_power_w", "power"),
    }

    return {
        "fingerprint": fp,
        "baseline_mode": "previous_matching_run",
        "insufficient_history": False,
        "reason": None,
        "prior_count": prior_count,
        "matches_used": prior_count,
        "baseline_run_id": b_rid,
        "baseline_run_date": b_date,
        "current_run_id": run_id,
        "metrics": metrics,
    }


def _dedupe_round_work_metric(value: Any, ndigits: int) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


def _work_fallback_duplicate_signature(
    conn: sqlite3.Connection, run_id: str
) -> tuple[Any, ...] | None:
    """Strict persisted work-only signature for older rows without ``fit_activity_key``."""
    wo = aggregate_work_only_session_for_run(conn, run_id)
    wbl = derive_work_block_label_for_run(conn, run_id)
    if wo is None or wbl is None:
        return None
    label = str(wbl.get("work_block_label") or "").strip()
    if not label or "no work" in label.lower():
        return None
    blocks = int(wo.get("work_block_count") or 0)
    if blocks < 1:
        return None
    rd = wo.get("run_date")
    if rd is None:
        return None
    tt = _dedupe_round_work_metric(wo.get("work_total_time_sec"), 1)
    pwr = _dedupe_round_work_metric(wo.get("work_mean_power_w"), 2)
    pace = _dedupe_round_work_metric(wo.get("work_mean_pace_sec_per_km"), 2)
    if tt is None or pace is None:
        return None
    return ("wsig", str(rd), label, tt, pwr, pace)


def _work_fallback_skip_reason(
    conn: sqlite3.Connection, run_id: str
) -> tuple[tuple[Any, ...] | None, str | None]:
    """Same acceptance rules as :func:`_work_fallback_duplicate_signature`; returns skip reason."""
    wo = aggregate_work_only_session_for_run(conn, run_id)
    if wo is None:
        return None, "aggregate_work_only_session_for_run returned None"
    wbl = derive_work_block_label_for_run(conn, run_id)
    if wbl is None:
        return None, "derive_work_block_label_for_run returned None"
    label = str(wbl.get("work_block_label") or "").strip()
    if not label:
        return None, "empty work_block_label"
    if "no work" in label.lower():
        return None, f"work_block_label excluded (no-work): {label!r}"
    blocks = int(wo.get("work_block_count") or 0)
    if blocks < 1:
        return None, f"work_block_count < 1 (got {blocks})"
    rd = wo.get("run_date")
    if rd is None:
        return None, "work aggregate run_date is None"
    tt = _dedupe_round_work_metric(wo.get("work_total_time_sec"), 1)
    pwr = _dedupe_round_work_metric(wo.get("work_mean_power_w"), 2)
    pace = _dedupe_round_work_metric(wo.get("work_mean_pace_sec_per_km"), 2)
    if tt is None:
        return None, "work_total_time_sec missing or not numeric after aggregate"
    if pace is None:
        return None, "work_mean_pace_sec_per_km missing or not numeric after aggregate"
    sig = ("wsig", str(rd), label, tt, pwr, pace)
    return sig, None


def _dedupe_projected_impact_for_remove_ids(
    conn: sqlite3.Connection, remove_ids: list[str]
) -> dict[str, int]:
    """Row counts that :func:`_delete_run_and_dependents` would touch for ``remove_ids``."""
    seg = 0
    zp = 0
    zn = 0
    for rid in remove_ids:
        r = str(rid)
        seg += int(
            conn.execute(
                "SELECT COUNT(*) FROM run_segments WHERE run_id = ?", (r,)
            ).fetchone()[0]
        )
        zp += int(
            conn.execute(
                "SELECT COUNT(*) FROM zone_profiles WHERE source_run_id = ?", (r,)
            ).fetchone()[0]
        )
        zn += int(
            conn.execute(
                "SELECT COUNT(*) FROM runs WHERE zone_model_source_run_id = ?", (r,)
            ).fetchone()[0]
        )
    return {
        "run_segments_delete": seg,
        "zone_profiles_delete": zp,
        "runs_zone_model_null_updates": zn,
        "runs_delete": len(remove_ids),
    }


def _delete_run_and_dependents_rowcounts(
    conn: sqlite3.Connection, run_id: str
) -> tuple[int, int, int, int]:
    """Same deletes as :func:`_delete_run_and_dependents`; returns affected row counts."""
    rid = str(run_id)
    cur_u = conn.execute(
        "UPDATE runs SET zone_model_source_run_id = NULL WHERE zone_model_source_run_id = ?",
        (rid,),
    )
    zn = cur_u.rowcount if cur_u.rowcount is not None and cur_u.rowcount >= 0 else 0
    cur_s = conn.execute("DELETE FROM run_segments WHERE run_id = ?", (rid,))
    ns = cur_s.rowcount if cur_s.rowcount is not None and cur_s.rowcount >= 0 else 0
    cur_z = conn.execute("DELETE FROM zone_profiles WHERE source_run_id = ?", (rid,))
    nz = cur_z.rowcount if cur_z.rowcount is not None and cur_z.rowcount >= 0 else 0
    cur_r = conn.execute("DELETE FROM runs WHERE run_id = ?", (rid,))
    nr = cur_r.rowcount if cur_r.rowcount is not None and cur_r.rowcount >= 0 else 0
    return (zn, ns, nz, nr)


def _delete_run_and_dependents(conn: sqlite3.Connection, run_id: str) -> None:
    """Remove ``run_id`` from ``run_segments``, ``zone_profiles``, and ``runs`` safely."""
    _delete_run_and_dependents_rowcounts(conn, run_id)


def _fit_activity_key_survivor_remediation(
    run_id_to_stripped_key: dict[str, str],
    *,
    keep_run_id: str,
    remove_run_ids: list[str],
) -> dict[str, Any] | None:
    """If survivor has no usable ``fit_activity_key``, plan promote or conflict from removed rows."""
    keep = str(keep_run_id)
    if run_id_to_stripped_key.get(keep, ""):
        return None
    keyed: list[tuple[str, str]] = []
    for rid in remove_run_ids:
        r = str(rid)
        k = run_id_to_stripped_key.get(r, "")
        if k:
            keyed.append((r, k))
    if not keyed:
        return None
    distinct = sorted({k for _, k in keyed})
    if len(distinct) == 1:
        only = distinct[0]
        source = min(r for r, k in keyed if k == only)
        return {"kind": "promote", "key": only, "source_removed_run": source}
    return {"kind": "conflict", "candidate_keys": distinct}


def dedupe_fit_import_duplicates(
    conn: sqlite3.Connection,
    *,
    apply: bool,
    dedupe_debug: bool = False,
) -> dict[str, Any]:
    """Detect duplicate Garmin FIT activities; optionally remove extras (one survivor per group).

    **Primary:** same non-empty ``runs.fit_activity_key``.
    **Fallback (``garmin_fit`` only, not consumed by primary duplicate groups):** strict
    work-only signature — same fields as above — for any row not in ``seen_run_ids`` after
    primary grouping, including rows with a non-null ``fit_activity_key`` when that key is
    unique (so mixed keyed / legacy-null pairs can still match on work signature).

    Survivor: lexicographically first ``(run_date, run_id)`` in each group (oldest / most
    canonical deterministic choice). Dependent rows in ``run_segments`` and ``zone_profiles``
    (``source_run_id``) are removed;     ``runs.zone_model_source_run_id`` pointers are nulled.
    """
    ensure_run_segments_table(conn)
    ensure_zone_profiles_table(conn)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT run_id, run_date, source_type, fit_activity_key, source_file
        FROM runs
        ORDER BY run_date ASC, run_id ASC
        """
    ).fetchall()

    keyed: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        fk = r["fit_activity_key"]
        if fk is not None and str(fk).strip():
            keyed.setdefault(str(fk).strip(), []).append(dict(r))

    groups: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()

    for k, members in keyed.items():
        if len(members) < 2:
            continue
        members_sorted = sorted(
            members, key=lambda m: (str(m.get("run_date") or ""), str(m.get("run_id") or ""))
        )
        keep = members_sorted[0]["run_id"]
        remove = [m["run_id"] for m in members_sorted[1:]]
        groups.append(
            {
                "kind": "fit_activity_key",
                "key": f"fit_activity_key:{k}",
                "keep": keep,
                "remove": remove,
            }
        )
        seen_run_ids.update(m["run_id"] for m in members_sorted)

    wsig_map: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for r in rows:
        rid = str(r["run_id"])
        if rid in seen_run_ids:
            continue
        if str(r["source_type"] or "") != "garmin_fit":
            continue
        sig = _work_fallback_duplicate_signature(conn, rid)
        if sig is None:
            continue
        wsig_map.setdefault(sig, []).append(dict(r))

    for sig, members in wsig_map.items():
        if len(members) < 2:
            continue
        members_sorted = sorted(
            members, key=lambda m: (str(m.get("run_date") or ""), str(m.get("run_id") or ""))
        )
        keep = members_sorted[0]["run_id"]
        remove = [m["run_id"] for m in members_sorted[1:]]
        key_s = (
            f"work_sig:date={sig[1]!s}|label={sig[2]!s}|t_s={sig[3]!s}"
            f"|pwr={sig[4]!s}|pace={sig[5]!s}"
        )
        groups.append(
            {
                "kind": "work_signature",
                "key": key_s,
                "keep": keep,
                "remove": remove,
            }
        )

    run_id_to_stripped_key: dict[str, str] = {}
    for r in rows:
        rid = str(r["run_id"])
        fk = r["fit_activity_key"]
        run_id_to_stripped_key[rid] = "" if fk is None else str(fk).strip()

    for g in groups:
        g["fit_activity_key_survivor"] = _fit_activity_key_survivor_remediation(
            run_id_to_stripped_key,
            keep_run_id=str(g["keep"]),
            remove_run_ids=[str(x) for x in g["remove"]],
        )

    for g in groups:
        g["impact_projected"] = _dedupe_projected_impact_for_remove_ids(
            conn, [str(x) for x in g["remove"]]
        )

    impact_totals: dict[str, int] = {
        "run_segments_delete": 0,
        "zone_profiles_delete": 0,
        "runs_zone_model_null_updates": 0,
        "runs_delete": 0,
    }
    for g in groups:
        imp = g["impact_projected"]
        impact_totals["run_segments_delete"] += imp["run_segments_delete"]
        impact_totals["zone_profiles_delete"] += imp["zone_profiles_delete"]
        impact_totals["runs_zone_model_null_updates"] += imp["runs_zone_model_null_updates"]
        impact_totals["runs_delete"] += imp["runs_delete"]

    if not apply and dedupe_debug:
        key_nonnull = 0
        key_null = 0
        for r in rows:
            fk = r["fit_activity_key"]
            if fk is not None and str(fk).strip():
                key_nonnull += 1
            else:
                key_null += 1

        fallback_prefilter = 0
        fallback_sig_ok = 0
        fallback_skipped_sig = 0
        for r in rows:
            rid = str(r["run_id"])
            if rid in seen_run_ids:
                continue
            if str(r["source_type"] or "") != "garmin_fit":
                continue
            fallback_prefilter += 1
            sig_d, reason = _work_fallback_skip_reason(conn, rid)
            if sig_d is None:
                fallback_skipped_sig += 1
            else:
                fallback_sig_ok += 1

        print(
            "dedupe_diag_summary: "
            f"fit_activity_key_nonnull={key_nonnull} "
            f"fit_activity_key_null_or_empty={key_null} "
            f"fallback_prefilter_garmin_not_primary_dup={fallback_prefilter} "
            f"fallback_signature_ok={fallback_sig_ok} "
            f"fallback_skipped_no_signature={fallback_skipped_sig}",
            flush=True,
        )

        diag_date = "2026-04-08"
        for r in rows:
            if str(r["run_date"] or "") != diag_date:
                continue
            rid = str(r["run_id"])
            fk = r["fit_activity_key"]
            fk_s = "" if fk is None else str(fk).strip()
            st = str(r["source_type"] or "")
            has_pk = bool(fk_s)
            k_for_row = fk_s if has_pk else ""
            primary_pool = len(keyed.get(k_for_row, [])) if has_pk else 0
            primary_would_group = has_pk and primary_pool >= 2
            in_seen = rid in seen_run_ids
            fallback_prefilter_row = not in_seen and st == "garmin_fit"
            sig_d, skip_reason = _work_fallback_skip_reason(conn, rid)
            fallback_sig_row = sig_d is not None
            print(f"dedupe_diag: run_date={diag_date} run_id={rid}", flush=True)
            print(f"  source_type={st!r} fit_activity_key={fk_s!r}", flush=True)
            print(
                f"  primary_key_nonempty={has_pk} "
                f"primary_same_key_run_count={primary_pool} "
                f"primary_would_duplicate_group={primary_would_group}",
                flush=True,
            )
            print(
                f"  in_seen_run_ids_after_primary={in_seen} "
                f"(excluded from fallback when True)",
                flush=True,
            )
            print(
                f"  fallback_prefilter_garmin_not_primary_dup={fallback_prefilter_row}",
                flush=True,
            )
            print(
                f"  fallback_enters_wsig_map={fallback_prefilter_row and sig_d is not None} "
                f"(prefilter and signature ok)",
                flush=True,
            )
            if sig_d is not None:
                print(f"  fallback_signature={sig_d!r}", flush=True)
            else:
                print(
                    f"  fallback_signature=None skipped_reason={skip_reason!r}",
                    flush=True,
                )

    total_remove = sum(len(g["remove"]) for g in groups)
    apply_impact: dict[str, int] | None = None
    if apply and total_remove > 0:
        apply_impact = {
            "run_segments_deleted": 0,
            "zone_profiles_deleted": 0,
            "runs_zone_model_nulled": 0,
            "runs_deleted": 0,
        }
        conn.execute("BEGIN")
        try:
            for g in groups:
                fe = g.get("fit_activity_key_survivor")
                if fe and fe.get("kind") == "promote":
                    conn.execute(
                        "UPDATE runs SET fit_activity_key = ? WHERE run_id = ?",
                        (fe["key"], str(g["keep"])),
                    )
                for rid in g["remove"]:
                    zn, ns, nz, nr = _delete_run_and_dependents_rowcounts(conn, str(rid))
                    apply_impact["runs_zone_model_nulled"] += zn
                    apply_impact["run_segments_deleted"] += ns
                    apply_impact["zone_profiles_deleted"] += nz
                    apply_impact["runs_deleted"] += nr
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    return {
        "groups": groups,
        "duplicate_group_count": len(groups),
        "rows_to_remove": total_remove,
        "apply": apply,
        "impact_projected_totals": impact_totals,
        "impact_apply_totals": apply_impact,
    }
