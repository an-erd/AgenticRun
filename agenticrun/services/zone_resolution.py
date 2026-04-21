from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import timezone
from typing import Any

import sqlite3

from agenticrun.core.fit_power_zones import (
    FIT_HR_REFERENCE_BPM,
    FIT_THRESHOLD_POWER_W,
    fit_hr_zone_seconds_from_records,
    fit_power_zone_seconds_from_records,
)
from agenticrun.core.models import RunRecord, RunState
from agenticrun.services.zone_profiles import (
    fetch_latest_zone_profile_at_or_before,
    insert_zone_profile_if_new,
)


def _agenticrun_debug() -> bool:
    return os.getenv("AGENTICRUN_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _fit_zone_seconds_source_label(gsnap: dict[str, Any], channel: str, used_garmin_array: bool) -> str:
    if not used_garmin_array:
        return "fit_record_recalc"
    zxs = gsnap.get("zone_extract_sources")
    if not isinstance(zxs, dict):
        return "fit_garmin_session_or_lap"
    key = "hr_zone_time_sec" if channel == "hr" else "power_zone_time_sec"
    src = str(zxs.get(key) or "")
    if "decoded_unknown_216" in src:
        return "fit_garmin_mesg216"
    return "fit_garmin_session_or_lap"


def run_timestamp_iso_for_lookup(run: RunRecord) -> str:
    session: dict[str, Any] = {}
    if isinstance(run.raw_summary, dict):
        session = run.raw_summary.get("session")  # type: ignore[assignment]
        if not isinstance(session, dict):
            session = {}
    st_raw = session.get("start_time")
    if hasattr(st_raw, "isoformat"):
        ts = st_raw
        if getattr(ts, "tzinfo", None) is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.isoformat()
    rd = (run.run_date or "").strip()
    if rd:
        return f"{rd}T12:00:00+00:00"
    return "1970-01-01T00:00:00+00:00"


def _hr_reference_from_snapshot(s: dict[str, Any]) -> float | None:
    thr = s.get("threshold_heart_rate")
    if thr is not None:
        try:
            t = float(thr)
            if t > 0:
                return t
        except (TypeError, ValueError):
            pass
    mx = s.get("max_heart_rate")
    if mx is not None:
        try:
            m = float(mx)
            if m > 0:
                return m
        except (TypeError, ValueError):
            pass
    return None


def _model_params_from_snapshot(s: dict[str, Any]) -> dict[str, Any]:
    hr_b = s.get("hr_zone_boundaries") or []
    pw_b = s.get("power_zone_boundaries") or []
    if not isinstance(hr_b, list):
        hr_b = []
    if not isinstance(pw_b, list):
        pw_b = []
    hr_bounds_f = [float(x) for x in hr_b if _is_pos_num(x)][:5]
    pw_bounds_f = [float(x) for x in pw_b if _is_pos_num(x)][:5]
    ftp = s.get("functional_threshold_power")
    try:
        ftp_f = float(ftp) if ftp is not None else None
    except (TypeError, ValueError):
        ftp_f = None
    if ftp_f is not None and ftp_f <= 0:
        ftp_f = None
    hr_ref = _hr_reference_from_snapshot(s)
    return {
        "hr_high_bounds": hr_bounds_f if len(hr_bounds_f) >= 5 else None,
        "power_high_bounds": pw_bounds_f if len(pw_bounds_f) >= 5 else None,
        "ftp_w": ftp_f,
        "hr_reference_bpm": hr_ref,
    }


def _is_pos_num(x: Any) -> bool:
    try:
        return float(x) > 0
    except (TypeError, ValueError):
        return False


def _internal_model_params() -> dict[str, Any]:
    return {
        "hr_high_bounds": None,
        "power_high_bounds": None,
        "ftp_w": FIT_THRESHOLD_POWER_W,
        "hr_reference_bpm": FIT_HR_REFERENCE_BPM,
    }


@dataclass
class ZoneResolution:
    zone_model_source: str
    effective_from_iso: str
    profile_source_run_id: str | None
    model_params: dict[str, Any]
    garmin_snapshot: dict[str, Any] | None
    persist_snapshot: dict[str, Any] | None
    debug_fallback_reason: str | None = None
    historical_profile_row_id: int | None = None
    prior_zone_model_source: str | None = None


def resolve_zone_resolution(conn: sqlite3.Connection, run: RunRecord) -> ZoneResolution:
    at_iso = run_timestamp_iso_for_lookup(run)
    raw = run.raw_summary if isinstance(run.raw_summary, dict) else {}
    extract = raw.get("fit_garmin_zone_extract")
    has_extract = isinstance(extract, dict) and _snapshot_has_zone_data(extract)

    if has_extract:
        params = _model_params_from_snapshot(extract)
        return ZoneResolution(
            zone_model_source="fit_profile_current_run",
            effective_from_iso=at_iso,
            profile_source_run_id=run.run_id,
            model_params=params,
            garmin_snapshot=extract,
            persist_snapshot=extract,
            debug_fallback_reason=None,
            historical_profile_row_id=None,
        )

    row = fetch_latest_zone_profile_at_or_before(conn, at_iso)
    if row:
        snap = row["snapshot"]
        params = _model_params_from_snapshot(snap)
        return ZoneResolution(
            zone_model_source="fit_profile_historical",
            effective_from_iso=str(row["effective_from"]),
            profile_source_run_id=row["source_run_id"],
            model_params=params,
            garmin_snapshot=snap,
            persist_snapshot=None,
            debug_fallback_reason=None,
            historical_profile_row_id=row.get("row_id"),
        )

    return ZoneResolution(
        zone_model_source="internal_fallback",
        effective_from_iso=at_iso,
        profile_source_run_id=None,
        model_params=_internal_model_params(),
        garmin_snapshot=None,
        persist_snapshot=None,
        debug_fallback_reason=(
            "No Garmin zone snapshot in this run's FIT-derived summary and no row in zone_profiles "
            f"with effective_from <= lookup time ({at_iso})."
        ),
        historical_profile_row_id=None,
    )


def _seq_has_positive_seconds(seq: Any) -> bool:
    if not isinstance(seq, list):
        return False
    for x in seq:
        try:
            if float(x) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _snapshot_has_zone_data(s: dict[str, Any]) -> bool:
    hr_b = s.get("hr_zone_boundaries") or []
    pw_b = s.get("power_zone_boundaries") or []
    if isinstance(hr_b, list) and len(hr_b) >= 1:
        return True
    if isinstance(pw_b, list) and len(pw_b) >= 1:
        return True
    if _seq_has_positive_seconds(s.get("hr_zone_time_sec")):
        return True
    if _seq_has_positive_seconds(s.get("power_zone_time_sec")):
        return True
    for k in ("threshold_heart_rate", "functional_threshold_power", "max_heart_rate"):
        if s.get(k) is not None:
            try:
                if float(s[k]) > 0:
                    return True
            except (TypeError, ValueError):
                continue
    return False


def _garmin_five_zone_seconds_tuple(
    seq: Any,
) -> tuple[float | None, float | None, float | None, float | None, float | None] | None:
    if not isinstance(seq, list):
        return None
    vals: list[float] = []
    for x in seq[:5]:
        try:
            vals.append(float(x))
        except (TypeError, ValueError):
            return None
    while len(vals) < 5:
        vals.append(0.0)
    if not any(v > 0 for v in vals):
        return None
    return (vals[0], vals[1], vals[2], vals[3], vals[4])


def _has_streams(bundle: tuple[list[object], list[float | None], list[float | None]] | None) -> tuple[bool, bool]:
    if not bundle:
        return False, False
    _, powers, hrs = bundle
    hp = any(p is not None for p in powers)
    hh = any(h is not None for h in hrs)
    return hp, hh


def finalize_zone_model_source(res: ZoneResolution, bundle: tuple | None) -> ZoneResolution:
    hp, hh = _has_streams(bundle)
    if not hp and not hh and res.zone_model_source == "internal_fallback":
        prior = (res.debug_fallback_reason or "").strip()
        extra = "Zone time-in-zone not computed: no power or heart-rate samples in FIT record stream."
        merged = f"{prior} {extra}".strip() if prior else extra
        return ZoneResolution(
            zone_model_source="unavailable",
            effective_from_iso=res.effective_from_iso,
            profile_source_run_id=res.profile_source_run_id,
            model_params=res.model_params,
            garmin_snapshot=res.garmin_snapshot,
            persist_snapshot=res.persist_snapshot,
            debug_fallback_reason=merged,
            historical_profile_row_id=res.historical_profile_row_id,
            prior_zone_model_source="internal_fallback",
        )
    return res


def apply_zone_resolution_to_state(conn: sqlite3.Connection, state: RunState) -> ZoneResolution:
    run = state.run_record
    if not run:
        return ZoneResolution(
            zone_model_source="unavailable",
            effective_from_iso="",
            profile_source_run_id=None,
            model_params=_internal_model_params(),
            garmin_snapshot=None,
            persist_snapshot=None,
            debug_fallback_reason="No run_record on state.",
            historical_profile_row_id=None,
        )

    res = resolve_zone_resolution(conn, run)
    res = finalize_zone_model_source(res, state.fit_stream_bundle)

    mp = res.model_params
    raw = run.raw_summary if isinstance(run.raw_summary, dict) else {}
    fm = raw.get("fit_session_metrics")
    if not isinstance(fm, dict):
        fm = {}

    gsnap = res.garmin_snapshot or {}
    zs = gsnap.get("zone_extract_sources") if isinstance(gsnap.get("zone_extract_sources"), dict) else {}
    if zs:
        fm["hr_zone_boundaries_fit_source"] = zs.get("hr_zone_boundaries")
        fm["power_zone_boundaries_fit_source"] = zs.get("power_zone_boundaries")
        fm["threshold_heart_rate_fit_source"] = zs.get("threshold_heart_rate")
        fm["functional_threshold_power_fit_source"] = zs.get("functional_threshold_power")
        fm["max_heart_rate_fit_source"] = zs.get("max_heart_rate")
        fm["resting_heart_rate_fit_source"] = zs.get("resting_heart_rate")
        fm["hr_calc_type_fit_source"] = zs.get("hr_calc_type")
        fm["pwr_calc_type_fit_source"] = zs.get("pwr_calc_type")

    bundle = state.fit_stream_bundle
    if bundle:
        use_garmin_session_times = res.zone_model_source == "fit_profile_current_run"
        g_pw = (
            _garmin_five_zone_seconds_tuple(gsnap.get("power_zone_time_sec"))
            if use_garmin_session_times
            else None
        )
        g_hr = (
            _garmin_five_zone_seconds_tuple(gsnap.get("hr_zone_time_sec"))
            if use_garmin_session_times
            else None
        )
        ts, powers, hrs = bundle
        pz = fit_power_zone_seconds_from_records(
            ts,
            powers,
            ftp_w=mp.get("ftp_w"),
            power_high_bounds=mp.get("power_high_bounds"),
        )
        hz = fit_hr_zone_seconds_from_records(
            ts,
            hrs,
            hr_reference_bpm=mp.get("hr_reference_bpm"),
            hr_high_bounds=mp.get("hr_high_bounds"),
        )
        if g_pw is not None:
            pz = g_pw
            fm["power_zone_seconds_source"] = _fit_zone_seconds_source_label(gsnap, "power", True)
        else:
            fm["power_zone_seconds_source"] = "fit_record_recalc"
        if g_hr is not None:
            hz = g_hr
            fm["hr_zone_seconds_source"] = _fit_zone_seconds_source_label(gsnap, "hr", True)
        else:
            fm["hr_zone_seconds_source"] = "fit_record_recalc"
        fm["power_zone_z1_sec"], fm["power_zone_z2_sec"], fm["power_zone_z3_sec"], fm["power_zone_z4_sec"], fm["power_zone_z5_sec"] = pz
        fm["hr_zone_z1_sec"], fm["hr_zone_z2_sec"], fm["hr_zone_z3_sec"], fm["hr_zone_z4_sec"], fm["hr_zone_z5_sec"] = hz

    fm["zone_model_source"] = res.zone_model_source
    fm["zone_model_effective_from"] = res.effective_from_iso
    fm["zone_model_source_run_id"] = res.profile_source_run_id
    fm["resolved_functional_threshold_power"] = mp.get("ftp_w")
    fm["resolved_threshold_heart_rate"] = mp.get("hr_reference_bpm")
    if res.garmin_snapshot:
        fm["resolved_max_heart_rate"] = res.garmin_snapshot.get("max_heart_rate")

    raw["fit_session_metrics"] = fm
    run.raw_summary = raw

    profile_store_attempted = bool(res.persist_snapshot and run.source_type == "garmin_fit")
    profile_stored: bool | None = None
    if profile_store_attempted:
        eff = run_timestamp_iso_for_lookup(run)
        profile_stored = insert_zone_profile_if_new(
            conn,
            effective_from=eff,
            source_run_id=run.run_id,
            source_type=run.source_type,
            snapshot=res.persist_snapshot,
            zone_source="fit_garmin",
        )
    elif _agenticrun_debug():
        if not res.persist_snapshot:
            print(
                "zone_profile_persist: skipped (no persist_snapshot; resolution used historical DB row or internal fallback)",
                flush=True,
            )
        else:
            print(
                f"zone_profile_persist: skipped (source_type={run.source_type!r}, not garmin_fit)",
                flush=True,
            )

    if _agenticrun_debug():
        mp = res.model_params
        gs = res.garmin_snapshot or {}
        print("--- zone analysis (resolved for this run) ---", flush=True)
        print(f"  active_zone_source: {res.zone_model_source}", flush=True)
        print(f"  effective_timestamp: {res.effective_from_iso}", flush=True)
        print(f"  profile_source_run_id: {res.profile_source_run_id or '-'}", flush=True)
        if res.historical_profile_row_id is not None:
            print(f"  historical_zone_profiles_row_id: {res.historical_profile_row_id}", flush=True)
        print(f"  hr_boundaries_used: {json.dumps(mp.get('hr_high_bounds'))}", flush=True)
        print(f"  power_boundaries_used: {json.dumps(mp.get('power_high_bounds'))}", flush=True)
        print(f"  threshold_hr_used: {mp.get('hr_reference_bpm')}", flush=True)
        print(f"  ftp_used: {mp.get('ftp_w')}", flush=True)
        if gs.get("max_heart_rate") is not None:
            print(f"  max_hr_from_profile: {gs.get('max_heart_rate')}", flush=True)
        if gs.get("resting_heart_rate") is not None:
            print(f"  resting_hr_from_profile: {gs.get('resting_heart_rate')}", flush=True)
        zxs = gs.get("zone_extract_sources")
        if isinstance(zxs, dict) and zxs:
            print(f"  garmin_zone_extract_sources: {json.dumps(zxs)}", flush=True)
        if gs.get("hr_zone_boundaries"):
            print(f"  garmin_hr_zone_boundaries_high_bpm: {json.dumps(gs.get('hr_zone_boundaries'))}", flush=True)
        if gs.get("hr_zone_boundaries_high_bpm"):
            print(
                f"  garmin_hr_zone_boundaries_high_bpm (unknown_216): "
                f"{json.dumps(gs.get('hr_zone_boundaries_high_bpm'))}",
                flush=True,
            )
        if gs.get("power_zone_boundaries"):
            print(f"  garmin_power_zone_boundaries_high_w: {json.dumps(gs.get('power_zone_boundaries'))}", flush=True)
        if gs.get("power_zone_boundaries_high_w"):
            print(
                f"  garmin_power_zone_boundaries_high_w (unknown_216): "
                f"{json.dumps(gs.get('power_zone_boundaries_high_w'))}",
                flush=True,
            )
        if gs.get("garmin_hr_zones_found") is not None:
            print(f"  garmin_hr_zones_found: {gs.get('garmin_hr_zones_found')}", flush=True)
        if gs.get("garmin_power_zones_found") is not None:
            print(f"  garmin_power_zones_found: {gs.get('garmin_power_zones_found')}", flush=True)
        if res.zone_model_source == "internal_fallback":
            print("  internal_fallback: yes (Garmin profile not used for model parameters)", flush=True)
        if res.prior_zone_model_source:
            print(
                f"  prior_zone_model_source (before stream check): {res.prior_zone_model_source}; "
                f"final active_zone_source is {res.zone_model_source!r}",
                flush=True,
            )
        if res.debug_fallback_reason:
            print(f"  resolution_note: {res.debug_fallback_reason}", flush=True)
        if fm.get("hr_zone_seconds_source"):
            print(f"  hr_zone_seconds_source: {fm.get('hr_zone_seconds_source')}", flush=True)
        if fm.get("power_zone_seconds_source"):
            print(f"  power_zone_seconds_source: {fm.get('power_zone_seconds_source')}", flush=True)
        for k in (
            "hr_zone_boundaries_fit_source",
            "power_zone_boundaries_fit_source",
            "threshold_heart_rate_fit_source",
            "functional_threshold_power_fit_source",
            "max_heart_rate_fit_source",
            "resting_heart_rate_fit_source",
        ):
            if fm.get(k):
                print(f"  {k}: {fm.get(k)}", flush=True)
        if gs.get("hr_zone_time_sec") is not None:
            _hr_note = (
                ""
                if res.zone_model_source == "fit_profile_current_run"
                else " [profile snapshot only; zone seconds for this run use record recalc]"
            )
            print(f"  garmin_hr_zone_time_sec: {json.dumps(gs.get('hr_zone_time_sec'))}{_hr_note}", flush=True)
        if gs.get("power_zone_time_sec") is not None:
            _pw_note = (
                ""
                if res.zone_model_source == "fit_profile_current_run"
                else " [profile snapshot only; zone seconds for this run use record recalc]"
            )
            print(f"  garmin_power_zone_time_sec: {json.dumps(gs.get('power_zone_time_sec'))}{_pw_note}", flush=True)
        if profile_store_attempted:
            print(f"  zone_profile_store_attempted: yes, stored_new_row: {profile_stored}", flush=True)

    state.fit_stream_bundle = None
    return res


def zone_debug_line(res: ZoneResolution) -> str:
    mp = res.model_params
    parts = [
        f"src={res.zone_model_source}",
        f"eff={res.effective_from_iso}",
        f"prof_run={res.profile_source_run_id or '-'}",
        f"hr_b={json.dumps(mp.get('hr_high_bounds'))}",
        f"pw_b={json.dumps(mp.get('power_high_bounds'))}",
        f"thr_hr={mp.get('hr_reference_bpm')}",
        f"ftp={mp.get('ftp_w')}",
    ]
    return "zone_profile: " + " ".join(parts)
