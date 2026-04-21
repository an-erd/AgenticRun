from __future__ import annotations

import struct
from typing import Any

from fitparse import FitFile

from agenticrun.core.fit_garmin_mesg216 import decode_garmin_mesg216_zone_arrays, run_mesg216_debug_if_enabled

# Garmin / FIT developer-data names seen on session/lap for zone high edges (forums / Connect exports).
_HR_BOUNDARY_FIELD_NAMES = frozenset(
    {
        "hr_zone_high_boundary",
        "heart_rate_zone_high_boundary",
    }
)
_POWER_BOUNDARY_FIELD_NAMES = frozenset(
    {
        "power_zone_high_boundary",
        "pwr_zone_high_boundary",
    }
)

# FIT global session=18, lap=19 — field def_nums for time-in-zone (per message type in SDK profile).
_TIME_IN_ZONE_DEFS_BY_MESG: dict[str, tuple[int, int]] = {
    "session": (65, 68),  # time_in_hr_zone, time_in_power_zone
    "lap": (57, 60),
    "segment_lap": (49, 52),
}


def _calc_type_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _msg_field_dict(msg: Any) -> dict[str, Any]:
    return {field.name: field.value for field in msg if field.name}


def _field_value_by_def_num(msg: Any, def_num: int) -> Any:
    for fd in msg.fields:
        if fd.def_num == def_num:
            return fd.value
    return None


def _norm_key(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _fuzzy_time_in_hr_raw(fm: dict[str, Any]) -> Any:
    for k, v in fm.items():
        if v is None:
            continue
        kn = _norm_key(k)
        if "time_in" not in kn or "zone" not in kn:
            continue
        if "power" in kn or "pwr" in kn or "cadence" in kn or "speed" in kn:
            continue
        if "hr" in kn or "heart" in kn:
            return v
    return None


def _fuzzy_time_in_power_raw(fm: dict[str, Any]) -> Any:
    for k, v in fm.items():
        if v is None:
            continue
        kn = _norm_key(k)
        if "time_in" not in kn or "zone" not in kn:
            continue
        if "power" in kn or "pwr" in kn:
            return v
    return None


def _read_hr_zone_seconds_from_message(msg: Any) -> tuple[list[float] | None, str | None]:
    fm = _msg_field_dict(msg)
    mesg_name = getattr(msg, "name", "") or ""
    for key, label in (
        ("time_in_hr_zone", "time_in_hr_zone"),
        ("time_in_heart_rate_zone", "time_in_heart_rate_zone"),
    ):
        seq = _zone_seconds_array(fm.get(key))
        if seq and _total_zone_time(seq) > 0:
            return seq, f"{mesg_name}:{label}"
    defs = _TIME_IN_ZONE_DEFS_BY_MESG.get(mesg_name)
    if defs:
        hr_def, _pw_def = defs
        seq = _zone_seconds_array(_field_value_by_def_num(msg, hr_def))
        if seq and _total_zone_time(seq) > 0:
            return seq, f"{mesg_name}:def_num_{hr_def}"
    raw_f = _fuzzy_time_in_hr_raw(fm)
    seq = _zone_seconds_array(raw_f)
    if seq and _total_zone_time(seq) > 0:
        return seq, f"{mesg_name}:fuzzy_time_in_hr_zone"
    return None, None


def _read_power_zone_seconds_from_message(msg: Any) -> tuple[list[float] | None, str | None]:
    fm = _msg_field_dict(msg)
    mesg_name = getattr(msg, "name", "") or ""
    seq = _zone_seconds_array(fm.get("time_in_power_zone"))
    if seq and _total_zone_time(seq) > 0:
        return seq, f"{mesg_name}:time_in_power_zone"
    defs = _TIME_IN_ZONE_DEFS_BY_MESG.get(mesg_name)
    if defs:
        _hr_def, pw_def = defs
        seq = _zone_seconds_array(_field_value_by_def_num(msg, pw_def))
        if seq and _total_zone_time(seq) > 0:
            return seq, f"{mesg_name}:def_num_{pw_def}"
    raw_f = _fuzzy_time_in_power_raw(fm)
    seq = _zone_seconds_array(raw_f)
    if seq and _total_zone_time(seq) > 0:
        return seq, f"{mesg_name}:fuzzy_time_in_power_zone"
    return None, None


def _to_float_seq(val: Any, *, max_n: int = 7) -> list[float] | None:
    """Coerce FIT array/tuple/bytes/single number to a list of floats (>=0)."""
    if val is None:
        return None
    if isinstance(val, (bytes, bytearray)):
        if len(val) < 4 or len(val) % 4 != 0:
            return None
        out: list[float] = []
        for i in range(0, len(val), 4):
            raw = struct.unpack_from("<I", val, i)[0]
            out.append(float(raw) / 1000.0)
        return out[:max_n] if out else None
    if isinstance(val, (int, float)):
        f = float(val)
        if f < 0:
            return None
        return [f]
    if isinstance(val, (tuple, list)):
        out = []
        for x in val[:max_n]:
            if x is None:
                continue
            try:
                f = float(x)
            except (TypeError, ValueError):
                continue
            if f < 0:
                continue
            out.append(f)
        return out if out else None
    return None


def _zone_seconds_array(val: Any) -> list[float] | None:
    """Session/lap time_in_* fields: seconds per zone (FIT scale already applied by fitparse when decoded)."""
    return _to_float_seq(val, max_n=7)


def _zone_boundary_array(val: Any) -> list[float] | None:
    """HR BPM or power watts high edges; allow uint8/uint16 raw values."""
    return _to_float_seq(val, max_n=7)


def _total_zone_time(seq: list[float] | None) -> float:
    if not seq:
        return 0.0
    return float(sum(seq))


def _pick_richer_boundaries(
    from_messages: list[float],
    from_scan: list[float],
    msg_source_label: str,
) -> tuple[list[float], str | None]:
    """Prefer >=5 zone edges; else longer non-empty list."""

    def score(b: list[float]) -> tuple[int, int]:
        return (1 if len(b) >= 5 else 0, len(b))

    if not from_scan:
        return from_messages, (msg_source_label if from_messages else None)
    if not from_messages:
        return from_scan, "session/lap developer boundary fields"
    if score(from_scan) > score(from_messages):
        return from_scan, "session/lap developer boundary fields"
    return from_messages, msg_source_label


def _merge_time_arrays_from_sessions_and_laps(fit: FitFile) -> tuple[
    list[float] | None,
    list[float] | None,
    dict[str, str | None],
]:
    """Read Garmin session/lap time_in_hr_zone and time_in_power_zone (and developer aliases)."""
    sources: dict[str, str | None] = {"hr_time": None, "pw_time": None}

    session_msgs = list(fit.get_messages("session"))
    best_hr: list[float] | None = None
    best_pw: list[float] | None = None
    best_hr_tot = -1.0
    best_hr_src: str | None = None
    best_pw_src: str | None = None

    for msg in session_msgs:
        hr, hr_src = _read_hr_zone_seconds_from_message(msg)
        pw, pw_src = _read_power_zone_seconds_from_message(msg)
        if hr is None:
            continue
        tot = _total_zone_time(hr)
        if tot > best_hr_tot:
            best_hr_tot = tot
            best_hr = hr
            best_pw = pw
            best_hr_src = hr_src
            if pw and _total_zone_time(pw) > 0:
                best_pw_src = pw_src

    if best_hr_src:
        sources["hr_time"] = best_hr_src
    if best_pw_src:
        sources["pw_time"] = best_pw_src

    if best_pw is None or _total_zone_time(best_pw) <= 0:
        for msg in session_msgs:
            pw, pw_src = _read_power_zone_seconds_from_message(msg)
            if pw and _total_zone_time(pw) > 0:
                best_pw = pw
                sources["pw_time"] = pw_src or "session:time_in_power_zone"
                break

    if not best_hr or best_hr_tot <= 0:
        lap_hr_rows: list[list[float]] = []
        lap_pw_rows: list[list[float]] = []
        lap_hr_srcs: list[str] = []
        lap_pw_srcs: list[str] = []
        for msg in list(fit.get_messages("lap")) + list(fit.get_messages("segment_lap")):
            hr, hr_src = _read_hr_zone_seconds_from_message(msg)
            pw, pw_src = _read_power_zone_seconds_from_message(msg)
            if hr and _total_zone_time(hr) > 0:
                lap_hr_rows.append(hr)
                if hr_src:
                    lap_hr_srcs.append(hr_src)
            if pw and _total_zone_time(pw) > 0:
                lap_pw_rows.append(pw)
                if pw_src:
                    lap_pw_srcs.append(pw_src)
        if lap_hr_rows:
            width = max(len(r) for r in lap_hr_rows)
            acc = [0.0] * width
            for r in lap_hr_rows:
                for i, v in enumerate(r):
                    if i < width:
                        acc[i] += v
            best_hr = acc
            sources["hr_time"] = (
                "lap+segment_lap:sum(" + ";".join(sorted(set(lap_hr_srcs)) or ["time_in_hr_zone"]) + ")"
            )
        if lap_pw_rows and (not best_pw or _total_zone_time(best_pw) <= 0):
            width = max(len(r) for r in lap_pw_rows)
            acc = [0.0] * width
            for r in lap_pw_rows:
                for i, v in enumerate(r):
                    if i < width:
                        acc[i] += v
            best_pw = acc
            sources["pw_time"] = (
                "lap+segment_lap:sum(" + ";".join(sorted(set(lap_pw_srcs)) or ["time_in_power_zone"]) + ")"
            )

    need_hr = not best_hr or _total_zone_time(best_hr) <= 0
    need_pw = best_pw is None or _total_zone_time(best_pw) <= 0
    if need_hr or need_pw:
        scan_hr, scan_pw, scan_src = _scan_all_messages_for_zone_times(fit, need_hr=need_hr, need_pw=need_pw)
        if need_hr and scan_hr and _total_zone_time(scan_hr) > 0:
            best_hr = scan_hr
            if scan_src.get("hr_time"):
                sources["hr_time"] = scan_src["hr_time"]
        if need_pw and scan_pw and _total_zone_time(scan_pw) > 0:
            best_pw = scan_pw
            if scan_src.get("pw_time"):
                sources["pw_time"] = scan_src["pw_time"]

    return best_hr, best_pw, sources


def _scan_all_messages_for_zone_times(
    fit: FitFile, *, need_hr: bool, need_pw: bool
) -> tuple[list[float] | None, list[float] | None, dict[str, str | None]]:
    """Last resort: any data message (e.g. nonstandard layout) with time-in-zone fields."""
    src: dict[str, str | None] = {"hr_time": None, "pw_time": None}
    if not need_hr and not need_pw:
        return None, None, src
    best_hr: list[float] | None = None
    best_pw: list[float] | None = None
    best_hr_tot = -1.0
    best_pw_tot = -1.0
    skip = frozenset({"record", "event"})
    for msg in fit.get_messages():
        if getattr(msg, "type", None) != "data":
            continue
        if msg.name in skip:
            continue
        if need_hr:
            hr, hr_label = _read_hr_zone_seconds_from_message(msg)
            if hr:
                t = _total_zone_time(hr)
                if t > best_hr_tot:
                    best_hr_tot = t
                    best_hr = hr
                    src["hr_time"] = hr_label
        if need_pw:
            pw, pw_label = _read_power_zone_seconds_from_message(msg)
            if pw:
                t = _total_zone_time(pw)
                if t > best_pw_tot:
                    best_pw_tot = t
                    best_pw = pw
                    src["pw_time"] = pw_label
    return best_hr, best_pw, src


def _scan_boundary_fields(fit: FitFile) -> tuple[list[float], list[float], dict[str, str | None]]:
    """Pick up Garmin developer boundary arrays from session/lap/activity messages."""
    sources: dict[str, str | None] = {"hr_bounds": None, "pw_bounds": None}
    best_hr: list[float] = []
    best_pw: list[float] = []
    best_hr_score = (-1, -1)
    best_pw_score = (-1, -1)

    def score_bounds(b: list[float]) -> tuple[int, int]:
        return (1 if len(b) >= 5 else 0, len(b))

    for mesg_name in ("session", "lap", "activity"):
        for msg in fit.get_messages(mesg_name):
            fm = _msg_field_dict(msg)
            for fname, raw in fm.items():
                if not fname:
                    continue
                key = str(fname).strip().lower()
                seq = _zone_boundary_array(raw)
                if not seq:
                    continue
                if key in _HR_BOUNDARY_FIELD_NAMES or (
                    "boundary" in key and "hr" in key and "power" not in key and "pwr" not in key
                ) or (
                    "zone" in key
                    and "high" in key
                    and ("hr" in key or "heart" in key)
                    and "power" not in key
                    and "pwr" not in key
                ):
                    sc = score_bounds(seq)
                    if sc > best_hr_score:
                        best_hr_score = sc
                        best_hr = seq
                        sources["hr_bounds"] = f"{mesg_name}:{fname}"
                elif key in _POWER_BOUNDARY_FIELD_NAMES or (
                    "boundary" in key and ("power" in key or "pwr" in key)
                ) or (
                    "zone" in key and "high" in key and ("power" in key or "pwr" in key)
                ):
                    sc = score_bounds(seq)
                    if sc > best_pw_score:
                        best_pw_score = sc
                        best_pw = seq
                        sources["pw_bounds"] = f"{mesg_name}:{fname}"

    return best_hr, best_pw, sources


def _bounds_score(b: list[float]) -> tuple[int, int]:
    return (1 if len(b) >= 5 else 0, len(b))


def extract_garmin_zone_snapshot_from_fit(
    fit: FitFile, *, fit_source_label: str | None = None
) -> dict[str, Any] | None:
    """Read Garmin zone boundaries, session/lap time-in-zone arrays, and zones_target thresholds."""
    label = fit_source_label or "fit"
    run_mesg216_debug_if_enabled(fit, label)
    hr_rows: list[tuple[int | None, float]] = []
    for msg in fit.get_messages("hr_zone"):
        fm = _msg_field_dict(msg)
        idx = fm.get("message_index")
        hi = fm.get("high_bpm")
        if hi is None:
            hi = _field_value_by_def_num(msg, 1)
        if hi is None:
            continue
        try:
            hi_f = float(hi)
        except (TypeError, ValueError):
            continue
        if hi_f <= 0:
            continue
        try:
            idx_i = int(idx) if idx is not None else None
        except (TypeError, ValueError):
            idx_i = None
        hr_rows.append((idx_i, hi_f))

    power_rows: list[tuple[int | None, float]] = []
    for msg in fit.get_messages("power_zone"):
        fm = _msg_field_dict(msg)
        idx = fm.get("message_index")
        hi = fm.get("high_value")
        if hi is None:
            hi = _field_value_by_def_num(msg, 1)
        if hi is None:
            continue
        try:
            hi_f = float(hi)
        except (TypeError, ValueError):
            continue
        if hi_f <= 0:
            continue
        try:
            idx_i = int(idx) if idx is not None else None
        except (TypeError, ValueError):
            idx_i = None
        power_rows.append((idx_i, hi_f))

    zones_target: dict[str, Any] = {}
    for msg in fit.get_messages("zones_target"):
        fm = _msg_field_dict(msg)
        for k, v in fm.items():
            if v is None:
                continue
            if zones_target.get(k) is None:
                zones_target[k] = v

    resting_hr_f: float | None = None
    for msg in fit.get_messages("user_profile"):
        fm = _msg_field_dict(msg)
        r = fm.get("resting_heart_rate")
        if r is not None:
            try:
                rf = float(r)
            except (TypeError, ValueError):
                continue
            if rf > 0:
                resting_hr_f = rf
                break

    def _ordered_bounds(rows: list[tuple[int | None, float]]) -> list[float]:
        rows_sorted = sorted(rows, key=lambda t: (t[0] is not None, t[0] if t[0] is not None else 0))
        return [r[1] for r in rows_sorted]

    hr_from_msgs = _ordered_bounds(hr_rows)
    pw_from_msgs = _ordered_bounds(power_rows)

    scan_hr_b, scan_pw_b, scan_b_src = _scan_boundary_fields(fit)
    hr_bounds, hr_b_src = _pick_richer_boundaries(hr_from_msgs, scan_hr_b, "hr_zone messages")
    pw_bounds, pw_b_src = _pick_richer_boundaries(pw_from_msgs, scan_pw_b, "power_zone messages")

    hr_times, pw_times, time_src = _merge_time_arrays_from_sessions_and_laps(fit)

    thr_hr = zones_target.get("threshold_heart_rate")
    ftp = zones_target.get("functional_threshold_power")
    max_hr = zones_target.get("max_heart_rate")

    def _to_float(v: Any) -> float | None:
        if v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if f > 0 else None

    thr_hr_f = _to_float(thr_hr)
    ftp_f = _to_float(ftp)
    max_hr_f = _to_float(max_hr)
    max_hr_src: str | None = "zones_target" if max_hr_f is not None else None
    if max_hr_f is None:
        for msg in fit.get_messages("user_profile"):
            fm = _msg_field_dict(msg)
            for uk in (
                "default_max_heart_rate",
                "default_max_running_heart_rate",
                "default_max_biking_heart_rate",
            ):
                mf = _to_float(fm.get(uk))
                if mf is not None:
                    max_hr_f = mf
                    max_hr_src = "user_profile"
                    break
            if max_hr_f is not None:
                break

    mesg216_notes: list[str] | None = None
    m216_extras: dict[str, Any] = {}
    m216_hr_calc_raw: Any = None
    m216_pwr_calc_raw: Any = None
    thr_from_m216 = False
    ftp_from_m216 = False
    resting_from_m216 = False
    m216_dec = decode_garmin_mesg216_zone_arrays(fit)
    if m216_dec:
        m216 = dict(m216_dec)
        m216.pop("_mesg216_selected_row", None)
        zone_time_prov_m216 = bool(m216.pop("garmin_mesg216_zone_time_sec_provisional", False))
        zone_time_conf_m216 = m216.pop("garmin_mesg216_zone_time_sec_confirmed", None)
        zone_time_conv_m216 = m216.pop("garmin_mesg216_zone_time_conversion", None)
        fk_hr_b = m216.pop("_mesg216_field_hr_zone_boundaries", None)
        fk_pw_b = m216.pop("_mesg216_field_power_zone_boundaries", None)
        fk_hr_t = m216.pop("_mesg216_field_hr_zone_time_sec", None)
        fk_pw_t = m216.pop("_mesg216_field_power_zone_time_sec", None)
        mesg216_notes = m216.pop("mesg216_decode_notes", None)

        m216_sel = m216_dec.get("_mesg216_selected_row")
        if m216_sel is not None:
            m216_extras["garmin_mesg216_selected_row"] = list(m216_sel)
        if zone_time_prov_m216:
            m216_extras["garmin_mesg216_zone_time_sec_provisional"] = True
        if zone_time_conf_m216 is not None:
            m216_extras["garmin_mesg216_zone_time_sec_confirmed"] = bool(zone_time_conf_m216)
        if zone_time_conv_m216:
            m216_extras["garmin_mesg216_zone_time_conversion"] = zone_time_conv_m216

        _m216_thr = m216.pop("threshold_heart_rate", None)
        _m216_ftp = m216.pop("functional_threshold_power", None)
        _m216_max = m216.pop("max_heart_rate", None)
        _m216_rst = m216.pop("resting_heart_rate", None)
        m216_hr_calc_raw = m216.pop("hr_calc_type", None)
        m216_pwr_calc_raw = m216.pop("power_calc_type", None)

        if thr_hr_f is None and _to_float(_m216_thr) is not None:
            thr_hr_f = _to_float(_m216_thr)
            thr_from_m216 = True
        if ftp_f is None and _to_float(_m216_ftp) is not None:
            ftp_f = _to_float(_m216_ftp)
            ftp_from_m216 = True
        if max_hr_f is None and _to_float(_m216_max) is not None:
            max_hr_f = _to_float(_m216_max)
            max_hr_src = "decoded_unknown_216"
        if resting_hr_f is None and _to_float(_m216_rst) is not None:
            resting_hr_f = _to_float(_m216_rst)
            resting_from_m216 = True

        for k in (
            "hr_zone_boundaries_high_bpm",
            "power_zone_boundaries_high_w",
            "hr_zone_time_raw",
            "power_zone_time_raw",
        ):
            if k not in m216:
                continue
            v = m216.pop(k)
            if v is not None:
                m216_extras[k] = v

        b216 = m216.get("hr_zone_boundaries") or []
        if b216 and (not hr_bounds or _bounds_score(b216) > _bounds_score(hr_bounds)):
            hr_bounds = list(b216)
            hr_b_src = (
                f"decoded_unknown_216:{fk_hr_b}" if fk_hr_b else "decoded_unknown_216"
            )
        p216 = m216.get("power_zone_boundaries") or []
        if p216 and (not pw_bounds or _bounds_score(p216) > _bounds_score(pw_bounds)):
            pw_bounds = list(p216)
            pw_b_src = (
                f"decoded_unknown_216:{fk_pw_b}" if fk_pw_b else "decoded_unknown_216"
            )

        t216h = m216.get("hr_zone_time_sec")
        if t216h and _total_zone_time(t216h) > _total_zone_time(hr_times):
            hr_times = list(t216h)
            time_src["hr_time"] = (
                f"decoded_unknown_216:{fk_hr_t}" if fk_hr_t else "decoded_unknown_216"
            )
        t216p = m216.get("power_zone_time_sec")
        if t216p and _total_zone_time(t216p) > _total_zone_time(pw_times):
            pw_times = list(t216p)
            time_src["pw_time"] = (
                f"decoded_unknown_216:{fk_pw_t}" if fk_pw_t else "decoded_unknown_216"
            )

    has_zones = bool(hr_bounds or pw_bounds)
    has_times = bool(
        (hr_times and _total_zone_time(hr_times) > 0) or (pw_times and _total_zone_time(pw_times) > 0)
    )
    has_targets = bool(thr_hr_f or ftp_f or max_hr_f or resting_hr_f)
    if not has_zones and not has_targets and not has_times:
        return None

    zone_extract_sources: dict[str, Any] = {
        "hr_zone_boundaries": hr_b_src,
        "power_zone_boundaries": pw_b_src,
        "hr_zone_time_sec": time_src.get("hr_time"),
        "power_zone_time_sec": time_src.get("pw_time"),
    }
    if scan_b_src.get("hr_bounds"):
        zone_extract_sources["hr_boundary_scan"] = scan_b_src["hr_bounds"]
    if scan_b_src.get("pw_bounds"):
        zone_extract_sources["power_boundary_scan"] = scan_b_src["pw_bounds"]
    if thr_hr_f is not None:
        zone_extract_sources["threshold_heart_rate"] = (
            "decoded_unknown_216" if thr_from_m216 else "zones_target"
        )
    if ftp_f is not None:
        zone_extract_sources["functional_threshold_power"] = (
            "decoded_unknown_216" if ftp_from_m216 else "zones_target"
        )
    if max_hr_src:
        zone_extract_sources["max_heart_rate"] = max_hr_src
    if resting_hr_f is not None:
        zone_extract_sources["resting_heart_rate"] = (
            "decoded_unknown_216" if resting_from_m216 else "user_profile"
        )
    if zones_target.get("hr_calc_type") is not None:
        zone_extract_sources["hr_calc_type"] = "zones_target"
    elif m216_hr_calc_raw is not None:
        zone_extract_sources["hr_calc_type"] = "decoded_unknown_216"
    if zones_target.get("pwr_calc_type") is not None:
        zone_extract_sources["pwr_calc_type"] = "zones_target"
    elif m216_pwr_calc_raw is not None:
        zone_extract_sources["pwr_calc_type"] = "decoded_unknown_216"

    out: dict[str, Any] = {
        "hr_zone_boundaries": hr_bounds,
        "power_zone_boundaries": pw_bounds,
        "threshold_heart_rate": thr_hr_f,
        "functional_threshold_power": ftp_f,
        "max_heart_rate": max_hr_f,
        "hr_calc_type": _calc_type_str(
            zones_target.get("hr_calc_type")
            if zones_target.get("hr_calc_type") is not None
            else m216_hr_calc_raw
        ),
        "power_calc_type": _calc_type_str(
            zones_target.get("pwr_calc_type")
            if zones_target.get("pwr_calc_type") is not None
            else m216_pwr_calc_raw
        ),
        "zone_extract_sources": zone_extract_sources,
    }
    if hr_times:
        out["hr_zone_time_sec"] = hr_times
    if pw_times:
        out["power_zone_time_sec"] = pw_times
    if resting_hr_f is not None:
        out["resting_heart_rate"] = resting_hr_f
    if mesg216_notes:
        out["garmin_mesg216_decode_notes"] = mesg216_notes
    if m216_extras:
        out.update(m216_extras)
    hb5 = out.get("hr_zone_boundaries_high_bpm")
    pb5 = out.get("power_zone_boundaries_high_w")
    out["garmin_hr_zones_found"] = bool(isinstance(hb5, list) and len(hb5) >= 5)
    out["garmin_power_zones_found"] = bool(isinstance(pb5, list) and len(pb5) >= 5)
    if not out["garmin_hr_zones_found"]:
        zs = out.get("zone_extract_sources") or {}
        hb = out.get("hr_zone_boundaries") or []
        if isinstance(hb, list) and len(hb) >= 5 and "decoded_unknown_216" in str(zs.get("hr_zone_boundaries") or ""):
            out["garmin_hr_zones_found"] = True
            if not isinstance(hb5, list) or len(hb5) < 5:
                out["hr_zone_boundaries_high_bpm"] = list(hb)
    if not out["garmin_power_zones_found"]:
        zs = out.get("zone_extract_sources") or {}
        pb = out.get("power_zone_boundaries") or []
        if isinstance(pb, list) and len(pb) >= 5 and "decoded_unknown_216" in str(zs.get("power_zone_boundaries") or ""):
            out["garmin_power_zones_found"] = True
            if not isinstance(pb5, list) or len(pb5) < 5:
                out["power_zone_boundaries_high_w"] = list(pb)
    return out
