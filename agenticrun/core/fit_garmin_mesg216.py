# Garmin Time in Zone: FIT global message #216 (fitparse: unknown_216). Decode helpers + debug.

from __future__ import annotations

import struct
from collections import Counter, defaultdict
from typing import Any, Iterator

from fitparse import FitFile

_GARMIN_TIME_IN_ZONE_MESG_NUM = 216
_MAX_DEBUG_SAMPLES = 5
_MAX_PATTERN_KEYS = 12

# Explicit developer-field mapping (Garmin Time in Zone payload, global_mesg_num=216).
_F216_HR_ZONE_HIGH = "unknown_6"
_F216_PWR_ZONE_HIGH = "unknown_9"
_F216_HR_CALC_TYPE = "unknown_10"
_F216_MAX_HR = "unknown_11"
_F216_RESTING_HR = "unknown_12"
_F216_THRESHOLD_HR = "unknown_13"
_F216_PWR_CALC_TYPE = "unknown_14"
_F216_FTP = "unknown_15"
_F216_HR_ZONE_TIME = "unknown_2"
_F216_PWR_ZONE_TIME = "unknown_5"
_F216_ROW_KIND = "unknown_0"
_F216_ROW_SUB = "unknown_1"

# Preferred session-level row: correlates with FIT session message type (18) and primary index (0).
_PREFERRED_SESSION_U0 = 18
_PREFERRED_SESSION_U1 = 0


def _mesg216_verbose_debug() -> bool:
    """Raw unknown_216 row dumps and pattern summaries (developer-only; off by default)."""
    import os

    return os.getenv("AGENTICRUN_FIT_MESG216_VERBOSE", "").lower() in {"1", "true", "yes", "on"}


def iter_garmin_mesg216_data_messages(fit: FitFile) -> Iterator[Any]:
    for msg in fit.messages:
        if getattr(msg, "type", None) != "data":
            continue
        if getattr(msg, "mesg_num", None) == _GARMIN_TIME_IN_ZONE_MESG_NUM or msg.name == "unknown_216":
            yield msg


def _row_value_map(msg: Any) -> dict[str, Any]:
    return {fd.name: fd.value for fd in msg.fields}


def _field_debug_line(fd: Any) -> str:
    bt = getattr(fd.base_type, "name", "?")
    rv = fd.raw_value
    rv_s = repr(rv)
    if len(rv_s) > 72:
        rv_s = rv_s[:69] + "..."
    return f"{fd.name} value={fd.value!r} type={type(fd.value).__name__} raw={rv_s} base_type={bt}"


def debug_print_mesg216_samples(fit: FitFile, file_label: str) -> None:
    """Print 2–5 sample rows with field names, decoded values, types, and raw/base_type."""
    msgs = list(iter_garmin_mesg216_data_messages(fit))
    if not msgs:
        return
    n = min(_MAX_DEBUG_SAMPLES, max(2, len(msgs)))
    print(f"--- FIT unknown_216 samples (global_mesg_num={_GARMIN_TIME_IN_ZONE_MESG_NUM}) {file_label} ---", flush=True)
    print(f"  total_rows: {len(msgs)}  showing: {n}", flush=True)
    for i, msg in enumerate(msgs[:n]):
        print(f"  row[{i}]:", flush=True)
        for fd in sorted(msg.fields, key=lambda x: x.def_num):
            print(f"    {_field_debug_line(fd)}", flush=True)


def _shape_one(v: Any) -> str:
    if v is None:
        return "None"
    if isinstance(v, (tuple, list)):
        return f"{type(v).__name__}[{len(v)}]"
    return type(v).__name__


def _structural_pattern(row: dict[str, Any]) -> str:
    parts = [f"{k}={_shape_one(row[k])}" for k in sorted(row.keys())]
    return "|".join(parts)


def _leading_triple(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (row.get("unknown_0"), row.get("unknown_1"), row.get("unknown_2"))


def debug_print_mesg216_patterns(fit: FitFile, file_label: str) -> None:
    """Compact pattern summaries: structural shapes and (unknown_0,1,2) histograms."""
    msgs = list(iter_garmin_mesg216_data_messages(fit))
    if not msgs:
        return
    rows = [_row_value_map(m) for m in msgs]
    pat_counts = Counter(_structural_pattern(r) for r in rows)
    lead_counts = Counter(_leading_triple(r) for r in rows)

    print(f"--- FIT unknown_216 pattern summary {file_label} ---", flush=True)
    print(f"  distinct_structural_patterns: {len(pat_counts)}", flush=True)
    for pat, cnt in pat_counts.most_common(_MAX_PATTERN_KEYS):
        print(f"    count={cnt}  pattern={pat[:300]}{'...' if len(pat) > 300 else ''}", flush=True)
    if len(pat_counts) > _MAX_PATTERN_KEYS:
        print(f"    ... ({len(pat_counts) - _MAX_PATTERN_KEYS} more patterns)", flush=True)

    print(f"  leading (unknown_0, unknown_1, unknown_2) histogram (top {min(15, len(lead_counts))}):", flush=True)
    for key, cnt in lead_counts.most_common(15):
        print(f"    count={cnt}  triple={key!r}", flush=True)


def _as_5_floats(val: Any) -> list[float] | None:
    if not isinstance(val, (tuple, list)) or len(val) != 5:
        return None
    out: list[float] = []
    for x in val:
        if x is None:
            return None
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            return None
    return out


def _list_floats_skip_none(val: Any, *, max_n: int = 16) -> list[float]:
    if not isinstance(val, (tuple, list)):
        return []
    out: list[float] = []
    for x in val[:max_n]:
        if x is None:
            continue
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            continue
    return out


def _mesg216_hr_zone_highs_bpm(val: Any) -> list[float] | None:
    """
    unknown_6: five ascending HR zone high edges (BPM); FIT often includes a 6th slot (e.g. max HR).
    """
    xs = _list_floats_skip_none(val, max_n=8)
    if len(xs) < 5:
        return None
    head = xs[:5]
    if not all(30.0 <= x <= 260.0 for x in head):
        return None
    if not all(head[i] <= head[i + 1] + 1e-6 for i in range(4)):
        return None
    return head


def _mesg216_power_zone_highs_w(val: Any) -> list[float] | None:
    """
    unknown_9: five ascending power zone high edges (W); later slots may hold sentinels (e.g. 4000).
    """
    xs = _list_floats_skip_none(val, max_n=12)
    if not xs:
        return None
    trimmed: list[float] = []
    for x in xs:
        if x > 1500.0:
            break
        trimmed.append(x)
        if len(trimmed) == 5:
            break
    if len(trimmed) >= 5:
        head = trimmed[:5]
        if all(head[i] <= head[i + 1] + 1e-6 for i in range(4)) and all(x > 0 for x in head):
            return head
    if len(xs) >= 5:
        head5 = xs[:5]
        if all(head5[i] <= head5[i + 1] + 1e-6 for i in range(4)) and all(0 < x <= 1500 for x in head5):
            return head5
    return None


def _session_total_timer_sec(fit: FitFile) -> float | None:
    for msg in fit.get_messages("session"):
        if getattr(msg, "type", None) != "data":
            continue
        for fd in msg.fields:
            if fd.name != "total_timer_time" or fd.value is None:
                continue
            try:
                t = float(fd.value)
            except (TypeError, ValueError):
                continue
            if t > 0:
                return t
    return None


def _seven_slot_floats(val: Any) -> list[float] | None:
    if not isinstance(val, (tuple, list)):
        return None
    out: list[float] = []
    for x in val[:7]:
        if x is None:
            out.append(0.0)
            continue
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            return None
    return out if out else None


def _mesg216_zone_times_seconds(
    val: Any,
    *,
    total_timer_sec: float | None,
    field_name: str,
) -> tuple[list[float] | None, str, bool]:
    """
    unknown_2 / unknown_5: uint32 per-zone buckets in FIT logs are milliseconds; divide by 1000 for seconds.
    When session total_timer_time is available, compare sum(zone_sec) to mark session-aligned (not provisional).
    """
    if val is None:
        return None, f"{field_name}: absent", False

    if isinstance(val, (bytes, bytearray)):
        if len(val) < 4 or len(val) % 4 != 0:
            return None, f"{field_name}: unparseable bytes", False
        raw: list[float] = []
        for i in range(0, len(val), 4):
            raw.append(float(struct.unpack_from("<I", val, i)[0]))
        slots = raw[:7]
        while len(slots) < 7:
            slots.append(0.0)
    else:
        slots = _seven_slot_floats(val)
        if slots is None:
            return None, f"{field_name}: not a 7-slot numeric tuple/list", False

    sum_raw = float(sum(slots))
    sec_as_ms_scaled = [x / 1000.0 for x in slots]
    sum_ms_conv = float(sum(sec_as_ms_scaled))

    use_as_raw_seconds = sec_as_ms_scaled
    note = f"{field_name}: uint32 slots interpreted as ms, divided by 1000 -> seconds (Garmin mesg216 evidence)"
    confirmed = False

    if total_timer_sec is not None and total_timer_sec > 30.0:
        err_ms = abs(sum_ms_conv - total_timer_sec)
        err_raw = abs(sum_raw - total_timer_sec)
        tol = max(120.0, 0.22 * total_timer_sec)
        if err_ms <= tol:
            confirmed = True
            note += f"; sum={sum_ms_conv:.1f}s vs session total_timer_time={total_timer_sec:.1f}s (within tol)"
        elif err_raw <= tol:
            use_as_raw_seconds = list(slots)
            sum_check = sum_raw
            confirmed = True
            note = (
                f"{field_name}: values treated as seconds (sum={sum_check:.1f}s aligned with "
                f"total_timer_time={total_timer_sec:.1f}s)"
            )
        else:
            note += (
                f"; sum_ms/1000={sum_ms_conv:.1f}s vs total_timer_time={total_timer_sec:.1f}s "
                f"(diff={sum_ms_conv - total_timer_sec:.1f}s — keep ms/1000; verify in Connect)"
            )
    else:
        note += "; no session total_timer_time — conversion plausible but not session-verified"

    if not any(x > 0 for x in use_as_raw_seconds):
        return None, f"{field_name}: all zero after conversion", False

    return use_as_raw_seconds, note, confirmed


def _classify_5_tuple(seq: list[float]) -> str | None:
    """Return hr_bounds | power_bounds | zone_time_seconds or None (conservative)."""
    if len(seq) != 5:
        return None
    s, mx, mn = sum(seq), max(seq), min(seq)
    if mn < 0:
        return None
    inc = all(seq[i] <= seq[i + 1] + 1e-6 for i in range(4))

    # HR zone high edges (bpm): ascending, plausible range
    if inc and all(35 <= x <= 230 for x in seq) and mx <= 230 and s < 2000:
        return "hr_bounds"

    # Power zone high edges (w): ascending, plausible watts (sum of edges rarely ~time totals)
    if inc and all(20 <= x for x in seq) and mx <= 520 and s < 2600:
        return "power_bounds"

    # Time-in-zone seconds (per-zone or cumulative layouts; often non-monotonic)
    if s >= 90 and mx >= 30:
        return "zone_time_seconds"

    return None


def _collect_5_tuple_candidates(rows: list[dict[str, Any]]) -> defaultdict[str, list[tuple[str, list[float]]]]:
    by_class: defaultdict[str, list[tuple[str, list[float]]]] = defaultdict(list)
    for row in rows:
        for k, v in row.items():
            seq = _as_5_floats(v)
            if not seq:
                continue
            label = _classify_5_tuple(seq)
            if label:
                by_class[label].append((k, seq))
    return by_class


def _copy_raw_for_debug(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, (bytes, bytearray)):
        return val.hex()
    return val


def _scalar_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f > 0 else None


def _select_mesg216_row(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], tuple[Any, Any]]:
    """Prefer unknown_0==18 and unknown_1==0; else conservative fallbacks."""
    for r in rows:
        if r.get(_F216_ROW_KIND) == _PREFERRED_SESSION_U0 and r.get(_F216_ROW_SUB) == _PREFERRED_SESSION_U1:
            return r, (r.get(_F216_ROW_KIND), r.get(_F216_ROW_SUB))
    for r in rows:
        if r.get(_F216_ROW_KIND) == _PREFERRED_SESSION_U0:
            return r, (r.get(_F216_ROW_KIND), r.get(_F216_ROW_SUB))
    for r in rows:
        if r.get(_F216_HR_ZONE_HIGH) is not None or r.get(_F216_HR_ZONE_TIME) is not None:
            return r, (r.get(_F216_ROW_KIND), r.get(_F216_ROW_SUB))
    r0 = rows[0]
    return r0, (r0.get(_F216_ROW_KIND), r0.get(_F216_ROW_SUB))


def _legacy_tuple_scan_decode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Previous heuristic: classify 5-tuples across all rows (fallback only)."""
    by_class = _collect_5_tuple_candidates(rows)
    out: dict[str, Any] = {}
    notes: list[str] = []

    hb = by_class.get("hr_bounds")
    if hb:
        key, seq = hb[0]
        out["hr_zone_boundaries"] = seq
        out["_mesg216_field_hr_zone_boundaries"] = key
        notes.append(f"hr_zone_boundaries from unknown_216 field {key!r} (5-tuple hr_bounds pattern)")

    pb = by_class.get("power_bounds")
    if pb:
        key, seq = pb[0]
        out["power_zone_boundaries"] = seq
        out["_mesg216_field_power_zone_boundaries"] = key
        notes.append(f"power_zone_boundaries from unknown_216 field {key!r} (5-tuple power_bounds pattern)")

    zt = by_class.get("zone_time_seconds")
    if zt:
        seen: list[list[float]] = []
        seen_keys: list[str] = []
        for key, seq in zt:
            if not any(seq == s for s in seen):
                seen.append(seq)
                seen_keys.append(key)
        if len(seen) >= 1:
            out["hr_zone_time_sec"] = seen[0]
            out["_mesg216_field_hr_zone_time_sec"] = seen_keys[0]
            notes.append(f"hr_zone_time_sec from unknown_216 field {seen_keys[0]!r} (zone_time_seconds pattern)")
        if len(seen) >= 2:
            out["power_zone_time_sec"] = seen[1]
            out["_mesg216_field_power_zone_time_sec"] = seen_keys[1]
            notes.append(f"power_zone_time_sec from unknown_216 field {seen_keys[1]!r} (zone_time_seconds pattern)")
        elif len(seen) == 1 and len(zt) > 1:
            notes.append("single distinct zone_time 5-tuple; power time not inferred")

    if notes:
        out["_legacy_notes"] = notes
    return out


def debug_print_mesg216_decoded(decoded: dict[str, Any], file_label: str) -> None:
    """Print semantic fields from decode_garmin_mesg216_zone_arrays (decoded_unknown_216)."""
    print(f"--- FIT decoded_unknown_216 (semantic) {file_label} ---", flush=True)
    sel = decoded.get("_mesg216_selected_row")
    print(f"  selected_unknown_216_row = {sel!r}", flush=True)
    print("  source: decoded_unknown_216 (global_mesg_num=216)", flush=True)
    for key in (
        "hr_zone_boundaries_high_bpm",
        "power_zone_boundaries_high_w",
        "hr_zone_time_raw",
        "power_zone_time_raw",
        "hr_calc_type",
        "max_heart_rate",
        "resting_heart_rate",
        "threshold_heart_rate",
        "power_calc_type",
        "functional_threshold_power",
    ):
        if key in decoded and decoded[key] is not None:
            print(f"  {key} = {decoded[key]!r}", flush=True)
    if decoded.get("hr_zone_time_sec") is not None:
        _t = "confirmed vs session total_timer_time" if decoded.get("garmin_mesg216_zone_time_sec_confirmed") else "provisional"
        print(f"  hr_zone_time_sec ({_t}) = {decoded['hr_zone_time_sec']!r}", flush=True)
    if decoded.get("power_zone_time_sec") is not None:
        _t = "confirmed vs session total_timer_time" if decoded.get("garmin_mesg216_zone_time_sec_confirmed") else "provisional"
        print(f"  power_zone_time_sec ({_t}) = {decoded['power_zone_time_sec']!r}", flush=True)
    zc = decoded.get("garmin_mesg216_zone_time_conversion")
    if zc:
        print(f"  garmin_mesg216_zone_time_conversion = {zc!r}", flush=True)
    print(f"  garmin_mesg216_zone_time_sec_confirmed = {decoded.get('garmin_mesg216_zone_time_sec_confirmed')!r}", flush=True)
    print(f"  garmin_mesg216_zone_time_sec_provisional = {decoded.get('garmin_mesg216_zone_time_sec_provisional')!r}", flush=True)
    notes = decoded.get("mesg216_decode_notes")
    if notes:
        print(f"  mesg216_decode_notes: {notes}", flush=True)


def debug_print_mesg216_timer_consistency(fit: FitFile, decoded: dict[str, Any], file_label: str) -> None:
    """Debug-only: compare sums of zone times to FIT session total_timer_time."""
    tt = _session_total_timer_sec(fit)
    print(f"--- FIT decoded_unknown_216 (timer consistency) {file_label} ---", flush=True)
    print(f"  session total_timer_time_sec = {tt!r}", flush=True)
    for label, key in (("HR", "hr_zone_time_sec"), ("power", "power_zone_time_sec")):
        arr = decoded.get(key)
        if not arr:
            print(f"  sum({key}): (none)", flush=True)
            continue
        try:
            s = float(sum(float(x) for x in arr))
        except (TypeError, ValueError):
            print(f"  sum({key}): (unreadable)", flush=True)
            continue
        print(f"  sum({key}) = {s:.3f}s  ({label})", flush=True)
        if tt is not None and tt > 0:
            print(f"  {key} sum - total_timer_time = {s - tt:.3f}s; ratio sum/timer = {s / tt:.4f}", flush=True)


def decode_garmin_mesg216_zone_arrays(fit: FitFile) -> dict[str, Any] | None:
    """
    Decode Garmin Time in Zone from unknown_216 using explicit field mapping on the
    preferred session row (unknown_0==18, unknown_1==0), with legacy 5-tuple scan as fallback.
    """
    msgs = list(iter_garmin_mesg216_data_messages(fit))
    if not msgs:
        return None
    rows = [_row_value_map(m) for m in msgs]
    row, selected = _select_mesg216_row(rows)

    out: dict[str, Any] = {}
    notes: list[str] = []
    out["_mesg216_selected_row"] = selected
    notes.append(
        f"selected unknown_216 row (unknown_0, unknown_1)={selected!r} "
        f"(preferred=({_PREFERRED_SESSION_U0}, {_PREFERRED_SESSION_U1}))"
    )

    hr_b = _mesg216_hr_zone_highs_bpm(row.get(_F216_HR_ZONE_HIGH))
    if not hr_b:
        hr_b = _as_5_floats(row.get(_F216_HR_ZONE_HIGH))
    pw_b = _mesg216_power_zone_highs_w(row.get(_F216_PWR_ZONE_HIGH))
    if not pw_b:
        pw_b = _as_5_floats(row.get(_F216_PWR_ZONE_HIGH))
    hr_raw = row.get(_F216_HR_ZONE_TIME)
    pw_raw = row.get(_F216_PWR_ZONE_TIME)

    out["hr_zone_time_raw"] = _copy_raw_for_debug(hr_raw)
    out["power_zone_time_raw"] = _copy_raw_for_debug(pw_raw)

    if hr_b:
        out["hr_zone_boundaries_high_bpm"] = hr_b
        out["hr_zone_boundaries"] = list(hr_b)
        out["_mesg216_field_hr_zone_boundaries"] = _F216_HR_ZONE_HIGH
        notes.append(f"hr_zone_boundaries_high_bpm from {_F216_HR_ZONE_HIGH!r} (explicit)")
    if pw_b:
        out["power_zone_boundaries_high_w"] = pw_b
        out["power_zone_boundaries"] = list(pw_b)
        out["_mesg216_field_power_zone_boundaries"] = _F216_PWR_ZONE_HIGH
        notes.append(f"power_zone_boundaries_high_w from {_F216_PWR_ZONE_HIGH!r} (explicit)")

    v10, v11, v12, v13 = (
        row.get(_F216_HR_CALC_TYPE),
        row.get(_F216_MAX_HR),
        row.get(_F216_RESTING_HR),
        row.get(_F216_THRESHOLD_HR),
    )
    if v10 is not None:
        out["hr_calc_type"] = v10
        notes.append(f"hr_calc_type from {_F216_HR_CALC_TYPE!r}")
    mf = _scalar_float(v11)
    if mf is not None:
        out["max_heart_rate"] = mf
    rf = _scalar_float(v12)
    if rf is not None:
        out["resting_heart_rate"] = rf
    tf = _scalar_float(v13)
    if tf is not None:
        out["threshold_heart_rate"] = tf

    v14, v15 = row.get(_F216_PWR_CALC_TYPE), row.get(_F216_FTP)
    if v14 is not None:
        out["power_calc_type"] = v14
        notes.append(f"power_calc_type from {_F216_PWR_CALC_TYPE!r}")
    ftp_f = _scalar_float(v15)
    if ftp_f is not None:
        out["functional_threshold_power"] = ftp_f

    timer_sec = _session_total_timer_sec(fit)
    hr_t, hr_time_note, hr_time_conf = _mesg216_zone_times_seconds(
        hr_raw, total_timer_sec=timer_sec, field_name=_F216_HR_ZONE_TIME
    )
    pw_t, pw_time_note, pw_time_conf = _mesg216_zone_times_seconds(
        pw_raw, total_timer_sec=timer_sec, field_name=_F216_PWR_ZONE_TIME
    )
    notes.append(hr_time_note)
    notes.append(pw_time_note)

    time_channel_conf: list[bool] = []
    if hr_raw is not None and hr_t is not None:
        out["hr_zone_time_sec"] = hr_t
        out["_mesg216_field_hr_zone_time_sec"] = _F216_HR_ZONE_TIME
        time_channel_conf.append(hr_time_conf)
    if pw_raw is not None and pw_t is not None:
        out["power_zone_time_sec"] = pw_t
        out["_mesg216_field_power_zone_time_sec"] = _F216_PWR_ZONE_TIME
        time_channel_conf.append(pw_time_conf)

    if time_channel_conf:
        all_conf = all(time_channel_conf)
        out["garmin_mesg216_zone_time_sec_confirmed"] = all_conf
        out["garmin_mesg216_zone_time_sec_provisional"] = not all_conf
        out["garmin_mesg216_zone_time_conversion"] = (
            "uint32_milliseconds_per_zone_divided_by_1000; "
            "session check uses FIT session.total_timer_time vs sum(zone seconds)"
        )

    legacy = _legacy_tuple_scan_decode(rows)
    leg_notes = legacy.pop("_legacy_notes", None)
    if leg_notes:
        notes.extend(leg_notes)

    if not out.get("hr_zone_boundaries"):
        k = legacy.get("hr_zone_boundaries")
        fk = legacy.get("_mesg216_field_hr_zone_boundaries")
        if k is not None:
            out["hr_zone_boundaries"] = k
            if fk:
                out["_mesg216_field_hr_zone_boundaries"] = fk
            notes.append("hr_zone_boundaries filled from legacy 5-tuple scan (explicit row lacked usable unknown_6)")
    if not out.get("power_zone_boundaries"):
        k = legacy.get("power_zone_boundaries")
        fk = legacy.get("_mesg216_field_power_zone_boundaries")
        if k is not None:
            out["power_zone_boundaries"] = k
            if fk:
                out["_mesg216_field_power_zone_boundaries"] = fk
            notes.append("power_zone_boundaries filled from legacy 5-tuple scan (explicit row lacked usable unknown_9)")
    legacy_filled_hr_time = False
    legacy_filled_pw_time = False
    if not out.get("hr_zone_time_sec"):
        k = legacy.get("hr_zone_time_sec")
        fk = legacy.get("_mesg216_field_hr_zone_time_sec")
        if k is not None:
            out["hr_zone_time_sec"] = k
            if fk:
                out["_mesg216_field_hr_zone_time_sec"] = fk
            notes.append("hr_zone_time_sec filled from legacy 5-tuple scan (explicit unknown_2 not interpreted)")
            legacy_filled_hr_time = True
    if not out.get("power_zone_time_sec"):
        k = legacy.get("power_zone_time_sec")
        fk = legacy.get("_mesg216_field_power_zone_time_sec")
        if k is not None:
            out["power_zone_time_sec"] = k
            if fk:
                out["_mesg216_field_power_zone_time_sec"] = fk
            notes.append("power_zone_time_sec filled from legacy 5-tuple scan (explicit unknown_5 not interpreted)")
            legacy_filled_pw_time = True
    if legacy_filled_hr_time or legacy_filled_pw_time:
        out["garmin_mesg216_zone_time_sec_confirmed"] = False
        out["garmin_mesg216_zone_time_sec_provisional"] = True
        prev = (out.get("garmin_mesg216_zone_time_conversion") or "").strip()
        leg = "legacy 5-tuple zone_time classifier (provisional; not unknown_2/5 ms path)"
        out["garmin_mesg216_zone_time_conversion"] = f"{prev} {leg}".strip() if prev else leg

    usable = (
        out.get("hr_zone_boundaries")
        or out.get("power_zone_boundaries")
        or out.get("hr_zone_time_sec")
        or out.get("power_zone_time_sec")
        or out.get("max_heart_rate")
        or out.get("threshold_heart_rate")
        or out.get("functional_threshold_power")
        or out.get("resting_heart_rate")
        or out.get("hr_zone_boundaries_high_bpm")
        or out.get("power_zone_boundaries_high_w")
        or (out.get("hr_zone_time_raw") is not None)
        or (out.get("power_zone_time_raw") is not None)
        or (out.get("hr_calc_type") is not None)
        or (out.get("power_calc_type") is not None)
    )
    if not usable:
        return None

    out["mesg216_decode_notes"] = notes
    return out


def run_mesg216_debug_if_enabled(fit: FitFile, file_label: str) -> None:
    if not _mesg216_verbose_debug():
        return
    if not list(iter_garmin_mesg216_data_messages(fit)):
        return
    debug_print_mesg216_samples(fit, file_label)
    debug_print_mesg216_patterns(fit, file_label)
    decoded = decode_garmin_mesg216_zone_arrays(fit)
    if decoded:
        debug_print_mesg216_decoded(decoded, file_label)
        debug_print_mesg216_timer_consistency(fit, decoded, file_label)
