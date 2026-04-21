from __future__ import annotations

from typing import Any

from agenticrun.utils.parsing import parse_float

_FIT_MESSAGE_ACTIVITY_ID_FIELDS: frozenset[str] = frozenset(
    {
        "activity_id_string",
        "leader_activity_id_string",
    }
)


def _scan_fit_messages_for_garmin_activity_string_id(fit: Any) -> str | None:
    """FIT fields that carry Garmin Connect–style decimal activity ids as strings."""
    for msg in fit.messages:
        for fd in getattr(msg, "fields", None) or []:
            name = getattr(fd, "name", None)
            if name not in _FIT_MESSAGE_ACTIVITY_ID_FIELDS:
                continue
            val = getattr(fd, "value", None)
            if val is None:
                continue
            s = str(val).strip()
            if s.isdigit() and 8 <= len(s) <= 14:
                return s
    return None


def _fit_session_content_signature_key(
    session_data: dict[str, Any],
    activity_data: dict[str, Any],
) -> str | None:
    """Stable key from session start + elapsed time + total distance (decoded FIT values)."""
    start = session_data.get("start_time") or activity_data.get("timestamp")
    if start is None or not hasattr(start, "timestamp"):
        return None
    try:
        unix = int(start.timestamp())
    except (TypeError, ValueError, OSError):
        return None

    elapsed = parse_float(session_data.get("total_elapsed_time"))
    if elapsed is None:
        elapsed = parse_float(session_data.get("total_timer_time"))
    if elapsed is None:
        elapsed = parse_float(activity_data.get("total_timer_time"))

    dist = parse_float(session_data.get("total_distance"))
    if dist is None:
        dist = parse_float(activity_data.get("total_distance"))

    if elapsed is None or dist is None:
        return None
    el_r = round(float(elapsed), 1)
    d_r = int(round(float(dist)))
    return f"fitsec:{unix}:{el_r}:{d_r}"


def derive_fit_activity_key(
    fit: Any,
    session_data: dict[str, Any],
    activity_data: dict[str, Any],
    _file_id_data: dict[str, Any],
) -> str | None:
    """Canonical duplicate-detection key from FIT content (not the filename).

    Prefers Garmin activity id strings found in FIT messages when present; otherwise a
    session summary signature (UTC start unix, elapsed s, total distance m).
    ``_file_id_data`` is reserved for future manufacturer-specific fields.
    """
    gstr = _scan_fit_messages_for_garmin_activity_string_id(fit)
    if gstr:
        return f"gaid:{gstr}"
    sig = _fit_session_content_signature_key(session_data, activity_data)
    if sig:
        return sig
    return None
