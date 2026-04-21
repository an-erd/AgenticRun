from __future__ import annotations

from typing import Any

# Zone-second arrays from FIT file (message 216 / session / lap), not recomputed from record stream.
GARMIN_FIT_ZONE_SECONDS_SOURCES = frozenset({"fit_garmin_mesg216", "fit_garmin_session_or_lap"})

_FIT_KEYS = (
    "moving_time_sec",
    "avg_moving_pace_sec_km",
    "stopped_time_sec",
    "power_zone_z1_sec",
    "power_zone_z2_sec",
    "power_zone_z3_sec",
    "power_zone_z4_sec",
    "power_zone_z5_sec",
    "hr_zone_z1_sec",
    "hr_zone_z2_sec",
    "hr_zone_z3_sec",
    "hr_zone_z4_sec",
    "hr_zone_z5_sec",
    "has_power",
    "has_hr",
    "has_cadence",
    "has_gps",
    "data_quality_score",
    "fit_parse_warnings",
    "lap_count",
    "zone_model_source",
    "zone_model_effective_from",
    "zone_model_source_run_id",
    "resolved_functional_threshold_power",
    "resolved_threshold_heart_rate",
    "resolved_max_heart_rate",
    "hr_zone_seconds_source",
    "power_zone_seconds_source",
)


def as_bool_flag(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def session_fit_metrics(source: Any) -> dict[str, Any]:
    """Return FIT-derived session metrics from a RunRecord or a flat DB history row."""
    if hasattr(source, "raw_summary"):
        rs = source.raw_summary
        if isinstance(rs, dict):
            nested = rs.get("fit_session_metrics")
            if isinstance(nested, dict):
                return {k: nested.get(k) for k in _FIT_KEYS}
        return {}
    if isinstance(source, dict):
        return {k: source.get(k) for k in _FIT_KEYS}
    return {}
