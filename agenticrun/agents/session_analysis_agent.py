from __future__ import annotations

import os
from typing import Any

from agenticrun.core.fit_power_zones import FIT_HR_REFERENCE_BPM, FIT_THRESHOLD_POWER_W
from agenticrun.core.models import RunRecord, RunState
from agenticrun.core.session_fit_metrics import (
    GARMIN_FIT_ZONE_SECONDS_SOURCES,
    as_bool_flag,
    session_fit_metrics,
)
from agenticrun.utils.parsing import format_pace_min_km

LOW_DATA_QUALITY_THRESHOLD = 50.0
MIN_ZONE_SECONDS = 120.0
HIGH_STOPPED_FRACTION = 0.28
# Below this, interval/VO2 and threshold labels are easy to get wrong from sparse streams.
SOFT_DATA_QUALITY_FOR_SHARP_TYPES = 52.0

_INTENSITY_RANK = {
    "unknown": -1,
    "easy": 0,
    "moderate": 1,
    "moderate_high": 2,
    "high": 3,
}


def _zone_seconds(prefix: str, fm: dict[str, Any]) -> tuple[float, list[float]]:
    zones = [float(fm.get(f"{prefix}_z{i}_sec") or 0) for i in range(1, 6)]
    return sum(zones), zones


def _intensity_label_from_zone_fractions(z4: float, z5: float, z3: float, z2: float, z1: float) -> str:
    total = z1 + z2 + z3 + z4 + z5
    if total <= 0:
        return "unknown"
    hi = (z4 + z5) / total
    z3f = z3 / total
    aerobic_mid = (z2 + z3) / total
    if hi >= 0.15:
        return "high"
    if hi >= 0.06 or z3f >= 0.28:
        return "moderate_high"
    if aerobic_mid >= 0.35 or z3f >= 0.12:
        return "moderate"
    return "easy"


def _intensity_from_power_zones(run: RunRecord, fm: dict[str, Any]) -> str | None:
    p_total, pz = _zone_seconds("power_zone", fm)
    if not (as_bool_flag(fm.get("has_power")) or run.avg_power is not None):
        return None
    if p_total < MIN_ZONE_SECONDS:
        return None
    return _intensity_label_from_zone_fractions(pz[3], pz[4], pz[2], pz[1], pz[0])


def _intensity_from_hr_zones(run: RunRecord, fm: dict[str, Any]) -> str | None:
    h_total, hz = _zone_seconds("hr_zone", fm)
    if not (as_bool_flag(fm.get("has_hr")) or run.avg_hr is not None):
        return None
    if h_total < MIN_ZONE_SECONDS:
        return None
    return _intensity_label_from_zone_fractions(hz[3], hz[4], hz[2], hz[1], hz[0])


def _merge_power_hr_intensity(
    power_lbl: str, hr_lbl: str, fm: dict[str, Any] | None = None
) -> str:
    """When both zone clocks agree, use them; when they diverge sharply, avoid extreme labels."""
    rp = _INTENSITY_RANK.get(power_lbl, -1)
    rh = _INTENSITY_RANK.get(hr_lbl, -1)
    if fm is not None:
        ps = fm.get("power_zone_seconds_source")
        hs = fm.get("hr_zone_seconds_source")
        if ps in GARMIN_FIT_ZONE_SECONDS_SOURCES and hs == "fit_record_recalc" and rp >= 0:
            return power_lbl
        if hs in GARMIN_FIT_ZONE_SECONDS_SOURCES and ps == "fit_record_recalc" and rh >= 0:
            return hr_lbl
    if rp < 0:
        return hr_lbl
    if rh < 0:
        return power_lbl
    if abs(rp - rh) >= 2:
        # e.g. power says easy while HR says high — pick middle rather than trusting either extreme.
        return "moderate"
    # Adjacent bands: take the higher stress reading so easy days are not mislabeled too soft.
    return power_lbl if rp >= rh else hr_lbl


def infer_intensity_from_fit(run: RunRecord, fm: dict[str, Any]) -> str:
    ip = _intensity_from_power_zones(run, fm)
    ih = _intensity_from_hr_zones(run, fm)
    if ip and ih:
        return _merge_power_hr_intensity(ip, ih, fm)
    if ip:
        return ip
    if ih:
        return ih
    return _infer_intensity_from_averages(run.avg_power, run.avg_hr, fm)


def _lap_count_value(fm: dict[str, Any]) -> int | None:
    v = fm.get("lap_count")
    if v is None:
        return None
    try:
        return max(0, int(float(v)))
    except (TypeError, ValueError):
        return None


def _vo2_interval_structure_flags(
    run: RunRecord,
    easy_share_p: float | None,
    high_share_p: float | None,
    z5_share_p: float | None,
    laps: int | None,
) -> tuple[bool, bool]:
    """Repeated hard/easy mix or clear top-end time / power — matches core hard-session gate."""
    interval_structure = (
        laps is not None
        and laps >= 6
        and easy_share_p is not None
        and high_share_p is not None
        and easy_share_p >= 0.12
        and high_share_p >= 0.10
    )
    vo2_hard = (
        (run.avg_power and run.avg_power >= 262)
        or (high_share_p is not None and high_share_p >= 0.20)
        or (
            z5_share_p is not None
            and high_share_p is not None
            and z5_share_p >= 0.055
            and high_share_p >= 0.13
        )
    )
    return interval_structure, vo2_hard


def _refine_hard_session_type(
    run: RunRecord,
    fm: dict[str, Any],
    moving_time: Any,
    easy_share_p: float | None,
    high_share_p: float | None,
    z5_share_p: float | None,
    z1_share_h: float | None,
    high_share_h: float | None,
    p_total: float,
    h_total: float,
    laps: int | None,
    dq_f: float | None,
    interval_structure: bool,
    vo2_hard: bool,
) -> tuple[str, str]:
    """
    Split the coarse VO2/interval core bucket into race / vo2_interval_session / test_session,
    or keep test_or_vo2_session when evidence is ambiguous.
    Returns (training_type, short_reason_for_debug).
    """
    if dq_f is not None and dq_f < SOFT_DATA_QUALITY_FOR_SHARP_TYPES:
        return "test_or_vo2_session", "weak_dq"
    if fm.get("fit_parse_warnings"):
        return "test_or_vo2_session", "fit_warnings"

    mt = float(moving_time) if moving_time is not None else 0.0
    if mt <= 0:
        return "test_or_vo2_session", "no_moving_time"

    resolved_mx = fm.get("resolved_max_heart_rate")
    try:
        mx_f = float(resolved_mx) if resolved_mx is not None else 0.0
    except (TypeError, ValueError):
        mx_f = 0.0
    max_hr = float(run.max_hr) if run.max_hr is not None else 0.0
    near_profile_max_hr = mx_f >= 150 and max_hr > 0 and (max_hr / mx_f) >= 0.96

    # Benchmark-style: near-max HR relative to profile, top-end zone time, not marathon-length.
    if (
        near_profile_max_hr
        and vo2_hard
        and mt <= 4200
        and (interval_structure or (z5_share_p is not None and z5_share_p >= 0.05))
        and (p_total >= MIN_ZONE_SECONDS or h_total >= MIN_ZONE_SECONDS)
    ):
        return "test_session", "near_max_hr_plus_vo2_pattern"

    vo2_interval = (
        (
            interval_structure
            and (
                vo2_hard
                or (z5_share_p is not None and z5_share_p >= 0.05)
                or (laps is not None and laps >= 8)
            )
        )
        or (
            laps is not None
            and laps >= 7
            and easy_share_p is not None
            and high_share_p is not None
            and easy_share_p >= 0.14
            and high_share_p >= 0.10
            and high_share_h is not None
            and high_share_h >= 0.14
        )
        or (
            z5_share_p is not None
            and high_share_p is not None
            and z5_share_p >= 0.055
            and high_share_p >= 0.13
        )
    )
    if vo2_interval:
        return "vo2_interval_session", "interval_structure_or_z5_spike"

    # Race-like: sustained hard block, little on/off interval signature, meaningful duration.
    if not interval_structure and mt >= 2700:
        dist_ok = run.distance_km is not None and float(run.distance_km) >= 9.5
        time_ok = mt >= 3300
        if dist_ok or time_ok:
            if p_total >= MIN_ZONE_SECONDS and high_share_p is not None:
                if high_share_p >= 0.19 and (easy_share_p is None or easy_share_p <= 0.32):
                    return "race", "sustained_power_z4plus"
            if h_total >= MIN_ZONE_SECONDS and high_share_h is not None:
                if high_share_h >= 0.24 and (z1_share_h is None or z1_share_h <= 0.14):
                    return "race", "sustained_hr_z4plus"

    return "test_or_vo2_session", "unspecified_hard"


def _moderate_non_vo2_guard(
    run: RunRecord,
    high_share_p: float | None,
    z5_share_p: float | None,
    high_share_h: float | None,
    laps: int | None,
) -> str | None:
    """
    Demote weakly-structured hard labels when top-end evidence is absent.

    Prevents moderate structured runs from being over-labeled as VO2 sessions.
    """
    try:
        avg_hr_f = float(run.avg_hr) if run.avg_hr is not None else None
    except (TypeError, ValueError):
        avg_hr_f = None
    try:
        max_hr_f = float(run.max_hr) if run.max_hr is not None else None
    except (TypeError, ValueError):
        max_hr_f = None
    try:
        avg_power_f = float(run.avg_power) if run.avg_power is not None else None
    except (TypeError, ValueError):
        avg_power_f = None

    weak_top_end = (
        (high_share_p is None or float(high_share_p) <= 0.11)
        and (z5_share_p is None or float(z5_share_p) < 0.03)
        and (high_share_h is None or float(high_share_h) <= 0.12)
    )
    moderate_effort = (
        (avg_hr_f is None or avg_hr_f <= 150)
        and (max_hr_f is None or max_hr_f <= 168)
        and (avg_power_f is None or avg_power_f < 235)
    )
    not_repeated_intervals = laps is None or laps < 7

    if weak_top_end and moderate_effort and not_repeated_intervals:
        return "steady_run"
    return None


def _classification_debug_enabled() -> bool:
    v = (os.getenv("AGENTICRUN_DEBUG_CLASSIFICATION") or "").strip().lower()
    return v in {"1", "true", "yes", "on"}


def _fmt_zone_summary(total: float, zones: list[float]) -> str:
    if total < MIN_ZONE_SECONDS:
        return "n/a"
    return "/".join(f"{(z / total):.2f}" for z in zones)


def _branch_diagnostics(
    run: RunRecord,
    moving_time: Any,
    easy_share_p: float | None,
    high_share_p: float | None,
    z3_share_p: float | None,
    z5_share_p: float | None,
    z1_share_h: float | None,
    high_share_h: float | None,
    laps: int | None,
) -> list[str]:
    interval_structure, vo2_hard = _vo2_interval_structure_flags(
        run, easy_share_p, high_share_p, z5_share_p, laps
    )
    threshold_zones = (
        moving_time
        and float(moving_time) >= 1800
        and z3_share_p is not None
        and z3_share_p >= 0.22
        and high_share_p is not None
        and high_share_p < 0.13
        and (run.avg_power is None or run.avg_power >= 185)
    )
    threshold_power = (
        run.avg_power
        and 225 <= run.avg_power < 255
        and moving_time
        and float(moving_time) >= 1800
    )
    long_match = False
    if run.distance_km and run.distance_km >= 14:
        if easy_share_p is not None:
            long_match = easy_share_p > 0.52 and (run.avg_power or 0) < 235
        elif (run.avg_power or 0) < 235:
            long_match = True
        elif (
            moving_time
            and float(moving_time) >= 5400
            and easy_share_p is not None
            and easy_share_p > 0.48
            and (run.avg_power or 0) < 240
        ):
            long_match = True
    easy_match = (
        (run.distance_km and run.distance_km <= 8 and easy_share_p is not None and easy_share_p > 0.68 and (run.avg_power or 0) < 215)
        or (run.distance_km and run.distance_km <= 8 and (run.avg_power or 0) < 205)
        or (run.avg_hr and run.avg_hr < 135)
        or (z1_share_h is not None and z1_share_h > 0.58)
    )
    return [
        f"branch.interval={'Y' if (vo2_hard or interval_structure) else 'N'}(vo2={int(bool(vo2_hard))},struct={int(bool(interval_structure))})",
        f"branch.threshold={'Y' if (threshold_zones or threshold_power) else 'N'}(zones={int(bool(threshold_zones))},power={int(bool(threshold_power))})",
        f"branch.long={'Y' if long_match else 'N'}",
        f"branch.easy_recovery={'Y' if easy_match else 'N'}",
    ]


def _easy_recovery_branch_match(
    run: RunRecord,
    easy_share_p: float | None,
    z1_share_h: float | None,
) -> bool:
    return bool(
        (run.distance_km and run.distance_km <= 8 and easy_share_p is not None and easy_share_p > 0.68 and (run.avg_power or 0) < 215)
        or (run.distance_km and run.distance_km <= 8 and (run.avg_power or 0) < 205)
        or (run.avg_hr and run.avg_hr < 135)
        or (z1_share_h is not None and z1_share_h > 0.58)
    )


def _easy_recovery_label_from_branch(
    run: RunRecord,
    z1_share_h: float | None,
) -> str:
    if (run.avg_hr and run.avg_hr < 135) or (z1_share_h is not None and z1_share_h > 0.58):
        return "recovery_run"
    return "easy_run"


def _is_generic_non_workout_title(title: str | None) -> bool:
    t = str(title or "").strip().lower()
    if not t:
        return True
    workout_cues = (
        "intervall",
        "interval",
        "vo2",
        "anaerob",
        "schwelle",
        "schwellenentwicklung",
        "tempo",
        "threshold",
        "ga1",
        "regenerativ",
        "recovery",
        "locker",
        "race",
        "wettkampf",
        "test",
        "berg",
        "hill",
    )
    if any(k in t for k in workout_cues):
        return False
    generic_exact = {
        "laufen",
        "run",
        "running",
        "jogging",
        "lauf",
        "easy run",
    }
    if t in generic_exact:
        return True
    # Keep this conservative: short single-token generic names are treated as non-workout.
    return (" " not in t) and (len(t) <= 10)


def _generic_title_easy_guard(
    run: RunRecord,
    fm: dict[str, Any],
    easy_share_p: float | None,
    high_share_p: float | None,
    z1_share_h: float | None,
    high_share_h: float | None,
    laps: int | None,
    candidate_type: str,
) -> str | None:
    """
    Prefer easy/recovery for generic-title runs unless interval evidence is truly strong.

    This is intentionally conservative and only applies to ambiguous or VO2-like candidates.
    """
    if candidate_type not in {"vo2_interval_session", "test_or_vo2_session", "mixed_unclear"}:
        return None
    if not _is_generic_non_workout_title(run.title):
        return None
    avg_moving_pace = fm.get("avg_moving_pace_sec_km")
    try:
        pace_f = (
            float(avg_moving_pace)
            if avg_moving_pace is not None
            else (float(run.avg_pace_sec_km) if run.avg_pace_sec_km is not None else None)
        )
    except (TypeError, ValueError):
        pace_f = None
    try:
        avg_power_f = float(run.avg_power) if run.avg_power is not None else None
    except (TypeError, ValueError):
        avg_power_f = None
    try:
        avg_hr_f = float(run.avg_hr) if run.avg_hr is not None else None
    except (TypeError, ValueError):
        avg_hr_f = None

    p_total, pz = _zone_seconds("power_zone", fm)
    z5_share_p = (pz[4] / p_total) if p_total >= MIN_ZONE_SECONDS else None

    strong_structured_interval = (
        (laps is not None and laps >= 8 and easy_share_p is not None and high_share_p is not None and easy_share_p >= 0.10 and high_share_p >= 0.14)
        or (high_share_p is not None and high_share_p >= 0.17)
        or (avg_power_f is not None and avg_power_f >= 250)
        or (high_share_h is not None and high_share_h >= 0.22 and avg_hr_f is not None and avg_hr_f >= 155)
        or (z5_share_p is not None and z5_share_p >= 0.055)
    )
    if strong_structured_interval:
        return None

    easy_zone_context = (
        (easy_share_p is not None and easy_share_p >= 0.55 and (high_share_p is None or high_share_p <= 0.11))
        or (z1_share_h is not None and z1_share_h >= 0.50 and (high_share_h is None or high_share_h <= 0.10))
    )
    easy_pace_context = pace_f is not None and pace_f >= 445
    easy_effort_context = (
        avg_power_f is not None
        and avg_power_f <= 230
        and (avg_hr_f is None or avg_hr_f <= 150 or (high_share_h is not None and high_share_h <= 0.12))
    )
    if not ((easy_zone_context or easy_pace_context) and easy_effort_context):
        return None
    return _easy_recovery_label_from_branch(run, z1_share_h)


def _generic_title_vo2_false_positive_guard(
    run: RunRecord,
    fm: dict[str, Any],
    high_share_p: float | None,
    z5_share_p: float | None,
) -> str | None:
    """
    Very narrow guard for generic-title runs that look interval-structured but not truly VO2-like.

    Keeps true VO2 intact by requiring both low top-end evidence and easy-ish overall effort.
    """
    if not _is_generic_non_workout_title(run.title):
        return None
    avg_moving_pace = fm.get("avg_moving_pace_sec_km")
    try:
        pace_f = (
            float(avg_moving_pace)
            if avg_moving_pace is not None
            else (float(run.avg_pace_sec_km) if run.avg_pace_sec_km is not None else None)
        )
    except (TypeError, ValueError):
        pace_f = None
    try:
        avg_power_f = float(run.avg_power) if run.avg_power is not None else None
    except (TypeError, ValueError):
        avg_power_f = None

    weak_vo2_top_end = (
        (z5_share_p is None or z5_share_p < 0.04)
        and (high_share_p is None or high_share_p < 0.40)
        and (avg_power_f is None or avg_power_f < 230)
    )
    easyish_context = pace_f is not None and pace_f >= 445
    if weak_vo2_top_end and easyish_context:
        return "easy_run"
    return None


def _easy_profile_guard(
    run: RunRecord,
    fm: dict[str, Any],
    moving_time: Any,
    easy_share_p: float | None,
    high_share_p: float | None,
    z1_share_h: float | None,
    high_share_h: float | None,
    laps: int | None,
    dq_f: float | None,
) -> str | None:
    """Conservative fallback when the full run profile looks easy despite a sharp label."""
    avg_moving_pace = fm.get("avg_moving_pace_sec_km")
    stopped = fm.get("stopped_time_sec")
    duration = run.duration_sec
    stopped_frac = None
    if stopped is not None and duration and float(duration) > 0:
        stopped_frac = float(stopped) / float(duration)

    low_zone_dominant = (
        (easy_share_p is not None and easy_share_p >= 0.62 and (high_share_p is None or high_share_p <= 0.10))
        or (z1_share_h is not None and z1_share_h >= 0.58 and (high_share_h is None or high_share_h <= 0.08))
    )
    hr_easy = run.avg_hr is not None and run.avg_hr < 140
    pace_slow = avg_moving_pace is not None and float(avg_moving_pace) >= 340
    power_not_sharp = run.avg_power is None or float(run.avg_power) < 225
    enough_time_for_evidence = moving_time is not None and float(moving_time) >= 1800
    fragmented = stopped_frac is not None and stopped_frac > HIGH_STOPPED_FRACTION
    garmin_power_trust = fm.get("power_zone_seconds_source") in GARMIN_FIT_ZONE_SECONDS_SOURCES
    suppress_hr_drift_sharp = (
        garmin_power_trust
        and high_share_p is not None
        and float(high_share_p) < 0.08
    )
    has_sharp_pattern = (
        (high_share_p is not None and float(high_share_p) >= 0.14)
        or (
            high_share_h is not None
            and float(high_share_h) >= 0.16
            and not suppress_hr_drift_sharp
        )
        or (run.avg_power is not None and float(run.avg_power) >= 245)
        or (laps is not None and laps >= 5)
    )
    weak_data = (
        (dq_f is not None and dq_f < LOW_DATA_QUALITY_THRESHOLD)
        or bool(fm.get("fit_parse_warnings"))
    )

    # Positive easy/recovery rule: prefer clear low-intensity profile when no sharp-workout pattern exists.
    if low_zone_dominant and (hr_easy or pace_slow or power_not_sharp) and not has_sharp_pattern:
        if fragmented and weak_data:
            return "mixed_unclear"
        if hr_easy and (pace_slow or power_not_sharp):
            return "recovery_run"
        return "easy_run"
    if fragmented and power_not_sharp and (high_share_p is None or high_share_p < 0.12):
        return "mixed_unclear"
    return None


def _threshold_evidence_is_strong(
    run: RunRecord,
    fm: dict[str, Any],
    moving_time: Any,
    z3_share_p: float | None,
    high_share_p: float | None,
    high_share_h: float | None,
    p_total: float,
    h_total: float,
    dq_f: float | None,
) -> bool:
    """Require sustained current-run evidence before allowing threshold classification."""
    if moving_time is None or float(moving_time) < 1800:
        return False
    if dq_f is not None and dq_f < SOFT_DATA_QUALITY_FOR_SHARP_TYPES:
        return False
    if fm.get("fit_parse_warnings"):
        return False

    power_threshold_pattern = (
        p_total >= MIN_ZONE_SECONDS
        and z3_share_p is not None
        and z3_share_p >= 0.24
        and high_share_p is not None
        and 0.03 <= high_share_p <= 0.14
        and run.avg_power is not None
        and 220 <= float(run.avg_power) < 255
    )
    hr_support = (
        h_total >= MIN_ZONE_SECONDS
        and high_share_h is not None
        and high_share_h <= 0.16
        and run.avg_hr is not None
        and float(run.avg_hr) >= 145
    )
    if not power_threshold_pattern:
        return False
    if fm.get("power_zone_seconds_source") in GARMIN_FIT_ZONE_SECONDS_SOURCES:
        if high_share_h is None or float(high_share_h) <= 0.18:
            return True
    return hr_support


def infer_training_type_from_fit(run: RunRecord, fm: dict[str, Any]) -> str:
    return infer_training_type_from_fit_with_trace(run, fm)[0]


def infer_training_type_from_fit_with_trace(run: RunRecord, fm: dict[str, Any]) -> tuple[str, list[str]]:
    trace: list[str] = []
    moving_time = fm.get("moving_time_sec")
    if moving_time is None:
        moving_time = run.duration_sec

    p_total, pz = _zone_seconds("power_zone", fm)
    h_total, hz = _zone_seconds("hr_zone", fm)
    z1p, z2p, z3p, z4p, z5p = pz
    z1h, z2h, z3h, z4h, z5h = hz

    easy_share_p = (z1p + z2p) / p_total if p_total >= MIN_ZONE_SECONDS else None
    high_share_p = (z4p + z5p) / p_total if p_total >= MIN_ZONE_SECONDS else None
    z3_share_p = z3p / p_total if p_total >= MIN_ZONE_SECONDS else None
    z5_share_p = z5p / p_total if p_total >= MIN_ZONE_SECONDS else None
    z1_share_h = z1h / h_total if h_total >= MIN_ZONE_SECONDS else None
    high_share_h = (z4h + z5h) / h_total if h_total >= MIN_ZONE_SECONDS else None

    laps = _lap_count_value(fm)
    dq = fm.get("data_quality_score")
    dq_f = float(dq) if dq is not None else None

    trace.append(
        f"metrics mt={moving_time}, pace={fm.get('avg_moving_pace_sec_km')}, hr={run.avg_hr}/{run.max_hr}, "
        f"power={run.avg_power}, laps={laps}, dq={dq_f}, fit_warn={int(bool(fm.get('fit_parse_warnings')))}"
    )
    trace.append(f"zones p={_fmt_zone_summary(p_total, pz)} h={_fmt_zone_summary(h_total, hz)}")
    trace.extend(
        _branch_diagnostics(
            run,
            moving_time,
            easy_share_p,
            high_share_p,
            z3_share_p,
            z5_share_p,
            z1_share_h,
            high_share_h,
            laps,
        )
    )

    t = _classify_training_type_core(
        run,
        moving_time,
        easy_share_p,
        high_share_p,
        z3_share_p,
        z5_share_p,
        z1_share_h,
        p_total,
        h_total,
        laps,
    )
    trace.append(f"core={t}")
    is_interval = t == "test_or_vo2_session"
    is_threshold = t == "threshold_run"
    is_long = t == "long_run"
    easy_recovery_branch = _easy_recovery_branch_match(run, easy_share_p, z1_share_h)

    # Hierarchy: interval/VO2 -> threshold -> long -> easy/recovery -> mixed fallback only.
    if is_interval:
        interval_structure, vo2_hard = _vo2_interval_structure_flags(
            run, easy_share_p, high_share_p, z5_share_p, laps
        )
        refined, why = _refine_hard_session_type(
            run,
            fm,
            moving_time,
            easy_share_p,
            high_share_p,
            z5_share_p,
            z1_share_h,
            high_share_h,
            p_total,
            h_total,
            laps,
            dq_f,
            interval_structure,
            vo2_hard,
        )
        if refined == "vo2_interval_session":
            fp_override = _generic_title_vo2_false_positive_guard(
                run, fm, high_share_p, z5_share_p
            )
            if fp_override == "easy_run":
                trace.append("generic_vo2_false_positive_guard=Y")
                trace.append("branch.mixed_fallback=N")
                trace.append("final=easy_run(reason=generic_vo2_false_positive_guard)")
                return "easy_run", trace
        generic_easy_override = _generic_title_easy_guard(
            run,
            fm,
            easy_share_p,
            high_share_p,
            z1_share_h,
            high_share_h,
            laps,
            refined,
        )
        if generic_easy_override in {"easy_run", "recovery_run"}:
            trace.append("generic_title_easy_guard=Y")
            trace.append("branch.mixed_fallback=N")
            trace.append(
                f"final={generic_easy_override}(reason=generic_title_easy_guard_interval)"
            )
            return generic_easy_override, trace
        if refined in {"vo2_interval_session", "test_or_vo2_session"}:
            moderate_non_vo2 = _moderate_non_vo2_guard(
                run, high_share_p, z5_share_p, high_share_h, laps
            )
            if moderate_non_vo2 is not None:
                trace.append("moderate_non_vo2_guard=Y")
                trace.append("branch.mixed_fallback=N")
                trace.append(
                    f"final={moderate_non_vo2}(reason=weak_top_end_non_vo2_profile)"
                )
                return moderate_non_vo2, trace
        trace.append("branch.mixed_fallback=N")
        trace.append(f"hard_refine={refined};why={why}")
        trace.append(f"final={refined}(reason=hard_session_refine:{why})")
        return refined, trace

    if is_threshold:
        # Guardrail: do not keep threshold unless this run itself has clear sustained evidence.
        if not _threshold_evidence_is_strong(
            run,
            fm,
            moving_time,
            z3_share_p,
            high_share_p,
            high_share_h,
            p_total,
            h_total,
            dq_f,
        ):
            trace.append("threshold_guard=failed")
            if easy_recovery_branch:
                promoted = _easy_recovery_label_from_branch(run, z1_share_h)
                trace.append("branch.mixed_fallback=N")
                trace.append(f"final={promoted}(reason=threshold_guard_failed_promoted_to_easy_recovery)")
                return promoted, trace
            fallback = _easy_profile_guard(
                run,
                fm,
                moving_time,
                easy_share_p,
                high_share_p,
                z1_share_h,
                high_share_h,
                laps,
                dq_f,
            )
            if fallback in {"easy_run", "recovery_run"}:
                trace.append("branch.mixed_fallback=N")
                trace.append(f"final={fallback}(reason=threshold_blocked_then_easy_profile)")
                return fallback, trace
            generic_easy_override = _generic_title_easy_guard(
                run,
                fm,
                easy_share_p,
                high_share_p,
                z1_share_h,
                high_share_h,
                laps,
                "mixed_unclear",
            )
            if generic_easy_override in {"easy_run", "recovery_run"}:
                trace.append("generic_title_easy_guard=Y")
                trace.append("branch.mixed_fallback=N")
                trace.append(
                    f"final={generic_easy_override}(reason=generic_title_easy_guard_threshold_failed)"
                )
                return generic_easy_override, trace
            trace.append("branch.mixed_fallback=Y")
            trace.append("final=mixed_unclear(reason=threshold_blocked_no_clear_easy)")
            return "mixed_unclear", trace
        trace.append("branch.mixed_fallback=N")
        trace.append("final=threshold_run(reason=threshold_guard_passed)")
        return "threshold_run", trace

    if is_long:
        trace.append("branch.mixed_fallback=N")
        trace.append("final=long_run(reason=long_branch_matched)")
        return "long_run", trace

    easy_override = _easy_profile_guard(
        run,
        fm,
        moving_time,
        easy_share_p,
        high_share_p,
        z1_share_h,
        high_share_h,
        laps,
        dq_f,
    )
    if easy_override in {"easy_run", "recovery_run"}:
        trace.append("branch.mixed_fallback=N")
        trace.append(f"final={easy_override}(reason=positive_easy_profile)")
        return easy_override, trace

    should_mixed = _should_mixed_unclear(
        run,
        t,
        easy_share_p,
        high_share_p,
        high_share_h,
        dq_f,
        laps,
        p_total,
        h_total,
    )
    trace.append(f"mixed_check={should_mixed}")
    if should_mixed:
        generic_easy_override = _generic_title_easy_guard(
            run,
            fm,
            easy_share_p,
            high_share_p,
            z1_share_h,
            high_share_h,
            laps,
            "mixed_unclear",
        )
        if generic_easy_override in {"easy_run", "recovery_run"}:
            trace.append("generic_title_easy_guard=Y")
            trace.append("branch.mixed_fallback=N")
            trace.append(
                f"final={generic_easy_override}(reason=generic_title_easy_guard_mixed)"
            )
            return generic_easy_override, trace
        trace.append("branch.mixed_fallback=Y")
        trace.append("final=mixed_unclear(reason=contradictory_or_weak_evidence)")
        return "mixed_unclear", trace

    trace.append("branch.mixed_fallback=N")
    trace.append(f"final={t}(reason=core_kept)")
    return t, trace


def _classify_training_type_core(
    run: RunRecord,
    moving_time: Any,
    easy_share_p: float | None,
    high_share_p: float | None,
    z3_share_p: float | None,
    z5_share_p: float | None,
    z1_share_h: float | None,
    p_total: float,
    h_total: float,
    laps: int | None,
) -> str:
    # Interval / VO2: repeated hard work (laps + on/off zones) or clear top-end time / power.
    interval_structure, vo2_hard = _vo2_interval_structure_flags(
        run, easy_share_p, high_share_p, z5_share_p, laps
    )
    if vo2_hard or interval_structure:
        return "test_or_vo2_session"

    # Threshold: sustained tempo — Z3-heavy with limited Z4+ time, or classic power band for long enough.
    thresh_zones = (
        moving_time
        and float(moving_time) >= 1800
        and z3_share_p is not None
        and z3_share_p >= 0.22
        and high_share_p is not None
        and high_share_p < 0.13
        and (run.avg_power is None or run.avg_power >= 185)
    )
    thresh_power = (
        run.avg_power
        and 225 <= run.avg_power < 255
        and moving_time
        and float(moving_time) >= 1800
    )
    if thresh_zones or thresh_power:
        return "threshold_run"

    # Long run: distance-led endurance; only after sharp-workout checks.
    if run.distance_km and run.distance_km >= 14:
        if easy_share_p is not None:
            if easy_share_p > 0.52 and (run.avg_power or 0) < 235:
                return "long_run"
        elif (run.avg_power or 0) < 235:
            return "long_run"
        else:
            # Easy-dominance failed but volume is still long — moving-time + zones as a softer check.
            if (
                moving_time
                and float(moving_time) >= 5400
                and easy_share_p is not None
                and easy_share_p > 0.48
                and (run.avg_power or 0) < 240
            ):
                return "long_run"

    # Easy: short session, easy power or clearly easy-dominant zones.
    if run.distance_km and run.distance_km <= 8:
        if easy_share_p is not None and easy_share_p > 0.68 and (run.avg_power or 0) < 215:
            return "easy_run"
        if (run.avg_power or 0) < 205:
            return "easy_run"

    # Recovery: low average HR or HR Z1-dominant distribution.
    if run.avg_hr and run.avg_hr < 135:
        return "recovery_run"
    if h_total >= MIN_ZONE_SECONDS and z1_share_h is not None and z1_share_h > 0.58:
        return "recovery_run"

    return "steady_run"


def _should_mixed_unclear(
    run: RunRecord,
    t: str,
    easy_share_p: float | None,
    high_share_p: float | None,
    high_share_h: float | None,
    dq_f: float | None,
    laps: int | None,
    p_total: float,
    h_total: float,
) -> bool:
    """Prefer a vague bucket when data quality or cross-stream agreement is weak."""
    if t == "mixed_unclear":
        return True

    # Blunt sharp types when the composite quality score is weak (threshold is slightly above "low" flag).
    if dq_f is not None and dq_f < SOFT_DATA_QUALITY_FOR_SHARP_TYPES:
        if t in {"threshold_run", "test_or_vo2_session"}:
            if not (run.avg_power and run.avg_power >= 270):
                return True

    # Lots of easy time and some hard time but few laps — warm-up + main set + cool-down without clear repeats.
    if (
        easy_share_p is not None
        and high_share_p is not None
        and easy_share_p > 0.33
        and 0.085 < high_share_p < 0.19
    ):
        if laps is None or laps < 4:
            if t == "steady_run":
                return True

    # Power vs HR zone story diverges strongly for the same session.
    if (
        p_total >= MIN_ZONE_SECONDS
        and h_total >= MIN_ZONE_SECONDS
        and high_share_p is not None
        and high_share_h is not None
        and high_share_p < 0.07
        and high_share_h > 0.16
        and t == "steady_run"
    ):
        return True

    return False


def _infer_intensity_from_averages(
    avg_power,
    avg_hr,
    fm: dict[str, Any] | None = None,
) -> str:
    ref_ftp = FIT_THRESHOLD_POWER_W
    ref_hr = FIT_HR_REFERENCE_BPM
    if fm:
        v = fm.get("resolved_functional_threshold_power")
        if v is not None:
            try:
                vf = float(v)
                if vf > 0:
                    ref_ftp = vf
            except (TypeError, ValueError):
                pass
        h = fm.get("resolved_threshold_heart_rate")
        if h is not None:
            try:
                hf = float(h)
                if hf > 0:
                    ref_hr = hf
            except (TypeError, ValueError):
                pass
    scale_p = ref_ftp / FIT_THRESHOLD_POWER_W if FIT_THRESHOLD_POWER_W else 1.0
    scale_h = ref_hr / FIT_HR_REFERENCE_BPM if FIT_HR_REFERENCE_BPM else 1.0
    if avg_power is not None:
        if avg_power >= 255 * scale_p:
            return "high"
        if avg_power >= 225 * scale_p:
            return "moderate_high"
        if avg_power >= 190 * scale_p:
            return "moderate"
        return "easy"
    if avg_hr is not None:
        if avg_hr >= 165 * scale_h:
            return "high"
        if avg_hr >= 150 * scale_h:
            return "moderate_high"
        if avg_hr >= 135 * scale_h:
            return "moderate"
        return "easy"
    return "unknown"


def _execution_quality_for_type(training_type: str, had_flags: bool) -> str:
    """Keep good/review surface; ambiguous type always reviews even without sensor flags."""
    if training_type == "mixed_unclear":
        return "review"
    return "good" if not had_flags else "review"


def _agenticrun_debug() -> bool:
    return os.getenv("AGENTICRUN_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _zone_interpretation_basis_label(fm: dict[str, Any]) -> str:
    ps = fm.get("power_zone_seconds_source")
    hs = fm.get("hr_zone_seconds_source")
    p_g = ps in GARMIN_FIT_ZONE_SECONDS_SOURCES
    h_g = hs in GARMIN_FIT_ZONE_SECONDS_SOURCES
    if p_g and h_g:
        return "Garmin FIT time-in-zone (power and HR)"
    if p_g:
        return "Garmin FIT time-in-zone (power)"
    if h_g:
        return "Garmin FIT time-in-zone (HR)"
    if ps is None and hs is None:
        return "no FIT zone-second sources"
    return "record stream or mixed sources"


class SessionAnalysisAgent:
    def run(self, state: RunState) -> RunState:
        run = state.run_record
        if not run:
            state.warnings.append("Session analysis skipped because no run record is present.")
            return state

        fm = session_fit_metrics(run)
        training_type, type_trace = infer_training_type_from_fit_with_trace(run, fm)
        intensity = infer_intensity_from_fit(run, fm)
        # Mixed type + a sharp intensity label over-claims; collapse toward the middle.
        if training_type == "mixed_unclear" and intensity in {"high", "moderate_high", "easy"}:
            intensity = "moderate"
        # Keep intensity consistent with conservative easy/recovery classification.
        if training_type in {"easy_run", "recovery_run"} and intensity in {"moderate_high", "high"}:
            intensity = "moderate" if training_type == "easy_run" else "easy"
        if training_type == "recovery_run" and intensity == "moderate":
            intensity = "easy"

        flags: list[str] = []

        if run.avg_hr and run.max_hr and run.avg_hr > run.max_hr:
            flags.append("avg_hr_above_max_hr_check_source")
        if run.distance_km and run.distance_km < 3:
            flags.append("short_session")
        if run.avg_power and run.avg_power > 280:
            flags.append("high_power_session")

        dq = fm.get("data_quality_score")
        if dq is not None and float(dq) < LOW_DATA_QUALITY_THRESHOLD:
            flags.append("low_data_quality")
        if fm.get("fit_parse_warnings"):
            flags.append("fit_parse_issues")

        duration_sec = run.duration_sec
        stopped = fm.get("stopped_time_sec")
        if (
            stopped is not None
            and duration_sec
            and duration_sec > 0
            and float(stopped) / float(duration_sec) > HIGH_STOPPED_FRACTION
        ):
            flags.append("high_stopped_time_fraction")

        if training_type == "mixed_unclear":
            flags.append("ambiguous_classification")

        had_flags = bool(flags)
        state.analysis.training_type = training_type
        state.analysis.intensity_label = intensity
        state.analysis.execution_quality = _execution_quality_for_type(training_type, had_flags)
        state.analysis.confidence = self._analysis_confidence(training_type, flags, fm)
        state.analysis.session_flags = flags
        state.analysis.classification_trace = " | ".join(type_trace)

        dq_note = ""
        if "low_data_quality" in flags and dq is not None:
            dq_note = f" Data quality score is low ({float(dq):.0f}/100), so this classification is tentative."
        elif "fit_parse_issues" in flags:
            dq_note = " FIT parsing reported gaps or missing fields; interpret metrics cautiously."

        sensor_note = ""
        if fm:
            laps_v = fm.get("lap_count")
            sensor_note = (
                f" Sensors present: power={as_bool_flag(fm.get('has_power'))}, hr={as_bool_flag(fm.get('has_hr'))}, "
                f"cadence={as_bool_flag(fm.get('has_cadence'))}, gps={as_bool_flag(fm.get('has_gps'))}."
                f" Laps: {laps_v if laps_v is not None else 'n/a'}."
            )
        state.analysis.summary = (
            f"Classified as {training_type} with {intensity} intensity.{dq_note} "
            f"Distance: {run.distance_km or 'n/a'} km, moving time: {fm.get('moving_time_sec') or 'n/a'} s, "
            f"avg moving pace: {format_pace_min_km(fm.get('avg_moving_pace_sec_km'))}, "
            f"stopped time (s): {fm.get('stopped_time_sec') or 'n/a'}, "
            f"avg HR: {run.avg_hr or 'n/a'}, avg power: {run.avg_power or 'n/a'}.{sensor_note}"
        )
        if _agenticrun_debug():
            state.analysis.summary += (
                f" Debug: primary zone basis for interpretation: {_zone_interpretation_basis_label(fm)}."
            )
            hard_note = next(
                (s for s in type_trace if s.startswith("hard_refine=")),
                None,
            )
            if hard_note:
                state.analysis.summary += f" Hard-session branch: {hard_note}."
        if _classification_debug_enabled():
            state.analysis.summary += f" Debug training-type trace: {' | '.join(type_trace)}."
        return state

    def _analysis_confidence(self, training_type: str, flags: list[str], fm: dict) -> float:
        base = 0.8 if training_type not in {"unknown", "mixed_unclear"} else 0.5
        if training_type == "mixed_unclear":
            base = min(base, 0.48)
        if "low_data_quality" in flags:
            base = min(base, 0.45)
        if "fit_parse_issues" in flags:
            base = min(base, 0.55)
        if not as_bool_flag(fm.get("has_power")) and not as_bool_flag(fm.get("has_hr")):
            if run_has_no_fit_streams(fm):
                pass
            else:
                base = min(base, 0.55)
        return max(0.25, base)


def run_has_no_fit_streams(fm: dict) -> bool:
    """True when FIT extras are absent (e.g. CSV import) — do not penalize confidence."""
    return not fm or all(fm.get(k) is None for k in ("data_quality_score", "moving_time_sec"))
