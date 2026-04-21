from __future__ import annotations

from collections import Counter
from typing import Any

from fitparse import FitFile

from agenticrun.utils.parsing import parse_float


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _timer_sec(field_map: dict[str, Any]) -> float | None:
    """Lap/segment total_timer_time in seconds.

    FIT stores sub-second resolution with scale 1000; fitparse already applies
    scale/offset on decode, so ``field_map`` values are seconds — do not divide again.
    """
    return parse_float(field_map.get("total_timer_time"))


def _distance_m(field_map: dict[str, Any]) -> float | None:
    """Lap/segment total_distance in meters.

    FIT uses scale 100 on the wire; fitparse decodes to meters — do not divide again.
    """
    return parse_float(field_map.get("total_distance"))


def _avg_speed_m_s(field_map: dict[str, Any]) -> float | None:
    """Average speed in m/s from lap/segment_lap fields.

    FIT uses scale 1000 for avg_speed; fitparse decodes to m/s. If a value is
    still implausibly large for endurance laps, treat it as unscaled wire units.
    """
    raw = field_map.get("enhanced_avg_speed")
    if raw is None:
        raw = field_map.get("avg_speed")
    v = parse_float(raw)
    if v is None:
        return None
    if v > 25.0:
        return v / 1000.0
    return float(v)


def _pace_sec_per_km(duration_sec: float | None, distance_m: float | None) -> float | None:
    """Seconds per km from duration (s) and segment distance (m), i.e. duration / (distance_m / 1000)."""
    if duration_sec is None or distance_m is None or distance_m <= 0:
        return None
    km = distance_m / 1000.0
    if km <= 0:
        return None
    return duration_sec / km


def _comparison_metrics(field_map: dict[str, Any]) -> dict[str, Any]:
    duration_sec = _timer_sec(field_map)
    distance_m = _distance_m(field_map)
    return {
        "duration_sec": duration_sec,
        "distance_m": distance_m,
        "avg_hr": parse_float(field_map.get("avg_heart_rate")),
        "max_hr": parse_float(field_map.get("max_heart_rate")),
        "avg_power": parse_float(field_map.get("avg_power")),
        "avg_speed_m_s": _avg_speed_m_s(field_map),
        "avg_pace_sec_per_km": _pace_sec_per_km(duration_sec, distance_m),
    }


def _intensity_raw_is_four(intensity: Any) -> bool:
    """FIT SDK extends lap intensity beyond fitparse's 0–3 table; raw 4 is recovery-like on Garmin intervals."""
    if intensity is None:
        return False
    if intensity == 4:
        return True
    if isinstance(intensity, str) and intensity.strip() == "4":
        return True
    try:
        return int(intensity) == 4
    except (TypeError, ValueError):
        return False


def _map_lap_intensity(intensity: Any) -> str:
    if intensity is None:
        return "other"
    if _intensity_raw_is_four(intensity):
        return "recovery"
    label = str(intensity).strip().lower()
    if label == "warmup":
        return "warmup"
    if label == "cooldown":
        return "cooldown"
    if label == "rest":
        return "recovery"
    if label == "active":
        return "work"
    return "other"


def _compact_hist(counter: Counter[str]) -> str:
    if not counter:
        return "-"
    return ",".join(f"{k}:{v}" for k, v in sorted(counter.items(), key=lambda x: (-x[1], x[0])))


def extract_run_segments_from_fit(fit: FitFile) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Build per-row segment records from FIT lap and segment_lap messages (Garmin / FIT SDK).
    Returns (rows for DB, meta for compact logging).
    """
    rows: list[dict[str, Any]] = []
    lap_intensity_hist: Counter[str] = Counter()
    lap_trigger_hist: Counter[str] = Counter()
    seg_sport_event_hist: Counter[str] = Counter()
    seg_status_hist: Counter[str] = Counter()

    lap_message_count = 0
    segment_lap_message_count = 0
    idx = 0
    lap_map_parts: list[str] = []
    unit_verify_samples: list[dict[str, Any]] = []

    for lap_index, msg in enumerate(fit.get_messages("lap")):
        lap_message_count += 1
        fm = {f.name: f.value for f in msg}
        intensity = fm.get("intensity")
        ir = _as_str(intensity) or "null"
        lap_intensity_hist[ir] += 1
        lt = fm.get("lap_trigger")
        ltr = _as_str(lt) or "null"
        lap_trigger_hist[ltr] += 1
        ev = fm.get("event")
        et = fm.get("event_type")
        evs = _as_str(ev) or "-"
        ets = _as_str(et) or "-"
        mapped = _map_lap_intensity(intensity)
        lap_map_parts.append(f"{lap_index}|i={ir}|tr={ltr}|ev={evs}|et={ets}|m={mapped}")
        cm = _comparison_metrics(fm)
        dur = cm["duration_sec"]
        dist = cm["distance_m"]
        if mapped in ("work", "recovery") and len(unit_verify_samples) < 3:
            unit_verify_samples.append(
                {
                    "idx": idx,
                    "mapped_type": mapped,
                    "fit_total_timer_time_s": parse_float(fm.get("total_timer_time")),
                    "fit_total_distance_m": parse_float(fm.get("total_distance")),
                    "fit_avg_speed_m_s": _avg_speed_m_s(fm),
                    "persisted_duration_sec": cm["duration_sec"],
                    "persisted_distance_m": cm["distance_m"],
                    "persisted_avg_speed_m_s": cm["avg_speed_m_s"],
                    "persisted_pace_sec_per_km": cm["avg_pace_sec_per_km"],
                }
            )
        rows.append(
            {
                "idx": idx,
                "fit_source": "fit_lap",
                "intensity_raw": _as_str(intensity),
                "lap_trigger_raw": _as_str(lt),
                "wkt_step_index": _safe_int(fm.get("wkt_step_index")),
                "sport_event_raw": None,
                "segment_status_raw": None,
                "segment_name": None,
                "mapped_type": mapped,
                "total_timer_sec": dur,
                "total_distance_m": dist,
                "segment_index": idx,
                "segment_type_mapped": mapped,
                **cm,
            }
        )
        idx += 1

    for msg in fit.get_messages("segment_lap"):
        segment_lap_message_count += 1
        fm = {f.name: f.value for f in msg}
        se = fm.get("sport_event")
        ser = _as_str(se) or "null"
        seg_sport_event_hist[ser] += 1
        st = fm.get("status")
        ssr = _as_str(st) or "null"
        seg_status_hist[ssr] += 1
        name = fm.get("name")
        name_s = _as_str(name)
        mapped_seg = "other"
        cm = _comparison_metrics(fm)
        dur = cm["duration_sec"]
        dist = cm["distance_m"]
        rows.append(
            {
                "idx": idx,
                "fit_source": "fit_segment_lap",
                "intensity_raw": None,
                "lap_trigger_raw": None,
                "wkt_step_index": _safe_int(fm.get("wkt_step_index")),
                "sport_event_raw": _as_str(se),
                "segment_status_raw": _as_str(st),
                "segment_name": name_s,
                "mapped_type": mapped_seg,
                "total_timer_sec": dur,
                "total_distance_m": dist,
                "segment_index": idx,
                "segment_type_mapped": mapped_seg,
                **cm,
            }
        )
        idx += 1

    meta: dict[str, Any] = {
        "lap_message_count": lap_message_count,
        "segment_lap_message_count": segment_lap_message_count,
        "lap_intensity_hist": dict(lap_intensity_hist),
        "lap_trigger_hist": dict(lap_trigger_hist),
        "segment_sport_event_hist": dict(seg_sport_event_hist),
        "segment_status_hist": dict(seg_status_hist),
        "extract_source": "FIT lap + segment_lap messages",
        "lap_map_compact": ";".join(lap_map_parts),
        "raw_compact": (
            f"lap_i[{_compact_hist(lap_intensity_hist)}]|"
            f"lap_tr[{_compact_hist(lap_trigger_hist)}]|"
            f"seg_ev[{_compact_hist(seg_sport_event_hist)}]|"
            f"seg_st[{_compact_hist(seg_status_hist)}]"
        ),
        "unit_verify_samples": unit_verify_samples,
    }
    return rows, meta
