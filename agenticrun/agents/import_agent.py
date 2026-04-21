from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from fitparse import FitFile

from agenticrun.core.fit_activity_identity import derive_fit_activity_key
from agenticrun.core.fit_segment_extract import extract_run_segments_from_fit
from agenticrun.core.fit_zone_extract import extract_garmin_zone_snapshot_from_fit
from agenticrun.core.models import RunRecord, RunState
from agenticrun.utils.parsing import (
    clean_header,
    infer_date_from_filename,
    pace_from_distance_duration,
    parse_duration_to_seconds,
    parse_float,
    slugify_filename,
)


def _agenticrun_debug() -> bool:
    return os.getenv("AGENTICRUN_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _fit_introspect_debug_enabled() -> bool:
    """Broad FIT message/field dump (developer-only); does not affect Garmin zone extract logging."""
    return os.getenv("AGENTICRUN_FIT_INTROSPECT", "").lower() in {"1", "true", "yes", "on"}


def _fit_session_sport_is_running(session_data: dict[str, object]) -> bool:
    """True for running disciplines; excludes cycling so RPM-style cadence is not doubled."""
    sport_label = str(session_data.get("sport") or "").strip().lower().replace(" ", "_")
    sub_sport_label = str(session_data.get("sub_sport") or "").strip().lower().replace(" ", "_")
    combined = f"{sport_label}_{sub_sport_label}"
    if "cycl" in combined or "bike" in combined:
        return False
    return "run" in combined


class ImportAgent:
    """Reads Garmin CSV exports and extracts one run-level record per file."""

    def run(self, state: RunState) -> RunState:
        try:
            path = Path(state.source_path)
            if path.suffix.lower() == ".fit":
                run, bundle = self._build_run_record_from_fit(path)
                state.fit_stream_bundle = bundle
                if not run.raw_summary.get("session"):
                    state.warnings.append("FIT import: no session message found; metrics may be incomplete.")
            else:
                df = pd.read_csv(path)
                df.columns = [clean_header(c) for c in df.columns]

                source_type = self._detect_source_type(df)
                row = self._select_summary_row(df, source_type)
                run = self._build_run_record(row, path.name, source_type)
                state.fit_stream_bundle = None
            state.run_record = run
            state.status = "imported"
            return state
        except Exception as exc:
            state.status = "error"
            state.warnings.append(f"Import failed: {exc}")
            return state

    def _detect_source_type(self, df: pd.DataFrame) -> str:
        cols = set(df.columns)
        if "intervall" in cols or "abschnitttyp" in cols:
            return "garmin_interval_export"
        if any("übersicht" in str(v).lower() for v in df.astype(str).values.flatten()):
            return "garmin_lap_export"
        return "garmin_csv_generic"

    def _select_summary_row(self, df: pd.DataFrame, source_type: str) -> pd.Series:
        if source_type == "garmin_lap_export":
            for _, row in df.iterrows():
                if any("übersicht" in str(v).lower() for v in row.tolist()):
                    return row
            return df.iloc[-1]
        if source_type == "garmin_interval_export":
            # Prefer a summary-style row, else choose the row with the largest duration.
            for _, row in df.iterrows():
                if any("übersicht" in str(v).lower() for v in row.tolist()):
                    return row
            duration_candidates = []
            for i, row in df.iterrows():
                duration = None
                for col in ["zeit", "dauer", "time"]:
                    if col in df.columns:
                        duration = parse_duration_to_seconds(row.get(col))
                        if duration is not None:
                            break
                duration_candidates.append((i, duration or 0))
            best_idx = sorted(duration_candidates, key=lambda x: x[1], reverse=True)[0][0]
            return df.loc[best_idx]
        return df.iloc[-1]

    def _build_run_record(self, row: pd.Series, file_name: str, source_type: str) -> RunRecord:
        row_map = {str(k): row[k] for k in row.index}

        distance_raw = self._pick(row_map, [
            "distanz", "distanz km", "gesamtstrecke", "distance"
        ])
        duration_raw = self._pick(row_map, ["zeit", "dauer", "time"])
        avg_hr = parse_float(self._pick(row_map, ["durchschn. herzfrequenz", "durchschnittliche herzfrequenz", "ø herzfrequenz", "avg hr"]))
        max_hr = parse_float(self._pick(row_map, ["max. herzfrequenz", "maximale herzfrequenz", "maximaler puls", "max hr"]))
        avg_power = parse_float(self._pick(row_map, ["durchschn. leistung", "durchschnittliche leistung", "ø leistung", "avg power"]))
        max_power = parse_float(self._pick(row_map, ["max. leistung", "max power"]))    
        avg_cadence = parse_float(self._pick(row_map, ["durchschn. laufkadenz", "ø schrittfrequenz (laufen)", "kadenz", "avg cadence"]))
        elevation = parse_float(self._pick(row_map, ["anstieg gesamt", "anstieg gesamt m", "höhenmeter bergauf", "elevation gain"]))
        training_load = parse_float(self._pick(row_map, ["trainingsbelastung", "training load"]))

        distance_val = parse_float(distance_raw)
        if distance_val and distance_val > 100:
            distance_km = distance_val / 1000.0
        else:
            distance_km = distance_val
        duration_sec = parse_duration_to_seconds(duration_raw)
        pace = pace_from_distance_duration(distance_km, duration_sec)

        title = Path(file_name).stem.replace("_", " ").replace("-", " ")
        run_date = infer_date_from_filename(file_name)
        run_id = f"{run_date}_{slugify_filename(file_name)}"

        return RunRecord(
            run_id=run_id,
            source_file=file_name,
            source_type=source_type,
            run_date=run_date,
            title=title,
            distance_km=distance_km,
            duration_sec=duration_sec,
            avg_pace_sec_km=pace,
            avg_hr=avg_hr,
            max_hr=max_hr,
            avg_power=avg_power,
            max_power=max_power,
            avg_cadence=avg_cadence,
            elevation_gain_m=elevation,
            training_load=training_load,
            raw_summary=row_map,
        )

    def _build_run_record_from_fit(
        self, path: Path
    ) -> tuple[RunRecord, tuple[list[object], list[float | None], list[float | None]]]:
        fit = FitFile(str(path))
        fit.parse()
        seg_rows, seg_meta = extract_run_segments_from_fit(fit)
        if _fit_introspect_debug_enabled():
            from agenticrun.core.fit_introspect_debug import print_fit_introspection_debug

            print_fit_introspection_debug(fit, path.name)

        session_data: dict[str, object] = {}
        for msg in fit.get_messages("session"):
            session_data = {field.name: field.value for field in msg}
            if session_data:
                break

        activity_data: dict[str, object] = {}
        for msg in fit.get_messages("activity"):
            activity_data = {field.name: field.value for field in msg}
            if activity_data:
                break

        device_data: dict[str, object] = {}
        for msg in fit.get_messages("device_info"):
            device_data = {field.name: field.value for field in msg}
            if device_data:
                break

        file_id_data: dict[str, object] = {}
        for msg in fit.get_messages("file_id"):
            file_id_data = {field.name: field.value for field in msg}
            if file_id_data:
                break

        is_run_fit = _fit_session_sport_is_running(session_data)

        lap_count = 0
        lap_data: dict[str, object] = {}
        lap_max_speed_mps: float | None = None
        lap_max_cadence_val: float | None = None
        lap_has_power = False
        lap_has_hr = False
        lap_has_cadence = False
        record_count = 0
        power_available = False
        hr_available = False
        cadence_available = False
        for msg in fit.get_messages("lap"):
            lap_count += 1
            field_map = {field.name: field.value for field in msg}
            if not lap_data and field_map:
                lap_data = field_map
            speed_val = parse_float(field_map.get("enhanced_max_speed"))
            if speed_val is None:
                speed_val = parse_float(field_map.get("max_speed"))
            if speed_val is not None:
                lap_max_speed_mps = max(lap_max_speed_mps or speed_val, speed_val)
            cadence_val = parse_float(field_map.get("max_running_cadence"))
            if cadence_val is None:
                cadence_val = parse_float(field_map.get("max_cadence"))
            elif is_run_fit:
                # FIT *_running_cadence is strides/min; store steps/min for consistency with typical run metrics.
                cadence_val *= 2.0
            if cadence_val is not None:
                lap_max_cadence_val = max(lap_max_cadence_val or cadence_val, cadence_val)
            lap_has_power = lap_has_power or (parse_float(field_map.get("avg_power")) is not None or parse_float(field_map.get("max_power")) is not None)
            lap_has_hr = lap_has_hr or (parse_float(field_map.get("avg_heart_rate")) is not None or parse_float(field_map.get("max_heart_rate")) is not None)
            lap_has_cadence = lap_has_cadence or (
                parse_float(field_map.get("avg_running_cadence")) is not None
                or parse_float(field_map.get("avg_cadence")) is not None
                or parse_float(field_map.get("max_running_cadence")) is not None
                or parse_float(field_map.get("max_cadence")) is not None
            )

        record_max_speed_mps: float | None = None
        record_max_cadence_val: float | None = None
        record_timestamps: list[object] = []
        record_speeds_mps: list[float | None] = []
        record_powers_w: list[float | None] = []
        record_heart_rates_bpm: list[float | None] = []
        gps_available = False
        for msg in fit.get_messages("record"):
            record_count += 1
            field_map = {field.name: field.value for field in msg}
            if not power_available and field_map.get("power") is not None:
                power_available = True
            if not hr_available and field_map.get("heart_rate") is not None:
                hr_available = True
            if not cadence_available and (field_map.get("cadence") is not None or field_map.get("running_cadence") is not None):
                cadence_available = True
            if (
                not gps_available
                and field_map.get("position_lat") is not None
                and field_map.get("position_long") is not None
            ):
                gps_available = True
            speed_val = parse_float(field_map.get("enhanced_speed"))
            if speed_val is None:
                speed_val = parse_float(field_map.get("speed"))
            if speed_val is not None:
                record_max_speed_mps = max(record_max_speed_mps or speed_val, speed_val)
            record_timestamps.append(field_map.get("timestamp"))
            record_speeds_mps.append(speed_val)
            record_powers_w.append(parse_float(field_map.get("power")))
            record_heart_rates_bpm.append(parse_float(field_map.get("heart_rate")))
            cadence_val = parse_float(field_map.get("running_cadence"))
            if cadence_val is None:
                cadence_val = parse_float(field_map.get("cadence"))
            elif is_run_fit:
                # record.running_cadence is strides/min for running activities.
                cadence_val *= 2.0
            if cadence_val is not None:
                record_max_cadence_val = max(record_max_cadence_val or cadence_val, cadence_val)

        zone_snap = extract_garmin_zone_snapshot_from_fit(fit, fit_source_label=path.name)

        if _agenticrun_debug():
            print(f"--- FIT Garmin zone extract: {path.name} ---", flush=True)
            if not zone_snap:
                print(
                    "  garmin_hr_zones_found: no; garmin_power_zones_found: no; "
                    "threshold_hr: not found; ftp: not found "
                    "(no hr_zone/power_zone/zones_target, session/lap time_in_*_zone, or boundary developer fields)",
                    flush=True,
                )
            else:
                hr_b = zone_snap.get("hr_zone_boundaries") or []
                pw_b = zone_snap.get("power_zone_boundaries") or []
                hr_bh = zone_snap.get("hr_zone_boundaries_high_bpm") or []
                pw_bh = zone_snap.get("power_zone_boundaries_high_w") or []
                hr_t = zone_snap.get("hr_zone_time_sec") or []
                pw_t = zone_snap.get("power_zone_time_sec") or []
                thr = zone_snap.get("threshold_heart_rate")
                ftp = zone_snap.get("functional_threshold_power")
                src = zone_snap.get("zone_extract_sources") or {}
                gh = zone_snap.get("garmin_hr_zones_found")
                gp = zone_snap.get("garmin_power_zones_found")
                hr_found_s = ("yes" if hr_b else "no") if gh is None else ("yes" if gh else "no")
                pw_found_s = ("yes" if pw_b else "no") if gp is None else ("yes" if gp else "no")
                print(f"  garmin_hr_zones_found: {hr_found_s}", flush=True)
                print(f"  garmin_power_zones_found: {pw_found_s}", flush=True)
                print(f"  garmin_hr_time_in_zone_found: {'yes' if hr_t else 'no'}", flush=True)
                print(f"  garmin_power_time_in_zone_found: {'yes' if pw_t else 'no'}", flush=True)
                print(f"  hr_zone_boundaries_high_bpm (decoded_unknown_216 unknown_6): {hr_bh or hr_b}", flush=True)
                print(f"  power_zone_boundaries_high_w (decoded_unknown_216 unknown_9): {pw_bh or pw_b}", flush=True)
                print(f"  hr_zone_time_sec (source {src.get('hr_zone_time_sec', '?')}): {hr_t}", flush=True)
                print(f"  power_zone_time_sec (source {src.get('power_zone_time_sec', '?')}): {pw_t}", flush=True)
                zconv = zone_snap.get("garmin_mesg216_zone_time_conversion")
                zcf = zone_snap.get("garmin_mesg216_zone_time_sec_confirmed")
                zpv = zone_snap.get("garmin_mesg216_zone_time_sec_provisional")
                if zconv is not None or zcf is not None or zpv is not None:
                    print(f"  garmin_mesg216_zone_time_conversion: {zconv!r}", flush=True)
                    print(f"  garmin_mesg216_zone_time_sec_confirmed: {zcf!r}", flush=True)
                    print(f"  garmin_mesg216_zone_time_sec_provisional: {zpv!r}", flush=True)
                print(f"  threshold_hr: {thr if thr is not None else 'not found'}", flush=True)
                print(f"  ftp: {ftp if ftp is not None else 'not found'}", flush=True)
                mx = zone_snap.get("max_heart_rate")
                if mx is not None:
                    print(f"  max_hr: {mx}", flush=True)
                rhr = zone_snap.get("resting_heart_rate")
                if rhr is not None:
                    print(f"  resting_hr: {rhr}", flush=True)
                hct = zone_snap.get("hr_calc_type")
                pct = zone_snap.get("power_calc_type")
                if hct is not None:
                    print(f"  hr_calc_type: {hct}", flush=True)
                if pct is not None:
                    print(f"  power_calc_type: {pct}", flush=True)
                if src:
                    print(f"  zone_extract_sources (FIT message:field): {src}", flush=True)
                n216 = zone_snap.get("garmin_mesg216_decode_notes")
                if n216:
                    print(f"  garmin_mesg216_decode_notes: {n216}", flush=True)

        start_time = session_data.get("start_time") or activity_data.get("timestamp")
        if hasattr(start_time, "strftime"):
            run_date = start_time.strftime("%Y-%m-%d")
        else:
            run_date = infer_date_from_filename(path.name)

        def pick_fit_value(*keys: str):
            for key in keys:
                value = session_data.get(key)
                if value is not None:
                    return value
                value = activity_data.get(key)
                if value is not None:
                    return value
            return None

        def to_text(value: object) -> str | None:
            if value is None:
                return None
            text = str(value).strip()
            return text or None

        def to_text_non_numeric(value: object) -> str | None:
            text = to_text(value)
            if text is None:
                return None
            if isinstance(value, (int, float)):
                return None
            if text.isdigit():
                return None
            return text

        def mps_to_kmh(value: float | None) -> float | None:
            if value is None:
                return None
            return value * 3.6

        distance_m = parse_float(pick_fit_value("total_distance"))
        distance_km = (distance_m / 1000.0) if distance_m is not None else None

        duration_sec = parse_float(pick_fit_value("total_elapsed_time"))
        if duration_sec is None:
            duration_sec = parse_float(pick_fit_value("total_timer_time"))

        pace = pace_from_distance_duration(distance_km, duration_sec)

        title = self._fit_title(path, session_data, activity_data)

        calories = parse_float(pick_fit_value("total_calories"))
        if calories is None:
            calories = parse_float(lap_data.get("total_calories"))

        avg_speed_mps = parse_float(pick_fit_value("enhanced_avg_speed", "avg_speed"))
        if avg_speed_mps is None:
            avg_speed_mps = parse_float(lap_data.get("enhanced_avg_speed"))
        if avg_speed_mps is None:
            avg_speed_mps = parse_float(lap_data.get("avg_speed"))
        if avg_speed_mps is None and distance_m is not None and duration_sec:
            avg_speed_mps = distance_m / duration_sec
        avg_speed = mps_to_kmh(avg_speed_mps)

        max_speed_mps = parse_float(pick_fit_value("enhanced_max_speed", "max_speed"))
        if max_speed_mps is None:
            max_speed_mps = lap_max_speed_mps
        if max_speed_mps is None:
            max_speed_mps = record_max_speed_mps
        max_speed = mps_to_kmh(max_speed_mps)

        max_cadence_run = parse_float(pick_fit_value("max_running_cadence"))
        max_cadence_generic = parse_float(pick_fit_value("max_cadence"))
        if max_cadence_run is not None and is_run_fit:
            max_cadence_run *= 2.0  # strides/min → steps/min
        max_cadence = max_cadence_run if max_cadence_run is not None else max_cadence_generic
        if max_cadence is None:
            max_cadence = lap_max_cadence_val
        if max_cadence is None:
            max_cadence = record_max_cadence_val

        avg_run_cad = parse_float(pick_fit_value("avg_running_cadence"))
        avg_gen_cad = parse_float(pick_fit_value("avg_cadence"))
        if avg_run_cad is not None and is_run_fit:
            avg_run_cad *= 2.0  # strides/min → steps/min
        avg_cadence_out = avg_run_cad if avg_run_cad is not None else avg_gen_cad

        moving_time_sec = parse_float(pick_fit_value("total_moving_time"))
        if moving_time_sec is None:
            moving_time_total = 0.0
            prev_ts = None
            prev_speed = None
            for ts, speed in zip(record_timestamps, record_speeds_mps):
                if (
                    prev_ts is not None
                    and hasattr(ts, "timestamp")
                    and hasattr(prev_ts, "timestamp")
                ):
                    delta = ts.timestamp() - prev_ts.timestamp()
                    if delta > 0 and (speed or prev_speed):
                        if (speed and speed > 0) or (prev_speed and prev_speed > 0):
                            moving_time_total += delta
                prev_ts = ts
                prev_speed = speed
            if moving_time_total > 0:
                moving_time_sec = moving_time_total

        avg_moving_pace_sec_km = pace_from_distance_duration(distance_km, moving_time_sec)
        stopped_time_sec = None
        if duration_sec is not None and moving_time_sec is not None:
            stopped_time_sec = max(duration_sec - moving_time_sec, 0.0)

        sport = to_text(pick_fit_value("sport"))
        if sport is None:
            sport = to_text(lap_data.get("sport"))
        if sport is None:
            sport = "unknown"
        sub_sport = to_text(pick_fit_value("sub_sport"))
        if sub_sport is None:
            sub_sport = to_text(lap_data.get("sub_sport"))
        if sub_sport is None:
            sub_sport = "unspecified"

        # Only persist human-readable device labels; do not map numeric FIT ids.
        device_name = to_text_non_numeric(
            device_data.get("device_name")
            or device_data.get("product_name")
            or device_data.get("descriptor")
            or file_id_data.get("product_name")
        )
        if not power_available:
            power_available = parse_float(pick_fit_value("avg_power", "max_power")) is not None
        if not power_available:
            power_available = lap_has_power
        if not hr_available:
            hr_available = parse_float(pick_fit_value("avg_heart_rate", "max_heart_rate")) is not None
        if not hr_available:
            hr_available = lap_has_hr
        if not cadence_available:
            cadence_available = parse_float(pick_fit_value("avg_running_cadence", "avg_cadence", "max_running_cadence", "max_cadence")) is not None
        if not cadence_available:
            cadence_available = lap_has_cadence

        fit_parse_warning_items: list[str] = []
        if not session_data:
            fit_parse_warning_items.append("missing_session")
        if record_count == 0:
            fit_parse_warning_items.append("no_records")
        if moving_time_sec is None:
            fit_parse_warning_items.append("missing_moving_time")
        if distance_km is None:
            fit_parse_warning_items.append("missing_distance")
        fit_parse_warnings = ";".join(fit_parse_warning_items) if fit_parse_warning_items else None

        # Simple 0-100 data quality score from key stream availability and timing completeness.
        data_quality_score = (
            (25.0 if power_available else 0.0)
            + (25.0 if hr_available else 0.0)
            + (20.0 if cadence_available else 0.0)
            + (20.0 if gps_available else 0.0)
            + (10.0 if (duration_sec is not None and moving_time_sec is not None) else 0.0)
        )

        raw_summary: dict[str, object] = {
            "session": session_data,
            "activity": activity_data,
            "device_info": device_data,
            "file_id": file_id_data,
            "fit_run_segments": seg_rows,
            "fit_segment_extract_meta": seg_meta,
            "fit_session_metrics": {
                "calories": calories,
                "avg_speed": avg_speed,
                "max_speed": max_speed,
                "max_cadence": max_cadence,
                "moving_time_sec": moving_time_sec,
                "avg_moving_pace_sec_km": avg_moving_pace_sec_km,
                "power_zone_z1_sec": None,
                "power_zone_z2_sec": None,
                "power_zone_z3_sec": None,
                "power_zone_z4_sec": None,
                "power_zone_z5_sec": None,
                "hr_zone_z1_sec": None,
                "hr_zone_z2_sec": None,
                "hr_zone_z3_sec": None,
                "hr_zone_z4_sec": None,
                "hr_zone_z5_sec": None,
                "has_power": power_available,
                "has_hr": hr_available,
                "has_cadence": cadence_available,
                "has_gps": gps_available,
                "stopped_time_sec": stopped_time_sec,
                "data_quality_score": data_quality_score,
                "fit_parse_warnings": fit_parse_warnings,
                "speed_unit": "km_h",
                # FIT running cadence is normalized to steps/min; generic cadence remains source cadence per minute.
                "cadence_unit": "per_min",
                "sport": sport,
                "sub_sport": sub_sport,
                "device_name": device_name,
                "lap_count": int(parse_float(pick_fit_value("num_laps")) or lap_count),
                "power_available": power_available,
                "hr_available": hr_available,
                "cadence_available": cadence_available,
                "record_count": record_count,
            },
        }
        if zone_snap:
            raw_summary["fit_garmin_zone_extract"] = zone_snap

        fit_activity_key = derive_fit_activity_key(
            fit, session_data, activity_data, file_id_data
        )

        bundle = (record_timestamps, record_powers_w, record_heart_rates_bpm)
        run = RunRecord(
            run_id=f"{run_date}_{slugify_filename(path.name)}",
            source_file=path.name,
            source_type="garmin_fit",
            run_date=run_date,
            title=title,
            distance_km=distance_km,
            duration_sec=duration_sec,
            avg_pace_sec_km=pace,
            avg_hr=parse_float(pick_fit_value("avg_heart_rate")),
            max_hr=parse_float(pick_fit_value("max_heart_rate")),
            avg_power=parse_float(pick_fit_value("avg_power")),
            max_power=parse_float(pick_fit_value("max_power")),
            avg_cadence=avg_cadence_out,
            elevation_gain_m=parse_float(pick_fit_value("total_ascent")),
            training_load=parse_float(pick_fit_value("training_load")),
            fit_activity_key=fit_activity_key,
            raw_summary=raw_summary,
        )
        return run, bundle

    def _fit_title(self, path: Path, session_data: dict[str, object], activity_data: dict[str, object]) -> str:
        for key in ("sport", "sub_sport", "event", "event_type"):
            value = session_data.get(key) or activity_data.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text.replace("_", " ")
        return path.stem.replace("_", " ").replace("-", " ")

    def _pick(self, row_map: dict, aliases: list[str]):
        normalized = {clean_header(str(k)): v for k, v in row_map.items()}
        for alias in aliases:
            alias_clean = clean_header(alias)
            if alias_clean in normalized:
                return normalized[alias_clean]
            for key, value in normalized.items():
                if alias_clean in key or key in alias_clean:
                    return value
        return None
