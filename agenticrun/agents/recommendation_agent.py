from __future__ import annotations

from datetime import date, datetime
from typing import Any

from agenticrun.core.models import RunState
from agenticrun.core.session_fit_metrics import (
    GARMIN_FIT_ZONE_SECONDS_SOURCES,
    session_fit_metrics,
)
from agenticrun.utils.parsing import format_pace_min_km

HARD_TRAINING_TYPES = frozenset(
    {
        "test_or_vo2_session",
        "vo2_interval_session",
        "test_session",
        "race",
        "threshold_run",
    }
)
HARD_INTENSITY = frozenset({"high", "moderate_high"})
# Prior runs only (excludes current); small window keeps rules transparent and robust to sparse history.
RECENT_PRIOR_WINDOW = 7

# --- Easy/recovery "drift" rule (deterministic; label-based) -----------------------------
# Count how many of the last N *prior* runs (see _prior_runs) are easy/recovery **and** have a
# session intensity_label in this set. If count >= EASY_RECOVERY_DRIFT_MIN_FLAGS, the
# RecommendationAgent emits the easy-drift next_session branch (before many other branches).
# This does NOT read power/HR zone seconds for that count — only stored training_type +
# intensity_label from session analysis.
EASY_RECOVERY_DRIFT_SLICE = 4
EASY_RECOVERY_DRIFT_MIN_FLAGS = 2
INTENSITY_LABEL_FLAGGED_EASY_DRIFT = frozenset({"moderate", "moderate_high", "high"})

# Primary coaching for these types should reflect the current aerobic session and recovery after it — not the
# easy/recovery drift counter (see RecommendationAgent.run ordering).
STEADY_LONG_TRAINING_TYPES = frozenset({"steady_run", "long_run"})

# Developer note (misclassification / false positives):
# - A run classified as easy_run/recovery_run with intensity_label "moderate" is always flagged.
#   True easy days sometimes receive "moderate" from the classifier (e.g. HR drift, terrain,
#   or conservative thresholds) even when the athlete executed as planned — so this rule can
#   over-trigger relative to coaching intent.
# - steady_run / long_run are never counted here; only rows with training_type easy_run or
#   recovery_run contribute to the counter.
# - _prior_runs() uses the last RECENT_PRIOR_WINDOW rows of *all* history rows except the
#   current run_id (ordered as in load_history). For a non-latest selected run, that window can
#   include chronologically later activities unless history is rebuilt per-date — same behavior
#   as ingest-time RecommendationAgent.


def _normalized_intensity_label(row_or_state) -> str:
    """Lowercase intensity for robust matching against stored labels."""
    if hasattr(row_or_state, "analysis"):
        v = getattr(row_or_state.analysis, "intensity_label", None)
    else:
        v = (row_or_state or {}).get("intensity_label") if isinstance(row_or_state, dict) else None
    if v is None:
        return ""
    return str(v).strip().lower()


def _prior_runs(history: list[dict] | None, current_run_id: str | None) -> list[dict]:
    if not history or not current_run_id:
        return []
    rows = [h for h in history if h.get("run_id") != current_run_id]
    return rows[-RECENT_PRIOR_WINDOW:]


def _parse_run_date(value: Any) -> date | None:
    if value is None:
        return None
    txt = str(value).strip()
    if not txt:
        return None
    try:
        return datetime.fromisoformat(txt).date()
    except ValueError:
        pass
    try:
        return date.fromisoformat(txt.split("T", 1)[0])
    except ValueError:
        return None


def _recent_prior_within_days(prior: list[dict], current_run_date: Any, days: int = 7) -> list[dict]:
    current_date = _parse_run_date(current_run_date)
    if current_date is None:
        # Fallback to the latest few prior runs when date parsing is unavailable.
        return prior[-4:]
    rows: list[dict] = []
    for row in prior:
        rd = _parse_run_date(row.get("run_date"))
        if rd is None:
            continue
        delta = (current_date - rd).days
        if 0 < delta <= days:
            rows.append(row)
    return rows


def _power_z45_fraction(fm: dict[str, Any]) -> float | None:
    total = sum(float(fm.get(f"power_zone_z{i}_sec") or 0) for i in range(1, 6))
    if total < 120:
        return None
    z45 = float(fm.get("power_zone_z4_sec") or 0) + float(fm.get("power_zone_z5_sec") or 0)
    return z45 / total


def _hr_z45_fraction(fm: dict[str, Any]) -> float | None:
    total = sum(float(fm.get(f"hr_zone_z{i}_sec") or 0) for i in range(1, 6))
    if total < 120:
        return None
    z45 = float(fm.get("hr_zone_z4_sec") or 0) + float(fm.get("hr_zone_z5_sec") or 0)
    return z45 / total


def _row_is_hard(row: dict) -> bool:
    """Hard if stored labels say so, or zone clocks show sustained upper-band time."""
    if row.get("training_type") in HARD_TRAINING_TYPES:
        return True
    if row.get("intensity_label") in HARD_INTENSITY:
        return True
    fm = session_fit_metrics(row)
    pf = _power_z45_fraction(fm)
    if pf is not None and pf > 0.13:
        return True
    hf = _hr_z45_fraction(fm)
    if hf is not None and hf > 0.15:
        return True
    return False


def _current_garmin_zone_seconds_backed(state: RunState) -> bool:
    if not state.run_record:
        return False
    fm = session_fit_metrics(state.run_record)
    ps = fm.get("power_zone_seconds_source")
    hs = fm.get("hr_zone_seconds_source")
    return (ps in GARMIN_FIT_ZONE_SECONDS_SOURCES) or (hs in GARMIN_FIT_ZONE_SECONDS_SOURCES)


def _current_is_hard(state: RunState) -> bool:
    if state.analysis.training_type in HARD_TRAINING_TYPES:
        return True
    if state.analysis.intensity_label in HARD_INTENSITY:
        return True
    if not state.run_record:
        return False
    fm = session_fit_metrics(state.run_record)
    pf = _power_z45_fraction(fm)
    if pf is not None and pf > 0.13:
        return True
    hf = _hr_z45_fraction(fm)
    if hf is not None and hf > 0.15:
        return True
    return False


def _easy_session_too_intense(state: RunState) -> bool:
    """Easy/recovery intent but averages/zones look harder than that prescription."""
    if state.analysis.training_type not in {"easy_run", "recovery_run"}:
        return False
    return _normalized_intensity_label(state) in INTENSITY_LABEL_FLAGGED_EASY_DRIFT


def _repeated_same_training_type(prior: list[dict], training_type: str, need: int = 3) -> bool:
    """Last N prior runs all the same non-trivial type — stimulus repetition without variety."""
    if training_type in {"easy_run", "recovery_run", "mixed_unclear", "unknown"}:
        return False
    if len(prior) < need:
        return False
    tail = prior[-need:]
    return sum(1 for r in tail if r.get("training_type") == training_type) >= need


def _row_easy_session_too_intense(row: dict) -> bool:
    if row.get("training_type") not in {"easy_run", "recovery_run"}:
        return False
    return _normalized_intensity_label(row) in INTENSITY_LABEL_FLAGGED_EASY_DRIFT


def _row_low_quality(row: dict) -> bool:
    dq = row.get("data_quality_score")
    if dq is not None and float(dq) < 60:
        return True
    if row.get("fit_parse_warnings"):
        return True
    fm = session_fit_metrics(row)
    flags: set[str] = set()
    if row.get("fit_parse_warnings"):
        flags.add("fit_parse_issues")
    return _evidence_is_weak(fm, flags)


def _row_upper_zone_heavy(row: dict) -> bool:
    fm = session_fit_metrics(row)
    pf = _power_z45_fraction(fm)
    hf = _hr_z45_fraction(fm)
    return (pf is not None and pf >= 0.18) or (hf is not None and hf >= 0.20)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _steady_long_comparable_signal(
    state: RunState, prior: list[dict]
) -> dict[str, Any]:
    """Comparable aerobic prior for steady/long primacy in coaching."""
    if not state.run_record:
        return {"available": False, "reason": "missing_current_run"}
    tt = (state.analysis.training_type or "").strip().lower()
    if tt not in STEADY_LONG_TRAINING_TYPES:
        return {"available": False, "reason": "not_steady_long"}

    pool = {"steady_run", "long_run", "easy_run", "recovery_run"}
    cur_pace = _as_float(state.run_record.avg_pace_sec_km)
    cur_hr = _as_float(state.run_record.avg_hr)
    cur_power = _as_float(state.run_record.avg_power)

    baseline: dict[str, Any] | None = None
    for row in reversed(prior):
        rtt = str(row.get("training_type") or "").strip().lower()
        if rtt not in pool:
            continue
        b_pace = _as_float(row.get("avg_pace_sec_km"))
        b_hr = _as_float(row.get("avg_hr"))
        b_power = _as_float(row.get("avg_power"))
        if any(v is not None for v in (b_pace, b_hr, b_power)):
            baseline = row
            break
    if baseline is None:
        return {"available": False, "reason": "no_comparable_aerobic_prior"}

    b_pace = _as_float(baseline.get("avg_pace_sec_km"))
    b_hr = _as_float(baseline.get("avg_hr"))
    b_power = _as_float(baseline.get("avg_power"))

    metrics: dict[str, Any] = {}
    direction_hits: dict[str, int] = {"better": 0, "stable": 0, "worse": 0}

    def pace_status(cur: float | None, base: float | None) -> dict[str, Any]:
        if cur is None or base is None:
            return {"delta": None, "status": "unknown"}
        d = cur - base
        if d <= -8.0:
            st = "better"
        elif d >= 8.0:
            st = "worse"
        else:
            st = "stable"
        direction_hits[st] += 1
        return {"delta": d, "status": st}

    def hr_status(cur: float | None, base: float | None) -> dict[str, Any]:
        if cur is None or base is None:
            return {"delta": None, "status": "unknown"}
        d = cur - base
        if d <= -3.0:
            st = "better"
        elif d >= 4.0:
            st = "worse"
        else:
            st = "stable"
        direction_hits[st] += 1
        return {"delta": d, "status": st}

    def power_status(cur: float | None, base: float | None) -> dict[str, Any]:
        if cur is None or base is None:
            return {"delta": None, "status": "unknown"}
        d = cur - base
        if d >= 8.0:
            st = "better"
        elif d <= -8.0:
            st = "worse"
        else:
            st = "stable"
        direction_hits[st] += 1
        return {"delta": d, "status": st}

    metrics["avg_pace_sec_km"] = pace_status(cur_pace, b_pace)
    metrics["avg_hr"] = hr_status(cur_hr, b_hr)
    metrics["avg_power"] = power_status(cur_power, b_power)

    if direction_hits["better"] >= 2:
        sig = "aerobic_progression"
    elif direction_hits["worse"] >= 2:
        sig = "aerobic_regression"
    elif direction_hits["stable"] >= 1:
        sig = "aerobic_consolidation"
    else:
        sig = "aerobic_normal_variation"

    return {
        "available": True,
        "signal": sig,
        "current_training_type": tt,
        "baseline_run": {
            "run_id": baseline.get("run_id"),
            "run_date": baseline.get("run_date"),
            "training_type": baseline.get("training_type"),
        },
        "metrics": metrics,
    }


def _stopped_fraction(fm: dict[str, Any]) -> float | None:
    moving = fm.get("moving_time_sec")
    stopped = fm.get("stopped_time_sec")
    if moving is None or stopped is None:
        return None
    total = float(moving) + float(stopped)
    if total <= 0:
        return None
    return float(stopped) / total


def _zone_totals(fm: dict[str, Any]) -> tuple[float, float]:
    power_total = sum(float(fm.get(f"power_zone_z{i}_sec") or 0) for i in range(1, 6))
    hr_total = sum(float(fm.get(f"hr_zone_z{i}_sec") or 0) for i in range(1, 6))
    return power_total, hr_total


def _evidence_is_weak(fm: dict[str, Any], flags: set[str]) -> bool:
    dq = fm.get("data_quality_score")
    if dq is not None and float(dq) < 60:
        return True
    if "low_data_quality" in flags or "fit_parse_issues" in flags:
        return True
    power_total, hr_total = _zone_totals(fm)
    has_zone_signal = power_total >= 120 or hr_total >= 120
    return not has_zone_signal and fm.get("moving_time_sec") is None


SIGNAL_SCHEMA_VERSION = 1


def _empty_recommendation_signals() -> dict[str, Any]:
    return {
        "schema_version": SIGNAL_SCHEMA_VERSION,
        "primary_run_read": "unknown",
        "recovery_need": "none",
        "load_action_candidate": "hold",
        "caution_signals": [],
        "secondary_flags": [],
        "recommendation_candidates": [],
        "dominant_rule_id": None,
        "triggered_heuristic_ids": [],
        "prioritization_hints": {
            "primary_for_llm_coaching": [],
            "secondary_context_only": [],
        },
    }


def _derive_primary_run_read(state: RunState, training_type: str) -> str:
    """Deterministic label for how to read *this* session (not prior-window heuristics)."""
    execution = state.analysis.execution_quality
    if execution == "review":
        return "execution_needs_review"
    if training_type == "long_run":
        return "long_aerobic_volume"
    if training_type == "steady_run":
        return "controlled_steady_aerobic"
    if training_type in {"easy_run", "recovery_run"}:
        return "easy_recovery_session"
    if training_type in HARD_TRAINING_TYPES:
        return "high_stimulus_session"
    if training_type == "threshold_run":
        return "threshold_quality_session"
    if training_type in {"test_or_vo2_session", "vo2_interval_session", "test_session"}:
        return "vo2_or_test_session"
    if training_type == "race":
        return "race_effort"
    if training_type == "mixed_unclear":
        return "mixed_or_unclear_structure"
    return "general_session"


def _recovery_need_for_branch(
    rule_id: str,
    load_action: str,
    *,
    fatigue: str,
    training_type: str,
) -> str:
    if fatigue in {"moderate", "high"}:
        return "easy_or_recovery_next"
    if rule_id == "easy_session_too_intense":
        return "true_easy_next"
    if rule_id == "easy_recovery_intensity_label_drift":
        return "true_easy_next"
    if rule_id in {"upper_zone_cluster", "upper_zone_garmin_backed", "repeated_high_exposure"}:
        return "space_high_intensity"
    if rule_id in {"hard_stack_today_hard", "hard_prior_cumulative"}:
        return "easy_before_next_quality"
    if rule_id in {"steady_long_aerobic", "steady_long_comparable_aerobic"}:
        if load_action == "reduce":
            return "easy_next_after_volume_or_fatigue"
        return "absorb_then_easy"
    if load_action == "slight_increase":
        return "progress_quality_slightly"
    return "none"


def _long_aerobic_primary_recommendation(state: RunState) -> tuple[str, str, bool]:
    """Default next-session line for steady/long when higher-priority branches did not match.

    Grounded in *this* session (execution, trend, fatigue/fitness), not the prior easy-drift counter.
    """
    training_type = state.analysis.training_type
    execution = state.analysis.execution_quality
    trend = state.trend.trend_label

    if execution == "review":
        return (
            "This aerobic session has execution items to review; keep the next outing easy, controlled, "
            "and short enough to stay truly regenerative.",
            "hold",
            True,
        )
    if trend == "possible_fatigue":
        return (
            "After this aerobic volume, prioritize simple easy recovery next and avoid stacking intensity "
            "until signals look clearer.",
            "reduce",
            True,
        )
    if training_type == "long_run":
        return (
            "Controlled long aerobic run; schedule the next session as easy or true recovery before the next "
            "demanding workout.",
            "hold",
            False,
        )
    return (
        "Controlled steady aerobic session; make the next run a true easy or recovery day to absorb today’s load.",
        "hold",
        False,
    )


class RecommendationAgent:
    def run(self, state: RunState, history: list[dict] | None = None) -> RunState:
        training_type = state.analysis.training_type
        fatigue = state.trend.fatigue_signal
        trend = state.trend.trend_label
        intensity = state.analysis.intensity_label
        execution = state.analysis.execution_quality
        flags = set(state.analysis.session_flags)
        fm = session_fit_metrics(state.run_record) if state.run_record else {}
        low_q = "low_data_quality" in flags or "fit_parse_issues" in flags
        weak_evidence = _evidence_is_weak(fm, flags)

        rid = state.run_record.run_id if state.run_record else None
        prior = _prior_runs(history, rid)
        current_run_date = state.run_record.run_date if state.run_record else None
        recent_7d = _recent_prior_within_days(prior, current_run_date, days=7)
        close_window = recent_7d if recent_7d else prior[-4:]
        hard_prior = sum(1 for r in prior if _row_is_hard(r))
        current_hard = _current_is_hard(state)
        easy_intense = _easy_session_too_intense(state)
        easy_too_hard_recent = sum(
            1 for r in prior[-EASY_RECOVERY_DRIFT_SLICE:] if _row_easy_session_too_intense(r)
        )
        upper_zone_close_count = sum(1 for r in close_window if _row_upper_zone_heavy(r))
        low_quality_recent_count = sum(1 for r in prior[-5:] if _row_low_quality(r))
        low_quality_recent_window = min(len(prior), 5)
        repeated_high_exposure = sum(1 for r in prior[-4:] if _row_is_hard(r)) + (1 if current_hard else 0)
        same_streak = _repeated_same_training_type(prior, training_type)
        stopped_frac = _stopped_fraction(fm)
        power_z45 = _power_z45_fraction(fm)
        hr_z45 = _hr_z45_fraction(fm)
        fit_parse_warnings = fm.get("fit_parse_warnings")

        signals = _empty_recommendation_signals()
        signals["primary_run_read"] = _derive_primary_run_read(state, training_type)
        comparable_signal = _steady_long_comparable_signal(state, prior)
        if comparable_signal.get("available"):
            signals["comparable_aerobic_signal"] = comparable_signal
            signals["recommendation_candidates"].append(
                {
                    "rule_id": "steady_long_comparable_aerobic",
                    "tier": "primary",
                    "signal": comparable_signal.get("signal"),
                    "baseline_run": comparable_signal.get("baseline_run"),
                    "note": (
                        "For steady/long coaching, anchor top-line narrative in this run vs comparable aerobic prior."
                    ),
                }
            )
        dominant_rule_id = "default_continue_planned"

        drift_secondary_applicable = (
            training_type in STEADY_LONG_TRAINING_TYPES
            and easy_too_hard_recent >= EASY_RECOVERY_DRIFT_MIN_FLAGS
        )
        if drift_secondary_applicable:
            detail = (
                f"{easy_too_hard_recent} prior easy/recovery rows in the last {EASY_RECOVERY_DRIFT_SLICE} "
                f"prior slots with intensity labels in {sorted(INTENSITY_LABEL_FLAGGED_EASY_DRIFT)} "
                "(label-only counter; excludes steady/long from the count)"
            )
            signals["caution_signals"].append(
                {
                    "id": "recent_easy_recovery_sessions_read_more_demanding",
                    "priority": "secondary",
                    "detail": detail,
                }
            )
            signals["secondary_flags"].append("easy_recovery_intensity_label_drift")
            signals["triggered_heuristic_ids"].append("easy_recovery_intensity_label_drift")
            signals["recommendation_candidates"].append(
                {
                    "rule_id": "easy_recovery_intensity_label_drift",
                    "tier": "secondary",
                    "next_session_hint": (
                        "On upcoming easy/recovery days, keep effort clearly easy (stricter pace/HR caps)."
                    ),
                    "load_action": "hold",
                    "note": (
                        "Secondary context for steady/long: prior easy/recovery rows often labeled moderate+; "
                        "do not override primary read of this aerobic session unless it fits the story."
                    ),
                }
            )

        if fatigue in {"moderate", "high"}:
            dominant_rule_id = "fatigue_elevated"
            state.recommendation.next_session = "Schedule an easy run or recovery day next."
            state.recommendation.load_action = "reduce"
            state.recommendation.warning_flag = True
        elif low_q:
            dominant_rule_id = "data_quality_low"
            state.recommendation.next_session = (
                "Prioritize a clean recording next (HR/power/GPS as available) before pushing intensity; "
                "keep the next run easy to control load."
            )
            state.recommendation.load_action = "hold"
            state.recommendation.warning_flag = True
        elif easy_intense:
            dominant_rule_id = "easy_session_too_intense"
            state.recommendation.next_session = (
                "Easy/recovery intent but effort read moderate-or-hard; keep the next session truly easy "
                "and shorter if you need to guard load."
            )
            state.recommendation.load_action = "hold"
            state.recommendation.warning_flag = True
        elif (
            easy_too_hard_recent >= EASY_RECOVERY_DRIFT_MIN_FLAGS
            and training_type not in STEADY_LONG_TRAINING_TYPES
        ):
            dominant_rule_id = "easy_recovery_intensity_label_drift"
            state.recommendation.next_session = (
                "Recent easy/recovery sessions were often executed too hard; on the next easy or recovery day "
                "keep it clearly easy and use stricter pace/HR caps."
            )
            state.recommendation.load_action = "reduce"
            state.recommendation.warning_flag = True
        elif (
            training_type in STEADY_LONG_TRAINING_TYPES
            and comparable_signal.get("available")
        ):
            dominant_rule_id = "steady_long_comparable_aerobic"
            ns, la, wf = _long_aerobic_primary_recommendation(state)
            state.recommendation.next_session = ns
            state.recommendation.load_action = la
            state.recommendation.warning_flag = wf
        elif (
            upper_zone_close_count >= 3
            and not (training_type in STEADY_LONG_TRAINING_TYPES and comparable_signal.get("available"))
        ):
            dominant_rule_id = "upper_zone_cluster"
            state.recommendation.next_session = (
                "Too many upper-zone sessions clustered recently; schedule easy aerobic work next and space quality sessions further apart."
            )
            state.recommendation.load_action = "reduce"
            state.recommendation.warning_flag = True
        elif (
            upper_zone_close_count >= 2
            and _current_garmin_zone_seconds_backed(state)
            and (hard_prior >= 1 or current_hard)
            and not (training_type in STEADY_LONG_TRAINING_TYPES and comparable_signal.get("available"))
        ):
            dominant_rule_id = "upper_zone_garmin_backed"
            state.recommendation.next_session = (
                "Recent runs show meaningful upper-zone time on reliable Garmin FIT zone clocks; "
                "take an easy aerobic day before adding more high-intensity work."
            )
            state.recommendation.load_action = "reduce"
            state.recommendation.warning_flag = True
        elif repeated_high_exposure >= 3:
            dominant_rule_id = "repeated_high_exposure"
            state.recommendation.next_session = (
                "Recent runs show repeated high-intensity exposure; keep the next session easy before adding another demanding workout."
            )
            state.recommendation.load_action = "reduce"
            state.recommendation.warning_flag = True
        elif hard_prior >= 2 and current_hard:
            dominant_rule_id = "hard_stack_today_hard"
            quality_hint = "easy aerobic run (30-50 min)" if fm.get("moving_time_sec") else "easy aerobic run"
            state.recommendation.next_session = f"Several recent sessions were demanding and today adds intensity; insert a {quality_hint} before the next quality workout."
            state.recommendation.load_action = "reduce"
            state.recommendation.warning_flag = True
        elif low_quality_recent_window >= 3 and low_quality_recent_count >= 2:
            dominant_rule_id = "data_quality_recent_cluster"
            state.recommendation.next_session = (
                "Recent data quality is inconsistent, so confidence is lower; keep the next session controlled and prioritize cleaner recording."
            )
            state.recommendation.load_action = "hold"
            state.recommendation.warning_flag = True
        elif hard_prior >= 3:
            dominant_rule_id = "hard_prior_cumulative"
            state.recommendation.next_session = (
                "Multiple hard sessions already in the last few runs; bias the next outings easy until the pattern lightens."
            )
            state.recommendation.load_action = "reduce"
            state.recommendation.warning_flag = True
        elif (
            trend == "positive_progress"
            and training_type in {"threshold_run", "steady_run"}
            and not weak_evidence
            and fatigue not in {"moderate", "high"}
        ):
            dominant_rule_id = "positive_progress_slight_increase"
            state.recommendation.next_session = "Keep the next quality session, but increase load only slightly."
            state.recommendation.load_action = "slight_increase"
            state.recommendation.warning_flag = False
        elif training_type in {
            "test_or_vo2_session",
            "vo2_interval_session",
            "test_session",
            "race",
        }:
            dominant_rule_id = "post_race_or_vo2"
            state.recommendation.next_session = (
                "Use the next session for recovery or controlled easy mileage; avoid back-to-back high-intensity work."
            )
            state.recommendation.load_action = "reduce"
            state.recommendation.warning_flag = False
        elif training_type == "mixed_unclear":
            dominant_rule_id = "mixed_unclear"
            state.recommendation.next_session = (
                "Session structure was ambiguous; keep the next run easy and repeat a clearer quality session later in the week."
            )
            state.recommendation.load_action = "hold"
            state.recommendation.warning_flag = False
        elif same_streak and training_type in {
            "threshold_run",
            "test_or_vo2_session",
            "vo2_interval_session",
            "test_session",
            "race",
            "steady_run",
        }:
            dominant_rule_id = "stimulus_streak"
            state.recommendation.next_session = (
                "Recent runs match this stimulus repeatedly; swap the next similar slot for easy or long aerobic volume."
            )
            state.recommendation.load_action = "hold"
            state.recommendation.warning_flag = False
        elif training_type in STEADY_LONG_TRAINING_TYPES:
            dominant_rule_id = "steady_long_aerobic"
            ns, la, wf = _long_aerobic_primary_recommendation(state)
            state.recommendation.next_session = ns
            state.recommendation.load_action = la
            state.recommendation.warning_flag = wf
        else:
            dominant_rule_id = "default_continue_planned"
            state.recommendation.next_session = (
                "Continue with the planned week, but keep the next session controlled and avoid adding extra intensity."
            )
            state.recommendation.load_action = "hold" if not weak_evidence else "reduce"
            state.recommendation.warning_flag = False

        signals["dominant_rule_id"] = dominant_rule_id
        signals["load_action_candidate"] = state.recommendation.load_action
        signals["recovery_need"] = _recovery_need_for_branch(
            dominant_rule_id,
            state.recommendation.load_action,
            fatigue=fatigue,
            training_type=training_type,
        )
        signals["recommendation_candidates"].insert(
            0,
            {
                "rule_id": dominant_rule_id,
                "tier": "primary",
                "next_session_hint": state.recommendation.next_session,
                "load_action": state.recommendation.load_action,
                "warning_flag": state.recommendation.warning_flag,
            },
        )
        if (
            training_type in STEADY_LONG_TRAINING_TYPES
            and comparable_signal.get("available")
        ):
            if upper_zone_close_count >= 3:
                signals["caution_signals"].append(
                    {
                        "id": "upper_zone_cluster_recent",
                        "priority": "secondary",
                        "detail": (
                            f"{upper_zone_close_count} upper-zone-heavy runs in close window; "
                            "keep as supporting load-spacing context for steady/long."
                        ),
                    }
                )
                signals["secondary_flags"].append("upper_zone_cluster")
                signals["triggered_heuristic_ids"].append("upper_zone_cluster")
            if (
                upper_zone_close_count >= 2
                and _current_garmin_zone_seconds_backed(state)
                and (hard_prior >= 1 or current_hard)
            ):
                signals["caution_signals"].append(
                    {
                        "id": "upper_zone_garmin_backed_recent",
                        "priority": "secondary",
                        "detail": (
                            "Recent Garmin-backed upper-zone exposure exists; use as secondary spacing context."
                        ),
                    }
                )
                signals["secondary_flags"].append("upper_zone_garmin_backed")
                signals["triggered_heuristic_ids"].append("upper_zone_garmin_backed")
            if repeated_high_exposure >= 3:
                signals["caution_signals"].append(
                    {
                        "id": "repeated_high_exposure_recent",
                        "priority": "secondary",
                        "detail": (
                            f"{repeated_high_exposure} hard exposures in recent window; "
                            "keep as secondary load-spacing context for steady/long."
                        ),
                    }
                )
                signals["secondary_flags"].append("repeated_high_exposure")
                signals["triggered_heuristic_ids"].append("repeated_high_exposure")
                signals["recommendation_candidates"].append(
                    {
                        "rule_id": "repeated_high_exposure",
                        "tier": "secondary",
                        "next_session_hint": (
                            "Keep the next session easy before adding another demanding workout."
                        ),
                        "load_action": "reduce",
                        "note": (
                            "Secondary caution for steady/long when comparable aerobic signal is available."
                        ),
                    }
                )

        pri = [signals["primary_run_read"]]
        if (
            training_type in STEADY_LONG_TRAINING_TYPES
            and comparable_signal.get("available")
        ):
            pri.append(f"comparable_{comparable_signal.get('signal')}")
        pri.append(dominant_rule_id)
        sec_only = [c["id"] for c in signals["caution_signals"] if c.get("priority") == "secondary"]
        signals["prioritization_hints"]["primary_for_llm_coaching"] = pri
        signals["prioritization_hints"]["secondary_context_only"] = sec_only
        state.recommendation.recommendation_signals = signals

        extras: list[str] = []
        if "low_data_quality" in flags:
            dq = fm.get("data_quality_score")
            if dq is not None:
                extras.append(f"low data quality ({float(dq):.0f}/100)")
            else:
                extras.append("low data quality")
        if "fit_parse_issues" in flags:
            extras.append("FIT parse warnings")
        if fit_parse_warnings:
            extras.append(f"fit parse detail: {fit_parse_warnings}")
        if trend == "uncertain_data_quality":
            extras.append("trend suppressed due to data quality")
        if prior:
            extras.append(f"hard sessions in last {len(prior)} prior runs: {hard_prior}")
            extras.append(f"upper-zone-heavy sessions in close window: {upper_zone_close_count}")
            extras.append(f"easy/recovery sessions executed too hard (recent): {easy_too_hard_recent}")
            if (
                training_type in STEADY_LONG_TRAINING_TYPES
                and easy_too_hard_recent >= EASY_RECOVERY_DRIFT_MIN_FLAGS
            ):
                if easy_too_hard_recent >= 3:
                    extras.append(
                        "secondary (not top-line for steady/long): easy/recovery drift window is strong "
                        f"({easy_too_hard_recent} label-flags in slice) — consider tightening easy days; "
                        "primary next_session reflects this aerobic session instead"
                    )
                else:
                    extras.append(
                        "secondary (not top-line for steady/long): prior easy/recovery rows in drift window "
                        f"often labeled moderate+ ({easy_too_hard_recent} in slice); see diagnostics if needed"
                    )
            if low_quality_recent_window:
                extras.append(
                    f"low-quality runs in last {low_quality_recent_window}: {low_quality_recent_count}"
                )
        if fm.get("moving_time_sec") is not None:
            extras.append(f"moving time {float(fm['moving_time_sec']):.0f}s")
        if fm.get("avg_moving_pace_sec_km") is not None:
            extras.append(f"avg moving pace {format_pace_min_km(fm['avg_moving_pace_sec_km'])}")
        if stopped_frac is not None and stopped_frac > 0.22:
            extras.append(f"high stopped fraction ({stopped_frac:.0%})")
        if power_z45 is not None:
            extras.append(f"power Z4+Z5 {power_z45:.0%}")
        if hr_z45 is not None:
            extras.append(f"HR Z4+Z5 {hr_z45:.0%}")
        if weak_evidence:
            extras.append("evidence weak -> conservative recommendation")
        extras.append(f"intensity {intensity}, execution {execution}")

        extra_txt = f" Notes: {', '.join(extras)}." if extras else ""
        state.recommendation.recommendation_summary = (
            f"Recommendation based on training type '{training_type}', trend '{trend}', "
            f"and fatigue signal '{fatigue}'. "
            f"Structured signals: dominant_rule={dominant_rule_id}, primary_run_read={signals['primary_run_read']}, "
            f"recovery_need={signals['recovery_need']}.{extra_txt}"
        )
        return state


def diagnose_easy_recovery_drift_rule(
    history: list[dict[str, Any]] | None,
    current_run_id: str | None,
) -> dict[str, Any]:
    """Deterministic inspectability for the easy/recovery drift counter (`easy_too_hard_recent`).

    Mirrors :meth:`RecommendationAgent.run` inputs: ``history`` from :func:`load_history` and the
    selected activity's ``run_id``. The drift rule counts rows in ``prior[-EASY_RECOVERY_DRIFT_SLICE:]``
    where ``training_type`` is ``easy_run``/``recovery_run`` and normalized ``intensity_label`` is in
    :data:`INTENSITY_LABEL_FLAGGED_EASY_DRIFT`. Fires when count >= :data:`EASY_RECOVERY_DRIFT_MIN_FLAGS`
    (if no higher-priority branch applies in the agent).

    Returns contributing runs, thresholds, and developer notes on likely misclassification.
    """
    rid = str(current_run_id or "").strip() or None
    prior = _prior_runs(history, rid)
    ev_slice = prior[-EASY_RECOVERY_DRIFT_SLICE:] if prior else []
    rows_out: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []
    for r in ev_slice:
        tt = str(r.get("training_type") or "").strip().lower() or "—"
        il = _normalized_intensity_label(r) or "—"
        is_easy = tt in {"easy_run", "recovery_run"}
        flag = bool(is_easy and il in INTENSITY_LABEL_FLAGGED_EASY_DRIFT)
        entry: dict[str, Any] = {
            "run_id": r.get("run_id"),
            "run_date": r.get("run_date"),
            "training_type": r.get("training_type"),
            "intensity_label_raw": r.get("intensity_label"),
            "intensity_label_normalized": il if il != "—" else None,
            "counts_for_rule": is_easy,
            "flagged_too_hard_by_rule": flag,
            "criterion_if_flagged": (
                f"training_type in {{easy_run, recovery_run}} AND intensity_label ∈ "
                f"{sorted(INTENSITY_LABEL_FLAGGED_EASY_DRIFT)} (label-only check)"
                if flag
                else None
            ),
            "observed_snapshot": {
                "avg_hr": r.get("avg_hr"),
                "avg_power": r.get("avg_power"),
                "avg_pace_sec_km": r.get("avg_pace_sec_km"),
            },
            "why": (
                "Counted: easy/recovery type and session intensity label in flagged set (no zone fractions)."
                if flag
                else (
                    f"Not counted toward drift: training_type is {tt!r} (need easy_run or recovery_run)."
                    if not is_easy
                    else f"Not flagged: intensity_label {il!r} is outside flagged set."
                )
            ),
        }
        rows_out.append(entry)
        if flag:
            flagged.append(entry)

    n_flag = sum(1 for e in rows_out if e.get("flagged_too_hard_by_rule"))
    fires = n_flag >= EASY_RECOVERY_DRIFT_MIN_FLAGS

    return {
        "rule_id": "easy_recovery_intensity_label_drift",
        "code_reference": "RecommendationAgent.run → easy_too_hard_recent / elif easy_too_hard_recent >= "
        f"{EASY_RECOVERY_DRIFT_MIN_FLAGS}",
        "how_it_works": (
            f"From prior = last {RECENT_PRIOR_WINDOW} history rows excluding run_id, take "
            f"prior[-{EASY_RECOVERY_DRIFT_SLICE}:]. "
            "For each row, if training_type is easy_run or recovery_run and intensity_label (case-insensitive) "
            f"is one of {sorted(INTENSITY_LABEL_FLAGGED_EASY_DRIFT)}, increment the counter. "
            f"If counter >= {EASY_RECOVERY_DRIFT_MIN_FLAGS}, the easy-drift recommendation text is eligible "
            "(unless a higher-priority rule such as fatigue or low_q matched first)."
        ),
        "thresholds": {
            "prior_window_max_rows": RECENT_PRIOR_WINDOW,
            "slice_for_counter": EASY_RECOVERY_DRIFT_SLICE,
            "min_flagged_in_slice_to_fire": EASY_RECOVERY_DRIFT_MIN_FLAGS,
            "intensity_labels_counted_as_too_hard_for_easy_recovery": sorted(
                INTENSITY_LABEL_FLAGGED_EASY_DRIFT
            ),
        },
        "current_run_id": rid,
        "prior_row_count_after_exclude_self": len(prior),
        "evaluation_slice_row_count": len(ev_slice),
        "easy_too_hard_recent_count": n_flag,
        "rule_counter_condition_met": fires,
        "evaluation_slice_rows": rows_out,
        "flagged_runs_in_slice": flagged,
        "misclassification_and_limits": [
            "Label-only: does not use power/HR zone seconds for this counter — only stored intensity_label.",
            "Any easy/recovery session labeled moderate, moderate_high, or high counts. Planned easy runs "
            "often receive 'moderate' from the classifier (acceptable variation or terrain), which can inflate the counter.",
            "steady_run / long_run are never included in this counter; mis-tagged easy days could appear as easy_run.",
            "_prior_runs uses global history minus current id; for a selected past run, the slice may include "
            "activities that occurred later in calendar time if they appear later in load_history order.",
        ],
    }
