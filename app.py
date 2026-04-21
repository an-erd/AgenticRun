import os
import re
import json
import sqlite3
import tempfile
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from agenticrun.services.db import (
    LLM_CONTEXT_METADATA_UNAVAILABLE,
    analyze_structured_work_reps_for_run,
    aggregate_work_only_session_for_run,
    build_interval_family_insight_summary,
    compare_interval_session_vs_prior,
    compare_selected_run_work_family_vs_prior,
    compare_threshold_session_family_latest_vs_prior,
    compare_vo2max_family_latest_vs_prior,
    connect,
    derive_work_block_label_for_run,
    derive_work_session_family_for_run,
    derive_easy_aerobic_efficiency_trend,
    fetch_dedup_easy_aerobic_run_history,
    fetch_dedup_work_family_session_history,
    format_selected_run_interval_family_insight,
    load_history,
    work_family_membership_diagnostic,
    work_segment_family_distribution_diagnostic,
)

DEBUG = os.getenv("AGENTICRUN_DEBUG", "").lower() in {"1", "true", "yes", "on"}

def dprint(*args, **kwargs):
    if DEBUG:
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)


def record_debug(debug_lines: list[str], message: str) -> None:
    debug_lines.append(message)
    dprint(message)


st.set_page_config(
    page_title="AgenticRun",
    page_icon="🏃",
    layout="wide",
)

DB_PATH = Path("agenticrun.db")
OUT_DIR = Path("out")
TRACE_DIR = OUT_DIR / "llm_traces"

if "import_message" not in st.session_state:
    st.session_state.import_message = None
if "import_status" not in st.session_state:
    st.session_state.import_status = None
if "import_debug" not in st.session_state:
    st.session_state.import_debug = None
if "import_summary" not in st.session_state:
    st.session_state.import_summary = None
if "import_snapshot_before" not in st.session_state:
    st.session_state.import_snapshot_before = None

LAST_IMPORT_SUMMARY_PATH = OUT_DIR / "last_import_summary.json"


def _ensure_run_table_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in ("status", "cached_from_run_id", "source_path"):
        if col not in out.columns:
            out[col] = pd.NA
    return out


@st.cache_data
def load_runs_from_db(db_path: str) -> pd.DataFrame:
    if not Path(db_path).exists():
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    try:
        query = """
        SELECT
            run_id,
            source_file,
            run_date,
            distance_km,
            duration_sec,
            avg_pace_sec_km,
            avg_hr,
            max_hr,
            avg_power,
            max_power,
            avg_cadence,
            elevation_gain_m,
            training_load,
            moving_time_sec,
            avg_moving_pace_sec_km,
            stopped_time_sec,
            power_zone_z1_sec,
            power_zone_z2_sec,
            power_zone_z3_sec,
            power_zone_z4_sec,
            power_zone_z5_sec,
            hr_zone_z1_sec,
            hr_zone_z2_sec,
            hr_zone_z3_sec,
            hr_zone_z4_sec,
            hr_zone_z5_sec,
            has_power,
            has_hr,
            has_cadence,
            has_gps,
            data_quality_score,
            fit_parse_warnings,
            training_type,
            execution_quality,
            fatigue_signal,
            fitness_signal,
            trend_label,
            next_session,
            load_action,
            recommendation_summary,
            analysis_summary,
            trend_summary,
            llm_summary,
            llm_summary_short,
            llm_what_next_short,
            llm_context_progress,
            llm_context_progress_short,
            recommendation_signals
        FROM runs
        ORDER BY run_date DESC
        """
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


def load_last_import_summary() -> dict | None:
    if not LAST_IMPORT_SUMMARY_PATH.is_file():
        return None
    try:
        return json.loads(LAST_IMPORT_SUMMARY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def _display_evidence_tier(tier: str) -> str:
    """Compact metric-card values for trend-confidence tiers (analytics stay on raw tier strings)."""
    t = str(tier or "").strip()
    return {
        "Insufficient": "Sparse",
        "Limited": "Limited",
        "Moderate": "Mid",
        "Strong": "Strong",
    }.get(t, t)


def _overview_metric_display(label: str) -> str:
    """Very compact metric-card values for performance overview (fuller text lives below)."""
    s = str(label or "").strip()
    sl = s.lower()
    if sl == "insufficient history":
        return "N/A"
    if sl == "unavailable":
        return "N/A"
    return s


def _family_compare_overview_label(cmp: dict) -> str:
    """Latest-vs-prior work metrics for one interval family → compact card label."""
    if cmp.get("insufficient_history"):
        return "N/A"
    mm = cmp.get("metrics") or {}
    good = 0
    bad = 0
    for key, g, b in (
        ("work_mean_pace_sec_per_km", "faster", "slower"),
        ("work_mean_power_w", "higher", "lower"),
        ("work_mean_hr_avg", "lower", "higher"),
        ("work_w_per_hr", "better", "worse"),
    ):
        st = (mm.get(key) or {}).get("status")
        if st is None or (isinstance(st, float) and pd.isna(st)):
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


def _overall_trend_card_label(trend_label) -> str:
    if trend_label is None or (isinstance(trend_label, float) and pd.isna(trend_label)):
        return "N/A"
    t = str(trend_label).strip().lower()
    mapping = {
        "positive_progress": "Improving",
        "stable": "Stable",
        "possible_fatigue": "Mixed",
        "insufficient_history": "N/A",
        "uncertain_data_quality": "Mixed",
    }
    return mapping.get(t, "Mixed")


def _load_fatigue_overview_label(latest: pd.Series) -> str:
    """Load / fatigue from latest stored run signals (deterministic)."""
    trend = str(latest.get("trend_label") or "").strip().lower()
    fatigue = str(latest.get("fatigue_signal") or "").strip().lower()
    if trend == "insufficient_history" and fatigue in ("", "unknown"):
        return "N/A"
    if fatigue == "low":
        return "Manageable"
    if fatigue == "moderate":
        return "Elevated"
    if fatigue == "unknown":
        return "Mixed"
    return fatigue.title()


def _performance_overview_summary_md(
    overall: str,
    threshold: str,
    vo2: str,
    easy_aerobic: str,
    load_fatigue: str,
) -> str:
    return (
        f"Overall running signals read as **{overall}**. "
        f"Threshold interval work: **{threshold}**; VO2 interval work: **{vo2}**; "
        f"easy / aerobic efficiency: **{easy_aerobic}**; "
        f"fatigue / load: **{load_fatigue}**."
    )


def _performance_overview_explanation_lines(
    overall: str,
    threshold: str,
    vo2: str,
    easy_aerobic: str,
    load_fatigue: str,
) -> list[str]:
    """2–4 short deterministic lines from overview labels only (no LLM)."""

    def n(s: str) -> str:
        return str(s or "").strip()

    o, th, v2, ez, ld = n(overall), n(threshold), n(vo2), n(easy_aerobic), n(load_fatigue)
    ol, thl, v2l, ezl, ldl = o.lower(), th.lower(), v2.lower(), ez.lower(), ld.lower()

    out: list[str] = []

    if ol == "improving":
        out.append(
            "Overall running trend looks positive compared with your recent comparable sessions."
        )
    elif ol == "stable":
        out.append("Overall running trend is steady versus recent comparable sessions.")
    elif ol == "mixed":
        out.append("Overall running trend is mixed versus recent comparable sessions.")
    elif ol in ("insufficient history", "not enough history", "n/a"):
        out.append("More run history is needed before the overall trend is very reliable.")

    if thl in {"unavailable"}:
        out.append("Threshold interval comparison is unavailable.")
    elif thl in ("insufficient history", "not enough history", "n/a"):
        out.append("Threshold interval history is still building.")
    elif thl == "improving":
        out.append("Threshold work is trending upward.")
    elif thl == "declining":
        out.append("Threshold work has softened compared with the prior comparable session.")
    elif thl == "stable":
        out.append("Threshold work is stable versus the last comparable session.")
    elif thl == "mixed":
        out.append("Threshold metrics are mixed versus the last comparable session.")

    if v2l in {"unavailable"}:
        out.append("VO2 interval comparison is unavailable.")
    elif v2l in ("insufficient history", "not enough history", "n/a"):
        out.append("VO2 interval history is still building.")
    elif v2l == "improving":
        out.append("VO2 work is improving.")
    elif v2l == "declining":
        out.append("VO2 work has softened compared with the prior comparable session.")
    elif v2l == "stable":
        out.append("VO2 work is stable versus the last comparable session.")
    elif v2l == "mixed":
        out.append("VO2 metrics are mixed versus the last comparable session.")

    if ezl in {"unavailable"}:
        out.append("Easy/aerobic comparison is unavailable.")
    elif ezl in ("insufficient history", "not enough history", "n/a"):
        out.append("Easy and recovery run history is still building for aerobic efficiency.")
    elif ezl == "improving":
        out.append("Easy/aerobic efficiency is improving versus your prior easy or recovery run.")
    elif ezl == "declining":
        out.append(
            "Easy/aerobic efficiency has dipped versus your prior easy or recovery run."
        )
    elif ezl == "stable":
        out.append("Easy/aerobic efficiency is stable versus the prior easy or recovery run.")
    elif ezl == "mixed":
        out.append("Easy/aerobic metrics are mixed versus the prior easy or recovery run.")

    if ldl in ("insufficient history", "not enough history", "n/a"):
        out.append("Fatigue and load context is limited until there is more comparable history.")
    elif ldl == "manageable":
        out.append("Current fatigue and load look manageable.")
    elif ldl == "elevated":
        out.append("Fatigue/load signals look elevated—worth watching recovery.")
    elif ldl == "mixed":
        out.append("Fatigue/load signals are mixed.")

    return out[:4]


def _dashboard_work_family_chart_df(hist_asc: list) -> pd.DataFrame:
    """Chronological rows (oldest → newest) for dashboard work-family line charts."""
    out: list[dict] = []
    for r in hist_asc:
        rid = str(r.get("run_id") or "").strip()
        rd = r.get("run_date")
        out.append(
            {
                "run_id": rid,
                "run_date_dt": pd.to_datetime(rd, errors="coerce"),
                "work_mean_pace_sec_per_km": _family_trend_numeric(
                    r.get("work_mean_pace_sec_per_km")
                ),
                "work_mean_power_w": _family_trend_numeric(r.get("work_mean_power_w")),
                "work_w_per_hr": _family_trend_numeric(r.get("work_w_per_hr")),
            }
        )
    df = pd.DataFrame(out)
    if not df.empty and "work_mean_pace_sec_per_km" in df.columns:
        df["pace_tooltip"] = _pace_tooltip_series(df["work_mean_pace_sec_per_km"])
    return df


def _dashboard_easy_aerobic_chart_df(hist_asc: list) -> pd.DataFrame:
    """Chronological easy/recovery session rows for pace, HR, and W/HR (power÷HR) trends."""
    out: list[dict] = []
    for r in hist_asc:
        rid = str(r.get("run_id") or "").strip()
        rd = r.get("run_date")
        p = _family_trend_numeric(r.get("avg_pace_sec_km"))
        h = _family_trend_numeric(r.get("avg_hr"))
        pwr = _family_trend_numeric(r.get("avg_power"))
        wh = None
        if pwr is not None and h is not None and float(h) > 0:
            wh = float(pwr) / float(h)
        out.append(
            {
                "run_id": rid,
                "run_date_dt": pd.to_datetime(rd, errors="coerce"),
                "avg_pace_sec_km": p,
                "avg_hr": h,
                "w_per_hr": wh,
            }
        )
    df = pd.DataFrame(out)
    if not df.empty and "avg_pace_sec_km" in df.columns:
        df["pace_tooltip"] = _pace_tooltip_series(df["avg_pace_sec_km"])
    return df


def _easy_aerobic_dashboard_caption(aer: dict) -> str:
    """One line from :func:`derive_easy_aerobic_efficiency_trend` result (no LLM)."""
    if aer.get("insufficient_history"):
        return "History is still building."
    sig = str(aer.get("easy_aerobic_signal") or "").strip()
    if sig == "Improving":
        return "Aerobic efficiency is improving."
    if sig == "Stable":
        return "Easy running is stable."
    if sig == "Declining":
        return "Easy and recovery aerobic efficiency has softened versus the prior session."
    if sig == "Mixed":
        return "Easy and recovery aerobic signals are mixed versus the prior session."
    if sig == "Insufficient history":
        return "History is still building."
    return "Easy and recovery progression from stored session averages."


_RECENT_INDICATOR_WINDOW = 12


def _indicator_metric_float(val) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _indicator_row_date(row: dict) -> pd.Timestamp | None:
    rd = pd.to_datetime(row.get("run_date"), errors="coerce")
    if pd.isna(rd):
        return None
    return rd


def _pick_best_recent_with_recency_tiebreak(
    rows: list[dict],
    metric_getter,
    *,
    higher_is_better: bool,
    tolerance_ratio: float = 0.03,
    max_age_days: int | None = None,
) -> dict | None:
    """
    Pick strongest row with recency-aware tie-break.

    Keeps deterministic "strong first" behavior but prefers newer sessions when quality is close.
    """
    scored: list[tuple[dict, float, pd.Timestamp]] = []
    for r in rows:
        v = metric_getter(r)
        if v is None:
            continue
        rd = _indicator_row_date(r)
        if rd is None:
            continue
        scored.append((r, float(v), rd))
    if not scored:
        return None

    best_metric = max(v for _, v, _ in scored) if higher_is_better else min(v for _, v, _ in scored)
    if higher_is_better:
        floor = best_metric * (1.0 - tolerance_ratio)
        candidates = [(r, rd) for r, v, rd in scored if v >= floor]
    else:
        ceil = best_metric * (1.0 + tolerance_ratio)
        candidates = [(r, rd) for r, v, rd in scored if v <= ceil]
    if max_age_days is not None and candidates:
        newest = max(rd for _, rd in candidates)
        cutoff = newest - pd.Timedelta(days=max_age_days)
        fresh = [(r, rd) for r, rd in candidates if rd >= cutoff]
        if fresh:
            candidates = fresh
    if not candidates:
        best_row = max(scored, key=lambda x: x[1])[0] if higher_is_better else min(scored, key=lambda x: x[1])[0]
        return best_row
    candidates.sort(key=lambda x: x[1])  # newest among near-best candidates
    return candidates[-1][0]


def _best_work_family_recent_anchor(
    hist_asc: list,
    *,
    domain_title: str,
) -> dict[str, str] | None:
    """Pick one strong recent row: best W/HR, else best power, else fastest work pace."""
    if not hist_asc:
        return None
    tail = hist_asc[-_RECENT_INDICATOR_WINDOW:]
    prefer_recency = str(domain_title).strip().lower() == "vo2"

    if prefer_recency:
        best = _pick_best_recent_with_recency_tiebreak(
            tail,
            lambda r: _indicator_metric_float(r.get("work_w_per_hr")),
            higher_is_better=True,
            tolerance_ratio=0.15,
            max_age_days=210,
        )
    else:
        best = None
        best_wh: float | None = None
        for r in tail:
            v = _indicator_metric_float(r.get("work_w_per_hr"))
            if v is not None and (best_wh is None or v > best_wh):
                best_wh = v
                best = r
    if best is not None:
        return {
            "date": str(best.get("run_date") or "-"),
            "label": str(best.get("work_block_label") or "Work").strip() or "Work",
            "reason": (
                f"Top recent work W/HR among your {domain_title} sessions "
                "(strong efficiency signal in this window)."
            ),
        }

    if prefer_recency:
        best = _pick_best_recent_with_recency_tiebreak(
            tail,
            lambda r: _indicator_metric_float(r.get("work_mean_power_w")),
            higher_is_better=True,
            tolerance_ratio=0.12,
            max_age_days=210,
        )
    else:
        best = None
        best_pwr: float | None = None
        for r in tail:
            v = _indicator_metric_float(r.get("work_mean_power_w"))
            if v is not None and (best_pwr is None or v > best_pwr):
                best_pwr = v
                best = r
    if best is not None:
        return {
            "date": str(best.get("run_date") or "-"),
            "label": str(best.get("work_block_label") or "Work").strip() or "Work",
            "reason": f"Top recent work power among your {domain_title} sessions in this window.",
        }

    if prefer_recency:
        best = _pick_best_recent_with_recency_tiebreak(
            tail,
            lambda r: _indicator_metric_float(r.get("work_mean_pace_sec_per_km")),
            higher_is_better=False,
            tolerance_ratio=0.08,
            max_age_days=210,
        )
    else:
        best = None
        best_pace: float | None = None
        for r in tail:
            v = _indicator_metric_float(r.get("work_mean_pace_sec_per_km"))
            if v is not None and v > 0:
                if best_pace is None or v < best_pace:
                    best_pace = v
                    best = r
    if best is not None:
        return {
            "date": str(best.get("run_date") or "-"),
            "label": str(best.get("work_block_label") or "Work").strip() or "Work",
            "reason": f"Fastest recent work pace among your {domain_title} sessions in this window.",
        }
    return None


def _best_easy_aerobic_recent_anchor(hist_asc: list) -> dict[str, str] | None:
    """Prefer best session W/HR (power÷HR); else fastest avg pace."""
    if not hist_asc:
        return None
    tail = hist_asc[-_RECENT_INDICATOR_WINDOW:]
    scored: list[tuple[dict, float]] = []
    for r in tail:
        ap = _indicator_metric_float(r.get("avg_power"))
        ah = _indicator_metric_float(r.get("avg_hr"))
        if ap is not None and ah is not None and ah > 0:
            scored.append((r, ap / ah))
    if scored:
        best_r, _ = max(scored, key=lambda x: x[1])
        lab = format_training_type_label(best_r.get("training_type") or "easy/recovery")
        return {
            "date": str(best_r.get("run_date") or "-"),
            "label": lab,
            "reason": "Best recent session W/HR (power vs HR) among easy and recovery runs.",
        }

    best = None
    best_pace: float | None = None
    for r in tail:
        v = _indicator_metric_float(r.get("avg_pace_sec_km"))
        if v is not None and v > 0:
            if best_pace is None or v < best_pace:
                best_pace = v
                best = r
    if best is not None:
        lab = format_training_type_label(best.get("training_type") or "easy/recovery")
        return {
            "date": str(best.get("run_date") or "-"),
            "label": lab,
            "reason": "Fastest recent session pace among easy and recovery runs.",
        }
    return None


def render_best_recent_indicators(db_path: Path) -> None:
    """Compact strongest-recent anchors from deterministic history (no LLM)."""
    st.subheader("Best recent indicators")
    st.caption(
        "One highlight per domain from recent history (up to 12 sessions). "
        "Intervals: W/HR, then power, then pace. Easy runs: W/HR, then pace."
    )
    try:
        conn = connect(str(db_path))
        try:
            th_hist = fetch_dedup_work_family_session_history(conn, "threshold_session")
            vo2_hist = fetch_dedup_work_family_session_history(conn, "vo2max_session")
            ea_hist = fetch_dedup_easy_aerobic_run_history(conn)
        finally:
            conn.close()
    except Exception as exc:
        st.caption(f"Best recent indicators unavailable: {exc}")
        return

    th_a = _best_work_family_recent_anchor(th_hist, domain_title="threshold")
    vo_a = _best_work_family_recent_anchor(vo2_hist, domain_title="VO2")
    ea_a = _best_easy_aerobic_recent_anchor(ea_hist)

    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        st.markdown("**Threshold**")
        if th_a:
            st.markdown(f"**{th_a['date']}** · {th_a['label']}")
            st.caption(th_a["reason"])
        else:
            st.caption("Not enough history yet for a threshold-style anchor.")
    with ic2:
        st.markdown("**VO2**")
        if vo_a:
            st.markdown(f"**{vo_a['date']}** · {vo_a['label']}")
            st.caption(vo_a["reason"])
        else:
            st.caption("Not enough history yet for a VO2-style anchor.")
    with ic3:
        st.markdown("**Easy / aerobic**")
        if ea_a:
            st.markdown(f"**{ea_a['date']}** · {ea_a['label']}")
            st.caption(ea_a["reason"])
        else:
            st.caption("Not enough history yet for an easy/aerobic anchor.")


_WEEKLY_HARD_TRAINING_TYPES = frozenset(
    {
        "threshold_run",
        "vo2_interval_session",
        "test_or_vo2_session",
        "test_session",
        "race",
    }
)
_WEEKLY_EASY_RECOVERY_TYPES = frozenset({"easy_run", "recovery_run"})
_WEEKLY_STEADY_LONG_TYPES = frozenset({"steady_run", "long_run"})


def _weekly_session_bucket(training_type: object) -> str:
    """Classify persisted ``training_type`` into hard / easy_recovery / steady_long / other."""
    if training_type is None or (isinstance(training_type, float) and pd.isna(training_type)):
        return "other"
    s = str(training_type).strip().lower()
    if s in _WEEKLY_HARD_TRAINING_TYPES:
        return "hard"
    if s in _WEEKLY_EASY_RECOVERY_TYPES:
        return "easy_recovery"
    if s in _WEEKLY_STEADY_LONG_TYPES:
        return "steady_long"
    return "other"


_ARCHIVE_VO2_TRAINING_TYPES = frozenset(
    {"vo2_interval_session", "test_or_vo2_session", "test_session", "race"}
)


def _format_history_span_days(earliest: pd.Timestamp, latest: pd.Timestamp) -> str:
    if pd.isna(earliest) or pd.isna(latest):
        return "—"
    e = earliest.normalize()
    l = latest.normalize()
    if e == l:
        return "single day"
    days = max(0, int((l - e).days))
    if days == 0:
        return "single day"
    y = days // 365
    mo = (days % 365) // 30
    if y >= 1:
        if mo >= 1:
            return f"~{y}y {mo}mo"
        return f"~{y} year{'s' if y != 1 else ''}"
    if days >= 30:
        return f"~{days // 30} months"
    return f"{days} days"


@st.cache_data
def _domain_history_counts_from_db(db_path: str) -> dict[str, int] | None:
    """Domain session counts from the same deduplicated histories used by dashboard panels."""
    if not Path(db_path).is_file():
        return None
    try:
        conn = connect(db_path)
        try:
            th_hist = fetch_dedup_work_family_session_history(conn, "threshold_session")
            vo2_hist = fetch_dedup_work_family_session_history(conn, "vo2max_session")
            ea_hist = fetch_dedup_easy_aerobic_run_history(conn)
        finally:
            conn.close()
    except Exception:
        return None
    return {
        "n_threshold": int(len(th_hist)),
        "n_vo2": int(len(vo2_hist)),
        "n_easy_recovery": int(len(ea_hist)),
    }


def _archive_coverage_payload(
    df_runs: pd.DataFrame, db_path: str | None = None
) -> dict[str, object] | None:
    """Deterministic run-span + domain counts; prefer DB-backed family histories when available."""
    if df_runs.empty or "run_date" not in df_runs.columns:
        return None
    rd = pd.to_datetime(df_runs["run_date"], errors="coerce")
    valid = df_runs.assign(_rd=rd).dropna(subset=["_rd"])
    if valid.empty:
        return None
    earliest = valid["_rd"].min()
    latest = valid["_rd"].max()
    n = int(len(valid))
    tt = (
        valid["training_type"].fillna("").astype(str).str.strip().str.lower()
        if "training_type" in valid.columns
        else pd.Series([""] * n, index=valid.index)
    )
    n_threshold = int((tt == "threshold_run").sum())
    n_vo2 = int(tt.isin(_ARCHIVE_VO2_TRAINING_TYPES).sum())
    n_easy = int(tt.isin(_WEEKLY_EASY_RECOVERY_TYPES).sum())
    out = {
        "n_runs": n,
        "earliest": earliest,
        "latest": latest,
        "n_threshold": n_threshold,
        "n_vo2": n_vo2,
        "n_easy_recovery": n_easy,
    }
    if db_path:
        db_counts = _domain_history_counts_from_db(db_path)
        if db_counts is not None:
            out["n_threshold"] = int(db_counts["n_threshold"])
            out["n_vo2"] = int(db_counts["n_vo2"])
            out["n_easy_recovery"] = int(db_counts["n_easy_recovery"])
    return out


def render_archive_coverage(df_runs: pd.DataFrame) -> None:
    """Compact archive / history coverage from stored runs (deterministic, no LLM)."""
    st.markdown("### History loaded")
    st.caption("Running activities available in the database for trends and analysis.")
    payload = _archive_coverage_payload(df_runs, str(DB_PATH))
    if payload is None:
        st.caption("No runs with valid dates yet.")
        return

    earliest: pd.Timestamp = payload["earliest"]  # type: ignore[assignment]
    latest: pd.Timestamp = payload["latest"]  # type: ignore[assignment]
    n = int(payload["n_runs"])
    n_threshold = int(payload["n_threshold"])
    n_vo2 = int(payload["n_vo2"])
    n_easy = int(payload["n_easy_recovery"])
    span = _format_history_span_days(earliest, latest)
    ed = earliest.strftime("%Y-%m-%d")
    ld = latest.strftime("%Y-%m-%d")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total activities", n)
    with m2:
        st.metric("Earliest run", ed)
    with m3:
        st.metric("Latest run", ld)
    with m4:
        st.metric("Approx. span", span)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Threshold sessions",
            n_threshold,
            help="deduplicated threshold work-family sessions from stored segments",
        )
    with c2:
        st.metric(
            "VO2-style sessions",
            n_vo2,
            help="deduplicated VO2 work-family sessions from stored segments",
        )
    with c3:
        st.metric(
            "Easy / recovery",
            n_easy,
            help="deduplicated easy/recovery runs",
        )

    st.markdown(
        f"AgenticRun currently covers **{ed}** to **{ld}** ({span}) across "
        f"**{n}** running activit{'y' if n == 1 else 'ies'} in the database."
    )


def _trend_evidence_strength(session_count: int) -> str:
    """Map labeled-session count to a compact evidence tier (deterministic, no LLM)."""
    if session_count <= 0:
        return "Insufficient"
    if session_count <= 3:
        return "Limited"
    if session_count <= 11:
        return "Moderate"
    return "Strong"


def _trend_confidence_summary_sentence(
    threshold_label: str, vo2_label: str, easy_label: str
) -> str:
    dt = _display_evidence_tier(threshold_label)
    dv = _display_evidence_tier(vo2_label)
    de = _display_evidence_tier(easy_label)
    return (
        f"**Threshold:** {dt}. **VO2:** {dv}. **Easy/aerobic:** {de}."
    )


def render_trend_confidence(df_runs: pd.DataFrame) -> None:
    """Evidence strength for domain trends from session-type counts (deterministic, no LLM)."""
    st.markdown("### Trend confidence")
    st.caption(
        "How much labeled history supports threshold, VO2-style, and easy/aerobic trend conclusions."
    )
    payload = _archive_coverage_payload(df_runs, str(DB_PATH))
    if payload is None:
        st.caption("Not enough dated runs to assess trend evidence yet.")
        return

    n_th = int(payload["n_threshold"])
    n_vo2 = int(payload["n_vo2"])
    n_easy = int(payload["n_easy_recovery"])
    lab_th = _trend_evidence_strength(n_th)
    lab_vo2 = _trend_evidence_strength(n_vo2)
    lab_easy = _trend_evidence_strength(n_easy)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Threshold trend evidence",
            _display_evidence_tier(lab_th),
            delta=f"{n_th} threshold session{'s' if n_th != 1 else ''}",
            delta_color="off",
        )
    with c2:
        st.metric(
            "VO2 trend evidence",
            _display_evidence_tier(lab_vo2),
            delta=f"{n_vo2} VO2-style session{'s' if n_vo2 != 1 else ''}",
            delta_color="off",
        )
    with c3:
        st.metric(
            "Easy / aerobic trend evidence",
            _display_evidence_tier(lab_easy),
            delta=f"{n_easy} easy/recovery run{'s' if n_easy != 1 else ''}",
            delta_color="off",
        )

    st.markdown(_trend_confidence_summary_sentence(lab_th, lab_vo2, lab_easy))
    if int(payload["n_runs"]) <= 12:
        st.caption(
            "With only a handful of activities loaded, these tiers stay conservative—"
            "they strengthen automatically as you add more runs."
        )


def _db_coverage_snapshot(db_path: str) -> dict[str, object]:
    """Point-in-time activity counts, date span, and evidence labels (no new schema)."""
    empty: dict[str, object] = {
        "n_runs": 0,
        "n_threshold": 0,
        "n_vo2": 0,
        "n_easy_recovery": 0,
        "earliest": None,
        "latest": None,
        "label_th": "Insufficient",
        "label_vo2": "Insufficient",
        "label_easy": "Insufficient",
    }
    if not Path(db_path).is_file():
        return empty
    df = load_runs_from_db(str(db_path))
    p = _archive_coverage_payload(df, db_path)
    if p is None:
        return empty
    nth = int(p["n_threshold"])
    nv = int(p["n_vo2"])
    ne = int(p["n_easy_recovery"])
    return {
        "n_runs": int(p["n_runs"]),
        "n_threshold": nth,
        "n_vo2": nv,
        "n_easy_recovery": ne,
        "earliest": p["earliest"],
        "latest": p["latest"],
        "label_th": _trend_evidence_strength(nth),
        "label_vo2": _trend_evidence_strength(nv),
        "label_easy": _trend_evidence_strength(ne),
    }


_EVIDENCE_ORDER = ("Insufficient", "Limited", "Moderate", "Strong")


def _evidence_rank(label: str) -> int:
    try:
        return _EVIDENCE_ORDER.index(str(label).strip())
    except ValueError:
        return 0


def render_what_changed_after_import(
    db_path: Path,
    summary: dict | None,
    snap_before: dict | None,
) -> None:
    """Impact of the last import vs prior DB state (deterministic, no LLM)."""
    if not summary:
        return
    u = int(summary.get("uploaded_files") or 0)
    n_new = int(summary.get("new_analyzed") or 0)
    n_dup = int(summary.get("duplicate_cached") or 0)
    n_err = int(summary.get("errors") or 0)
    n_inc = int(summary.get("incomplete_or_unsupported_fit") or 0)
    if u <= 0 and n_new <= 0 and n_dup <= 0 and n_err <= 0 and n_inc <= 0:
        return

    st.markdown("#### What changed after import?")
    st.caption(
        "Impact of the most recent import (from the summary below). "
        "History-range and trend deltas need a snapshot taken before import on this page."
    )

    snap_after = _db_coverage_snapshot(str(db_path))

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("New activities analyzed", n_new)
    with m2:
        st.metric("Duplicate / cached", n_dup)
    with m3:
        st.metric("Import errors", n_err)

    if n_inc > 0:
        st.caption(
            f"Bulk-style skips (incomplete/unsupported FIT): **{n_inc}** files not analyzed."
        )

    d_th = d_vo2 = d_easy = None
    if snap_before is not None:
        d_th = int(snap_after["n_threshold"]) - int(snap_before["n_threshold"])
        d_vo2 = int(snap_after["n_vo2"]) - int(snap_before["n_vo2"])
        d_easy = int(snap_after["n_easy_recovery"]) - int(snap_before["n_easy_recovery"])
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("New threshold sessions (est.)", d_th)
        with c2:
            st.metric("New VO2-style sessions (est.)", d_vo2)
        with c3:
            st.metric("New easy/recovery runs (est.)", d_easy)

    range_bits: list[str] = []
    if snap_before is not None:
        e0, l0 = snap_before.get("earliest"), snap_before.get("latest")
        e1, l1 = snap_after.get("earliest"), snap_after.get("latest")
        if e1 is not None and l1 is not None:
            e1s = pd.Timestamp(e1).strftime("%Y-%m-%d")
            l1s = pd.Timestamp(l1).strftime("%Y-%m-%d")
            if e0 is None or l0 is None:
                range_bits.append(f"Visible history is now **{e1s}** → **{l1s}**.")
            else:
                te0, tl0 = pd.Timestamp(e0), pd.Timestamp(l0)
                te1, tl1 = pd.Timestamp(e1), pd.Timestamp(l1)
                back = te1 < te0
                fwd = tl1 > tl0
                if back and fwd:
                    range_bits.append("Date range **extended backward and forward**.")
                elif back:
                    range_bits.append("Earliest visible run **moved earlier** (more history backward).")
                elif fwd:
                    range_bits.append("Latest visible run **moved forward**.")
                else:
                    range_bits.append("Visible date span **unchanged** (same earliest/latest).")
    else:
        st.caption(
            "_Import from this page once to record a pre-import snapshot; "
            "then you’ll see date-range and domain deltas here._"
        )

    ev_lines: list[str] = []
    if snap_before is not None:
        for title, key in (
            ("Threshold", "label_th"),
            ("VO2", "label_vo2"),
            ("Easy / aerobic", "label_easy"),
        ):
            b = str(snap_before.get(key) or "Insufficient")
            a = str(snap_after.get(key) or "Insufficient")
            if _evidence_rank(a) > _evidence_rank(b):
                ev_lines.append(f"**{title}** evidence: {b} → **{a}**")

    if ev_lines:
        st.caption("Trend evidence (tier changes)")
        for line in ev_lines:
            st.markdown(f"- {line}")

    if range_bits:
        st.markdown(range_bits[0])

    ext_left = ext_right = False
    if snap_before is not None:
        e0, l0 = snap_before.get("earliest"), snap_before.get("latest")
        e1, l1 = snap_after.get("earliest"), snap_after.get("latest")
        if e0 is not None and l0 is not None and e1 is not None and l1 is not None:
            te0, tl0 = pd.Timestamp(e0), pd.Timestamp(l0)
            te1, tl1 = pd.Timestamp(e1), pd.Timestamp(l1)
            ext_left = te1 < te0
            ext_right = tl1 > tl0

    domain_growth = False
    if snap_before is not None:
        domain_growth = (d_th + d_vo2 + d_easy) >= 8

    material = (
        n_new >= 10
        or ext_left
        or ext_right
        or domain_growth
        or len(ev_lines) > 0
    )

    tail = ""
    if material and (n_new > 0 or ext_left or ext_right):
        tail = " This materially expanded your running history in AgenticRun."
    elif material and n_new > 0:
        tail = " This added meaningful new data for trend analysis."

    summ = (
        f"The last import processed **{u}** candidate file(s), added **{n_new}** new activit"
        f"{'y' if n_new == 1 else 'ies'}, and **{n_dup}** duplicate/cached match(es)."
    )
    if n_err:
        summ += f" **{n_err}** error(s) occurred."
    summ += tail

    st.markdown(summ)


def _weekly_window_stats(df: pd.DataFrame) -> dict[str, float | int]:
    """Aggregate runs in ``df`` (already date-filtered)."""
    n = len(df)
    dist = float(df["distance_km"].fillna(0).sum()) if n and "distance_km" in df.columns else 0.0
    dur = 0.0
    if n and "duration_sec" in df.columns:
        ds = df["duration_sec"].dropna()
        dur = float(ds.sum()) if len(ds) else 0.0
    hard = easy = steady = other = 0
    if n and "training_type" in df.columns:
        for tt in df["training_type"]:
            b = _weekly_session_bucket(tt)
            if b == "hard":
                hard += 1
            elif b == "easy_recovery":
                easy += 1
            elif b == "steady_long":
                steady += 1
            else:
                other += 1
    return {
        "runs": n,
        "distance_km": dist,
        "duration_sec": int(dur),
        "hard": hard,
        "easy_recovery": easy,
        "steady_long": steady,
        "other": other,
    }


def _weekly_load_direction(
    curr_dur: float, prev_dur: float, prev_runs: int
) -> str:
    """Compare total duration vs prior window → up / stable / down / insufficient_history."""
    if prev_runs <= 0 or prev_dur <= 0:
        return "insufficient_history"
    if curr_dur <= 0:
        return "down"
    ratio = curr_dur / prev_dur
    if ratio > 1.08:
        return "up"
    if ratio < 0.92:
        return "down"
    return "stable"


def _weekly_training_summary_line(
    cur: dict[str, float | int],
    prev: dict[str, float | int],
    direction: str,
) -> str:
    """One deterministic sentence (no LLM)."""
    if cur["runs"] <= 0:
        return "No runs recorded in the last 7 days."
    h = int(cur["hard"])
    e = int(cur["easy_recovery"])
    s = int(cur["steady_long"])
    o = int(cur["other"])
    mix_parts: list[str] = []
    if h:
        mix_parts.append(f"{h} hard session{'s' if h != 1 else ''}")
    if e:
        mix_parts.append(f"{e} easy or recovery run{'s' if e != 1 else ''}")
    if s:
        mix_parts.append(f"{s} steady or long aerobic run{'s' if s != 1 else ''}")
    if o > 0:
        mix_parts.append(f"{o} other run{'s' if o != 1 else ''}")
    mix = ", ".join(mix_parts) if mix_parts else "runs without a clear session-type label"

    if direction == "insufficient_history":
        tail = "Not enough activity in the prior 7 days to compare weekly load."
    elif direction == "up":
        tail = "Load is up relative to the previous week."
    elif direction == "down":
        tail = "Load is down relative to the previous week."
    else:
        tail = "Load is stable relative to the previous week."

    return f"Recent mix: {mix}. {tail}"


def render_weekly_training_summary(df_runs: pd.DataFrame) -> None:
    """Compact last 7 days vs prior 7 days from stored runs (deterministic, no LLM)."""
    st.subheader("Weekly training summary")
    st.caption("Last 7 days vs the previous 7 days (from your run dates and session types).")
    if df_runs.empty or "run_date_dt" not in df_runs.columns:
        st.caption("No run dates available for a weekly summary yet.")
        return

    work = df_runs.dropna(subset=["run_date_dt"]).copy()
    if work.empty:
        st.caption("No valid run dates for a weekly summary yet.")
        return

    rd = pd.to_datetime(work["run_date_dt"], errors="coerce")
    work["_day"] = rd.dt.normalize()

    end = pd.Timestamp.today().normalize()
    curr_start = end - pd.Timedelta(days=6)
    curr_end = end
    prev_start = end - pd.Timedelta(days=13)
    prev_end = end - pd.Timedelta(days=7)

    cur_df = work[(work["_day"] >= curr_start) & (work["_day"] <= curr_end)]
    prev_df = work[(work["_day"] >= prev_start) & (work["_day"] <= prev_end)]

    cur = _weekly_window_stats(cur_df)
    prev = _weekly_window_stats(prev_df)
    direction = _weekly_load_direction(
        float(cur["duration_sec"]), float(prev["duration_sec"]), int(prev["runs"])
    )

    dir_label = {
        "up": "Up",
        "stable": "Stable",
        "down": "Down",
        "insufficient_history": "N/A",
    }.get(direction, "—")

    m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)
    with m1:
        st.metric("Runs (7d)", int(cur["runs"]))
    with m2:
        st.metric("Distance (7d)", format_number(cur["distance_km"], digits=1, suffix=" km"))
    with m3:
        st.metric("Duration (7d)", format_duration(cur["duration_sec"]))
    with m4:
        st.metric("Hard sessions", int(cur["hard"]))
    with m5:
        st.metric("Easy / recovery", int(cur["easy_recovery"]))
    with m6:
        st.metric("Steady / long", int(cur["steady_long"]))
    with m7:
        st.metric("Other", int(cur["other"]))
    with m8:
        st.metric("Load vs prior week", dir_label)

    st.caption(
        f"Prior 7 days: {int(prev['runs'])} runs, "
        f"{format_number(prev['distance_km'], digits=1, suffix=' km')}, "
        f"{format_duration(prev['duration_sec'])} — "
        f"{int(prev['hard'])} hard, {int(prev['easy_recovery'])} easy/recovery, "
        f"{int(prev['steady_long'])} steady/long, {int(prev['other'])} other."
    )
    st.markdown(_weekly_training_summary_line(cur, prev, direction))


def _four_week_rolling_blocks(end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Four consecutive 7-day blocks ending on ``end`` (oldest first, newest last)."""
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for i in range(3, -1, -1):
        week_end = end - pd.Timedelta(days=7 * i)
        week_start = week_end - pd.Timedelta(days=6)
        out.append((week_start, week_end))
    return out


def _four_week_rows_from_work(
    work: pd.DataFrame,
    end: pd.Timestamp,
) -> tuple[list[dict[str, float | int]], list[str]]:
    """Per-week stats (oldest→newest) and display labels for each block."""
    rows: list[dict[str, float | int]] = []
    labels: list[str] = []
    for week_start, week_end in _four_week_rolling_blocks(end):
        sub = work[(work["_day"] >= week_start) & (work["_day"] <= week_end)]
        st_w = _weekly_window_stats(sub)
        rows.append(
            {
                "runs": int(st_w["runs"]),
                "distance_km": float(st_w["distance_km"]),
                "hard": int(st_w["hard"]),
            }
        )
        labels.append(
            f"{week_start.strftime('%b %d')}–{week_end.strftime('%b %d')}"
        )
    return rows, labels


def _four_week_classify_pattern(rows: list[dict[str, float | int]]) -> tuple[str, bool]:
    """Return pattern label and whether hard-session counts look uneven."""
    dists = [float(r["distance_km"]) for r in rows]
    hards = [int(r["hard"]) for r in rows]
    runs = [int(r["runs"]) for r in rows]
    total_runs = sum(runs)
    weeks_with = sum(1 for r in runs if r > 0)

    if total_runs < 4 or weeks_with < 2:
        return "Insufficient history", False

    older = dists[0] + dists[1]
    recent = dists[2] + dists[3]
    mean_d = sum(dists) / 4.0
    if mean_d > 1e-9:
        variance = sum((x - mean_d) ** 2 for x in dists) / 4.0
        cv = (variance ** 0.5) / mean_d
    else:
        cv = 0.0

    hard_uneven = (max(hards) - min(hards) >= 2) and (sum(hards) >= 3)

    label: str | None = None
    if older > 0:
        if recent < 0.88 * older:
            label = "Dropping"
        elif recent > 1.12 * older:
            label = "Building"
    elif recent > 0:
        label = "Building"

    if label is None:
        if cv > 0.32 or hard_uneven:
            label = "Uneven"
        else:
            label = "Consistent"

    return label, hard_uneven


def _four_week_consistency_summary_line(pattern: str, hard_uneven: bool) -> str:
    """One deterministic sentence under the 4-week section (no LLM)."""
    if pattern == "Insufficient history":
        return "Not enough runs across the last four weeks to judge consistency yet."
    if pattern == "Consistent":
        return "Training has been consistent across the last 4 weeks."
    if pattern == "Dropping":
        return "Training volume has declined compared with earlier weeks."
    if pattern == "Building":
        if hard_uneven:
            return "Volume is building, but hard-session distribution is uneven."
        return "Training volume is building compared with earlier weeks."
    return "Weekly volume or intensity varies; the last four weeks look uneven."


def render_four_week_consistency_progression(df_runs: pd.DataFrame) -> None:
    """Four rolling 7-day blocks (28 days) — distance, runs, hard sessions; deterministic label."""
    st.subheader("4-week consistency / progression")
    st.caption("Four 7-day blocks ending today. Wk 1 = oldest, Wk 4 = most recent.")
    if df_runs.empty or "run_date_dt" not in df_runs.columns:
        st.caption("No run dates available for a 4-week view yet.")
        return

    work = df_runs.dropna(subset=["run_date_dt"]).copy()
    if work.empty:
        st.caption("No valid run dates for a 4-week view yet.")
        return

    rd = pd.to_datetime(work["run_date_dt"], errors="coerce")
    work["_day"] = rd.dt.normalize()
    end = pd.Timestamp.today().normalize()

    rows, labels = _four_week_rows_from_work(work, end)
    pattern, hard_uneven = _four_week_classify_pattern(rows)

    st.markdown(f"**Pattern:** {pattern}")

    tbl = pd.DataFrame(
        {
            "Week": labels,
            "Runs": [r["runs"] for r in rows],
            "Distance (km)": [round(float(r["distance_km"]), 2) for r in rows],
            "Hard sessions": [r["hard"] for r in rows],
        }
    )
    plot_df = pd.DataFrame(
        {"Distance (km)": [float(r["distance_km"]) for r in rows]},
        # Zero-pad so Streamlit bar_chart lexicographic index order matches week number.
        index=[f"Wk {i + 1:02d}" for i in range(4)],
    )

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("**Distance by week**")
        st.bar_chart(plot_df, width="stretch")
    with c_right:
        st.markdown("**Weekly totals**")
        st.dataframe(tbl, hide_index=True, width="stretch")

    st.markdown(_four_week_consistency_summary_line(pattern, hard_uneven))


def _twelve_week_rolling_blocks(end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Twelve consecutive 7-day blocks ending on ``end`` (oldest first, newest last)."""
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for i in range(11, -1, -1):
        week_end = end - pd.Timedelta(days=7 * i)
        week_start = week_end - pd.Timedelta(days=6)
        out.append((week_start, week_end))
    return out


def _twelve_week_rows_from_work(
    work: pd.DataFrame,
    end: pd.Timestamp,
) -> tuple[list[dict[str, float | int]], list[str]]:
    """Per-week stats (oldest→newest) and display labels for each block."""
    rows: list[dict[str, float | int]] = []
    labels: list[str] = []
    for week_start, week_end in _twelve_week_rolling_blocks(end):
        sub = work[(work["_day"] >= week_start) & (work["_day"] <= week_end)]
        st_w = _weekly_window_stats(sub)
        rows.append(
            {
                "runs": int(st_w["runs"]),
                "distance_km": float(st_w["distance_km"]),
                "hard": int(st_w["hard"]),
            }
        )
        labels.append(
            f"{week_start.strftime('%b %d')}–{week_end.strftime('%b %d')}"
        )
    return rows, labels


def _twelve_week_classify_pattern(
    rows: list[dict[str, float | int]],
) -> tuple[str, bool]:
    """Medium-term label (Building / Stable / Uneven / Dropping / Insufficient history)."""
    dists = [float(r["distance_km"]) for r in rows]
    hards = [int(r["hard"]) for r in rows]
    runs = [int(r["runs"]) for r in rows]
    total_runs = sum(runs)
    weeks_with = sum(1 for r in runs if r > 0)

    if total_runs < 8 or weeks_with < 4:
        return "Insufficient history", False

    older = sum(dists[0:6])
    recent = sum(dists[6:12])
    mean_d = sum(dists) / 12.0
    if mean_d > 1e-9:
        variance = sum((x - mean_d) ** 2 for x in dists) / 12.0
        cv = (variance ** 0.5) / mean_d
    else:
        cv = 0.0

    hard_uneven = (max(hards) - min(hards) >= 3) and (sum(hards) >= 6)

    label: str | None = None
    if older > 0:
        if recent < 0.90 * older:
            label = "Dropping"
        elif recent > 1.10 * older:
            label = "Building"
    elif recent > 0:
        label = "Building"

    if label is None:
        if cv > 0.30 or hard_uneven:
            label = "Uneven"
        else:
            label = "Stable"

    r4 = dists[8:12]
    m4 = sum(r4) / 4.0
    if m4 > 1e-9:
        v4 = sum((x - m4) ** 2 for x in r4) / 4.0
        cv4 = (v4 ** 0.5) / m4
    else:
        cv4 = 0.0
    runs_recent = sum(runs[8:12])
    recent_month_stable = (
        label == "Uneven"
        and cv4 < 0.22
        and runs_recent >= 3
    )

    return label, recent_month_stable


def _twelve_week_progression_summary_line(pattern: str, recent_month_stable: bool) -> str:
    """One deterministic sentence under the 12-week section (no LLM)."""
    if pattern == "Insufficient history":
        return "Not enough history over the last 12 weeks to summarize progression yet."
    if pattern == "Building":
        return "Training volume has been building over the last 12 weeks."
    if pattern == "Dropping":
        return "Training volume has softened over the last 12 weeks."
    if pattern == "Stable":
        return "Training volume has been steady over the last 12 weeks."
    if pattern == "Uneven" and recent_month_stable:
        return "Longer-term progression is uneven despite a stable recent month."
    return "Weekly training has been uneven across the last 12 weeks."


def render_twelve_week_progression(df_runs: pd.DataFrame) -> None:
    """Twelve rolling 7-day blocks (84 days) — distance, runs, hard; medium-term label."""
    st.subheader("12-week progression")
    st.caption("Twelve 7-day blocks ending today. Wk 1 = oldest, Wk 12 = most recent.")
    if df_runs.empty or "run_date_dt" not in df_runs.columns:
        st.caption("No run dates available for a 12-week view yet.")
        return

    work = df_runs.dropna(subset=["run_date_dt"]).copy()
    if work.empty:
        st.caption("No valid run dates for a 12-week view yet.")
        return

    rd = pd.to_datetime(work["run_date_dt"], errors="coerce")
    work["_day"] = rd.dt.normalize()
    end = pd.Timestamp.today().normalize()

    rows, labels = _twelve_week_rows_from_work(work, end)
    pattern, recent_month_stable = _twelve_week_classify_pattern(rows)

    st.markdown(f"**Pattern:** {pattern}")

    # Zero-pad so Streamlit bar_chart lexicographic index order matches week number.
    idx = [f"Wk {i + 1:02d}" for i in range(12)]
    dist_plot = pd.DataFrame({"Distance (km)": [float(r["distance_km"]) for r in rows]}, index=idx)
    hard_plot = pd.DataFrame({"Hard sessions": [int(r["hard"]) for r in rows]}, index=idx)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Distance by week**")
        st.bar_chart(dist_plot, width="stretch")
    with c2:
        st.markdown("**Hard sessions by week**")
        st.bar_chart(hard_plot, width="stretch")

    tbl = pd.DataFrame(
        {
            "Week": labels,
            "Runs": [r["runs"] for r in rows],
            "Distance (km)": [round(float(r["distance_km"]), 2) for r in rows],
            "Hard sessions": [r["hard"] for r in rows],
        }
    )
    st.markdown("**Weekly totals**")
    st.dataframe(tbl, hide_index=True, width="stretch")

    st.markdown(_twelve_week_progression_summary_line(pattern, recent_month_stable))


def _runs_in_calendar_window(
    work: pd.DataFrame, end: pd.Timestamp, n_calendar_days: int
) -> int:
    """Count run rows with ``_day`` in the last ``n_calendar_days`` days ending ``end``."""
    start = end - pd.Timedelta(days=n_calendar_days - 1)
    sub = work[(work["_day"] >= start) & (work["_day"] <= end)]
    return int(len(sub))


def _active_days_in_calendar_window(
    work: pd.DataFrame, end: pd.Timestamp, n_calendar_days: int
) -> int:
    """Distinct calendar days with at least one run."""
    start = end - pd.Timedelta(days=n_calendar_days - 1)
    sub = work[(work["_day"] >= start) & (work["_day"] <= end)]
    if sub.empty:
        return 0
    return int(sub["_day"].nunique())


def _weekly_run_counts_cv(weekly_runs: list[int]) -> tuple[float, float]:
    """Mean weekly run count and coefficient of variation (0 if degenerate)."""
    n = len(weekly_runs)
    if n == 0:
        return 0.0, 0.0
    total = float(sum(weekly_runs))
    mean = total / float(n)
    if mean <= 1e-9:
        return 0.0, 0.0
    variance = sum((float(x) - mean) ** 2 for x in weekly_runs) / float(n)
    cv = (variance ** 0.5) / mean
    return mean, cv


def _training_regularity_label(
    *,
    total_runs_28: int,
    avg_runs_4w: float,
    avg_runs_12w: float,
    active_14: int,
    active_28: int,
    weekly_runs_4: list[int],
) -> str:
    """Very regular / Regular / Uneven / Sparse / Insufficient history."""
    if total_runs_28 < 3:
        return "Insufficient history"

    mean_w, cv_w = _weekly_run_counts_cv(weekly_runs_4)

    if avg_runs_4w < 2.0 or (total_runs_28 < 6 and active_28 < 5):
        return "Sparse"

    if (
        cv_w > 0.45
        and mean_w >= 1.5
        and total_runs_28 >= 6
        and max(weekly_runs_4) - min(weekly_runs_4) >= 3
    ):
        return "Uneven"

    if avg_runs_4w >= 3.5 and avg_runs_12w >= 2.5 and cv_w <= 0.40 and active_14 >= 4:
        return "Very regular"

    return "Regular"


def _training_regularity_summary_line(label: str) -> str:
    """One deterministic sentence (no LLM)."""
    if label == "Insufficient history":
        return "Not enough recent runs to judge training regularity yet."
    if label == "Sparse":
        return "Training frequency is low; sessions are relatively sparse."
    if label == "Uneven":
        return "Availability is uneven, with noticeable gaps between sessions."
    if label == "Very regular":
        return "Training frequency has been regular and consistent over the last month."
    return "Training frequency has been regular over the last month."


def render_training_regularity_availability(df_runs: pd.DataFrame) -> None:
    """Runs/week (4w & 12w), active days (14d & 28d), simple regularity label."""
    st.subheader("Training regularity / availability")
    st.caption("How often you run: weekly averages and distinct days with a run (through today).")
    if df_runs.empty or "run_date_dt" not in df_runs.columns:
        st.caption("No run dates available for regularity yet.")
        return

    work = df_runs.dropna(subset=["run_date_dt"]).copy()
    if work.empty:
        st.caption("No valid run dates for regularity yet.")
        return

    rd = pd.to_datetime(work["run_date_dt"], errors="coerce")
    work["_day"] = rd.dt.normalize()
    end = pd.Timestamp.today().normalize()

    total_runs_28 = _runs_in_calendar_window(work, end, 28)
    total_runs_84 = _runs_in_calendar_window(work, end, 84)
    avg_runs_4w = total_runs_28 / 4.0
    avg_runs_12w = total_runs_84 / 12.0

    active_14 = _active_days_in_calendar_window(work, end, 14)
    active_28 = _active_days_in_calendar_window(work, end, 28)

    rows4, _ = _four_week_rows_from_work(work, end)
    weekly_runs_4 = [int(r["runs"]) for r in rows4]

    label = _training_regularity_label(
        total_runs_28=total_runs_28,
        avg_runs_4w=avg_runs_4w,
        avg_runs_12w=avg_runs_12w,
        active_14=active_14,
        active_28=active_28,
        weekly_runs_4=weekly_runs_4,
    )

    st.markdown(f"**Pattern:** {label}")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Avg runs / week (4 wk)", f"{avg_runs_4w:.1f}")
    with m2:
        st.metric("Avg runs / week (12 wk)", f"{avg_runs_12w:.1f}")
    with m3:
        st.metric("Active days (14 d)", str(active_14))
    with m4:
        st.metric("Active days (28 d)", str(active_28))

    st.markdown(_training_regularity_summary_line(label))


def _rel_session_trend(
    cur: float, prev: float, *, higher_is_better: bool
) -> str:
    """rising / stable / softening vs prior (deterministic thresholds)."""
    if prev <= 0:
        return "insufficient_history"
    if higher_is_better:
        rel = (cur - prev) / prev
    else:
        rel = (prev - cur) / prev
    if rel > 0.02:
        return "rising"
    if rel < -0.02:
        return "softening"
    return "stable"


def _work_level_metric_display(row: dict) -> tuple[str, str]:
    """One primary metric for a work-family row: W/HR, else power, else pace."""
    wh = _indicator_metric_float(row.get("work_w_per_hr"))
    if wh is not None:
        return "W/HR", f"{wh:.2f}"
    pwr = _indicator_metric_float(row.get("work_mean_power_w"))
    if pwr is not None:
        return "Power", format_power_w(pwr)
    pace = _indicator_metric_float(row.get("work_mean_pace_sec_per_km"))
    if pace is not None and pace > 0:
        return "Pace", format_pace(pace)
    return "—", "—"


def _work_session_level_trend(latest: dict, prior: dict | None) -> str:
    if prior is None:
        return "insufficient_history"
    for key, higher in (
        ("work_w_per_hr", True),
        ("work_mean_power_w", True),
        ("work_mean_pace_sec_per_km", False),
    ):
        c = _indicator_metric_float(latest.get(key))
        p = _indicator_metric_float(prior.get(key))
        if c is None or p is None:
            continue
        if key == "work_mean_pace_sec_per_km" and (c <= 0 or p <= 0):
            continue
        return _rel_session_trend(c, p, higher_is_better=higher)
    return "insufficient_history"


def _easy_row_w_per_hr(row: dict) -> float | None:
    ap = _indicator_metric_float(row.get("avg_power"))
    ah = _indicator_metric_float(row.get("avg_hr"))
    if ap is None or ah is None or ah <= 0:
        return None
    return ap / ah


def _easy_level_metric_display(row: dict) -> tuple[str, str]:
    wh = _easy_row_w_per_hr(row)
    if wh is not None:
        return "W/HR", f"{wh:.2f}"
    pace = _indicator_metric_float(row.get("avg_pace_sec_km"))
    if pace is not None and pace > 0:
        return "Pace", format_pace(pace)
    return "—", "—"


def _easy_session_level_trend(latest: dict, prior: dict | None) -> str:
    if prior is None:
        return "insufficient_history"
    lc = _easy_row_w_per_hr(latest)
    lp = _easy_row_w_per_hr(prior)
    if lc is not None and lp is not None:
        return _rel_session_trend(lc, lp, higher_is_better=True)
    c = _indicator_metric_float(latest.get("avg_pace_sec_km"))
    p = _indicator_metric_float(prior.get("avg_pace_sec_km"))
    if c is not None and p is not None and c > 0 and p > 0:
        return _rel_session_trend(c, p, higher_is_better=False)
    return "insufficient_history"


def _trend_vs_prior_caption(trend: str) -> str:
    return {
        "rising": "vs prior: better",
        "stable": "vs prior: similar",
        "softening": "vs prior: softer",
        "insufficient_history": "vs prior: not enough history",
    }.get(trend, "")


def _current_level_baseline_summary_line(
    t_th: str, t_vo2: str, t_ea: str
) -> str:
    """One deterministic sentence (no LLM)."""

    def th_phrase(t: str) -> str:
        if t == "insufficient_history":
            return "threshold has no prior session to compare yet"
        if t == "rising":
            return "current threshold level is rising"
        if t == "stable":
            return "threshold work is stable"
        return "threshold work has softened versus the prior session"

    def vo2_phrase(t: str) -> str:
        if t == "insufficient_history":
            return "VO2 work has no prior session to compare yet"
        if t == "rising":
            return "VO2 is strong"
        if t == "stable":
            return "VO2 is stable"
        return "VO2 work has softened versus the prior session"

    def ea_phrase(t: str) -> str:
        if t == "insufficient_history":
            return "easy aerobic base has no prior session to compare yet"
        if t == "rising":
            return "aerobic base is improving"
        if t == "stable":
            return "aerobic base is stable"
        return "easy aerobic base has softened versus the prior session"

    return (
        f"{th_phrase(t_th).capitalize()}, {vo2_phrase(t_vo2)}, and {ea_phrase(t_ea)}."
    )


def render_current_level_personal_baseline(db_path: Path) -> None:
    """Latest threshold / VO2 / easy session vs prior in-family row (deterministic)."""
    st.subheader("Current level / personal baseline")
    st.caption(
        "Latest session in each domain vs the previous one in the same family "
        "(intervals: W/HR, else power, else pace; easy: W/HR or pace)."
    )
    try:
        conn = connect(str(db_path))
        try:
            th_hist = fetch_dedup_work_family_session_history(conn, "threshold_session")
            vo2_hist = fetch_dedup_work_family_session_history(conn, "vo2max_session")
            ea_hist = fetch_dedup_easy_aerobic_run_history(conn)
        finally:
            conn.close()
    except Exception as exc:
        st.caption(f"Current level snapshot unavailable: {exc}")
        return

    th_last = th_hist[-1] if th_hist else None
    th_prior = th_hist[-2] if len(th_hist) >= 2 else None
    vo2_last = vo2_hist[-1] if vo2_hist else None
    vo2_prior = vo2_hist[-2] if len(vo2_hist) >= 2 else None
    ea_last = ea_hist[-1] if ea_hist else None
    ea_prior = ea_hist[-2] if len(ea_hist) >= 2 else None

    t_th = _work_session_level_trend(th_last, th_prior) if th_last else "insufficient_history"
    t_vo2 = _work_session_level_trend(vo2_last, vo2_prior) if vo2_last else "insufficient_history"
    t_ea = _easy_session_level_trend(ea_last, ea_prior) if ea_last else "insufficient_history"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Current threshold level**")
        if th_last is None:
            st.caption("No threshold family sessions in history yet.")
        else:
            lab, val = _work_level_metric_display(th_last)
            st.markdown(f"{lab} **{val}**")
            st.caption(f"Latest · {str(th_last.get('run_date') or '-')}")
            st.caption(_trend_vs_prior_caption(t_th))

    with c2:
        st.markdown("**Current VO2 level**")
        if vo2_last is None:
            st.caption("No VO2 family sessions in history yet.")
        else:
            lab, val = _work_level_metric_display(vo2_last)
            st.markdown(f"{lab} **{val}**")
            st.caption(f"Latest · {str(vo2_last.get('run_date') or '-')}")
            st.caption(_trend_vs_prior_caption(t_vo2))

    with c3:
        st.markdown("**Current aerobic base**")
        if ea_last is None:
            st.caption("No easy or recovery runs in history yet.")
        else:
            lab, val = _easy_level_metric_display(ea_last)
            st.markdown(f"{lab} **{val}**")
            st.caption(f"Latest · {str(ea_last.get('run_date') or '-')}")
            st.caption(_trend_vs_prior_caption(t_ea))

    st.markdown(_current_level_baseline_summary_line(t_th, t_vo2, t_ea))


def _dashboard_work_family_caption(fam_display: str, cmp: dict) -> str:
    """One deterministic sentence from latest-vs-prior family compare (no LLM)."""
    if cmp.get("insufficient_history"):
        return (
            f"{fam_display} progression: not enough sessions in this work family yet "
            "(need at least two comparable runs)."
        )
    lbl = _family_compare_overview_label(cmp)
    mm = cmp.get("metrics") or {}
    pace = (mm.get("work_mean_pace_sec_per_km") or {}).get("status")
    pwr = (mm.get("work_mean_power_w") or {}).get("status")
    whr = (mm.get("work_w_per_hr") or {}).get("status")

    if lbl == "Improving":
        return f"{fam_display} performance is improving."
    if lbl == "Declining":
        return f"{fam_display} performance has softened versus the prior session."
    if lbl == "Mixed":
        return (
            f"{fam_display} performance is mixed across pace, power, and W/HR "
            "versus the prior session."
        )
    # Stable — add detail when a metric moved but aggregate stayed stable
    bits: list[str] = []
    if pwr == "higher":
        bits.append("higher power")
    elif pwr == "lower":
        bits.append("lower power")
    if pace == "faster":
        bits.append("faster pace")
    elif pace == "slower":
        bits.append("slower pace")
    if whr == "better":
        bits.append("better W/HR")
    elif whr == "worse":
        bits.append("lower W/HR")
    if bits:
        return f"{fam_display} performance is stable with {', '.join(bits)}."
    return f"{fam_display} performance is stable."


def _chart_y_tooltip_format(y_col: str) -> str:
    """Altair tooltip number format: integer watts for power series."""
    if y_col in {"work_mean_power_w", "avg_power_w", "avg_power"}:
        return ".0f"
    if y_col in {"avg_hr", "work_mean_hr_avg"}:
        return ".0f"
    return ".2f"


def _dashboard_chart_tooltip_label(y_col: str) -> str:
    """Tooltip column title when Y-axis has no unit title (matches compact chart row style)."""
    if y_col in ("work_mean_pace_sec_per_km", "avg_pace_sec_km"):
        return "Pace"
    if y_col == "work_mean_power_w":
        return "Power (W)"
    if y_col == "avg_hr":
        return "Heart rate (bpm)"
    if y_col in ("work_w_per_hr", "w_per_hr"):
        return "W / HR"
    if y_col == "distance_km":
        return "Distance (km)"
    if y_col in {"avg_power_w", "avg_power"}:
        return "Average power (W)"
    return "Value"


def _is_pace_sec_per_km_column(y_col: str) -> bool:
    return y_col in ("work_mean_pace_sec_per_km", "avg_pace_sec_km")


def _pace_tooltip_series(sec_series: pd.Series) -> pd.Series:
    """Human-readable m:ss min/km for Altair tooltips (main dashboard charts)."""

    def _one(v) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        try:
            return format_pace(float(v))
        except (TypeError, ValueError):
            return ""

    return sec_series.map(_one)


# Seconds/km on scale; tick labels shown as m:ss (no raw 350/600-style numbers).
_PACE_AXIS_LABEL_EXPR = (
    "datum.value == null ? '' : "
    "(floor(datum.value / 60) + ':' + "
    "((floor(datum.value) % 60 < 10) ? '0' : '') + (floor(datum.value) % 60))"
)


def _pace_axis_sec_per_km() -> alt.Axis:
    """Y-axis for pace: sec/km internally; tick labels as m:ss (same title=None style as Power / W·HR).

    Tooltips use ``pace_tooltip`` (``m:ss min/km`` via ``format_pace``).
    """
    return alt.Axis(title=None, labelExpr=_PACE_AXIS_LABEL_EXPR)


def _dashboard_work_family_line_chart(
    chart_df: pd.DataFrame,
    *,
    y_col: str,
    title: str,
) -> None:
    if chart_df.empty or y_col not in chart_df.columns:
        return
    if chart_df[y_col].notna().sum() == 0:
        st.caption(f"{title}: no numeric data")
        return
    _ok = chart_df["run_date_dt"].notna() & chart_df[y_col].notna()
    if int(_ok.sum()) <= 1:
        st.caption(
            f"{title}: one comparable session so far—trend lines appear after more "
            "dated points in this family."
        )
        return
    df = chart_df
    if _is_pace_sec_per_km_column(y_col):
        df = chart_df.copy()
        if "pace_tooltip" not in df.columns:
            df["pace_tooltip"] = _pace_tooltip_series(df[y_col])
        y_enc = alt.Y(
            f"{y_col}:Q",
            axis=_pace_axis_sec_per_km(),
            scale=alt.Scale(zero=False),
        )
        tips: list = [
            alt.Tooltip("run_date_dt:T", title="Date"),
            alt.Tooltip("pace_tooltip:N", title="Pace"),
        ]
    else:
        y_enc = alt.Y(f"{y_col}:Q", title=None, scale=alt.Scale(zero=False))
        _tf = _chart_y_tooltip_format(y_col)
        _ttl = _dashboard_chart_tooltip_label(y_col)
        tips = [
            alt.Tooltip("run_date_dt:T", title="Date"),
            alt.Tooltip(f"{y_col}:Q", title=_ttl, format=_tf),
        ]
    base = alt.Chart(df).encode(
        x=alt.X(
            "run_date_dt:T",
            axis=alt.Axis(format="%b %Y", title=None),
        ),
        y=y_enc,
    )
    line = base.mark_line()
    points = base.mark_circle(size=45, color="#4c78a8")
    hover = base.mark_circle(size=240, opacity=0).encode(tooltip=tips)
    ch = (
        (line + points + hover)
        .encode(
            x=alt.X(
                "run_date_dt:T",
                axis=alt.Axis(format="%b %Y", title=None),
            ),
            y=y_enc,
        )
        .properties(height=110, title=title)
        .configure_axis(labelFontSize=9, titleFontSize=10)
        .configure_title(fontSize=12)
    )
    st.altair_chart(ch, width="stretch")


def _main_dashboard_trends_pace_chart(chart_df: pd.DataFrame) -> None:
    """Average pace over time (main dashboard): section header + m:ss min/km tooltips; no Y title."""
    if chart_df.empty or "avg_pace_sec_km" not in chart_df.columns:
        return
    if chart_df["avg_pace_sec_km"].notna().sum() == 0:
        st.caption("Average pace: no numeric data")
        return
    tdf = chart_df[["run_date_dt", "avg_pace_sec_km"]].copy()
    tdf["pace_tooltip"] = _pace_tooltip_series(tdf["avg_pace_sec_km"])
    base = alt.Chart(tdf).encode(
        x=alt.X("run_date_dt:T", axis=alt.Axis(format="%b %Y", title=None)),
        y=alt.Y(
            "avg_pace_sec_km:Q",
            axis=_pace_axis_sec_per_km(),
            scale=alt.Scale(zero=False),
        ),
    )
    ch = (
        (
            base.mark_line()
            + base.mark_circle(size=45, color="#4c78a8")
            + base.mark_circle(size=260, opacity=0).encode(
                tooltip=[
                    alt.Tooltip("run_date_dt:T", title="Date"),
                    alt.Tooltip("pace_tooltip:N", title="Pace"),
                ]
            )
        )
        .properties(height=200)
        .configure_axis(labelFontSize=10, titleFontSize=11)
    )
    st.altair_chart(ch, width="stretch")


def _filter_implausible_trend_rows(chart_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Chart-only plausibility filter for long-range trends (does not modify stored data)."""
    if chart_df.empty:
        return chart_df, 0

    df = chart_df.copy()
    n = len(df)
    bad = pd.Series(False, index=df.index, dtype=bool)

    dist = pd.to_numeric(df.get("distance_km"), errors="coerce")
    dur = pd.to_numeric(df.get("duration_sec"), errors="coerce")
    pace = pd.to_numeric(df.get("avg_pace_sec_km"), errors="coerce")
    hr = pd.to_numeric(df.get("avg_hr"), errors="coerce")
    tt = (
        df["training_type"].fillna("").astype(str).str.strip().str.lower()
        if "training_type" in df.columns
        else pd.Series([""] * n, index=df.index)
    )
    is_hard = tt.isin(_WEEKLY_HARD_TRAINING_TYPES)
    is_easy_rec = tt.isin(_WEEKLY_EASY_RECOVERY_TYPES)

    # Basic invalid activity rows (only when values exist and are non-positive).
    bad |= dist.notna() & (dist <= 0)
    bad |= dur.notna() & (dur <= 0)
    bad |= pace.notna() & (pace <= 0)

    # Very fast pace outliers for meaningful run distances.
    bad |= pace.notna() & dist.notna() & (dist >= 1.5) & (pace < 120)

    # Easy/recovery: sustained “race-adjacent” average pace over real distance is not credible here.
    bad |= (
        pace.notna()
        & dist.notna()
        & is_easy_rec
        & (dist >= 3.0)
        & (pace < 185)
    )

    # Non-interval types: very fast average over distance with low HR (bad watch / bad typing).
    bad |= (
        pace.notna()
        & dist.notna()
        & hr.notna()
        & ~is_hard
        & (dist >= 5.0)
        & (pace < 195)
        & (hr < 138)
    )

    # Ultra-fast pace paired with low average HR across longer runs is usually malformed history.
    bad |= (
        pace.notna()
        & dist.notna()
        & hr.notna()
        & (dist >= 8.0)
        & (pace < 170)
        & (hr < 145)
    )

    # Extreme slow average pace (GPS / moving-time glitches) — dominates long-range pace axis.
    bad |= pace.notna() & dist.notna() & (dist >= 1.5) & (pace > 780)

    # Cross-check persisted pace against duration/distance-derived pace when all are present.
    calc_pace = (dur / dist).where(dist.notna() & dur.notna() & (dist > 0))
    pace_mismatch = (
        pace.notna()
        & calc_pace.notna()
        & (pace > 0)
        & ((pace - calc_pace).abs() / pace > 0.35)
    )
    bad |= pace_mismatch

    filtered = df.loc[~bad].copy()
    return filtered, int(bad.sum())


def render_main_dashboard_trends_charts(chart_df: pd.DataFrame) -> None:
    """Distance, HR, pace, power — main dashboard Trends block (inside expander)."""
    plaus_df, excluded = _filter_implausible_trend_rows(chart_df)
    dated = plaus_df.dropna(subset=["run_date_dt"])
    if len(dated) <= 1:
        st.info(
            "These charts need **at least two dated runs** to show change over time. "
            "With a single activity, there is nothing to connect yet—keep importing and check back."
        )
        return
    if excluded > 0:
        st.caption("Some implausible historical outliers were excluded from this chart.")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Distance over time**")
        st.line_chart(
            dated.set_index("run_date_dt")[["distance_km"]],
            width="stretch",
        )

        st.markdown("**Average heart rate over time**")
        st.line_chart(
            dated.set_index("run_date_dt")[["avg_hr"]],
            width="stretch",
        )

    with chart_col2:
        st.markdown("**Pace**")
        _main_dashboard_trends_pace_chart(dated)

        st.markdown("**Average power (W)**")
        _pwr = dated[["run_date_dt", "avg_power"]].copy()
        _pwr["avg_power_w"] = pd.to_numeric(_pwr["avg_power"], errors="coerce").round()
        st.line_chart(
            _pwr.set_index("run_date_dt")[["avg_power_w"]],
            width="stretch",
        )


def render_work_family_progression_panels(db_path: Path) -> None:
    """Dashboard-level threshold, VO2, and easy/aerobic metrics over time (deterministic)."""
    try:
        conn = connect(str(db_path))
        try:
            th_hist = fetch_dedup_work_family_session_history(conn, "threshold_session")
            vo2_hist = fetch_dedup_work_family_session_history(conn, "vo2max_session")
            th_cmp = compare_threshold_session_family_latest_vs_prior(conn)
            vo2_cmp = compare_vo2max_family_latest_vs_prior(conn)
            ea_hist = fetch_dedup_easy_aerobic_run_history(conn)
            aer_cmp = derive_easy_aerobic_efficiency_trend(conn)
        finally:
            conn.close()
    except Exception as exc:
        st.caption(f"Work family progression unavailable: {exc}")
        return

    st.subheader("Threshold performance")
    st.caption("Work intervals from stored segments (deduplicated).")
    if len(th_hist) < 2:
        st.info(
            "Threshold work family: not enough history yet—import more threshold-style "
            "runs with work segments to see progression."
        )
    else:
        rows_tbl: list[dict[str, str]] = []
        for r in reversed(th_hist):
            rows_tbl.append(
                {
                    "Date": str(r.get("run_date") if r.get("run_date") is not None else "-"),
                    "Pace (min/km)": _format_work_metric_cell(
                        r.get("work_mean_pace_sec_per_km"), kind="pace"
                    ),
                    "Power (W)": _format_work_metric_cell(
                        r.get("work_mean_power_w"), kind="int"
                    ),
                    "W/HR": _format_work_metric_cell(r.get("work_w_per_hr"), kind="w_per_hr"),
                }
            )
        st.dataframe(
            pd.DataFrame(rows_tbl),
            width="stretch",
            hide_index=True,
        )
        cdf = _dashboard_work_family_chart_df(th_hist)
        if not cdf["run_date_dt"].isna().all():
            c1, c2, c3 = st.columns(3)
            with c1:
                _dashboard_work_family_line_chart(
                    cdf,
                    y_col="work_mean_pace_sec_per_km",
                    title="Pace",
                )
            with c2:
                _dashboard_work_family_line_chart(
                    cdf,
                    y_col="work_mean_power_w",
                    title="Power",
                )
            with c3:
                _dashboard_work_family_line_chart(
                    cdf,
                    y_col="work_w_per_hr",
                    title="W / HR",
                )
        else:
            st.caption("Chart needs valid run dates in this history.")
    st.caption(_dashboard_work_family_caption("Threshold", th_cmp))

    st.subheader("VO2 performance")
    st.caption("Work intervals from stored segments (deduplicated).")
    if len(vo2_hist) < 2:
        st.info(
            "VO2 work family: not enough history yet—import more VO2-style "
            "runs with work segments to see progression."
        )
    else:
        rows_tbl_v: list[dict[str, str]] = []
        for r in reversed(vo2_hist):
            rows_tbl_v.append(
                {
                    "Date": str(r.get("run_date") if r.get("run_date") is not None else "-"),
                    "Pace (min/km)": _format_work_metric_cell(
                        r.get("work_mean_pace_sec_per_km"), kind="pace"
                    ),
                    "Power (W)": _format_work_metric_cell(
                        r.get("work_mean_power_w"), kind="int"
                    ),
                    "W/HR": _format_work_metric_cell(r.get("work_w_per_hr"), kind="w_per_hr"),
                }
            )
        st.dataframe(
            pd.DataFrame(rows_tbl_v),
            width="stretch",
            hide_index=True,
        )
        cdfv = _dashboard_work_family_chart_df(vo2_hist)
        if not cdfv["run_date_dt"].isna().all():
            v1, v2, v3 = st.columns(3)
            with v1:
                _dashboard_work_family_line_chart(
                    cdfv,
                    y_col="work_mean_pace_sec_per_km",
                    title="Pace",
                )
            with v2:
                _dashboard_work_family_line_chart(
                    cdfv,
                    y_col="work_mean_power_w",
                    title="Power",
                )
            with v3:
                _dashboard_work_family_line_chart(
                    cdfv,
                    y_col="work_w_per_hr",
                    title="W / HR",
                )
        else:
            st.caption("Chart needs valid run dates in this history.")
    st.caption(_dashboard_work_family_caption("VO2", vo2_cmp))

    st.subheader("Easy / Aerobic performance")
    st.caption(
        "Easy and recovery runs (deduplicated). W/HR = average power ÷ average HR when both exist."
    )
    if len(ea_hist) < 2:
        st.info(
            "Easy and recovery aerobic history: not enough sessions yet—import more easy or "
            "recovery runs to see progression."
        )
    else:
        rows_ea: list[dict[str, str]] = []
        for r in reversed(ea_hist):
            ap = r.get("avg_power")
            ah = r.get("avg_hr")
            wh_raw = None
            try:
                if ap is not None and ah is not None and float(ah) > 0:
                    wh_raw = float(ap) / float(ah)
            except (TypeError, ValueError):
                wh_raw = None
            rows_ea.append(
                {
                    "Date": str(r.get("run_date") if r.get("run_date") is not None else "-"),
                    "Type": str(r.get("training_type") or "-"),
                    "Pace (min/km)": _format_work_metric_cell(
                        r.get("avg_pace_sec_km"), kind="pace"
                    ),
                    "Avg HR": format_int(r.get("avg_hr"))
                    if r.get("avg_hr") is not None
                    and not (isinstance(r.get("avg_hr"), float) and pd.isna(r.get("avg_hr")))
                    else "-",
                    "Power (W)": format_int(r.get("avg_power"))
                    if r.get("avg_power") is not None
                    and not (
                        isinstance(r.get("avg_power"), float) and pd.isna(r.get("avg_power"))
                    )
                    else "-",
                    "W/HR": format_number(wh_raw, 2)
                    if wh_raw is not None
                    else "-",
                }
            )
        st.dataframe(
            pd.DataFrame(rows_ea),
            width="stretch",
            hide_index=True,
        )
        ecdf = _dashboard_easy_aerobic_chart_df(ea_hist)
        if not ecdf["run_date_dt"].isna().all():
            e1, e2, e3 = st.columns(3)
            with e1:
                _dashboard_work_family_line_chart(
                    ecdf,
                    y_col="avg_pace_sec_km",
                    title="Pace",
                )
            with e2:
                _dashboard_work_family_line_chart(
                    ecdf,
                    y_col="avg_hr",
                    title="Heart rate",
                )
            with e3:
                _dashboard_work_family_line_chart(
                    ecdf,
                    y_col="w_per_hr",
                    title="W / HR",
                )
        else:
            st.caption("Chart needs valid run dates in this history.")
    st.caption(_easy_aerobic_dashboard_caption(aer_cmp))


def _render_work_family_membership_one(payload: dict[str, object]) -> None:
    """Single-family lines + optional recent table from :func:`work_family_membership_diagnostic`."""
    if not payload.get("ok"):
        st.caption("Unavailable.")
        return
    n_scan = int(payload["n_runs_with_work_segments"])
    n_raw = int(payload["n_raw_in_target_family"])
    n_ded = int(payload["n_deduped_in_target_family"])
    n_ded_collapsed = int(payload["n_collapsed_by_dedup"])
    n_other = int(payload["n_classified_other_family"])
    n_noclass = int(payload["n_not_classified"])
    n_agg = int(payload["n_aggregate_failed_after_family_match"])
    st.caption(
        f"Scanned **{n_scan}** run(s) with ≥1 stored work segment. In this family: **{n_raw}** before "
        f"dedup, **{n_ded}** after dedup (**{n_ded_collapsed}** merged as duplicate activity). "
        f"Excluded from this family: **{n_other}** classified to another family / other interval, "
        f"**{n_noclass}** could not derive family from work segments, **{n_agg}** missing work aggregate."
    )
    rows = payload.get("included_recent") or []
    if rows:
        st.caption("Most recent included sessions (deduped):")
        _rows_labelled: list[dict[str, object]] = []
        _fam_hint = payload.get("work_session_family")
        for r in rows:
            _tt = _training_type_with_family_hint(r.get("training_type"), _fam_hint)
            _rows_labelled.append(
                {
                    "Run label": format_run_display_label(
                        r.get("run_date"),
                        _tt,
                        r.get("distance_km"),
                        str(r.get("work_block_label") or "").strip() or None,
                    ),
                    "Date": r.get("run_date"),
                    "Run ID": r.get("run_id"),
                    "Work block label": r.get("work_block_label"),
                }
            )
        tbl = pd.DataFrame(_rows_labelled)
        st.dataframe(tbl, hide_index=True, width="stretch")
    else:
        st.caption("No sessions in this family yet.")


def render_work_family_membership_diagnostics(db_path: Path) -> None:
    """Secondary view: why threshold/VO2 dashboard counts match a small deduped family list."""
    st.markdown("#### Work-family membership (diagnostic)")
    st.caption(
        "Top-level threshold / VO2 totals use deduplicated **work-family** rows from stored "
        "**work** segments and deterministic interval rules—not `training_type` alone."
    )
    try:
        conn = connect(str(db_path))
        try:
            th = work_family_membership_diagnostic(conn, "threshold_session")
            vo2 = work_family_membership_diagnostic(conn, "vo2max_session")
        finally:
            conn.close()
    except Exception as exc:
        st.caption(f"Diagnostic unavailable: {exc}")
        return
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Threshold (`threshold_session`)**")
        _render_work_family_membership_one(th)
    with c2:
        st.markdown("**VO2 (`vo2max_session`)**")
        _render_work_family_membership_one(vo2)


def render_work_segment_family_distribution_diagnostic(db_path: Path) -> None:
    """How work-segment runs split across derived families (same rules as classification; read-only)."""
    st.markdown("#### Work-segment family distribution (diagnostic)")
    st.caption(
        "Each run with ≥1 stored **work** segment gets one derived `work_session_family` "
        "(threshold vs VO2 vs other interval). Totals are raw counts, not deduplicated."
    )
    try:
        conn = connect(str(db_path))
        try:
            dist = work_segment_family_distribution_diagnostic(conn)
        finally:
            conn.close()
    except Exception as exc:
        st.caption(f"Diagnostic unavailable: {exc}")
        return
    if not dist.get("ok"):
        st.caption("Unavailable.")
        return
    n_scan = int(dist["n_runs_with_work_segments"])
    n_nd = int(dist["n_not_derived"])
    st.caption(f"Runs with work segments scanned: **{n_scan}** · Could not derive family: **{n_nd}**.")
    by = dist.get("by_family") or {}
    preferred = ("threshold_session", "vo2max_session", "other_interval_session")
    lines: list[str] = []
    for key in preferred:
        lines.append(f"- **{key}:** {int(by.get(key, 0))}")
    rest_keys = sorted(k for k in by if k not in preferred)
    for key in rest_keys:
        lines.append(f"- **{key}:** {int(by[key])}")
    if lines:
        st.markdown("\n".join(lines))
    other_recent = dist.get("other_interval_recent") or []
    if other_recent:
        st.caption(
            f"Most recent **other_interval_session** rows (up to {len(other_recent)} shown; inspect labels):"
        )
        _other_labelled: list[dict[str, object]] = []
        for r in other_recent:
            _tt = _training_type_with_family_hint(
                r.get("training_type"),
                "other_interval_session",
            )
            _other_labelled.append(
                {
                    "Run label": format_run_display_label(
                        r.get("run_date"),
                        _tt,
                        r.get("distance_km"),
                        str(r.get("work_block_label") or "").strip() or None,
                    ),
                    "Date": r.get("run_date"),
                    "Run ID": r.get("run_id"),
                    "Training type": format_training_type_label(_tt),
                    "Work block label": r.get("work_block_label"),
                }
            )
        tbl = pd.DataFrame(_other_labelled)
        st.dataframe(tbl, hide_index=True, width="stretch")
    else:
        st.caption("No `other_interval_session` rows in this scan.")


def render_performance_overview(df_runs: pd.DataFrame, db_path: Path) -> None:
    """Top-level deterministic snapshot (no LLM)."""
    st.markdown("### Performance overview")
    st.caption(
        "Snapshot from your latest run’s signals, threshold and VO2 family comparisons, "
        "and easy/recovery efficiency vs the prior easy or recovery run."
    )
    if df_runs.empty:
        st.info("Add analyzed runs to see a performance overview.")
        return

    latest = df_runs.iloc[0]
    overall = _overall_trend_card_label(latest.get("trend_label"))
    load_lbl = _load_fatigue_overview_label(latest)
    th_lbl = "Unavailable"
    vo2_lbl = "Unavailable"
    easy_lbl = "Unavailable"
    try:
        conn = connect(str(db_path))
        try:
            insight = build_interval_family_insight_summary(conn)
            th_lbl = _family_compare_overview_label(
                insight.get("threshold_latest_vs_prior") or {}
            )
            vo2_lbl = _family_compare_overview_label(
                insight.get("vo2max_latest_vs_prior") or {}
            )
            aer = derive_easy_aerobic_efficiency_trend(conn)
            easy_lbl = str(aer.get("easy_aerobic_signal") or "Unavailable")
        finally:
            conn.close()
    except Exception:
        pass

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Overall trend", _overview_metric_display(overall))
    with c2:
        st.metric("Threshold trend", _overview_metric_display(th_lbl))
    with c3:
        st.metric("VO2 trend", _overview_metric_display(vo2_lbl))
    with c4:
        st.metric("Easy / Aerobic trend", _overview_metric_display(easy_lbl))

    expl = _performance_overview_explanation_lines(
        overall, th_lbl, vo2_lbl, easy_lbl, load_lbl
    )
    if expl:
        st.markdown("##### What this means")
        st.markdown("\n".join(f"- {line}" for line in expl))

    st.markdown(
        _performance_overview_summary_md(
            _overview_metric_display(overall),
            _overview_metric_display(th_lbl),
            _overview_metric_display(vo2_lbl),
            _overview_metric_display(easy_lbl),
            _overview_metric_display(load_lbl),
        )
    )


def format_import_summary_lines(summary: dict) -> list[str]:
    """User-facing lines from ingest_folder summary dict / last_import_summary.json."""
    u = int(summary.get("uploaded_files") or 0)
    n = int(summary.get("new_analyzed") or 0)
    d = int(summary.get("duplicate_cached") or 0)
    sk = int(summary.get("skipped_cache_miss") or 0)
    err = int(summary.get("errors") or 0)
    llm_calls = int(summary.get("llm_api_calls") or 0)
    llm_reu = int(summary.get("llm_reused_stored") or 0)
    use_llm = bool(summary.get("use_llm_requested"))

    lines: list[str] = []
    if u == 1:
        lines.append(f"**{u}** file uploaded.")
    else:
        lines.append(f"**{u}** files uploaded.")

    if n == 1:
        lines.append(f"**{n}** new run analyzed end-to-end.")
    elif n:
        lines.append(f"**{n}** new runs analyzed end-to-end.")

    if d == 1:
        lines.append(
            f"**{d}** duplicate activity reused from cache (no new analysis; stored data shown)."
        )
    elif d:
        lines.append(
            f"**{d}** duplicate activities reused from cache (no new analysis; stored data shown)."
        )

    if sk == 1:
        lines.append(
            f"**{sk}** file skipped (duplicate activity present in DB but cache payload missing)."
        )
    elif sk:
        lines.append(
            f"**{sk}** files skipped (duplicate activity present in DB but cache payload missing)."
        )

    if err == 1:
        lines.append(f"**{err}** file failed to import.")
    elif err:
        lines.append(f"**{err}** files failed to import.")

    if use_llm:
        if llm_calls == 1:
            lc = f"**{llm_calls}** AI summary generated (API call)"
        else:
            lc = f"**{llm_calls}** AI summaries generated (API calls)"
        if llm_reu == 1:
            lr = f"**{llm_reu}** duplicate reused a saved summary (no new call)"
        elif llm_reu:
            lr = f"**{llm_reu}** duplicates reused saved summaries (no new calls)"
        else:
            lr = None
        if lr:
            lines.append(f"{lc}; {lr}.")
        else:
            lines.append(f"{lc}.")
    else:
        lines.append("AI summaries were **off** for this import (rule-based text only; no API calls).")

    return lines


def load_runs_table_data(db_path: str, out_dir: Path) -> pd.DataFrame:
    """Prefer live SQLite data; fallback to last ingest JSON when DB is empty/unavailable."""
    try:
        db_df = load_runs_from_db(db_path)
        if db_df is not None and not db_df.empty:
            return _ensure_run_table_columns(db_df)
    except Exception:
        pass

    jp = out_dir / "runs_normalized.json"
    if jp.is_file():
        try:
            raw = json.loads(jp.read_text(encoding="utf-8"))
            if isinstance(raw, list) and raw:
                return _ensure_run_table_columns(pd.DataFrame(raw))
        except Exception:
            pass
    return _ensure_run_table_columns(pd.DataFrame())


@st.cache_data
def load_work_block_label_map(db_path: str, run_ids: tuple[str, ...]) -> dict[str, str]:
    """Selector helper: compact work-block labels by run id where available."""
    labels: dict[str, str] = {}
    if not run_ids:
        return labels
    conn = connect(db_path)
    try:
        for rid in run_ids:
            rid_s = str(rid).strip()
            if not rid_s:
                continue
            wbl = derive_work_block_label_for_run(conn, rid_s)
            if not isinstance(wbl, dict):
                continue
            label = str(wbl.get("work_block_label") or "").strip()
            if not label or label.lower() == "no work blocks":
                continue
            labels[rid_s] = label
        return labels
    finally:
        conn.close()


def load_llm_trace(run_id: str) -> dict | None:
    trace_path = TRACE_DIR / f"{run_id}.json"
    if not trace_path.exists():
        return None
    try:
        return json.loads(trace_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_llm_trace_for_run_row(selected_run_id: str, run_row: pd.Series) -> dict | None:
    """Load trace JSON; for duplicate_cached rows, fall back to the canonical run id if needed."""
    payload = load_llm_trace(selected_run_id)
    if payload is not None:
        return payload
    if _scalar_str(run_row, "status") != "duplicate_cached":
        return None
    canon = run_row.get("cached_from_run_id")
    if canon is None or (isinstance(canon, float) and pd.isna(canon)) or str(canon).strip() == "":
        canon = run_row.get("run_id")
    if canon is None or (isinstance(canon, float) and pd.isna(canon)):
        return None
    cid = str(canon).strip()
    if cid and cid != selected_run_id:
        return load_llm_trace(cid)
    return None


def _llm_meta_is_placeholder(meta: dict) -> bool:
    """True when no structured add-ons were recorded (older trace or unavailable bundle)."""
    if meta == LLM_CONTEXT_METADATA_UNAVAILABLE:
        return True
    for key in (
        "deterministic_run_takeaway_used",
        "interval_insight_used",
        "family_history_used",
        "recommendation_summary_used",
    ):
        if str(meta.get(key) or "").strip().lower() == "yes":
            return False
    fam = meta.get("work_session_family")
    if fam is not None and not (isinstance(fam, float) and pd.isna(fam)) and str(fam).strip():
        return False
    return True


def _llm_grounding_at_a_glance(meta: dict | None) -> str:
    """One short line for whether prompt context / grounding was present."""
    if not isinstance(meta, dict) or not meta:
        return "Prompt context: not recorded for this trace."
    if _llm_meta_is_placeholder(meta):
        return (
            "Prompt context: no add-on blocks (older import, or the deterministic bundle was not built)."
        )
    parts: list[str] = []
    if str(meta.get("deterministic_run_takeaway_used") or "").strip().lower() == "yes":
        parts.append("run takeaway")
    if str(meta.get("interval_insight_used") or "").strip().lower() == "yes":
        parts.append("interval insight")
    if str(meta.get("family_history_used") or "").strip().lower() == "yes":
        parts.append("session family history")
    if str(meta.get("recommendation_summary_used") or "").strip().lower() == "yes":
        parts.append("recommendation summary")
    fam = meta.get("work_session_family")
    fam_bit = ""
    if fam is not None and not (isinstance(fam, float) and pd.isna(fam)) and str(fam).strip():
        fam_bit = f" · Session family: **{str(fam).strip()}**"
    if parts:
        return "Prompt context: " + ", ".join(parts) + "." + fam_bit
    return "Prompt context: metadata present; no add-on blocks flagged as included." + fam_bit


def _friendly_llm_trace_status(status: object, used_llm: bool) -> str:
    s = str(status or "").strip() or "—"
    if s == "success" and used_llm:
        return "Finished (model response used)"
    if s == "success":
        return "Finished"
    if s == "fallback":
        return "Template text (model not available)"
    if s == "error_fallback":
        return "Template text after a model error"
    if s == "disabled_fallback":
        return "Template text (AI summaries off)"
    return s.replace("_", " ")


def _llm_context_meta_display_lines(meta: dict) -> list[str]:
    """Human-readable lines for llm_context_metadata (full detail inside expanders)."""
    fam = meta.get("work_session_family")
    if fam is None or (isinstance(fam, float) and pd.isna(fam)):
        fam_s = "—"
    else:
        fam_s = str(fam).strip() or "—"

    def _yn(key: str) -> str:
        v = meta.get(key)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        return str(v)

    return [
        f"Run takeaway in prompt: {_yn('deterministic_run_takeaway_used')}",
        f"Interval insight in prompt: {_yn('interval_insight_used')}",
        f"Session family history in prompt: {_yn('family_history_used')}",
        f"Recommendation summary in prompt: {_yn('recommendation_summary_used')}",
        f"Work session family label: {fam_s}",
    ]


def format_duration(seconds):
    if pd.isna(seconds):
        return "-"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_pace(seconds_per_km):
    """Format pace for UI: clock time as min/km (same numeric basis as stored sec/km)."""
    if pd.isna(seconds_per_km):
        return "-"
    total_seconds = int(seconds_per_km)
    m = total_seconds // 60
    s = total_seconds % 60
    return f"{m}:{s:02d} min/km"


def _sanitize_legacy_pace_s_km_in_text(text: str) -> str:
    """Best-effort rewrite of older stored copy that used ``… N s/km`` for display-only."""
    if not text or not isinstance(text, str):
        return text

    def _sub_plain(m):
        try:
            return format_pace(float(m.group(1)))
        except (TypeError, ValueError):
            return m.group(0)

    out = re.sub(r"(\d+(?:\.\d+)?)\s*s/km", _sub_plain, text, flags=re.IGNORECASE)

    def _sub_paren(m):
        try:
            return format_pace(float(m.group(1)))
        except (TypeError, ValueError):
            return m.group(0)

    return re.sub(r"\(s/km\):\s*(\d+(?:\.\d+)?)", _sub_paren, out, flags=re.IGNORECASE)


def format_power_w(value) -> str:
    """Integer watts for display, or dash when missing."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    try:
        return f"{int(round(float(value)))} W"
    except (TypeError, ValueError):
        return "-"


def format_number(value, digits=2, suffix=""):
    if pd.isna(value):
        return "-"
    return f"{value:.{digits}f}{suffix}"


def format_int(value):
    if pd.isna(value):
        return "-"
    return str(int(value))


def first_non_empty(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        if str(value).strip():
            return str(value)
    return "No recommendation available."


_TRAINING_TYPE_DISPLAY_MAP: dict[str, str] = {
    "easy_run": "Easy run",
    "recovery_run": "Recovery run",
    "steady_run": "Steady run",
    "long_run": "Long run",
    "threshold_run": "Threshold run",
    "vo2_interval_session": "VO2 interval session",
    "test_session": "Benchmark session",
    "test_or_vo2_session": "Test or VO2 session",
    "race": "Race",
    "mixed_unclear": "Mixed / unclear",
    "unknown": "Unknown",
}

_LONG_AEROBIC_TRAINING_TYPES_UI: frozenset[str] = frozenset({"steady_run", "long_run"})
_LEGACY_NEXT_SESSION_EASY_DRIFT = (
    "Recent easy/recovery sessions were often executed too hard; make the next run clearly easy "
    "and use stricter pace/HR caps."
)


def format_training_type_label(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    raw = str(value).strip()
    if not raw:
        return "-"
    norm = raw.lower()
    if norm in _TRAINING_TYPE_DISPLAY_MAP:
        return _TRAINING_TYPE_DISPLAY_MAP[norm]
    return raw.replace("_", " ").strip().capitalize()


def _quick_coaching_this_session_line(run_row: pd.Series) -> str | None:
    """One line: today’s session labels only (steady/long), so ‘what next’ is not misread as critique."""
    tt = str(run_row.get("training_type") or "").strip().lower()
    if tt not in _LONG_AEROBIC_TRAINING_TYPES_UI:
        return None
    tlab = format_training_type_label(run_row.get("training_type"))
    ex = _scalar_str(run_row, "execution_quality")
    intf = _scalar_str(run_row, "intensity_label")
    return (
        f"**This session:** {tlab} · execution {ex} · intensity {intf}. "
        "These describe **today’s run** only."
    )


def _markdown_quick_coaching_what_next(run_row: pd.Series) -> str:
    """Prefer LLM ``what_next`` when present; else deterministic next_session (legacy steady/long scoping)."""
    la = _scalar_str(run_row, "load_action") or "-"
    wn_ai = str(run_row.get("llm_what_next_short") or "").strip()
    if wn_ai:
        return f"{wn_ai}\n\n**Deterministic load default:** {la}"
    ns = (_scalar_str(run_row, "next_session") or "").strip()
    tt = str(run_row.get("training_type") or "").strip().lower()
    if tt not in _LONG_AEROBIC_TRAINING_TYPES_UI:
        return f"**Next session:** {ns}\n\n**Load action:** {la}"
    if not ns:
        return f"**Next session:** —\n\n**Load action:** {la}"

    if ns == _LEGACY_NEXT_SESSION_EASY_DRIFT:
        body = (
            "**Reason (recent history, not this run):** "
            "Recent easy/recovery sessions were often executed harder than intended.\n\n"
            "**Next easy/recovery session:** "
            "Make the next easy or recovery run clearly easy with stricter pace/HR caps."
        )
        return f"{body}\n\n**Load action:** {la}"

    if "\n" in ns:
        parts = [p.strip() for p in ns.split("\n") if p.strip()]
        if len(parts) >= 2:
            body = (
                f"**Reason (recent history, not this run):** {parts[0]}\n\n"
                f"**Next easy/recovery session:** {parts[1]}"
            )
            return f"{body}\n\n**Load action:** {la}"
        return f"**Next session:** {ns}\n\n**Load action:** {la}"

    nsl = ns.lower()
    if nsl.startswith("follow the long run"):
        return f"**Next easy/recovery session:** {ns}\n\n**Load action:** {la}"

    return f"**Next session:** {ns}\n\n**Load action:** {la}"


_USER_FACING_NON_STRUCTURED_TYPES: frozenset[str] = frozenset(
    {"easy_run", "recovery_run", "long_run", "steady_run"}
)
_USER_FACING_STRUCTURED_TYPES: frozenset[str] = frozenset(
    {
        "test_session",
        "threshold_run",
        "threshold_session",
        "vo2_interval_session",
        "vo2max_session",
        "test_or_vo2_session",
    }
)


def is_user_facing_structured_run(training_type: object) -> bool:
    if training_type is None or (isinstance(training_type, float) and pd.isna(training_type)):
        return False
    tt = str(training_type).strip().lower()
    if not tt:
        return False
    if tt in _USER_FACING_NON_STRUCTURED_TYPES:
        return False
    if tt in _USER_FACING_STRUCTURED_TYPES:
        return True
    return any(k in tt for k in ("interval", "vo2", "threshold", "test"))


def format_run_key_detail(
    run_date: object,
    training_type: object,
    distance_km: object,
    work_block_label: str | None,
) -> str:
    if is_user_facing_structured_run(training_type):
        if work_block_label and work_block_label.strip():
            return work_block_label.strip()
    if distance_km is not None and not (isinstance(distance_km, float) and pd.isna(distance_km)):
        try:
            return format_number(float(distance_km), 2, " km")
        except (TypeError, ValueError):
            pass
    ds = str(run_date).strip() if run_date is not None else ""
    return f"{ds} session".strip() if ds else "Details unavailable"


def format_run_display_label(
    run_date: object,
    training_type: object,
    distance_km: object,
    work_block_label: str | None,
) -> str:
    date_s = str(run_date).strip() if run_date is not None and str(run_date).strip() else "-"
    tt_s = format_training_type_label(training_type)
    detail_s = format_run_key_detail(run_date, training_type, distance_km, work_block_label)
    return f"{date_s} | {tt_s} | {detail_s}"


def _training_type_with_family_hint(
    training_type: object, work_session_family: object | None
) -> object:
    """Use family hint only when row-level training_type is missing in diagnostics."""
    if training_type is not None and not (isinstance(training_type, float) and pd.isna(training_type)):
        if str(training_type).strip():
            return training_type
    fam = str(work_session_family or "").strip().lower()
    if fam == "threshold_session":
        return "threshold_run"
    if fam in {"vo2max_session", "other_interval_session"}:
        return "vo2_interval_session"
    return training_type


def format_run_selector_label(run_row: pd.Series, work_block_label: str | None = None) -> str:
    """Readable selector label: date | training type | key detail."""
    return format_run_display_label(
        run_row.get("run_date"),
        run_row.get("training_type"),
        run_row.get("distance_km"),
        work_block_label,
    )


_TREND_DISPLAY_MAP: dict[str, str] = {
    "positive_progress": "Positive progress",
    "possible_fatigue": "Possible fatigue",
    "insufficient_history": "Insufficient history",
    "uncertain_data_quality": "Uncertain data quality",
}


def format_trend_label(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    raw = str(value).strip()
    if not raw:
        return "-"
    norm = raw.lower()
    if norm in _TREND_DISPLAY_MAP:
        return _TREND_DISPLAY_MAP[norm]
    return raw.replace("_", " ").strip().capitalize()


def _format_recommendation_text_for_display(text: str) -> str:
    """Display-only cleanup for recommendation copy (no logic changes)."""
    if not text or not isinstance(text, str):
        return text

    out = text
    out = re.sub(
        r"(training type\s+')([^']+)(')",
        lambda m: f"{m.group(1)}{format_training_type_label(m.group(2))}{m.group(3)}",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        r"(trend\s+')([^']+)(')",
        lambda m: f"{m.group(1)}{format_trend_label(m.group(2))}{m.group(3)}",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        r"(fatigue signal\s+')([^']+)(')",
        lambda m: f"{m.group(1)}{str(m.group(2)).replace('_', ' ').strip().capitalize()}{m.group(3)}",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        r"\bmoving time\s+(\d+)s\b",
        lambda m: f"moving time {format_duration(int(m.group(1)))}",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        r"\bintensity\s+([a-z_]+)\b",
        lambda m: f"intensity {str(m.group(1)).replace('_', ' ').strip().capitalize()}",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        r"\bexecution\s+([a-z_]+)\b",
        lambda m: f"execution {str(m.group(1)).replace('_', ' ').strip().capitalize()}",
        out,
        flags=re.IGNORECASE,
    )
    return out


def _recommendation_mentions_training_type(text: str) -> str | None:
    m = re.search(r"training type\s+'([^']+)'", text, flags=re.IGNORECASE)
    if not m:
        return None
    t = m.group(1).strip()
    return t if t else None


def _selected_run_recommendation_text(run_row: pd.Series) -> str:
    """
    Prefer current-state-consistent recommendation text.

    If stored recommendation text references a different training_type than the current row,
    fall back to deterministic current fields for consistency.
    """
    current_type = _scalar_str(run_row, "training_type")
    rec = _scalar_str(run_row, "recommendation_summary")
    if rec:
        mentioned = _recommendation_mentions_training_type(rec)
        if mentioned is None or current_type is None or mentioned == current_type:
            return _format_recommendation_text_for_display(rec)

    trend = format_trend_label(_scalar_str(run_row, "trend_label") or "unknown")
    fatigue = str(_scalar_str(run_row, "fatigue_signal") or "unknown").replace("_", " ").capitalize()
    t = format_training_type_label(current_type or "unknown")
    next_s = _scalar_str(run_row, "next_session")
    load = _scalar_str(run_row, "load_action")
    factors: list[str] = [f"{t.lower()} session", f"overall trend {trend.lower()}", f"fatigue {fatigue.lower()}"]
    summary_bits: list[str] = [f"This recommendation reflects your {', '.join(factors)}."]
    if next_s:
        summary_bits.append(f"Planned next session: {next_s}")
    if load:
        summary_bits.append(f"Load guidance: {load}")
    return _format_recommendation_text_for_display(" ".join(summary_bits))


def _deterministic_top_ai_summary(text: str) -> str:
    """Deterministic compact 2-4 sentence run-specific coaching summary."""
    if not text:
        return ""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if not sentences:
        return ""

    def _score(s: str, keywords: tuple[str, ...]) -> int:
        low = s.lower()
        return sum(1 for kw in keywords if kw in low)

    interpretation_keywords = (
        "session",
        "workout",
        "today",
        "run",
        "test session",
        "high-intensity",
        "high intensity",
        "easy day",
        "easy run",
        "recovery run",
        "tempo",
        "threshold",
        "vo2",
        "quality",
        "well-executed",
        "well executed",
        "executed",
        "solid effort",
        "fatigue",
        "fitness",
        "load",
        "recovery",
        "heart rate",
        "hr",
        "pace",
        "power",
        "stable",
        "controlled",
        "aerobic",
        "stress",
    )
    meaning_keywords = (
        "suggesting",
        "indicating",
        "indicates",
        "suggests",
        "means",
        "reflects",
        "implies",
        "points to",
        "likely",
        "shows",
        "pattern",
        "readiness",
        "without undue fatigue",
    )
    implication_keywords = (
        "moving forward",
        "next",
        "recommend",
        "should",
        "focus",
        "keep",
        "maintain",
        "support recovery",
        "load steady",
        "plan",
        "priority",
        "consequence",
    )
    recommendation_starters = (
        "to support",
        "the next session",
        "next session",
        "moving forward",
        "focus on",
        "keep the next",
        "you should",
        "following this guidance",
    )
    run_specific_keywords = (
        "this run",
        "this session",
        "easy run",
        "interval",
        "long run",
        "effort",
        "pace",
        "power",
        "hr",
        "heart rate",
        "cadence",
        "workout",
        "test",
        "execution",
    )
    abstract_state_only_keywords = (
        "trend",
        "fitness",
        "fatigue",
        "stable",
        "neutral",
        "managed load",
        "load balance",
    )
    cleanup_patterns = (
        (r"\boverall signals? (?:point|points) to\b", ""),
        (r"\b(?:heart rate|hr) and power data (?:indicate|indicates|suggest|suggests)\b", ""),
        (r"\bthis suggests that\b", ""),
        (r"\bit suggests that\b", ""),
    )

    def _clean(sentence: str) -> str:
        s = sentence.strip()
        for pattern, repl in cleanup_patterns:
            s = re.sub(pattern, repl, s, flags=re.IGNORECASE)
        s = re.sub(r"\s+,", ",", s)
        s = re.sub(r"\s{2,}", " ", s).strip(" ,;:-")
        if s and not s.endswith((".", "!", "?")):
            s += "."
        return s

    def _pick_index(role: str, excluded: set[int] | None = None) -> int:
        excluded = excluded or set()
        candidates = [i for i in range(len(sentences)) if i not in excluded]
        if not candidates:
            return 0
        if role == "interpretation":
            return max(
                candidates,
                key=lambda i: (
                    3 * _score(sentences[i], run_specific_keywords)
                    + 2 * _score(sentences[i], interpretation_keywords)
                    + _score(sentences[i], meaning_keywords)
                    - _score(sentences[i], abstract_state_only_keywords)
                    - 3 * _score(sentences[i], recommendation_starters),
                    -i,
                ),
            )
        if role == "meaning":
            return max(
                candidates,
                key=lambda i: (
                    2 * _score(sentences[i], meaning_keywords)
                    + _score(sentences[i], interpretation_keywords)
                    + _score(sentences[i], run_specific_keywords)
                    - _score(sentences[i], recommendation_starters),
                    -abs(i - (len(sentences) // 2)),
                ),
            )
        return max(
            candidates,
            key=lambda i: (
                3 * _score(sentences[i], implication_keywords) + _score(sentences[i], recommendation_starters),
                i,
            ),
        )

    interp_idx = _pick_index("interpretation")
    meaning_idx = _pick_index("meaning", {interp_idx})
    implication_idx = _pick_index("implication", {interp_idx, meaning_idx})
    if _score(sentences[implication_idx], implication_keywords) == 0 and len(sentences) > 1:
        implication_idx = len(sentences) - 1

    chosen: list[str] = []
    for idx in (interp_idx, meaning_idx, implication_idx):
        cleaned = _clean(sentences[idx])
        if not cleaned:
            continue
        low = cleaned.lower()
        if any(low == s.lower() or low.rstrip(".!?") in s.lower() for s in chosen):
            continue
        chosen.append(cleaned)

    if not chosen:
        return _clean(sentences[0])

    if len(chosen) == 1 and len(sentences) > 1:
        tail = _clean(sentences[len(sentences) - 1])
        if tail and tail.lower() != chosen[0].lower():
            chosen.append(tail)

    if len(chosen) < 4:
        extra_idx = _pick_index("meaning", {interp_idx, meaning_idx, implication_idx})
        extra = _clean(sentences[extra_idx])
        if extra:
            low = extra.lower()
            if not any(low == s.lower() or low.rstrip(".!?") in s.lower() for s in chosen):
                chosen.insert(min(2, len(chosen)), extra)

    out = " ".join(chosen[:4]).strip()
    words = out.split()
    if len(words) > 90:
        out = " ".join(words[:90]).rstrip(" ,;:-")
    if not out.endswith((".", "!", "?")):
        out += "."
    return out


def _top_ai_summary_card_text(
    run_row: pd.Series, trace_payload: dict | None
) -> tuple[str | None, str | None]:
    """Return (card_text, full_text_for_lower_section) preferring explicit short summaries."""
    full_from_row = str(run_row.get("llm_summary") or "").strip() or None
    short_from_row = str(run_row.get("llm_summary_short") or "").strip() or None

    trace = trace_payload.get("trace") if isinstance(trace_payload, dict) else None
    full_from_trace = None
    short_from_trace = None
    if isinstance(trace, dict):
        fs = str(trace.get("final_summary") or "").strip()
        full_from_trace = fs or None
        ss = str(trace.get("short_summary") or "").strip()
        short_from_trace = ss or None

    full = full_from_row or full_from_trace
    short = short_from_row or short_from_trace
    if short:
        return short, full
    # Legacy fallback: only derive if no explicit short summary exists anywhere.
    if full:
        return _deterministic_top_ai_summary(full), full
    return None, None


def _deterministic_top_context_summary(text: str) -> str:
    """Compact 2-4 sentence comparative summary derived from full context interpretation."""
    if not text:
        return ""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if not sentences:
        return ""

    def _score(s: str, keywords: tuple[str, ...]) -> int:
        low = s.lower()
        return sum(1 for kw in keywords if kw in low)

    comparative_keywords = (
        "compared",
        "vs",
        "versus",
        "prior",
        "previous",
        "last",
        "baseline",
        "relative to",
        "earlier",
        "than usual",
        "trend",
        "change",
        "improved",
        "higher",
        "lower",
    )
    meaning_keywords = (
        "means",
        "suggests",
        "indicates",
        "implies",
        "points to",
        "likely",
        "reflects",
        "readiness",
        "fatigue",
        "fitness",
    )
    practical_keywords = (
        "next",
        "action",
        "focus",
        "keep",
        "adjust",
        "maintain",
        "reduce",
        "increase",
        "should",
        "plan",
        "session",
    )

    ranked = sorted(
        range(len(sentences)),
        key=lambda i: (
            3 * _score(sentences[i], comparative_keywords)
            + 2 * _score(sentences[i], meaning_keywords)
            + _score(sentences[i], practical_keywords),
            -i,
        ),
        reverse=True,
    )

    chosen_idx: list[int] = []
    for i in ranked:
        if len(chosen_idx) >= 4:
            break
        if not chosen_idx:
            chosen_idx.append(i)
            continue
        low = sentences[i].lower().rstrip(".!?")
        if any(low == sentences[j].lower().rstrip(".!?") for j in chosen_idx):
            continue
        chosen_idx.append(i)

    if len(chosen_idx) < 2:
        for i in range(min(len(sentences), 4)):
            if i not in chosen_idx:
                chosen_idx.append(i)
            if len(chosen_idx) >= 2:
                break

    chosen_idx = sorted(chosen_idx[:4])
    out = " ".join(sentences[i].strip() for i in chosen_idx if sentences[i].strip()).strip()
    words = out.split()
    if len(words) > 95:
        out = " ".join(words[:95]).rstrip(" ,;:-")
    if out and not out.endswith((".", "!", "?")):
        out += "."
    return out


def _context_pace_is_slower_from_trace_payload(trace_payload: dict | None) -> bool:
    """Check saved deterministic context bundle for slower-pace signal (sec/km increased)."""
    if not isinstance(trace_payload, dict):
        return False
    bundle = trace_payload.get("llm_context_progress_bundle") or {}
    if not isinstance(bundle, dict):
        return False
    rec_sig = bundle.get("recommendation_signals") or {}
    if not isinstance(rec_sig, dict):
        return False
    cmp_sig = rec_sig.get("comparable_aerobic_signal") or {}
    if not isinstance(cmp_sig, dict):
        return False
    metrics = cmp_sig.get("metrics") or {}
    if not isinstance(metrics, dict):
        return False
    pace = metrics.get("avg_pace_sec_km") or {}
    if not isinstance(pace, dict):
        return False
    status = str(pace.get("status") or "").strip().lower()
    if status == "worse":
        return True
    try:
        delta = float(pace.get("delta"))
    except (TypeError, ValueError):
        return False
    return delta > 0


def _sanitize_top_context_pace_wording(text: str, *, pace_is_slower: bool) -> str:
    """Prevent contradictory 'pace gain/improved pace' wording in top What happened box."""
    if not text or not pace_is_slower:
        return text
    out = text
    replacements: list[tuple[str, str]] = [
        (r"\bmeaningful pace gain\b", "slightly slower pace with lower heart rate"),
        (r"\bpace gain\b", "slightly slower pace with lower heart rate"),
        (r"\bimproved pace\b", "slower pace with better control"),
        (r"\bfaster pace\b", "lower speed with better aerobic control"),
    ]
    for pat, repl in replacements:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    return out


def _is_missing_series_value(val) -> bool:
    try:
        return pd.isna(val)
    except (TypeError, ValueError):
        return val is None


def _format_zone_times_sec(prefix: str, row: pd.Series, keys: list[str]) -> str | None:
    def _compact_time(seconds: float) -> str:
        total = max(0, int(round(seconds)))
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    parts: list[str] = []
    for i, key in enumerate(keys, start=1):
        v = row.get(key)
        if _is_missing_series_value(v):
            continue
        parts.append(f"Z{i}: {_compact_time(float(v))}")
    if not parts:
        return None
    return f"{prefix}: " + ", ".join(parts)


def build_fit_derived_metrics_lines(row: pd.Series) -> list[str]:
    """Compact FIT-derived lines; omits null or empty values."""
    lines: list[str] = []

    if not _is_missing_series_value(row.get("moving_time_sec")):
        lines.append(f"Moving time: {format_duration(row['moving_time_sec'])}")
    if not _is_missing_series_value(row.get("avg_moving_pace_sec_km")):
        lines.append(f"Avg moving pace: {format_pace(row['avg_moving_pace_sec_km'])}")
    if not _is_missing_series_value(row.get("stopped_time_sec")):
        lines.append(f"Stopped time: {format_duration(row['stopped_time_sec'])}")

    pz = _format_zone_times_sec(
        "Power zones",
        row,
        [f"power_zone_z{i}_sec" for i in range(1, 6)],
    )
    if pz:
        lines.append(pz)
    hz = _format_zone_times_sec(
        "HR zones",
        row,
        [f"hr_zone_z{i}_sec" for i in range(1, 6)],
    )
    if hz:
        lines.append(hz)

    for label, key in (
        ("Power", "has_power"),
        ("HR", "has_hr"),
        ("Cadence", "has_cadence"),
        ("GPS", "has_gps"),
    ):
        v = row.get(key)
        if _is_missing_series_value(v):
            continue
        yn = "Yes" if int(v) else "No"
        lines.append(f"Has {label.lower()}: {yn}")

    warn = row.get("fit_parse_warnings")
    if warn is not None and str(warn).strip():
        lines.append(f"Parse / completeness notes: {str(warn).strip()}")

    return lines


def render_fit_derived_metrics(row: pd.Series) -> None:
    """Compact FIT-derived fields for validation; omits null or empty values."""
    lines = build_fit_derived_metrics_lines(row)

    if not lines:
        st.caption("No additional FIT metrics available for this run.")
        return

    st.caption("From the database (typically Garmin FIT imports).")
    st.markdown("\n".join(f"- {line}" for line in lines))


_INTERVAL_FAMILY_TABLE_FAMILIES: frozenset[str] = frozenset(
    {"threshold_session", "vo2max_session"}
)


def _format_work_metric_cell(val, *, kind: str) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "-"
    if kind == "duration":
        return format_duration(val)
    if kind == "pace":
        return format_pace(val)
    if kind == "w_per_hr":
        return format_number(val, 2)
    return format_int(val)


def _structured_training_type(training_type: str | None) -> bool:
    t = str(training_type or "").strip().lower()
    return t in {
        "threshold_run",
        "vo2_interval_session",
        "test_session",
        "test_or_vo2_session",
    }


def _is_structured_selected_run(
    run_row: pd.Series,
    *,
    work_only: dict | None,
    work_block_label: str | None,
    work_family: str | None,
    sel_payload: dict | None,
) -> bool:
    tt = _scalar_str(run_row, "training_type")
    tt_norm = str(tt or "").strip().lower()

    # Explicitly suppress common non-structured endurance sessions.
    if tt_norm in {"easy_run", "recovery_run", "steady_run", "long_run"}:
        return False

    # Strong direct signal from training-type classification.
    if _structured_training_type(tt):
        return True

    # Strong structured family evidence only (avoid generic interval-like buckets).
    fam = str(work_family or "").strip().lower()
    if fam in {"threshold_session", "vo2max_session"}:
        return True

    if sel_payload and sel_payload.get("applicable"):
        sel_fam = str(sel_payload.get("work_session_family") or "").strip().lower()
        if sel_fam in {"threshold_session", "vo2max_session"}:
            return True

    # Do not gate on generic segment/work aggregate presence alone.
    return False


def _scalar_str(row: pd.Series, key: str) -> str | None:
    v = row.get(key)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    return s if s else None


def _takeaway_interval_vs_prior_line(sel_payload: dict | None) -> str | None:
    if not sel_payload or not sel_payload.get("applicable"):
        return None
    fam = sel_payload.get("work_session_family")
    fam_word = (
        "threshold"
        if fam == "threshold_session"
        else ("VO2max" if fam == "vo2max_session" else "interval")
    )
    if sel_payload.get("insufficient_history"):
        if sel_payload.get("reason") == "no_prior_family_session":
            return (
                f"Earliest stored {fam_word} family session in this history—no prior to compare."
            )
        return "Interval comparison: not enough prior history in this family."

    mm = sel_payload.get("metrics") or {}
    bits: list[str] = []
    for metric_key, short in (
        ("work_mean_pace_sec_per_km", "pace"),
        ("work_mean_power_w", "power"),
        ("work_mean_hr_avg", "HR"),
        ("work_w_per_hr", "W/HR"),
    ):
        st = (mm.get(metric_key) or {}).get("status")
        if st is not None and str(st).strip():
            bits.append(f"{short} {st}")
    if not bits:
        return None
    return f"Same-family vs prior session: {' · '.join(bits)}."


def _takeaway_signals_line(run_row: pd.Series) -> str | None:
    parts: list[str] = []
    t = _scalar_str(run_row, "trend_label")
    if t:
        parts.append(f"trend {format_trend_label(t)}")
    eq = _scalar_str(run_row, "execution_quality")
    if eq:
        parts.append(f"execution {eq}")
    fs = _scalar_str(run_row, "fitness_signal")
    if fs:
        parts.append(f"fitness {fs}")
    fg = _scalar_str(run_row, "fatigue_signal")
    if fg:
        parts.append(f"fatigue {fg}")
    if not parts:
        return None
    return " · ".join(parts)


def build_run_takeaway_bullets(
    run_row: pd.Series, sel_payload: dict | None
) -> list[str]:
    """2–4 deterministic markdown lines (no LLM)."""
    session = format_training_type_label(_scalar_str(run_row, "training_type") or "Unlabeled session")
    lines: list[str] = [f"**Session:** {session}"]

    iv = _takeaway_interval_vs_prior_line(sel_payload)
    sig = _takeaway_signals_line(run_row)
    structured_types = {
        "threshold_run",
        "vo2_interval_session",
        "test_or_vo2_session",
        "test_session",
        "race",
    }
    show_interval_vs_prior = iv is not None and session in structured_types

    if show_interval_vs_prior:
        lines.append(f"**Vs prior (segments):** {iv}")
        if sig:
            lines.append(f"**Signals:** {sig}")
    elif sig:
        lines.append(f"**Signals:** {sig}")
    else:
        lines.append("**Signals:** — (see analysis; no same-family interval comparison)")

    next_bits: list[str] = []
    ns = _scalar_str(run_row, "next_session")
    la = _scalar_str(run_row, "load_action")
    if ns:
        next_bits.append(ns)
    if la:
        next_bits.append(f"load: {la}")
    if next_bits:
        lines.append(f"**Next step:** {' · '.join(next_bits)}")
    else:
        lines.append("**Next step:** —")

    return lines[:4]


def render_run_takeaway(run_row: pd.Series, sel_payload: dict | None) -> None:
    bullets = build_run_takeaway_bullets(run_row, sel_payload)
    st.markdown("### Run takeaway")
    st.markdown("\n".join(f"- {b}" for b in bullets))


def render_interval_family_history_table(sel_payload: dict) -> None:
    """Compact family history next to interval insight; no-op unless threshold/VO2 with rows."""
    fam = sel_payload.get("work_session_family")
    if fam not in _INTERVAL_FAMILY_TABLE_FAMILIES:
        return
    window = sel_payload.get("family_history_window")
    selected = str(sel_payload.get("selected_run_id") or "").strip()
    baseline_rid = sel_payload.get("baseline_run_id_for_comparison")
    baseline_s = str(baseline_rid).strip() if baseline_rid else ""

    if not window:
        st.caption(
            "Family history preview is unavailable for this run (missing from stored family history)."
        )
        return

    rows_out: list[dict[str, str]] = []
    for r in window:
        rid = str(r.get("run_id") or "").strip()
        role = ""
        if rid == selected:
            role = "This run"
        elif baseline_s and rid == baseline_s:
            role = "Prior (comparison)"
        rows_out.append(
            {
                "run_date": str(r.get("run_date") if r.get("run_date") is not None else "-"),
                "work_block_label": str(r.get("work_block_label") or "-"),
                "work_total_time_sec": _format_work_metric_cell(
                    r.get("work_total_time_sec"), kind="duration"
                ),
                "work_mean_pace_sec_per_km": _format_work_metric_cell(
                    r.get("work_mean_pace_sec_per_km"), kind="pace"
                ),
                "work_mean_power_w": _format_work_metric_cell(
                    r.get("work_mean_power_w"), kind="int"
                ),
                "work_mean_hr_avg": _format_work_metric_cell(
                    r.get("work_mean_hr_avg"), kind="int"
                ),
                "work_w_per_hr": _format_work_metric_cell(
                    r.get("work_w_per_hr"), kind="w_per_hr"
                ),
                "role": role,
            }
        )

    display_df = pd.DataFrame(rows_out)
    display_df.columns = [
        "Run date",
        "Work block",
        "Work time",
        "Pace (min/km)",
        "Power (W)",
        "HR",
        "W/HR",
        "Role",
    ]

    def _row_bg(row: pd.Series) -> list[str]:
        tag = row.iloc[-1]
        n = len(row)
        if tag == "This run":
            return [f"background-color: rgba(46, 160, 67, 0.16)"] * n
        if tag == "Prior (comparison)":
            return [f"background-color: rgba(255, 193, 7, 0.22)"] * n
        return [""] * n

    styled = display_df.style.hide(axis="index").apply(_row_bg, axis=1)
    st.dataframe(styled, width="stretch", hide_index=True)


def _family_trend_numeric(val) -> float:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return float("nan")
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def _family_trend_chart_df(window: list) -> pd.DataFrame:
    """Oldest → newest rows for progression; includes run_id for selection highlight."""
    rows = list(reversed(window))
    out: list[dict] = []
    for r in rows:
        rid = str(r.get("run_id") or "").strip()
        rd = r.get("run_date")
        out.append(
            {
                "run_id": rid,
                "run_date_dt": pd.to_datetime(rd, errors="coerce"),
                "work_mean_pace_sec_per_km": _family_trend_numeric(
                    r.get("work_mean_pace_sec_per_km")
                ),
                "work_mean_power_w": _family_trend_numeric(r.get("work_mean_power_w")),
                "work_w_per_hr": _family_trend_numeric(r.get("work_w_per_hr")),
            }
        )
    df = pd.DataFrame(out)
    if not df.empty and "work_mean_pace_sec_per_km" in df.columns:
        df["pace_tooltip"] = _pace_tooltip_series(df["work_mean_pace_sec_per_km"])
    return df


def _compact_family_metric_chart(
    chart_df: pd.DataFrame,
    *,
    y_col: str,
    title: str,
    selected_run_id: str,
) -> alt.Chart:
    if chart_df.empty or y_col not in chart_df.columns:
        return (
            alt.Chart(chart_df)
            .mark_text(text="No data", align="center")
            .properties(height=130, title=title)
        )

    if _is_pace_sec_per_km_column(y_col):
        df = chart_df.copy()
        if "pace_tooltip" not in df.columns:
            df["pace_tooltip"] = _pace_tooltip_series(df[y_col])
        base = alt.Chart(df).encode(
            x=alt.X("run_date_dt:T", axis=alt.Axis(format="%b %d", title=None)),
        )
        y_line = alt.Y(
            f"{y_col}:Q",
            axis=_pace_axis_sec_per_km(),
            scale=alt.Scale(zero=False),
        )
        line = base.mark_line(point=False).encode(y=y_line)
        hover_points = (
            alt.Chart(df)
            .mark_circle(size=260, opacity=0)
            .encode(
                x=alt.X("run_date_dt:T"),
                y=alt.Y(f"{y_col}:Q", title=None, scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip("run_date_dt:T", title="Date"),
                    alt.Tooltip("pace_tooltip:N", title="Pace"),
                ],
            )
        )
        sel_df = df[df["run_id"] == selected_run_id]
        points = (
            alt.Chart(sel_df)
            .mark_circle(size=95, color="#2ea043", stroke="white", strokeWidth=1)
            .encode(
                x=alt.X("run_date_dt:T"),
                y=alt.Y(f"{y_col}:Q", title=None, scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip("run_date_dt:T", title="Date"),
                    alt.Tooltip("pace_tooltip:N", title="Pace"),
                ],
            )
        )
    else:
        _ttl = _dashboard_chart_tooltip_label(y_col)
        base = alt.Chart(chart_df).encode(
            x=alt.X("run_date_dt:T", axis=alt.Axis(format="%b %d", title=None)),
        )
        line = base.mark_line(point=False).encode(
            y=alt.Y(f"{y_col}:Q", title=None, scale=alt.Scale(zero=False)),
        )
        hover_points = (
            alt.Chart(chart_df)
            .mark_circle(size=260, opacity=0)
            .encode(
                x=alt.X("run_date_dt:T"),
                y=alt.Y(f"{y_col}:Q", title=None, scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip("run_date_dt:T", title="Date"),
                    alt.Tooltip(
                        f"{y_col}:Q",
                        title=_ttl,
                        format=_chart_y_tooltip_format(y_col),
                    ),
                ],
            )
        )
        sel_df = chart_df[chart_df["run_id"] == selected_run_id]
        points = (
            alt.Chart(sel_df)
            .mark_circle(size=95, color="#2ea043", stroke="white", strokeWidth=1)
            .encode(
                x=alt.X("run_date_dt:T"),
                y=alt.Y(f"{y_col}:Q", title=None, scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip("run_date_dt:T", title="Date"),
                    alt.Tooltip(
                        f"{y_col}:Q",
                        title=_ttl,
                        format=_chart_y_tooltip_format(y_col),
                    ),
                ],
            )
        )
    return (
        (line + hover_points + points)
        .properties(height=130, title=title)
        .configure_axis(labelFontSize=10, titleFontSize=11)
        .configure_title(fontSize=13)
    )


def render_interval_family_trend_charts(sel_payload: dict) -> None:
    """Compact pace / power / W·HR trends from deterministic family history (threshold & VO2max only)."""
    fam = sel_payload.get("work_session_family")
    if fam not in _INTERVAL_FAMILY_TABLE_FAMILIES:
        return

    window = sel_payload.get("family_history_window")
    selected = str(sel_payload.get("selected_run_id") or "").strip()
    fam_label = "Threshold" if fam == "threshold_session" else "VO2max"

    st.markdown("#### Family trend")
    if window:
        st.caption(f"{fam_label}: pace, power, W/HR (oldest → newest; same series as the table).")
    else:
        st.caption(
            "Family trend is unavailable for this run (not in stored family history window)."
        )
        return

    chart_df = _family_trend_chart_df(window)
    if chart_df["run_date_dt"].isna().all():
        st.caption("Family trend needs valid run dates in family history.")
        return

    c1, c2, c3 = st.columns(3)
    specs = (
        (c1, "work_mean_pace_sec_per_km", "Pace"),
        (c2, "work_mean_power_w", "Power"),
        (c3, "work_w_per_hr", "W / HR"),
    )
    for col, y_col, title in specs:
        _cols = ["run_date_dt", "run_id", y_col]
        if _is_pace_sec_per_km_column(y_col) and "pace_tooltip" in chart_df.columns:
            _cols.append("pace_tooltip")
        sub = chart_df[_cols].copy()
        with col:
            if sub[y_col].notna().sum() == 0:
                st.caption(f"{title}: — (no numeric data)")
                continue
            ch = _compact_family_metric_chart(
                sub,
                y_col=y_col,
                title=title,
                selected_run_id=selected,
            )
            st.altair_chart(ch, width="stretch")


def _metric_status_words(metric_key: str, delta: float | None) -> str:
    if delta is None:
        return "unknown"
    if metric_key == "pace":
        if delta <= -3:
            return "improved"
        if delta >= 3:
            return "softened"
        return "stable"
    if metric_key == "power":
        if delta >= 3:
            return "improved"
        if delta <= -3:
            return "softened"
        return "stable"
    if metric_key == "hr":
        if delta <= -2:
            return "improved"
        if delta >= 2:
            return "higher"
        return "stable"
    if metric_key == "whr":
        if delta >= 0.04:
            return "improved"
        if delta <= -0.04:
            return "softened"
        return "stable"
    return "stable"


def _structured_comparison_verdict(bits: dict[str, str]) -> str:
    good = sum(1 for v in bits.values() if v == "improved")
    bad = sum(1 for v in bits.values() if v in {"softened", "higher"})
    if good >= 2 and bad == 0:
        return "improved versus last similar workout"
    if bad >= 2 and good == 0:
        return "softened versus last similar workout"
    return "consolidated versus last similar workout"


def _structured_rep_shape(rep_analysis: dict | None) -> dict | None:
    if not rep_analysis or not rep_analysis.get("available"):
        return None
    reps = list(rep_analysis.get("reps") or [])
    if not reps:
        return None
    dist_vals = [float(r.get("distance_m")) for r in reps if r.get("distance_m") is not None]
    dur_vals = [float(r.get("duration_sec")) for r in reps if r.get("duration_sec") is not None]
    rep_count = len(reps)
    if dist_vals and (len(dist_vals) >= max(1, rep_count // 2)):
        core = int(round((sum(dist_vals) / len(dist_vals)) / 10.0) * 10)
        return {"rep_count": rep_count, "rep_type": "distance", "core_size": core}
    if dur_vals:
        core = max(1, int(round((sum(dur_vals) / len(dur_vals)) / 60.0)))
        return {"rep_count": rep_count, "rep_type": "time", "core_size": core}
    return None


def _pick_near_exact_structured_baseline(
    conn,
    *,
    selected_run_id: str,
    selected_rep_analysis: dict | None,
    work_family: str | None,
) -> tuple[str | None, str | None]:
    fam = str(work_family or "").strip()
    if fam not in {"threshold_session", "vo2max_session"}:
        return None, None
    target = _structured_rep_shape(selected_rep_analysis)
    if not target:
        return None, None

    hist = fetch_dedup_work_family_session_history(conn, fam)
    idx = next((i for i, r in enumerate(hist) if str(r.get("run_id") or "") == selected_run_id), None)
    if idx is None or idx < 1:
        return None, None

    best: tuple[int, int, str, str] | None = None
    for prior_idx, r in enumerate(hist[:idx]):
        rid = str(r.get("run_id") or "").strip()
        if not rid:
            continue
        cand_rep = analyze_structured_work_reps_for_run(conn, rid)
        cand = _structured_rep_shape(cand_rep)
        if not cand:
            continue
        if cand["rep_type"] != target["rep_type"]:
            continue
        if int(cand["core_size"]) != int(target["core_size"]):
            continue
        rep_diff = abs(int(cand["rep_count"]) - int(target["rep_count"]))
        if rep_diff > 2:
            continue
        # Prioritize smallest rep-count gap; if tied, most recent prior wins (later idx in hist slice).
        score = (
            rep_diff,
            idx - prior_idx,
            rid,
            str(r.get("work_block_label") or "").strip(),
        )
        if best is None or (score[0], -score[1]) < (best[0], -best[1]):
            best = score
    if best is None:
        return None, None
    _, _, rid, lbl = best
    return rid, lbl


def render_structured_workout_analysis_block(
    *,
    show_structured_details: bool,
    run_row: pd.Series,
    selected_run_id: str,
    rep_analysis: dict | None,
    exact_compare: dict | None,
    sel_payload: dict | None,
    current_work_only: dict | None,
    baseline_work_only: dict | None,
    baseline_rep_analysis: dict | None,
    comparison_mode: str | None,
) -> None:
    if not show_structured_details:
        return
    st.markdown("#### Structured workout analysis")

    if not rep_analysis or not rep_analysis.get("available"):
        st.caption("Rep-by-rep interval analysis unavailable for this run.")
        return

    reps = list(rep_analysis.get("reps") or [])
    summary = rep_analysis.get("summary") or {}
    intr = rep_analysis.get("interpretation") or {}

    if reps:
        rows = [
            {
                "Rep": str(int(r.get("rep_no") or 0)),
                "Block": str(r.get("rep_label") or "work rep"),
                "Time": format_duration(r.get("duration_sec")),
                "Pace": format_pace(r.get("pace_sec_per_km")),
                "Power": format_power_w(r.get("power_w")),
                "Avg HR": format_int(r.get("hr_avg")),
                "Max HR": format_int(r.get("hr_max")),
            }
            for r in reps
        ]
        rows.append(
            {
                "Rep": "All",
                "Block": "Work reps summary",
                "Time": format_duration(summary.get("total_work_time_sec")),
                "Pace": format_pace(summary.get("work_mean_pace_sec_per_km")),
                "Power": format_power_w(summary.get("work_mean_power_w")),
                "Avg HR": format_int(summary.get("work_mean_hr_avg")),
                "Max HR": format_int(summary.get("work_max_hr_avg")),
            }
        )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.markdown("**Intra-workout interpretation**")
    st.markdown(
        "\n".join(
            [
                f"- Interval stability: **{intr.get('interval_stability') or 'unknown'}**",
                f"- Rep dynamics: {intr.get('coupling') or 'no meaningful execution breakdown'}",
            ]
        )
    )

    if not current_work_only:
        st.caption("Comparable-session analysis unavailable (missing current work summary).")
        return

    if not baseline_work_only:
        st.caption("No prior comparable structured workout found yet.")
        return

    cur_pace = current_work_only.get("work_mean_pace_sec_per_km")
    base_pace = baseline_work_only.get("work_mean_pace_sec_per_km")
    cur_pwr = current_work_only.get("work_mean_power_w")
    base_pwr = baseline_work_only.get("work_mean_power_w")
    cur_hr = current_work_only.get("work_mean_hr_avg")
    base_hr = baseline_work_only.get("work_mean_hr_avg")
    cur_whr = current_work_only.get("work_w_per_hr")
    base_whr = baseline_work_only.get("work_w_per_hr")
    deltas = {
        "pace": (float(cur_pace) - float(base_pace)) if cur_pace is not None and base_pace is not None else None,
        "power": (float(cur_pwr) - float(base_pwr)) if cur_pwr is not None and base_pwr is not None else None,
        "hr": (float(cur_hr) - float(base_hr)) if cur_hr is not None and base_hr is not None else None,
        "whr": (float(cur_whr) - float(base_whr)) if cur_whr is not None and base_whr is not None else None,
    }
    statuses = {
        "pace": _metric_status_words("pace", deltas["pace"]),
        "power": _metric_status_words("power", deltas["power"]),
        "hr": _metric_status_words("hr", deltas["hr"]),
        "whr": _metric_status_words("whr", deltas["whr"]),
    }
    verdict = _structured_comparison_verdict(statuses)

    st.markdown("**Compared with last similar workout**")
    st.caption(comparison_mode or "Closest comparable structured session")

    comp_rows = [
        {
            "Session": "Current",
            "Work pace": format_pace(cur_pace),
            "Work power": format_power_w(cur_pwr),
            "Work HR": format_int(cur_hr),
            "Work W/HR": format_number(cur_whr, 2) if not _is_missing_series_value(cur_whr) else "-",
            "Execution stability": str(intr.get("interval_stability") or "-"),
        },
        {
            "Session": "Prior comparable",
            "Work pace": format_pace(base_pace),
            "Work power": format_power_w(base_pwr),
            "Work HR": format_int(base_hr),
            "Work W/HR": format_number(base_whr, 2) if not _is_missing_series_value(base_whr) else "-",
            "Execution stability": str((baseline_rep_analysis or {}).get("interpretation", {}).get("interval_stability") or "-"),
        },
    ]
    st.dataframe(pd.DataFrame(comp_rows), width="stretch", hide_index=True)

    st.markdown("**Deterministic coaching view**")
    st.markdown(
        "\n".join(
            [
                f"- Execution quality: {str(_scalar_str(run_row, 'execution_quality') or '—')}",
                f"- Interval stability: {intr.get('interval_stability') or 'unknown'}",
                f"- Similar-session comparison: **{verdict}**",
                f"- Practical interpretation: pace {statuses['pace']}, power {statuses['power']}, HR {statuses['hr']}, W/HR {statuses['whr']}.",
            ]
        )
    )


def import_uploaded_files(
    uploaded_files, use_llm: bool
) -> tuple[bool, str, str, dict | None]:
    debug_lines = []

    if not uploaded_files:
        return False, "Please upload at least one CSV file.", "No uploaded files received.", None

    record_debug(debug_lines, f"Received {len(uploaded_files)} uploaded file(s).")
    try:
        from agenticrun.services.llm import LLMService

        _llm_status = LLMService()
        record_debug(
            debug_lines,
            _llm_status.format_ingest_runtime_status_line(
                enabled_for_run=use_llm,
                used_for_run=False,
            )
            + " (used_for_run=n/a until ingest runs; console shows per-step updates)",
        )
    except Exception as exc:
        record_debug(debug_lines, f"LLM runtime status snapshot failed: {exc}")

    try:
        from main import ingest_folder
        record_debug(debug_lines, "Imported ingest_folder from main.py successfully.")
    except Exception as e:
        return False, f"Could not import ingest pipeline: {e}", "\n".join(debug_lines), None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    record_debug(debug_lines, f"Output directory: {OUT_DIR.resolve()}")

    snap_before = _db_coverage_snapshot(str(DB_PATH))

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        record_debug(debug_lines, f"Temporary upload directory: {tmp_path}")

        saved_files = []
        for uploaded_file in uploaded_files:
            target = tmp_path / uploaded_file.name
            with open(target, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_files.append(target.name)

            record_debug(debug_lines, f"Saved files: {', '.join(saved_files)}")

        try:
            summary = ingest_folder(
                input_dir=str(tmp_path),
                db_path=str(DB_PATH),
                out_dir=str(OUT_DIR),
                use_llm=use_llm,
            )
            record_debug(debug_lines, "ingest_folder completed successfully.")
            st.session_state.import_snapshot_before = snap_before
            return (
                True,
                f"Imported {len(uploaded_files)} file(s) successfully.",
                "\n".join(debug_lines),
                summary,
            )
        except Exception as e:
            record_debug(debug_lines, f"Import failed with exception: {e}")
            return False, f"Import failed: {e}", "\n".join(debug_lines), None


def render_recommendation_easy_drift_debug(db_path: Path, selected_run_id: str | None) -> None:
    """Diagnostics: deterministic source of `easy_too_hard_recent` / easy-drift recommendation copy."""
    from agenticrun.agents.recommendation_agent import diagnose_easy_recovery_drift_rule

    st.caption(
        "Deterministic rule behind copy like “recent easy/recovery sessions… harder than intended”. "
        "Same `load_history` + prior slice as RecommendationAgent."
    )
    rid = str(selected_run_id or "").strip()
    if not rid:
        st.caption("No run selected.")
        return
    if not db_path.is_file():
        st.caption("Database not found.")
        return
    try:
        conn = connect(str(db_path))
        try:
            history = load_history(conn)
        finally:
            conn.close()
    except Exception as exc:
        st.warning(f"Could not load run history: {exc}")
        return

    payload = diagnose_easy_recovery_drift_rule(history, rid)
    st.markdown(str(payload.get("how_it_works") or ""))
    th = payload.get("thresholds") or {}
    labs = th.get("intensity_labels_counted_as_too_hard_for_easy_recovery") or []
    st.markdown(
        f"- **Prior window:** up to **{th.get('prior_window_max_rows', '?')}** rows after removing the "
        f"current `run_id` from full history.\n"
        f"- **Counter slice:** `prior[-{th.get('slice_for_counter', '?')}:]` (last N of that window).\n"
        f"- **Trigger:** count of flagged rows in slice ≥ **{th.get('min_flagged_in_slice_to_fire', '?')}** "
        "(the drift branch still loses to earlier rules such as fatigue or low_q).\n"
        f"- **Flag row when:** `training_type` ∈ {{easy_run, recovery_run}} and normalized `intensity_label` ∈ "
        f"`{labs}` — **label only**; zone seconds are not used for this counter."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("easy_too_hard_recent", int(payload.get("easy_too_hard_recent_count") or 0))
    with c2:
        st.metric(
            "Slice condition met",
            "yes" if payload.get("rule_counter_condition_met") else "no",
        )
    with c3:
        st.metric("Prior rows (excl. self)", int(payload.get("prior_row_count_after_exclude_self") or 0))

    rows = list(payload.get("evaluation_slice_rows") or [])
    if rows:
        disp: list[dict[str, object]] = []
        for r in rows:
            o = r.get("observed_snapshot") or {}
            disp.append(
                {
                    "run_date": r.get("run_date"),
                    "run_id": r.get("run_id"),
                    "training_type": r.get("training_type"),
                    "intensity_label": r.get("intensity_label_raw"),
                    "flagged": r.get("flagged_too_hard_by_rule"),
                    "avg_hr": o.get("avg_hr"),
                    "avg_power": o.get("avg_power"),
                    "avg_pace_sec_km": o.get("avg_pace_sec_km"),
                    "reason": r.get("why"),
                }
            )
        st.markdown("**Evaluation slice (contributing rows)**")
        st.dataframe(pd.DataFrame(disp), width="stretch", hide_index=True)
    else:
        st.caption("No prior rows in the evaluation slice (empty history or missing prior window).")

    for line in payload.get("misclassification_and_limits") or []:
        st.caption(f"— {line}")

    with st.expander("Full diagnostic JSON", expanded=False):
        st.json(payload)


def render_import_section(*, expanded: bool = False) -> None:
    with st.expander("Import and operations", expanded=expanded):
        uploaded_files = st.file_uploader(
            "Upload one or more Garmin exports (CSV, FIT, or ZIP)",
            type=["csv", "fit", "zip"],
            accept_multiple_files=True,
        )
        use_llm = st.checkbox("Generate AI summaries for new analyses", value=False)

        if st.button("Import runs", type="primary"):
            with st.spinner("Importing runs..."):
                success, message, debug_text, summary = import_uploaded_files(
                    uploaded_files, use_llm
                )
                st.session_state.import_status = "success" if success else "error"
                st.session_state.import_message = message
                st.session_state.import_debug = debug_text
                if success and summary is not None:
                    st.session_state.import_summary = summary

                if success:
                    load_runs_from_db.clear()
                    st.rerun()

        _import_summary_display = st.session_state.get("import_summary")
        if _import_summary_display is None:
            _import_summary_display = load_last_import_summary()

        if _import_summary_display:
            st.markdown("#### Import summary")
            st.caption("Last successful import in this session, or the saved summary from disk.")
            for _line in format_import_summary_lines(_import_summary_display):
                st.markdown(f"- {_line}")
            render_what_changed_after_import(
                DB_PATH,
                _import_summary_display,
                st.session_state.get("import_snapshot_before"),
            )

        if st.session_state.import_status == "success":
            st.success(st.session_state.import_message)

        if st.session_state.import_status == "error":
            st.error(st.session_state.import_message)

        if st.session_state.import_debug:
            with st.expander("Import debug", expanded=False):
                st.code(st.session_state.import_debug, language="text")


st.title("🏃 AgenticRun")
st.caption("Garmin runs, analysis, trends, and recommendations")

if not DB_PATH.exists():
    st.info("No database found yet. Import Garmin CSV files below to get started.")
    render_import_section(expanded=True)
    st.stop()

df = load_runs_table_data(str(DB_PATH), OUT_DIR)

if df.empty:
    st.info("No runs found yet. Import Garmin CSV files below to populate the app.")
    render_import_section(expanded=True)
    st.stop()

df = df.reset_index(drop=True)

df_chart = load_runs_from_db(str(DB_PATH))
df_chart = df_chart.copy()
df_chart["run_date_dt"] = pd.to_datetime(df_chart["run_date"], errors="coerce")

df["run_date_dt"] = pd.to_datetime(df["run_date"], errors="coerce")

display_df = df.copy()
display_df["duration"] = display_df["duration_sec"].apply(format_duration)
display_df["pace"] = display_df["avg_pace_sec_km"].apply(format_pace)

st.subheader("Selected run")
_selector_indices = list(range(len(display_df)))
_selector_run_ids = tuple(str(df.iloc[i].get("run_id") or "").strip() for i in _selector_indices)
_selector_wbl_map = load_work_block_label_map(str(DB_PATH), _selector_run_ids)
_selector_labels = [
    format_run_selector_label(
        display_df.iloc[i],
        _selector_wbl_map.get(str(df.iloc[i].get("run_id") or "").strip()),
    )
    for i in _selector_indices
]
display_df["session_label"] = _selector_labels
selected_idx = st.selectbox(
    "Select run",
    _selector_indices,
    format_func=lambda i: _selector_labels[int(i)],
)
run_row = df.iloc[selected_idx]
selected_run_id = str(run_row["run_id"]) if pd.notna(run_row.get("run_id")) else ""
analysis_run_id = selected_run_id
if _scalar_str(run_row, "status") == "duplicate_cached":
    _canon = run_row.get("cached_from_run_id")
    if _canon is None or (isinstance(_canon, float) and pd.isna(_canon)) or str(_canon).strip() == "":
        _canon = run_row.get("run_id")
    if _canon is not None and not (isinstance(_canon, float) and pd.isna(_canon)):
        _cid = str(_canon).strip()
        if _cid:
            analysis_run_id = _cid

is_duplicate_cached = _scalar_str(run_row, "status") == "duplicate_cached"
_has_full_llm = bool(str(run_row.get("llm_summary") or "").strip())
_has_short_llm = bool(str(run_row.get("llm_summary_short") or "").strip())
has_llm_summary_text = _has_full_llm or _has_short_llm
_has_full_context_llm = bool(str(run_row.get("llm_context_progress") or "").strip())
_has_short_context_llm = bool(str(run_row.get("llm_context_progress_short") or "").strip())
has_llm_context_text = _has_full_context_llm or _has_short_context_llm

_sel_payload: dict | None = None
_work_only_payload: dict | None = None
_work_block_label: str | None = None
_work_session_family: str | None = None
_structured_rep_analysis: dict | None = None
_exact_structured_compare: dict | None = None
_baseline_work_only_payload: dict | None = None
_baseline_rep_analysis: dict | None = None
_structured_comparison_mode: str | None = None
_insight_exc: Exception | None = None
try:
    _insight_conn = connect(str(DB_PATH))
    try:
        _hint_tt = None
        if "training_type" in run_row.index:
            _tv = run_row.get("training_type")
            if _tv is not None and not (isinstance(_tv, float) and pd.isna(_tv)):
                _hs = str(_tv).strip()
                _hint_tt = _hs if _hs else None
        _sel_payload = compare_selected_run_work_family_vs_prior(
            _insight_conn,
            analysis_run_id,
            training_type_hint=_hint_tt,
        )
        _work_only_payload = aggregate_work_only_session_for_run(_insight_conn, analysis_run_id)
        _structured_rep_analysis = analyze_structured_work_reps_for_run(
            _insight_conn, analysis_run_id
        )
        _exact_structured_compare = compare_interval_session_vs_prior(
            _insight_conn, analysis_run_id
        )
        _wbl = derive_work_block_label_for_run(_insight_conn, analysis_run_id)
        _work_block_label = (
            str(_wbl.get("work_block_label")).strip()
            if isinstance(_wbl, dict) and _wbl.get("work_block_label") is not None
            else None
        )
        _wsf = derive_work_session_family_for_run(
            _insight_conn,
            analysis_run_id,
            training_type_hint=_hint_tt,
        )
        _work_session_family = (
            str(_wsf.get("work_session_family")).strip()
            if isinstance(_wsf, dict) and _wsf.get("work_session_family") is not None
            else None
        )

        _baseline_rid: str | None = None
        _baseline_label: str | None = None
        if (
            isinstance(_exact_structured_compare, dict)
            and not _exact_structured_compare.get("insufficient_history")
            and _exact_structured_compare.get("baseline_run_id")
        ):
            _baseline_rid = str(_exact_structured_compare.get("baseline_run_id")).strip()
            _b_wbl = derive_work_block_label_for_run(_insight_conn, _baseline_rid)
            _baseline_label = (
                str(_b_wbl.get("work_block_label")).strip()
                if isinstance(_b_wbl, dict) and _b_wbl.get("work_block_label") is not None
                else None
            )
            _structured_comparison_mode = (
                f"Exact comparable: prior {_baseline_label} session"
                if _baseline_label
                else "Exact comparable: prior same-structure session"
            )

        if _baseline_rid is None:
            _near_rid, _near_label = _pick_near_exact_structured_baseline(
                _insight_conn,
                selected_run_id=analysis_run_id,
                selected_rep_analysis=_structured_rep_analysis,
                work_family=_work_session_family,
            )
            if _near_rid:
                _baseline_rid = _near_rid
                _structured_comparison_mode = (
                    f"Near match: prior {_near_label} session"
                    if _near_label
                    else "Near match: prior similar structured session"
                )

        if _baseline_rid is None and (
            isinstance(_sel_payload, dict)
            and _sel_payload.get("applicable")
            and not _sel_payload.get("insufficient_history")
        ):
            _base = _sel_payload.get("baseline") or {}
            _candidate = _base.get("run_id")
            if _candidate:
                _baseline_rid = str(_candidate).strip()
                _structured_comparison_mode = "Closest comparable: prior session in same workout family"

        if _baseline_rid and _baseline_rid != analysis_run_id:
            _baseline_work_only_payload = aggregate_work_only_session_for_run(
                _insight_conn, _baseline_rid
            )
            _baseline_rep_analysis = analyze_structured_work_reps_for_run(
                _insight_conn, _baseline_rid
            )
    finally:
        _insight_conn.close()
except Exception as exc:
    _insight_exc = exc

_table_cols = [
    "session_label",
    "run_date",
    "source_file",
    "status",
    "cached_from_run_id",
    "training_type",
    "distance_km",
    "duration",
    "pace",
    "avg_hr",
    "avg_power",
    "trend_label",
    "next_session",
]
_table_cols = [c for c in _table_cols if c in display_df.columns]
table_df = display_df[_table_cols].rename(
    columns={
        "session_label": "Run label",
        "run_date": "Date",
        "source_file": "Source file",
        "status": "Status",
        "cached_from_run_id": "Cached from run",
        "training_type": "Training type",
        "distance_km": "Distance (km)",
        "duration": "Duration",
        "pace": "Avg pace",
        "avg_hr": "Avg HR",
        "avg_power": "Avg power",
        "trend_label": "Trend",
        "next_session": "Next session",
    }
)
if "Run label" in table_df.columns:
    table_df["Run label"] = display_df["session_label"].fillna("-")
if "Training type" in table_df.columns:
    table_df["Training type"] = display_df["training_type"].apply(format_training_type_label)
if "Trend" in table_df.columns:
    table_df["Trend"] = display_df["trend_label"].apply(format_trend_label)
if "Status" in table_df.columns:
    table_df["Status"] = table_df["Status"].fillna("-")
if "Cached from run" in table_df.columns:
    table_df["Cached from run"] = table_df["Cached from run"].fillna("-")
if "Avg power" in table_df.columns:
    table_df = table_df.copy()
    ap = display_df["avg_power"]
    table_df["Avg power"] = [
        format_power_w(v) if v is not None and not (isinstance(v, float) and pd.isna(v)) else "-"
        for v in ap
    ]

fit_lines = build_fit_derived_metrics_lines(run_row)
show_structured_details = _is_structured_selected_run(
    run_row,
    work_only=_work_only_payload,
    work_block_label=_work_block_label,
    work_family=_work_session_family,
    sel_payload=_sel_payload,
)

if is_duplicate_cached:
    canon = run_row.get("cached_from_run_id")
    if canon is None or (isinstance(canon, float) and pd.isna(canon)) or str(canon).strip() == "":
        canon = run_row.get("run_id")
    llm_line = (
        "**Using saved AI summary for this activity.**"
        if has_llm_summary_text
        else "No saved AI summary; text below comes from stored analysis in the database."
    )
    st.info(
        f"**Showing a cached result** (same activity as run `{canon}`). {llm_line}"
    )
    st.caption(
        "Duplicate import: no new analysis run; "
        f"saved summary present: **{'yes' if has_llm_summary_text else 'no'}**."
    )

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Date", run_row["run_date"] if pd.notna(run_row["run_date"]) else "-")
    st.metric("Distance", format_number(run_row["distance_km"], 2, " km"))
    st.metric("Duration", format_duration(run_row["duration_sec"]))

with col2:
    st.metric("Avg pace", format_pace(run_row["avg_pace_sec_km"]))
    st.metric("Avg HR", format_int(run_row["avg_hr"]))
    st.metric("Max HR", format_int(run_row["max_hr"]))

with col3:
    st.metric("Avg power", format_power_w(run_row["avg_power"]))
    st.metric("Training type", format_training_type_label(run_row.get("training_type")))
    st.metric("Trend", format_trend_label(run_row.get("trend_label")))

_dq_val = run_row.get("data_quality_score")
_dq_warn = (
    _dq_val is not None
    and not (isinstance(_dq_val, float) and pd.isna(_dq_val))
    and float(_dq_val) < 50.0
)
if _dq_warn:
    st.warning(
        f"Limited data quality for this run ({format_number(_dq_val, 0)}/100) may reduce interpretation confidence."
    )

if show_structured_details:
    st.caption(
        "Top metrics reflect the full run, including warm-up, recovery, and cool-down."
    )

_exec_label = str(_scalar_str(run_row, "execution_quality") or "-").replace("_", " ").strip().title()
_fit_label = str(_scalar_str(run_row, "fitness_signal") or "-").replace("_", " ").strip().title()
_fat_label = str(_scalar_str(run_row, "fatigue_signal") or "-").replace("_", " ").strip().title()
st.markdown(
    (
        "<div style='margin-top:0.2rem; margin-bottom:0.6rem; display:flex; flex-wrap:wrap; gap:0.5rem;'>"
        f"<span style='display:inline-block; background:#1f2937; color:#e5e7eb; border:1px solid #374151; "
        "border-radius:999px; padding:0.48rem 0.95rem; font-size:1.5rem; font-weight:600; line-height:1.1;'>"
        f"Execution: {_exec_label}</span>"
        f"<span style='display:inline-block; background:#1f2937; color:#e5e7eb; border:1px solid #374151; "
        "border-radius:999px; padding:0.48rem 0.95rem; font-size:1.5rem; font-weight:600; line-height:1.1;'>"
        f"Fitness: {_fit_label}</span>"
        f"<span style='display:inline-block; background:#1f2937; color:#e5e7eb; border:1px solid #374151; "
        "border-radius:999px; padding:0.48rem 0.95rem; font-size:1.5rem; font-weight:600; line-height:1.1;'>"
        f"Fatigue: {_fat_label}</span>"
        "</div>"
    ),
    unsafe_allow_html=True,
)

if show_structured_details:
    st.markdown("**Workout details (work intervals only)**")
    wd_col1, wd_col2, wd_col3 = st.columns(3)
    work_label_display = _work_block_label if _work_block_label else "-"
    if work_label_display.lower() == "no work blocks":
        work_label_display = "-"
    with wd_col1:
        st.metric("Work block", work_label_display)
        st.metric(
            "Total work time",
            format_duration(
                _work_only_payload.get("work_total_time_sec")
                if isinstance(_work_only_payload, dict)
                else None
            ),
        )
    with wd_col2:
        st.metric(
            "Work pace",
            format_pace(
                _work_only_payload.get("work_mean_pace_sec_per_km")
                if isinstance(_work_only_payload, dict)
                else None
            ),
        )
        st.metric(
            "Work power",
            format_power_w(
                _work_only_payload.get("work_mean_power_w")
                if isinstance(_work_only_payload, dict)
                else None
            ),
        )
    with wd_col3:
        st.metric(
            "Work HR",
            format_int(
                _work_only_payload.get("work_mean_hr_avg")
                if isinstance(_work_only_payload, dict)
                else None
            ),
        )
        _work_w_per_hr = (
            _work_only_payload.get("work_w_per_hr")
            if isinstance(_work_only_payload, dict)
            else None
        )
        st.metric(
            "Work W/HR",
            format_number(_work_w_per_hr, 2) if not _is_missing_series_value(_work_w_per_hr) else "-",
        )

st.subheader("Quick coaching view")
_top_trace_payload = load_llm_trace_for_run_row(selected_run_id, run_row)
_card_text, _full_ai_for_lower = _top_ai_summary_card_text(run_row, _top_trace_payload)
_ctx_full_text = str(run_row.get("llm_context_progress") or "").strip()
if not _ctx_full_text and _top_trace_payload is not None:
    _ctx_trace = _top_trace_payload.get("context_progress_trace", {})
    _ctx_full_text = str(_ctx_trace.get("context_interpretation") or "").strip()
_ctx_short_text = str(run_row.get("llm_context_progress_short") or "").strip()
if not _ctx_short_text and _top_trace_payload is not None:
    _ctx_trace = _top_trace_payload.get("context_progress_trace", {})
    _ctx_short_text = str(_ctx_trace.get("context_insight_short") or "").strip()
if not _ctx_short_text and _ctx_full_text:
    _ctx_short_text = _deterministic_top_context_summary(_ctx_full_text)
_pace_is_slower_ctx = _context_pace_is_slower_from_trace_payload(_top_trace_payload)
_ctx_short_text = _sanitize_top_context_pace_wording(
    _ctx_short_text, pace_is_slower=_pace_is_slower_ctx
)

_this_sess_line = _quick_coaching_this_session_line(run_row)
if _this_sess_line:
    st.markdown(_this_sess_line)

st.markdown("**What happened**")
if _ctx_short_text:
    st.info(_ctx_short_text)
elif _card_text:
    st.info(_card_text)
else:
    st.info("No AI context/progress insight stored for this run yet.")

st.markdown("**What next**")
st.info(_markdown_quick_coaching_what_next(run_row))

st.subheader("Current training picture")

render_performance_overview(df_chart, DB_PATH)

render_weekly_training_summary(df_chart)

render_best_recent_indicators(DB_PATH)

with st.expander("Context & trust", expanded=False):
    render_archive_coverage(df_chart)
    render_trend_confidence(df_chart)

with st.expander("Deeper analysis", expanded=False):
    st.markdown("#### Interval insight")
    if _insight_exc is not None:
        st.caption(f"Interval insight unavailable: {_insight_exc}")
    elif _sel_payload is not None:
        st.text(format_selected_run_interval_family_insight(_sel_payload))
        render_interval_family_trend_charts(_sel_payload)
        render_interval_family_history_table(_sel_payload)
    else:
        st.caption("Interval insight unavailable.")

    render_structured_workout_analysis_block(
        show_structured_details=show_structured_details,
        run_row=run_row,
        selected_run_id=selected_run_id,
        rep_analysis=_structured_rep_analysis,
        exact_compare=_exact_structured_compare,
        sel_payload=_sel_payload,
        current_work_only=_work_only_payload,
        baseline_work_only=_baseline_work_only_payload,
        baseline_rep_analysis=_baseline_rep_analysis,
        comparison_mode=_structured_comparison_mode,
    )

    st.markdown("#### Analysis")

    analysis_col1, analysis_col2 = st.columns(2)

    with analysis_col1:
        st.write(
            f"**Execution quality:** {run_row['execution_quality']}"
            if pd.notna(run_row["execution_quality"])
            else "**Execution quality:** -"
        )
        st.write(
            f"**Fatigue signal:** {run_row['fatigue_signal']}"
            if pd.notna(run_row["fatigue_signal"])
            else "**Fatigue signal:** -"
        )
        st.write(
            f"**Fitness signal:** {run_row['fitness_signal']}"
            if pd.notna(run_row["fitness_signal"])
            else "**Fitness signal:** -"
        )

    with analysis_col2:
        st.write(f"**Elevation gain:** {format_number(run_row['elevation_gain_m'], 0, ' m')}")
        st.write(f"**Avg cadence (/min):** {format_int(run_row['avg_cadence'])}")
        st.write(f"**Training load:** {format_number(run_row['training_load'], 0)}")

    if fit_lines:
        with st.expander("FIT file metrics (zones, moving time)", expanded=False):
            st.caption("From the database (typically Garmin FIT imports).")
            st.markdown("\n".join(f"- {line}" for line in fit_lines))

    st.caption(
        "Threshold / VO2 / easy panels, 4- and 12-week views, training regularity, "
        "personal baseline, and long-range trend charts."
    )
    st.markdown("#### Performance domains")
    render_work_family_progression_panels(DB_PATH)

    _trend_df = df_chart.sort_values("run_date_dt").copy()
    _trend_df = _trend_df[_trend_df["run_date_dt"].notna()]

    st.markdown("#### Multi-week training context")
    render_four_week_consistency_progression(df_chart)

    render_twelve_week_progression(df_chart)

    render_training_regularity_availability(df_chart)

    render_current_level_personal_baseline(DB_PATH)

    st.markdown("#### Trends over time")
    if _trend_df.empty:
        st.info("No valid dates available yet for charts.")
    else:
        render_main_dashboard_trends_charts(_trend_df)

    with st.expander("Diagnostics", expanded=False):
        render_work_family_membership_diagnostics(DB_PATH)
        render_work_segment_family_distribution_diagnostic(DB_PATH)
        with st.expander("Recommendation source details (easy/recovery drift)", expanded=False):
            render_recommendation_easy_drift_debug(DB_PATH, selected_run_id)
        st.subheader("Runs")
        st.dataframe(table_df, width="stretch", hide_index=True)

render_import_section(expanded=False)

trace_payload = load_llm_trace_for_run_row(selected_run_id, run_row)

_full_ai_text = _full_ai_for_lower
if not _full_ai_text:
    if trace_payload is None:
        if _has_full_llm:
            _full_ai_text = str(run_row.get("llm_summary") or "")
    else:
        _full_trace = trace_payload.get("trace", {})
        _full_ai_text = str(_full_trace.get("final_summary") or "").strip() or None

_full_context_ai_text = str(run_row.get("llm_context_progress") or "").strip() or None
if not _full_context_ai_text and trace_payload is not None:
    _ctx_trace = trace_payload.get("context_progress_trace", {})
    _full_context_ai_text = str(_ctx_trace.get("context_interpretation") or "").strip() or None

with st.expander("LLM response and debug", expanded=False):
    if _full_ai_text:
        st.markdown("**Full run interpretation**")
        with st.container(border=True):
            st.markdown(_full_ai_text)
    else:
        st.caption("No detailed AI interpretation stored for this run yet.")

    st.markdown("")
    if _full_context_ai_text:
        st.markdown("**Full context/progress interpretation**")
        with st.container(border=True):
            st.markdown(_full_context_ai_text)
    else:
        st.caption("No AI context/progress interpretation stored for this run yet.")

    _rs_raw = run_row.get("recommendation_signals")
    if _rs_raw is not None and not (isinstance(_rs_raw, float) and pd.isna(_rs_raw)):
        _rs_s = str(_rs_raw).strip()
        if _rs_s and _rs_s != "{}":
            st.markdown("")
            st.markdown("**Deterministic recommendation signals (debug)**")
            try:
                st.json(json.loads(_rs_s))
            except json.JSONDecodeError:
                st.text(_rs_s)

    st.markdown("")
    st.markdown("**Run technical details**")
    with st.container(border=True):
        _run_id_display = _scalar_str(run_row, "run_id") or "-"
        _src_file_display = _scalar_str(run_row, "source_file") or "-"
        _cached_from_display = _scalar_str(run_row, "cached_from_run_id") or "-"
        _status_display = _scalar_str(run_row, "status") or "-"
        _dq_display = (
            format_number(run_row.get("data_quality_score"), 0)
            if not _is_missing_series_value(run_row.get("data_quality_score"))
            else "-"
        )
        st.markdown(f"- Run ID: `{_run_id_display}`")
        st.markdown(f"- Source file: `{_src_file_display}`")
        st.markdown(f"- Cached-from run ID: `{_cached_from_display}`")
        st.markdown(f"- Status: `{_status_display}`")
        st.markdown(f"- Data quality: `{_dq_display}`")

    st.markdown("")
    st.markdown("**Technical metadata**")

    if trace_payload is None:
        st.caption("No stored AI trace for this run yet.")
        if is_duplicate_cached:
            st.caption(
                "Duplicate imports reuse the earlier analysis; a trace may exist under the original run id "
                f"from the banner above. Saved summary text in the database: "
                f"**{'yes' if has_llm_summary_text else 'no'}**."
            )
    else:
        trace = trace_payload.get("trace", {})
        used_llm = bool(trace.get("used_llm"))
        trace_run_id = str(trace_payload.get("run_id") or "").strip()
        sel_id = str(selected_run_id or "").strip()
        same_run_trace = not trace_run_id or not sel_id or trace_run_id == sel_id

        st.markdown(
            f"**Model call:** {'Yes' if used_llm else 'No'} · "
            f"**Model:** {trace.get('model') or '—'} · "
            f"**Outcome:** {_friendly_llm_trace_status(trace.get('status'), used_llm)}"
        )
        if same_run_trace:
            st.caption("Summary and trace belong to this run.")
        else:
            st.caption(
                f"Stored summary and trace match the earlier run **{trace_run_id}** "
                "(same activity re-imported; not generated again for this row)."
            )

        lcm = trace_payload.get("llm_context_metadata")
        if isinstance(lcm, dict) and lcm:
            st.caption(_llm_grounding_at_a_glance(lcm))
        else:
            st.caption("Prompt context: not available (older import or missing fields).")

        _pga = trace_payload.get("prompt_grounding_audit")
        st.markdown("")
        st.markdown("**Prompt bundle audit**")
        if isinstance(_pga, dict) and _pga:
            st.markdown(
                f"- Structured recommendation signals in prompt: **{_pga.get('structured_recommendation_signals_in_prompt', '—')}**\n"
                f"- Family history block in prompt: **{_pga.get('family_history_block_in_prompt', '—')}**\n"
                f"- Recommendation candidates (signals JSON) in prompt: **{_pga.get('recommendation_candidates_in_prompt', '—')}**"
            )
        else:
            st.caption(
                "Not recorded for this trace (re-run ingest or `backfill-ai-summaries` with current AgenticRun)."
            )

        st.markdown("")
        st.markdown("**Technical details**")
        with st.expander("What was sent to the model", expanded=False):
            st.caption("Structured context flags and the full prompt text from the saved trace.")
            if isinstance(lcm, dict) and lcm:
                for line in _llm_context_meta_display_lines(lcm):
                    st.markdown(f"- {line}")
            else:
                st.caption("No structured context metadata on file.")
            with st.expander("Full prompt text", expanded=False):
                st.code(trace.get("prompt") or "", language="text")

        with st.expander("Model response (raw)", expanded=False):
            st.code(trace.get("raw_response") or "", language="text")

        if trace.get("error"):
            st.error(trace["error"])