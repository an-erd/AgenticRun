from __future__ import annotations

from statistics import mean

from agenticrun.core.models import RunState
from agenticrun.core.session_fit_metrics import as_bool_flag, session_fit_metrics

LOW_DATA_QUALITY_THRESHOLD = 50.0
PACE_TREND_DATA_QUALITY_THRESHOLD = 60.0
PACE_TREND_DELTA_SEC_KM = 8.0


class TrendAgent:
    def run(self, state: RunState, history: list[dict]) -> RunState:
        current = state.run_record
        if not current:
            return state

        history_without_current = [h for h in history if h.get("run_id") != current.run_id]
        state.trend.history_count = len(history_without_current)
        if len(history_without_current) < 2:
            state.trend.trend_label = "insufficient_history"
            state.trend.fitness_signal = "unknown"
            state.trend.fatigue_signal = "unknown"
            state.trend.trend_summary = "Not enough history yet for meaningful comparison."
            return state

        same_type = [h for h in history_without_current if h.get("training_type") == state.analysis.training_type]
        comparable = same_type if same_type else history_without_current[-5:]
        state.trend.similar_count = len(comparable)

        fm = session_fit_metrics(current)
        dq = fm.get("data_quality_score")
        if dq is not None and float(dq) < LOW_DATA_QUALITY_THRESHOLD:
            state.trend.trend_label = "uncertain_data_quality"
            state.trend.fitness_signal = "unknown"
            state.trend.fatigue_signal = "unknown"
            state.trend.trend_summary = (
                f"Trend comparison suppressed: data quality score is low ({float(dq):.0f}/100). "
                f"Gather cleaner HR/power/GPS streams before reading fitness or fatigue from trends."
            )
            return state

        avg_power_hist = [h.get("avg_power") for h in comparable if h.get("avg_power") is not None]
        avg_hr_hist = [h.get("avg_hr") for h in comparable if h.get("avg_hr") is not None]

        power_signal = "unknown"
        hr_signal = "unknown"
        moving_pace_trend = "insufficient_data"

        pace_hist_rows: list[tuple[float, float | None, float | None]] = []
        for h in comparable:
            hfm = session_fit_metrics(h)
            pace_val = hfm.get("avg_moving_pace_sec_km")
            if pace_val is None:
                continue
            pace_hist_rows.append((
                float(pace_val),
                hfm.get("moving_time_sec"),
                hfm.get("data_quality_score"),
            ))
        pace_curr = fm.get("avg_moving_pace_sec_km")
        if pace_curr is not None and len(pace_hist_rows) >= 2:
            recent_hist = pace_hist_rows[-3:]
            hist_vals = [p for (p, _mt, _dq) in recent_hist]
            weak_quality_count = 0
            for _p, mt, dqv in recent_hist:
                if mt is None or (dqv is not None and float(dqv) < PACE_TREND_DATA_QUALITY_THRESHOLD):
                    weak_quality_count += 1
            curr_mt = fm.get("moving_time_sec")
            curr_dq = fm.get("data_quality_score")
            curr_weak = curr_mt is None or (
                curr_dq is not None and float(curr_dq) < PACE_TREND_DATA_QUALITY_THRESHOLD
            )
            if not curr_weak and weak_quality_count <= 1:
                hist_pace = mean(hist_vals)
                pace_delta = hist_pace - float(pace_curr)  # positive -> faster now (lower sec/km)
                if pace_delta >= PACE_TREND_DELTA_SEC_KM:
                    moving_pace_trend = "improving"
                elif pace_delta <= -PACE_TREND_DELTA_SEC_KM:
                    moving_pace_trend = "slowing"
                else:
                    moving_pace_trend = "stable"

        if current.avg_power is not None and avg_power_hist:
            hist_power = mean(avg_power_hist)
            power_signal = "positive" if current.avg_power > hist_power + 5 else "neutral_or_lower"
        elif as_bool_flag(fm.get("has_gps")):
            pace_curr = fm.get("avg_moving_pace_sec_km")
            pace_hist = [
                session_fit_metrics(h).get("avg_moving_pace_sec_km")
                for h in comparable
                if session_fit_metrics(h).get("avg_moving_pace_sec_km") is not None
            ]
            if pace_curr is not None and pace_hist:
                hist_pace = mean(pace_hist)
                if pace_curr < hist_pace - 5:
                    power_signal = "positive"
                elif pace_curr > hist_pace + 5:
                    power_signal = "neutral_or_lower"
                else:
                    power_signal = "stable"

        if current.avg_hr is not None and avg_hr_hist:
            hist_hr = mean(avg_hr_hist)
            hr_signal = "elevated" if current.avg_hr > hist_hr + 4 else "stable"

        if (power_signal == "positive" or moving_pace_trend == "improving") and hr_signal == "stable":
            state.trend.trend_label = "positive_progress"
            state.trend.fitness_signal = "positive"
            state.trend.fatigue_signal = "low"
        elif hr_signal == "elevated" and power_signal != "positive" and moving_pace_trend != "improving":
            state.trend.trend_label = "possible_fatigue"
            state.trend.fitness_signal = "neutral"
            state.trend.fatigue_signal = "moderate"
        elif moving_pace_trend == "slowing":
            state.trend.trend_label = "possible_fatigue"
            state.trend.fitness_signal = "neutral"
            state.trend.fatigue_signal = "moderate"
        else:
            state.trend.trend_label = "stable"
            state.trend.fitness_signal = "neutral"
            state.trend.fatigue_signal = "low"

        state.trend.trend_summary = (
            f"Compared against {len(comparable)} similar historical sessions. "
            f"Power signal: {power_signal}. HR signal: {hr_signal}. "
            f"Moving-pace trend: {moving_pace_trend}."
        )
        return state
