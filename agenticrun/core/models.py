from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class RunRecord:
    run_id: str
    source_file: str
    source_type: str
    run_date: str
    title: str
    distance_km: Optional[float] = None
    duration_sec: Optional[float] = None
    avg_pace_sec_km: Optional[float] = None
    avg_hr: Optional[float] = None
    max_hr: Optional[float] = None
    avg_power: Optional[float] = None
    max_power: Optional[float] = None
    avg_cadence: Optional[float] = None
    elevation_gain_m: Optional[float] = None
    training_load: Optional[float] = None
    notes: str = ""
    fit_activity_key: Optional[str] = None
    raw_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["raw_summary"] = str(self.raw_summary)
        return payload


@dataclass
class RunAnalysis:
    training_type: str = "unknown"
    intensity_label: str = "unknown"
    execution_quality: str = "unknown"
    confidence: float = 0.5
    session_flags: list[str] = field(default_factory=list)
    summary: str = ""
    classification_trace: str = ""


@dataclass
class TrendAssessment:
    history_count: int = 0
    similar_count: int = 0
    trend_label: str = "insufficient_history"
    fitness_signal: str = "unknown"
    fatigue_signal: str = "unknown"
    trend_summary: str = ""


@dataclass
class Recommendation:
    next_session: str = "Keep the next session easy until more evidence is available."
    load_action: str = "hold"
    warning_flag: bool = False
    recommendation_summary: str = ""
    # Structured deterministic signals for hybrid LLM coaching (facts/classifications/candidates).
    recommendation_signals: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunState:
    source_path: str
    status: str = "initialized"
    run_record: Optional[RunRecord] = None
    analysis: RunAnalysis = field(default_factory=RunAnalysis)
    trend: TrendAssessment = field(default_factory=TrendAssessment)
    recommendation: Recommendation = field(default_factory=Recommendation)
    warnings: list[str] = field(default_factory=list)
    llm_summary: str = ""
    llm_summary_short: str = ""
    # Separate context/progress interpretation layer (historical-comparison focused).
    llm_context_progress: str = ""
    llm_context_progress_short: str = ""
    # Top “What next” line in Quick coaching; LLM-grounded, separate from short_summary.
    llm_what_next_short: str = ""
    # Injected into LLM prompt during ingest (deterministic bundle from DB + analysis).
    llm_prompt_deterministic: dict[str, Any] = field(default_factory=dict)
    # Structured deterministic comparison bundle for context/progress interpretation.
    llm_context_progress_bundle: dict[str, Any] = field(default_factory=dict)
    # When set, this row was built from an existing DB run (duplicate FIT upload).
    cached_from_run_id: Optional[str] = None
    # Populated during FIT import; consumed when resolving zone profiles during ingest.
    fit_stream_bundle: tuple[list[object], list[float | None], list[float | None]] | None = None

    def as_flat_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"source_path": self.source_path, "status": self.status}
        if self.cached_from_run_id:
            result["cached_from_run_id"] = self.cached_from_run_id
        if self.run_record:
            result.update(self.run_record.to_dict())
        result.update({
            "training_type": self.analysis.training_type,
            "intensity_label": self.analysis.intensity_label,
            "execution_quality": self.analysis.execution_quality,
            "analysis_confidence": self.analysis.confidence,
            "session_flags": ", ".join(self.analysis.session_flags),
            "analysis_summary": self.analysis.summary,
            "classification_trace": self.analysis.classification_trace,
            "history_count": self.trend.history_count,
            "similar_count": self.trend.similar_count,
            "trend_label": self.trend.trend_label,
            "fitness_signal": self.trend.fitness_signal,
            "fatigue_signal": self.trend.fatigue_signal,
            "trend_summary": self.trend.trend_summary,
            "next_session": self.recommendation.next_session,
            "load_action": self.recommendation.load_action,
            "warning_flag": self.recommendation.warning_flag,
            "recommendation_summary": self.recommendation.recommendation_summary,
            "recommendation_signals": (
                json.dumps(self.recommendation.recommendation_signals, ensure_ascii=False)
                if self.recommendation.recommendation_signals
                else "{}"
            ),
            "warnings": "; ".join(self.warnings),
            "llm_summary": self.llm_summary,
            "llm_summary_short": self.llm_summary_short,
            "llm_what_next_short": self.llm_what_next_short,
            "llm_context_progress": self.llm_context_progress,
            "llm_context_progress_short": self.llm_context_progress_short,
        })
        return result
