from __future__ import annotations

import json
import os
import re
import traceback
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

from agenticrun.core.models import RunState

# Lazy import: `import openai` can block or be very slow at import time; defer until client is needed.
_OPENAI_SENTINEL = object()
_openai_client_cls: object = _OPENAI_SENTINEL


def _get_openai_client_cls() -> Any | None:
    """Return OpenAI client class or None if the SDK is unavailable."""
    global _openai_client_cls
    if _openai_client_cls is not _OPENAI_SENTINEL:
        return _openai_client_cls  # type: ignore[return-value]
    try:
        from openai import OpenAI

        _openai_client_cls = OpenAI
    except Exception:
        _openai_client_cls = None
    return _openai_client_cls  # type: ignore[return-value]


class LLMService:
    def __init__(self) -> None:
        self.debug = os.getenv("AGENTICRUN_DEBUG", "").lower() in {"1", "true", "yes", "on"}
        self.llm_enabled = os.getenv("AGENTICRUN_LLM_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
        self.env_path = ".env"
        self.env_loaded = load_dotenv(self.env_path)
        self.env_source = self.env_path if self.env_loaded else "process_env"
        self.api_key = (os.getenv("OPENAI_API_KEY", "") or "").strip()
        self.model = (os.getenv("OPENAI_MODEL", "gpt-4.1-mini") or "").strip()
        self.base_url = (os.getenv("OPENAI_BASE_URL", "") or "").strip()
        self.provider = "openai"
        self.client = None
        self.init_status = "not_initialized"
        self.init_error: str | None = None

        readiness = self.check_readiness()
        if not readiness["ready"]:
            self.init_status = "config_not_ready"
            self.init_error = "; ".join(readiness.get("issues", [])) or "LLM configuration not ready."
            self._debug(f"LLM not ready: {self.init_error}")

        self._debug(
            "LLM init: "
            f"provider={self.provider}, model={self.model or '<empty>'}, "
            f"AGENTICRUN_LLM_ENABLED={'yes' if self.llm_enabled else 'no'} "
            f"(env gate; separate from per-ingest enabled_for_run), "
            f"api_key_present={'yes' if bool(self.api_key) else 'no'}, "
            f"base_url_set={'yes' if bool(self.base_url) else 'no'}, "
            f"env_loaded={'yes' if self.env_loaded else 'no'} from {self.env_source}"
        )

    def available(self) -> bool:
        return self.client is not None

    def format_ingest_runtime_status_line(
        self,
        *,
        enabled_for_run: bool,
        used_for_run: bool | None = None,
    ) -> str:
        """Single-line snapshot: config vs env gate vs CLI/UI gate vs actual API use.

        - *configured*: API key and model string present (credentials to target a model).
        - *available*: :meth:`check_readiness` is satisfied (SDK, key, model, URL, env gate, …).
        - *enabled_for_run*: ``--llm`` / Streamlit checkbox for this ingest only.
        - *used_for_run*: ``yes`` / ``no`` if an API call was made this ingest; ``n/a`` at batch start.
        - *cache_reuse_possible*: duplicate FIT path can surface stored ``llm_summary`` without a new call.
        """
        readiness = self.check_readiness()
        configured = bool(self.api_key) and bool(self.model)
        available = bool(readiness.get("ready"))
        if used_for_run is None:
            used_s = "n/a"
        else:
            used_s = "yes" if used_for_run else "no"
        return (
            "llm_runtime_status: "
            f"configured={'yes' if configured else 'no'} "
            f"available={'yes' if available else 'no'} "
            f"enabled_for_run={'yes' if enabled_for_run else 'no'} "
            f"used_for_run={used_s} "
            f"cache_reuse_possible=yes"
        )

    def _ensure_client(self) -> bool:
        if self.client is not None:
            return True
        readiness = self.check_readiness()
        if not readiness.get("ready"):
            self.init_status = "config_not_ready"
            self.init_error = "; ".join(readiness.get("issues", [])) or "LLM configuration not ready."
            return False
        try:
            OpenAI = _get_openai_client_cls()
            client_kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self.client = OpenAI(**client_kwargs) if OpenAI else None
            if self.client is None:
                self.init_status = "client_init_error"
                self.init_error = "OpenAI client unavailable."
                return False
            self.init_status = "ok"
            self.init_error = None
            return True
        except Exception as exc:
            self.init_status = "client_init_error"
            self.init_error = f"{type(exc).__name__}: {exc}"
            self._debug(f"LLM client initialization failed: {self.init_error}")
            return False

    def check_readiness(self) -> dict[str, Any]:
        issues: list[str] = []
        warnings: list[str] = []

        if not self.llm_enabled:
            issues.append("LLM mode is disabled via AGENTICRUN_LLM_ENABLED.")
        if _get_openai_client_cls() is None:
            issues.append("OpenAI SDK is not available.")
        if not self.model:
            issues.append("OPENAI_MODEL is missing or empty.")
        if not self.api_key:
            issues.append("OPENAI_API_KEY is missing or empty.")
        elif not self.api_key.startswith("sk-"):
            warnings.append("OPENAI_API_KEY format does not match expected 'sk-' prefix.")
        if self.base_url and not (self.base_url.startswith("http://") or self.base_url.startswith("https://")):
            issues.append("OPENAI_BASE_URL must start with http:// or https://.")

        return {
            "ready": len(issues) == 0,
            "provider": self.provider,
            "model": self.model or None,
            "api_key_present": bool(self.api_key),
            "env_source": self.env_source,
            "issues": issues,
            "warnings": warnings,
        }

    def readiness_report(self) -> dict[str, Any]:
        readiness = self.check_readiness()
        return {
            "ready": readiness["ready"],
            "provider": self.provider,
            "model": self.model or None,
            "llm_enabled": self.llm_enabled,
            "api_key_present": bool(self.api_key),
            "base_url_set": bool(self.base_url),
            "env_loaded": self.env_loaded,
            "env_source": self.env_source,
            "init_status": self.init_status,
            "init_error": self.init_error,
            "issues": readiness.get("issues", []),
            "warnings": readiness.get("warnings", []),
        }

    def build_prompt(self, state: RunState) -> str:
        det = state.llm_prompt_deterministic or {}
        context_line = det.get(
            "llm_prompt_context_line",
            "llm_prompt_context: deterministic_run_takeaway=no interval_insight=no family_history=no",
        )
        findings = (det.get("findings_text") or "").strip()

        rr = state.run_record
        file_facts = (
            f"Run date: {rr.run_date if rr else 'unknown'}\n"
            f"Title: {rr.title if rr else 'unknown'}\n"
            f"Distance km: {rr.distance_km if rr else 'unknown'}\n"
            f"Duration sec: {rr.duration_sec if rr else 'unknown'}\n"
            f"Avg HR: {rr.avg_hr if rr else 'unknown'}\n"
            f"Avg Power: {rr.avg_power if rr else 'unknown'}\n"
        )
        warn_s = ", ".join(state.warnings) if state.warnings else "none"

        intro = """You are a careful running-analysis assistant for AgenticRun.
Your job is to write two coach-style outputs that explain what the app already concluded—not to re-score the workout or add new analysis.
Rules: use only the data below; do not invent metrics, sessions, or trends; do not contradict the deterministic findings; do not make medical claims.
Avoid repeating raw numbers that are already spelled out in the findings (pace, HR, power, distance) unless one figure is needed to nail a single point."""

        output_contract = """Return valid JSON only, with exactly these keys:
{
  "short_summary": "...",
  "final_summary": "...",
  "what_next": "..."
}

short_summary requirements:
- concise top-card coaching summary for this selected run
- 2-4 sentences, around 50-90 words
- first state what this run was / what stood out
- then what it means
- then practical implication for next step
- no bullets, no technical metadata, no generic filler
- do not start with generic advice before run interpretation
- prioritize the primary story from structured recommendation signals / primary_run_read; do not let a secondary-only caution (e.g. prior-window easy/recovery label drift on steady/long) dominate the opening unless it truly matches the athlete-facing story
- for steady_run / long_run: if recommendation_signals includes comparable_aerobic_signal, treat that comparison as the primary narrative source for "what happened"
- pace wording clarity (important): pace is in sec/km, so a larger pace value means slower running. Avoid ambiguous phrases like "increase in pace" or "improved pace" unless the numeric direction is explicitly faster. Prefer plain wording such as "slightly slower pace with lower heart rate", "lower speed but better control", or "easier aerobic execution at lower cardiovascular cost" when pace is slower but HR/control is improved.

final_summary requirements:
- fuller coaching interpretation for lower section
- more detailed than short_summary
- explain what happened, what it means, and what to do next
- no bullets, no technical metadata

what_next requirements:
- 1-3 short sentences, Quick coaching “What next” line
- grounded ONLY in deterministic signals + default next-session/load candidates + comparisons already given (no new metrics)
- choose emphasis: primary coaching vs secondary cautions must follow prioritization_hints in the signals block when present
- steady_run / long_run: if easy/recovery drift appears only as secondary caution, you may mention it briefly but must not let it read like the main critique of this run"""

        if findings:
            return f"""{intro}

{context_line}

--- AgenticRun deterministic findings (authoritative) ---
{findings}

--- Recording / file facts (context only; do not recite as a table) ---
{file_facts}
Intensity: {state.analysis.intensity_label}
Warnings: {warn_s}

Write a brief coach note (about three short paragraphs or fewer, tight prose):

Priority 1 — If the block above includes an interval / same-work-family comparison vs a prior session, lead with that: what this run was, how it compares to that in-family baseline (pace/power/HR/W·HR as already labeled), and what that suggests in plain language.

Priority 2 — Decide primary vs secondary emphasis using the structured recommendation signals block (primary_run_read, dominant_rule_id, caution_signals tiers). The default next-session line is a candidate, not an order to repeat verbatim if a secondary historical flag would mis-rank the story (notably steady/long vs easy-drift).

Priority 3 — For steady_run / long_run, when comparable_aerobic_signal is present, anchor the story in current run vs that comparable aerobic prior; keep upper-zone clustering or easy/recovery drift as supporting context unless no meaningful comparable exists.

Priority 4 — State practical next-step implications (recovery, load, session type) consistent with the signals you treated as primary; phrase as actionable advice, not a copy-paste of labels.

Priority 5 — Weave in training type, trend, execution, and fatigue/fitness signals only where they sharpen the story; skip generic filler and redundant metric lists.

Wording guardrail — running terminology:
- If pace is slower but HR is lower/stable and power/efficiency is stable, describe it as easier/more controlled aerobic execution (not as "pace improvement").
- Use "faster/slower" or "higher/lower speed" when helpful to avoid pace-direction confusion.
- Strict consistency rule: never say "improved pace" when pace status is "worse" or pace delta is positive (sec/km increased). In that case, describe it as slower pace / lower speed with improved control if HR-cost metrics are favorable.

End with a clear takeaway sentence the athlete can remember after closing the app.

{output_contract}
""".strip()

        return f"""{intro}

{context_line}

--- Recording / file facts ---
{file_facts}
Training type: {state.analysis.training_type}
Intensity: {state.analysis.intensity_label}
Execution quality: {state.analysis.execution_quality}
Trend: {state.trend.trend_label}
Fitness signal: {state.trend.fitness_signal}
Fatigue signal: {state.trend.fatigue_signal}
Structured recommendation signals (JSON): {json.dumps(getattr(state.recommendation, "recommendation_signals", None) or {}, ensure_ascii=False)}
Default next-session candidate: {state.recommendation.next_session}
Default load action candidate: {state.recommendation.load_action}
Warnings: {warn_s}

(Deterministic interval-family bundle was not available for this prompt—no segment-based vs-prior comparison in the block above.)

Write the same style of brief coach note (three short paragraphs or fewer). Describe what kind of run this was, how it fits the trend and signals, and the practical next step—without inventing an interval comparison that was not provided.

{output_contract}
""".strip()

    def summarize(self, state: RunState) -> str:
        trace = self.summarize_with_trace(state)
        return trace["final_summary"]

    def _what_next_short_fallback(
        self, state: RunState, short_summary: str, final_summary: str
    ) -> str:
        """Grounded next-step line when the model omits ``what_next``."""
        ns = (state.recommendation.next_session or "").strip() if state.recommendation else ""
        if ns:
            return self._shorten_for_top_card(ns)
        return self._shorten_for_top_card(final_summary or short_summary)

    def _context_pace_is_slower(self, context_bundle: dict[str, Any] | None) -> bool:
        """True when deterministic context says pace is slower (sec/km increased)."""
        bundle = context_bundle or {}
        rec_sig = bundle.get("recommendation_signals") or {}
        cmp_sig = rec_sig.get("comparable_aerobic_signal") if isinstance(rec_sig, dict) else None
        if not isinstance(cmp_sig, dict):
            return False
        metrics = cmp_sig.get("metrics")
        if not isinstance(metrics, dict):
            return False
        pace = metrics.get("avg_pace_sec_km")
        if not isinstance(pace, dict):
            return False
        status = str(pace.get("status") or "").strip().lower()
        if status == "worse":
            return True
        try:
            delta = float(pace.get("delta"))
        except (TypeError, ValueError):
            delta = None
        return bool(delta is not None and delta > 0)

    def _sanitize_context_pace_direction(self, text: str, context_bundle: dict[str, Any] | None) -> str:
        """Guardrail against contradictory pace wording in context/progress output."""
        if not text or not self._context_pace_is_slower(context_bundle):
            return text
        out = text
        replacements: list[tuple[str, str]] = [
            (r"\bmeaningful pace gain\b", "slightly slower pace with lower heart rate"),
            (r"\bpace gain\b", "slightly slower pace with lower heart rate"),
            (r"\bimproved pace\b", "slower pace with better control"),
            (r"\bfaster pace\b", "lower speed with better aerobic control"),
            (r"\bpace improved\b", "pace was slightly slower while aerobic control improved"),
            (r"\bpace improvement\b", "slower pace with lower cardiovascular cost"),
        ]
        for pat, repl in replacements:
            out = re.sub(pat, repl, out, flags=re.IGNORECASE)
        return out

    def summarize_with_trace(self, state: RunState) -> dict[str, Any]:
        stage = "request_build"
        prompt = self.build_prompt(state)
        timestamp = self._now_iso()
        self._debug(
            "LLM request starting: "
            f"AGENTICRUN_LLM_ENABLED={'yes' if self.llm_enabled else 'no'}, "
            f"model={self.model or '<empty>'}, "
            f"api_key_present={'yes' if bool(self.api_key) else 'no'}, "
            f"base_url_set={'yes' if bool(self.base_url) else 'no'}"
        )

        readiness = self.readiness_report()
        self._ensure_client()
        if not self.client:
            fallback = self._fallback(state)
            fb_short = self._shorten_for_top_card(fallback)
            what_next_fb = self._what_next_short_fallback(state, fb_short, fallback)
            diagnostics = self._config_diagnostics(readiness)
            self._debug(
                "LLM unavailable, using fallback: "
                f"{diagnostics['error_type']} - {diagnostics['error_message']}"
            )
            return {
                "success": False,
                "status": "fallback",
                "model": None,
                "used_llm": False,
                "fallback_used": True,
                "content": fallback,
                "provider": self.provider,
                "prompt": prompt,
                "raw_response": None,
                "final_summary": fallback,
                "short_summary": fb_short,
                "what_next": what_next_fb,
                "error": diagnostics["error_message"],
                "stage": "initialization",
                "error_type": diagnostics["error_type"],
                "error_message": diagnostics["error_message"],
                "likely_cause": diagnostics["likely_cause"],
                "suggested_fix": diagnostics["suggested_fix"],
                "timestamp": timestamp,
                "readiness": readiness,
                "diagnostics": diagnostics,
            }
        try:
            stage = "request_send"
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
            )
            stage = "response_parse"
            output_text = (response.output_text or "").strip()
            if not output_text:
                raise ValueError("Model response did not include output_text.")
            parsed = self._parse_structured_summaries(output_text)
            final_summary = str(parsed.get("final_summary") or "").strip()
            short_summary = str(parsed.get("short_summary") or "").strip()
            parse_ok = bool(final_summary and short_summary)
            parse_error = None if parse_ok else "LLM output did not provide valid short_summary + final_summary JSON."
            if not final_summary:
                final_summary = output_text
            if not short_summary:
                short_summary = self._shorten_for_top_card(final_summary)
            what_next = str(parsed.get("what_next") or "").strip()
            if not what_next:
                what_next = self._what_next_short_fallback(state, short_summary, final_summary)
            self._debug("LLM request completed successfully.")

            return {
                "success": True,
                "status": "success" if parse_ok else "success_parse_fallback",
                "model": self.model,
                "used_llm": True,
                "fallback_used": False,
                "content": final_summary,
                "provider": self.provider,
                "prompt": prompt,
                "raw_response": output_text,
                "final_summary": final_summary,
                "short_summary": short_summary,
                "what_next": what_next,
                "error": parse_error,
                "stage": "completed",
                "error_type": None if parse_ok else "response_parse_error",
                "error_message": None if parse_ok else parse_error,
                "likely_cause": None if parse_ok else "Model returned text that did not match required JSON shape.",
                "suggested_fix": None if parse_ok else "Retry; ensure model returns valid JSON with short_summary and final_summary.",
                "timestamp": timestamp,
                "parse_ok": parse_ok,
                "readiness": readiness,
                "diagnostics": {
                    "stage": "completed",
                    "error_type": None if parse_ok else "response_parse_error",
                    "error_message": None if parse_ok else parse_error,
                    "likely_cause": None if parse_ok else "Model returned text that did not match required JSON shape.",
                    "suggested_fix": None if parse_ok else "Retry; ensure model returns valid JSON with short_summary and final_summary.",
                },
            }
        except Exception as e:
            fallback = self._fallback(state)
            err_short = self._shorten_for_top_card(fallback)
            what_next_err = self._what_next_short_fallback(state, err_short, fallback)
            diagnostics = self._classify_error(e, stage=stage)
            self._debug(
                "LLM request failed; using fallback: "
                f"type={diagnostics['error_type']}, message={diagnostics['error_message']}"
            )
            tb_summary = traceback.format_exc(limit=3)
            self._debug(f"LLM traceback (short): {tb_summary}")
            return {
                "success": False,
                "status": "error_fallback",
                "model": self.model,
                "used_llm": False,
                "fallback_used": True,
                "content": fallback,
                "provider": self.provider,
                "prompt": prompt,
                "raw_response": None,
                "final_summary": fallback,
                "short_summary": err_short,
                "what_next": what_next_err,
                "error": diagnostics["error_message"],
                "stage": diagnostics["stage"],
                "error_type": diagnostics["error_type"],
                "error_message": diagnostics["error_message"],
                "likely_cause": diagnostics["likely_cause"],
                "suggested_fix": diagnostics["suggested_fix"],
                "timestamp": timestamp,
                "exception_type": type(e).__name__,
                "traceback_summary": tb_summary,
                "readiness": readiness,
                "diagnostics": diagnostics,
            }

    def build_context_progress_prompt(
        self, state: RunState, context_bundle: dict[str, Any] | None
    ) -> str:
        bundle = context_bundle or {}
        current_run = bundle.get("current_run") or {}
        comparable_run = bundle.get("comparable_run") or {}
        family_context = bundle.get("family_context") or {}
        rec_sig = bundle.get("recommendation_signals") or {}

        intro = """You are a careful running-analysis assistant for AgenticRun.
This is the second, separate interpretation layer: context/progress relative to comparable prior runs.
Use only the deterministic fields below. Do NOT recalculate comparisons from scratch and do NOT invent data.
Interpret what the deterministic comparison already indicates.
When comparable_run.match_type is long_steady_aerobic_prior, the prior session may be easy_run or recovery_run within the shared aerobic pool — treat that as a valid recurring long-aerobic anchor, not a category error.
Do not claim that no comparable prior exists when comparable_run includes a prior_run and non-empty metrics.
If recommendation_signals includes secondary caution_signals (e.g. easy/recovery drift) while primary_run_read is steady/long aerobic, treat drift as background context for this narrative — do not frame it as the headline critique of the selected run."""

        output_contract = """Return valid JSON only, with exactly these keys:
{
  "context_insight_short": "...",
  "context_interpretation": "..."
}

context_insight_short requirements:
- 2-4 sentences
- concise, coaching-oriented, user-facing
- clearly state comparison vs comparable prior, what it likely means, and practical implication
- no bullets, no technical metadata, no raw dump phrasing
- pace wording consistency: pace is sec/km (higher = slower). Never describe pace as improved if pace status is worse / delta positive.

context_interpretation requirements:
- fuller than context_insight_short
- explicitly classify this signal as one of: progress, consolidation, regression, normal_variation
- explain why using comparable_run and family_context
- include practical next-step implication consistent with recommendation_signals prioritization and load_action/next_session candidates (do not invent new metrics)
- strict pace-direction consistency: when pace status is worse or pace delta is positive (sec/km increased), do NOT claim pace gain / improved pace / faster pace; describe it as slower pace (lower speed) with better control if HR/cost is improved"""

        return f"""{intro}

Deterministic context bundle:
- current_run:
{json.dumps(current_run, ensure_ascii=False, indent=2)}
- comparable_run:
{json.dumps(comparable_run, ensure_ascii=False, indent=2)}
- family_context:
{json.dumps(family_context, ensure_ascii=False, indent=2)}
- recommendation_signals (deterministic prioritization bundle):
{json.dumps(rec_sig, ensure_ascii=False, indent=2)}

{output_contract}
""".strip()

    def summarize_context_progress_with_trace(
        self, state: RunState, context_bundle: dict[str, Any] | None
    ) -> dict[str, Any]:
        stage = "request_build"
        prompt = self.build_context_progress_prompt(state, context_bundle)
        timestamp = self._now_iso()
        readiness = self.readiness_report()
        self._ensure_client()
        if not self.client:
            full_fb, short_fb = self._fallback_context_progress_fields(state, context_bundle)
            diagnostics = self._config_diagnostics(readiness)
            return {
                "success": False,
                "status": "fallback",
                "model": None,
                "used_llm": False,
                "fallback_used": True,
                "content": full_fb,
                "provider": self.provider,
                "prompt": prompt,
                "raw_response": None,
                "context_interpretation": full_fb,
                "context_insight_short": short_fb,
                "error": diagnostics["error_message"],
                "stage": "initialization",
                "error_type": diagnostics["error_type"],
                "error_message": diagnostics["error_message"],
                "likely_cause": diagnostics["likely_cause"],
                "suggested_fix": diagnostics["suggested_fix"],
                "timestamp": timestamp,
                "readiness": readiness,
                "diagnostics": diagnostics,
            }

        try:
            stage = "request_send"
            response = self.client.responses.create(model=self.model, input=prompt)
            stage = "response_parse"
            output_text = (response.output_text or "").strip()
            if not output_text:
                raise ValueError("Model response did not include output_text.")

            parsed = self._parse_structured_context_progress(output_text)
            final_text = str(parsed.get("context_interpretation") or "").strip()
            short_text = str(parsed.get("context_insight_short") or "").strip()
            parse_ok = bool(final_text and short_text)
            parse_error = None if parse_ok else "LLM output missing context_insight_short/context_interpretation JSON."

            if not parse_ok:
                fb_full, fb_short = self._fallback_context_progress_fields(
                    state, context_bundle, raw_text=output_text
                )
                final_text = fb_full
                short_text = fb_short
            final_text = self._sanitize_context_pace_direction(final_text, context_bundle)
            short_text = self._sanitize_context_pace_direction(short_text, context_bundle)

            return {
                "success": True,
                "status": "success" if parse_ok else "success_parse_fallback",
                "model": self.model,
                "used_llm": True,
                "fallback_used": False,
                "content": final_text,
                "provider": self.provider,
                "prompt": prompt,
                "raw_response": output_text,
                "context_interpretation": final_text,
                "context_insight_short": short_text,
                "error": parse_error,
                "stage": "completed",
                "error_type": None if parse_ok else "response_parse_error",
                "error_message": None if parse_ok else parse_error,
                "likely_cause": None if parse_ok else "Model returned text that did not match required JSON shape.",
                "suggested_fix": None if parse_ok else "Retry; ensure model returns valid JSON with context_insight_short/context_interpretation.",
                "timestamp": timestamp,
                "parse_ok": parse_ok,
                "readiness": readiness,
                "diagnostics": {
                    "stage": "completed",
                    "error_type": None if parse_ok else "response_parse_error",
                    "error_message": None if parse_ok else parse_error,
                    "likely_cause": None if parse_ok else "Model returned text that did not match required JSON shape.",
                    "suggested_fix": None if parse_ok else "Retry; ensure model returns valid JSON with context_insight_short/context_interpretation.",
                },
            }
        except Exception as e:
            fb_full, fb_short = self._fallback_context_progress_fields(state, context_bundle)
            diagnostics = self._classify_error(e, stage=stage)
            tb_summary = traceback.format_exc(limit=3)
            return {
                "success": False,
                "status": "error_fallback",
                "model": self.model,
                "used_llm": False,
                "fallback_used": True,
                "content": fb_full,
                "provider": self.provider,
                "prompt": prompt,
                "raw_response": None,
                "context_interpretation": fb_full,
                "context_insight_short": fb_short,
                "error": diagnostics["error_message"],
                "stage": diagnostics["stage"],
                "error_type": diagnostics["error_type"],
                "error_message": diagnostics["error_message"],
                "likely_cause": diagnostics["likely_cause"],
                "suggested_fix": diagnostics["suggested_fix"],
                "timestamp": timestamp,
                "exception_type": type(e).__name__,
                "traceback_summary": tb_summary,
                "readiness": readiness,
                "diagnostics": diagnostics,
            }

    def _fallback(self, state: RunState) -> str:
        return (
            f"Session interpreted as {state.analysis.training_type} with {state.analysis.intensity_label} intensity. "
            f"Trend status: {state.trend.trend_label}. "
            f"Recommendation: {state.recommendation.next_session}"
        )

    def _parse_structured_summaries(self, raw_text: str) -> dict[str, str]:
        """Parse model output expecting JSON with short_summary and final_summary."""
        if not raw_text:
            return {}

        def _try_parse(candidate: str) -> dict[str, str]:
            try:
                obj = json.loads(candidate)
            except Exception:
                return {}
            if not isinstance(obj, dict):
                return {}
            short = str(obj.get("short_summary") or "").strip()
            final = str(obj.get("final_summary") or "").strip()
            wn = str(obj.get("what_next") or "").strip()
            if not short or not final:
                return {}
            out = {"short_summary": short, "final_summary": final}
            if wn:
                out["what_next"] = wn
            return out

        parsed = _try_parse(raw_text)
        if parsed:
            return parsed

        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            parsed = _try_parse(fenced.group(1).strip())
            if parsed:
                return parsed

        brace = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if brace:
            parsed = _try_parse(brace.group(0).strip())
            if parsed:
                return parsed

        return {}

    def _parse_structured_context_progress(self, raw_text: str) -> dict[str, str]:
        """Parse model output expecting JSON with context_insight_short/context_interpretation."""
        if not raw_text:
            return {}

        def _try_parse(candidate: str) -> dict[str, str]:
            try:
                obj = json.loads(candidate)
            except Exception:
                return {}
            if not isinstance(obj, dict):
                return {}
            short = str(obj.get("context_insight_short") or "").strip()
            full = str(obj.get("context_interpretation") or "").strip()
            if not short or not full:
                return {}
            short_n = len([s for s in re.split(r"(?<=[.!?])\s+", short) if s.strip()])
            full_n = len([s for s in re.split(r"(?<=[.!?])\s+", full) if s.strip()])
            # Keep strict enough to reject malformed outputs while avoiding brittle failures.
            if short_n < 2 or short_n > 5:
                return {}
            if full_n < 3:
                return {}
            return {
                "context_insight_short": short,
                "context_interpretation": full,
            }

        parsed = _try_parse(raw_text)
        if parsed:
            return parsed
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            parsed = _try_parse(fenced.group(1).strip())
            if parsed:
                return parsed
        brace = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if brace:
            parsed = _try_parse(brace.group(0).strip())
            if parsed:
                return parsed
        return {}

    def _fallback_context_progress(
        self, state: RunState, context_bundle: dict[str, Any] | None
    ) -> str:
        full, _ = self._fallback_context_progress_fields(state, context_bundle)
        return full

    def _fallback_context_progress_fields(
        self,
        state: RunState,
        context_bundle: dict[str, Any] | None,
        raw_text: str | None = None,
    ) -> tuple[str, str]:
        bundle = context_bundle or {}
        comp = bundle.get("comparable_run") or {}
        fam = bundle.get("family_context") or {}
        match_type = str(comp.get("match_type") or "none")
        fam_name = str(fam.get("session_family") or "other")
        trend = state.trend.trend_label
        fitness = state.trend.fitness_signal
        fatigue = state.trend.fatigue_signal
        direction = state.recommendation.load_action

        metrics = comp.get("metrics") or {}
        status_words: list[str] = []
        for m in metrics.values():
            if isinstance(m, dict):
                st = str(m.get("status") or "").strip()
                if st and st != "unknown":
                    status_words.append(st)

        if any(s in {"better", "faster", "lower", "higher"} for s in status_words):
            signal = "progress"
        elif any(s in {"worse", "slower"} for s in status_words):
            signal = "regression"
        elif any(s == "stable" for s in status_words):
            signal = "consolidation"
        else:
            signal = "normal_variation"

        short = (
            f"This run compares to recent {fam_name} history via {match_type} with a {signal} signal. "
            f"Trend context is {trend} with fitness {fitness} and fatigue {fatigue}. "
            f"Practical implication: keep load action {direction} and follow the next-session guidance."
        )
        short = self._shorten_for_top_card(short)

        full = (
            f"Relative to comparable prior runs ({match_type}), this session most likely reads as {signal}. "
            f"The family context ({fam_name}) and recent labels suggest the observed difference is better interpreted as training context, not an isolated one-off. "
            f"Current trend is {trend}, with fitness signal {fitness} and fatigue signal {fatigue}, which frames how much weight to place on the comparison. "
            f"Practical implication: stay aligned with load action {direction} and execute the deterministic next-session guidance rather than forcing an abrupt change."
        )
        if raw_text:
            full = f"{full} The model output was replaced by deterministic fallback due to malformed JSON."
        # Ensure short is concise and full is fuller.
        if len(full.split()) < len(short.split()) + 10:
            full = f"{full} This is best treated as structured context evidence, not a standalone verdict."
        return full, short

    def _shorten_for_top_card(self, text: str) -> str:
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
            # implication
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

        # Ensure interpretation leads and implication ends.
        if len(chosen) == 1 and len(sentences) > 1:
            tail = _clean(sentences[len(sentences) - 1])
            if tail and tail.lower() != chosen[0].lower():
                chosen.append(tail)

        # Allow one additional support sentence if concise and non-duplicate.
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

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _debug(self, message: str) -> None:
        if self.debug:
            print(f"[LLM] {message}", flush=True)

    def _config_diagnostics(self, readiness: dict[str, Any]) -> dict[str, str]:
        issues = readiness.get("issues", [])
        first_issue = issues[0] if issues else (self.init_error or "LLM client is not initialized.")
        return {
            "stage": "init",
            "error_type": "config_error",
            "error_message": first_issue,
            "likely_cause": "Invalid or incomplete LLM configuration.",
            "suggested_fix": "Set AGENTICRUN_LLM_ENABLED=1, OPENAI_API_KEY, OPENAI_MODEL, and optional OPENAI_BASE_URL.",
        }

    def _classify_error(self, exc: Exception, stage: str) -> dict[str, str]:
        msg = str(exc).strip() or "Unknown LLM error."
        lower = msg.lower()
        error_type = "unexpected_error"
        likely_cause = "Unexpected runtime error while calling the LLM API."
        suggested_fix = "Check logs/traceback and verify API/client compatibility."

        if "timeout" in lower or "timed out" in lower:
            error_type = "timeout_error"
            likely_cause = "The request timed out before the model responded."
            suggested_fix = "Retry, check network quality, and consider a smaller/faster model."
        elif "connection" in lower or "dns" in lower or "network" in lower:
            error_type = "connection_error"
            likely_cause = "Network/connectivity issue while reaching the LLM endpoint."
            suggested_fix = "Verify internet access, proxy/firewall settings, and OPENAI_BASE_URL."
        elif "401" in lower or "authentication" in lower or "invalid api key" in lower or "unauthorized" in lower:
            error_type = "auth_error"
            likely_cause = "API key is invalid, expired, or not authorized."
            suggested_fix = "Set a valid OPENAI_API_KEY with model access."
        elif "429" in lower or "rate limit" in lower or "quota" in lower:
            error_type = "rate_limit_error"
            likely_cause = "Rate limit or account quota was exceeded."
            suggested_fix = "Wait/retry later or increase quota/limits."
        elif "model" in lower and ("not found" in lower or "unsupported" in lower or "does not exist" in lower):
            error_type = "model_error"
            likely_cause = "Configured model is unavailable or unsupported."
            suggested_fix = "Set OPENAI_MODEL to a supported model."
        elif "output_text" in lower or "response" in lower and "empty" in lower:
            error_type = "response_error"
            likely_cause = "Model/API response format was missing expected text output."
            suggested_fix = "Inspect raw response handling and SDK compatibility."

        return {
            "stage": stage,
            "error_type": error_type,
            "error_message": msg,
            "likely_cause": likely_cause,
            "suggested_fix": suggested_fix,
        }

    def self_test(self, live_test: bool = False) -> dict[str, Any]:
        readiness = self.readiness_report()
        live = {
            "success": None,
            "stage": None,
            "error_type": None,
            "error_message": None,
            "likely_cause": None,
            "suggested_fix": None,
        }
        if live_test:
            self._debug("self_test: live test requested")
            if not self._ensure_client():
                diag = self._config_diagnostics(readiness)
                live = {
                    "success": False,
                    "stage": "init",
                    "error_type": diag["error_type"],
                    "error_message": diag["error_message"],
                    "likely_cause": diag["likely_cause"],
                    "suggested_fix": diag["suggested_fix"],
                }
            else:
                try:
                    self._debug("self_test: starting live API call (timeout=8s)")
                    self.client.responses.create(model=self.model, input="Reply with: OK", timeout=8.0)
                    live = {
                        "success": True,
                        "stage": "request_send",
                        "error_type": None,
                        "error_message": None,
                        "likely_cause": None,
                        "suggested_fix": None,
                    }
                except Exception as exc:
                    diag = self._classify_error(exc, stage="request_send")
                    live = {
                        "success": False,
                        "stage": diag["stage"],
                        "error_type": diag["error_type"],
                        "error_message": diag["error_message"],
                        "likely_cause": diag["likely_cause"],
                        "suggested_fix": diag["suggested_fix"],
                    }
        return {
            "ok": readiness.get("ready", False) and (live["success"] is not False),
            "provider": self.provider,
            "model": self.model or None,
            "api_key_present": bool(self.api_key),
            "used_llm": False,
            "readiness": readiness,
            "live_test": live,
            "note": "Readiness check for configured LLM path.",
        }