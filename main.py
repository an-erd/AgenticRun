from __future__ import annotations

import os
import sys
import argparse
import json
import tempfile
import time
import zipfile
from datetime import date, datetime, timezone
from shutil import copyfileobj
import csv
import re
from pathlib import Path
from typing import Any

from agenticrun.core.models import RunRecord
from agenticrun.utils.parsing import clean_header, parse_float

DEBUG = os.getenv("AGENTICRUN_DEBUG", "").lower() in {"1", "true", "yes", "on"}
SUPPORTED_IMPORT_EXTENSIONS = {".csv", ".fit", ".zip"}  # TODO: .fit parsing and .zip extraction are handled downstream

def dprint(*args, **kwargs):
    if DEBUG:
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)


def _collect_scan_targets(
    input_dir: Path, tmp_zip_root: Path
) -> tuple[list[Path], dict[str, Any]]:
    """Recursive discovery: .csv / .fit anywhere under input_dir; .zip members like ingest."""
    all_files = [p for p in input_dir.rglob("*") if p.is_file()]
    total_walk = len(all_files)
    targets: list[Path] = []
    zip_no_fit: list[str] = []
    zip_read_errors: list[str] = []
    for p in sorted(all_files, key=lambda x: str(x).lower()):
        ext = p.suffix.lower()
        if ext in (".csv", ".fit"):
            targets.append(p)
            continue
        if ext != ".zip":
            continue
        try:
            with zipfile.ZipFile(p, "r") as zf:
                n_fit = 0
                for member in zf.infolist():
                    member_path = Path(member.filename)
                    if member.is_dir() or member_path.suffix.lower() != ".fit":
                        continue
                    safe_name = f"{p.stem}__{member_path.name}"
                    extracted_path = tmp_zip_root / safe_name
                    with zf.open(member, "r") as src, open(extracted_path, "wb") as dst:
                        copyfileobj(src, dst)
                    targets.append(extracted_path)
                    n_fit += 1
            if n_fit == 0:
                zip_no_fit.append(str(p))
        except Exception as exc:
            zip_read_errors.append(f"{p}: {exc}")
    meta = {
        "total_files_walked": total_walk,
        "import_candidate_count": len(targets),
        "zip_without_fit": zip_no_fit,
        "zip_read_errors": zip_read_errors,
    }
    targets = sorted(targets, key=lambda x: str(x).lower())
    return targets, meta


BULK_CHECKPOINT_FILENAME = "bulk_import_checkpoint.json"
BULK_CHECKPOINT_VERSION = 1


def _bulk_checkpoint_path(out_dir: str) -> Path:
    return Path(out_dir) / BULK_CHECKPOINT_FILENAME


def _bulk_clear_normalized_artifacts(out_dir: str) -> None:
    root = Path(out_dir)
    for name in (
        "runs_normalized.csv",
        "runs_normalized.xlsx",
        "runs_normalized.json",
        "summary.md",
    ):
        p = root / name
        if p.is_file():
            try:
                p.unlink()
            except OSError:
                pass


def _bulk_write_checkpoint(
    out_dir: str,
    *,
    input_resolved: str,
    db_resolved: str,
    chunk_size: int,
    use_llm: bool,
    date_from: str | None,
    n_total: int,
    n_chunks: int,
    last_completed_chunk_index: int,
    files_processed: int,
    cum_stats: dict[str, Any],
) -> None:
    out_resolved = str(Path(out_dir).resolve())
    payload = {
        "schema_version": BULK_CHECKPOINT_VERSION,
        "input_resolved": input_resolved,
        "db_resolved": db_resolved,
        "out_resolved": out_resolved,
        "chunk_size": chunk_size,
        "use_llm": use_llm,
        "date_from": date_from,
        "n_total": n_total,
        "n_chunks": n_chunks,
        "last_completed_chunk_index": last_completed_chunk_index,
        "files_processed": files_processed,
        "cum_stats": cum_stats,
    }
    path = _bulk_checkpoint_path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _bulk_read_checkpoint(out_dir: str) -> dict[str, Any] | None:
    path = _bulk_checkpoint_path(out_dir)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _bulk_validate_checkpoint_for_resume(
    cp: dict[str, Any],
    *,
    input_resolved: str,
    db_resolved: str,
    out_resolved: str,
    chunk_size: int,
    use_llm: bool,
    date_from: str | None,
    n_total: int,
) -> str | None:
    if int(cp.get("schema_version", 0)) != BULK_CHECKPOINT_VERSION:
        return "checkpoint schema_version mismatch"
    if cp.get("input_resolved") != input_resolved:
        return "checkpoint input path mismatch"
    if cp.get("db_resolved") != db_resolved:
        return "checkpoint db path mismatch"
    if cp.get("out_resolved") != out_resolved:
        return "checkpoint out path mismatch"
    if int(cp.get("chunk_size", -1)) != chunk_size:
        return "checkpoint chunk_size mismatch"
    if bool(cp.get("use_llm")) != bool(use_llm):
        return "checkpoint use_llm mismatch"
    cp_date_from = cp.get("date_from")
    if cp_date_from is not None:
        cp_date_from = str(cp_date_from)
    run_date_from = date_from if date_from is not None else None
    if cp_date_from != run_date_from:
        return "checkpoint date_from mismatch"
    if int(cp.get("n_total", -1)) != n_total:
        return "checkpoint n_total mismatch (input tree changed?)"
    return None


BULK_IMPORT_SUMMARY_JSON = "bulk_import_last_summary.json"
BULK_IMPORT_SUMMARY_MD = "bulk_import_last_summary.md"
BULK_IMPORT_SUMMARY_SCHEMA = 1


def _write_bulk_import_last_summary(
    out_dir: str,
    *,
    input_resolved: str,
    db_resolved: str,
    out_resolved: str,
    chunk_size: int,
    resumed: bool,
    date_from: str | None,
    total_import_candidates: int,
    elapsed_sec: float,
    cum_stats: dict[str, Any],
) -> None:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    completed_at = datetime.now(timezone.utc).replace(microsecond=0)
    completed_iso = completed_at.isoformat().replace("+00:00", "Z")

    payload: dict[str, Any] = {
        "schema_version": BULK_IMPORT_SUMMARY_SCHEMA,
        "completed_at_utc": completed_iso,
        "input": input_resolved,
        "db": db_resolved,
        "out": out_resolved,
        "chunk_size": chunk_size,
        "resumed": resumed,
        "date_from": date_from,
        "total_import_candidates": total_import_candidates,
        "new_analyzed": int(cum_stats.get("new_analyzed") or 0),
        "duplicate_cached": int(cum_stats.get("duplicate_cached") or 0),
        "incomplete_or_unsupported_fit": int(
            cum_stats.get("incomplete_or_unsupported_fit") or 0
        ),
        "errors": int(cum_stats.get("errors") or 0),
        "dup_cache_miss": int(cum_stats.get("skipped_cache_miss") or 0),
        "skipped_non_running": int(cum_stats.get("skipped_non_running") or 0),
        "skipped_by_date": int(cum_stats.get("skipped_by_date") or 0),
        "skipped_date_unknown": int(cum_stats.get("skipped_date_unknown") or 0),
        "elapsed_sec": round(float(elapsed_sec), 3),
    }

    (root / BULK_IMPORT_SUMMARY_JSON).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    md = "\n".join(
        [
            "# Bulk import summary",
            "",
            f"- completed_at_utc: {completed_iso}",
            f"- input: `{input_resolved}`",
            f"- db: `{db_resolved}`",
            f"- out: `{out_resolved}`",
            f"- chunk_size: {chunk_size}",
            f"- resumed: {'yes' if resumed else 'no'}",
            f"- date_from: {date_from or '-'}",
            f"- total_import_candidates: {total_import_candidates}",
            f"- new_analyzed: {payload['new_analyzed']}",
            f"- duplicate_cached: {payload['duplicate_cached']}",
            f"- incomplete_or_unsupported_fit: {payload['incomplete_or_unsupported_fit']}",
            f"- errors: {payload['errors']}",
            f"- dup_cache_miss: {payload['dup_cache_miss']}",
            f"- skipped_non_running: {payload['skipped_non_running']}",
            f"- skipped_by_date: {payload['skipped_by_date']}",
            f"- skipped_date_unknown: {payload['skipped_date_unknown']}",
            f"- elapsed_sec: {payload['elapsed_sec']}",
            "",
        ]
    )
    (root / BULK_IMPORT_SUMMARY_MD).write_text(md, encoding="utf-8")


def _scan_real_activity_candidate(run: RunRecord) -> tuple[bool, str]:
    """Classify import readiness from parsed FIT metadata (aligned with show-fit-meta)."""
    if run.source_type != "garmin_fit":
        return True, "tabular import (non-FIT)"
    raw = run.raw_summary or {}
    sess: dict[str, Any] = dict(raw.get("session") or {})
    metrics: dict[str, Any] = dict(raw.get("fit_session_metrics") or {})
    rc = int(metrics.get("record_count") or 0)
    warn = str(metrics.get("fit_parse_warnings") or "")
    sport = str(metrics.get("sport") or "").strip().lower().replace(" ", "_")
    sub_sport = str(metrics.get("sub_sport") or "").strip().lower().replace(" ", "_")
    combo = f"{sport}_{sub_sport}".strip("_")
    if combo:
        if "cycl" in combo or "bike" in combo:
            return False, f"non-running FIT activity (sport={sport or '-'} sub_sport={sub_sport or '-'})"
        if "run" not in combo:
            return False, f"non-running FIT activity (sport={sport or '-'} sub_sport={sub_sport or '-'})"
    if not sess:
        return False, "incomplete / unsupported (no session message)"
    if rc == 0:
        return False, "not a normal activity stream (no FIT record points)"
    if warn:
        return False, "partial / thin data (" + warn.replace(";", ", ") + ")"
    return True, "likely a real activity"


def _parse_run_date_to_date(value: object) -> date | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if len(s) >= 10:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d").date()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _parse_date_from_arg(date_from: str | None) -> date | None:
    if date_from is None:
        return None
    return datetime.strptime(date_from, "%Y-%m-%d").date()


def cmd_backfill_ai_summaries(
    *,
    db_path: str,
    latest: int,
    use_llm: bool,
    force: bool,
) -> dict[str, int]:
    """Generate and store AI summaries for the latest N runs using existing LLM logic."""
    import sqlite3

    from agenticrun.agents.recommendation_agent import RecommendationAgent

    from agenticrun.services.db import (
        LLM_CONTEXT_METADATA_UNAVAILABLE,
        build_llm_context_progress_bundle,
        build_llm_prompt_deterministic_bundle,
        connect,
        load_cached_run_state_from_db,
        load_history,
        persistence_audit_for_run,
    )
    from agenticrun.services.llm import LLMService

    n = max(0, int(latest))
    db = Path(db_path)
    if not db.is_file():
        print(f"[backfill] No database file at {db.resolve()}", flush=True)
        return {
            "selected": 0,
            "generated": 0,
            "skipped_existing": 0,
            "failed": 0,
        }

    conn = connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT run_id, run_date, llm_summary, llm_summary_short,
                   llm_context_progress, llm_context_progress_short, source_file
            FROM runs
            WHERE sport = 'running' OR sport IS NULL
            ORDER BY run_date DESC, run_id DESC
            LIMIT ?
            """,
            (n,),
        ).fetchall()
        selected = len(rows)
        print(f"[backfill] Selected {selected} run(s) from {db.resolve()} (requested latest={n})", flush=True)

        if selected == 0:
            return {
                "selected": 0,
                "generated": 0,
                "skipped_existing": 0,
                "failed": 0,
            }

        llm = LLMService()
        stats = {
            "selected": selected,
            "generated": 0,
            "skipped_existing": 0,
            "failed": 0,
        }

        for idx, row in enumerate(rows, start=1):
            rid = str(row["run_id"])
            existing_full = str(row["llm_summary"] or "").strip()
            existing_short = str(row["llm_summary_short"] or "").strip()
            existing_ctx_full = str(row["llm_context_progress"] or "").strip()
            existing_ctx_short = str(row["llm_context_progress_short"] or "").strip()
            if (
                existing_full
                and existing_short
                and existing_ctx_full
                and existing_ctx_short
                and not force
            ):
                print(
                    f"[backfill] [{idx}/{selected}] run_id={rid} – skipping (existing summary+context outputs)",
                    flush=True,
                )
                stats["skipped_existing"] += 1
                continue
            if existing_full and not existing_short and not force and not use_llm:
                # LLM disabled: keep deterministic fallback so the field is still populated.
                short_only = llm._shorten_for_top_card(existing_full)
                if short_only:
                    conn.execute(
                        "UPDATE runs SET llm_summary_short = ? WHERE run_id = ?",
                        (short_only, rid),
                    )
                    conn.commit()
                    print(
                        f"[backfill] [{idx}/{selected}] run_id={rid} – filled missing short summary only (fallback)",
                        flush=True,
                    )
                    stats["generated"] += 1
                continue

            print(f"[backfill] [{idx}/{selected}] run_id={rid} – generating summary...", flush=True)
            try:
                src_file = str(row["source_file"] or "").strip()
                incoming_source_path = src_file if src_file else f"{rid}.fit"
                state = load_cached_run_state_from_db(conn, rid, incoming_source_path)
                if state is None or state.run_record is None:
                    print(f"[backfill]   ! no cached RunState for run_id={rid}; skipping", flush=True)
                    stats["failed"] += 1
                    continue

                history = load_history(conn)
                state = RecommendationAgent().run(state, history)
                try:
                    state.llm_prompt_deterministic = build_llm_prompt_deterministic_bundle(
                        conn, state
                    )
                except Exception as exc:
                    print(
                        f"[backfill]   ! llm_prompt_deterministic bundle failed run_id={rid}: {exc}",
                        flush=True,
                    )
                    state.llm_prompt_deterministic = {
                        "llm_prompt_context_line": (
                            "llm_prompt_context: deterministic_run_takeaway=no "
                            "interval_insight=no family_history=no"
                        ),
                        "findings_text": "",
                        "llm_context_metadata": dict(LLM_CONTEXT_METADATA_UNAVAILABLE),
                        "prompt_grounding_audit": {
                            "structured_recommendation_signals_in_prompt": "no",
                            "family_history_block_in_prompt": "no",
                            "recommendation_candidates_in_prompt": "no",
                        },
                    }

                if not use_llm:
                    fb = llm._fallback(state)
                    ctx_fb_bundle = build_llm_context_progress_bundle(conn, state)
                    ctx_fb = llm._fallback_context_progress(state, ctx_fb_bundle)
                    _fb_sh = llm._shorten_for_top_card(fb)
                    trace = {
                        "status": "disabled_fallback",
                        "model": None,
                        "used_llm": False,
                        "prompt": llm.build_prompt(state),
                        "raw_response": None,
                        "final_summary": fb,
                        "short_summary": _fb_sh,
                        "what_next": llm._what_next_short_fallback(state, _fb_sh, fb),
                        "error": None,
                    }
                    context_trace = {
                        "status": "disabled_fallback",
                        "model": None,
                        "used_llm": False,
                        "prompt": llm.build_context_progress_prompt(state, ctx_fb_bundle),
                        "raw_response": None,
                        "context_interpretation": ctx_fb,
                        "context_insight_short": llm._shorten_for_top_card(ctx_fb),
                        "error": None,
                    }
                else:
                    trace = llm.summarize_with_trace(state)
                    ctx_bundle = build_llm_context_progress_bundle(conn, state)
                    context_trace = llm.summarize_context_progress_with_trace(
                        state, ctx_bundle
                    )

                state.llm_summary = trace.get("final_summary") or ""
                short_top = str(trace.get("short_summary") or "").strip()
                if short_top:
                    state.llm_summary_short = short_top
                wn = str(trace.get("what_next") or "").strip()
                if wn:
                    state.llm_what_next_short = wn
                state.llm_context_progress = str(
                    context_trace.get("context_interpretation") or ""
                ).strip()
                ctx_short = str(context_trace.get("context_insight_short") or "").strip()
                if ctx_short:
                    state.llm_context_progress_short = ctx_short

                write_llm_trace(
                    str(Path("out")),
                    state,
                    trace,
                    context_trace=context_trace,
                )
                conn.execute(
                    "UPDATE runs SET llm_summary = ?, llm_summary_short = ?, llm_what_next_short = ?, "
                    "llm_context_progress = ?, llm_context_progress_short = ?, "
                    "recommendation_signals = ?, recommendation_summary = ?, next_session = ?, "
                    "load_action = ?, warning_flag = ? "
                    "WHERE run_id = ?",
                    (
                        state.llm_summary,
                        state.llm_summary_short,
                        state.llm_what_next_short,
                        state.llm_context_progress,
                        state.llm_context_progress_short,
                        state.as_flat_dict().get("recommendation_signals") or "{}",
                        state.recommendation.recommendation_summary,
                        state.recommendation.next_session,
                        state.recommendation.load_action,
                        1 if state.recommendation.warning_flag else 0,
                        rid,
                    ),
                )
                conn.commit()

                audit = persistence_audit_for_run(conn, rid)
                llm_ok = audit.get("llm_summary_stored")
                print(
                    f"[backfill]   done run_id={rid} status={trace.get('status')} "
                    f"used_llm={trace.get('used_llm')} llm_summary_stored={'yes' if llm_ok else 'no'}",
                    flush=True,
                )
                stats["generated"] += 1
            except Exception as exc:
                print(f"[backfill]   ! failed run_id={rid}: {exc}", flush=True)
                stats["failed"] += 1

        print(
            "[backfill] Summary: "
            f"selected={stats['selected']} generated={stats['generated']} "
            f"skipped_existing={stats['skipped_existing']} failed={stats['failed']}",
            flush=True,
        )
        return stats
    finally:
        conn.close()


def scan_folder(input_dir: str, db_path: str, *, date_from: date | None = None) -> dict[str, Any]:
    """Preflight: parse-only walk; no DB writes, no LLM. Duplicate check matches ingest (FIT keys)."""
    from agenticrun.agents.import_agent import ImportAgent
    from agenticrun.core.models import RunState
    from agenticrun.services.db import (
        connect,
        load_cached_run_state_from_db,
        lookup_run_id_by_fit_activity_key,
    )

    input_path = Path(input_dir)
    if not input_path.is_dir():
        print(f"scan: not a directory: {input_dir}", flush=True)
        return {"error": "not_a_directory"}

    print(
        f"scan: input={input_path.resolve()} db={Path(db_path).resolve()} "
        f"date_from={date_from.isoformat() if date_from else '-'}",
        flush=True,
    )

    conn = connect(db_path)
    importer = ImportAgent()

    out: dict[str, Any] = {
        "input": str(input_path.resolve()),
        "db": str(Path(db_path).resolve()),
        "total_files_seen": 0,
        "import_candidates": 0,
        "parsable": 0,
        "real_activity_candidates": 0,
        "incomplete_or_unsupported_fit": 0,
        "parse_failed": 0,
        "fit_duplicate_in_db": 0,
        "fit_duplicate_cache_miss": 0,
        "likely_new": 0,
        "zip_errors": 0,
        "zips_without_fit": 0,
        "skipped_non_running": 0,
        "skipped_by_date": 0,
        "skipped_date_unknown": 0,
    }

    try:
        with tempfile.TemporaryDirectory(prefix="agenticrun_scan_zip_") as tmp_zip_dir:
            tmp_zip_path = Path(tmp_zip_dir)
            targets, meta = _collect_scan_targets(input_path, tmp_zip_path)
            out["total_files_seen"] = meta["total_files_walked"]
            out["import_candidates"] = meta["import_candidate_count"]
            out["zips_without_fit"] = len(meta["zip_without_fit"])
            out["zip_errors"] = len(meta["zip_read_errors"])

            if meta["zip_read_errors"]:
                print("scan: zip read issue(s) (first up to 3):", flush=True)
                for line in meta["zip_read_errors"][:3]:
                    print(f"  {line}", flush=True)

            parse_fail_examples: list[str] = []
            dup_examples: list[str] = []
            incomplete_fit_examples: list[str] = []

            for path in targets:
                state = RunState(source_path=str(path))
                state = importer.run(state)
                if state.status == "error":
                    out["parse_failed"] += 1
                    if len(parse_fail_examples) < 3:
                        parse_fail_examples.append(path.name)
                    continue

                out["parsable"] += 1
                rr = state.run_record
                if not rr:
                    continue
                if date_from is not None:
                    rr_date = _parse_run_date_to_date(rr.run_date)
                    if rr_date is None:
                        out["skipped_date_unknown"] += 1
                        continue
                    if rr_date < date_from:
                        out["skipped_by_date"] += 1
                        continue
                is_real, reason = _scan_real_activity_candidate(rr)
                if rr.source_type == "garmin_fit":
                    if is_real:
                        out["real_activity_candidates"] += 1
                    else:
                        if reason.startswith("non-running FIT activity"):
                            out["skipped_non_running"] += 1
                        out["incomplete_or_unsupported_fit"] += 1
                        if len(incomplete_fit_examples) < 3:
                            incomplete_fit_examples.append(path.name)
                else:
                    out["real_activity_candidates"] += 1

                if (
                    rr.source_type == "garmin_fit"
                    and rr.fit_activity_key
                ):
                    existing = lookup_run_id_by_fit_activity_key(
                        conn, rr.fit_activity_key
                    )
                    if existing:
                        cached = load_cached_run_state_from_db(
                            conn, existing, str(path)
                        )
                        if cached is None:
                            out["fit_duplicate_cache_miss"] += 1
                        else:
                            out["fit_duplicate_in_db"] += 1
                            if len(dup_examples) < 2:
                                dup_examples.append(path.name)
                    else:
                        if is_real:
                            out["likely_new"] += 1
                else:
                    if rr.source_type != "garmin_fit":
                        out["likely_new"] += 1
                    elif not rr.fit_activity_key and is_real:
                        out["likely_new"] += 1
    finally:
        conn.close()

    # Summary line (compact)
    pf = out["parse_failed"]
    zr = out["zip_errors"]
    zw = out["zips_without_fit"]
    rac = out["real_activity_candidates"]
    inc = out["incomplete_or_unsupported_fit"]
    summary = (
        f"scan_preflight: total_files={out['total_files_seen']} "
        f"candidates={out['import_candidates']} "
        f"real_activity_candidates={rac} "
        f"duplicate_fit={out['fit_duplicate_in_db']} "
        f"dup_cache_miss={out['fit_duplicate_cache_miss']} "
        f"likely_new={out['likely_new']} "
        f"incomplete_or_unsupported_fit={inc} "
        f"skipped_non_running={out['skipped_non_running']} "
        f"skipped_by_date={out['skipped_by_date']} "
        f"skipped_date_unknown={out['skipped_date_unknown']} "
        f"parse_failed={pf} zip_err={zr} zip_no_fit={zw}"
    )
    print(summary, flush=True)

    if parse_fail_examples:
        print(f"scan: parse_failed examples: {', '.join(parse_fail_examples)}", flush=True)
    if incomplete_fit_examples:
        print(
            f"scan: incomplete_or_unsupported_fit examples: {', '.join(incomplete_fit_examples)}",
            flush=True,
        )
    if dup_examples:
        print(f"scan: duplicate_fit examples: {', '.join(dup_examples)}", flush=True)

    return out


def _empty_ingest_summary(use_llm: bool = False) -> dict[str, Any]:
    return {
        "uploaded_files": 0,
        "new_analyzed": 0,
        "duplicate_cached": 0,
        "errors": 0,
        "skipped_cache_miss": 0,
        "skipped_non_running": 0,
        "skipped_by_date": 0,
        "skipped_date_unknown": 0,
        "incomplete_or_unsupported_fit": 0,
        "llm_api_calls": 0,
        "llm_reused_stored": 0,
        "use_llm_requested": use_llm,
    }


def _fmt_hist_field(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)

def write_llm_trace(
    out_dir: str,
    state: RunState,
    trace: dict,
    *,
    context_trace: dict | None = None,
) -> None:
    from agenticrun.services.db import LLM_CONTEXT_METADATA_UNAVAILABLE

    def _shorten_for_top_card(text: str) -> str:
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

    trace_dir = Path(out_dir) / "llm_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    run_id = state.run_record.run_id if state.run_record else "unknown_run"
    trace_path = trace_dir / f"{run_id}.json"

    det = getattr(state, "llm_prompt_deterministic", None) or {}
    meta = det.get("llm_context_metadata")
    if meta is None:
        meta = dict(LLM_CONTEXT_METADATA_UNAVAILABLE)

    if isinstance(trace, dict) and not str(trace.get("short_summary") or "").strip():
        fs = str(trace.get("final_summary") or "").strip()
        if fs:
            trace["short_summary"] = _shorten_for_top_card(fs)

    rec_sig = (
        state.recommendation.recommendation_signals
        if state.recommendation and state.recommendation.recommendation_signals
        else {}
    )
    prompt_audit = {}
    if isinstance(det, dict):
        pa = det.get("prompt_grounding_audit")
        if isinstance(pa, dict):
            prompt_audit = pa

    payload = {
        "run_id": run_id,
        "source_file": state.source_path,
        "run_date": state.run_record.run_date if state.run_record else None,
        "training_type": state.analysis.training_type if state.analysis else None,
        "trend_label": state.trend.trend_label if state.trend else None,
        "next_session": state.recommendation.next_session if state.recommendation else None,
        "recommendation_signals": rec_sig,
        "prompt_grounding_audit": prompt_audit,
        "llm_context_metadata": meta,
        "llm_context_progress_bundle": getattr(state, "llm_context_progress_bundle", None) or {},
        "trace": trace,
        "context_progress_trace": context_trace or {},
    }

    trace_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _ingest_process_paths(
    paths: list[Path],
    *,
    conn: Any,
    out_dir: str,
    use_llm: bool,
    importer: Any,
    analyzer: Any,
    trend_agent: Any,
    recommender: Any,
    llm: Any,
    date_from: date | None = None,
    skip_incomplete_fit: bool = False,
    ingest_quiet: bool = False,
) -> tuple[list[Any], dict[str, Any]]:
    """Run the per-file ingest pipeline used by ``ingest`` on an explicit path list."""
    from agenticrun.core.models import RunState
    from agenticrun.services.db import (
        aggregate_work_only_session_for_run,
        aggregate_work_recovery_segments_for_run,
        derive_work_block_label_for_run,
        derive_work_session_family_for_run,
        compare_interval_session_vs_prior,
        build_llm_context_progress_bundle,
        build_llm_prompt_deterministic_bundle,
        format_llm_context_applied_log_line,
        LLM_CONTEXT_METADATA_UNAVAILABLE,
        fetch_comparable_interval_sessions_by_fingerprint,
        fetch_work_recovery_segments_history,
        fetch_work_recovery_session_summaries,
        interval_structure_fingerprint_for_run,
        load_history,
        load_cached_run_state_from_db,
        lookup_run_id_by_fit_activity_key,
        persistence_audit_for_run,
        upsert_state,
    )
    from agenticrun.services.zone_resolution import apply_zone_resolution_to_state, zone_debug_line

    states: list[RunState] = []
    stats: dict[str, Any] = {
        "uploaded_files": len(paths),
        "new_analyzed": 0,
        "duplicate_cached": 0,
        "errors": 0,
        "skipped_cache_miss": 0,
        "skipped_non_running": 0,
        "skipped_by_date": 0,
        "skipped_date_unknown": 0,
        "incomplete_or_unsupported_fit": 0,
        "llm_api_calls": 0,
        "llm_context_api_calls": 0,
        "llm_reused_stored": 0,
        "use_llm_requested": use_llm,
    }

    for path in paths:
        dprint(f"\n--- START {path.name} ---")

        state = RunState(source_path=str(path))
        dprint("created RunState")

        state = importer.run(state)
        dprint(f"after import_agent: status={state.status}")

        if state.status == "error":
            states.append(state)
            stats["errors"] += 1
            if ingest_quiet:
                warn = "; ".join(state.warnings) if state.warnings else "unknown error"
                print(f"import_error: file={path.name} {warn}", flush=True)
            dprint("state had error, skipping remaining steps")
            continue

        if date_from is not None and state.run_record is not None:
            rr_date = _parse_run_date_to_date(state.run_record.run_date)
            if rr_date is None:
                stats["skipped_date_unknown"] += 1
                if ingest_quiet:
                    print(
                        f"date_filter_skip: file={path.name} reason=missing_or_invalid_run_date",
                        flush=True,
                    )
                continue
            if rr_date < date_from:
                stats["skipped_by_date"] += 1
                if ingest_quiet:
                    print(
                        "date_filter_skip: "
                        f"file={path.name} run_date={rr_date.isoformat()} "
                        f"date_from={date_from.isoformat()}",
                        flush=True,
                    )
                continue

        if state.run_record and state.run_record.source_type == "garmin_fit":
            _is_real_fit, _fit_reason = _scan_real_activity_candidate(state.run_record)
            if not _is_real_fit and _fit_reason.startswith("non-running FIT activity"):
                stats["skipped_non_running"] += 1
                if ingest_quiet:
                    print(
                        f"skip_non_running_fit: file={path.name} reason={_fit_reason}",
                        flush=True,
                    )
                continue

        if (
            skip_incomplete_fit
            and state.run_record
            and state.run_record.source_type == "garmin_fit"
        ):
            is_real, _ = _scan_real_activity_candidate(state.run_record)
            if not is_real:
                stats["incomplete_or_unsupported_fit"] += 1
                dprint(f"skip incomplete/unsupported FIT: {path.name}")
                continue

        if (
            state.run_record
            and state.run_record.source_type == "garmin_fit"
            and state.run_record.fit_activity_key
        ):
            existing_rid = lookup_run_id_by_fit_activity_key(
                conn, state.run_record.fit_activity_key
            )
            if existing_rid:
                print(
                    "duplicate_activity_skip: "
                    f"activity_key={state.run_record.fit_activity_key!s} "
                    f"existing_run_id={existing_rid!s} "
                    f"incoming_file={path.name!s}",
                    flush=True,
                )
                cached_state = load_cached_run_state_from_db(
                    conn, existing_rid, str(path)
                )
                if cached_state is None:
                    stats["skipped_cache_miss"] += 1
                    if ingest_quiet:
                        print(
                            "duplicate_activity_cache_miss: "
                            f"incoming_file={path.name} existing_run_id={existing_rid}",
                            flush=True,
                        )
                    else:
                        dprint(
                            "duplicate_activity_cache_miss: "
                            f"incoming_file={path.name!s} existing_run_id={existing_rid!s}"
                        )
                    continue
                cached_state.status = "duplicate_cached"
                cached_state.cached_from_run_id = existing_rid
                llm_reused = bool((cached_state.llm_summary or "").strip())
                print(
                    "duplicate_activity_cache_hit: "
                    f"incoming_file={path.name!s} existing_run_id={existing_rid!s} "
                    f"reused_llm_from_store={'yes' if llm_reused else 'no'}",
                    flush=True,
                )
                states.append(cached_state)
                stats["duplicate_cached"] += 1
                if llm_reused:
                    stats["llm_reused_stored"] += 1
                dprint(
                    f"Processed {path.name}: duplicate_cached / "
                    f"{cached_state.analysis.training_type} / {cached_state.trend.trend_label}",
                    flush=True,
                )
                continue

        zr = apply_zone_resolution_to_state(conn, state)
        dprint(zone_debug_line(zr))

        state = analyzer.run(state)

        dprint("after session_analysis_agent")
        if state.analysis.classification_trace:
            dprint(f"classification_trace: {state.analysis.classification_trace}")

        history = load_history(conn)
        dprint(f"after load_history: {len(history)} history rows")

        state = trend_agent.run(state, history)
        dprint("after trend_agent")

        state = recommender.run(state, history)
        dprint("after recommendation_agent")

        rid_ingest = ""
        if state.run_record and state.run_record.run_id:
            rid_ingest = str(state.run_record.run_id).strip()

        upsert_state(conn, state)
        pre_audit = (
            persistence_audit_for_run(conn, rid_ingest) if rid_ingest else None
        )
        dprint("after upsert_state (pre-LLM persist for segment-based prompt context)")

        state.llm_prompt_deterministic = {}
        state.llm_context_progress_bundle = {}
        try:
            state.llm_prompt_deterministic = build_llm_prompt_deterministic_bundle(
                conn, state
            )
        except Exception as exc:
            dprint(f"llm_prompt_deterministic bundle skipped: {exc}")
            state.llm_prompt_deterministic = {
                "llm_prompt_context_line": (
                    "llm_prompt_context: deterministic_run_takeaway=no "
                    "interval_insight=no family_history=no"
                ),
                "findings_text": "",
                "llm_context_metadata": dict(LLM_CONTEXT_METADATA_UNAVAILABLE),
                "prompt_grounding_audit": {
                    "structured_recommendation_signals_in_prompt": "no",
                    "family_history_block_in_prompt": "no",
                    "recommendation_candidates_in_prompt": "no",
                },
            }
        try:
            state.llm_context_progress_bundle = build_llm_context_progress_bundle(
                conn, state
            )
        except Exception as exc:
            dprint(f"llm_context_progress bundle skipped: {exc}")
            state.llm_context_progress_bundle = {"available": False, "reason": "bundle_build_failed"}

        if use_llm:
            trace = llm.summarize_with_trace(state)
            context_trace = llm.summarize_context_progress_with_trace(
                state, state.llm_context_progress_bundle
            )
            dprint(
                llm.format_ingest_runtime_status_line(
                    enabled_for_run=use_llm,
                    used_for_run=bool(trace.get("used_llm")),
                )
            )
            dprint(
                "after llm summarize_with_trace: "
                f"status={trace.get('status')} used_llm={trace.get('used_llm')} "
                f"error_type={trace.get('error_type')} "
                f"error_message={trace.get('error_message')}"
            )
            dprint(
                "after llm context summarize_with_trace: "
                f"status={context_trace.get('status')} used_llm={context_trace.get('used_llm')} "
                f"error_type={context_trace.get('error_type')} "
                f"error_message={context_trace.get('error_message')}"
            )
        else:
            fallback_summary = llm._fallback(state)
            fallback_context = llm._fallback_context_progress(
                state, state.llm_context_progress_bundle
            )
            _fb_short = llm._shorten_for_top_card(fallback_summary)
            trace = {
                "status": "disabled_fallback",
                "model": None,
                "used_llm": False,
                "prompt": llm.build_prompt(state),
                "raw_response": None,
                "final_summary": fallback_summary,
                "short_summary": _fb_short,
                "what_next": llm._what_next_short_fallback(state, _fb_short, fallback_summary),
                "error": None,
            }
            context_trace = {
                "status": "disabled_fallback",
                "model": None,
                "used_llm": False,
                "prompt": llm.build_context_progress_prompt(
                    state, state.llm_context_progress_bundle
                ),
                "raw_response": None,
                "context_interpretation": fallback_context,
                "context_insight_short": llm._shorten_for_top_card(fallback_context),
                "error": None,
            }
            dprint(
                llm.format_ingest_runtime_status_line(
                    enabled_for_run=use_llm,
                    used_for_run=False,
                )
            )
            dprint("after llm fallback (enabled_for_run=no; no API call)")

        state.llm_summary = trace["final_summary"]
        # Persist a dedicated short AI summary for the top card when available.
        short_top = str(trace.get("short_summary") or "").strip()
        if short_top:
            state.llm_summary_short = short_top
        wn = str(trace.get("what_next") or "").strip()
        if wn:
            state.llm_what_next_short = wn
        state.llm_context_progress = str(
            context_trace.get("context_interpretation") or ""
        ).strip()
        ctx_short = str(context_trace.get("context_insight_short") or "").strip()
        if ctx_short:
            state.llm_context_progress_short = ctx_short
        write_llm_trace(out_dir, state, trace, context_trace=context_trace)
        _meta = (state.llm_prompt_deterministic or {}).get("llm_context_metadata")
        if not ingest_quiet:
            print(format_llm_context_applied_log_line(_meta), flush=True)
        dprint("after write_llm_trace")

        upsert_state(conn, state)
        dprint("after upsert_state")

        post_audit = (
            persistence_audit_for_run(conn, rid_ingest) if rid_ingest else None
        )
        if rid_ingest and pre_audit is not None and post_audit is not None:
            seg_stable = (
                "yes"
                if pre_audit["segments_rowcount"] == post_audit["segments_rowcount"]
                else "no"
            )
            llm_st = "yes" if post_audit["llm_summary_stored"] else "no"
            llm_ctx_st = (
                "yes" if post_audit.get("llm_context_progress_stored") else "no"
            )
            if not ingest_quiet:
                print(
                    "persistence_flow: pre_llm_upsert=yes post_llm_upsert=yes "
                    f"segments_rowcount={post_audit['segments_rowcount']} "
                    f"run_rowcount={post_audit['run_rowcount']} "
                    f"llm_summary_stored={llm_st} "
                    f"llm_context_progress_stored={llm_ctx_st} "
                    f"segments_stable_across_steps={seg_stable}",
                    flush=True,
                )
            if DEBUG and post_audit["run_rowcount"] != 1:
                dprint(
                    "persistence_audit: expected run_rowcount=1 for this run_id, "
                    f"got {post_audit['run_rowcount']}"
                )

        stats["new_analyzed"] += 1
        if use_llm and bool(trace.get("used_llm")):
            stats["llm_api_calls"] += 1
        if use_llm and bool(context_trace.get("used_llm")):
            stats["llm_context_api_calls"] += 1

        if DEBUG:
            rid_agg = state.run_record.run_id if state.run_record else None
            if rid_agg:
                agg = aggregate_work_recovery_segments_for_run(conn, rid_agg)
                if agg is not None:
                    dprint(
                        "run_work_recovery_session_agg (from persisted run_segments): "
                        f"run_date={agg.get('run_date')!s} run_id={agg.get('run_id')!s} | "
                        "work: "
                        f"n={agg.get('work_count')} "
                        f"mean_dur_s={_fmt_hist_field(agg.get('work_mean_duration_sec'))} "
                        f"mean_pwr_w={_fmt_hist_field(agg.get('work_mean_power_w'))} "
                        f"mean_hr_avg={_fmt_hist_field(agg.get('work_mean_hr_avg'))} "
                        f"mean_pace_sec_per_km={_fmt_hist_field(agg.get('work_mean_pace_sec_per_km'))} | "
                        "recovery: "
                        f"n={agg.get('recovery_count')} "
                        f"mean_dur_s={_fmt_hist_field(agg.get('recovery_mean_duration_sec'))} "
                        f"mean_pwr_w={_fmt_hist_field(agg.get('recovery_mean_power_w'))} "
                        f"mean_hr_avg={_fmt_hist_field(agg.get('recovery_mean_hr_avg'))} "
                        f"mean_pace_sec_per_km={_fmt_hist_field(agg.get('recovery_mean_pace_sec_per_km'))}"
                    )

                wo = aggregate_work_only_session_for_run(conn, rid_agg)
                if wo is not None:
                    dprint(
                        "work_only_session_summary: "
                        f"run_date={wo.get('run_date')!s} run_id={wo.get('run_id')!s} | "
                        f"blocks={wo.get('work_block_count')} "
                        f"t_total_s={_fmt_hist_field(wo.get('work_total_time_sec'))} "
                        f"dist_total_m={_fmt_hist_field(wo.get('work_total_distance_m'))} "
                        f"pwr_mean_w={_fmt_hist_field(wo.get('work_mean_power_w'))} "
                        f"hr_mean={_fmt_hist_field(wo.get('work_mean_hr_avg'))} "
                        f"pace_mean_s_km={_fmt_hist_field(wo.get('work_mean_pace_sec_per_km'))} "
                        f"w_per_hr={_fmt_hist_field(wo.get('work_w_per_hr'))}"
                    )

                wbl = derive_work_block_label_for_run(conn, rid_agg)
                if wbl is not None:
                    dprint(
                        "work_block_label: "
                        f"run_id={wbl.get('run_id')!s} "
                        f"label={wbl.get('work_block_label')!s}"
                    )

                hint_tt = (
                    state.analysis.training_type
                    if state.analysis is not None
                    else None
                )
                wsf = derive_work_session_family_for_run(
                    conn, rid_agg, training_type_hint=hint_tt
                )
                if wsf is not None:
                    dprint(
                        "work_session_family: "
                        f"run_id={wsf.get('run_id')!s} "
                        f"family={wsf.get('work_session_family')!s}"
                    )

                ist = interval_structure_fingerprint_for_run(conn, rid_agg)
                if ist is not None:
                    dprint(
                        "interval_structure_fingerprint: "
                        f"work_n={ist.get('work_count')} rec_n={ist.get('recovery_count')} "
                        f"work_distances_m={ist.get('work_distances_m')} "
                        f"rec_durations_s={ist.get('recovery_durations_s')} "
                        f"fingerprint={ist.get('fingerprint')}"
                    )

                cmp_iv = fetch_comparable_interval_sessions_by_fingerprint(
                    conn, rid_agg, newest_first=False, limit=8
                )
                fp_c = cmp_iv.get("fingerprint")
                dprint(
                    "comparable_interval_sessions_by_fingerprint: "
                    f"fingerprint={fp_c!s} matches={cmp_iv.get('match_count')}"
                )
                for m in cmp_iv.get("matches") or []:
                    dprint(
                        "  "
                        f"date={m.get('run_date')!s} run_id={m.get('run_id')!s} | "
                        f"work n={m.get('work_count')} "
                        f"dur_s={_fmt_hist_field(m.get('work_mean_duration_sec'))} "
                        f"pwr={_fmt_hist_field(m.get('work_mean_power_w'))} "
                        f"hr={_fmt_hist_field(m.get('work_mean_hr_avg'))} "
                        f"pace={_fmt_hist_field(m.get('work_mean_pace_sec_per_km'))} | "
                        f"rec n={m.get('recovery_count')} "
                        f"dur_s={_fmt_hist_field(m.get('recovery_mean_duration_sec'))} "
                        f"pwr={_fmt_hist_field(m.get('recovery_mean_power_w'))} "
                        f"hr={_fmt_hist_field(m.get('recovery_mean_hr_avg'))} "
                        f"pace={_fmt_hist_field(m.get('recovery_mean_pace_sec_per_km'))}"
                    )

                ivd = compare_interval_session_vs_prior(conn, rid_agg)
                if ivd.get("insufficient_history"):
                    dprint(
                        "comparable_interval_delta: "
                        f"fingerprint={ivd.get('fingerprint')!s} "
                        f"insufficient_history=1 reason={ivd.get('reason')!s} "
                        f"prior_count={ivd.get('prior_count')}"
                    )
                else:
                    mm = ivd.get("metrics") or {}
                    wp = mm.get("work_mean_pace_sec_per_km") or {}
                    wq = mm.get("work_mean_power_w") or {}
                    rh = mm.get("recovery_mean_hr_avg") or {}
                    rp = mm.get("recovery_mean_power_w") or {}
                    dprint(
                        "comparable_interval_delta: "
                        f"fingerprint={ivd.get('fingerprint')!s} "
                        f"baseline={ivd.get('baseline_mode')!s} "
                        f"baseline_run={ivd.get('baseline_run_id')!s}@"
                        f"{ivd.get('baseline_run_date')!s} "
                        f"matches_used={ivd.get('matches_used')} "
                        f"work_pace status={wp.get('status')!s} "
                        f"delta_s_per_km={_fmt_hist_field(wp.get('delta'))} | "
                        f"work_pwr status={wq.get('status')!s} "
                        f"delta_w={_fmt_hist_field(wq.get('delta'))} | "
                        f"rec_hr status={rh.get('status')!s} "
                        f"delta_bpm={_fmt_hist_field(rh.get('delta'))} | "
                        f"rec_pwr status={rp.get('status')!s} "
                        f"delta_w={_fmt_hist_field(rp.get('delta'))}"
                    )

            sums = fetch_work_recovery_session_summaries(
                conn, newest_first=True, limit=10
            )
            dprint(
                f"work_recovery_session_summaries_preview (latest {len(sums)} run(s)):"
            )
            for s in sums:
                dprint(
                    "  "
                    f"date={s.get('run_date')!s} run_id={s.get('run_id')!s} | "
                    f"work n={s.get('work_count')} "
                    f"dur_s={_fmt_hist_field(s.get('work_mean_duration_sec'))} "
                    f"pwr={_fmt_hist_field(s.get('work_mean_power_w'))} "
                    f"hr={_fmt_hist_field(s.get('work_mean_hr_avg'))} "
                    f"pace={_fmt_hist_field(s.get('work_mean_pace_sec_per_km'))} | "
                    f"rec n={s.get('recovery_count')} "
                    f"dur_s={_fmt_hist_field(s.get('recovery_mean_duration_sec'))} "
                    f"pwr={_fmt_hist_field(s.get('recovery_mean_power_w'))} "
                    f"hr={_fmt_hist_field(s.get('recovery_mean_hr_avg'))} "
                    f"pace={_fmt_hist_field(s.get('recovery_mean_pace_sec_per_km'))}"
                )

            hist = fetch_work_recovery_segments_history(conn, limit=10, newest_first=True)
            dprint(
                f"run_segments_history_preview (latest {len(hist)} work/recovery row(s) in DB):"
            )
            for h in hist:
                dprint(
                    "  "
                    f"date={h.get('run_date')!s} run_id={h.get('run_id')!s} "
                    f"seg_idx={_fmt_hist_field(h.get('segment_index'))} "
                    f"type={h.get('segment_type_mapped')!s} "
                    f"duration_sec={_fmt_hist_field(h.get('duration_sec'))} "
                    f"distance_m={_fmt_hist_field(h.get('distance_m'))} "
                    f"hr_avg_max={_fmt_hist_field(h.get('avg_hr'))}/"
                    f"{_fmt_hist_field(h.get('max_hr'))} "
                    f"avg_power_w={_fmt_hist_field(h.get('avg_power'))} "
                    f"avg_speed_m_s={_fmt_hist_field(h.get('avg_speed_m_s'))} "
                    f"pace_sec_per_km={_fmt_hist_field(h.get('avg_pace_sec_per_km'))}"
                )

        state.status = "completed"
        states.append(state)

        dprint(
            f"Processed {path.name}: "
            f"{state.analysis.training_type} / {state.trend.trend_label}",
            flush=True,
        )

    return states, stats


def _ingest_finalize_outputs(
    conn: Any,
    states: list[Any],
    stats: dict[str, Any],
    *,
    db_path: str,
    out_dir: str,
    use_llm: bool,
    writer: Any,
    skip_write_batch: bool = False,
) -> dict[str, Any]:
    """DEBUG tail (optional), normalized exports, and ``last_import_summary.json``."""
    from agenticrun.services.db import (
        fetch_work_family_session_history,
        compare_threshold_session_family_latest_vs_prior,
        compare_vo2max_family_latest_vs_prior,
        derive_easy_aerobic_efficiency_trend,
        build_interval_family_insight_summary,
        format_interval_family_insight_summary,
    )

    if DEBUG:
        for fam in ("threshold_session", "vo2max_session"):
            hist = fetch_work_family_session_history(conn, fam)
            dprint(f"work_family_history {fam}: n={len(hist)}")
            for row in hist:
                dprint(
                    "  "
                    f"date={row.get('run_date')!s} run_id={row.get('run_id')!s} "
                    f"label={row.get('work_block_label')!s} "
                    f"t_s={_fmt_hist_field(row.get('work_total_time_sec'))} "
                    f"pace={_fmt_hist_field(row.get('work_mean_pace_sec_per_km'))} "
                    f"pwr={_fmt_hist_field(row.get('work_mean_power_w'))} "
                    f"hr={_fmt_hist_field(row.get('work_mean_hr_avg'))} "
                    f"w/hr={_fmt_hist_field(row.get('work_w_per_hr'))}"
                )
            if fam == "threshold_session":
                th = compare_threshold_session_family_latest_vs_prior(conn)
                if th.get("insufficient_history"):
                    dprint(
                        "work_family_history threshold_vs_prior: insufficient_history "
                        f"reason={th.get('reason')!s}"
                    )
                else:
                    mm = th.get("metrics") or {}
                    lb = th.get("work_block_label") or {}
                    mp = mm.get("work_mean_pace_sec_per_km") or {}
                    mw = mm.get("work_mean_power_w") or {}
                    mh = mm.get("work_mean_hr_avg") or {}
                    mr = mm.get("work_w_per_hr") or {}
                    dprint(
                        "work_family_history threshold_vs_prior: "
                        f"cur_date={(th.get('current') or {}).get('run_date')!s} "
                        f"prev_date={(th.get('baseline') or {}).get('run_date')!s} "
                        f"label_cmp={lb.get('status')!s}"
                    )
                    dprint(
                        "  "
                        f"pace d={_fmt_hist_field(mp.get('delta'))} {mp.get('status')!s} | "
                        f"pwr d={_fmt_hist_field(mw.get('delta'))} {mw.get('status')!s} | "
                        f"hr d={_fmt_hist_field(mh.get('delta'))} {mh.get('status')!s} | "
                        f"w/hr d={_fmt_hist_field(mr.get('delta'))} {mr.get('status')!s}"
                    )
            if fam == "vo2max_session":
                v2 = compare_vo2max_family_latest_vs_prior(conn)
                if v2.get("insufficient_history"):
                    dprint(
                        "work_family_history vo2max_vs_prior: insufficient_history "
                        f"reason={v2.get('reason')!s}"
                    )
                else:
                    mm = v2.get("metrics") or {}
                    lb = v2.get("work_block_label") or {}
                    mp = mm.get("work_mean_pace_sec_per_km") or {}
                    mw = mm.get("work_mean_power_w") or {}
                    mh = mm.get("work_mean_hr_avg") or {}
                    mr = mm.get("work_w_per_hr") or {}
                    dprint(
                        "work_family_history vo2max_vs_prior: "
                        f"cur_date={(v2.get('current') or {}).get('run_date')!s} "
                        f"prev_date={(v2.get('baseline') or {}).get('run_date')!s} "
                        f"label_cmp={lb.get('status')!s}"
                    )
                    dprint(
                        "  "
                        f"pace d={_fmt_hist_field(mp.get('delta'))} {mp.get('status')!s} | "
                        f"pwr d={_fmt_hist_field(mw.get('delta'))} {mw.get('status')!s} | "
                        f"hr d={_fmt_hist_field(mh.get('delta'))} {mh.get('status')!s} | "
                        f"w/hr d={_fmt_hist_field(mr.get('delta'))} {mr.get('status')!s}"
                    )

        aer = derive_easy_aerobic_efficiency_trend(conn)
        if aer.get("debug_line"):
            dprint(aer["debug_line"])

        insight = build_interval_family_insight_summary(conn)
        th_ins = insight.get("threshold_latest_vs_prior") or {}
        v2_ins = insight.get("vo2max_latest_vs_prior") or {}
        th_avail = not th_ins.get("insufficient_history")
        v2_avail = not v2_ins.get("insufficient_history")
        dprint(
            "interval_family_insight_summary: "
            f"threshold={'available' if th_avail else 'insufficient'} "
            f"vo2={'available' if v2_avail else 'insufficient'}"
        )
        if th_avail:
            th_cur = th_ins.get("current") or {}
            th_base = th_ins.get("baseline") or {}
            th_mm = th_ins.get("metrics") or {}
            dprint(
                "  threshold: "
                f"cur_date={th_cur.get('run_date')!s} base_date={th_base.get('run_date')!s} "
                f"pace={(th_mm.get('work_mean_pace_sec_per_km') or {}).get('status')!s} "
                f"pwr={(th_mm.get('work_mean_power_w') or {}).get('status')!s} "
                f"hr={(th_mm.get('work_mean_hr_avg') or {}).get('status')!s} "
                f"w_hr={(th_mm.get('work_w_per_hr') or {}).get('status')!s}"
            )
        else:
            dprint(
                "  threshold: insufficient "
                f"reason={th_ins.get('reason')!s}"
            )
        if v2_avail:
            v2_cur = v2_ins.get("current") or {}
            v2_base = v2_ins.get("baseline") or {}
            v2_mm = v2_ins.get("metrics") or {}
            dprint(
                "  vo2max: "
                f"cur_date={v2_cur.get('run_date')!s} base_date={v2_base.get('run_date')!s} "
                f"pace={(v2_mm.get('work_mean_pace_sec_per_km') or {}).get('status')!s} "
                f"pwr={(v2_mm.get('work_mean_power_w') or {}).get('status')!s} "
                f"hr={(v2_mm.get('work_mean_hr_avg') or {}).get('status')!s} "
                f"w_hr={(v2_mm.get('work_w_per_hr') or {}).get('status')!s}"
            )
        else:
            dprint(
                "  vo2max: insufficient "
                f"reason={v2_ins.get('reason')!s}"
            )

        dprint(
            "interval_family_insight_text:\n"
            + format_interval_family_insight_summary(insight)
        )

    if not skip_write_batch:
        writer.write_batch(states)
    print(f"\nDone. Database: {db_path}\nOutputs: {out_dir}")

    stats["use_llm_requested"] = use_llm
    _out = Path(out_dir)
    _out.mkdir(parents=True, exist_ok=True)
    (_out / "last_import_summary.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return stats


def ingest_folder(
    input_dir: str,
    db_path: str,
    out_dir: str,
    use_llm: bool,
    *,
    date_from: date | None = None,
) -> dict[str, Any]:
    from agenticrun.agents.import_agent import ImportAgent
    from agenticrun.agents.output_agent import OutputAgent
    from agenticrun.agents.recommendation_agent import RecommendationAgent
    from agenticrun.agents.session_analysis_agent import SessionAnalysisAgent
    from agenticrun.agents.trend_agent import TrendAgent
    from agenticrun.services.db import connect
    from agenticrun.services.llm import LLMService

    conn = connect(db_path)
    importer = ImportAgent()
    analyzer = SessionAnalysisAgent()
    trend_agent = TrendAgent()
    recommender = RecommendationAgent()
    writer = OutputAgent(out_dir)
    llm = LLMService()
    print(
        llm.format_ingest_runtime_status_line(enabled_for_run=use_llm, used_for_run=False),
        flush=True,
    )

    input_path = Path(input_dir)
    with tempfile.TemporaryDirectory(prefix="agenticrun_zip_") as tmp_zip_dir:
        tmp_zip_path = Path(tmp_zip_dir)
        files: list[Path] = []

        files.extend(sorted(input_path.glob("*.csv")))
        files.extend(sorted(input_path.glob("*.fit")))

        for zip_path in sorted(input_path.glob("*.zip")):
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    for member in zf.infolist():
                        member_path = Path(member.filename)
                        if member.is_dir() or member_path.suffix.lower() != ".fit":
                            continue
                        safe_name = f"{zip_path.stem}__{member_path.name}"
                        extracted_path = tmp_zip_path / safe_name
                        with zf.open(member, "r") as src, open(extracted_path, "wb") as dst:
                            copyfileobj(src, dst)
                        files.append(extracted_path)
            except Exception as exc:
                dprint(f"Skipping {zip_path.name}: {exc}")

        files = sorted(files, key=lambda p: p.name.lower())
        if date_from is not None:
            print(f"ingest: date_from={date_from.isoformat()}", flush=True)
        if not files:
            print(f"No CSV, FIT, or ZIP-containing-FIT files found in {input_dir}")
            return _empty_ingest_summary()

        states, stats = _ingest_process_paths(
            files,
            conn=conn,
            out_dir=out_dir,
            use_llm=use_llm,
            importer=importer,
            analyzer=analyzer,
            trend_agent=trend_agent,
            recommender=recommender,
            llm=llm,
            date_from=date_from,
        )

    return _ingest_finalize_outputs(
        conn,
        states,
        stats,
        db_path=db_path,
        out_dir=out_dir,
        use_llm=use_llm,
        writer=writer,
    )


def bulk_import_folder(
    input_dir: str,
    db_path: str,
    out_dir: str,
    use_llm: bool,
    chunk_size: int,
    date_from: date | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Recursive tree import like ``scan``; processes candidates in chunks with progress lines."""
    from agenticrun.agents.import_agent import ImportAgent
    from agenticrun.agents.output_agent import OutputAgent
    from agenticrun.agents.recommendation_agent import RecommendationAgent
    from agenticrun.agents.session_analysis_agent import SessionAnalysisAgent
    from agenticrun.agents.trend_agent import TrendAgent
    from agenticrun.services.db import connect
    from agenticrun.services.llm import LLMService

    input_path = Path(input_dir)
    if not input_path.is_dir():
        print(f"bulk-import: not a directory: {input_dir}", flush=True)
        return {"error": "not_a_directory"}

    print(
        f"bulk-import: input={input_path.resolve()} db={Path(db_path).resolve()} "
        f"out={Path(out_dir).resolve()} chunk_size={chunk_size} resume={resume} "
        f"llm={'on' if use_llm else 'off'} "
        f"date_from={date_from.isoformat() if date_from else '-'}",
        flush=True,
    )

    conn = connect(db_path)
    importer = ImportAgent()
    analyzer = SessionAnalysisAgent()
    trend_agent = TrendAgent()
    recommender = RecommendationAgent()
    writer = OutputAgent(out_dir)
    llm = LLMService()
    print(
        llm.format_ingest_runtime_status_line(enabled_for_run=use_llm, used_for_run=False),
        flush=True,
    )

    all_states: list[Any] = []

    with tempfile.TemporaryDirectory(prefix="agenticrun_bulk_zip_") as tmp_zip_dir:
        tmp_zip_path = Path(tmp_zip_dir)
        targets, meta = _collect_scan_targets(input_path, tmp_zip_path)

        if meta["zip_read_errors"]:
            print("bulk-import: zip read issue(s) (first up to 3):", flush=True)
            for line in meta["zip_read_errors"][:3]:
                print(f"  {line}", flush=True)

        n_total = len(targets)
        if n_total == 0:
            print(
                f"bulk-import: no import candidates under {input_path} "
                f"(tree_files={meta['total_files_walked']})",
                flush=True,
            )
            return _empty_ingest_summary(use_llm)

        n_chunks = (n_total + chunk_size - 1) // chunk_size
        input_resolved = str(input_path.resolve())
        db_resolved = str(Path(db_path).resolve())
        out_resolved = str(Path(out_dir).resolve())

        if resume:
            cp = _bulk_read_checkpoint(out_dir)
            if cp is None:
                print(
                    "bulk-import: --resume but no readable "
                    f"{BULK_CHECKPOINT_FILENAME} in --out",
                    flush=True,
                )
                return {"error": "no_checkpoint"}
            err = _bulk_validate_checkpoint_for_resume(
                cp,
                input_resolved=input_resolved,
                db_resolved=db_resolved,
                out_resolved=out_resolved,
                chunk_size=chunk_size,
                use_llm=use_llm,
                date_from=date_from.isoformat() if date_from else None,
                n_total=n_total,
            )
            if err:
                print(f"bulk-import: resume aborted: {err}", flush=True)
                return {"error": "checkpoint_mismatch"}
            base = _empty_ingest_summary(use_llm)
            cum_stats = {**base, **(cp.get("cum_stats") or {})}
            for k, v in base.items():
                cum_stats.setdefault(k, v)
            start_ci = int(cp["last_completed_chunk_index"]) + 1
            print(
                f"bulk_import_resume: resuming at chunk {start_ci + 1}/{n_chunks}",
                flush=True,
            )
        else:
            if _bulk_checkpoint_path(out_dir).is_file():
                print(
                    "bulk-import: bulk_import_checkpoint.json exists in --out; "
                    "use --resume to continue or remove the file to start a new run.",
                    flush=True,
                )
                return {"error": "checkpoint_exists"}
            _bulk_clear_normalized_artifacts(out_dir)
            cum_stats = _empty_ingest_summary(use_llm)
            start_ci = 0

        cum_stats["uploaded_files"] = n_total
        t0 = time.perf_counter()

        if start_ci >= n_chunks:
            try:
                _bulk_checkpoint_path(out_dir).unlink()
            except OSError:
                pass
            print(
                "bulk-import: checkpoint had no remaining chunks; removed checkpoint file",
                flush=True,
            )
        else:
            for ci in range(start_ci, n_chunks):
                lo = ci * chunk_size
                hi = min(lo + chunk_size, n_total)
                chunk_paths = targets[lo:hi]
                states, st = _ingest_process_paths(
                    chunk_paths,
                    conn=conn,
                    out_dir=out_dir,
                    use_llm=use_llm,
                    importer=importer,
                    analyzer=analyzer,
                    trend_agent=trend_agent,
                    recommender=recommender,
                    llm=llm,
                    date_from=date_from,
                    skip_incomplete_fit=True,
                    ingest_quiet=True,
                )
                writer.append_batch(states)
                cum_stats["new_analyzed"] += st["new_analyzed"]
                cum_stats["duplicate_cached"] += st["duplicate_cached"]
                cum_stats["errors"] += st["errors"]
                cum_stats["skipped_cache_miss"] += st["skipped_cache_miss"]
                cum_stats["skipped_non_running"] += st["skipped_non_running"]
                cum_stats["skipped_by_date"] += st["skipped_by_date"]
                cum_stats["skipped_date_unknown"] += st["skipped_date_unknown"]
                cum_stats["incomplete_or_unsupported_fit"] += st["incomplete_or_unsupported_fit"]
                cum_stats["llm_api_calls"] += st["llm_api_calls"]
                cum_stats["llm_reused_stored"] += st["llm_reused_stored"]

                _bulk_write_checkpoint(
                    out_dir,
                    input_resolved=input_resolved,
                    db_resolved=db_resolved,
                    chunk_size=chunk_size,
                    use_llm=use_llm,
                    date_from=date_from.isoformat() if date_from else None,
                    n_total=n_total,
                    n_chunks=n_chunks,
                    last_completed_chunk_index=ci,
                    files_processed=hi,
                    cum_stats=dict(cum_stats),
                )

                elapsed = time.perf_counter() - t0
                print(
                    f"bulk_import_progress: chunk {ci + 1}/{n_chunks} "
                    f"files={hi}/{n_total} "
                    f"new={cum_stats['new_analyzed']} dup={cum_stats['duplicate_cached']} "
                    f"err={cum_stats['errors']} dup_cache_miss={cum_stats['skipped_cache_miss']} "
                    f"non_running_skip={cum_stats['skipped_non_running']} "
                    f"date_skip={cum_stats['skipped_by_date']} "
                    f"date_unknown={cum_stats['skipped_date_unknown']} "
                    f"incomplete_or_unsupported_fit={cum_stats['incomplete_or_unsupported_fit']} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

            try:
                _bulk_checkpoint_path(out_dir).unlink()
            except OSError:
                pass

        elapsed_total = time.perf_counter() - t0
        print(
            f"bulk_import_summary: files={n_total} "
            f"new={cum_stats['new_analyzed']} dup={cum_stats['duplicate_cached']} "
            f"err={cum_stats['errors']} dup_cache_miss={cum_stats['skipped_cache_miss']} "
            f"non_running_skip={cum_stats['skipped_non_running']} "
            f"date_skip={cum_stats['skipped_by_date']} "
            f"date_unknown={cum_stats['skipped_date_unknown']} "
            f"incomplete_or_unsupported_fit={cum_stats['incomplete_or_unsupported_fit']} "
            f"elapsed={elapsed_total:.1f}s",
            flush=True,
        )

    _write_bulk_import_last_summary(
        out_dir,
        input_resolved=input_resolved,
        db_resolved=db_resolved,
        out_resolved=out_resolved,
        chunk_size=chunk_size,
        resumed=resume,
        date_from=date_from.isoformat() if date_from else None,
        total_import_candidates=n_total,
        elapsed_sec=elapsed_total,
        cum_stats=cum_stats,
    )
    print(
        f"bulk_import_last_summary: wrote {BULK_IMPORT_SUMMARY_JSON} "
        f"and {BULK_IMPORT_SUMMARY_MD} under {out_resolved}",
        flush=True,
    )

    return _ingest_finalize_outputs(
        conn,
        all_states,
        cum_stats,
        db_path=db_path,
        out_dir=out_dir,
        use_llm=use_llm,
        writer=writer,
        skip_write_batch=True,
    )


def cmd_dedupe(db_path: str, apply: bool, *, dedupe_debug: bool = False) -> None:
    from agenticrun.services.db import connect, dedupe_fit_import_duplicates

    p = Path(db_path)
    if not p.is_file():
        print(f"No database file at {p}", flush=True)
        sys.exit(1)
    conn = connect(str(p))
    try:
        result = dedupe_fit_import_duplicates(conn, apply=apply, dedupe_debug=dedupe_debug)
        for g in result["groups"]:
            imp = g.get("impact_projected") or {}
            zn = int(imp.get("runs_zone_model_null_updates") or 0)
            print(
                f"duplicate_group: key={g['key']!s} keep={g['keep']!s} "
                f"remove={g['remove']!s} "
                f"impact_run_segments_delete={imp.get('run_segments_delete', 0)} "
                f"impact_zone_profiles_delete={imp.get('zone_profiles_delete', 0)} "
                f"impact_runs_zone_model_null_updates={zn} "
                f"impact_runs_delete={imp.get('runs_delete', 0)} "
                f"zone_model_nulls_would_apply={'yes' if zn > 0 else 'no'}",
                flush=True,
            )
            fe = g.get("fit_activity_key_survivor")
            if fe and fe.get("kind") == "promote":
                print(
                    "survivor_fit_activity_key_promoted: "
                    f"survivor={g['keep']!s} key={fe['key']!s} "
                    f"source_removed_run={fe['source_removed_run']!s}",
                    flush=True,
                )
            elif fe and fe.get("kind") == "conflict":
                print(
                    "survivor_fit_activity_key_conflict: "
                    f"survivor={g['keep']!s} candidate_keys={fe['candidate_keys']!s}",
                    flush=True,
                )
        mode = "apply" if apply else "dry_run"
        pt = result.get("impact_projected_totals") or {}
        print(
            f"dedupe_summary: duplicate_groups={result['duplicate_group_count']} "
            f"rows_to_remove={result['rows_to_remove']} mode={mode} "
            f"proj_run_segments_delete={pt.get('run_segments_delete', 0)} "
            f"proj_zone_profiles_delete={pt.get('zone_profiles_delete', 0)} "
            f"proj_runs_zone_model_null_updates={pt.get('runs_zone_model_null_updates', 0)} "
            f"proj_runs_delete={pt.get('runs_delete', 0)}",
            flush=True,
        )
        if apply:
            at = result.get("impact_apply_totals")
            if at:
                print(
                    "dedupe_summary_apply: "
                    f"run_segments_deleted={at.get('run_segments_deleted', 0)} "
                    f"zone_profiles_deleted={at.get('zone_profiles_deleted', 0)} "
                    f"runs_zone_model_nulled={at.get('runs_zone_model_nulled', 0)} "
                    f"runs_deleted={at.get('runs_deleted', 0)}",
                    flush=True,
                )
            else:
                print(
                    "dedupe_summary_apply: run_segments_deleted=0 zone_profiles_deleted=0 "
                    "runs_zone_model_nulled=0 runs_deleted=0",
                    flush=True,
                )
            for g in result["groups"]:
                fe = g.get("fit_activity_key_survivor")
                if not fe or fe.get("kind") != "promote":
                    continue
                surv = str(g["keep"])
                expected_key = str(fe["key"])
                donor = str(fe["source_removed_run"])
                remove_ids = [str(x) for x in g["remove"]]
                cur = conn.execute(
                    "SELECT fit_activity_key FROM runs WHERE run_id = ?",
                    (surv,),
                ).fetchone()
                surv_key = str(cur[0] or "").strip() if cur else ""
                ok = (
                    cur is not None
                    and surv_key == expected_key.strip()
                    and conn.execute(
                        "SELECT 1 FROM runs WHERE run_id = ?",
                        (donor,),
                    ).fetchone()
                    is None
                )
                for rid in remove_ids:
                    if conn.execute(
                        "SELECT 1 FROM runs WHERE run_id = ?",
                        (rid,),
                    ).fetchone():
                        ok = False
                print(
                    "dedupe_promotion_validate: "
                    f"survivor={surv!s} promoted_key={expected_key!s} removed_run={donor!s} "
                    f"ok={'yes' if ok else 'no'}",
                    flush=True,
                )
    finally:
        conn.close()


def cmd_delete_runs(
    *,
    db_path: str,
    latest: int | None,
    newer_than: date | None,
    apply: bool,
    include_non_running: bool = False,
) -> dict[str, int]:
    """Delete selected runs (dry-run by default) with dependent row cleanup."""
    import sqlite3

    from agenticrun.services.db import connect, _delete_run_and_dependents_rowcounts

    p = Path(db_path)
    if not p.is_file():
        print(f"No database file at {p}", flush=True)
        return {"selected": 0, "deleted": 0, "failed": 0, "trace_files_deleted": 0}

    conn = connect(str(p))
    conn.row_factory = sqlite3.Row
    trace_dir = Path("out") / "llm_traces"

    where_parts: list[str] = []
    params: list[Any] = []
    if not include_non_running:
        where_parts.append("(sport = 'running' OR sport IS NULL)")
    if newer_than is not None:
        where_parts.append("run_date >= ?")
        params.append(newer_than.isoformat())

    where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""
    limit_sql = ""
    if latest is not None:
        n = max(0, int(latest))
        limit_sql = f" LIMIT {n}"

    rows = conn.execute(
        f"""
        SELECT run_id, run_date, source_file, training_type, sport
        FROM runs
        {where_sql}
        ORDER BY run_date DESC, run_id DESC
        {limit_sql}
        """,
        tuple(params),
    ).fetchall()

    selected = len(rows)
    mode = "apply" if apply else "dry-run"
    print(
        f"delete-runs: mode={mode} selected={selected} "
        f"scope={'all_sports' if include_non_running else 'running_only'} "
        f"latest={latest if latest is not None else '-'} "
        f"newer_than={newer_than.isoformat() if newer_than else '-'}",
        flush=True,
    )
    print(
        "delete-runs: cleanup targets = runs, run_segments, zone_profiles(source_run_id), "
        "runs.zone_model_source_run_id(null update), out/llm_traces/<run_id>.json",
        flush=True,
    )

    if selected == 0:
        conn.close()
        return {"selected": 0, "deleted": 0, "failed": 0, "trace_files_deleted": 0}

    for i, r in enumerate(rows, start=1):
        print(
            f"  [{i}/{selected}] run_date={r['run_date'] or '-'} run_id={r['run_id']} "
            f"source_file={r['source_file'] or '-'} training_type={r['training_type'] or '-'} "
            f"sport={r['sport'] or '-'}",
            flush=True,
        )

    if not apply:
        conn.close()
        return {"selected": selected, "deleted": 0, "failed": 0, "trace_files_deleted": 0}

    deleted = 0
    failed = 0
    trace_files_deleted = 0
    runs_zone_model_nulled = 0
    run_segments_deleted = 0
    zone_profiles_deleted = 0
    runs_deleted = 0

    try:
        for i, r in enumerate(rows, start=1):
            rid = str(r["run_id"])
            print(f"delete-runs: applying [{i}/{selected}] run_id={rid}", flush=True)
            try:
                zn, ns, nz, nr = _delete_run_and_dependents_rowcounts(conn, rid)
                conn.commit()
                runs_zone_model_nulled += int(zn)
                run_segments_deleted += int(ns)
                zone_profiles_deleted += int(nz)
                runs_deleted += int(nr)
                if nr > 0:
                    deleted += 1
                tpath = trace_dir / f"{rid}.json"
                if tpath.is_file():
                    tpath.unlink()
                    trace_files_deleted += 1
                print(
                    f"delete-runs: deleted run_id={rid} run_segments={ns} zone_profiles={nz} "
                    f"runs_zone_model_nulled={zn} runs_deleted={nr}",
                    flush=True,
                )
            except Exception as exc:
                conn.rollback()
                failed += 1
                print(f"delete-runs: failed run_id={rid} error={exc}", flush=True)
    finally:
        conn.close()

    print(
        "delete-runs summary: "
        f"selected={selected} deleted={deleted} failed={failed} "
        f"run_segments_deleted={run_segments_deleted} zone_profiles_deleted={zone_profiles_deleted} "
        f"runs_zone_model_nulled={runs_zone_model_nulled} runs_deleted={runs_deleted} "
        f"trace_files_deleted={trace_files_deleted}",
        flush=True,
    )
    return {
        "selected": selected,
        "deleted": deleted,
        "failed": failed,
        "trace_files_deleted": trace_files_deleted,
    }


def cmd_audit_non_running(db_path: str, *, limit: int = 50) -> None:
    """List likely non-running FIT rows already imported in runs table (diagnostic only)."""
    import sqlite3

    p = Path(db_path)
    if not p.is_file():
        print(f"No database file at {p}", flush=True)
        sys.exit(1)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT run_id, run_date, source_file, source_type, sport, sub_sport, training_type
            FROM runs
            WHERE source_type = 'garmin_fit'
            ORDER BY run_date ASC, run_id ASC
            """
        ).fetchall()
    finally:
        conn.close()

    flagged: list[sqlite3.Row] = []
    for r in rows:
        sport = str(r["sport"] or "").strip().lower().replace(" ", "_")
        sub = str(r["sub_sport"] or "").strip().lower().replace(" ", "_")
        combo = f"{sport}_{sub}".strip("_")
        if not combo:
            continue
        if "cycl" in combo or "bike" in combo or "run" not in combo:
            flagged.append(r)

    print(
        "non_running_audit: "
        f"fit_rows={len(rows)} flagged_non_running={len(flagged)} "
        f"limit={max(0, int(limit))}",
        flush=True,
    )
    if not flagged:
        return
    n_show = max(0, int(limit))
    for r in flagged[:n_show]:
        print(
            f"- date={r['run_date'] or '-'} run_id={r['run_id']} "
            f"sport={r['sport'] or '-'} sub_sport={r['sub_sport'] or '-'} "
            f"training_type={r['training_type'] or '-'} file={r['source_file'] or '-'}",
            flush=True,
        )


def _normalize_csv_date_to_iso(value: object) -> str | None:
    """Parse common Garmin CSV date cells to ``YYYY-MM-DD``."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    m = re.match(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$", s)
    if m:
        dd, mm, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{yy:04d}-{mm:02d}-{dd:02d}"
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", s)
    if m:
        mm, dd, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{yy:04d}-{mm:02d}-{dd:02d}"
    try:
        dt = datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
        return dt.date().isoformat()
    except ValueError:
        pass
    try:
        dt = datetime.strptime(s[:10], "%d.%m.%Y")
        return dt.date().isoformat()
    except ValueError:
        pass
    return None


def _parse_pace_cell_to_sec_per_km(value: object) -> float | None:
    """Best-effort pace → seconds/km for CSV cells (m:ss, decimal min/km, or sec/km)."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        v = float(value)
        if 200 <= v <= 900:
            return v
        if 3.0 <= v <= 20.0:
            return v * 60.0
        return None
    s = str(value).strip().lower()
    if not s or s in {"nan", "-", "—"}:
        return None
    s = re.sub(r"\s*(min/km|min\s*km|/km)\s*$", "", s, flags=re.I).strip()
    m = re.match(r"^(\d+):(\d{1,2})(?::(\d{2}))?$", s)
    if m:
        if m.group(3) is not None:
            h, mi, se = int(m.group(1)), int(m.group(2)), int(m.group(3))
            total = h * 3600 + mi * 60 + se
        else:
            mi, se = int(m.group(1)), int(m.group(2))
            total = mi * 60 + se
        if 120 <= total <= 1200:
            return float(total)
        return None
    pf = parse_float(s.replace(",", "."))
    if pf is None:
        return None
    if 200 <= pf <= 900:
        return float(pf)
    if 3.0 <= pf <= 20.0:
        return float(pf * 60.0)
    return None


def _expected_label_from_reference(title: str, pace_sec_km: float | None) -> str | None:
    """User reference rules for offline validation (deterministic)."""
    t = (title or "").strip().lower()
    if "schwellenentwicklung" in t or "schwelle" in t:
        return "threshold"
    if "anaerob" in t:
        return "vo2"
    if "ga1" in t or "regenerativ" in t:
        return "easy"
    if pace_sec_km is not None and pace_sec_km > 7 * 60 + 30:
        return "easy"
    return None


def _pick_csv_columns(headers: list[str]) -> tuple[str | None, str | None, str | None]:
    date_col = None
    title_col = None
    pace_col = None
    for h in headers:
        if date_col is None and any(
            k in h for k in ("date", "datum", "day", "tag", "start", "zeit")
        ):
            date_col = h
        if title_col is None and (
            "activity name" in h
            or any(k in h for k in ("aktivität", "titel", "beschreibung"))
            or (("activity" in h or "sport" in h) and "file" not in h and "id" not in h)
            or (h == "title" or h.endswith(" title") or h.startswith("title "))
            or ("title" in h and "file" not in h and "subtitle" not in h)
        ):
            title_col = h
        if pace_col is None and "pace" in h and any(
            x in h for x in ("avg", "durch", "ø", "mean", "mittel")
        ):
            pace_col = h
    if date_col is None and headers:
        date_col = headers[0]
    if title_col is None:
        for h in headers:
            if "aktivität" in h or "titel" in h or h == "title" or "activity name" in h:
                title_col = h
                break
    if pace_col is None:
        for h in headers:
            if "pace" in h:
                pace_col = h
                break
    return date_col, title_col, pace_col


def _agenticrun_bucket_label(
    conn: Any, run_id: str, training_type: str | None
) -> tuple[str, str]:
    """Return (bucket, detail) bucket in easy|threshold|vo2|other."""
    from agenticrun.services.db import derive_work_session_family_for_run

    tt = (training_type or "").strip() or None
    tt_l = (training_type or "").strip().lower()
    wsf = derive_work_session_family_for_run(conn, run_id, training_type_hint=tt)
    fam = str((wsf or {}).get("work_session_family") or "other_interval_session")
    if tt_l in ("easy_run", "recovery_run"):
        return "easy", f"training_type={tt_l} family={fam}"
    if fam == "threshold_session":
        return "threshold", f"training_type={tt_l or '-'} family={fam}"
    if fam == "vo2max_session":
        return "vo2", f"training_type={tt_l or '-'} family={fam}"
    if tt_l == "threshold_run":
        return "threshold", f"training_type={tt_l} family={fam}"
    if "vo2" in tt_l or tt_l in ("test_or_vo2_session", "test_session", "race"):
        return "vo2", f"training_type={tt_l} family={fam}"
    return "other", f"training_type={tt_l or '-'} family={fam}"


def cmd_validate_classification(csv_path: str, db_path: str, *, max_mismatches: int = 25) -> None:
    """Compare reference CSV labels vs DB classification for matching run dates (offline)."""
    import sqlite3

    p = Path(csv_path)
    if not p.is_file():
        print(f"validate-classification: not a file: {csv_path}", flush=True)
        sys.exit(1)
    dbp = Path(db_path)
    if not dbp.is_file():
        print(f"validate-classification: no database at {db_path}", flush=True)
        sys.exit(1)

    with open(p, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print("validate-classification: CSV has no header row", flush=True)
            sys.exit(1)
        raw_fields = list(reader.fieldnames)
        headers = [clean_header(h) for h in raw_fields]
        rows: list[dict[str, str | None]] = []
        for raw in reader:
            rows.append(
                {headers[i]: raw.get(h) for i, h in enumerate(raw_fields)}
            )

    date_col, title_col, pace_col = _pick_csv_columns(headers)
    if date_col is None or date_col not in headers:
        print("validate-classification: could not detect a date column", flush=True)
        sys.exit(1)
    if title_col is None or title_col not in headers:
        print("validate-classification: could not detect a title column", flush=True)
        sys.exit(1)

    from agenticrun.services.db import connect

    conn = connect(str(dbp))
    conn.row_factory = sqlite3.Row
    try:
        matched = 0
        labeled_in_csv = 0
        agree = 0
        disagree = 0
        no_db = 0
        mismatches: list[tuple[str, str, str, str, str]] = []

        for raw in rows:
            iso = _normalize_csv_date_to_iso(raw.get(date_col))
            if not iso:
                continue
            title = str(raw.get(title_col) or "").strip()
            pace_raw = raw.get(pace_col) if pace_col else None
            pace_sec = _parse_pace_cell_to_sec_per_km(pace_raw)
            expected = _expected_label_from_reference(title, pace_sec)
            if expected is None:
                continue

            labeled_in_csv += 1
            cur = conn.execute(
                "SELECT run_id, title, training_type FROM runs WHERE run_date = ? "
                "ORDER BY run_id ASC LIMIT 1",
                (iso,),
            ).fetchone()
            if cur is None:
                no_db += 1
                continue
            matched += 1
            rid = str(cur["run_id"])
            ar_bucket, ar_detail = _agenticrun_bucket_label(conn, rid, cur["training_type"])
            if ar_bucket == expected:
                agree += 1
            else:
                disagree += 1
                if len(mismatches) < max(0, int(max_mismatches)):
                    db_title = str(cur["title"] or "").strip()[:60]
                    mismatches.append(
                        (
                            iso,
                            title[:80],
                            expected,
                            ar_bucket,
                            f"{ar_detail[:100]} | db_title={db_title!r}",
                        )
                    )
    finally:
        conn.close()

    print(
        "classification_validate: "
        f"csv_rows={len(rows)} labeled_in_csv={labeled_in_csv} matched_date={matched} "
        f"no_db_row={no_db} agree={agree} disagree={disagree} "
        f"date_col={date_col!r} title_col={title_col!r} pace_col={pace_col!r}",
        flush=True,
    )
    if mismatches:
        print("mismatches (date | title | expected | ar_bucket | ar_detail):", flush=True)
        for iso, title, exp, ab, det in mismatches:
            print(f"  {iso} | {title!r} | {exp} | {ab} | {det}", flush=True)


def _fmt_fit_timestamp(value: object) -> str:
    if value is None:
        return "-"
    if hasattr(value, "strftime"):
        try:
            return value.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(value)
    return str(value)


def show_fit_meta(input_path: str) -> dict[str, Any]:
    """Print a compact FIT summary using the same parse path as ingest (no DB, no LLM)."""
    from agenticrun.agents.import_agent import ImportAgent

    path = Path(input_path)
    if not path.is_file():
        print(f"show-fit-meta: not a file: {input_path}", flush=True)
        return {"ok": False, "reason": "not_a_file"}
    if path.suffix.lower() != ".fit":
        print(f"show-fit-meta: expected a single .fit file: {input_path}", flush=True)
        return {"ok": False, "reason": "not_fit"}

    size_b = path.stat().st_size
    print(f"path: {path.resolve()}", flush=True)
    print(f"basename: {path.name}", flush=True)
    print(f"size_bytes: {size_b}", flush=True)

    try:
        run, _bundle = ImportAgent()._build_run_record_from_fit(path)
    except Exception as exc:
        print("parse_ok: no", flush=True)
        print(f"error: {exc}", flush=True)
        return {"ok": False, "reason": "parse_failed", "error": str(exc)}

    raw = run.raw_summary or {}
    sess: dict[str, Any] = dict(raw.get("session") or {})
    act: dict[str, Any] = dict(raw.get("activity") or {})
    fid: dict[str, Any] = dict(raw.get("file_id") or {})
    metrics: dict[str, Any] = dict(raw.get("fit_session_metrics") or {})

    def _pick_sess(*keys: str) -> Any:
        for k in keys:
            if k in sess and sess[k] is not None:
                return sess[k]
            if k in act and act[k] is not None:
                return act[k]
        return None

    elapsed = _pick_sess("total_elapsed_time")
    timer = _pick_sess("total_timer_time")
    dist_m = _pick_sess("total_distance")
    start = sess.get("start_time") or act.get("timestamp")

    key = run.fit_activity_key
    gaid = None
    if key and key.startswith("gaid:"):
        gaid = key[5:]

    warn = metrics.get("fit_parse_warnings") or ""
    rc = int(metrics.get("record_count") or 0)
    lap_n = metrics.get("lap_count")
    dq = metrics.get("data_quality_score")

    _, assessment = _scan_real_activity_candidate(run)

    print("parse_ok: yes", flush=True)
    print(f"fit_activity_key: {key or '-'}", flush=True)
    if gaid:
        print(f"garmin_activity_id: {gaid}", flush=True)
    print(f"run_id_derived: {run.run_id}", flush=True)
    if fid:
        serial = fid.get("serial_number")
        prod = fid.get("product_name") or fid.get("product")
        manu = fid.get("manufacturer")
        bits = []
        if manu is not None:
            bits.append(f"manufacturer={manu}")
        if prod is not None:
            bits.append(f"product={prod}")
        if serial is not None:
            bits.append(f"serial_number={serial}")
        if bits:
            print("file_id: " + " ".join(bits), flush=True)
    print(f"start_time: {_fmt_fit_timestamp(start)}", flush=True)
    print(f"total_elapsed_s: {elapsed if elapsed is not None else '-'}", flush=True)
    print(f"total_timer_s: {timer if timer is not None else '-'}", flush=True)
    if dist_m is not None:
        try:
            print(f"total_distance_km: {float(dist_m) / 1000.0:.3f}", flush=True)
        except (TypeError, ValueError):
            print(f"total_distance_m: {dist_m}", flush=True)
    else:
        print("total_distance_km: -", flush=True)
    print(
        f"sport: {metrics.get('sport', '-')}  "
        f"sub_sport: {metrics.get('sub_sport', '-')}",
        flush=True,
    )
    print(f"title: {run.title}", flush=True)
    print(
        f"streams: record_messages={rc}  lap_messages={lap_n if lap_n is not None else '-'}  "
        f"data_quality_score={dq if dq is not None else '-'}",
        flush=True,
    )
    if warn:
        print(f"parse_notes: {warn}", flush=True)
    print(f"activity_assessment: {assessment}", flush=True)
    return {"ok": True}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AgenticRun prototype")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Parse Garmin CSV files from a folder")
    ingest.add_argument("--input", required=True, help="Folder containing Garmin CSV files")
    ingest.add_argument("--db", default="agenticrun.db", help="SQLite database path")
    ingest.add_argument("--out", default="out", help="Output folder")
    ingest.add_argument("--llm", action="store_true", help="Enable LLM summaries if API key is configured")
    ingest.add_argument(
        "--date-from",
        help="Only process runs on/after YYYY-MM-DD (parsed run date required)",
    )

    scan = sub.add_parser(
        "scan",
        help="Preflight: recurse input, parse candidates only (no import, no LLM); report duplicates vs DB",
    )
    scan.add_argument("--input", required=True, help="Folder tree to scan (recursive)")
    scan.add_argument("--db", default="agenticrun.db", help="SQLite database path for duplicate detection")
    scan.add_argument(
        "--date-from",
        help="Only include runs on/after YYYY-MM-DD in scan tallies",
    )

    bulk_imp = sub.add_parser(
        "bulk-import",
        help="Large archive import: chunked, reduced per-run logging, LLM off unless --llm",
    )
    bulk_imp.add_argument("--input", required=True, help="Folder tree to import (recursive)")
    bulk_imp.add_argument("--db", default="agenticrun.db", help="SQLite database path")
    bulk_imp.add_argument("--out", default="out", help="Output folder")
    bulk_imp.add_argument(
        "--llm",
        action="store_true",
        help="Enable per-run LLM summaries (default off for bulk; requires API configuration)",
    )
    bulk_imp.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        metavar="N",
        help="Maximum import candidates per chunk (default 100)",
    )
    bulk_imp.add_argument(
        "--resume",
        action="store_true",
        help="Resume from bulk_import_checkpoint.json in --out (same paths, chunk size, and --llm as the interrupted run)",
    )
    bulk_imp.add_argument(
        "--date-from",
        help="Only process runs on/after YYYY-MM-DD (parsed run date required)",
    )

    show_fit = sub.add_parser(
        "show-fit-meta",
        help="Inspect one .fit file: compact metadata (no DB import, no LLM)",
    )
    show_fit.add_argument(
        "--input",
        required=True,
        help="Path to a single .fit file",
    )

    llm_check = sub.add_parser("llm-check", help="Print LLM readiness diagnostics")
    llm_check.add_argument("--live", action="store_true", help="Run a tiny live LLM request test")

    zone_profiles = sub.add_parser("zone-profiles", help="List rows in zone_profiles (debug / audit)")
    zone_profiles.add_argument("--db", default="agenticrun.db", help="SQLite database path")

    audit_non_running = sub.add_parser(
        "audit-non-running",
        help="List likely non-running FIT rows already imported in runs table",
    )
    audit_non_running.add_argument("--db", default="agenticrun.db", help="SQLite database path")
    audit_non_running.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max flagged rows to print (default 50)",
    )

    backfill = sub.add_parser(
        "backfill-ai-summaries",
        help="Generate and store AI summaries for recent runs using existing LLM logic",
    )
    backfill.add_argument(
        "--db", default="agenticrun.db", help="SQLite database path (default: agenticrun.db)"
    )
    backfill.add_argument(
        "--latest",
        type=int,
        default=15,
        help="Number of most recent runs to process (default 15)",
    )
    backfill.add_argument(
        "--force",
        action="store_true",
        help="Regenerate summaries even when llm_summary/trace already exist",
    )
    backfill.add_argument(
        "--llm",
        action="store_true",
        help="Use live LLM calls (same behavior as ingest --llm); otherwise use deterministic fallback",
    )

    validate_cls = sub.add_parser(
        "validate-classification",
        help="Compare reference CSV labels vs DB classification by run date (offline, no LLM)",
    )
    validate_cls.add_argument(
        "--csv",
        required=True,
        help="Path to exported activities CSV (Garmin-style columns)",
    )
    validate_cls.add_argument("--db", default="agenticrun.db", help="SQLite database path")
    validate_cls.add_argument(
        "--max-mismatches",
        type=int,
        default=25,
        metavar="N",
        help="Max mismatch rows to print (default 25)",
    )

    dedupe = sub.add_parser(
        "dedupe",
        help="Detect or remove duplicate Garmin FIT activity rows (by fit_activity_key or work signature)",
    )
    dedupe.add_argument("--db", default="agenticrun.db", help="SQLite database path")
    dedupe_mx = dedupe.add_mutually_exclusive_group(required=True)
    dedupe_mx.add_argument(
        "--dry-run",
        action="store_true",
        help="List duplicate groups and rows that would be removed; do not change the database",
    )
    dedupe_mx.add_argument(
        "--apply",
        action="store_true",
        help="Remove duplicate rows (keep oldest run_id per group) and dependent segment/zone rows",
    )
    dedupe.add_argument(
        "--dedupe-debug",
        action="store_true",
        help="Verbose dedupe diagnostics (dry-run only: dedupe_diag_summary / dedupe_diag lines)",
    )

    delete_runs = sub.add_parser(
        "delete-runs",
        help="Delete recent runs and dependent rows (dry-run by default)",
    )
    delete_runs.add_argument("--db", default="agenticrun.db", help="SQLite database path")
    delete_runs.add_argument(
        "--latest",
        type=int,
        default=None,
        help="Select latest N runs (ordered by run_date DESC, run_id DESC)",
    )
    delete_runs.add_argument(
        "--newer-than",
        default=None,
        help="Select runs with run_date >= YYYY-MM-DD",
    )
    delete_runs.add_argument(
        "--include-non-running",
        action="store_true",
        help="Broaden scope to all sports (default is running-only)",
    )
    delete_runs.add_argument(
        "--apply",
        action="store_true",
        help="Apply deletions (default is dry-run preview)",
    )
    return parser


def main() -> None:
    dprint("main: starting CLI parse")
    parser = build_parser()
    args = parser.parse_args()
    date_from: date | None = None
    if hasattr(args, "date_from"):
        try:
            date_from = _parse_date_from_arg(args.date_from)
        except ValueError:
            print("invalid --date-from (expected YYYY-MM-DD)", flush=True)
            sys.exit(2)
    dprint(f"main: parsed command={args.command}")
    if args.command == "ingest":
        ingest_folder(args.input, args.db, args.out, args.llm, date_from=date_from)
    if args.command == "scan":
        result = scan_folder(args.input, args.db, date_from=date_from)
        if result.get("error") == "not_a_directory":
            sys.exit(1)
    if args.command == "bulk-import":
        if args.chunk_size < 1:
            print("bulk-import: --chunk-size must be >= 1", flush=True)
            sys.exit(2)
        result = bulk_import_folder(
            args.input,
            args.db,
            args.out,
            args.llm,
            args.chunk_size,
            date_from=date_from,
            resume=bool(args.resume),
        )
        err = result.get("error")
        if err == "not_a_directory":
            sys.exit(1)
        if err in ("checkpoint_exists", "no_checkpoint", "checkpoint_mismatch"):
            sys.exit(1)
    if args.command == "show-fit-meta":
        r = show_fit_meta(args.input)
        sys.exit(0 if r.get("ok") else 1)
    if args.command == "audit-non-running":
        if int(args.limit) < 0:
            print("audit-non-running: --limit must be >= 0", flush=True)
            sys.exit(2)
        cmd_audit_non_running(args.db, limit=int(args.limit))
        sys.exit(0)
    if args.command == "backfill-ai-summaries":
        selected = cmd_backfill_ai_summaries(
            db_path=args.db,
            latest=int(args.latest),
            use_llm=bool(args.llm),
            force=bool(args.force),
        )
        sys.exit(0 if selected.get("failed", 0) == 0 else 1)
    if args.command == "validate-classification":
        if int(args.max_mismatches) < 0:
            print("validate-classification: --max-mismatches must be >= 0", flush=True)
            sys.exit(2)
        cmd_validate_classification(
            args.csv, args.db, max_mismatches=int(args.max_mismatches)
        )
        sys.exit(0)
    if args.command == "dedupe":
        cmd_dedupe(args.db, apply=bool(args.apply), dedupe_debug=bool(args.dedupe_debug))
        sys.exit(0)
    if args.command == "delete-runs":
        newer_than = None
        if args.newer_than:
            try:
                newer_than = _parse_date_from_arg(args.newer_than)
            except ValueError:
                print("delete-runs: invalid --newer-than (expected YYYY-MM-DD)", flush=True)
                sys.exit(2)
        if args.latest is not None and int(args.latest) < 0:
            print("delete-runs: --latest must be >= 0", flush=True)
            sys.exit(2)
        if args.latest is None and newer_than is None:
            print("delete-runs: provide at least one selector: --latest N and/or --newer-than YYYY-MM-DD", flush=True)
            sys.exit(2)
        result = cmd_delete_runs(
            db_path=args.db,
            latest=(int(args.latest) if args.latest is not None else None),
            newer_than=newer_than,
            apply=bool(args.apply),
            include_non_running=bool(args.include_non_running),
        )
        if bool(args.apply) and result.get("failed", 0) > 0:
            sys.exit(1)
        sys.exit(0)
    if args.command == "zone-profiles":
        import sqlite3

        from agenticrun.services.zone_profiles import ensure_zone_profiles_table

        db_path = Path(args.db)
        if not db_path.is_file():
            print(f"No database file at {db_path}", flush=True)
            sys.exit(1)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            ensure_zone_profiles_table(conn)
            rows = conn.execute(
                """
                SELECT id, effective_from, source_run_id, source_type, zone_source,
                       hr_zone_boundaries, power_zone_boundaries,
                       hr_zone_time_sec, power_zone_time_sec,
                       threshold_heart_rate, functional_threshold_power, profile_fingerprint
                FROM zone_profiles
                ORDER BY effective_from ASC, id ASC
                """
            ).fetchall()
            print(f"zone_profiles: {len(rows)} row(s) in {db_path.resolve()}", flush=True)
            for r in rows:
                print(
                    f"  id={r['id']} effective_from={r['effective_from']!r} "
                    f"source_run_id={r['source_run_id']!r} zone_source={r['zone_source']!r}",
                    flush=True,
                )
                print(f"    hr_zone_boundaries={r['hr_zone_boundaries']}", flush=True)
                print(f"    power_zone_boundaries={r['power_zone_boundaries']}", flush=True)
                print(
                    f"    threshold_heart_rate={r['threshold_heart_rate']} "
                    f"functional_threshold_power={r['functional_threshold_power']}",
                    flush=True,
                )
                print(f"    hr_zone_time_sec={r['hr_zone_time_sec']}", flush=True)
                print(f"    power_zone_time_sec={r['power_zone_time_sec']}", flush=True)
                fp = r["profile_fingerprint"]
                fp_short = f"{fp[:12]}…" if isinstance(fp, str) and len(fp) > 12 else fp
                print(f"    profile_fingerprint={fp_short}", flush=True)
        finally:
            conn.close()
        sys.exit(0)
    if args.command == "llm-check":
        print("[llm-check] entered command dispatch", flush=True)
        dprint("main: dispatching llm-check")
        print("[llm-check] importing LLMService...", flush=True)
        from agenticrun.services.llm import LLMService
        print("[llm-check] constructing LLMService...", flush=True)
        dprint("main: constructing LLMService")
        llm = LLMService()
        print("[llm-check] running readiness/self-test...", flush=True)
        dprint("main: starting readiness/self-test")
        result = llm.self_test(live_test=args.live)
        readiness = result.get("readiness", {})
        issues = readiness.get("issues", [])
        warnings = readiness.get("warnings", [])
        live = result.get("live_test", {})
        ready = bool(readiness.get("ready"))
        live_failed = args.live and live.get("success") is False

        print("[llm-check] rendering final output...", flush=True)
        dprint("main: printing llm-check result")
        print("READY" if ready and not live_failed else "NOT READY")
        print(f"provider: {result.get('provider') or '-'}")
        print(f"model: {result.get('model') or '-'}")
        print(f"api_key_present: {'yes' if result.get('api_key_present') else 'no'}")
        print(f"env_source: {readiness.get('env_source') or '-'}")

        if issues:
            print(f"failure_category: config_error")
            print(f"reason: {issues[0]}")
            print("suggested_fix: Set AGENTICRUN_LLM_ENABLED=1, OPENAI_API_KEY, OPENAI_MODEL, and optional OPENAI_BASE_URL.")
        elif args.live:
            if live.get("success"):
                print("LIVE TEST OK")
            else:
                print("LIVE TEST FAILED")
                print(f"failure_category: {live.get('error_type') or 'unexpected_error'}")
                print(f"reason: {live.get('error_message') or 'unknown error'}")
                print(f"likely_cause: {live.get('likely_cause') or 'unknown'}")
                print(f"suggested_fix: {live.get('suggested_fix') or 'check logs and configuration'}")

        if warnings:
            print(f"warning: {warnings[0]}")

        if DEBUG:
            print(f"debug.readiness={json.dumps(readiness)}")
            if args.live:
                print(f"debug.live_test={json.dumps(live)}")

        if not ready or live_failed:
            sys.exit(1)
        sys.exit(0)


if __name__ == "__main__":
    main()