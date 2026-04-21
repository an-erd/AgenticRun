# Optional broad FIT message/field introspection when AGENTICRUN_FIT_INTROSPECT is set.
# Call site: import_agent._build_run_record_from_fit (Garmin zone logging uses AGENTICRUN_DEBUG only).

from __future__ import annotations

from collections import defaultdict
from typing import Any

from fitparse import FitFile

_KEYWORDS = (
    "zone",
    "time",
    "hr",
    "heart",
    "power",
    "threshold",
    "resting",
    "calc",
    "reference",
)


def _kw_hit(text: str) -> list[str]:
    t = text.lower()
    out: list[str] = []
    for k in _KEYWORDS:
        if k not in t:
            continue
        # Avoid treating every *timestamp* field as a "time" zone-related hit.
        if k == "time" and "timestamp" in t:
            continue
        out.append(k)
    return out


def _compact_val(val: Any, *, max_len: int = 96) -> str:
    if val is None:
        return "None"
    if isinstance(val, (bytes, bytearray)):
        return f"<{type(val).__name__} len={len(val)}>"
    if isinstance(val, (list, tuple)):
        n = len(val)
        if n == 0:
            return f"{type(val).__name__}[]"
        if n <= 5:
            return repr(val)
        return f"{type(val).__name__}[{n}] head={repr(val[:3])} ..."
    s = repr(val)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def print_fit_introspection_debug(fit: FitFile, file_label: str) -> None:
    """Print a compact parser-visible map of data messages (counts, field names, candidate samples)."""
    by_name: dict[str, dict[str, Any]] = {}
    samples: dict[str, list[dict[str, str]]] = defaultdict(list)

    for msg in fit.messages:
        if getattr(msg, "type", None) != "data":
            continue
        name = msg.name
        if name not in by_name:
            by_name[name] = {"count": 0, "fields": set(), "mesg_num": getattr(msg, "mesg_num", None)}
        ent = by_name[name]
        ent["count"] += 1
        if ent.get("mesg_num") is None:
            ent["mesg_num"] = getattr(msg, "mesg_num", None)

        row_values: dict[str, str] = {}
        field_names: list[str] = []
        for fd in msg.fields:
            fn = fd.name
            field_names.append(fn)
            ent["fields"].add(fn)
            row_values[fn] = _compact_val(fd.value)

        mesg_kws = _kw_hit(name)
        flat_fk: list[str] = []
        for fn in field_names:
            flat_fk.extend(_kw_hit(fn))
        is_candidate = bool(mesg_kws or flat_fk)
        if is_candidate and len(samples[name]) < 2:
            samples[name].append(
                {
                    "_introspect_keywords_mesg": ",".join(sorted(set(mesg_kws))) or "-",
                    "_introspect_keywords_fields": ",".join(sorted(set(flat_fk))) or "-",
                    **row_values,
                }
            )

    print("--- FIT introspection (AGENTICRUN_FIT_INTROSPECT; see fit_introspect_debug.py) ---", flush=True)
    print(f"  file: {file_label}", flush=True)
    print("  data_message_sections:", flush=True)

    for name in sorted(by_name.keys()):
        ent = by_name[name]
        flds = sorted(ent["fields"])
        mesg_num = ent.get("mesg_num")
        num_s = f" global_mesg_num={mesg_num}" if mesg_num is not None else ""
        mesg_kws = _kw_hit(name)
        field_kw_union: set[str] = set()
        for fn in flds:
            field_kw_union.update(_kw_hit(fn))
        highlight = ""
        if mesg_kws or field_kw_union:
            bits = []
            if mesg_kws:
                bits.append(f"name~{','.join(sorted(set(mesg_kws)))}")
            if field_kw_union:
                bits.append(f"fields~{','.join(sorted(field_kw_union))}")
            highlight = f"  <<< CANDIDATE ({'; '.join(bits)})"

        rf = repr(flds)
        if len(rf) <= 220:
            fields_compact = rf
        else:
            fields_compact = repr(flds[:20])[:-1] + f", ... (+{len(flds) - 20} more)]"
        print(f"    {name}: count={ent['count']}{num_s} fields={fields_compact}{highlight}", flush=True)

    printed_samples = False
    for name in sorted(samples.keys()):
        rows = samples[name]
        if not rows:
            continue
        if not printed_samples:
            print("  candidate_samples (max 2 rows per message type; compact values):", flush=True)
            printed_samples = True
        for i, row in enumerate(rows, 1):
            print(f"    [{name}] sample {i}/{len(rows)}:", flush=True)
            mk = row.get("_introspect_keywords_mesg", "")
            fk = row.get("_introspect_keywords_fields", "")
            print(f"      _introspect_keywords_mesg: {mk}  _introspect_keywords_fields: {fk}", flush=True)
            for k in sorted(row.keys()):
                if k.startswith("_introspect_keywords"):
                    continue
                print(f"      {k}: {row[k]}", flush=True)

    if not printed_samples:
        print("  candidate_samples: (none — no message/field names matched highlight terms)", flush=True)
