from __future__ import annotations

import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional


def clean_header(value: str) -> str:
    value = value.replace("\n", " ").replace("\r", " ").strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def parse_float(value) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    text = text.replace(".", "") if re.fullmatch(r"\d{1,3}(\.\d{3})+,\d+", text) else text
    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def parse_duration_to_seconds(value) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace(",", ".")
    if not text:
        return None
    parts = text.split(":")
    try:
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        if len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        return float(text)
    except ValueError:
        return None


def pace_from_distance_duration(distance_km: Optional[float], duration_sec: Optional[float]) -> Optional[float]:
    if not distance_km or not duration_sec or distance_km <= 0:
        return None
    return duration_sec / distance_km


def format_pace_min_km(seconds_per_km) -> str:
    """User-facing pace string: ``m:ss min/km`` (stored values are seconds per km)."""
    if seconds_per_km is None:
        return "n/a"
    try:
        v = float(seconds_per_km)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(v):
        return "n/a"
    total_seconds = int(v)
    m = total_seconds // 60
    s = total_seconds % 60
    return f"{m}:{s:02d} min/km"


def infer_date_from_filename(path: str) -> str:
    name = Path(path).stem
    m6 = re.search(r"(\d{2})(\d{2})(\d{2})", name)
    if m6:
        dd, mm, yy = m6.groups()
        return f"20{yy}-{mm}-{dd}"
    m4 = re.search(r"(\d{2})(\d{2})", name)
    if m4:
        dd, mm = m4.groups()
        current_year = datetime.now().year
        return f"{current_year}-{mm}-{dd}"
    return datetime.now().strftime("%Y-%m-%d")


def slugify_filename(path: str) -> str:
    base = os.path.basename(path)
    base = re.sub(r"[^a-zA-Z0-9]+", "-", base).strip("-").lower()
    return base
