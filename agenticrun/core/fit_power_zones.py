from __future__ import annotations

# Functional threshold (FTP-style) in watts — single place to configure zone scaling.
FIT_THRESHOLD_POWER_W: float = 250.0
FIT_HR_REFERENCE_BPM: float = 190.0

# Upper bounds as fractions of threshold: Z1 < u0, Z2 < u1, Z3 < u2, Z4 < u3, else Z5.
_FIT_POWER_ZONE_UPPER_RATIOS: tuple[float, ...] = (0.55, 0.75, 0.90, 1.05)
_FIT_HR_ZONE_UPPER_RATIOS: tuple[float, ...] = (0.60, 0.70, 0.80, 0.90)


def fit_power_zone_index(
    power_w: float,
    *,
    ftp_w: float | None = None,
    power_high_bounds: list[float] | None = None,
) -> int:
    bounds = [float(x) for x in (power_high_bounds or []) if x is not None and float(x) > 0]
    if len(bounds) >= 5:
        b = sorted(bounds)[:5]
        for i, hi in enumerate(b):
            if power_w <= hi:
                return min(i, 4)
        return 4
    ref = float(ftp_w) if ftp_w is not None and float(ftp_w) > 0 else FIT_THRESHOLD_POWER_W
    r = power_w / ref
    for i, upper in enumerate(_FIT_POWER_ZONE_UPPER_RATIOS):
        if r < upper:
            return i
    return 4


def fit_power_zone_seconds_from_records(
    timestamps: list[object],
    powers_w: list[float | None],
    *,
    ftp_w: float | None = None,
    power_high_bounds: list[float] | None = None,
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """Time in each zone (seconds) from FIT record rows with power; left-aligned to each sample."""
    ref = float(ftp_w) if ftp_w is not None and float(ftp_w) > 0 else FIT_THRESHOLD_POWER_W
    if ref <= 0 and not (power_high_bounds and len([x for x in power_high_bounds if x]) >= 5):
        return (None, None, None, None, None)
    if len(timestamps) != len(powers_w):
        return (None, None, None, None, None)
    if not any(p is not None for p in powers_w):
        return (None, None, None, None, None)

    totals = [0.0, 0.0, 0.0, 0.0, 0.0]
    last_ts: object | None = None
    last_power: float | None = None
    for ts, power in zip(timestamps, powers_w):
        if power is None:
            continue
        if last_ts is not None and hasattr(ts, "timestamp") and hasattr(last_ts, "timestamp"):
            delta = ts.timestamp() - last_ts.timestamp()
            if delta > 0 and last_power is not None:
                zi = fit_power_zone_index(
                    last_power,
                    ftp_w=ftp_w,
                    power_high_bounds=power_high_bounds,
                )
                totals[zi] += delta
        last_ts = ts
        last_power = power

    return (totals[0], totals[1], totals[2], totals[3], totals[4])


def fit_hr_zone_index(
    hr_bpm: float,
    *,
    hr_reference_bpm: float | None = None,
    hr_high_bounds: list[float] | None = None,
) -> int:
    bounds = [float(x) for x in (hr_high_bounds or []) if x is not None and float(x) > 0]
    if len(bounds) >= 5:
        b = sorted(bounds)[:5]
        for i, hi in enumerate(b):
            if hr_bpm <= hi:
                return min(i, 4)
        return 4
    ref = (
        float(hr_reference_bpm)
        if hr_reference_bpm is not None and float(hr_reference_bpm) > 0
        else FIT_HR_REFERENCE_BPM
    )
    r = hr_bpm / ref
    for i, upper in enumerate(_FIT_HR_ZONE_UPPER_RATIOS):
        if r < upper:
            return i
    return 4


def fit_hr_zone_seconds_from_records(
    timestamps: list[object],
    heart_rates_bpm: list[float | None],
    *,
    hr_reference_bpm: float | None = None,
    hr_high_bounds: list[float] | None = None,
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """Time in each HR zone (seconds) from FIT record rows with heart rate; left-aligned to each sample."""
    ref = (
        float(hr_reference_bpm)
        if hr_reference_bpm is not None and float(hr_reference_bpm) > 0
        else FIT_HR_REFERENCE_BPM
    )
    if ref <= 0 and not (hr_high_bounds and len([x for x in hr_high_bounds if x]) >= 5):
        return (None, None, None, None, None)
    if len(timestamps) != len(heart_rates_bpm):
        return (None, None, None, None, None)
    if not any(hr is not None for hr in heart_rates_bpm):
        return (None, None, None, None, None)

    totals = [0.0, 0.0, 0.0, 0.0, 0.0]
    last_ts: object | None = None
    last_hr: float | None = None
    for ts, hr in zip(timestamps, heart_rates_bpm):
        if hr is None:
            continue
        if last_ts is not None and hasattr(ts, "timestamp") and hasattr(last_ts, "timestamp"):
            delta = ts.timestamp() - last_ts.timestamp()
            if delta > 0 and last_hr is not None:
                zi = fit_hr_zone_index(
                    last_hr,
                    hr_reference_bpm=hr_reference_bpm,
                    hr_high_bounds=hr_high_bounds,
                )
                totals[zi] += delta
        last_ts = ts
        last_hr = hr

    return (totals[0], totals[1], totals[2], totals[3], totals[4])
