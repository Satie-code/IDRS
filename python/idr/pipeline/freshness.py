"""GNSS fix availability and position freshness.

Phase 1 established two facts this module encodes:

1. **Every row of every stream reports a valid fix.** There are no GNSS
   outages in IO-VNBD, confirmed or otherwise.
2. **The smartphone position is forward-filled**, refreshing roughly every 9 s
   while rows arrive at ~10 Hz. A "100% availability" stream is therefore a
   *sparse* position source.

So a fix flag alone is misleading. Every row also gets ``position_fresh`` and
``stale_duration_s``, and a row whose position has been held beyond a
data-derived threshold is flagged ``GNSS_STALE``.

``GNSS_STALE`` means *the position has not been updated recently*. It does
**not** mean loss of signal, and it must never be relabelled as an outage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from idr.pipeline.canonical import QualityFlag

#: A held position is considered stale once it exceeds this multiple of the
#: stream's own median position-update interval. Derived from the stream rather
#: than fixed, because the smartphone updates ~9 s and the vehicle ~0.1 s —
#: one absolute threshold cannot serve both.
STALE_UPDATE_MULTIPLE = 2.0

#: Absolute floor, so a stream with a very fast position cadence is not
#: declared stale on ordinary jitter.
STALE_MIN_SECONDS = 1.0


@dataclass
class FreshnessReport:
    """Measured GNSS behaviour for one stream."""

    source_file: str
    stream: str
    row_count: int
    rows_with_fix: int
    availability_fraction: float
    position_update_count: int
    median_update_interval_s: float | None
    p90_update_interval_s: float | None
    max_update_interval_s: float | None
    measured_update_rate_hz: float | None
    stale_threshold_s: float | None
    stale_rows: int
    stale_fraction: float
    max_stale_duration_s: float
    #: Always 0 for IO-VNBD; kept explicit so the report can state it.
    confirmed_outages: int = 0


def annotate_freshness(
    frame: pd.DataFrame, *, source_file: str, stream: str
) -> tuple[pd.DataFrame, FreshnessReport]:
    """Add fix/freshness columns to a canonical frame and summarize them."""
    working = frame
    row_count = len(working)

    latitude = working["latitude"].to_numpy(dtype="float64")
    longitude = working["longitude"].to_numpy(dtype="float64")
    # Use the continuous analysis clock: the raw source counter restarts
    # mid-recording in some files, which would corrupt staleness durations.
    times = working["analysis_time_s"].to_numpy(dtype="float64")

    # (0, 0) is the classic "no fix" sentinel, not a position off West Africa.
    fix = np.isfinite(latitude) & np.isfinite(longitude) & ~((latitude == 0.0) & (longitude == 0.0))

    changed = np.zeros(row_count, dtype=bool)
    if row_count > 1:
        changed[1:] = (latitude[1:] != latitude[:-1]) | (longitude[1:] != longitude[:-1])
    # The first fixed row is itself a fresh observation.
    first_fix = np.flatnonzero(fix)
    if first_fix.size:
        changed[first_fix[0]] = True
    changed &= fix

    update_idx = np.flatnonzero(changed)
    update_times = times[update_idx] if update_idx.size else np.empty(0)
    finite_updates = update_times[np.isfinite(update_times)]

    median_interval = p90 = maximum = rate = None
    if finite_updates.size > 1:
        intervals = np.diff(finite_updates)
        intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
        if intervals.size:
            median_interval = float(np.median(intervals))
            p90 = float(np.percentile(intervals, 90))
            maximum = float(intervals.max())
            span = float(finite_updates[-1] - finite_updates[0])
            if span > 0:
                rate = float(finite_updates.size - 1) / span

    # Seconds since the position last changed.
    stale_duration = np.full(row_count, np.nan)
    last_update_time = np.nan
    for index in range(row_count):
        if changed[index] and np.isfinite(times[index]):
            last_update_time = times[index]
        if np.isfinite(times[index]) and np.isfinite(last_update_time):
            stale_duration[index] = times[index] - last_update_time

    threshold = None
    if median_interval is not None and median_interval > 0:
        threshold = max(median_interval * STALE_UPDATE_MULTIPLE, STALE_MIN_SECONDS)

    stale = np.zeros(row_count, dtype=bool)
    if threshold is not None:
        stale = np.isfinite(stale_duration) & (stale_duration > threshold)

    flags = working["quality_flags"].to_numpy(dtype="int64").copy()
    flags[stale] |= QualityFlag.GNSS_STALE.value

    working = working.copy()
    working["gnss_fix_available"] = pd.array(fix, dtype="boolean")
    working["gnss_position_fresh"] = pd.array(changed, dtype="boolean")
    working["gnss_stale_duration_s"] = stale_duration.astype("float32")
    working["quality_flags"] = flags.astype("int32")

    report = FreshnessReport(
        source_file=source_file,
        stream=stream,
        row_count=row_count,
        rows_with_fix=int(fix.sum()),
        availability_fraction=float(fix.mean()) if row_count else 0.0,
        position_update_count=int(update_idx.size),
        median_update_interval_s=median_interval,
        p90_update_interval_s=p90,
        max_update_interval_s=maximum,
        measured_update_rate_hz=rate,
        stale_threshold_s=threshold,
        stale_rows=int(stale.sum()),
        stale_fraction=float(stale.mean()) if row_count else 0.0,
        max_stale_duration_s=float(np.nanmax(stale_duration)) if row_count else 0.0,
    )
    return working, report
