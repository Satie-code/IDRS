"""GNSS availability analysis and outage-interval extraction.

The distinction this module exists to protect: a **documented** outage is one
the dataset itself declares; an **observed** gap is one this tool measured.
IO-VNBD ships no outage annotation files, so every interval produced here is
observed or inferred, and is labelled as such. Calling an inferred gap a
confirmed GNSS outage would misrepresent the dataset to Phase 2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from idr.dataset.evidence import Evidence
from idr.dataset.scan import StreamScan

#: Source vocabulary for an outage interval.
SOURCE_DOCUMENTED = "documented"
SOURCE_OBSERVED_MISSINGNESS = "observed_missingness"
SOURCE_OBSERVED_POSITION_HOLD = "observed_position_hold"
SOURCE_INFERRED_FROM_QUALITY = "inferred_from_quality"

#: A held position lasting at least this long is reported as a staleness
#: interval. Chosen well above the smartphone's typical ~9 s update cadence so
#: routine updates are not mistaken for a fault.
POSITION_HOLD_THRESHOLD_S = 30.0

#: Consecutive invalid-fix samples shorter than this are sampling jitter,
#: not a navigation-relevant outage.
MIN_OUTAGE_DURATION_S = 1.0

#: Reported GNSS accuracy worse than this (metres) is treated as degraded
#: rather than usable. Screening threshold only — no data is altered.
POOR_ACCURACY_M = 50.0


@dataclass(frozen=True)
class OutageInterval:
    """One contiguous interval of unavailable or degraded GNSS."""

    session_id: str
    stream: str
    relative_path: str
    start_time_s: float
    end_time_s: float
    duration_s: float
    start_index: int
    end_index: int
    source: str
    confidence: str
    evidence: str


@dataclass
class OutageReport:
    """GNSS availability summary for one stream, plus its outage intervals."""

    session_id: str
    stream: str
    relative_path: str
    total_rows: int
    rows_with_fix: int
    availability_fraction: float
    has_gnss_fields: bool
    has_accuracy_field: bool
    mean_accuracy_m: float | None
    worst_accuracy_m: float | None
    outage_count: int
    total_outage_s: float
    longest_outage_s: float
    documented_outages_found: bool
    #: Rate at which the *position actually changes*, which on a forward-filled
    #: stream is far lower than the row rate.
    position_update_rate_hz: float | None
    position_update_count: int
    median_update_interval_s: float | None
    p90_update_interval_s: float | None
    max_update_interval_s: float | None
    position_hold_count: int
    longest_position_hold_s: float
    evidence: str
    note: str = ""


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive ``(start, end)`` index pairs for each True run."""
    if mask.size == 0 or not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b - 1)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def analyze_outages(
    scan: StreamScan,
    session_id: str,
    stream: str,
    *,
    min_duration_s: float = MIN_OUTAGE_DURATION_S,
) -> tuple[OutageReport, list[OutageInterval]]:
    """Measure GNSS availability and extract observed outage intervals."""
    has_gnss = scan.gnss_valid is not None
    accuracy = scan.column_by_role("gnss_accuracy")

    report = OutageReport(
        session_id=session_id,
        stream=stream,
        relative_path=scan.relative_path,
        total_rows=scan.row_count,
        rows_with_fix=0,
        availability_fraction=0.0,
        has_gnss_fields=has_gnss,
        has_accuracy_field=accuracy is not None,
        mean_accuracy_m=accuracy.mean if accuracy else None,
        worst_accuracy_m=accuracy.maximum if accuracy else None,
        outage_count=0,
        total_outage_s=0.0,
        longest_outage_s=0.0,
        # No IO-VNBD file declares outage periods; see the forensic report.
        documented_outages_found=False,
        position_update_rate_hz=None,
        position_update_count=0,
        median_update_interval_s=None,
        p90_update_interval_s=None,
        max_update_interval_s=None,
        position_hold_count=0,
        longest_position_hold_s=0.0,
        evidence=Evidence.VERIFIED_FROM_FILE if has_gnss else Evidence.UNVERIFIED,
        note="" if has_gnss else "schema has no GNSS latitude/longitude columns",
    )
    if not has_gnss or scan.gnss_valid is None:
        return report, []

    valid = scan.gnss_valid
    report.rows_with_fix = int(valid.sum())
    report.availability_fraction = float(valid.mean()) if valid.size else 0.0

    # Prefer the elapsed clock for interval timing; fall back to sample index
    # scaled by the median interval when no clock exists.
    times = scan.time_seconds
    if times is None or times.size != valid.size:
        times = np.arange(valid.size, dtype="float64") * 0.1

    intervals: list[OutageInterval] = []
    for start, end in _contiguous_runs(~valid):
        start_time = float(times[start]) if np.isfinite(times[start]) else float("nan")
        # The fix is absent until the next *valid* sample, so the outage runs
        # to that sample's timestamp. Measuring only first-to-last bad sample
        # would understate every gap by one sampling interval, and would call
        # a single dropped sample a zero-length outage.
        boundary = end + 1 if end + 1 < times.size else end
        end_time = float(times[boundary]) if np.isfinite(times[boundary]) else float("nan")
        duration = end_time - start_time
        if not np.isfinite(duration) or duration < min_duration_s:
            continue
        intervals.append(
            OutageInterval(
                session_id=session_id,
                stream=stream,
                relative_path=scan.relative_path,
                start_time_s=start_time,
                end_time_s=end_time,
                duration_s=duration,
                start_index=start,
                end_index=end,
                source=SOURCE_OBSERVED_MISSINGNESS,
                # Measured with certainty; the *cause* (tunnel, logger fault,
                # sensor dropout) is not established, hence not "confirmed".
                confidence="observed_high",
                evidence=Evidence.VERIFIED_FROM_FILE,
            )
        )

    report.outage_count = len(intervals)
    report.total_outage_s = float(sum(i.duration_s for i in intervals))
    report.longest_outage_s = max((i.duration_s for i in intervals), default=0.0)
    if not intervals:
        report.note = "no GNSS gaps at or above the minimum duration threshold"

    intervals.extend(_analyze_position_holds(scan, session_id, stream, times, report))
    return report, intervals


def _analyze_position_holds(
    scan: StreamScan,
    session_id: str,
    stream: str,
    times: np.ndarray,
    report: OutageReport,
) -> list[OutageInterval]:
    """Measure the real position-update cadence and long held-position spans.

    A stream can report a valid fix on every row while the *position* only
    refreshes occasionally. That staleness is invisible to availability
    counting but matters enormously to a navigation filter, so it is measured
    separately here.
    """
    changed = scan.gnss_position_changed
    if changed is None or changed.size < 2:
        return []

    # The first sample is itself a position report, so it counts as an update
    # event. Without it a stream that holds one position for a long time before
    # its first change would show no staleness at all.
    indices = np.unique(np.concatenate(([0], np.flatnonzero(changed))))
    report.position_update_count = int(indices.size)

    if indices.size < 2:
        report.note = (report.note + "; " if report.note else "") + (
            "position never changed after the first sample"
        )
        return []

    update_times = times[indices]
    # Rate from the intervals between updates, not events divided by file span:
    # trailing time after the last update would otherwise depress the figure.
    event_span = float(update_times[-1] - update_times[0])
    if event_span > 0:
        report.position_update_rate_hz = float(indices.size - 1) / event_span
    gaps = np.diff(update_times)
    gaps = gaps[np.isfinite(gaps)]
    if gaps.size:
        report.median_update_interval_s = float(np.median(gaps))
        report.p90_update_interval_s = float(np.percentile(gaps, 90))
        report.max_update_interval_s = float(gaps.max())

    holds: list[OutageInterval] = []
    for position, gap in enumerate(gaps):
        if gap < POSITION_HOLD_THRESHOLD_S:
            continue
        start_index = int(indices[position])
        end_index = int(indices[position + 1])
        holds.append(
            OutageInterval(
                session_id=session_id,
                stream=stream,
                relative_path=scan.relative_path,
                start_time_s=float(times[start_index]),
                end_time_s=float(times[end_index]),
                duration_s=float(gap),
                start_index=start_index,
                end_index=end_index,
                source=SOURCE_OBSERVED_POSITION_HOLD,
                # The hold is measured directly; whether it reflects lost
                # reception or merely a slow logger is not established.
                confidence="observed_high",
                evidence=Evidence.VERIFIED_FROM_FILE,
            )
        )

    report.position_hold_count = len(holds)
    report.longest_position_hold_s = max((h.duration_s for h in holds), default=0.0)
    return holds


def outage_report_to_dict(report: OutageReport) -> dict[str, object]:
    return asdict(report)


def outage_interval_to_dict(interval: OutageInterval) -> dict[str, object]:
    return asdict(interval)
