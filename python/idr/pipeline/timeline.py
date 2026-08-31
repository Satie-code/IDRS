"""Timestamp normalization, duplicate/ordering diagnosis, and trip segmentation.

Phase 1 found files that concatenate several trips, with internal gaps up to
~661 s inside otherwise-10 Hz streams, plus duplicate and non-monotonic
timestamps. This module turns those findings into explicit structure:

- every row keeps its original clock (``source_time_s``) untouched;
- ``elapsed_s`` is derived per segment as a clean, monotonic analysis axis;
- duplicates and out-of-order rows are **flagged and grouped**, never deleted;
- large gaps split a file into logical segments, which become the unit of
  train/test splitting and sequence generation.

No resampling happens here. Actual sample timing is preserved, because Phase 1
established that the smartphone stream is not uniformly 10 Hz.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from idr.pipeline.canonical import QualityFlag

#: A gap at least this many multiples of the stream's own median sampling
#: interval starts a new segment. Relative rather than absolute because the
#: dataset contains ~2 Hz, ~10 Hz, ~24 Hz and ~1000 Hz streams — a fixed
#: seconds threshold would over-segment slow streams and under-segment fast ones.
SEGMENT_GAP_MULTIPLE = 100.0

#: Floor for the same rule, so a very fast stream cannot be chopped up by
#: ordinary jitter. A gap must exceed BOTH conditions to split a segment.
SEGMENT_GAP_MIN_SECONDS = 10.0

#: Segments shorter than this are still emitted but marked, since they are
#: rarely usable for sequence generation.
SHORT_SEGMENT_SECONDS = 5.0

#: A backward step of at least this size is treated as a *clock reset* — the
#: recorder starting a new trip and restarting its elapsed counter — rather
#: than local ordering noise. Verified in 15 smartphone files, e.g. S-S2.csv
#: drops from t=186.3 s to t=0.010 s at row 1863.
#:
#: Splitting on resets BEFORE any reordering is essential: sorting a file that
#: contains two trips each starting near t=0 interleaves them row by row and
#: produces a trajectory that never existed.
CLOCK_RESET_MIN_DROP_S = 1.0


@dataclass
class TimingDiagnosis:
    """What the timestamps of one stream actually look like."""

    source_file: str
    row_count: int
    valid_timestamps: int
    invalid_timestamps: int
    duplicate_timestamp_rows: int
    duplicate_groups: int
    non_monotonic_rows: int
    reordered: bool
    reorder_reason: str
    median_interval_s: float | None
    measured_rate_hz: float | None
    segment_count: int
    gap_threshold_s: float | None
    #: Backward clock jumps interpreted as a new trip starting.
    clock_reset_count: int = 0
    #: Forward gaps large enough to end a trip.
    large_gap_count: int = 0
    #: Resets of the raw source counter that the continuous clock does not
    #: show — reported, but not treated as trip boundaries.
    source_counter_reset_count: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SegmentRecord:
    """One logical trip carved out of a source file."""

    source_file: str
    session_id: str
    stream: str
    dataset_family: str
    payload_hash: str
    segment_id: str
    segment_index: int
    start_row: int
    end_row: int
    row_count: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    gap_before_s: float | None
    median_interval_s: float | None
    measured_rate_hz: float | None
    segmentation_reason: str
    is_short: bool


def _median_positive_interval(times: np.ndarray) -> float | None:
    finite = times[np.isfinite(times)]
    if finite.size < 2:
        return None
    deltas = np.diff(finite)
    positive = deltas[deltas > 0]
    if positive.size == 0:
        return None
    return float(np.median(positive))


def gap_threshold_for(median_interval: float | None) -> float | None:
    """The gap size that starts a new segment, given a stream's own cadence.

    Returns ``None`` when the cadence is unknown, in which case the caller must
    not segment (guessing a threshold would fabricate trip boundaries).
    """
    if median_interval is None or median_interval <= 0:
        return None
    return max(median_interval * SEGMENT_GAP_MULTIPLE, SEGMENT_GAP_MIN_SECONDS)


def _segmentation_reason(
    index: int,
    start: int,
    gaps: dict[int, float],
    reset_rows: set[int],
    threshold: float | None,
    median_interval: float | None,
) -> str:
    """Explain, per segment, exactly why it starts where it does."""
    if index == 0:
        return "first segment of file"
    delta = gaps.get(start, float("nan"))
    if start in reset_rows:
        return (
            f"clock reset: elapsed time jumped backwards by {abs(delta):.1f}s "
            f"(>= {CLOCK_RESET_MIN_DROP_S}s), indicating a new recording"
        )
    if threshold is not None and median_interval:
        return (
            f"forward gap of {delta:.1f}s exceeds threshold {threshold:.1f}s "
            f"(max({SEGMENT_GAP_MULTIPLE}x median interval {median_interval:.4f}s, "
            f"{SEGMENT_GAP_MIN_SECONDS}s))"
        )
    return "segment boundary"


def diagnose_and_segment(
    frame: pd.DataFrame,
    *,
    source_file: str,
    session_id: str,
    stream: str,
    dataset_family: str,
    payload_hash: str,
) -> tuple[pd.DataFrame, TimingDiagnosis, list[SegmentRecord]]:
    """Diagnose timing, assign segments, and derive ``elapsed_s``.

    Returns the annotated frame (same rows, same order as input unless a
    reorder was justified), the diagnosis, and one record per segment.
    """
    notes: list[str] = []
    working = frame.copy()
    # Segment on the continuous analysis clock, not the raw source counter:
    # the smartphone's elapsed counter restarts mid-drive in some files while
    # its wall clock runs through, so the counter is not a trip boundary.
    times = working["analysis_time_s"].to_numpy(dtype="float64")
    source_times = working["source_time_s"].to_numpy(dtype="float64")
    finite_mask = np.isfinite(times)
    row_count = len(working)

    flags = working["quality_flags"].to_numpy(dtype="int64").copy()
    flags[~finite_mask] |= QualityFlag.INVALID_TIMESTAMP.value

    # --- duplicates: flag and group, never drop ---------------------------
    duplicate_rows = 0
    duplicate_groups = 0
    if finite_mask.any():
        valid_times = pd.Series(times).where(pd.Series(finite_mask))
        duplicated_mask = valid_times.duplicated(keep=False) & pd.Series(finite_mask)
        duplicate_rows = int(duplicated_mask.sum())
        if duplicate_rows:
            duplicate_groups = int(valid_times[duplicated_mask].nunique())
            flags[duplicated_mask.to_numpy()] |= QualityFlag.DUPLICATE_TIMESTAMP.value
            notes.append(
                f"{duplicate_rows} row(s) share a timestamp across {duplicate_groups} group(s); "
                "all retained and flagged"
            )

    # --- segmentation, in SOURCE ORDER ------------------------------------
    #
    # Boundaries are found before any reordering. A file may concatenate
    # several trips, each restarting its elapsed clock at ~0; sorting such a
    # file by timestamp interleaves the trips into a trajectory that never
    # happened. Row order as written is therefore the authority for where one
    # trip ends and the next begins.
    median_interval = _median_positive_interval(times)
    rate = (1.0 / median_interval) if median_interval else None
    threshold = gap_threshold_for(median_interval)

    boundaries: list[int] = [0]
    gaps: dict[int, float] = {}
    reset_rows: set[int] = set()
    clock_resets = 0
    large_gaps = 0
    source_counter_resets = 0

    if finite_mask.sum() > 1:
        finite_idx = np.flatnonzero(finite_mask)
        deltas = np.diff(times[finite_idx])

        for position in np.flatnonzero(deltas <= -CLOCK_RESET_MIN_DROP_S):
            start_row = int(finite_idx[position + 1])
            boundaries.append(start_row)
            gaps[start_row] = float(deltas[position])
            reset_rows.add(start_row)
            clock_resets += 1

        if threshold is not None:
            for position in np.flatnonzero(deltas > threshold):
                start_row = int(finite_idx[position + 1])
                boundaries.append(start_row)
                gaps[start_row] = float(deltas[position])
                large_gaps += 1
        else:
            notes.append("no usable cadence; forward-gap segmentation was not applied")

    if clock_resets:
        notes.append(
            f"{clock_resets} reset(s) detected on the analysis clock; the file concatenates "
            "separate recordings and was split at those points WITHOUT sorting"
        )

    # A reset of the raw source counter that does NOT appear on the analysis
    # clock is a logger restarting its elapsed timer mid-recording. It is a
    # real property of the file and is reported, but it is not a trip boundary.
    source_finite = np.isfinite(source_times)
    if source_finite.sum() > 1:
        source_deltas = np.diff(source_times[np.flatnonzero(source_finite)])
        counter_resets = int((source_deltas <= -CLOCK_RESET_MIN_DROP_S).sum())
        if counter_resets:
            source_counter_resets = counter_resets
            notes.append(
                f"{counter_resets} reset(s) of the raw source counter (TIME SINCE START) that "
                "do not appear on the continuous analysis clock; the recording is continuous "
                "and was NOT split there"
            )

    boundaries = sorted(set(boundaries))

    # --- ordering, strictly WITHIN each segment ---------------------------
    #
    # Any remaining backward step inside a segment is local ordering noise (a
    # logger writing two samples out of order), not a trip boundary, so a
    # stable sort within the segment is safe: it cannot merge distinct trips,
    # because those are already separated above.
    non_monotonic = 0
    reordered = False
    reorder_reason = "input order preserved"
    order = np.arange(row_count)

    for index, start in enumerate(boundaries):
        end = boundaries[index + 1] if index + 1 < len(boundaries) else row_count
        span_times = times[start:end]
        span_finite = np.isfinite(span_times)
        if span_finite.sum() < 2:
            continue
        span_idx = np.flatnonzero(span_finite)
        backwards = np.flatnonzero(np.diff(span_times[span_idx]) < 0)
        if backwards.size == 0:
            continue
        non_monotonic += int(backwards.size)
        flags[start + span_idx[backwards + 1]] |= QualityFlag.NON_MONOTONIC_TIMESTAMP.value
        sortable = np.where(span_finite, span_times, np.inf)
        local_order = np.lexsort((np.arange(end - start), sortable))
        order[start:end] = start + local_order
        reordered = True

    if reordered:
        reorder_reason = (
            f"{non_monotonic} backward step(s) remained inside segment boundaries; rows were "
            "stably sorted WITHIN each segment only. No row was added, removed, or moved "
            "across a segment boundary."
        )
        notes.append(reorder_reason)

    working = working.iloc[order].reset_index(drop=True)
    flags = flags[order]
    times = times[order]
    finite_mask = np.isfinite(times)
    segment_ids = np.empty(row_count, dtype=object)
    segments: list[SegmentRecord] = []

    for index, start in enumerate(boundaries):
        end = boundaries[index + 1] - 1 if index + 1 < len(boundaries) else row_count - 1
        segment_id = f"{payload_hash[:12]}:{index:03d}"
        segment_ids[start : end + 1] = segment_id

        segment_times = times[start : end + 1]
        segment_finite = segment_times[np.isfinite(segment_times)]
        start_time = float(segment_finite[0]) if segment_finite.size else float("nan")
        end_time = float(segment_finite[-1]) if segment_finite.size else float("nan")
        duration = end_time - start_time if segment_finite.size > 1 else 0.0
        seg_median = _median_positive_interval(segment_times)

        flags[start] |= QualityFlag.SEGMENT_BOUNDARY.value
        segments.append(
            SegmentRecord(
                source_file=source_file,
                session_id=session_id,
                stream=stream,
                dataset_family=dataset_family,
                payload_hash=payload_hash,
                segment_id=segment_id,
                segment_index=index,
                start_row=start,
                end_row=end,
                row_count=end - start + 1,
                start_time_s=start_time,
                end_time_s=end_time,
                duration_s=duration,
                gap_before_s=gaps.get(start),
                median_interval_s=seg_median,
                measured_rate_hz=(1.0 / seg_median) if seg_median else None,
                segmentation_reason=_segmentation_reason(
                    index, start, gaps, reset_rows, threshold, median_interval
                ),
                is_short=bool(segment_finite.size > 1 and duration < SHORT_SEGMENT_SECONDS),
            )
        )

    working["segment_id"] = segment_ids

    # --- elapsed_s, derived per segment -----------------------------------
    elapsed = np.full(row_count, np.nan)
    for segment in segments:
        span = slice(segment.start_row, segment.end_row + 1)
        segment_times = times[span]
        if np.isfinite(segment.start_time_s):
            elapsed[span] = segment_times - segment.start_time_s
    working["elapsed_s"] = elapsed
    working["quality_flags"] = flags.astype("int32")

    diagnosis = TimingDiagnosis(
        source_file=source_file,
        row_count=row_count,
        valid_timestamps=int(finite_mask.sum()),
        invalid_timestamps=int((~finite_mask).sum()),
        duplicate_timestamp_rows=duplicate_rows,
        duplicate_groups=duplicate_groups,
        non_monotonic_rows=non_monotonic,
        reordered=reordered,
        reorder_reason=reorder_reason,
        median_interval_s=median_interval,
        measured_rate_hz=rate,
        segment_count=len(segments),
        gap_threshold_s=threshold,
        clock_reset_count=clock_resets,
        large_gap_count=large_gaps,
        source_counter_reset_count=source_counter_resets,
        notes=notes,
    )
    return working, diagnosis, segments
