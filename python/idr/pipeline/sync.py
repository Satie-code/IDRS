"""Per-pair smartphone/vehicle time alignment.

Phase 1 measured, and this module refines, the relationship between a paired
S/V recording. Two properties drive the design:

- **There is no universal offset.** Most pairs are near-aligned, a minority sit
  ~1 h apart, and some drift. A single hardcoded shift would be wrong for most
  of the dataset.
- **Nothing documents the true alignment**, so every estimate is an inference
  and is reported with a method and a confidence, never as ground truth.

The estimator is deterministic: given the same two streams it always returns
the same offset. When the evidence is too weak it returns ``UNRESOLVED``
rather than forcing a number.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

#: Candidate offsets are searched on this grid, in seconds, around the coarse
#: clock difference. 0.1 s matches the 10 Hz cadence of both streams.
SEARCH_STEP_S = 0.1

#: Half-width of the fine search window around the coarse estimate.
SEARCH_HALF_WIDTH_S = 30.0

#: Minimum overlap after alignment for a correlation estimate to be trusted.
MIN_OVERLAP_S = 60.0

#: Minimum correlation for a speed-based alignment to count as confirmed.
MIN_CORRELATION = 0.5

#: A correlation peak must beat the median candidate by this margin to be
#: considered distinct rather than a flat, ambiguous surface.
MIN_PEAK_MARGIN = 0.05


class SyncStatus(StrEnum):
    """Outcome of aligning one pair."""

    #: Speed cross-correlation found a clear, well-separated peak.
    CONFIRMED_BY_SIGNAL = "CONFIRMED_BY_SIGNAL"
    #: Clocks overlap and agree, but no independent signal confirmation.
    CLOCK_ONLY = "CLOCK_ONLY"
    #: No trustworthy alignment could be established.
    UNRESOLVED = "UNRESOLVED"


@dataclass
class SyncResult:
    """Estimated temporal relationship for one S/V pair."""

    pair_id: str
    session_id: str
    dataset_family: str
    smartphone_file: str
    vehicle_file: str
    smartphone_segment: str
    vehicle_segment: str
    smartphone_start_s: float | None
    smartphone_end_s: float | None
    vehicle_start_s: float | None
    vehicle_end_s: float | None
    overlap_start_s: float | None
    overlap_end_s: float | None
    overlap_duration_s: float | None
    clock_offset_s: float | None
    estimated_offset_s: float | None
    offset_method: str
    offset_confidence: str
    correlation: float | None
    peak_margin: float | None
    drift_detected: bool
    drift_estimate_s: float | None
    drift_ppm: float | None
    status: str
    note: str = ""


def _resample(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Sample a signal onto a uniform grid for correlation.

    This is a *derived analysis convenience only*: the resampled series is used
    to estimate an offset and is never written to canonical data, which keeps
    its original timing.
    """
    finite = np.isfinite(times) & np.isfinite(values)
    if finite.sum() < 2:
        return np.full(grid.size, np.nan)
    return np.interp(grid, times[finite], values[finite], left=np.nan, right=np.nan)


def _correlate(a: np.ndarray, b: np.ndarray) -> float:
    both = np.isfinite(a) & np.isfinite(b)
    if both.sum() < 10:
        return float("nan")
    x, y = a[both], b[both]
    if x.std() < 1e-9 or y.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def estimate_offset(
    phone_time: np.ndarray,
    phone_speed: np.ndarray,
    vehicle_time: np.ndarray,
    vehicle_speed: np.ndarray,
    *,
    coarse_offset: float,
    step_s: float = SEARCH_STEP_S,
    half_width_s: float = SEARCH_HALF_WIDTH_S,
) -> tuple[float | None, float | None, float | None]:
    """Refine ``coarse_offset`` by maximizing speed cross-correlation.

    Both streams carry a speed signal (smartphone GNSS speed, vehicle
    velocity) that responds to the same physical motion, which makes it the
    natural alignment signal. Returns ``(offset, correlation, peak_margin)``;
    the offset is ``None`` when no usable candidate exists.

    ``offset`` is defined so that ``phone_time - offset`` lands on the vehicle
    clock.
    """
    candidates = np.arange(
        coarse_offset - half_width_s, coarse_offset + half_width_s + step_s, step_s
    )
    if candidates.size == 0:
        return None, None, None

    v_finite = np.isfinite(vehicle_time) & np.isfinite(vehicle_speed)
    if v_finite.sum() < 10:
        return None, None, None
    v_start = float(vehicle_time[v_finite].min())
    v_end = float(vehicle_time[v_finite].max())
    if v_end - v_start < MIN_OVERLAP_S:
        return None, None, None

    grid = np.arange(v_start, v_end, step_s)
    if grid.size < 10:
        return None, None, None
    vehicle_grid = _resample(vehicle_time, vehicle_speed, grid)

    scores = np.full(candidates.size, np.nan)
    for index, candidate in enumerate(candidates):
        shifted = _resample(phone_time - candidate, phone_speed, grid)
        scores[index] = _correlate(shifted, vehicle_grid)

    if not np.isfinite(scores).any():
        return None, None, None

    best = int(np.nanargmax(scores))
    peak = float(scores[best])
    median = float(np.nanmedian(scores))
    return float(candidates[best]), peak, peak - median


def analyze_pair(
    pair_id: str,
    session_id: str,
    dataset_family: str,
    phone: dict[str, object],
    vehicle: dict[str, object],
) -> SyncResult:
    """Estimate alignment for one paired segment.

    ``phone``/``vehicle`` carry ``file``, ``segment``, ``time`` (time-of-day
    seconds) and ``speed`` (m/s) arrays.
    """
    phone_time = np.asarray(phone["time"], dtype="float64")
    vehicle_time = np.asarray(vehicle["time"], dtype="float64")
    phone_speed = np.asarray(phone["speed"], dtype="float64")
    vehicle_speed = np.asarray(vehicle["speed"], dtype="float64")

    result = SyncResult(
        pair_id=pair_id,
        session_id=session_id,
        dataset_family=dataset_family,
        smartphone_file=str(phone["file"]),
        vehicle_file=str(vehicle["file"]),
        smartphone_segment=str(phone["segment"]),
        vehicle_segment=str(vehicle["segment"]),
        smartphone_start_s=None,
        smartphone_end_s=None,
        vehicle_start_s=None,
        vehicle_end_s=None,
        overlap_start_s=None,
        overlap_end_s=None,
        overlap_duration_s=None,
        clock_offset_s=None,
        estimated_offset_s=None,
        offset_method="none",
        offset_confidence="none",
        correlation=None,
        peak_margin=None,
        drift_detected=False,
        drift_estimate_s=None,
        drift_ppm=None,
        status=SyncStatus.UNRESOLVED,
        note="",
    )

    p_finite = phone_time[np.isfinite(phone_time)]
    v_finite = vehicle_time[np.isfinite(vehicle_time)]
    if p_finite.size < 2 or v_finite.size < 2:
        result.note = "one or both streams lack a usable time-of-day clock"
        return result

    result.smartphone_start_s = float(p_finite[0])
    result.smartphone_end_s = float(p_finite[-1])
    result.vehicle_start_s = float(v_finite[0])
    result.vehicle_end_s = float(v_finite[-1])

    # Coarse offset from the clocks themselves, plus the drift implied by the
    # difference between the start-aligned and end-aligned offsets.
    offset_start = result.smartphone_start_s - result.vehicle_start_s
    offset_end = result.smartphone_end_s - result.vehicle_end_s
    result.clock_offset_s = (offset_start + offset_end) / 2.0
    result.drift_estimate_s = offset_end - offset_start
    vehicle_duration = result.vehicle_end_s - result.vehicle_start_s
    if vehicle_duration > 0:
        result.drift_ppm = 1e6 * result.drift_estimate_s / vehicle_duration
    result.drift_detected = abs(result.drift_estimate_s) > 1.0

    offset, correlation, margin = estimate_offset(
        phone_time,
        phone_speed,
        vehicle_time,
        vehicle_speed,
        coarse_offset=result.clock_offset_s,
    )

    if (
        offset is not None
        and correlation is not None
        and margin is not None
        and correlation >= MIN_CORRELATION
        and margin >= MIN_PEAK_MARGIN
    ):
        result.estimated_offset_s = offset
        result.correlation = correlation
        result.peak_margin = margin
        result.offset_method = "speed_cross_correlation"
        result.offset_confidence = "high" if correlation >= 0.8 else "medium"
        result.status = SyncStatus.CONFIRMED_BY_SIGNAL
        result.note = (
            f"speed correlation {correlation:.3f} at offset {offset:.2f}s "
            f"(peak margin {margin:.3f})"
        )
    else:
        result.correlation = correlation
        result.peak_margin = margin
        result.estimated_offset_s = result.clock_offset_s
        result.offset_method = "clock_difference"
        result.offset_confidence = "low"
        result.status = SyncStatus.CLOCK_ONLY
        reason = (
            "insufficient overlap or flat correlation surface"
            if correlation is None or not np.isfinite(correlation)
            else f"correlation {correlation:.3f} below {MIN_CORRELATION} or peak not distinct"
        )
        result.note = f"signal alignment not confirmed ({reason}); clock difference reported only"

    # Overlap on the vehicle clock after applying the chosen offset.
    shifted_start = result.smartphone_start_s - (result.estimated_offset_s or 0.0)
    shifted_end = result.smartphone_end_s - (result.estimated_offset_s or 0.0)
    result.overlap_start_s = max(shifted_start, result.vehicle_start_s)
    result.overlap_end_s = min(shifted_end, result.vehicle_end_s)
    result.overlap_duration_s = max(0.0, result.overlap_end_s - result.overlap_start_s)

    if result.overlap_duration_s < MIN_OVERLAP_S:
        result.status = SyncStatus.UNRESOLVED
        result.offset_confidence = "none"
        result.note = (
            f"aligned overlap {result.overlap_duration_s:.1f}s is below the "
            f"{MIN_OVERLAP_S:.0f}s minimum; alignment left UNRESOLVED"
        )
    return result
