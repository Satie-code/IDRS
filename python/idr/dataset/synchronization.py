"""Diagnose the temporal relationship between paired smartphone and vehicle streams.

Phase 1 **diagnoses only** — nothing here resamples, shifts, or corrects a
stream. The output tells Phase 2 which pairs are safe to align and what
correction they would need.

Both streams are compared on a seconds-since-local-midnight clock: the vehicle
provides one directly, and the smartphone's ``DATE`` column is parsed into
one. Anything derived from that comparison (a constant offset, a drift rate)
is marked INFERRED, because the dataset ships no documented ground-truth
alignment to verify it against.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from idr.dataset.evidence import Evidence
from idr.dataset.scan import StreamScan

#: An offset within this many seconds of a whole hour is almost certainly a
#: timezone/DST bookkeeping difference rather than a sensor timing fault.
TIMEZONE_TOLERANCE_S = 5.0

#: Drift below this magnitude over a session is treated as no meaningful drift.
NEGLIGIBLE_DRIFT_S = 0.05


@dataclass
class SynchronizationReport:
    """Measured temporal relationship for one S/V session pair."""

    session_id: str
    dataset_family: str
    smartphone_file: str
    vehicle_file: str
    smartphone_rows: int
    vehicle_rows: int
    row_counts_match: bool
    smartphone_start_s: float | None
    smartphone_end_s: float | None
    vehicle_start_s: float | None
    vehicle_end_s: float | None
    raw_overlap_s: float | None
    offset_start_s: float | None
    offset_end_s: float | None
    mean_offset_s: float | None
    drift_s: float | None
    drift_ppm: float | None
    whole_hour_offset: int | None
    residual_offset_s: float | None
    aligned_overlap_s: float | None
    interpretation: str
    evidence: str
    note: str = ""


def _finite_bounds(values: np.ndarray | None) -> tuple[float, float] | None:
    if values is None:
        return None
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return None
    return float(finite[0]), float(finite[-1])


def analyze_pair(
    session_id: str,
    dataset_family: str,
    smartphone: StreamScan,
    vehicle: StreamScan,
) -> SynchronizationReport:
    """Compare one smartphone stream against its paired vehicle stream."""
    phone_bounds = _finite_bounds(smartphone.time_of_day_s)
    vehicle_bounds = _finite_bounds(vehicle.time_of_day_s)

    report = SynchronizationReport(
        session_id=session_id,
        dataset_family=dataset_family,
        smartphone_file=smartphone.relative_path,
        vehicle_file=vehicle.relative_path,
        smartphone_rows=smartphone.row_count,
        vehicle_rows=vehicle.row_count,
        row_counts_match=smartphone.row_count == vehicle.row_count,
        smartphone_start_s=phone_bounds[0] if phone_bounds else None,
        smartphone_end_s=phone_bounds[1] if phone_bounds else None,
        vehicle_start_s=vehicle_bounds[0] if vehicle_bounds else None,
        vehicle_end_s=vehicle_bounds[1] if vehicle_bounds else None,
        raw_overlap_s=None,
        offset_start_s=None,
        offset_end_s=None,
        mean_offset_s=None,
        drift_s=None,
        drift_ppm=None,
        whole_hour_offset=None,
        residual_offset_s=None,
        aligned_overlap_s=None,
        interpretation="unknown",
        evidence=Evidence.UNVERIFIED,
        note="",
    )

    if phone_bounds is None or vehicle_bounds is None:
        report.note = "one or both streams lack a usable time-of-day clock"
        return report

    phone_start, phone_end = phone_bounds
    vehicle_start, vehicle_end = vehicle_bounds

    report.raw_overlap_s = max(0.0, min(phone_end, vehicle_end) - max(phone_start, vehicle_start))
    report.offset_start_s = phone_start - vehicle_start
    report.offset_end_s = phone_end - vehicle_end
    report.mean_offset_s = (report.offset_start_s + report.offset_end_s) / 2.0
    report.drift_s = report.offset_end_s - report.offset_start_s

    vehicle_duration = vehicle_end - vehicle_start
    if vehicle_duration > 0:
        report.drift_ppm = 1e6 * report.drift_s / vehicle_duration

    # Split the offset into a whole-hour component (timezone bookkeeping) and
    # the residual sub-hour component (genuine start-time difference).
    hours = round(report.mean_offset_s / 3600.0)
    residual = report.mean_offset_s - hours * 3600.0
    report.whole_hour_offset = int(hours)
    report.residual_offset_s = residual

    shifted_start = phone_start - report.mean_offset_s
    shifted_end = phone_end - report.mean_offset_s
    report.aligned_overlap_s = max(
        0.0, min(shifted_end, vehicle_end) - max(shifted_start, vehicle_start)
    )

    drift_is_negligible = abs(report.drift_s) <= NEGLIGIBLE_DRIFT_S
    if hours != 0 and abs(residual) <= TIMEZONE_TOLERANCE_S and drift_is_negligible:
        report.interpretation = "constant_offset_whole_hour_likely_timezone"
        report.note = (
            f"streams differ by ~{hours}h ({report.mean_offset_s:.3f}s) with "
            f"{report.drift_s:+.4f}s drift; consistent with a UTC/local-time "
            f"difference rather than clock error"
        )
    elif drift_is_negligible:
        report.interpretation = "constant_offset"
        report.note = f"stable offset {report.mean_offset_s:.3f}s, drift {report.drift_s:+.4f}s"
    else:
        report.interpretation = "variable_offset_possible_drift"
        report.note = (
            f"offset moved {report.drift_s:+.4f}s over {vehicle_duration:.1f}s "
            f"({report.drift_ppm:+.1f} ppm)"
        )

    # Measured from file content, but the alignment conclusion is our
    # inference: the dataset documents no reference synchronization.
    report.evidence = Evidence.INFERRED
    return report


def synchronization_to_dict(report: SynchronizationReport) -> dict[str, object]:
    return asdict(report)
