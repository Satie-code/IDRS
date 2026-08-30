"""Timestamp forensics and sampling-rate classification.

Every rate reported here is *measured* from the data. The dataset
documentation states a nominal 10 Hz; this module never assumes it, and the
forensic report presents measured behaviour alongside the nominal claim so
the two can be compared.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from idr.dataset.evidence import Evidence, StatKind
from idr.dataset.scan import StreamScan

# Sampling-rate stability thresholds, expressed as the coefficient of
# variation of the inter-sample interval (std / mean). These are project
# conventions chosen for 10 Hz vehicle data, recorded in every report so a
# reader can re-derive the classification.
STABLE_CV_MAX = 0.05
MOSTLY_STABLE_CV_MAX = 0.20
IRREGULAR_CV_MAX = 1.00

SAMPLING_THRESHOLDS = {
    "stable": f"cv <= {STABLE_CV_MAX}",
    "mostly_stable": f"{STABLE_CV_MAX} < cv <= {MOSTLY_STABLE_CV_MAX}",
    "irregular": f"{MOSTLY_STABLE_CV_MAX} < cv <= {IRREGULAR_CV_MAX}",
    "severely_irregular": f"cv > {IRREGULAR_CV_MAX}",
    "unknown": "no usable timestamp column, or fewer than 2 valid timestamps",
}


@dataclass
class TimestampReport:
    """Measured timing behaviour of one stream."""

    relative_path: str
    time_column_role: str | None
    time_is_absolute: bool
    time_unit: str | None
    valid_timestamps: int
    missing_timestamps: int
    first_timestamp_s: float | None
    last_timestamp_s: float | None
    duration_s: float | None
    duration_kind: str
    is_monotonic: bool
    non_monotonic_count: int
    duplicate_timestamp_count: int
    zero_interval_count: int
    negative_interval_count: int
    delta_mean_s: float | None
    delta_median_s: float | None
    delta_std_s: float | None
    delta_min_s: float | None
    delta_max_s: float | None
    delta_p99_s: float | None
    estimated_rate_hz: float | None
    rate_kind: str
    sampling_class: str
    coefficient_of_variation: float | None
    #: True when the file has rows and a recognized time column, yet not one
    #: value in it parses as a number — the signature of a column-shifted file.
    suspected_column_misalignment: bool
    evidence: str
    note: str = ""


def _empty_report(
    scan: StreamScan, note: str, *, suspected_misalignment: bool = False
) -> TimestampReport:
    return TimestampReport(
        relative_path=scan.relative_path,
        time_column_role=scan.time_role,
        time_is_absolute=False,
        time_unit=scan.time_unit,
        valid_timestamps=0,
        missing_timestamps=scan.row_count,
        first_timestamp_s=None,
        last_timestamp_s=None,
        duration_s=None,
        duration_kind=StatKind.ESTIMATED,
        is_monotonic=False,
        non_monotonic_count=0,
        duplicate_timestamp_count=0,
        zero_interval_count=0,
        negative_interval_count=0,
        delta_mean_s=None,
        delta_median_s=None,
        delta_std_s=None,
        delta_min_s=None,
        delta_max_s=None,
        delta_p99_s=None,
        estimated_rate_hz=None,
        rate_kind=StatKind.EXACT,
        sampling_class="unknown",
        coefficient_of_variation=None,
        suspected_column_misalignment=suspected_misalignment,
        evidence=Evidence.UNVERIFIED,
        note=note,
    )


def classify_sampling(coefficient_of_variation: float | None) -> str:
    """Map an interval coefficient of variation to a stability class."""
    if coefficient_of_variation is None:
        return "unknown"
    if coefficient_of_variation <= STABLE_CV_MAX:
        return "stable"
    if coefficient_of_variation <= MOSTLY_STABLE_CV_MAX:
        return "mostly_stable"
    if coefficient_of_variation <= IRREGULAR_CV_MAX:
        return "irregular"
    return "severely_irregular"


def analyze_timestamps(scan: StreamScan) -> TimestampReport:
    """Compute exact timing statistics from a completed :class:`StreamScan`."""
    if scan.time_seconds is None:
        return _empty_report(scan, "no recognized timestamp column in this schema")

    raw = scan.time_seconds
    finite = np.isfinite(raw)
    missing = int((~finite).sum())
    values = raw[finite]

    if values.size < 2:
        # A file with plenty of rows whose time column yields no numbers at all
        # is almost certainly shifted: some other field has landed in the
        # timestamp column. Flagging it stops Phase 2 trusting the labels.
        misaligned = values.size == 0 and scan.row_count > 1
        note = (
            "no value in the time column parses as a number despite "
            f"{scan.row_count} rows — column misalignment suspected"
            if misaligned
            else "fewer than 2 valid timestamps"
        )
        report = _empty_report(scan, note, suspected_misalignment=misaligned)
        report.valid_timestamps = int(values.size)
        report.missing_timestamps = missing
        return report

    deltas = np.diff(values)
    positive = deltas[deltas > 0]

    mean = float(positive.mean()) if positive.size else None
    std = float(positive.std()) if positive.size > 1 else None
    cv = (std / mean) if (mean not in (None, 0.0) and std is not None) else None

    # Rate is derived from the median interval rather than the mean: a handful
    # of large gaps would drag the mean and understate the true cadence.
    median = float(np.median(positive)) if positive.size else None
    rate = (1.0 / median) if median else None

    return TimestampReport(
        relative_path=scan.relative_path,
        time_column_role=scan.time_role,
        # "time since start" is elapsed; "time of day" is an absolute
        # within-day clock. Neither carries a date, so neither is a full
        # absolute timestamp on its own.
        time_is_absolute=scan.time_role == "time_of_day_s",
        time_unit=scan.time_unit,
        valid_timestamps=int(values.size),
        missing_timestamps=missing,
        first_timestamp_s=float(values[0]),
        last_timestamp_s=float(values[-1]),
        duration_s=float(values[-1] - values[0]),
        duration_kind=StatKind.ESTIMATED,
        is_monotonic=bool(np.all(deltas > 0)),
        non_monotonic_count=int((deltas < 0).sum()),
        duplicate_timestamp_count=int(values.size - np.unique(values).size),
        zero_interval_count=int((deltas == 0).sum()),
        negative_interval_count=int((deltas < 0).sum()),
        delta_mean_s=mean,
        delta_median_s=median,
        delta_std_s=std,
        delta_min_s=float(positive.min()) if positive.size else None,
        delta_max_s=float(positive.max()) if positive.size else None,
        delta_p99_s=float(np.percentile(positive, 99)) if positive.size else None,
        estimated_rate_hz=rate,
        rate_kind=StatKind.EXACT,
        sampling_class=classify_sampling(cv),
        coefficient_of_variation=cv,
        suspected_column_misalignment=False,
        evidence=Evidence.VERIFIED_FROM_FILE,
    )


def timestamp_report_to_dict(report: TimestampReport) -> dict[str, object]:
    return asdict(report)
