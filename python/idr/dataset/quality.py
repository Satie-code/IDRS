"""Missing-data analysis and non-destructive range/sanity diagnostics.

Nothing in this module alters, filters, or drops data. Suspicious values are
*classified*, never removed — a 1.2 g longitudinal acceleration is a plausible
hard-braking event, not automatically an error, so findings carry an explicit
interpretation rather than a verdict.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from idr.dataset.evidence import Evidence, StatKind
from idr.dataset.scan import StreamScan
from idr.dataset.timestamps import TimestampReport

#: Interpretation vocabulary for a value that falls outside an expected band.
POSSIBLE_VALID_EXTREME = "possible_valid_extreme"
POSSIBLE_SENSOR_ARTIFACT = "possible_sensor_artifact"
POSSIBLE_DATA_ERROR = "possible_data_error"
UNKNOWN_ANOMALY = "unknown"

# Physically-motivated plausibility bands, per semantic role, as
# ``(low, high, interpretation_when_outside)``. These are sanity screens for a
# road vehicle, not validity rules: exceeding a band produces a finding for a
# human to read, never a modification.
_RANGE_CHECKS: dict[str, tuple[float, float, str]] = {
    "gnss_latitude": (-90.0, 90.0, POSSIBLE_DATA_ERROR),
    "gnss_longitude": (-180.0, 180.0, POSSIBLE_DATA_ERROR),
    "gnss_altitude": (-500.0, 9000.0, POSSIBLE_DATA_ERROR),
    "gnss_speed": (0.0, 250.0, POSSIBLE_VALID_EXTREME),
    "gnss_accuracy": (0.0, 1000.0, POSSIBLE_SENSOR_ARTIFACT),
    "accel_x": (-50.0, 50.0, POSSIBLE_SENSOR_ARTIFACT),
    "accel_y": (-50.0, 50.0, POSSIBLE_SENSOR_ARTIFACT),
    "accel_z": (-50.0, 50.0, POSSIBLE_SENSOR_ARTIFACT),
    "gyro_yaw": (-35.0, 35.0, POSSIBLE_SENSOR_ARTIFACT),
    "gyro_pitch": (-35.0, 35.0, POSSIBLE_SENSOR_ARTIFACT),
    "gyro_roll": (-35.0, 35.0, POSSIBLE_SENSOR_ARTIFACT),
    "velocity": (0.0, 250.0, POSSIBLE_VALID_EXTREME),
    "vehicle_speed": (0.0, 250.0, POSSIBLE_VALID_EXTREME),
    "heading": (-360.0, 360.0, POSSIBLE_DATA_ERROR),
    "yaw_rate": (-200.0, 200.0, POSSIBLE_VALID_EXTREME),
    "accel_longitudinal": (-2.0, 2.0, POSSIBLE_VALID_EXTREME),
    "accel_lateral": (-2.0, 2.0, POSSIBLE_VALID_EXTREME),
    "steering_angle": (-720.0, 720.0, POSSIBLE_VALID_EXTREME),
    "engine_speed": (0.0, 10000.0, POSSIBLE_SENSOR_ARTIFACT),
}


@dataclass(frozen=True)
class RangeFinding:
    """One column whose observed range falls outside its plausibility band."""

    relative_path: str
    column: str
    semantic_role: str
    observed_min: float | None
    observed_max: float | None
    expected_low: float
    expected_high: float
    interpretation: str
    evidence: str
    detail: str


@dataclass
class MissingnessRecord:
    """Missing/unparseable counts for one column."""

    relative_path: str
    column: str
    semantic_role: str
    total_rows: int
    missing_rows: int
    missing_percentage: float
    non_numeric_rows: int
    is_constant: bool
    kind: str


@dataclass
class SessionQuality:
    """Aggregate quality profile for one stream.

    ``quality_score`` is a transparent mean of four sub-scores, each in [0, 1]
    and each retained separately so the score can never hide the metric that
    produced it. It ranks streams for attention; it never excludes one.
    """

    relative_path: str
    session_id: str
    stream: str
    schema_valid: bool
    row_count: int
    timestamp_quality: float
    missingness_score: float
    sampling_quality: float
    gnss_availability: float
    sampling_class: str
    measured_rate_hz: float | None
    range_finding_count: int
    parse_error: str | None
    notes: list[str] = field(default_factory=list)

    @property
    def quality_score(self) -> float:
        return round(
            (
                self.timestamp_quality
                + self.missingness_score
                + self.sampling_quality
                + self.gnss_availability
            )
            / 4.0,
            4,
        )

    @property
    def score_formula(self) -> str:
        return (
            "mean(timestamp_quality, missingness_score, sampling_quality, "
            "gnss_availability); each in [0,1]; never used to drop a session"
        )


def analyze_missingness(scan: StreamScan) -> list[MissingnessRecord]:
    """Per-column missing and unparseable counts (exact, whole file)."""
    return [
        MissingnessRecord(
            relative_path=scan.relative_path,
            column=stats.normalized_name,
            semantic_role=stats.semantic_role,
            total_rows=stats.total_rows,
            missing_rows=stats.missing_rows,
            missing_percentage=round(stats.missing_percentage, 6),
            non_numeric_rows=stats.non_numeric_rows,
            is_constant=stats.is_constant,
            kind=StatKind.EXACT,
        )
        for stats in scan.columns.values()
    ]


def analyze_ranges(scan: StreamScan) -> list[RangeFinding]:
    """Report columns whose observed extremes leave their plausibility band."""
    findings: list[RangeFinding] = []
    for stats in scan.columns.values():
        check = _RANGE_CHECKS.get(stats.semantic_role)
        if check is None or stats.numeric_count == 0:
            continue
        low, high, interpretation = check
        if stats.minimum is None or stats.maximum is None:
            continue
        if stats.minimum >= low and stats.maximum <= high:
            continue
        findings.append(
            RangeFinding(
                relative_path=scan.relative_path,
                column=stats.normalized_name,
                semantic_role=stats.semantic_role,
                observed_min=stats.minimum,
                observed_max=stats.maximum,
                expected_low=low,
                expected_high=high,
                interpretation=interpretation,
                evidence=Evidence.VERIFIED_FROM_FILE,
                detail=(
                    f"observed [{stats.minimum:.4g}, {stats.maximum:.4g}] "
                    f"outside screen [{low:g}, {high:g}]"
                ),
            )
        )

    # A sensor stuck at one value across a whole session is a distinct
    # pathology from an out-of-range spike, so it gets its own finding.
    for stats in scan.columns.values():
        if stats.is_constant and stats.semantic_role in _RANGE_CHECKS:
            findings.append(
                RangeFinding(
                    relative_path=scan.relative_path,
                    column=stats.normalized_name,
                    semantic_role=stats.semantic_role,
                    observed_min=stats.minimum,
                    observed_max=stats.maximum,
                    expected_low=float("nan"),
                    expected_high=float("nan"),
                    interpretation=POSSIBLE_SENSOR_ARTIFACT,
                    evidence=Evidence.VERIFIED_FROM_FILE,
                    detail=(
                        f"constant value {stats.minimum!r} across "
                        f"{stats.numeric_count} rows (possible stuck channel)"
                    ),
                )
            )
    return findings


def _gnss_availability(scan: StreamScan) -> float:
    if scan.gnss_valid is None or scan.gnss_valid.size == 0:
        return 0.0
    return float(scan.gnss_valid.mean())


def build_session_quality(
    scan: StreamScan,
    timestamps: TimestampReport,
    session_id: str,
    stream: str,
    range_findings: list[RangeFinding],
) -> SessionQuality:
    """Combine per-stream metrics into a transparent quality profile."""
    notes: list[str] = []

    if timestamps.valid_timestamps:
        bad = (
            timestamps.non_monotonic_count
            + timestamps.duplicate_timestamp_count
            + timestamps.zero_interval_count
        )
        timestamp_quality = max(0.0, 1.0 - bad / timestamps.valid_timestamps)
    else:
        timestamp_quality = 0.0
        notes.append("no usable timestamps")

    columns = list(scan.columns.values())
    if columns and scan.row_count:
        mean_missing = sum(c.missing_percentage for c in columns) / len(columns)
        missingness_score = max(0.0, 1.0 - mean_missing / 100.0)
    else:
        missingness_score = 0.0
        notes.append("no rows scanned")

    sampling_quality = {
        "stable": 1.0,
        "mostly_stable": 0.75,
        "irregular": 0.4,
        "severely_irregular": 0.1,
        "unknown": 0.0,
    }[timestamps.sampling_class]

    if scan.parse_error:
        notes.append(f"parse error: {scan.parse_error}")

    return SessionQuality(
        relative_path=scan.relative_path,
        session_id=session_id,
        stream=stream,
        schema_valid=scan.schema.column_count > 0 and scan.parse_error is None,
        row_count=scan.row_count,
        timestamp_quality=round(timestamp_quality, 4),
        missingness_score=round(missingness_score, 4),
        sampling_quality=sampling_quality,
        gnss_availability=round(_gnss_availability(scan), 4),
        sampling_class=timestamps.sampling_class,
        measured_rate_hz=(
            round(timestamps.estimated_rate_hz, 4) if timestamps.estimated_rate_hz else None
        ),
        range_finding_count=len(range_findings),
        parse_error=scan.parse_error,
        notes=notes,
    )


def quality_to_dict(quality: SessionQuality) -> dict[str, object]:
    data = asdict(quality)
    data["quality_score"] = quality.quality_score
    data["score_formula"] = quality.score_formula
    return data
