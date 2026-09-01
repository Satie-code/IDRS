"""Running the baseline over real IO-VNBD segments, and measuring what it does.

The honest framing, stated once here and repeated in the report: this module
does not validate the mechanization. Synthetic fixtures do that, because they
are the only place the answer is known. What this module establishes is how the
validated mechanization behaves when the inputs are real — how quickly it
drifts, which segments it refuses, and how much of the apparent error is
actually the residual synchronization Phase 3 measured.

Segments are selected on *stated criteria* (§34: alignment available, timing
usable, enough data), never on how good the result looks. Segments where the
alignment is unavailable are deliberately included, because the failure
behaviour is part of what has to be shown.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from idr.frames.alignment import (
    AlignmentEstimate,
    CalibrationStatus,
    estimate_alignment,
    linear_accel_for_alignment,
)
from idr.frames.conventions import STANDARD_GRAVITY_MPS2
from idr.frames.diagnostics import choose_speed_reference
from idr.frames.orientation import OrientationTrack, estimate_orientation
from idr.frames.sources import SegmentData, find_segments, load_field_mapping, load_segment
from idr.frames.vectors import VectorSeries
from idr.navigation.bias import ImuBias, estimate_bias_from_rest, perturbed
from idr.navigation.geodesy import TangentPlane, origin_from_fixes
from idr.navigation.integration import median_interval
from idr.navigation.mechanization import (
    AlignmentPolicy,
    AttitudeMode,
    MechanizationConfig,
)
from idr.navigation.reference_alignment import (
    LagEstimate,
    ReferenceComparison,
    ReferenceTrack,
    compare,
)
from idr.navigation.state import (
    VELOCITY_FROM_REFERENCE,
    VELOCITY_ZERO_STATIONARY,
    InitialState,
)
from idr.navigation.stationarity import (
    MIN_STATIONARY_WINDOW_S,
    StationarityResult,
    detect_stationarity,
)
from idr.navigation.trajectory import Trajectory, mechanize_series

LOGGER = logging.getLogger(__name__)

#: Fewest samples a segment needs before a trajectory is worth propagating.
#: Below roughly a minute of 10 Hz data the drift has not had time to reveal
#: itself and the segment contributes noise to every aggregate.
MIN_SEGMENT_SAMPLES = 300

#: Perturbations for the bias-sensitivity experiment (§37). Chosen from sensor
#: physics, not from the results: 0.01 rad/s is ~0.6°/s, a typical consumer
#: MEMS run-to-run gyro bias, and 0.05 m/s² is a comparable accelerometer
#: offset. They are applied to whatever bias the segment already has, so the
#: experiment measures a *difference* and cannot be gamed by the baseline.
GYRO_PERTURBATIONS_RPS = (0.001, 0.005, 0.01, 0.02)
ACCEL_PERTURBATIONS_MPS2 = (0.005, 0.02, 0.05, 0.1)


@dataclass
class SegmentInputs:
    """One canonical segment with the Phase 3 chain already run over it."""

    segment: SegmentData
    accel: VectorSeries
    gyro: VectorSeries | None
    orientation: OrientationTrack
    alignment: AlignmentEstimate
    stationarity: StationarityResult
    reference: ReferenceTrack
    median_dt_s: float | None
    origin: TangentPlane | None

    @property
    def segment_id(self) -> str:
        return self.segment.segment_id


@dataclass
class SegmentEvaluation:
    """The baseline result for one segment, with everything needed to explain it."""

    segment_id: str
    session_id: str
    source_file: str
    path: Path
    sample_count: int
    duration_s: float
    alignment_status: CalibrationStatus
    alignment_confidence: float
    alignment_available: bool
    heading_observed: bool
    bias: ImuBias
    trajectory: Trajectory | None
    comparison: ReferenceComparison | None
    stationary_fraction: float
    seconds_per_sample: float
    elapsed_s: float
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.trajectory is not None and self.error is None

    def row(self) -> dict[str, object]:
        """One flat record for the diagnostics CSV."""
        row: dict[str, object] = {
            "segment_id": self.segment_id,
            "session_id": self.session_id,
            "source_file": self.source_file,
            "sample_count": self.sample_count,
            "duration_s": round(self.duration_s, 3),
            "alignment_status": str(self.alignment_status),
            "alignment_confidence": round(self.alignment_confidence, 4),
            "alignment_available": self.alignment_available,
            "heading_observed": self.heading_observed,
            "stationary_fraction": round(self.stationary_fraction, 4),
            "gyro_bias_magnitude_rps": round(self.bias.gyroscope.magnitude_rps, 6),
            "accel_bias_magnitude_mps2": round(self.bias.accelerometer.magnitude_mps2, 6),
            "seconds_per_sample": self.seconds_per_sample,
            "error": self.error or "",
        }
        if self.trajectory is not None:
            row.update(
                {
                    "final_speed_mps": round(float(self.trajectory.speed_mps[-1]), 3),
                    "max_speed_mps": round(float(np.nanmax(self.trajectory.speed_mps)), 3),
                    "path_length_m": round(self.trajectory.distance_travelled_m(), 1),
                    "integrated_steps": self.trajectory.counters.integrated,
                    "skipped_steps": self.trajectory.counters.skipped,
                    "irregular_steps": self.trajectory.counters.irregular,
                }
            )
        comparison = self.comparison
        if comparison is not None:
            zero = comparison.zero_lag.summary()
            best = comparison.best_lag.summary()
            row.update(
                {
                    "lag_s": round(comparison.lag.lag_s, 2),
                    "lag_accepted": comparison.lag.accepted,
                    "zero_lag_speed_rmse_mps": _round(zero["speed_rmse_mps"], 3),
                    "best_lag_speed_rmse_mps": _round(best["speed_rmse_mps"], 3),
                    "zero_lag_position_rmse_m": _round(zero["horizontal_position_rmse_m"], 1),
                    "best_lag_position_rmse_m": _round(best["horizontal_position_rmse_m"], 1),
                    "final_horizontal_error_m": _round(best["final_horizontal_error_m"], 1),
                    "position_drift_mps": _round(comparison.drift_best_lag.position_drift_mps, 4),
                    "yaw_offset_removed_deg": _round(best["yaw_offset_removed_deg"], 2),
                }
            )
        return row


@dataclass
class BiasSensitivityPoint:
    """One perturbation and the divergence it caused."""

    axis: str
    magnitude: float
    final_horizontal_error_m: float | None
    final_speed_error_mps: float | None
    path_length_m: float
    #: Displacement between this trajectory and the unperturbed one, metres.
    divergence_from_baseline_m: float
    #: Which attitude source was active. Gyroscope perturbations are measured
    #: under gyro propagation because the baseline's attitude comes from the
    #: Phase 3 filter, which Phase 4 does not re-run — see :func:`bias_sensitivity`.
    attitude_mode: str = str(AttitudeMode.PHASE3_FILTER)


@dataclass
class BiasSensitivityResult:
    segment_id: str
    duration_s: float
    baseline_final_error_m: float | None
    points: list[BiasSensitivityPoint] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class DiagnosticRun:
    """Every segment Phase 4 processed, plus the aggregate figures."""

    evaluations: list[SegmentEvaluation]
    processed_dir: Path
    started_at: str
    elapsed_s: float
    bias_sensitivity: list[BiasSensitivityResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> list[SegmentEvaluation]:
        return [item for item in self.evaluations if item.succeeded]

    @property
    def with_alignment(self) -> list[SegmentEvaluation]:
        return [item for item in self.evaluations if item.alignment_available]

    @property
    def without_alignment(self) -> list[SegmentEvaluation]:
        return [item for item in self.evaluations if not item.alignment_available]

    def summary(self) -> dict[str, object]:
        compared = [item.comparison for item in self.succeeded if item.comparison is not None]
        zero_rmse = _finite([item.zero_lag.summary()["speed_rmse_mps"] for item in compared])
        best_rmse = _finite([item.best_lag.summary()["speed_rmse_mps"] for item in compared])
        drift = _finite([item.drift_best_lag.position_drift_mps for item in compared])
        lags = [item.lag.lag_s for item in compared if item.lag.accepted]
        return {
            "segments_processed": len(self.evaluations),
            "segments_propagated": len(self.succeeded),
            "segments_with_alignment": len(self.with_alignment),
            "segments_without_alignment": len(self.without_alignment),
            "segments_compared": len(compared),
            "median_zero_lag_speed_rmse_mps": _median(zero_rmse),
            "median_best_lag_speed_rmse_mps": _median(best_rmse),
            "median_position_drift_mps": _median(drift),
            "segments_with_accepted_lag": len(lags),
            "median_abs_lag_s": _median([abs(value) for value in lags]),
            "elapsed_s": round(self.elapsed_s, 2),
        }


def build_reference(segment: SegmentData, origin: TangentPlane | None) -> ReferenceTrack:
    """Assemble every reference signal the segment carries, with provenance.

    Speed comes from Phase 3's chooser, so the tiering and the freshness gate
    are applied in exactly one place. Position comes from GNSS and **only from
    rows Phase 2 marked fresh** — the smartphone position is forward-filled and
    refreshes about every 9 s, so a stale row repeats the previous fix and would
    contribute a stationary reference during motion.
    """
    speed = choose_speed_reference(segment)
    times = segment.frame_data["analysis_time_s"].to_numpy(dtype="float64")
    notes: list[str] = [speed.note]

    position: np.ndarray | None = None
    position_valid: np.ndarray | None = None
    position_source = "none"
    latitude = segment.column("latitude")
    longitude = segment.column("longitude")
    if origin is not None and latitude is not None and longitude is not None:
        fresh = segment.frame_data.get("gnss_position_fresh")
        valid = np.isfinite(latitude) & np.isfinite(longitude)
        if fresh is not None:
            valid &= fresh.fillna(False).to_numpy(dtype=bool)
        if valid.any():
            position = origin.to_enu(latitude, longitude, segment.column("altitude_m"))
            position_valid = valid
            position_source = "gnss_fresh_fixes"
            notes.append(
                f"position reference from {int(valid.sum())} fresh GNSS fixes "
                f"of {valid.size} rows, projected onto {origin.describe()}"
            )

    heading: np.ndarray | None = None
    heading_valid: np.ndarray | None = None
    heading_source = "none"
    for column, label in (("ref_heading_deg", "vehicle"), ("gnss_heading_deg", "gnss")):
        candidate = segment.column(column)
        if candidate is not None and np.isfinite(candidate).any():
            heading = candidate
            heading_valid = np.isfinite(candidate)
            heading_source = f"{column} ({label} course over ground)"
            break

    return ReferenceTrack(
        analysis_time_s=times,
        speed_mps=speed.values,
        speed_valid=speed.valid,
        speed_source=speed.source,
        position_enu_m=position,
        position_valid=position_valid,
        position_source=position_source,
        heading_deg=heading,
        heading_valid=heading_valid,
        heading_source=heading_source,
        origin=origin,
        # IO-VNBD documents no ground truth for any of these quantities, and
        # Phase 2 rates its own best velocity label "high confidence", not truth.
        is_ground_truth=False,
        notes=notes,
    )


def prepare_segment(
    path: Path, *, field_mapping: dict[str, dict[str, bool]] | None = None
) -> SegmentInputs | None:
    """Load a segment and run the Phase 3 chain over it.

    Returns ``None`` when the segment cannot support a mechanization at all —
    no accelerometer, or too few samples. That is a stated selection criterion,
    not a quality judgement about the result.
    """
    segment = load_segment(path, field_mapping=field_mapping)
    accel = segment.series("accel")
    if accel is None or len(accel) < MIN_SEGMENT_SAMPLES:
        return None

    gyro = segment.series("gyro")
    mag = segment.series("mag")
    gravity_channel = segment.series("gravity")

    track = estimate_orientation(
        gyro=gyro,
        accel=accel,
        mag=mag,
        gravity_channel=gravity_channel,
        magnetometer_available=segment.availability.get("has_magnetometer", False),
    )
    speed = choose_speed_reference(segment)
    alignment = estimate_alignment(
        gravity=track.gravity,
        linear_accel=linear_accel_for_alignment(accel, track.gravity),
        speed_mps=speed.values,
        valid_speed=speed.valid,
        gyro=gyro,
        magnetometer=track.magnetometer,
        duration_s=segment.timing.duration_s,
    )
    stationarity = detect_stationarity(accel=accel, gyro=gyro)

    latitude = segment.column("latitude")
    longitude = segment.column("longitude")
    fresh = segment.frame_data.get("gnss_position_fresh")
    origin = (
        origin_from_fixes(
            latitude,
            longitude,
            segment.column("altitude_m"),
            valid=None if fresh is None else fresh.fillna(False).to_numpy(dtype=bool),
        )
        if latitude is not None and longitude is not None
        else None
    )

    return SegmentInputs(
        segment=segment,
        accel=accel,
        gyro=gyro,
        orientation=track,
        alignment=alignment,
        stationarity=stationarity,
        reference=build_reference(segment, origin),
        median_dt_s=median_interval(accel.analysis_time_s),
        origin=origin,
    )


def initial_state_for(inputs: SegmentInputs) -> InitialState:
    """Build the starting state, and record where every component came from.

    **Position.** The first fresh GNSS fix, which is the tangent-plane origin,
    so the trajectory starts at zero by construction. GNSS is used here and
    nowhere else: §14 makes it an initialization value, not a continuous input,
    and nothing in the mechanization can read it.

    **Velocity.** Zero when the opening window is detected stationary — the
    defensible case. Otherwise the reference velocity, read at the *lag-
    corrected* instant rather than at the identically-timestamped row, because
    Phase 3 measured a median residual offset of 4.6 s and picking the row whose
    timestamp matches would import that error straight into the initial state.

    **Attitude.** The Phase 3 orientation at the first sample. When the
    magnetometer was not trusted, its yaw is an arbitrary origin rather than a
    measurement, and ``heading_observed`` records that.
    """
    notes: list[str] = []
    times = inputs.accel.analysis_time_s
    start = float(times[0])

    position = np.zeros(3, dtype="float64")
    position_source = (
        f"first fresh GNSS fix, tangent-plane origin ({inputs.origin.describe()})"
        if inputs.origin is not None
        else "no GNSS fix; the origin is the start of the trajectory"
    )

    stationary_at_start = _begins_at_rest(inputs)
    velocity = np.zeros(3, dtype="float64")
    velocity_source = VELOCITY_ZERO_STATIONARY
    if stationary_at_start:
        notes.append(
            "the opening window is detected stationary, so the initial velocity is zero "
            "on evidence rather than by default"
        )
    else:
        speed = _initial_reference_speed(inputs)
        if speed is None:
            notes.append(
                "the segment does not begin at rest and no trustworthy reference speed "
                "was available at the start, so the initial velocity is zero by "
                "assumption. Every velocity below inherits that error."
            )
        else:
            magnitude, source_note = speed
            # The direction has to come from somewhere. The propagated attitude's
            # forward axis is the only self-consistent choice; using the GNSS
            # course would import a second reference into the initial state.
            heading = _initial_heading(inputs)
            velocity = magnitude * heading
            velocity_source = VELOCITY_FROM_REFERENCE
            notes.append(source_note)

    yaw_observed = inputs.orientation.magnetometer.is_trusted
    if not yaw_observed:
        notes.append(
            "the magnetometer was not trusted, so the initial yaw is an arbitrary "
            "origin. East and north are correct only up to a constant rotation."
        )

    return InitialState(
        analysis_time_s=start,
        position_m=position,
        velocity_mps=velocity,
        orientation=inputs.orientation.quaternions[0].copy(),
        heading_observed=yaw_observed,
        velocity_source=velocity_source,
        position_source=position_source,
        orientation_source="Phase 3 complementary filter at the first sample",
        notes=notes,
    )


#: How soon a detected rest interval must begin for the segment to count as
#: starting from rest, seconds. The detector's window is trailing, so the very
#: first samples carry no verdict at all and requiring rest at sample zero would
#: reject every genuinely stationary start.
START_REST_TOLERANCE_S = 3.0


def _begins_at_rest(inputs: SegmentInputs) -> bool:
    """Whether the segment opens with a defensible stationary interval.

    Asks the detector's own longest run where it starts, rather than testing the
    first N samples: a trailing window needs samples to fill before it can
    report anything, and treating "no verdict yet" as "moving" would make an
    initial velocity of zero unavailable on exactly the segments that begin
    parked.
    """
    stationarity = inputs.stationarity
    span = stationarity.longest_span
    if span is None or stationarity.longest_interval_s < MIN_STATIONARY_WINDOW_S:
        return False
    times = inputs.accel.analysis_time_s
    start_index = min(span[0], times.size - 1)
    return bool(times[start_index] - times[0] <= START_REST_TOLERANCE_S)


def _initial_reference_speed(inputs: SegmentInputs) -> tuple[float, str] | None:
    """The reference speed at the start, read at the lag-corrected instant."""
    reference = inputs.reference
    if reference.speed_mps is None:
        return None
    valid = (
        np.isfinite(reference.speed_mps)
        if reference.speed_valid is None
        else np.isfinite(reference.speed_mps) & reference.speed_valid
    )
    if not valid.any():
        return None
    start = float(inputs.accel.analysis_time_s[0])
    lag = inputs.alignment.quality.forward_lag_s or 0.0
    target = start + lag
    stamps = reference.analysis_time_s[valid]
    values = reference.speed_mps[valid]
    index = int(np.argmin(np.abs(stamps - target)))
    if abs(float(stamps[index]) - target) > 1.0:
        return None
    note = (
        f"initial speed {float(values[index]):.2f} m/s from {reference.speed_source}, "
        f"read at t{lag:+.1f}s to undo the residual synchronization offset Phase 3 "
        "measured for this segment"
    )
    return float(values[index]), note


def _initial_heading(inputs: SegmentInputs) -> np.ndarray:
    """A unit horizontal direction from the initial attitude's forward axis.

    Uses the alignment when there is one, because that is what defines "forward"
    for the vehicle; falls back to the phone's projected X axis, which is
    arbitrary but at least consistent with the attitude the run starts from.
    """
    from idr.frames import quaternion as quat

    rotation_nav_phone = quat.to_rotation_matrix(inputs.orientation.quaternions[0])
    if inputs.alignment.rotation is not None:
        forward_phone = inputs.alignment.rotation.matrix[0]
    else:
        forward_phone = np.array([1.0, 0.0, 0.0])
    direction = rotation_nav_phone @ forward_phone
    direction[2] = 0.0
    magnitude = float(np.linalg.norm(direction))
    if magnitude < 1e-9:
        return np.array([1.0, 0.0, 0.0])
    return direction / magnitude


def evaluate_segment(
    inputs: SegmentInputs,
    *,
    estimate_bias: bool = True,
    attitude_mode: AttitudeMode = AttitudeMode.PHASE3_FILTER,
    bias_override: ImuBias | None = None,
    lag_estimate: LagEstimate | None = None,
) -> SegmentEvaluation:
    """Propagate one segment and compare it against the reference.

    ``lag_estimate`` reuses a residual lag measured elsewhere instead of
    searching for one. The bias-sensitivity experiment passes the unperturbed
    run's lag, and that is a correctness point rather than a speed one: the
    residual synchronization offset is a property of the *data*, so letting each
    perturbed run re-search would allow the lag to absorb part of the effect the
    experiment is trying to measure. An ablation holds everything else fixed.
    """
    started = time.perf_counter()
    alignment_available = inputs.alignment.rotation is not None

    bias = bias_override or ImuBias.zero()
    if bias_override is None and estimate_bias:
        window = inputs.stationarity.longest_window_mask()
        if window.any():
            bias = estimate_bias_from_rest(
                gyro=inputs.gyro,
                accel=inputs.accel,
                mask=window,
                gravity=inputs.orientation.gravity,
                expected_gravity_mps2=STANDARD_GRAVITY_MPS2,
            )

    config = MechanizationConfig(
        attitude_mode=attitude_mode,
        # A segment without an alignment is still propagated, as a phone-frame
        # diagnostic. §34 asks for the failure behaviour to be shown, and
        # showing it means running it, not skipping it.
        alignment_policy=(
            AlignmentPolicy.REQUIRE
            if alignment_available
            else AlignmentPolicy.PHONE_FRAME_DIAGNOSTIC
        ),
        bias=bias,
        median_dt_s=inputs.median_dt_s,
    )

    initial = initial_state_for(inputs)
    error: str | None = None
    trajectory: Trajectory | None = None
    comparison: ReferenceComparison | None = None
    try:
        trajectory = mechanize_series(
            initial=initial,
            specific_force=inputs.accel,
            angular_rate=inputs.gyro,
            orientations=inputs.orientation.quaternions,
            alignment=inputs.alignment if alignment_available else None,
            config=config,
            stationary_mask=inputs.stationarity.mask,
            origin=inputs.origin,
        )
        comparison = compare(trajectory, inputs.reference, lag_estimate=lag_estimate)
    except Exception as failure:  # noqa: BLE001 - recorded, never swallowed silently
        error = f"{type(failure).__name__}: {failure}"
        LOGGER.warning("segment %s could not be propagated: %s", inputs.segment_id, error)

    elapsed = time.perf_counter() - started
    return SegmentEvaluation(
        segment_id=inputs.segment_id,
        session_id=inputs.segment.session_id,
        source_file=inputs.segment.source_file,
        path=inputs.segment.path,
        sample_count=len(inputs.accel),
        duration_s=inputs.segment.timing.duration_s,
        alignment_status=inputs.alignment.status,
        alignment_confidence=inputs.alignment.confidence,
        alignment_available=alignment_available,
        heading_observed=initial.heading_observed,
        bias=bias,
        trajectory=trajectory,
        comparison=comparison,
        stationary_fraction=inputs.stationarity.fraction,
        seconds_per_sample=elapsed / max(1, len(inputs.accel)),
        elapsed_s=elapsed,
        error=error,
        notes=[*initial.notes, *(trajectory.notes if trajectory else [])],
    )


def bias_sensitivity(
    inputs: SegmentInputs,
    *,
    gyro_magnitudes: tuple[float, ...] = GYRO_PERTURBATIONS_RPS,
    accel_magnitudes: tuple[float, ...] = ACCEL_PERTURBATIONS_MPS2,
) -> BiasSensitivityResult:
    """Perturb the bias and measure how far the trajectory moves (§37).

    The perturbations are fixed engineering constants taken from sensor
    datasheet behaviour, declared in :data:`GYRO_PERTURBATIONS_RPS` and
    :data:`ACCEL_PERTURBATIONS_MPS2`, and are *not* chosen to produce a
    particular picture. Each is applied along a single phone axis so the effect
    is attributable.

    **The two sensors are measured under different attitude modes, and the
    reason is structural rather than a convenience.** In the baseline's
    ``PHASE3_FILTER`` mode the attitude is the Phase 3 filter's output, computed
    upstream from the unperturbed gyroscope; the Phase 4 bias is applied to the
    angular rate *after* that, where nothing consumes it. A gyroscope
    perturbation therefore has exactly zero effect on the baseline trajectory —
    which is true, and reporting it as "gyroscope bias does not matter" would be
    badly misleading, because a gyroscope bias matters enormously wherever the
    gyroscope actually drives attitude. So gyroscope perturbations are measured
    under ``GYRO_PROPAGATION``, against a baseline in that same mode, and each
    point records which mode produced it.

    ``divergence_from_baseline_m`` is measured against the unperturbed run in
    the matching mode rather than against the reference. That makes the
    experiment independent of how good the baseline itself was, which is the
    point: it isolates sensitivity to bias from every other error source. The
    residual lag is held at the unperturbed run's value throughout, so the
    comparison cannot absorb part of the effect being measured.
    """
    baseline = evaluate_segment(inputs)
    if baseline.trajectory is None:
        return BiasSensitivityResult(
            segment_id=inputs.segment_id,
            duration_s=inputs.segment.timing.duration_s,
            baseline_final_error_m=None,
            notes=[f"the unperturbed run failed: {baseline.error}"],
        )

    baseline_error = (
        baseline.comparison.best_lag.summary()["final_horizontal_error_m"]
        if baseline.comparison is not None
        else None
    )
    held_lag = baseline.comparison.lag if baseline.comparison is not None else None

    propagated_baseline = evaluate_segment(
        inputs, attitude_mode=AttitudeMode.GYRO_PROPAGATION, lag_estimate=held_lag
    )
    points: list[BiasSensitivityPoint] = []
    notes = [
        "perturbations are fixed constants from sensor behaviour, not tuned; "
        "divergence is measured against the unperturbed run, not the reference",
        "gyroscope perturbations are measured under gyro propagation, because the "
        "baseline takes its attitude from the Phase 3 filter and is structurally "
        "insensitive to a gyroscope bias applied downstream of it",
    ]

    for axis_index, axis_name in enumerate("xyz"):
        for magnitude in gyro_magnitudes:
            delta = np.zeros(3)
            delta[axis_index] = magnitude
            points.append(
                _sensitivity_point(
                    inputs,
                    propagated_baseline,
                    f"gyro_{axis_name}",
                    magnitude,
                    gyro_rps=delta,
                    lag_estimate=held_lag,
                    attitude_mode=AttitudeMode.GYRO_PROPAGATION,
                )
            )
        for magnitude in accel_magnitudes:
            delta = np.zeros(3)
            delta[axis_index] = magnitude
            points.append(
                _sensitivity_point(
                    inputs,
                    baseline,
                    f"accel_{axis_name}",
                    magnitude,
                    accel_mps2=delta,
                    lag_estimate=held_lag,
                )
            )

    return BiasSensitivityResult(
        segment_id=inputs.segment_id,
        duration_s=baseline.trajectory.duration_s,
        baseline_final_error_m=baseline_error,
        points=points,
        notes=notes,
    )


def _sensitivity_point(
    inputs: SegmentInputs,
    baseline: SegmentEvaluation,
    axis: str,
    magnitude: float,
    *,
    gyro_rps: np.ndarray | None = None,
    accel_mps2: np.ndarray | None = None,
    lag_estimate: LagEstimate | None = None,
    attitude_mode: AttitudeMode = AttitudeMode.PHASE3_FILTER,
) -> BiasSensitivityPoint:
    if baseline.trajectory is None:
        return BiasSensitivityPoint(
            axis, magnitude, None, None, 0.0, float("nan"), str(attitude_mode)
        )
    disturbed = evaluate_segment(
        inputs,
        estimate_bias=False,
        attitude_mode=attitude_mode,
        bias_override=perturbed(baseline.bias, gyro_rps=gyro_rps, accel_mps2=accel_mps2),
        lag_estimate=lag_estimate,
    )
    if disturbed.trajectory is None:
        return BiasSensitivityPoint(
            axis, magnitude, None, None, 0.0, float("nan"), str(attitude_mode)
        )

    count = min(len(baseline.trajectory), len(disturbed.trajectory))
    divergence = float(
        np.linalg.norm(
            disturbed.trajectory.position_m[count - 1] - baseline.trajectory.position_m[count - 1]
        )
    )
    summary = disturbed.comparison.best_lag.summary() if disturbed.comparison else {}
    return BiasSensitivityPoint(
        axis=axis,
        magnitude=magnitude,
        final_horizontal_error_m=_as_float(summary.get("final_horizontal_error_m")),
        final_speed_error_mps=_as_float(summary.get("speed_rmse_mps")),
        path_length_m=disturbed.trajectory.distance_travelled_m(),
        divergence_from_baseline_m=divergence,
        attitude_mode=str(attitude_mode),
    )


def run_diagnostics(
    processed_dir: Path,
    metadata_dir: Path | None = None,
    *,
    limit: int | None = None,
    sensitivity_segments: int = 3,
    progress_every: int = 25,
) -> DiagnosticRun:
    """Propagate every usable smartphone segment and collect the results."""
    started = time.perf_counter()
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    field_mapping = load_field_mapping(metadata_dir) if metadata_dir else None
    paths = find_segments(processed_dir)
    if limit is not None:
        paths = paths[:limit]

    evaluations: list[SegmentEvaluation] = []
    prepared: list[SegmentInputs] = []
    skipped = 0
    for index, path in enumerate(paths, start=1):
        inputs = prepare_segment(path, field_mapping=field_mapping)
        if inputs is None:
            skipped += 1
            continue
        evaluations.append(evaluate_segment(inputs))
        prepared.append(inputs)
        if progress_every and index % progress_every == 0:
            LOGGER.info("Propagated %d/%d segment(s)", index, len(paths))

    # Sensitivity on segments of *typical* duration, not the longest. The
    # effect of a bias grows quadratically with time, so this dataset's
    # three-hour outliers would yield divergences in the hundreds of kilometres
    # — arithmetically correct and useless as a description of what a bias costs
    # on an ordinary drive. The rule is stated before any result is seen: the
    # segments whose duration is closest to the median of the successfully
    # propagated, aligned set.
    usable = [
        (inputs, evaluation)
        for inputs, evaluation in zip(prepared, evaluations, strict=True)
        if evaluation.succeeded and evaluation.alignment_available
    ]
    sensitivity: list[BiasSensitivityResult] = []
    if usable and sensitivity_segments > 0:
        median_duration = float(np.median([item[1].duration_s for item in usable]))
        candidates = sorted(usable, key=lambda pair: abs(pair[1].duration_s - median_duration))
        sensitivity = [bias_sensitivity(inputs) for inputs, _ in candidates[:sensitivity_segments]]
        notes_extra = (
            f"bias sensitivity was measured on the {len(sensitivity)} segment(s) closest "
            f"to the median duration of {median_duration:.0f}s, not the longest"
        )
    else:
        notes_extra = ""

    notes = []
    if notes_extra:
        notes.append(notes_extra)
    if skipped:
        notes.append(
            f"{skipped} segment(s) were not propagated: no accelerometer channel, or "
            f"fewer than {MIN_SEGMENT_SAMPLES} samples"
        )
    return DiagnosticRun(
        evaluations=evaluations,
        processed_dir=processed_dir,
        started_at=stamp,
        elapsed_s=time.perf_counter() - started,
        bias_sensitivity=sensitivity,
        notes=notes,
    )


def performance_summary(run: DiagnosticRun) -> dict[str, float]:
    """Per-sample processing cost. A baseline to improve on, not a result."""
    per_sample = np.array(
        [item.seconds_per_sample for item in run.evaluations if item.seconds_per_sample > 0.0]
    )
    if per_sample.size == 0:
        return {"segments": 0.0}
    per_segment = np.array([item.elapsed_s for item in run.evaluations])
    return {
        "mean_us_per_sample": float(per_sample.mean() * 1e6),
        "p95_us_per_sample": float(np.percentile(per_sample, 95) * 1e6),
        "mean_s_per_segment": float(per_segment.mean()),
        "p95_s_per_segment": float(np.percentile(per_segment, 95)),
        "segments": float(per_sample.size),
    }


def _round(value: object, digits: int) -> object:
    if isinstance(value, float) and math.isfinite(value):
        return round(value, digits)
    return "" if value is None else value


def _as_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _finite(values: list[object]) -> list[float]:
    return [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None
