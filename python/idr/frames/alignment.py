"""Phone→vehicle alignment estimation.

This is the central Phase 3 feature: recovering ``R_vehicle_phone`` when the
phone is mounted at an unknown attitude.

The problem splits cleanly into two halves that are **not** equally solvable,
and keeping them separate is the whole design:

* **Tilt (roll and pitch) — 2 degrees of freedom.** Gravity gives this
  directly. A phone at rest tells you which way is down, and that fixes the
  vehicle's vertical axis in phone coordinates. Cheap, robust, and available
  while stationary.

* **Heading (yaw) — 1 degree of freedom.** Gravity says nothing about it. A
  rotation of the phone about the vertical axis leaves every gravity reading
  identical. Yaw therefore requires *motion*, and if the vehicle has not moved
  there is no honest answer.

The temptation is to fill yaw in from the magnetometer and return a complete
rotation. This module refuses to: a magnetometer inside a car body is measuring
the car, and a confidently-wrong 30° yaw error is worse than an admitted
unknown, because everything downstream would inherit it silently. When yaw
cannot be resolved, :attr:`AlignmentEstimate.rotation` is ``None`` and only the
tilt is offered.

**On the IO-VNBD mounting.** The dataset was collected with the phone
reportedly placed in a fixed vehicle position, and it would be convenient to
assume ``R_vehicle_phone`` is close to identity. Phase 3 does not assume it —
the deployed system must handle a phone in a cupholder or a pocket, and an
assumption that holds only for the training corpus is exactly the kind that
survives validation and fails in the field. The alignment is estimated from the
data every time, and how close it lands to identity is a *finding*, not an
input.

**The forward-direction estimator.** Under longitudinal acceleration, the
vehicle's specific force in the horizontal plane points along ±forward. So the
direction along which horizontal acceleration co-varies with the rate of change
of speed is the forward axis, and the sign of that covariance disambiguates
forward from reverse. Formally, with ``h_i`` the horizontal linear acceleration
and ``s_i = dv/dt`` from an independent speed reference::

    f ∝ Σ h_i · s_i

which is the least-squares direction, and it is only meaningful when the
``s_i`` actually vary — hence ``forward_direction_confidence`` rather than a
bare direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from idr.frames import rotations
from idr.frames.conventions import MIN_VECTOR_NORM, Frame
from idr.frames.gravity import GravityEstimate, linear_acceleration
from idr.frames.magnetometer import MagnetometerQuality
from idr.frames.rotations import FrameRotation
from idr.frames.vectors import Vector3, VectorSeries

#: Minimum |dv/dt| for a sample to count as a longitudinal-dynamics event,
#: m/s². 0.4 m/s² is gentle — well inside normal driving — but comfortably
#: above the noise of a speed signal differenced over ~0.1 s.
MIN_LONGITUDINAL_ACCEL_MPS2 = 0.4

#: Minimum speed for a sample to inform heading, m/s. Below ~2 m/s a GNSS
#: course is dominated by position noise and a vehicle may be manoeuvring
#: rather than travelling forward.
MIN_HEADING_SPEED_MPS = 2.0

#: Minimum number of qualifying dynamic samples before yaw is claimed at all.
MIN_FORWARD_SAMPLES = 50

#: Correlation between predicted and observed longitudinal acceleration below
#: which the forward direction is not accepted.
MIN_FORWARD_CORRELATION = 0.3

#: Agreement (radians) required between forward directions estimated from
#: disjoint halves of the data. 0.26 rad ≈ 15°: two independent halves of the
#: same drive should agree far better than this if the estimate means anything.
MAX_SPLIT_HALF_DISAGREEMENT_RAD = 0.26

#: Trailing-average window applied to both the horizontal acceleration and the
#: speed derivative before they are correlated, in seconds.
#:
#: A vehicle's longitudinal acceleration is a sub-Hz quantity. A phone's
#: accelerometer at 10 Hz is dominated by road vibration and mount rattle: on
#: real IO-VNBD segments the horizontal specific force has a median magnitude
#: of about 1.2 m/s² while the vehicle's actual dv/dt has a median of 0.37
#: m/s². Correlating the two raw gives essentially nothing (measured: 0.02).
#: Averaging both over 2 s keeps the band the vehicle occupies and discards the
#: band it does not. Both signals get the *same* window, so the smoothing adds
#: no relative delay.
SMOOTHING_WINDOW_S = 2.0

#: Half-width of the lag search between the acceleration and the speed
#: reference, in seconds.
#:
#: Phase 2 measured a per-pair smartphone/vehicle offset but deliberately left
#: both streams untouched, and its own contract says not to assume the
#: synchronization is perfect. It is not: across 16 real segments the residual
#: lag between the phone's accelerometer and the interpolated reference
#: velocity has a median magnitude of 5.2 s, clustered around +4 to +7 s, and
#: correcting it lifts the median correlation from 0.02 to 0.73. So the lag is
#: estimated here rather than assumed to be zero. 15 s covers the observed
#: spread with margin.
MAX_LAG_SEARCH_S = 15.0

#: Step of the lag search, in seconds. Finer than the residual matters, coarser
#: than the smoothing window makes meaningful.
LAG_SEARCH_STEP_S = 0.2

#: How far the best lag's correlation must exceed the median across the search
#: for the peak to count as a peak rather than the argmax of a flat curve.
MIN_LAG_PEAK_MARGIN = 0.1

#: Angle from vehicle vertical that the transformed gravity may sit at before
#: the alignment is called inconsistent, in degrees.
MAX_GRAVITY_RESIDUAL_DEG = 10.0


class CalibrationStatus(StrEnum):
    """Outcome of an alignment attempt.

    Every failure state is distinct because the operator response differs:
    "drive in a straight line" is the fix for INSUFFICIENT_MOTION and useless
    for MAG_DISTURBED.
    """

    NOT_STARTED = "NOT_STARTED"
    COLLECTING = "COLLECTING"
    #: The vehicle never accelerated enough to reveal a forward direction.
    INSUFFICIENT_MOTION = "INSUFFICIENT_MOTION"
    #: Gravity could not be pinned down, so even tilt is unavailable.
    LOW_GRAVITY_CONFIDENCE = "LOW_GRAVITY_CONFIDENCE"
    #: Tilt is solid, heading is not. Roll and pitch are returned; yaw is not.
    LOW_HEADING_CONFIDENCE = "LOW_HEADING_CONFIDENCE"
    #: The magnetometer was the only heading candidate and it is untrustworthy.
    MAG_DISTURBED = "MAG_DISTURBED"
    #: Full 3-DOF alignment with acceptable confidence.
    SUCCESS = "SUCCESS"
    #: Something structural: no usable samples, degenerate geometry.
    FAILED = "FAILED"

    @property
    def is_terminal(self) -> bool:
        return self not in (CalibrationStatus.NOT_STARTED, CalibrationStatus.COLLECTING)

    @property
    def has_tilt(self) -> bool:
        return self in (
            CalibrationStatus.SUCCESS,
            CalibrationStatus.LOW_HEADING_CONFIDENCE,
            CalibrationStatus.INSUFFICIENT_MOTION,
            CalibrationStatus.MAG_DISTURBED,
        )


@dataclass
class ForwardDirectionEstimate:
    """The vehicle's forward axis in phone coordinates, with its evidence."""

    direction: Vector3 | None
    confidence: float
    correlation: float | None
    dynamic_sample_count: int
    candidate_count: int
    split_half_disagreement_rad: float | None
    reversed_fraction: float | None
    method: str
    #: Lag applied to the speed reference to align it with the accelerometer.
    #: Positive means the reference had to be shifted later in time. Non-zero
    #: values are a measurement of residual Phase 2 synchronization error, not
    #: a correction applied to any stored data.
    estimated_lag_s: float = 0.0
    #: How far the best lag's correlation stood above the median of the search.
    lag_peak_margin: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def is_available(self) -> bool:
        return self.direction is not None


@dataclass
class AlignmentQuality:
    """The individual diagnostics behind an alignment, kept separate.

    Deliberately not collapsed into one number. A caller that needs a scalar
    has :attr:`AlignmentEstimate.confidence`; a caller that needs to know *why*
    the confidence is low needs these.
    """

    gravity_confidence: float
    gravity_residual_deg: float | None
    forward_confidence: float
    forward_correlation: float | None
    longitudinal_correlation: float | None
    lateral_correlation: float | None
    angular_rate_consistency: float | None
    magnetometer_weight: float
    dynamic_sample_count: int
    total_sample_count: int
    duration_s: float
    #: Residual lag the forward-direction estimator had to remove between the
    #: accelerometer and the speed reference. A measurement of leftover Phase 2
    #: synchronization error, not a correction written anywhere.
    forward_lag_s: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "gravity_confidence": self.gravity_confidence,
            "gravity_residual_deg": self.gravity_residual_deg,
            "forward_confidence": self.forward_confidence,
            "forward_correlation": self.forward_correlation,
            "longitudinal_correlation": self.longitudinal_correlation,
            "lateral_correlation": self.lateral_correlation,
            "angular_rate_consistency": self.angular_rate_consistency,
            "magnetometer_weight": self.magnetometer_weight,
            "dynamic_sample_count": float(self.dynamic_sample_count),
            "total_sample_count": float(self.total_sample_count),
            "duration_s": self.duration_s,
            "forward_lag_s": self.forward_lag_s,
        }


@dataclass
class AlignmentEstimate:
    """Result of a phone→vehicle alignment attempt.

    ``rotation`` is ``R_vehicle_phone`` and is **None whenever yaw could not be
    resolved**. ``tilt_only_rotation`` is always available when gravity was,
    but its yaw is a placeholder chosen for reproducibility, not a measurement
    — check :attr:`yaw_resolved` before using it as a full alignment.
    """

    status: CalibrationStatus
    rotation: FrameRotation | None
    tilt_only_rotation: FrameRotation | None
    up_direction_phone: Vector3 | None
    forward_direction_phone: Vector3 | None
    roll_deg: float | None
    pitch_deg: float | None
    yaw_deg: float | None
    yaw_resolved: bool
    confidence: float
    quality: AlignmentQuality
    sample_count: int
    duration_s: float
    notes: list[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        """Whether a full 3-DOF vehicle-frame transform can be applied."""
        return self.rotation is not None and self.status is CalibrationStatus.SUCCESS

    def to_vehicle_frame(self, series: VectorSeries) -> VectorSeries:
        """Rotate a phone-frame series into the vehicle frame.

        Refuses when yaw is unresolved rather than applying the placeholder,
        because the caller's data would then be silently rotated by an
        arbitrary heading.
        """
        if self.rotation is None:
            raise ValueError(
                f"no full alignment is available (status {self.status}); "
                "yaw was never resolved, so phone-frame data cannot be expressed "
                "in the vehicle frame"
            )
        return self.rotation.apply_series(series)


def _unit(vector: np.ndarray) -> np.ndarray | None:
    magnitude = float(np.linalg.norm(vector))
    if not math.isfinite(magnitude) or magnitude < MIN_VECTOR_NORM:
        return None
    return vector / magnitude


def _time_smooth(values: np.ndarray, times: np.ndarray, window_s: float) -> np.ndarray:
    """Trailing moving average over a *time* window, not a sample count.

    A fixed sample count would mean different things at 2 Hz and 1000 Hz, and
    Phase 2 found both in this dataset. Non-finite entries are skipped rather
    than propagated, so one bad sample does not blank a whole window.
    """
    array = np.asarray(values, dtype="float64")
    flat = array.reshape(array.shape[0], -1)
    finite = np.isfinite(flat)
    filled = np.where(finite, flat, 0.0)

    totals = np.vstack([np.zeros((1, flat.shape[1])), np.cumsum(filled, axis=0)])
    counts = np.vstack([np.zeros((1, flat.shape[1])), np.cumsum(finite, axis=0)])
    start = np.searchsorted(times, times - window_s, side="left")
    stop = np.arange(1, flat.shape[0] + 1)

    total = totals[stop] - totals[start]
    count = counts[stop] - counts[start]
    smoothed = np.where(count > 0, total / np.maximum(count, 1.0), np.nan)
    return smoothed.reshape(array.shape)


def _shift_signal(values: np.ndarray, times: np.ndarray, lag_s: float) -> np.ndarray:
    """Resample ``values`` onto ``times + lag_s``.

    Interpolation on the actual timestamps, not ``np.roll``: rolling by a
    sample count assumes a uniform rate, which this dataset does not have.
    Samples that fall outside the original span become NaN rather than being
    clamped to the endpoints, so the edges do not contribute invented data.
    """
    if lag_s == 0.0:
        return values
    finite = np.isfinite(values) & np.isfinite(times)
    if int(finite.sum()) < 2:
        return np.full_like(values, np.nan)
    shifted = np.interp(times + lag_s, times[finite], values[finite], left=np.nan, right=np.nan)
    return shifted


def _speed_derivative(speed: np.ndarray, times: np.ndarray) -> np.ndarray:
    """dv/dt from an irregularly sampled speed signal.

    Central differences over *actual* timestamps, not a fixed rate. Steps with
    non-positive Δt (Phase 2 retains duplicate timestamps rather than dropping
    them) yield NaN rather than a division by zero.
    """
    derivative = np.full(speed.size, np.nan, dtype="float64")
    if speed.size < 3:
        return derivative
    dt = times[2:] - times[:-2]
    usable = np.isfinite(dt) & (dt > 0.0)
    delta = speed[2:] - speed[:-2]
    derivative[1:-1] = np.where(usable, delta / np.where(usable, dt, 1.0), np.nan)
    return derivative


def _direction_at_lag(
    horizontal: np.ndarray,
    accel_rate: np.ndarray,
    times: np.ndarray,
    base_valid: np.ndarray,
    up: np.ndarray,
    lag_s: float,
) -> tuple[np.ndarray | None, float | None, np.ndarray]:
    """Best-fit forward direction and its correlation at one candidate lag."""
    shifted = _shift_signal(accel_rate, times, lag_s)
    mask = base_valid & np.isfinite(shifted) & (np.abs(shifted) >= MIN_LONGITUDINAL_ACCEL_MPS2)
    if int(mask.sum()) < MIN_FORWARD_SAMPLES:
        return None, None, mask

    selected_h = horizontal[mask]
    selected_s = shifted[mask]
    direction = _unit(selected_h.T @ selected_s)
    if direction is None:
        return None, None, mask
    direction = _unit(direction - float(direction @ up) * up)
    if direction is None:
        return None, None, mask
    return direction, _correlation(selected_h @ direction, selected_s), mask


def _search_lag(
    *,
    horizontal: np.ndarray,
    accel_rate: np.ndarray,
    times: np.ndarray,
    base_valid: np.ndarray,
    up: np.ndarray,
    max_lag_s: float,
) -> tuple[float, np.ndarray | None, float | None, np.ndarray, float | None, list[str]]:
    """Find the lag that best aligns the speed reference with the accelerometer.

    Phase 2 left both streams as recorded and warns against assuming its
    synchronization is exact, so the residual is measured here. The peak must
    also *be* a peak: if the correlation curve is flat, its argmax carries no
    information and the zero-lag result is used instead, with a note.
    """
    notes: list[str] = []
    zero_direction, zero_correlation, zero_mask = _direction_at_lag(
        horizontal, accel_rate, times, base_valid, up, 0.0
    )
    if max_lag_s <= 0.0:
        return 0.0, zero_direction, zero_correlation, zero_mask, None, notes

    steps = int(max_lag_s / LAG_SEARCH_STEP_S)
    candidates = [index * LAG_SEARCH_STEP_S for index in range(-steps, steps + 1)]
    scored: list[tuple[float, float, np.ndarray | None, np.ndarray]] = []
    for lag in candidates:
        direction, correlation, mask = _direction_at_lag(
            horizontal, accel_rate, times, base_valid, up, lag
        )
        if correlation is not None and math.isfinite(correlation):
            scored.append((lag, correlation, direction, mask))

    if not scored:
        return 0.0, zero_direction, zero_correlation, zero_mask, None, notes

    best_lag, best_correlation, best_direction, best_mask = max(scored, key=lambda item: item[1])
    median_correlation = float(np.median([correlation for _, correlation, _, _ in scored]))
    margin = best_correlation - median_correlation

    if margin < MIN_LAG_PEAK_MARGIN:
        notes.append(
            f"the lag search found no distinct peak (best {best_correlation:.2f} versus a "
            f"median of {median_correlation:.2f} across ±{max_lag_s:.0f}s); "
            "no lag was applied"
        )
        return 0.0, zero_direction, zero_correlation, zero_mask, margin, notes

    if abs(best_lag) >= LAG_SEARCH_STEP_S:
        notes.append(
            f"the speed reference had to be shifted by {best_lag:+.1f}s to line up with "
            f"the accelerometer, lifting the correlation to {best_correlation:.2f}. "
            "This is residual Phase 2 synchronization error, measured here and not "
            "written back to any stored data."
        )
    if abs(best_lag) > max_lag_s - LAG_SEARCH_STEP_S:
        notes.append(
            f"the best lag sits at the edge of the ±{max_lag_s:.0f}s search window, so the "
            "true offset may be larger; treat the heading as unconfirmed"
        )
    return best_lag, best_direction, best_correlation, best_mask, margin, notes


def estimate_forward_direction(
    *,
    linear_accel: VectorSeries,
    up_direction: Vector3,
    speed_mps: np.ndarray,
    valid_speed: np.ndarray | None = None,
    max_lag_s: float = MAX_LAG_SEARCH_S,
) -> ForwardDirectionEstimate:
    """Recover the vehicle's forward axis from longitudinal dynamics.

    ``speed_mps`` is an independent scalar speed reference — the Phase 2
    reference velocity where a synchronized vehicle stream exists, or GNSS
    speed on rows Phase 2 marked fresh. ``valid_speed`` masks rows where that
    reference is not trustworthy; without it every finite row is used.
    """
    notes: list[str] = []
    times = linear_accel.analysis_time_s
    count = len(linear_accel)
    up = up_direction.to_array()

    if count < MIN_FORWARD_SAMPLES:
        return ForwardDirectionEstimate(
            direction=None,
            confidence=0.0,
            correlation=None,
            dynamic_sample_count=0,
            candidate_count=count,
            split_half_disagreement_rad=None,
            reversed_fraction=None,
            method="longitudinal_covariance",
            notes=[f"only {count} sample(s); need at least {MIN_FORWARD_SAMPLES}"],
        )

    speed = np.asarray(speed_mps, dtype="float64")
    raw_rate = _speed_derivative(speed, times)

    # Horizontal component of the linear acceleration: remove whatever lies
    # along the vertical, because that axis is already fixed by gravity and
    # letting it contribute would tilt the recovered forward axis.
    horizontal = linear_accel.samples - np.outer(linear_accel.samples @ up, up)

    # Low-pass both sides into the band the vehicle actually occupies. Without
    # this, phone vibration dominates and the covariance is noise.
    horizontal = _time_smooth(horizontal, times, SMOOTHING_WINDOW_S)
    accel_rate = _time_smooth(raw_rate, times, SMOOTHING_WINDOW_S)

    base_valid = np.isfinite(horizontal).all(axis=1) & np.isfinite(speed)
    if valid_speed is not None:
        base_valid &= np.asarray(valid_speed, dtype=bool)
    base_valid &= speed >= MIN_HEADING_SPEED_MPS

    lag_s, direction, correlation, dynamic, margin, lag_notes = _search_lag(
        horizontal=horizontal,
        accel_rate=accel_rate,
        times=times,
        base_valid=base_valid,
        up=up,
        max_lag_s=max_lag_s,
    )
    notes.extend(lag_notes)

    dynamic_count = int(dynamic.sum())
    if dynamic_count < MIN_FORWARD_SAMPLES:
        return ForwardDirectionEstimate(
            direction=None,
            confidence=0.0,
            correlation=None,
            dynamic_sample_count=dynamic_count,
            candidate_count=count,
            split_half_disagreement_rad=None,
            reversed_fraction=None,
            method="longitudinal_covariance",
            estimated_lag_s=lag_s,
            lag_peak_margin=margin,
            notes=[
                *notes,
                f"{dynamic_count} sample(s) show longitudinal acceleration above "
                f"{MIN_LONGITUDINAL_ACCEL_MPS2} m/s² while moving above "
                f"{MIN_HEADING_SPEED_MPS} m/s; need {MIN_FORWARD_SAMPLES}. "
                "The vehicle never accelerated decisively enough to reveal its heading.",
            ],
        )

    shifted_rate = _shift_signal(accel_rate, times, lag_s)
    selected_h = horizontal[dynamic]
    selected_s = shifted_rate[dynamic]
    if direction is None:
        return ForwardDirectionEstimate(
            direction=None,
            confidence=0.0,
            correlation=None,
            dynamic_sample_count=dynamic_count,
            candidate_count=count,
            split_half_disagreement_rad=None,
            reversed_fraction=None,
            method="longitudinal_covariance",
            estimated_lag_s=lag_s,
            lag_peak_margin=margin,
            notes=[*notes, "acceleration events cancelled out; no consistent forward axis"],
        )
    # Re-project onto the horizontal plane: the covariance sum can pick up a
    # small vertical component from noise, and forward must be horizontal.
    projected = _unit(direction - float(direction @ up) * up)
    if projected is None:
        return ForwardDirectionEstimate(
            direction=None,
            confidence=0.0,
            correlation=None,
            dynamic_sample_count=dynamic_count,
            candidate_count=count,
            split_half_disagreement_rad=None,
            reversed_fraction=None,
            method="longitudinal_covariance",
            estimated_lag_s=lag_s,
            lag_peak_margin=margin,
            notes=[*notes, "recovered direction is vertical; it carries no heading"],
        )
    direction = projected

    predicted = selected_h @ direction
    correlation = _correlation(predicted, selected_s)
    reversed_fraction = float(np.mean(np.sign(predicted) != np.sign(selected_s)))

    # Split-half check: two disjoint halves of the same drive must agree, or
    # the estimate is fitting whatever each half happened to contain.
    half = dynamic_count // 2
    disagreement: float | None = None
    if half >= MIN_FORWARD_SAMPLES // 2:
        first = _unit(selected_h[:half].T @ selected_s[:half])
        second = _unit(selected_h[half:].T @ selected_s[half:])
        if first is not None and second is not None:
            disagreement = float(np.arccos(np.clip(first @ second, -1.0, 1.0)))
            if disagreement > MAX_SPLIT_HALF_DISAGREEMENT_RAD:
                notes.append(
                    f"the two halves of the data disagree by "
                    f"{math.degrees(disagreement):.1f}°, above the "
                    f"{math.degrees(MAX_SPLIT_HALF_DISAGREEMENT_RAD):.0f}° limit"
                )

    confidence = _forward_confidence(correlation, dynamic_count, disagreement)
    if correlation is not None and correlation < MIN_FORWARD_CORRELATION:
        notes.append(
            f"predicted and observed longitudinal acceleration correlate at only "
            f"{correlation:.2f}; the recovered axis is not supported by the data"
        )

    return ForwardDirectionEstimate(
        direction=Vector3.from_array(direction, Frame.PHONE),
        confidence=confidence,
        correlation=correlation,
        dynamic_sample_count=dynamic_count,
        candidate_count=count,
        split_half_disagreement_rad=disagreement,
        reversed_fraction=reversed_fraction,
        method="longitudinal_covariance",
        estimated_lag_s=lag_s,
        lag_peak_margin=margin,
        notes=notes,
    )


def _correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pearson correlation, or None when either side has no variance."""
    if a.size < 2 or b.size != a.size:
        return None
    finite = np.isfinite(a) & np.isfinite(b)
    if int(finite.sum()) < 2:
        return None
    x, y = a[finite], b[finite]
    sx, sy = float(np.std(x)), float(np.std(y))
    if sx < MIN_VECTOR_NORM or sy < MIN_VECTOR_NORM:
        return None
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))


def _forward_confidence(
    correlation: float | None, sample_count: int, disagreement: float | None
) -> float:
    """Product of three named factors, so a low score is always attributable."""
    if correlation is None:
        return 0.0
    strength = max(
        0.0, min(1.0, (correlation - MIN_FORWARD_CORRELATION) / (1.0 - MIN_FORWARD_CORRELATION))
    )
    support = min(1.0, sample_count / (5.0 * MIN_FORWARD_SAMPLES))
    if disagreement is None:
        agreement = 0.5  # unmeasured, so neither credited nor punished fully
    else:
        agreement = max(0.0, 1.0 - disagreement / MAX_SPLIT_HALF_DISAGREEMENT_RAD)
    return float(strength * (0.4 + 0.6 * support) * (0.3 + 0.7 * agreement))


def _tilt_rotation(up: np.ndarray) -> FrameRotation:
    """A 2-DOF rotation fixing the vertical, with a reproducible placeholder yaw.

    The yaw is *not* a measurement. It is chosen deterministically (the phone's
    own X axis, projected into the horizontal plane) purely so the same input
    always produces the same object. Anything that uses this must have checked
    ``yaw_resolved`` first.
    """
    candidates = (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    for candidate in candidates:
        projected = candidate - float(candidate @ up) * up
        if float(np.linalg.norm(projected)) > 1e-3:
            matrix = rotations.basis_from_up_and_forward(up, projected)
            return FrameRotation(matrix, Frame.PHONE, Frame.VEHICLE).inverse()
    raise ValueError("no axis is usable as a placeholder heading")


def estimate_alignment(
    *,
    gravity: GravityEstimate,
    linear_accel: VectorSeries | None = None,
    speed_mps: np.ndarray | None = None,
    valid_speed: np.ndarray | None = None,
    gyro: VectorSeries | None = None,
    reference_yaw_rate_rps: np.ndarray | None = None,
    magnetometer: MagnetometerQuality | None = None,
    duration_s: float = 0.0,
    minimum_gravity_confidence: float = 0.2,
) -> AlignmentEstimate:
    """Estimate ``R_vehicle_phone`` from gravity plus longitudinal dynamics.

    Returns an estimate in every case; the ``status`` field says what was
    achieved. In particular a stationary calibration returns
    ``INSUFFICIENT_MOTION`` with tilt available and ``rotation=None``, which is
    the correct answer rather than a failure.
    """
    notes: list[str] = []
    total_samples = len(linear_accel) if linear_accel is not None else 0
    magnetometer_weight = magnetometer.weight if magnetometer is not None else 0.0

    def quality(
        forward: ForwardDirectionEstimate | None,
        residual: float | None = None,
        longitudinal: float | None = None,
        lateral: float | None = None,
        angular: float | None = None,
    ) -> AlignmentQuality:
        return AlignmentQuality(
            gravity_confidence=gravity.confidence,
            gravity_residual_deg=residual,
            forward_confidence=forward.confidence if forward else 0.0,
            forward_correlation=forward.correlation if forward else None,
            longitudinal_correlation=longitudinal,
            lateral_correlation=lateral,
            angular_rate_consistency=angular,
            magnetometer_weight=magnetometer_weight,
            dynamic_sample_count=forward.dynamic_sample_count if forward else 0,
            total_sample_count=total_samples,
            duration_s=duration_s,
            forward_lag_s=forward.estimated_lag_s if forward else None,
        )

    # --- 1. gravity fixes the vertical -----------------------------------
    if gravity.direction is None or gravity.confidence < minimum_gravity_confidence:
        return AlignmentEstimate(
            status=CalibrationStatus.LOW_GRAVITY_CONFIDENCE,
            rotation=None,
            tilt_only_rotation=None,
            up_direction_phone=None,
            forward_direction_phone=None,
            roll_deg=None,
            pitch_deg=None,
            yaw_deg=None,
            yaw_resolved=False,
            confidence=0.0,
            quality=quality(None),
            sample_count=total_samples,
            duration_s=duration_s,
            notes=[
                *notes,
                f"gravity confidence {gravity.confidence:.2f} is below "
                f"{minimum_gravity_confidence:.2f} ({gravity.method}); without a reliable "
                "vertical, neither tilt nor heading can be established",
                *gravity.notes,
            ],
        )

    up_vector = gravity.direction.scaled(-1.0)
    up = up_vector.to_array()
    tilt = _tilt_rotation(up)
    roll_deg, pitch_deg, _ = tilt.euler_deg()

    # --- 2. motion is required for heading --------------------------------
    if linear_accel is None or speed_mps is None:
        return AlignmentEstimate(
            status=CalibrationStatus.INSUFFICIENT_MOTION,
            rotation=None,
            tilt_only_rotation=tilt,
            up_direction_phone=up_vector,
            forward_direction_phone=None,
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            yaw_deg=None,
            yaw_resolved=False,
            confidence=0.0,
            quality=quality(None),
            sample_count=total_samples,
            duration_s=duration_s,
            notes=[
                *notes,
                "no motion data was supplied; roll and pitch are available from "
                "gravity, yaw is unresolved",
            ],
        )

    forward = estimate_forward_direction(
        linear_accel=linear_accel,
        up_direction=up_vector,
        speed_mps=np.asarray(speed_mps, dtype="float64"),
        valid_speed=valid_speed,
    )

    if forward.direction is None or forward.confidence <= 0.0:
        status = (
            CalibrationStatus.INSUFFICIENT_MOTION
            if forward.dynamic_sample_count < MIN_FORWARD_SAMPLES
            else CalibrationStatus.LOW_HEADING_CONFIDENCE
        )
        if (
            status is CalibrationStatus.LOW_HEADING_CONFIDENCE
            and magnetometer is not None
            and not magnetometer.is_trusted
        ):
            # The magnetometer was the remaining fallback and it is disturbed,
            # so name that specifically: it tells the operator the failure is
            # environmental, not a matter of driving differently.
            status = CalibrationStatus.MAG_DISTURBED
            notes.append(f"magnetometer could not substitute: {magnetometer.describe()}")
        return AlignmentEstimate(
            status=status,
            rotation=None,
            tilt_only_rotation=tilt,
            up_direction_phone=up_vector,
            forward_direction_phone=None,
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            yaw_deg=None,
            yaw_resolved=False,
            confidence=0.0,
            quality=quality(forward),
            sample_count=total_samples,
            duration_s=duration_s,
            notes=[*notes, *forward.notes, "yaw is reported as unresolved rather than guessed"],
        )

    # --- 3. assemble the full rotation ------------------------------------
    try:
        phone_from_vehicle = rotations.basis_from_up_and_forward(up, forward.direction.to_array())
    except ValueError as exc:
        return AlignmentEstimate(
            status=CalibrationStatus.FAILED,
            rotation=None,
            tilt_only_rotation=tilt,
            up_direction_phone=up_vector,
            forward_direction_phone=forward.direction,
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            yaw_deg=None,
            yaw_resolved=False,
            confidence=0.0,
            quality=quality(forward),
            sample_count=total_samples,
            duration_s=duration_s,
            notes=[*notes, f"degenerate geometry: {exc}"],
        )

    rotation = FrameRotation(phone_from_vehicle, Frame.PHONE, Frame.VEHICLE).inverse()
    checks = check_alignment(
        rotation,
        linear_accel=linear_accel,
        gravity=gravity,
        speed_mps=np.asarray(speed_mps, dtype="float64"),
        gyro=gyro,
        reference_yaw_rate_rps=reference_yaw_rate_rps,
    )
    roll_deg, pitch_deg, yaw_deg = rotation.euler_deg()

    confidence = float(
        gravity.confidence
        * forward.confidence
        * (
            1.0
            if checks.gravity_residual_deg is None
            else _residual_factor(checks.gravity_residual_deg)
        )
    )
    status = CalibrationStatus.SUCCESS
    if checks.gravity_residual_deg is not None and (
        checks.gravity_residual_deg > MAX_GRAVITY_RESIDUAL_DEG
    ):
        status = CalibrationStatus.LOW_HEADING_CONFIDENCE
        notes.append(
            f"transformed gravity sits {checks.gravity_residual_deg:.1f}° from vehicle "
            f"vertical, beyond the {MAX_GRAVITY_RESIDUAL_DEG:.0f}° limit"
        )

    return AlignmentEstimate(
        status=status,
        rotation=rotation if status is CalibrationStatus.SUCCESS else None,
        tilt_only_rotation=tilt,
        up_direction_phone=up_vector,
        forward_direction_phone=forward.direction,
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg if status is CalibrationStatus.SUCCESS else None,
        yaw_resolved=status is CalibrationStatus.SUCCESS,
        confidence=confidence,
        quality=quality(
            forward,
            residual=checks.gravity_residual_deg,
            longitudinal=checks.longitudinal_correlation,
            lateral=checks.lateral_correlation,
            angular=checks.angular_rate_consistency,
        ),
        sample_count=total_samples,
        duration_s=duration_s,
        notes=[*notes, *forward.notes, *checks.notes],
    )


def _residual_factor(residual_deg: float) -> float:
    return max(0.0, 1.0 - residual_deg / MAX_GRAVITY_RESIDUAL_DEG)


@dataclass
class AlignmentChecks:
    """Physical plausibility diagnostics for a candidate alignment.

    These are checks, not proofs. Passing every one is consistent with the
    alignment being right; it does not establish it, because the true
    phone→vehicle rotation for this dataset is not independently documented.
    """

    gravity_residual_deg: float | None
    longitudinal_correlation: float | None
    lateral_correlation: float | None
    angular_rate_consistency: float | None
    notes: list[str] = field(default_factory=list)


def check_alignment(
    rotation: FrameRotation,
    *,
    linear_accel: VectorSeries,
    gravity: GravityEstimate,
    speed_mps: np.ndarray | None = None,
    gyro: VectorSeries | None = None,
    reference_yaw_rate_rps: np.ndarray | None = None,
) -> AlignmentChecks:
    """Apply an alignment and test whether the result behaves like a vehicle.

    Four independent questions:

    1. Does gravity end up along vehicle vertical?
    2. Does forward acceleration land on the longitudinal axis?
    3. Does cornering produce lateral acceleration of the right sign and size?
       For a vehicle turning at yaw rate ω at speed v, the centripetal term is
       ``a_lat ≈ v·ω`` — with Z up and Y left, a left turn (ω > 0) pushes the
       measured specific force toward +Y.
    4. Are the transformed angular rates consistent with an independent yaw
       rate reference, where one exists?
    """
    notes: list[str] = []
    vehicle_accel = rotation.apply_series(linear_accel)

    residual: float | None = None
    if gravity.direction is not None:
        transformed = rotation.apply_vector(gravity.direction)
        vehicle_down = Vector3(0.0, 0.0, -1.0, Frame.VEHICLE)
        residual = math.degrees(transformed.angle_to(vehicle_down))
        if residual > MAX_GRAVITY_RESIDUAL_DEG:
            notes.append(f"gravity check: transformed gravity is {residual:.1f}° from vehicle down")

    longitudinal: float | None = None
    lateral: float | None = None
    if speed_mps is not None:
        speed = np.asarray(speed_mps, dtype="float64")
        rate = _speed_derivative(speed, linear_accel.analysis_time_s)
        longitudinal = _correlation(vehicle_accel.samples[:, 0], rate)
        if longitudinal is not None and longitudinal < 0.3:
            notes.append(
                f"longitudinal check: vehicle-frame a_x correlates with dv/dt at only "
                f"{longitudinal:.2f}"
            )

        if gyro is not None and len(gyro) == len(linear_accel):
            vehicle_gyro = rotation.apply_series(gyro)
            expected_lateral = speed * vehicle_gyro.samples[:, 2]
            lateral = _correlation(vehicle_accel.samples[:, 1], expected_lateral)
            if lateral is not None and lateral < 0.2:
                notes.append(
                    f"lateral check: vehicle-frame a_y correlates with v·ω_z at only "
                    f"{lateral:.2f}; cornering does not look like cornering"
                )

    angular: float | None = None
    if gyro is not None and reference_yaw_rate_rps is not None and len(gyro) == len(linear_accel):
        vehicle_gyro = rotation.apply_series(gyro)
        angular = _correlation(
            vehicle_gyro.samples[:, 2], np.asarray(reference_yaw_rate_rps, dtype="float64")
        )
        if angular is not None and angular < 0.3:
            notes.append(
                f"angular check: transformed yaw rate correlates with the reference at "
                f"{angular:.2f}"
            )

    return AlignmentChecks(
        gravity_residual_deg=residual,
        longitudinal_correlation=longitudinal,
        lateral_correlation=lateral,
        angular_rate_consistency=angular,
        notes=notes,
    )


def alignment_from_known_rotation(q_vehicle_phone: np.ndarray) -> FrameRotation:
    """Wrap a known quaternion as a phone→vehicle transform.

    For tests and for the day a mount is measured rather than estimated.
    """
    return FrameRotation.from_quaternion(q_vehicle_phone, Frame.VEHICLE, Frame.PHONE)


def linear_accel_for_alignment(
    accel: VectorSeries, gravity: GravityEstimate
) -> VectorSeries | None:
    """Gravity-removed acceleration, or None when gravity is unavailable."""
    if not gravity.is_available:
        return None
    return linear_acceleration(accel, gravity)


__all__ = [
    "AlignmentChecks",
    "AlignmentEstimate",
    "AlignmentQuality",
    "CalibrationStatus",
    "ForwardDirectionEstimate",
    "alignment_from_known_rotation",
    "check_alignment",
    "estimate_alignment",
    "estimate_forward_direction",
    "linear_accel_for_alignment",
]
