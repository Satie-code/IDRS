"""Comparing a propagated trajectory against the reference — evaluation only.

This module reads the reference. The mechanization does not, and cannot: there
is no path by which anything here writes into a
:class:`~idr.navigation.trajectory.Trajectory`. That separation is the whole
point of §26–§28 of the phase brief, and it is what makes a Phase 5 improvement
measurable rather than assumed.

**Three different times, kept distinct.** Conflating them is the mistake this
module exists to avoid:

``clock alignment``
    What Phase 2 did: interpolate the vehicle stream onto smartphone sample
    times using the two source clocks. Its contract explicitly warns that the
    result is not exact.

``residual alignment``
    The leftover offset Phase 3 measured — median 4.6 s, maximum 14.6 s across
    154 segments. It is real, it is large, and it is *not* corrected anywhere in
    the stored data.

``navigation time``
    ``analysis_time_s`` as the mechanization used it. Never shifted. Shifting
    IMU timestamps to improve a correlation would move the trajectory's own
    clock to flatter a comparison, which is the opposite of measuring anything.

So a lag-corrected comparison resamples **the reference** onto navigation time
and reports the shift it applied. The reference at navigation time ``t`` is
therefore the reference recorded at ``t + lag``, and it is labelled that way
rather than being passed off as the raw value at ``t``.

Both comparisons are produced side by side, always. The difference between them
is the amount of apparent navigation error that is actually synchronization
error, and reporting only the flattering one would misattribute it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from idr.frames import quaternion as quat
from idr.navigation.geodesy import TangentPlane
from idr.navigation.trajectory import Trajectory

#: Half-width of the lag search, seconds. Phase 3 measured a maximum residual
#: of 14.6 s over 154 segments using the same ±15 s window, so the window is
#: known to contain the effect rather than merely being generous.
MAX_LAG_SEARCH_S = 15.0
#: Lag search resolution, seconds. Finer than this is below the ~0.1 s sample
#: spacing that dominates the dataset and would report precision it does not have.
LAG_SEARCH_STEP_S = 0.2
#: How far the best correlation must stand above the median of the search curve
#: before the peak is believed. A flat curve's argmax is noise, and accepting it
#: would manufacture a lag on segments that do not have one.
MIN_LAG_PEAK_MARGIN = 0.1
#: Absolute correlation the peak must also reach. The margin test alone is not
#: enough, and the reason is multiple comparisons: the search evaluates 151
#: candidate lags, and after 2 s smoothing the signals have roughly one
#: independent sample every 2 s, so a two-minute segment carries about 60 of
#: them. The largest of 151 draws with a standard error of 1/√60 ≈ 0.13 reaches
#: ~0.3 by chance while the median stays near zero — which clears a 0.1 margin
#: on pure noise. An absolute floor closes that: below this the two streams
#: simply do not agree well enough for their relative timing to mean anything.
#: 0.3 matches Phase 3's ``MIN_FORWARD_CORRELATION``, chosen for the same reason.
MIN_LAG_CORRELATION = 0.3
#: Speed below which a course-over-ground heading is meaningless — a nearly
#: stationary receiver reports an essentially random bearing.
MIN_HEADING_SPEED_MPS = 2.0
#: Smoothing window applied to both acceleration signals before the lag search,
#: seconds. Matches Phase 3's ``SMOOTHING_WINDOW_S``: differentiating a 10 Hz
#: speed amplifies exactly the road-vibration band that carries no vehicle
#: dynamics, and 2 s separates that from the sub-Hz band that does.
TREND_WINDOW_S = 2.0


@dataclass
class ReferenceTrack:
    """The reference signals available for one segment, and their provenance.

    ``speed_mps`` is a *reference*, not ground truth. Phase 2's tiering makes
    that concrete: tier 1 is a synchronized vehicle stream, tier 3 is smartphone
    GNSS speed whose underlying position refreshes about every 9 s. Neither is
    a truth measurement of the phone's motion, and the label carries through to
    every metric computed from it.
    """

    analysis_time_s: np.ndarray
    speed_mps: np.ndarray | None
    speed_valid: np.ndarray | None
    speed_source: str
    position_enu_m: np.ndarray | None = None
    position_valid: np.ndarray | None = None
    position_source: str = "none"
    heading_deg: np.ndarray | None = None
    heading_valid: np.ndarray | None = None
    heading_source: str = "none"
    origin: TangentPlane | None = None
    #: True only where the source documentation establishes ground truth. It
    #: never does for IO-VNBD, so this stays False and the reports say
    #: "reference" throughout.
    is_ground_truth: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def has_speed(self) -> bool:
        return self.speed_mps is not None and bool(np.isfinite(self.speed_mps).any())

    @property
    def has_position(self) -> bool:
        return self.position_enu_m is not None and bool(np.isfinite(self.position_enu_m).any())


@dataclass
class LagEstimate:
    """The residual synchronization offset, measured and reported, never stored."""

    lag_s: float
    correlation: float
    zero_lag_correlation: float
    peak_margin: float | None
    search_half_width_s: float
    accepted: bool
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        if not self.accepted:
            return (
                f"no distinct correlation peak within ±{self.search_half_width_s:g}s; "
                "the zero-lag comparison is the only one supported"
            )
        return (
            f"the reference had to be read {self.lag_s:+.1f}s away from navigation time "
            f"to line up, lifting the speed correlation from "
            f"{self.zero_lag_correlation:.2f} to {self.correlation:.2f}"
        )


@dataclass
class ErrorCurves:
    """Per-sample error against the reference, at one particular lag."""

    lag_s: float
    analysis_time_s: np.ndarray
    speed_error_mps: np.ndarray | None
    position_error_m: np.ndarray | None
    horizontal_position_error_m: np.ndarray | None
    heading_error_deg: np.ndarray | None
    #: Constant rotation about the vertical removed before comparing position.
    #: ``None`` when no position comparison was possible; a *number* means the
    #: comparison is heading-agnostic and cannot detect a heading error.
    yaw_offset_removed_deg: float | None = None
    valid: np.ndarray | None = None

    def summary(self) -> dict[str, float | None]:
        return {
            "lag_s": self.lag_s,
            "speed_rmse_mps": _rmse(self.speed_error_mps),
            "speed_bias_mps": _mean(self.speed_error_mps),
            "speed_max_abs_error_mps": _max_abs(self.speed_error_mps),
            "position_rmse_m": _rmse(self.position_error_m),
            "horizontal_position_rmse_m": _rmse(self.horizontal_position_error_m),
            "final_position_error_m": _final(self.position_error_m),
            "final_horizontal_error_m": _final(self.horizontal_position_error_m),
            "heading_rmse_deg": _rmse(self.heading_error_deg),
            "yaw_offset_removed_deg": self.yaw_offset_removed_deg,
        }


@dataclass
class DriftMetrics:
    """How fast the solution leaves the reference, rather than by how much.

    A single final error conflates a long drive with a short one. These are the
    quantities that let two segments of different length be compared, and the
    quantities Phase 5 will have to beat.
    """

    duration_s: float
    #: Horizontal position error divided by elapsed time, m/s.
    position_drift_mps: float | None
    #: Horizontal error divided by elapsed time squared, m/s². A constant
    #: acceleration error — which is what an accelerometer bias is — produces a
    #: quadratic position error, so this is the quantity that stays flat when
    #: the dominant term really is bias.
    position_drift_mps2: float | None
    speed_drift_mps_per_s: float | None
    #: Error as a fraction of distance travelled, the usual odometry figure.
    error_per_distance: float | None
    distance_travelled_m: float

    def as_dict(self) -> dict[str, float | None]:
        return {
            "duration_s": self.duration_s,
            "position_drift_mps": self.position_drift_mps,
            "position_drift_mps2": self.position_drift_mps2,
            "speed_drift_mps_per_s": self.speed_drift_mps_per_s,
            "error_per_distance": self.error_per_distance,
            "distance_travelled_m": self.distance_travelled_m,
        }


@dataclass
class ReferenceComparison:
    """Zero-lag and lag-corrected comparisons, side by side. Never one alone."""

    zero_lag: ErrorCurves
    best_lag: ErrorCurves
    lag: LagEstimate
    drift_zero_lag: DriftMetrics
    drift_best_lag: DriftMetrics
    reference_source: str
    is_ground_truth: bool
    heading_observed: bool
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "reference_source": self.reference_source,
            "is_ground_truth": self.is_ground_truth,
            "heading_observed": self.heading_observed,
            "lag_s": self.lag.lag_s,
            "lag_accepted": self.lag.accepted,
            "lag_correlation": self.lag.correlation,
            "zero_lag_correlation": self.lag.zero_lag_correlation,
            "zero_lag": self.zero_lag.summary(),
            "best_lag": self.best_lag.summary(),
            "drift_zero_lag": self.drift_zero_lag.as_dict(),
            "drift_best_lag": self.drift_best_lag.as_dict(),
            "notes": list(self.notes),
        }


def resample_onto(
    values: np.ndarray,
    source_times: np.ndarray,
    target_times: np.ndarray,
    lag_s: float = 0.0,
) -> np.ndarray:
    """Read ``values`` at ``target_times + lag_s``, interpolating on real timestamps.

    Vector-capable, unlike the scalar shift Phase 3 used internally for its
    forward-direction search: a reference *track* has a position as well as a
    speed, and rolling by a sample count would assume a uniform rate this
    dataset does not have.

    Samples falling outside the source span become NaN rather than being clamped
    to the endpoints. Clamping would extend the first and last reference values
    across the shifted region and quietly reduce the error there.
    """
    array = np.asarray(values, dtype="float64")
    source = np.asarray(source_times, dtype="float64")
    target = np.asarray(target_times, dtype="float64") + lag_s
    if array.ndim == 1:
        finite = np.isfinite(array) & np.isfinite(source)
        if int(finite.sum()) < 2:
            return np.full(target.shape, np.nan)
        return np.interp(target, source[finite], array[finite], left=np.nan, right=np.nan)

    out = np.full((target.size, array.shape[1]), np.nan, dtype="float64")
    for column in range(array.shape[1]):
        finite = np.isfinite(array[:, column]) & np.isfinite(source)
        if int(finite.sum()) < 2:
            continue
        out[:, column] = np.interp(
            target, source[finite], array[finite, column], left=np.nan, right=np.nan
        )
    return out


def estimate_residual_lag(
    trajectory: Trajectory,
    reference: ReferenceTrack,
    *,
    max_lag_s: float = MAX_LAG_SEARCH_S,
    step_s: float = LAG_SEARCH_STEP_S,
) -> LagEstimate:
    """Find the shift that best aligns the reference speed with the inertial speed.

    Correlation of *speed*, not of acceleration: over any interval long enough
    for the inertial solution to have drifted, the speeds still rise and fall
    together even when their magnitudes have diverged, and correlation is
    invariant to the scale error that drift introduces.

    The peak has to be a peak. A flat curve's argmax is the largest sample of
    noise, and accepting it would report a confident lag for a segment whose
    reference contains no usable timing information at all.
    """
    notes: list[str] = []
    if not reference.has_speed or reference.speed_mps is None:
        return LagEstimate(
            0.0,
            float("nan"),
            float("nan"),
            None,
            max_lag_s,
            False,
            ["no reference speed, so no lag could be measured"],
        )

    times = trajectory.analysis_time_s
    speed = reference.speed_mps
    valid_reference = (
        np.isfinite(speed)
        if reference.speed_valid is None
        else np.isfinite(speed) & reference.speed_valid
    )
    # Correlate *rates of change*, not the speeds themselves. An unaided
    # inertial speed carries a large, near-monotone drift, and correlating a
    # drifting ramp against anything monotone gives a high score that improves
    # steadily as the reference is shifted — so a raw-speed search pins itself
    # to the edge of the window and reports a lag that is an artifact of the
    # drift rather than a measurement of timing. Differencing removes a linear
    # trend exactly and a slow drift very nearly, leaving the accelerations,
    # which is where timing information actually lives. Both sides are smoothed
    # over 2 s first, for the same reason Phase 3 does it: the phone's
    # high-frequency content is road vibration, and the vehicle's dynamics are
    # sub-Hz.
    # ``_gradient``, not ``np.gradient``: Phase 2 retains duplicate timestamps,
    # and differencing across a Δt of exactly zero divides by zero and poisons
    # the correlation with a non-finite value at every lag.
    inertial = _smooth(_gradient(trajectory.speed_mps, times), times, TREND_WINDOW_S)

    def correlation_at(lag: float) -> float:
        shifted = resample_onto(
            np.where(valid_reference, speed, np.nan),
            reference.analysis_time_s,
            times,
            lag,
        )
        rate = _smooth(_gradient(shifted, times), times, TREND_WINDOW_S)
        usable = np.isfinite(rate) & np.isfinite(inertial)
        if int(usable.sum()) < 10:
            return float("nan")
        return _pearson(inertial[usable], rate[usable])

    zero = correlation_at(0.0)
    steps = int(max_lag_s / step_s)
    scored = [
        (index * step_s, correlation_at(index * step_s)) for index in range(-steps, steps + 1)
    ]
    finite = [(lag, value) for lag, value in scored if math.isfinite(value)]
    if not finite:
        return LagEstimate(
            0.0,
            zero,
            zero,
            None,
            max_lag_s,
            False,
            [
                "the correlation was undefined at every shift: either the two "
                "streams never overlap, or one of them has no variation in its "
                "rate of change for a correlation to align. A constant "
                "acceleration is the common case — it looks identical at every "
                "lag, so it carries no timing information at all."
            ],
        )

    best_lag, best_correlation = max(finite, key=lambda item: item[1])
    median = float(np.median([value for _, value in finite]))
    margin = best_correlation - median
    accepted = (
        margin >= MIN_LAG_PEAK_MARGIN
        and best_correlation >= MIN_LAG_CORRELATION
        and abs(best_lag) >= step_s
    )

    if not accepted:
        notes.append(
            f"best correlation {best_correlation:.2f} against a search-curve median of "
            f"{median:.2f}: no distinct peak above the {MIN_LAG_CORRELATION:g} floor, "
            "so no lag was applied"
        )
        return LagEstimate(0.0, zero, zero, margin, max_lag_s, False, notes)
    # ``>=``, not ``>``: a peak sitting *on* the outermost-but-one candidate is
    # already unsafe, because the true optimum may be the next step out where
    # there is no candidate to compare against.
    if abs(best_lag) >= max_lag_s - step_s:
        notes.append(
            f"the best lag {best_lag:+.1f}s sits at the edge of the ±{max_lag_s:g}s "
            "search window, so the true offset may be larger than reported"
        )
    return LagEstimate(best_lag, best_correlation, zero, margin, max_lag_s, True, notes)


def compare(
    trajectory: Trajectory,
    reference: ReferenceTrack,
    *,
    lag_estimate: LagEstimate | None = None,
    remove_yaw_offset: bool | None = None,
) -> ReferenceComparison:
    """Compare a trajectory against a reference at zero lag and at the best lag.

    ``remove_yaw_offset`` defaults to ``not trajectory.heading_observed``. When
    the initial yaw was an arbitrary origin — the usual case in this dataset —
    the propagated east and north components are correct only up to a constant
    rotation about the vertical, and comparing them directly measures that
    unknown rather than the navigation error. Removing it is an **evaluation**
    operation: the trajectory is not modified, the angle removed is reported,
    and the resulting position error is explicitly blind to a heading error.
    """
    lag = lag_estimate if lag_estimate is not None else estimate_residual_lag(trajectory, reference)
    remove = (not trajectory.heading_observed) if remove_yaw_offset is None else remove_yaw_offset

    zero_curves = _curves_at(trajectory, reference, 0.0, remove)
    best_curves = (
        zero_curves if not lag.accepted else _curves_at(trajectory, reference, lag.lag_s, remove)
    )

    notes: list[str] = list(lag.notes)
    if remove:
        notes.append(
            "a constant rotation about the vertical was removed before comparing "
            "position, because the trajectory's initial yaw was an arbitrary origin "
            "rather than a measurement. The position error below is therefore blind "
            "to a heading error, and is a lower bound on the true position error."
        )
    if not reference.is_ground_truth:
        notes.append(
            f"the comparison is against {reference.speed_source}, a reference rather "
            "than ground truth; a disagreement is not by itself evidence that the "
            "inertial solution is the side that is wrong"
        )

    return ReferenceComparison(
        zero_lag=zero_curves,
        best_lag=best_curves,
        lag=lag,
        drift_zero_lag=drift_metrics(trajectory, zero_curves),
        drift_best_lag=drift_metrics(trajectory, best_curves),
        reference_source=reference.speed_source,
        is_ground_truth=reference.is_ground_truth,
        heading_observed=trajectory.heading_observed,
        notes=notes,
    )


def _curves_at(
    trajectory: Trajectory, reference: ReferenceTrack, lag_s: float, remove_yaw: bool
) -> ErrorCurves:
    times = trajectory.analysis_time_s

    speed_error: np.ndarray | None = None
    if reference.speed_mps is not None:
        masked = (
            reference.speed_mps
            if reference.speed_valid is None
            else np.where(reference.speed_valid, reference.speed_mps, np.nan)
        )
        resampled = resample_onto(masked, reference.analysis_time_s, times, lag_s)
        speed_error = trajectory.speed_mps - resampled

    position_error: np.ndarray | None = None
    horizontal_error: np.ndarray | None = None
    yaw_offset_deg: float | None = None
    if reference.position_enu_m is not None:
        masked_position = reference.position_enu_m.astype("float64", copy=True)
        if reference.position_valid is not None:
            masked_position[~reference.position_valid] = np.nan
        resampled_position = resample_onto(masked_position, reference.analysis_time_s, times, lag_s)
        propagated = trajectory.position_m
        if remove_yaw:
            angle = _best_yaw_offset(propagated[:, :2], resampled_position[:, :2])
            if angle is not None:
                yaw_offset_deg = math.degrees(angle)
                propagated = _rotate_about_z(propagated, angle)
        difference = propagated - resampled_position
        position_error = np.linalg.norm(difference, axis=1)
        horizontal_error = np.linalg.norm(difference[:, :2], axis=1)

    heading_error: np.ndarray | None = None
    if reference.heading_deg is not None:
        heading_error = _heading_error(trajectory, reference, times, lag_s)

    return ErrorCurves(
        lag_s=lag_s,
        analysis_time_s=times,
        speed_error_mps=speed_error,
        position_error_m=position_error,
        horizontal_position_error_m=horizontal_error,
        heading_error_deg=heading_error,
        yaw_offset_removed_deg=yaw_offset_deg,
    )


def _heading_error(
    trajectory: Trajectory, reference: ReferenceTrack, times: np.ndarray, lag_s: float
) -> np.ndarray | None:
    """Course-over-ground comparison, gated on speed.

    This is a *heading* reference, not an attitude reference: it says which way
    the vehicle was travelling, which equals which way it was pointing only for
    a non-skidding vehicle. And below a few m/s a reported bearing is noise, so
    slow samples are excluded rather than contributing a random angle.

    IO-VNBD carries no attitude ground truth at all — Phase 3 measured 86–132°
    median RMSE against the dataset's own orientation columns and established
    that neither series is truth for the other — so this is the only orientation
    comparison in Phase 4 that means anything, and it constrains one axis.
    """
    if reference.heading_deg is None:
        return None
    masked = (
        reference.heading_deg
        if reference.heading_valid is None
        else np.where(reference.heading_valid, reference.heading_deg, np.nan)
    )
    # Interpolate the unit vector rather than the angle: interpolating degrees
    # across the 360°/0° wrap produces a sweep through every intermediate
    # bearing, which is a large error that never happened.
    radians = np.radians(masked)
    east = resample_onto(np.sin(radians), reference.analysis_time_s, times, lag_s)
    north = resample_onto(np.cos(radians), reference.analysis_time_s, times, lag_s)
    reference_heading = np.degrees(np.arctan2(east, north))

    velocity = trajectory.velocity_mps
    speed = trajectory.speed_mps
    inertial_heading = np.degrees(np.arctan2(velocity[:, 0], velocity[:, 1]))
    error = _wrap_deg(inertial_heading - reference_heading)
    error[speed < MIN_HEADING_SPEED_MPS] = np.nan
    return error


def _best_yaw_offset(propagated_en: np.ndarray, reference_en: np.ndarray) -> float | None:
    """The rotation about the vertical that best maps propagated onto reference.

    Closed form. Minimizing ``Σ‖R(θ)p − r‖²`` over rotations about Z gives
    ``θ = atan2(Σ(pₓrᵧ − pᵧrₓ) … )`` directly — no search, no tuning, and no
    opportunity to nudge it toward a nicer number.

    Note what this cannot do: it removes exactly one degree of freedom, and it
    is applied only when the trajectory's own heading was never observed. It is
    not a fit to the reference in any broader sense — no scale, no translation,
    no time warp.
    """
    usable = np.isfinite(propagated_en).all(axis=1) & np.isfinite(reference_en).all(axis=1)
    if int(usable.sum()) < 3:
        return None
    p = propagated_en[usable]
    r = reference_en[usable]
    numerator = float(np.sum(p[:, 0] * r[:, 1] - p[:, 1] * r[:, 0]))
    denominator = float(np.sum(p[:, 0] * r[:, 0] + p[:, 1] * r[:, 1]))
    if abs(numerator) < 1e-12 and abs(denominator) < 1e-12:
        return None
    return math.atan2(numerator, denominator)


def _rotate_about_z(positions: np.ndarray, angle_rad: float) -> np.ndarray:
    cos, sin = math.cos(angle_rad), math.sin(angle_rad)
    out = positions.copy()
    out[:, 0] = cos * positions[:, 0] - sin * positions[:, 1]
    out[:, 1] = sin * positions[:, 0] + cos * positions[:, 1]
    return out


def drift_metrics(trajectory: Trajectory, curves: ErrorCurves) -> DriftMetrics:
    """Reduce an error curve to rates that survive comparison across segments."""
    duration = trajectory.duration_s
    distance = trajectory.distance_travelled_m()
    horizontal = curves.horizontal_position_error_m
    final_horizontal = _final(horizontal)

    position_rate = (
        final_horizontal / duration if final_horizontal is not None and duration > 0 else None
    )
    position_quadratic = (
        2.0 * final_horizontal / (duration * duration)
        if final_horizontal is not None and duration > 0
        else None
    )
    speed_final = _final(curves.speed_error_mps)
    speed_rate = abs(speed_final) / duration if speed_final is not None and duration > 0 else None
    per_distance = (
        final_horizontal / distance if final_horizontal is not None and distance > 1.0 else None
    )
    return DriftMetrics(
        duration_s=duration,
        position_drift_mps=position_rate,
        position_drift_mps2=position_quadratic,
        speed_drift_mps_per_s=speed_rate,
        error_per_distance=per_distance,
        distance_travelled_m=distance,
    )


def attitude_error_deg(estimated: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Angular distance between two attitude sequences, degrees.

    Only usable where a truth attitude exists — that is, on synthetic data.
    Real IO-VNBD has none, and this function is deliberately not wired to the
    dataset's orientation columns.
    """
    if estimated.shape != truth.shape:
        raise ValueError(f"{estimated.shape} against {truth.shape}")
    return np.array(
        [
            math.degrees(quat.angular_distance(estimated[index], truth[index]))
            for index in range(estimated.shape[0])
        ],
        dtype="float64",
    )


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    a_centred = a - a.mean()
    b_centred = b - b.mean()
    denominator = float(np.linalg.norm(a_centred) * np.linalg.norm(b_centred))
    if denominator < 1e-12:
        return float("nan")
    return float(a_centred @ b_centred / denominator)


def _wrap_deg(values: np.ndarray) -> np.ndarray:
    return (values + 180.0) % 360.0 - 180.0


def _rmse(values: np.ndarray | None) -> float | None:
    if values is None:
        return None
    finite = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(finite * finite))) if finite.size else None


def _mean(values: np.ndarray | None) -> float | None:
    if values is None:
        return None
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else None


def _max_abs(values: np.ndarray | None) -> float | None:
    if values is None:
        return None
    finite = values[np.isfinite(values)]
    return float(np.abs(finite).max()) if finite.size else None


def _final(values: np.ndarray | None) -> float | None:
    """The last finite value, which is what "final error" has to mean when the
    reference is missing at the very end of a segment."""
    if values is None:
        return None
    finite = values[np.isfinite(values)]
    return float(finite[-1]) if finite.size else None


def _gradient(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    """d(values)/dt on actual timestamps, tolerating gaps and NaNs.

    ``np.gradient`` refuses non-monotonic sample positions and propagates a NaN
    across three output samples, so the finite subset is differentiated and the
    result scattered back. Gaps stay gaps: no value is invented where the
    reference had none.
    """
    out = np.full(values.shape, np.nan, dtype="float64")
    finite = np.isfinite(values) & np.isfinite(times)
    if int(finite.sum()) < 3:
        return out
    stamps = times[finite]
    # Strictly increasing timestamps are required; duplicates reach here from
    # Phase 2 and would make np.gradient divide by zero.
    keep = np.concatenate([[True], np.diff(stamps) > 0.0])
    indices = np.flatnonzero(finite)[keep]
    if indices.size < 3:
        return out
    out[indices] = np.gradient(values[indices], times[indices])
    return out


def _smooth(values: np.ndarray, times: np.ndarray, window_s: float) -> np.ndarray:
    """Trailing mean over a *time* window, skipping non-finite entries.

    Time-based rather than sample-count-based because this dataset spans 2 Hz to
    1000 Hz, and a fixed sample count would mean two very different things
    across it.
    """
    array = np.asarray(values, dtype="float64")
    finite = np.isfinite(array)
    filled = np.where(finite, array, 0.0)
    totals = np.concatenate([[0.0], np.cumsum(filled)])
    counts = np.concatenate([[0.0], np.cumsum(finite)])
    start = np.searchsorted(times, times - window_s, side="left")
    stop = np.arange(1, array.size + 1)
    count = counts[stop] - counts[start]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, (totals[stop] - totals[start]) / np.maximum(count, 1.0), np.nan)
