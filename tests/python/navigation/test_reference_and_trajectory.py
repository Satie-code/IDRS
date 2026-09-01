"""Reference comparison, lag handling, geodesy, and the trajectory container.

The invariant these tests defend is §28: the reference informs *evaluation* and
never the solution. Several of them assert that a thing does **not** happen —
that a lag-corrected comparison leaves the trajectory untouched, that no method
exists to inject a fix — because "we did not aid the baseline" is a claim the
Phase 5 comparison depends on and is not verifiable by reading alone.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from idr.frames import quaternion as quat
from idr.navigation import synthetic as syn
from idr.navigation.geodesy import (
    WGS84_SEMI_MAJOR_M,
    TangentPlane,
    origin_from_fixes,
)
from idr.navigation.mechanization import (
    AlignmentPolicy,
    MechanizationConfig,
    StrapdownMechanization,
)
from idr.navigation.reference_alignment import (
    ReferenceTrack,
    attitude_error_deg,
    compare,
    drift_metrics,
    estimate_residual_lag,
    resample_onto,
)
from idr.navigation.state import NavigationFlag, NavigationState
from idr.navigation.trajectory import concatenate, mechanize_series


def _trajectory(drive=None, **kwargs):
    drive = drive or syn.constant_acceleration(acceleration_mps2=1.0, duration_s=60.0)
    return drive, mechanize_series(
        initial=drive.initial_state(),
        specific_force=drive.specific_force_phone,
        angular_rate=drive.angular_rate_phone,
        orientations=drive.orientation,
        config=MechanizationConfig(
            alignment_policy=AlignmentPolicy.PHONE_FRAME_DIAGNOSTIC, **kwargs
        ),
    )


def _structured_trajectory(duration_s: float = 120.0, rate_hz: float = 10.0):
    """A trajectory whose speed has features a lag search can lock onto.

    Built directly rather than mechanized, because the lag estimator's contract
    is about a :class:`Trajectory` and a :class:`ReferenceTrack` — not about how
    the trajectory was produced — and a profile with structure isolates the
    search from everything else.
    """
    from idr.navigation.trajectory import Trajectory

    count = int(duration_s * rate_hz) + 1
    times = np.arange(count, dtype="float64") / rate_hz
    # Accelerate/cruise/brake cycles: the peaks and troughs are what a
    # correlation can align. A monotonic ramp has none, by construction.
    speed = (
        12.0
        + 8.0 * np.sin(2.0 * math.pi * times / 25.0)
        + 3.0 * np.sin(2.0 * math.pi * times / 7.0)
    )
    velocity = np.column_stack([speed, np.zeros(count), np.zeros(count)])
    position = np.column_stack([np.cumsum(speed) / rate_hz, np.zeros(count), np.zeros(count)])
    return Trajectory(
        analysis_time_s=times,
        position_m=position,
        velocity_mps=velocity,
        orientation=np.tile(quat.identity(), (count, 1)),
        acceleration_nav=np.zeros((count, 3)),
        specific_force_phone=np.zeros((count, 3)),
        flags=np.zeros(count, dtype="int64"),
        heading_observed=True,
    )


def _reference_from(drive, *, lag_s: float = 0.0, noise: float = 0.0, seed: int = 3):
    """A reference speed built from the truth, optionally shifted in time.

    ``lag_s`` shifts the reference *timestamps*, which is how a residual
    synchronization error actually presents: the values are right, the clock
    they are stamped with is not.
    """
    speed = np.linalg.norm(drive.velocity_nav_mps, axis=1)
    if noise:
        speed = speed + np.random.default_rng(seed).normal(0.0, noise, size=speed.shape)
    return ReferenceTrack(
        analysis_time_s=drive.analysis_time_s + lag_s,
        speed_mps=speed,
        speed_valid=np.ones(speed.size, dtype=bool),
        speed_source="synthetic reference",
    )


# --- resampling -------------------------------------------------------------


def test_resample_interpolates_on_real_timestamps():
    source = np.array([0.0, 1.0, 2.0, 3.0])
    values = np.array([0.0, 10.0, 20.0, 30.0])
    out = resample_onto(values, source, np.array([0.5, 1.5]))
    assert out == pytest.approx([5.0, 15.0])


def test_resample_does_not_clamp_beyond_the_span():
    """Outside the source span the answer is NaN, not the nearest endpoint.

    Clamping would extend the first and last reference values across a shifted
    region and quietly reduce the error there.
    """
    source = np.array([0.0, 1.0, 2.0])
    values = np.array([5.0, 6.0, 7.0])
    out = resample_onto(values, source, np.array([-1.0, 3.0]))
    assert np.isnan(out).all()


def test_resample_handles_vectors():
    source = np.array([0.0, 1.0])
    values = np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]])
    out = resample_onto(values, source, np.array([0.5]))
    assert out.shape == (1, 3)
    assert out[0] == pytest.approx([5.0, 10.0, 15.0])


def test_resample_survives_a_degenerate_source():
    assert np.isnan(resample_onto(np.array([1.0]), np.array([0.0]), np.array([0.0]))).all()
    columns = resample_onto(np.array([[1.0, 2.0]]), np.array([0.0]), np.array([0.0]))
    assert np.isnan(columns).all()


def test_resample_with_zero_lag_is_the_identity_on_its_own_grid():
    source = np.array([0.0, 1.0, 2.0])
    values = np.array([5.0, 6.0, 7.0])
    assert resample_onto(values, source, source, 0.0) == pytest.approx(values)


# --- lag estimation ---------------------------------------------------------


def test_a_known_lag_is_recovered():
    """A reference shifted by 4 s must be found at +4 s, within one search step."""
    trajectory = _structured_trajectory()
    reference = ReferenceTrack(
        analysis_time_s=trajectory.analysis_time_s + 4.0,
        speed_mps=trajectory.speed_mps.copy(),
        speed_valid=np.ones(len(trajectory), dtype=bool),
        speed_source="synthetic reference",
    )
    estimate = estimate_residual_lag(trajectory, reference)
    assert estimate.accepted
    assert estimate.lag_s == pytest.approx(4.0, abs=0.3)
    assert estimate.correlation > estimate.zero_lag_correlation


def test_a_monotonic_ramp_carries_no_timing_information():
    """A constant acceleration is identical at every lag, so no lag is reported.

    This is why the search correlates *rates of change* rather than speeds. A
    linear speed ramp shifted in time is still the same ramp plus an offset, so
    a raw-speed correlation scores ~1.0 everywhere and its argmax is noise —
    and on real data, where the inertial speed carries a large monotone drift,
    that artifact pins the search to the edge of its window and reports a
    confident lag that measures nothing.

    Differencing removes the trend and leaves a constant, which has no variance
    for a correlation to work with, so the estimator refuses outright. Refusing
    is the correct answer: this signal genuinely contains no timing information.
    """
    drive, trajectory = _trajectory(
        syn.analytic_trajectory(
            acceleration_nav_mps2=[1.0, 0.0, 0.0],
            initial_velocity_nav_mps=[5.0, 0.0, 0.0],
            duration_s=120.0,
        )
    )
    estimate = estimate_residual_lag(trajectory, _reference_from(drive, lag_s=4.0))
    assert not estimate.accepted
    assert estimate.lag_s == 0.0
    assert any("no timing information" in note for note in estimate.notes)


def test_the_search_is_not_fooled_by_drift():
    """A drifting inertial speed must not manufacture a lag out of its own trend.

    The failure this guards: correlating raw speeds, an inertial solution that
    ramps away from the reference scores better and better as the reference is
    shifted, so the search saturates at its window edge on nearly every segment.
    Here the reference is aligned and the trajectory carries a large added
    drift; the answer must still be "no lag".
    """
    trajectory = _structured_trajectory()
    reference = ReferenceTrack(
        analysis_time_s=trajectory.analysis_time_s,
        speed_mps=trajectory.speed_mps.copy(),
        speed_valid=np.ones(len(trajectory), dtype=bool),
        speed_source="synthetic reference",
    )
    elapsed = trajectory.analysis_time_s - trajectory.analysis_time_s[0]
    drifted = trajectory.velocity_mps.copy()
    drifted[:, 0] += 0.6 * elapsed  # 0.6 m/s² of unmodelled acceleration error
    trajectory.velocity_mps = drifted

    estimate = estimate_residual_lag(trajectory, reference)
    assert abs(estimate.lag_s) < 1.0, (
        f"a drifting solution produced a spurious {estimate.lag_s:+.1f}s lag"
    )


def test_no_lag_is_reported_when_there_is_none():
    """A perfectly aligned reference must not acquire a spurious shift."""
    drive, trajectory = _trajectory()
    estimate = estimate_residual_lag(trajectory, _reference_from(drive))
    assert estimate.lag_s == 0.0


def test_a_flat_correlation_curve_is_rejected():
    """A curve with no peak yields no lag, with a note saying why.

    The trajectory has real structure and the reference is noise, so the
    correlation is finite at every shift and near zero everywhere. The
    peak-margin gate must decline rather than returning the largest sample of
    the noise as a measurement.
    """
    trajectory = _structured_trajectory()
    noise = np.random.default_rng(11).normal(10.0, 3.0, size=len(trajectory))
    reference = ReferenceTrack(
        analysis_time_s=trajectory.analysis_time_s,
        speed_mps=noise,
        speed_valid=np.ones(noise.size, dtype=bool),
        speed_source="pure noise",
    )
    estimate = estimate_residual_lag(trajectory, reference)
    assert not estimate.accepted
    assert estimate.lag_s == 0.0
    assert any("no distinct peak" in note for note in estimate.notes)


def test_a_missing_reference_yields_no_lag():
    _, trajectory = _trajectory()
    empty = ReferenceTrack(
        analysis_time_s=trajectory.analysis_time_s,
        speed_mps=None,
        speed_valid=None,
        speed_source="none",
    )
    estimate = estimate_residual_lag(trajectory, empty)
    assert not estimate.accepted
    assert "no reference speed" in estimate.notes[0]
    assert "only one supported" in estimate.describe()


def test_an_edge_lag_is_flagged_as_possibly_larger():
    """A peak at the edge of the window may not be the real peak, and says so."""
    trajectory = _structured_trajectory()
    reference = ReferenceTrack(
        analysis_time_s=trajectory.analysis_time_s + 4.0,
        speed_mps=trajectory.speed_mps.copy(),
        speed_valid=np.ones(len(trajectory), dtype=bool),
        speed_source="synthetic reference",
    )
    estimate = estimate_residual_lag(trajectory, reference, max_lag_s=4.2)
    assert estimate.accepted
    assert any("edge of the" in note for note in estimate.notes)


# --- comparison -------------------------------------------------------------


def test_both_lags_are_always_reported():
    """Never one alone: the difference is the synchronization contribution."""
    trajectory = _structured_trajectory()
    comparison = compare(
        trajectory,
        ReferenceTrack(
            analysis_time_s=trajectory.analysis_time_s + 4.0,
            speed_mps=trajectory.speed_mps.copy(),
            speed_valid=np.ones(len(trajectory), dtype=bool),
            speed_source="synthetic reference",
        ),
    )
    assert comparison.zero_lag.lag_s == 0.0
    assert comparison.best_lag.lag_s != 0.0
    zero = comparison.zero_lag.summary()["speed_rmse_mps"]
    best = comparison.best_lag.summary()["speed_rmse_mps"]
    assert best < zero, "removing a real lag must reduce the error"


def test_comparison_does_not_modify_the_trajectory():
    """§28, as an assertion. Evaluation must leave the solution untouched."""
    drive, trajectory = _trajectory()
    before_position = trajectory.position_m.copy()
    before_velocity = trajectory.velocity_mps.copy()
    compare(trajectory, _reference_from(drive, lag_s=3.0))
    assert trajectory.position_m == pytest.approx(before_position)
    assert trajectory.velocity_mps == pytest.approx(before_velocity)


def test_mechanization_exposes_no_correction_hook():
    """There is no way to inject a fix, and that is deliberate."""
    engine = StrapdownMechanization()
    for forbidden in ("update", "correct", "apply_fix", "reset_velocity", "set_position"):
        assert not hasattr(engine, forbidden), (
            f"{forbidden}() would let the baseline be aided, which makes the "
            "Phase 5 comparison meaningless"
        )


def test_a_reference_is_never_called_ground_truth():
    drive, trajectory = _trajectory()
    comparison = compare(trajectory, _reference_from(drive))
    assert not comparison.is_ground_truth
    assert any("reference rather" in note for note in comparison.notes)


def test_yaw_offset_is_removed_only_when_heading_was_unobserved():
    """The evaluation degree of freedom, applied on evidence and reported."""
    drive = syn.analytic_trajectory(acceleration_nav_mps2=[1.0, 0.0, 0.0], duration_s=60.0)
    trajectory = mechanize_series(
        initial=drive.initial_state(heading_observed=False),
        specific_force=drive.specific_force_phone,
        angular_rate=drive.angular_rate_phone,
        orientations=drive.orientation,
        config=MechanizationConfig(alignment_policy=AlignmentPolicy.PHONE_FRAME_DIAGNOSTIC),
    )
    assert not trajectory.heading_observed

    # A reference rotated 40° about the vertical from the propagated track.
    angle = math.radians(40.0)
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = drive.position_nav_m @ rotation.T
    reference = ReferenceTrack(
        analysis_time_s=drive.analysis_time_s,
        speed_mps=np.linalg.norm(drive.velocity_nav_mps, axis=1),
        speed_valid=np.ones(len(drive), dtype=bool),
        speed_source="synthetic",
        position_enu_m=rotated,
        position_valid=np.ones(len(drive), dtype=bool),
        position_source="synthetic",
    )
    comparison = compare(trajectory, reference)
    removed = comparison.best_lag.yaw_offset_removed_deg
    assert removed is not None
    assert removed == pytest.approx(40.0, abs=0.5)
    # With the unknown removed, the residual position error is at rounding.
    assert comparison.best_lag.summary()["horizontal_position_rmse_m"] < 1e-6
    assert any("blind to a heading error" in note for note in comparison.notes)


def test_observed_heading_keeps_the_position_comparison_honest():
    """When heading *was* observed, the offset is not removed and the error stands."""
    drive, trajectory = _trajectory()
    assert trajectory.heading_observed
    angle = math.radians(40.0)
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    reference = ReferenceTrack(
        analysis_time_s=drive.analysis_time_s,
        speed_mps=np.linalg.norm(drive.velocity_nav_mps, axis=1),
        speed_valid=np.ones(len(drive), dtype=bool),
        speed_source="synthetic",
        position_enu_m=drive.position_nav_m @ rotation.T,
        position_valid=np.ones(len(drive), dtype=bool),
        position_source="synthetic",
    )
    comparison = compare(trajectory, reference)
    assert comparison.best_lag.yaw_offset_removed_deg is None
    assert comparison.best_lag.summary()["horizontal_position_rmse_m"] > 100.0


def test_heading_error_wraps_and_gates_on_speed():
    """Bearings interpolate as vectors, and a crawling vehicle contributes nothing."""
    drive = syn.analytic_trajectory(initial_velocity_nav_mps=[0.0, 20.0, 0.0], duration_s=20.0)
    _, trajectory = _trajectory(drive)
    reference = ReferenceTrack(
        analysis_time_s=drive.analysis_time_s,
        speed_mps=np.full(len(drive), 20.0),
        speed_valid=np.ones(len(drive), dtype=bool),
        speed_source="synthetic",
        heading_deg=np.zeros(len(drive)),  # due north
        heading_valid=np.ones(len(drive), dtype=bool),
        heading_source="synthetic",
    )
    comparison = compare(trajectory, reference)
    errors = comparison.best_lag.heading_error_deg
    assert errors is not None
    finite = errors[np.isfinite(errors)]
    assert finite.size > 0
    assert np.abs(finite).max() < 1e-6


def test_heading_error_is_nan_while_slow():
    drive = syn.stationary(duration_s=20.0)
    _, trajectory = _trajectory(drive)
    reference = ReferenceTrack(
        analysis_time_s=drive.analysis_time_s,
        speed_mps=np.zeros(len(drive)),
        speed_valid=np.ones(len(drive), dtype=bool),
        speed_source="synthetic",
        heading_deg=np.zeros(len(drive)),
        heading_valid=np.ones(len(drive), dtype=bool),
    )
    comparison = compare(trajectory, reference)
    errors = comparison.best_lag.heading_error_deg
    assert errors is not None and np.isnan(errors).all()


def test_drift_metrics_report_rates_not_just_a_final_number():
    drive, trajectory = _trajectory()
    reference = ReferenceTrack(
        analysis_time_s=drive.analysis_time_s,
        speed_mps=np.linalg.norm(drive.velocity_nav_mps, axis=1),
        speed_valid=np.ones(len(drive), dtype=bool),
        speed_source="synthetic",
        position_enu_m=drive.position_nav_m + np.array([50.0, 0.0, 0.0]),
        position_valid=np.ones(len(drive), dtype=bool),
    )
    comparison = compare(trajectory, reference)
    metrics = comparison.drift_best_lag
    assert metrics.duration_s == pytest.approx(60.0, abs=0.2)
    assert metrics.position_drift_mps == pytest.approx(50.0 / 60.0, rel=0.05)
    assert metrics.distance_travelled_m > 1000.0
    assert metrics.error_per_distance is not None
    assert set(metrics.as_dict()) >= {"position_drift_mps", "position_drift_mps2"}


def test_drift_metrics_survive_an_empty_error_curve():
    _, trajectory = _trajectory()
    from idr.navigation.reference_alignment import ErrorCurves

    empty = ErrorCurves(
        lag_s=0.0,
        analysis_time_s=trajectory.analysis_time_s,
        speed_error_mps=None,
        position_error_m=None,
        horizontal_position_error_m=None,
        heading_error_deg=None,
    )
    metrics = drift_metrics(trajectory, empty)
    assert metrics.position_drift_mps is None
    assert metrics.speed_drift_mps_per_s is None


def test_attitude_error_requires_matching_shapes():
    with pytest.raises(ValueError):
        attitude_error_deg(np.zeros((3, 4)), np.zeros((4, 4)))
    identical = np.tile(quat.identity(), (5, 1))
    assert attitude_error_deg(identical, identical) == pytest.approx(np.zeros(5), abs=1e-9)


# --- geodesy ----------------------------------------------------------------


def test_tangent_plane_round_trips_a_small_displacement():
    """100 m north must project to 100 m north, to well under a metre."""
    origin = TangentPlane(51.5, -0.12, 0.0)
    metres_per_degree_north = origin.meridional_radius_m * math.pi / 180.0
    offset_deg = 100.0 / metres_per_degree_north
    enu = origin.to_enu(np.array([51.5 + offset_deg]), np.array([-0.12]))
    assert enu[0, 1] == pytest.approx(100.0, abs=1e-6)
    assert enu[0, 0] == pytest.approx(0.0, abs=1e-9)


def test_tangent_plane_radii_are_wgs84():
    equator = TangentPlane(0.0, 0.0)
    assert equator.transverse_radius_m == pytest.approx(WGS84_SEMI_MAJOR_M)
    pole = TangentPlane(90.0, 0.0)
    assert pole.meridional_radius_m > equator.meridional_radius_m


def test_tangent_plane_preserves_missing_fixes():
    origin = TangentPlane(51.5, -0.12)
    enu = origin.to_enu(np.array([51.5, np.nan]), np.array([-0.12, -0.12]))
    assert np.isfinite(enu[0]).all()
    assert np.isnan(enu[1, :2]).all()


def test_tangent_plane_checks_its_inputs():
    origin = TangentPlane(0.0, 0.0)
    with pytest.raises(ValueError, match="must match"):
        origin.to_enu(np.array([1.0, 2.0]), np.array([1.0]))


def test_origin_is_the_first_valid_fix_not_the_centroid():
    """A centroid origin would look exactly like an initialization error."""
    latitude = np.array([np.nan, 51.5, 51.6, 51.7])
    longitude = np.array([np.nan, -0.12, -0.13, -0.14])
    origin = origin_from_fixes(latitude, longitude)
    assert origin is not None
    assert origin.latitude_deg == pytest.approx(51.5)


def test_null_island_is_not_an_origin():
    """(0, 0) is what a receiver writes with no fix; it is in the Atlantic."""
    latitude = np.array([0.0, 0.0, 51.5])
    longitude = np.array([0.0, 0.0, -0.12])
    origin = origin_from_fixes(latitude, longitude)
    assert origin is not None
    assert origin.latitude_deg == pytest.approx(51.5)


def test_no_valid_fix_gives_no_origin():
    assert origin_from_fixes(np.array([np.nan]), np.array([np.nan])) is None
    assert origin_from_fixes(np.array([51.5]), np.array([-0.12]), valid=np.array([False])) is None


def test_origin_takes_altitude_when_it_is_finite():
    origin = origin_from_fixes(np.array([51.5]), np.array([-0.12]), np.array([35.0]))
    assert origin is not None and origin.altitude_m == pytest.approx(35.0)
    without = origin_from_fixes(np.array([51.5]), np.array([-0.12]), np.array([np.nan]))
    assert without is not None and without.altitude_m == 0.0
    assert "tangent plane" in origin.describe()


# --- the trajectory container ------------------------------------------------


def test_trajectory_summary_is_scalar_only():
    _, trajectory = _trajectory()
    payload = trajectory.as_dict()
    assert payload["samples"] == len(trajectory)
    assert payload["displacement_m"] > 0.0
    assert payload["path_length_m"] >= payload["displacement_m"]
    assert isinstance(payload["flags"], dict)
    for value in payload.values():
        assert not isinstance(value, np.ndarray)


def test_state_at_reconstructs_a_navigation_state():
    _, trajectory = _trajectory()
    state = trajectory.state_at(10)
    assert isinstance(state, NavigationState)
    assert state.analysis_time_s == pytest.approx(trajectory.analysis_time_s[10])
    assert state.speed_mps == pytest.approx(float(trajectory.speed_mps[10]))
    assert state.position.frame.value == "navigation"
    assert len(state.euler_deg()) == 3


def test_navigation_state_validates_its_shapes():
    with pytest.raises(ValueError, match="position_m"):
        NavigationState(0.0, np.zeros(2), np.zeros(3), quat.identity())
    with pytest.raises(ValueError, match="orientation"):
        NavigationState(0.0, np.zeros(3), np.zeros(3), np.zeros(3))


def test_flags_describe_themselves():
    combined = NavigationFlag.IRREGULAR_DT | NavigationFlag.STATIONARY
    described = combined.describe()
    assert "irregular dt" in described and "stationary" in described
    assert NavigationFlag.GOOD.describe() == "good"


def test_with_flags_accumulates():
    state = NavigationState(0.0, np.zeros(3), np.zeros(3), quat.identity())
    updated = state.with_flags(NavigationFlag.STATIONARY).with_flags(NavigationFlag.IRREGULAR_DT)
    assert updated.flags & NavigationFlag.STATIONARY
    assert updated.flags & NavigationFlag.IRREGULAR_DT


def test_concatenate_keeps_the_joins_visible():
    """Segmented pieces are joined for plotting, not made continuous."""
    _, first = _trajectory()
    _, second = _trajectory()
    joined = concatenate([first, second])
    assert len(joined) == len(first) + len(second)
    assert joined.counters.integrated == first.counters.integrated + second.counters.integrated
    assert any("discontinuous" in note for note in joined.notes)
    assert concatenate([first]) is first
    with pytest.raises(ValueError, match="nothing to concatenate"):
        concatenate([])


def test_empty_trajectory_summaries_are_safe():
    _, trajectory = _trajectory()
    trajectory.analysis_time_s = trajectory.analysis_time_s[:1]
    trajectory.position_m = trajectory.position_m[:1]
    assert trajectory.duration_s == 0.0
    assert trajectory.distance_travelled_m() == 0.0
    assert trajectory.displacement_m() == 0.0
