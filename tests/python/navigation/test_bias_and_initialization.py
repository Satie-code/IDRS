"""Bias representation, stationary initialization, and stationarity detection.

The centrepiece is :func:`test_accelerometer_bias_is_only_observable_along_gravity`.
It asserts a *limitation* rather than a capability, which is unusual for a test
and is the point: §21 of the phase brief warns against reading
``accelerometer_mean`` as a bias, and the failure it warns about produces a
number that looks entirely reasonable. A test that only checked the estimate
"worked" would pass against the wrong implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from idr.frames.conventions import STANDARD_GRAVITY_MPS2, Frame
from idr.frames.gravity import GravityEstimate, GravityMethod, estimate_gravity
from idr.frames.vectors import Vector3, VectorSeries
from idr.navigation import synthetic as syn
from idr.navigation.bias import (
    MAX_PLAUSIBLE_ACCEL_BIAS_MPS2,
    MAX_PLAUSIBLE_GYRO_BIAS_RPS,
    AccelerometerBias,
    GyroscopeBias,
    ImuBias,
    estimate_bias_from_rest,
    perturbed,
    phone_frame_check,
)
from idr.navigation.mechanization import (
    AlignmentPolicy,
    MechanizationConfig,
)
from idr.navigation.stationarity import (
    detect_stationarity,
    stationary_duration_s,
)
from idr.navigation.trajectory import mechanize_series


def _series(samples: np.ndarray, times: np.ndarray, frame: Frame = Frame.PHONE) -> VectorSeries:
    return VectorSeries(samples=samples, analysis_time_s=times, frame=frame, source="test")


def _resting(count: int = 400, rate_hz: float = 10.0, bias: np.ndarray | None = None):
    """A resting phone lying flat: accelerometer reads +g on Z, gyro reads zero."""
    times = np.arange(count, dtype="float64") / rate_hz
    force = np.tile([0.0, 0.0, STANDARD_GRAVITY_MPS2], (count, 1))
    if bias is not None:
        force = force + bias
    gyro = np.zeros((count, 3))
    return times, _series(force, times), _series(gyro, times)


# --- the bias containers ---------------------------------------------------


def test_bias_correction_subtracts():
    """``measured = true + bias``, so correction subtracts. Stated once, tested once."""
    gyro = GyroscopeBias(np.array([0.01, -0.02, 0.03]))
    assert gyro.correct(np.array([0.11, 0.0, 0.0])) == pytest.approx([0.10, 0.02, -0.03])
    accel = AccelerometerBias(np.array([0.1, 0.0, 0.0]))
    assert accel.correct(np.array([1.0, 2.0, 3.0])) == pytest.approx([0.9, 2.0, 3.0])


def test_zero_bias_is_a_no_op():
    bias = ImuBias.zero()
    assert bias.is_zero
    sample = np.array([1.0, 2.0, 3.0])
    assert bias.accelerometer.correct(sample) == pytest.approx(sample)
    assert bias.gyroscope.correct(sample) == pytest.approx(sample)
    assert bias.accelerometer.observable_axes == "none"


def test_bias_rejects_malformed_values():
    with pytest.raises(ValueError, match="shape"):
        GyroscopeBias(np.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        AccelerometerBias(np.array([np.nan, 0.0, 0.0]))


def test_plausibility_bounds():
    assert GyroscopeBias(np.array([0.01, 0.0, 0.0])).is_plausible
    assert not GyroscopeBias(np.array([MAX_PLAUSIBLE_GYRO_BIAS_RPS * 2, 0.0, 0.0])).is_plausible
    assert AccelerometerBias(np.array([0.05, 0.0, 0.0])).is_plausible
    assert not AccelerometerBias(
        np.array([MAX_PLAUSIBLE_ACCEL_BIAS_MPS2 * 2, 0.0, 0.0])
    ).is_plausible


def test_bias_serializes_with_its_provenance():
    bias = ImuBias(
        gyroscope=GyroscopeBias(np.array([0.01, 0.0, 0.0]), source="rest window"),
        accelerometer=AccelerometerBias(
            np.zeros(3), source="unobservable", observable_axes="gravity_parallel"
        ),
    )
    payload = bias.as_dict()
    assert payload["gyro_bias_source"] == "rest window"
    assert payload["accel_bias_observable_axes"] == "gravity_parallel"


def test_phone_frame_check_guards_the_ordering():
    """A bias must be applied before any rotation, never after."""
    times = np.zeros(2)
    phone_frame_check(_series(np.zeros((2, 3)), times, Frame.PHONE))
    with pytest.raises(ValueError, match="before any frame transformation"):
        phone_frame_check(_series(np.zeros((2, 3)), times, Frame.VEHICLE))


# --- stationary estimation -------------------------------------------------


def test_gyro_bias_is_the_mean_at_rest():
    """At rest the true rate is zero, so the mean is the bias. This one is sound."""
    times, accel, _ = _resting()
    offset = np.array([0.004, -0.002, 0.001])
    gyro = _series(np.tile(offset, (len(times), 1)), times)
    bias = estimate_bias_from_rest(
        gyro=gyro,
        accel=accel,
        mask=np.ones(len(times), dtype=bool),
        gravity=estimate_gravity(accel, gyro=gyro),
        expected_gravity_mps2=STANDARD_GRAVITY_MPS2,
    )
    assert bias.gyroscope.values_rps == pytest.approx(offset, abs=1e-12)
    assert "stationary samples" in bias.gyroscope.source


def test_accelerometer_bias_is_only_observable_along_gravity():
    """The limitation §21 warns about, asserted rather than assumed away.

    A phone at rest with a 0.2 m/s² bias on X reads ``(0.2, 0, g)``. Gravity is
    estimated from that same mean, so it points along ``(0.2, 0, g)`` too — and
    the X component is *indistinguishable* from the phone being tilted by
    0.2/9.81 radians. No stationary data can separate them.

    So the estimate must recover essentially nothing on X, and must say so
    through ``observable_axes``. An implementation that reported 0.2 would be
    reporting a tilt as a bias.
    """
    times, _, gyro = _resting()
    offset = np.array([0.2, 0.0, 0.0])
    accel = _series(np.tile([0.0, 0.0, STANDARD_GRAVITY_MPS2], (len(times), 1)) + offset, times)
    bias = estimate_bias_from_rest(
        gyro=gyro,
        accel=accel,
        mask=np.ones(len(times), dtype=bool),
        gravity=estimate_gravity(accel, gyro=gyro),
        expected_gravity_mps2=STANDARD_GRAVITY_MPS2,
    )
    assert bias.accelerometer.observable_axes == "gravity_parallel"
    # The horizontal offset is not recovered: it is second order in the tilt.
    assert bias.accelerometer.values_mps2[0] == pytest.approx(0.0, abs=0.01)
    assert any("perpendicular components are unobservable" in note for note in bias.notes)


def test_accelerometer_bias_along_gravity_is_recovered():
    """The one component that *is* separable, given the local gravity magnitude."""
    times, _, gyro = _resting()
    excess = 0.05
    accel = _series(np.tile([0.0, 0.0, STANDARD_GRAVITY_MPS2 + excess], (len(times), 1)), times)
    bias = estimate_bias_from_rest(
        gyro=gyro,
        accel=accel,
        mask=np.ones(len(times), dtype=bool),
        gravity=estimate_gravity(accel, gyro=gyro),
        expected_gravity_mps2=STANDARD_GRAVITY_MPS2,
    )
    assert bias.accelerometer.values_mps2[2] == pytest.approx(excess, abs=1e-9)
    assert bias.accelerometer.magnitude_mps2 == pytest.approx(excess, abs=1e-9)


def test_no_expected_gravity_means_no_accelerometer_bias():
    """Without an external magnitude, nothing about the bias is observable."""
    times, accel, gyro = _resting()
    bias = estimate_bias_from_rest(
        gyro=gyro,
        accel=accel,
        mask=np.ones(len(times), dtype=bool),
        gravity=estimate_gravity(accel, gyro=gyro),
        expected_gravity_mps2=None,
    )
    assert bias.accelerometer.magnitude_mps2 == 0.0
    assert any("no component" in note for note in bias.notes)


def test_implausible_estimates_are_rejected_not_returned():
    """A window that was not really stationary must not become a calibration."""
    times, accel, _ = _resting()
    wild = _series(np.tile([1.0, 0.0, 0.0], (len(times), 1)), times)
    bias = estimate_bias_from_rest(
        gyro=wild,
        accel=accel,
        mask=np.ones(len(times), dtype=bool),
        gravity=estimate_gravity(accel),
        expected_gravity_mps2=STANDARD_GRAVITY_MPS2,
    )
    assert bias.gyroscope.magnitude_rps == 0.0
    assert any("rejected a gyroscope bias" in note for note in bias.notes)


def test_missing_inputs_leave_the_bias_at_zero_with_a_reason():
    times, accel, _ = _resting()
    mask = np.ones(len(times), dtype=bool)
    bias = estimate_bias_from_rest(
        gyro=None, accel=accel, mask=mask, expected_gravity_mps2=STANDARD_GRAVITY_MPS2
    )
    assert bias.gyroscope.magnitude_rps == 0.0
    assert any("no gyroscope series" in note for note in bias.notes)

    empty = estimate_bias_from_rest(gyro=None, accel=None, mask=mask)
    assert empty.is_zero
    assert any("no accelerometer series" in note for note in empty.notes)

    unmatched = estimate_bias_from_rest(
        gyro=_series(np.zeros((5, 3)), np.arange(5.0)), accel=accel, mask=mask
    )
    assert any("against a" in note for note in unmatched.notes)


def test_no_stationary_samples_leaves_the_bias_at_zero():
    times, accel, gyro = _resting()
    bias = estimate_bias_from_rest(gyro=gyro, accel=accel, mask=np.zeros(len(times), dtype=bool))
    assert bias.gyroscope.magnitude_rps == 0.0
    assert any("no stationary samples" in note for note in bias.notes)


def test_unavailable_gravity_blocks_the_accelerometer_estimate():
    times, accel, gyro = _resting()
    unavailable = GravityEstimate(
        direction=None,
        magnitude_mps2=None,
        method=GravityMethod.UNAVAILABLE,
        confidence=0.0,
        sample_count=0,
        candidate_count=0,
        direction_spread_rad=None,
        magnitude_error_mps2=None,
    )
    bias = estimate_bias_from_rest(
        gyro=gyro,
        accel=accel,
        mask=np.ones(len(times), dtype=bool),
        gravity=unavailable,
        expected_gravity_mps2=STANDARD_GRAVITY_MPS2,
    )
    assert bias.accelerometer.magnitude_mps2 == 0.0
    assert any("no gravity direction" in note for note in bias.notes)


# --- stationarity detection -------------------------------------------------


def test_rest_is_detected():
    times, accel, gyro = _resting(count=600)
    result = detect_stationarity(accel=accel, gyro=gyro)
    assert result.fraction > 0.95
    assert result.longest_interval_s > 50.0
    assert result.confidence > 0.9


def test_constant_acceleration_is_not_stationary():
    """The condition that earns its place: steady is not the same as at rest.

    A vehicle in smooth cruise passes the variance and rotation checks
    comfortably. Only the comparison against gravity's magnitude rejects it.
    """
    drive = syn.constant_acceleration(acceleration_mps2=3.0, duration_s=30.0)
    result = detect_stationarity(accel=drive.specific_force_phone, gyro=drive.angular_rate_phone)
    assert result.fraction < 0.05


def test_gentle_constant_acceleration_is_a_known_blind_spot():
    """The limitation, asserted rather than left to be discovered later.

    Horizontal acceleration adds to gravity in quadrature, so 1 m/s² raises the
    measured magnitude by only 0.05 m/s² — inside any tolerance loose enough to
    accommodate a real sensor. This detector therefore *cannot* separate gentle
    constant acceleration from rest, and neither can any IMU-only detector.

    The test exists so the boundary is a stated property with a number attached
    rather than a surprise, and so that anyone tempted to feed this into a
    zero-velocity correction sees the reason not to.
    """
    drive = syn.constant_acceleration(acceleration_mps2=1.0, duration_s=30.0)
    result = detect_stationarity(accel=drive.specific_force_phone, gyro=drive.angular_rate_phone)
    assert result.fraction > 0.5, (
        "if this ever fails, the blind spot has been closed and the docstrings "
        "in stationarity.py and the Phase 4 report need updating"
    )
    # And the boundary is where the quadrature says it is.
    boundary = np.sqrt((STANDARD_GRAVITY_MPS2 + 0.15) ** 2 - STANDARD_GRAVITY_MPS2**2)
    assert 1.6 < boundary < 1.8


def test_rotation_prevents_a_stationary_verdict():
    drive = syn.pure_rotation(rate_rps=0.5, duration_s=20.0, rate_hz=50.0)
    result = detect_stationarity(accel=drive.specific_force_phone, gyro=drive.angular_rate_phone)
    assert result.fraction < 0.05


def test_detection_works_without_a_gyroscope_and_says_so():
    times, accel, _ = _resting()
    result = detect_stationarity(accel=accel, gyro=None)
    assert result.fraction > 0.9
    assert any("no gyroscope" in note for note in result.notes)

    mismatched = detect_stationarity(accel=accel, gyro=_series(np.zeros((5, 3)), np.arange(5.0)))
    assert any("samples against" in note for note in mismatched.notes)


def test_longest_window_selects_one_contiguous_run():
    """Bias estimation needs one interval, not scattered quiet samples."""
    times = np.arange(600, dtype="float64") / 10.0
    force = np.tile([0.0, 0.0, STANDARD_GRAVITY_MPS2], (600, 1))
    # A loud middle section splits the rest into two runs of unequal length.
    force[200:260] += np.array([4.0, 0.0, 0.0])
    result = detect_stationarity(accel=_series(force, times), gyro=None)
    window = result.longest_window_mask()
    assert window.any()
    indices = np.flatnonzero(window)
    assert (np.diff(indices) == 1).all(), "the window must be contiguous"


def test_short_runs_are_refused_for_bias_estimation():
    times = np.arange(20, dtype="float64") / 10.0  # 2 s total
    force = np.tile([0.0, 0.0, STANDARD_GRAVITY_MPS2], (20, 1))
    result = detect_stationarity(accel=_series(force, times), gyro=None, window_s=0.5)
    assert not result.longest_window_mask(minimum_s=5.0).any()


def test_detection_rejects_bad_arguments():
    times, accel, _ = _resting(count=4)
    with pytest.raises(ValueError, match="at least one"):
        detect_stationarity(accel=_series(np.zeros((0, 3)), np.zeros(0)))
    with pytest.raises(ValueError, match="window_s"):
        detect_stationarity(accel=accel, window_s=0.0)


def test_stationary_duration_sums_only_contiguous_intervals():
    times = np.array([0.0, 0.1, 0.2, 5.0, 5.1])
    mask = np.array([True, True, False, True, True])
    assert stationary_duration_s(mask, times) == pytest.approx(0.2)
    assert stationary_duration_s(np.zeros(1, dtype=bool), np.zeros(1)) == 0.0


# --- bias applied through the mechanization --------------------------------


def test_a_configured_bias_is_removed_by_the_mechanization():
    """Injecting a bias and configuring the same value must restore the truth."""
    offset = np.array([0.05, -0.03, 0.02])
    clean = syn.constant_acceleration(acceleration_mps2=1.0, duration_s=20.0)
    dirty = syn.constant_acceleration(
        acceleration_mps2=1.0, duration_s=20.0, accel_bias_mps2=offset
    )
    trajectory = mechanize_series(
        initial=clean.initial_state(),
        specific_force=dirty.specific_force_phone,
        angular_rate=dirty.angular_rate_phone,
        orientations=dirty.orientation,
        config=MechanizationConfig(
            alignment_policy=AlignmentPolicy.PHONE_FRAME_DIAGNOSTIC,
            bias=ImuBias(
                gyroscope=GyroscopeBias.zero(),
                accelerometer=AccelerometerBias(offset, source="test"),
            ),
        ),
    )
    assert np.abs(trajectory.velocity_mps - clean.velocity_nav_mps).max() < 1e-11
    assert np.abs(trajectory.position_m - clean.position_nav_m).max() < 1e-9


def test_an_uncorrected_bias_diverges_quadratically():
    """The justification for Phase 5, as an assertion rather than a claim.

    A 0.05 m/s² accelerometer bias left uncorrected must produce ½·a·t² of
    position error — 90 m over 60 s. If this test ever passed with a small
    number, the bias would not be reaching the integrator.
    """
    offset = np.array([0.05, 0.0, 0.0])
    clean = syn.constant_acceleration(acceleration_mps2=0.0, duration_s=60.0)
    dirty = syn.constant_acceleration(
        acceleration_mps2=0.0, duration_s=60.0, accel_bias_mps2=offset
    )
    trajectory = mechanize_series(
        initial=clean.initial_state(),
        specific_force=dirty.specific_force_phone,
        angular_rate=dirty.angular_rate_phone,
        orientations=dirty.orientation,
        config=MechanizationConfig(alignment_policy=AlignmentPolicy.PHONE_FRAME_DIAGNOSTIC),
    )
    expected = 0.5 * 0.05 * 60.0**2
    assert float(np.linalg.norm(trajectory.position_m[-1])) == pytest.approx(expected, rel=1e-6)


def test_gyro_bias_leaks_gravity_into_the_horizontal():
    """A gyro bias tilts the attitude, and a tilt turns gravity into acceleration.

    This is the mechanism behind the real-data drift, isolated: with attitude
    propagated from a biased gyro, a stationary device accumulates position
    error even though it never moved.
    """
    offset = np.array([0.0, 0.01, 0.0])  # ~0.57°/s about phone Y
    clean = syn.stationary(duration_s=60.0, rate_hz=20.0)
    dirty = syn.stationary(duration_s=60.0, rate_hz=20.0, gyro_bias_rps=offset)
    from idr.navigation.mechanization import AttitudeMode

    trajectory = mechanize_series(
        initial=clean.initial_state(),
        specific_force=dirty.specific_force_phone,
        angular_rate=dirty.angular_rate_phone,
        config=MechanizationConfig(
            attitude_mode=AttitudeMode.GYRO_PROPAGATION,
            alignment_policy=AlignmentPolicy.PHONE_FRAME_DIAGNOSTIC,
        ),
    )
    drift = float(np.linalg.norm(trajectory.position_m[-1]))
    assert drift > 100.0, "a 0.57°/s gyro bias must produce large drift, not a small one"


def test_perturbed_adds_to_both_channels():
    base = ImuBias(
        gyroscope=GyroscopeBias(np.array([0.001, 0.0, 0.0]), source="base"),
        accelerometer=AccelerometerBias(np.array([0.01, 0.0, 0.0]), source="base"),
    )
    shifted = perturbed(
        base, gyro_rps=np.array([0.01, 0.0, 0.0]), accel_mps2=np.array([0.0, 0.05, 0.0])
    )
    assert shifted.gyroscope.values_rps == pytest.approx([0.011, 0.0, 0.0])
    assert shifted.accelerometer.values_mps2 == pytest.approx([0.01, 0.05, 0.0])
    assert any("perturbed by" in note for note in shifted.notes)
    assert "no perturbation" in perturbed(base).notes[-1]


def test_gravity_estimate_direction_points_down():
    """Cross-check against Phase 3: the gravity *vector* points down, not up.

    The convention this whole phase depends on, re-asserted here so a change in
    Phase 3 would break a Phase 4 test rather than silently invert Phase 4.
    """
    times, accel, gyro = _resting()
    gravity = estimate_gravity(accel, gyro=gyro)
    assert gravity.direction is not None
    assert gravity.direction.to_array()[2] == pytest.approx(-1.0, abs=1e-9)
    assert gravity.up_direction == Vector3(0.0, 0.0, 1.0, Frame.PHONE)
    assert gravity.magnitude_mps2 == pytest.approx(STANDARD_GRAVITY_MPS2, rel=1e-9)
