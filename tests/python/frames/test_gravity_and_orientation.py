"""Gravity, magnetometer-quality and orientation tests.

The single most important assertion in this file is the sign one: both the
accelerometer and the gravity channel read specific force, so both point *away*
from gravity at rest, and the estimator must negate them. Getting that backwards
inverts the vertical axis and flips every recovered heading by 180° — which is
exactly what a synthetic round trip caught during development, after the
gravity-channel path and the accelerometer path silently disagreed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from idr.frames import quaternion as quat
from idr.frames.conventions import STANDARD_GRAVITY_MPS2, Frame
from idr.frames.gravity import (
    GravityMethod,
    estimate_gravity,
    linear_acceleration,
    low_dynamic_mask,
)
from idr.frames.magnetometer import (
    MagnetometerIssue,
    assess_magnetometer,
)
from idr.frames.orientation import AxisSupport, estimate_orientation
from idr.frames.synthetic import (
    disturbed_magnetometer,
    stationary_only,
    synthetic_drive,
    with_duplicate_timestamps,
)
from idr.frames.vectors import Vector3, VectorSeries

Z_AXIS = np.array([0.0, 0.0, 1.0])


def _series(samples: np.ndarray, rate_hz: float = 10.0) -> VectorSeries:
    return VectorSeries(
        samples=samples,
        analysis_time_s=np.arange(samples.shape[0], dtype="float64") / rate_hz,
        frame=Frame.PHONE,
    )


def _resting(direction: np.ndarray, count: int = 200) -> VectorSeries:
    """A phone at rest: specific force of 1 g pointing *away* from gravity."""
    unit = direction / np.linalg.norm(direction)
    return _series(np.tile(unit * STANDARD_GRAVITY_MPS2, (count, 1)))


# --- gravity sign ----------------------------------------------------------


def test_gravity_points_opposite_to_the_accelerometer_at_rest() -> None:
    """A phone flat on a table reads +g on Z; gravity points −Z."""
    estimate = estimate_gravity(_resting(Z_AXIS))
    assert estimate.is_available
    assert estimate.direction is not None
    assert np.allclose(estimate.direction.to_array(), [0.0, 0.0, -1.0], atol=1e-12)
    assert estimate.magnitude_mps2 == pytest.approx(STANDARD_GRAVITY_MPS2)


def test_up_direction_is_the_negation_of_gravity() -> None:
    estimate = estimate_gravity(_resting(Z_AXIS))
    up = estimate.up_direction
    assert up is not None
    assert np.allclose(up.to_array(), [0.0, 0.0, 1.0], atol=1e-12)


def test_gravity_channel_and_accelerometer_agree_in_sign() -> None:
    """The two paths must produce the same physical direction.

    Verified against the real dataset: over 66 segments with a quasi-static
    interval the median cosine between the two channels is +0.999.
    """
    accel = _resting(np.array([0.1, 0.2, 0.97]))
    from_accel = estimate_gravity(accel)
    from_channel = estimate_gravity(accel, gravity_channel=accel)
    assert from_channel.method is GravityMethod.GRAVITY_CHANNEL
    assert from_accel.method is GravityMethod.ACCEL_LOW_DYNAMIC
    assert from_accel.direction is not None
    assert from_channel.direction is not None
    assert from_accel.direction.angle_to(from_channel.direction) < 1e-9


@pytest.mark.parametrize("tilt_deg", [0.0, 15.0, 30.0, 45.0, 90.0])
def test_known_tilt_produces_the_expected_gravity_direction(tilt_deg: float) -> None:
    """Rotate a resting phone by a known angle; gravity must follow exactly."""
    q = quat.from_axis_angle(np.array([1.0, 0.0, 0.0]), math.radians(tilt_deg))
    rotated_up = quat.rotate_vector(q, Z_AXIS)
    estimate = estimate_gravity(_resting(rotated_up))
    assert estimate.direction is not None
    assert np.allclose(estimate.direction.to_array(), -rotated_up, atol=1e-9)
    assert estimate.tilt_from_phone_z_deg() == pytest.approx(tilt_deg, abs=1e-6)


# --- gravity vs raw acceleration -------------------------------------------


def test_aggressive_motion_is_excluded_from_the_gravity_estimate() -> None:
    """The whole reason the low-dynamic filter exists."""
    resting = np.tile([0.0, 0.0, STANDARD_GRAVITY_MPS2], (200, 1))
    braking = np.tile([6.0, 0.0, STANDARD_GRAVITY_MPS2], (200, 1))
    estimate = estimate_gravity(_series(np.vstack([resting, braking])))
    assert estimate.direction is not None
    # Only the resting half qualifies, so gravity stays vertical rather than
    # tilting toward the braking direction.
    assert np.allclose(estimate.direction.to_array(), [0.0, 0.0, -1.0], atol=1e-9)
    assert estimate.sample_count == 200
    assert estimate.candidate_count == 400


def test_gravity_is_unavailable_when_nothing_is_quasi_static() -> None:
    """A refusal, not a mean of whatever the accelerometer happened to read."""
    moving = np.tile([5.0, 5.0, STANDARD_GRAVITY_MPS2], (300, 1))
    estimate = estimate_gravity(_series(moving))
    assert not estimate.is_available
    assert estimate.method is GravityMethod.UNAVAILABLE
    assert estimate.confidence == 0.0
    assert any("below the minimum" in note for note in estimate.notes)


def test_gravity_is_unavailable_with_too_few_samples() -> None:
    estimate = estimate_gravity(_resting(Z_AXIS, count=5))
    assert not estimate.is_available


def test_gravity_is_unavailable_with_no_samples() -> None:
    estimate = estimate_gravity(_series(np.zeros((0, 3))))
    assert not estimate.is_available
    assert "no accelerometer samples" in estimate.notes


def test_a_steady_turn_is_rejected_only_when_the_gyro_is_present() -> None:
    """‖a‖ can stay near 1 g through a turn while the direction sweeps."""
    count = 300
    angles = np.linspace(0.0, 1.5, count)
    samples = STANDARD_GRAVITY_MPS2 * np.column_stack(
        [np.sin(angles) * 0.3, np.zeros(count), np.cos(angles * 0.3)]
    )
    accel = _series(samples)
    gyro = _series(np.tile([0.0, 0.5, 0.0], (count, 1)))

    without_gyro = estimate_gravity(accel)
    with_gyro = estimate_gravity(accel, gyro=gyro)
    assert without_gyro.is_available
    assert any("magnitude test alone" in note for note in without_gyro.notes)
    assert not with_gyro.is_available


def test_confidence_falls_when_the_contributing_samples_disagree() -> None:
    rng = np.random.default_rng(5)
    steady = estimate_gravity(_resting(Z_AXIS))
    noisy_samples = np.tile([0.0, 0.0, STANDARD_GRAVITY_MPS2], (300, 1))
    noisy_samples += rng.normal(0.0, 0.35, size=noisy_samples.shape)
    noisy = estimate_gravity(_series(noisy_samples))
    assert steady.confidence > noisy.confidence
    assert steady.direction_spread_rad == pytest.approx(0.0, abs=1e-12)
    assert noisy.direction_spread_rad is not None
    assert noisy.direction_spread_rad > 0.0


def test_linear_acceleration_recovers_the_true_acceleration() -> None:
    """a = f + g. Subtracting the gravity vector instead of adding it doubles
    the vertical term rather than cancelling it."""
    resting = np.tile([0.0, 0.0, STANDARD_GRAVITY_MPS2], (200, 1))
    accel = _series(resting)
    linear = linear_acceleration(accel, estimate_gravity(accel))
    assert np.allclose(linear.samples[0], [0.0, 0.0, 0.0], atol=1e-9)
    assert any("gravity removed" in note for note in linear.notes)

    pushed = np.tile([4.0, 0.0, STANDARD_GRAVITY_MPS2], (200, 1))
    combined = _series(np.vstack([resting, pushed]))
    estimate = estimate_gravity(combined)
    recovered = linear_acceleration(combined, estimate)
    assert np.allclose(recovered.samples[-1], [4.0, 0.0, 0.0], atol=1e-9)


def test_the_magnitude_gate_cannot_see_small_perpendicular_acceleration() -> None:
    """A documented blind spot, asserted so it cannot change unnoticed.

    Acceleration at right angles to gravity changes ‖a‖ only as
    ``√(g² + a²) − g``, so a 2 m/s² horizontal push shifts the magnitude by
    0.2 m/s² and slips under the 0.5 m/s² tolerance. The estimate then tilts.
    """
    resting = np.tile([0.0, 0.0, STANDARD_GRAVITY_MPS2], (100, 1))
    pushed = np.tile([2.0, 0.0, STANDARD_GRAVITY_MPS2], (100, 1))
    accel = _series(np.vstack([resting, pushed]))

    mask = low_dynamic_mask(accel)
    assert bool(mask.all()), "the perpendicular push is expected to pass the gate"

    estimate = estimate_gravity(accel)
    assert estimate.direction is not None
    tilt = math.degrees(estimate.direction.angle_to(Vector3(0.0, 0.0, -1.0, Frame.PHONE)))
    assert 3.0 < tilt < 8.0, f"expected a few degrees of induced tilt, got {tilt:.2f}°"
    # The spread is what exposes it: the samples visibly disagree.
    assert estimate.direction_spread_rad is not None
    assert estimate.direction_spread_rad > 0.05
    assert estimate.confidence < 0.9


def test_linear_acceleration_refuses_an_unavailable_estimate() -> None:
    accel = _series(np.tile([5.0, 5.0, STANDARD_GRAVITY_MPS2], (300, 1)))
    with pytest.raises(ValueError, match="without an available estimate"):
        linear_acceleration(accel, estimate_gravity(accel))


def test_low_dynamic_mask_on_empty_input() -> None:
    assert low_dynamic_mask(_series(np.zeros((0, 3)))).size == 0


# --- magnetometer quality --------------------------------------------------


def _clean_field(count: int = 300) -> VectorSeries:
    dip = math.radians(60.0)
    field = 48.0 * np.array([0.0, math.cos(dip), -math.sin(dip)])
    return _series(np.tile(field, (count, 1)))


def test_a_clean_earth_field_is_trusted() -> None:
    quality = assess_magnetometer(_clean_field(), gravity=estimate_gravity(_resting(Z_AXIS)))
    assert quality.is_trusted
    assert quality.issues is MagnetometerIssue.NONE
    assert quality.weight > 0.9
    assert quality.describe() == "no issues detected"


def test_an_implausible_field_strength_is_flagged() -> None:
    quality = assess_magnetometer(_series(np.tile([300.0, 0.0, 0.0], (200, 1))))
    assert quality.issues & MagnetometerIssue.MAGNITUDE_OUT_OF_RANGE
    assert "magnitude out of range" in quality.describe()


def test_a_wildly_varying_field_is_distrusted_entirely() -> None:
    """Two independent signs of disturbance drop the magnetometer to zero."""
    rng = np.random.default_rng(9)
    samples = rng.normal(0.0, 60.0, size=(300, 3))
    quality = assess_magnetometer(_series(samples))
    assert quality.weight == 0.0
    assert not quality.is_trusted


def test_an_abrupt_direction_change_is_flagged() -> None:
    field = np.tile([0.0, 40.0, -20.0], (200, 1))
    field[100:] = [0.0, -40.0, 20.0]  # instantaneous 180° reversal
    quality = assess_magnetometer(_series(field))
    assert quality.issues & MagnetometerIssue.ABRUPT_DIRECTION_CHANGE


def test_a_structurally_absent_magnetometer_is_distinguished_from_a_disturbed_one() -> None:
    absent = assess_magnetometer(_clean_field(), available=False)
    assert absent.issues & MagnetometerIssue.UNUSABLE
    assert "no magnetometer channel" in absent.notes[0]
    disturbed = assess_magnetometer(_series(np.tile([300.0, 0.0, 0.0], (200, 1))))
    assert not disturbed.issues & MagnetometerIssue.UNUSABLE


def test_missing_or_degenerate_magnetometer_input() -> None:
    assert assess_magnetometer(None).weight == 0.0
    assert assess_magnetometer(_series(np.zeros((0, 3)))).weight == 0.0
    assert assess_magnetometer(_series(np.zeros((1, 3)))).weight == 0.0
    assert assess_magnetometer(_series(np.zeros((100, 3)))).weight == 0.0


def test_a_steady_in_vehicle_bias_is_caught() -> None:
    """The realistic failure: a constant added field looks perfectly stable."""
    drive = disturbed_magnetometer(synthetic_drive(), offset_ut=80.0)
    gravity = estimate_gravity(drive.accel_phone, gravity_channel=drive.gravity_phone)
    quality = assess_magnetometer(drive.mag_phone, gravity=gravity)
    assert quality.weight < 1.0
    assert quality.issues != MagnetometerIssue.NONE


def test_dip_is_not_checked_without_a_gravity_estimate() -> None:
    quality = assess_magnetometer(_clean_field(), gravity=None)
    assert quality.dip_variation_deg is None
    assert any("inclination" in note for note in quality.notes)


# --- orientation -----------------------------------------------------------


def test_orientation_needs_an_accelerometer() -> None:
    with pytest.raises(ValueError, match="needs an accelerometer"):
        estimate_orientation(gyro=None, accel=None)


def test_a_level_stationary_phone_reports_zero_roll_and_pitch() -> None:
    drive = stationary_only()
    track = estimate_orientation(
        gyro=drive.gyro_phone, accel=drive.accel_phone, gravity_channel=drive.gravity_phone
    )
    estimate = track.at(len(track) - 1)
    assert estimate.roll_deg == pytest.approx(0.0, abs=0.5)
    assert estimate.pitch_deg == pytest.approx(0.0, abs=0.5)
    assert estimate.euler_well_conditioned


def test_roll_and_pitch_are_observed_while_yaw_is_only_dead_reckoned() -> None:
    """The honest description of a phone in a car with no trusted compass."""
    drive = synthetic_drive()
    track = estimate_orientation(
        gyro=drive.gyro_phone,
        accel=drive.accel_phone,
        gravity_channel=drive.gravity_phone,
        mag=None,
    )
    assert track.roll_support is AxisSupport.OBSERVED
    assert track.pitch_support is AxisSupport.OBSERVED
    assert track.yaw_support is AxisSupport.DEAD_RECKONED
    assert any("drifts in absolute terms" in note for note in track.notes)


def test_yaw_is_observed_when_the_magnetometer_is_trusted() -> None:
    drive = synthetic_drive()
    track = estimate_orientation(
        gyro=drive.gyro_phone,
        accel=drive.accel_phone,
        mag=drive.mag_phone,
        gravity_channel=drive.gravity_phone,
    )
    assert track.yaw_support is AxisSupport.OBSERVED
    assert track.mean_confidence() > 0.5


def test_a_disturbed_magnetometer_is_excluded_and_yaw_downgraded() -> None:
    drive = disturbed_magnetometer(synthetic_drive(), offset_ut=300.0)
    track = estimate_orientation(
        gyro=drive.gyro_phone,
        accel=drive.accel_phone,
        mag=drive.mag_phone,
        gravity_channel=drive.gravity_phone,
    )
    assert not track.magnetometer.is_trusted
    assert track.yaw_support is AxisSupport.DEAD_RECKONED
    assert any("magnetometer excluded" in note for note in track.notes)


def test_confidence_is_capped_when_yaw_has_no_absolute_reference() -> None:
    """An excellent roll/pitch with a free-running yaw is not a confident
    3-DOF orientation, and must not be reported as one."""
    drive = synthetic_drive()
    track = estimate_orientation(
        gyro=drive.gyro_phone,
        accel=drive.accel_phone,
        mag=None,
        gravity_channel=drive.gravity_phone,
    )
    assert track.mean_confidence() <= 0.5


def test_yaw_is_unresolved_without_a_gyro_or_a_magnetometer() -> None:
    drive = synthetic_drive()
    track = estimate_orientation(gyro=None, accel=drive.accel_phone, mag=None)
    assert track.yaw_support is AxisSupport.UNRESOLVED
    assert any("yaw is unresolved" in note for note in track.notes)


def test_gyro_integration_tracks_a_known_rotation() -> None:
    """90° of yaw at a constant rate, integrated over actual timestamps."""
    count = 900
    rate_hz = 100.0
    omega = math.radians(90.0) / (count / rate_hz)
    gyro = _series(np.tile([0.0, 0.0, omega], (count, 1)), rate_hz)
    accel = _series(np.tile([0.0, 0.0, STANDARD_GRAVITY_MPS2], (count, 1)), rate_hz)
    track = estimate_orientation(gyro=gyro, accel=accel, mag=None)
    _, _, yaw = quat.to_euler(track.quaternions[-1])
    assert math.degrees(yaw) == pytest.approx(90.0, abs=1.0)
    assert track.integrated_steps == count - 1


def test_duplicate_timestamps_are_not_integrated_across() -> None:
    """Phase 2 retains duplicates, so Δt = 0 reaches the filter legitimately."""
    drive = with_duplicate_timestamps(synthetic_drive(), count=5)
    track = estimate_orientation(
        gyro=drive.gyro_phone, accel=drive.accel_phone, gravity_channel=drive.gravity_phone
    )
    assert track.skipped_steps >= 5
    assert np.isfinite(track.quaternions).all()
    assert any("duplicate timestamp" in note for note in track.notes)


def test_irregular_sampling_uses_actual_intervals() -> None:
    """A jittered clock must not change the integrated angle."""
    regular = synthetic_drive(jitter_s=0.0)
    jittered = synthetic_drive(jitter_s=0.03)
    tracks = [
        estimate_orientation(
            gyro=drive.gyro_phone, accel=drive.accel_phone, gravity_channel=drive.gravity_phone
        )
        for drive in (regular, jittered)
    ]
    angle = quat.angular_distance(tracks[0].quaternions[-1], tracks[1].quaternions[-1])
    assert math.degrees(angle) < 5.0


def test_a_saturated_gyroscope_sample_is_not_integrated() -> None:
    count = 200
    gyro_samples = np.zeros((count, 3))
    gyro_samples[100] = [0.0, 0.0, 500.0]  # far beyond any real MEMS range
    accel = _series(np.tile([0.0, 0.0, STANDARD_GRAVITY_MPS2], (count, 1)))
    track = estimate_orientation(gyro=_series(gyro_samples), accel=accel, mag=None)
    assert track.saturated_samples == 1
    assert any("saturation" in note for note in track.notes)


def test_a_long_gap_is_not_bridged_by_integration() -> None:
    count = 100
    times = np.arange(count, dtype="float64") * 0.1
    times[50:] += 30.0  # a 30-second hole
    accel = VectorSeries(
        samples=np.tile([0.0, 0.0, STANDARD_GRAVITY_MPS2], (count, 1)),
        analysis_time_s=times,
        frame=Frame.PHONE,
    )
    gyro = VectorSeries(
        samples=np.tile([0.0, 0.0, 0.2], (count, 1)),
        analysis_time_s=times,
        frame=Frame.PHONE,
    )
    track = estimate_orientation(gyro=gyro, accel=accel, mag=None)
    assert track.skipped_steps >= 1


def test_orientation_track_exposes_euler_and_axis_support() -> None:
    drive = synthetic_drive()
    track = estimate_orientation(
        gyro=drive.gyro_phone, accel=drive.accel_phone, gravity_channel=drive.gravity_phone
    )
    angles = track.euler_deg()
    assert angles.shape == (len(track), 3)
    assert np.isfinite(angles).all()
    support = track.at(0).axis_support()
    assert set(support) == {"roll", "pitch", "yaw"}
    assert np.allclose(track.at(0).rotation_matrix, quat.to_rotation_matrix(track.quaternions[0]))
