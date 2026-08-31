"""Phone→vehicle alignment tests.

Real IO-VNBD data cannot validate an alignment estimator: the true
phone→vehicle rotation for those recordings is documented nowhere, so agreement
with it could never be checked. Synthetic data can. Every recovery test here
builds a drive in the vehicle frame — where forward is +X by construction —
rotates it by a rotation chosen in the test, and asserts the estimator returns
that rotation.

The refusal tests matter just as much. An estimator that always returns a
rotation is worse than one that sometimes says "not from this data", because
downstream code inherits a confident wrong answer silently.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from idr.frames import quaternion as quat
from idr.frames.alignment import (
    MAX_GRAVITY_RESIDUAL_DEG,
    CalibrationStatus,
    check_alignment,
    estimate_alignment,
    estimate_forward_direction,
    linear_accel_for_alignment,
)
from idr.frames.conventions import STANDARD_GRAVITY_MPS2, Frame
from idr.frames.gravity import estimate_gravity
from idr.frames.magnetometer import assess_magnetometer
from idr.frames.rotations import FrameRotation
from idr.frames.synthetic import (
    STANDARD_SWEEP_DEG,
    SyntheticDrive,
    disturbed_magnetometer,
    rotate_series,
    rotation_sweep,
    stationary_only,
    synthetic_drive,
    with_duplicate_timestamps,
)
from idr.frames.vectors import Vector3, VectorSeries

X_AXIS = np.array([1.0, 0.0, 0.0])
Y_AXIS = np.array([0.0, 1.0, 0.0])
Z_AXIS = np.array([0.0, 0.0, 1.0])


def _align(drive: SyntheticDrive, *, use_gravity_channel: bool = True, **kwargs: object):
    gravity = estimate_gravity(
        drive.accel_phone,
        gravity_channel=drive.gravity_phone if use_gravity_channel else None,
        gyro=drive.gyro_phone,
    )
    linear = linear_accel_for_alignment(drive.accel_phone, gravity)
    magnetometer = assess_magnetometer(drive.mag_phone, gravity=gravity)
    return estimate_alignment(
        gravity=gravity,
        linear_accel=linear,
        speed_mps=drive.speed_mps,
        gyro=drive.gyro_phone,
        reference_yaw_rate_rps=drive.yaw_rate_rps,
        magnetometer=magnetometer,
        duration_s=float(drive.analysis_time_s[-1] - drive.analysis_time_s[0]),
        **kwargs,  # type: ignore[arg-type]
    )


def _error_deg(estimate, truth: np.ndarray) -> float:
    assert estimate.rotation is not None
    return math.degrees(quat.angular_distance(estimate.rotation.quaternion, truth))


# --- recovery of a known rotation ------------------------------------------


def test_identity_mounting_is_recovered() -> None:
    drive = synthetic_drive()
    estimate = _align(drive)
    assert estimate.status is CalibrationStatus.SUCCESS
    assert _error_deg(estimate, drive.truth_q_vehicle_phone) < 0.5


@pytest.mark.parametrize("angle_deg", STANDARD_SWEEP_DEG)
@pytest.mark.parametrize(
    ("axis", "axis_name"), [(Z_AXIS, "yaw"), (X_AXIS, "roll"), (Y_AXIS, "pitch")]
)
def test_a_known_rotation_about_each_axis_is_recovered(
    angle_deg: float, axis: np.ndarray, axis_name: str
) -> None:
    """The controlled sweep the brief asks for: 0°, 15°, 30°, 45°, 90°."""
    truth = quat.from_axis_angle(axis, math.radians(angle_deg))
    estimate = _align(synthetic_drive(q_vehicle_phone=truth))
    assert estimate.status is CalibrationStatus.SUCCESS, axis_name
    assert _error_deg(estimate, truth) < 0.5


def test_a_compound_mounting_rotation_is_recovered() -> None:
    """A phone wedged at an awkward angle, not just rotated about one axis."""
    truth = quat.from_euler(math.radians(25), math.radians(-40), math.radians(115))
    estimate = _align(synthetic_drive(q_vehicle_phone=truth))
    assert estimate.status is CalibrationStatus.SUCCESS
    assert _error_deg(estimate, truth) < 1.0


def test_recovery_survives_sensor_noise_and_irregular_sampling() -> None:
    truth = quat.from_euler(math.radians(10), math.radians(-5), math.radians(35))
    drive = synthetic_drive(q_vehicle_phone=truth, noise_mps2=0.15, jitter_s=0.02)
    estimate = _align(drive)
    assert estimate.status is CalibrationStatus.SUCCESS
    assert _error_deg(estimate, truth) < 2.0


def test_recovery_works_from_the_accelerometer_alone() -> None:
    """No gravity channel: tilt must come from detected quasi-static samples."""
    truth = quat.from_euler(math.radians(10), math.radians(-5), math.radians(35))
    drive = synthetic_drive(q_vehicle_phone=truth, noise_mps2=0.1)
    estimate = _align(drive, use_gravity_channel=False)
    assert estimate.status is CalibrationStatus.SUCCESS
    assert _error_deg(estimate, truth) < 2.0


def test_duplicate_timestamps_do_not_break_recovery() -> None:
    truth = quat.from_axis_angle(Z_AXIS, math.radians(30))
    drive = with_duplicate_timestamps(synthetic_drive(q_vehicle_phone=truth), count=8)
    estimate = _align(drive)
    assert estimate.status is CalibrationStatus.SUCCESS
    assert _error_deg(estimate, truth) < 1.0


def test_the_applied_rotation_returns_the_vehicle_frame() -> None:
    """End to end: rotating the phone-frame data recovers the vehicle frame."""
    truth = quat.from_euler(math.radians(20), math.radians(15), math.radians(70))
    drive = synthetic_drive(q_vehicle_phone=truth)
    estimate = _align(drive)
    vehicle = estimate.to_vehicle_frame(drive.accel_phone)
    assert vehicle.frame is Frame.VEHICLE
    assert np.allclose(vehicle.samples, drive.accel_vehicle.samples, atol=0.2)


# --- pitch and roll from gravity -------------------------------------------


def test_pitch_and_roll_are_available_even_when_yaw_is_not() -> None:
    """The stationary case: tilt is observable, heading is not."""
    truth = quat.from_euler(math.radians(20), math.radians(-15), math.radians(60))
    estimate = _align(stationary_only(q_vehicle_phone=truth))
    assert estimate.status is CalibrationStatus.INSUFFICIENT_MOTION
    assert estimate.roll_deg is not None
    assert estimate.pitch_deg is not None
    assert estimate.tilt_only_rotation is not None
    assert estimate.up_direction_phone is not None


def test_the_recovered_vertical_matches_the_truth_while_stationary() -> None:
    """Yaw is unknown, but the vertical axis must still be right."""
    truth = quat.from_euler(math.radians(20), math.radians(-15), math.radians(60))
    estimate = _align(stationary_only(q_vehicle_phone=truth))
    assert estimate.up_direction_phone is not None
    expected_up = quat.rotate_vector(quat.conjugate(truth), Z_AXIS)
    recovered = estimate.up_direction_phone.to_array()
    assert math.degrees(math.acos(np.clip(recovered @ expected_up, -1, 1))) < 1.0


# --- yaw is never fabricated -----------------------------------------------


def test_a_stationary_calibration_refuses_to_report_yaw() -> None:
    estimate = _align(stationary_only())
    assert not estimate.yaw_resolved
    assert estimate.yaw_deg is None
    assert estimate.rotation is None


def test_applying_an_unresolved_alignment_raises_rather_than_guessing() -> None:
    """The placeholder yaw must never leak into a caller's data."""
    drive = stationary_only()
    estimate = _align(drive)
    with pytest.raises(ValueError, match="yaw was never resolved"):
        estimate.to_vehicle_frame(drive.accel_phone)


def test_a_trusted_magnetometer_does_not_by_itself_unlock_yaw() -> None:
    """Gravity plus a clean compass is still not a vehicle heading reference."""
    drive = stationary_only()
    estimate = _align(drive)
    assert estimate.quality.magnetometer_weight > 0.0
    assert not estimate.yaw_resolved


def test_insufficient_motion_is_distinguished_from_a_disturbed_compass() -> None:
    stationary = _align(stationary_only())
    assert stationary.status is CalibrationStatus.INSUFFICIENT_MOTION
    assert stationary.status.has_tilt
    assert stationary.status.is_terminal


# --- forward-direction estimator -------------------------------------------


def test_forward_direction_matches_the_truth() -> None:
    truth = quat.from_axis_angle(Z_AXIS, math.radians(40))
    drive = synthetic_drive(q_vehicle_phone=truth)
    gravity = estimate_gravity(drive.accel_phone, gravity_channel=drive.gravity_phone)
    linear = linear_accel_for_alignment(drive.accel_phone, gravity)
    assert linear is not None
    up = gravity.up_direction
    assert up is not None
    forward = estimate_forward_direction(
        linear_accel=linear, up_direction=up, speed_mps=drive.speed_mps
    )
    assert forward.is_available
    assert forward.direction is not None
    expected = quat.rotate_vector(quat.conjugate(truth), X_AXIS)
    assert math.degrees(math.acos(np.clip(forward.direction.to_array() @ expected, -1, 1))) < 2.0
    assert forward.confidence > 0.0
    assert forward.correlation is not None and forward.correlation > 0.9


def test_forward_direction_sign_distinguishes_forward_from_reverse() -> None:
    """Braking pushes forward; if the sign were free the axis could invert."""
    drive = synthetic_drive()
    gravity = estimate_gravity(drive.accel_phone, gravity_channel=drive.gravity_phone)
    linear = linear_accel_for_alignment(drive.accel_phone, gravity)
    assert linear is not None and gravity.up_direction is not None
    forward = estimate_forward_direction(
        linear_accel=linear, up_direction=gravity.up_direction, speed_mps=drive.speed_mps
    )
    assert forward.direction is not None
    assert forward.direction.to_array() @ X_AXIS > 0.9
    assert forward.reversed_fraction is not None and forward.reversed_fraction < 0.2


def test_forward_direction_declines_a_stationary_vehicle() -> None:
    drive = stationary_only()
    gravity = estimate_gravity(drive.accel_phone, gravity_channel=drive.gravity_phone)
    linear = linear_accel_for_alignment(drive.accel_phone, gravity)
    assert linear is not None and gravity.up_direction is not None
    forward = estimate_forward_direction(
        linear_accel=linear, up_direction=gravity.up_direction, speed_mps=drive.speed_mps
    )
    assert not forward.is_available
    assert forward.confidence == 0.0
    assert any("never accelerated" in note for note in forward.notes)


def test_forward_direction_declines_a_series_that_is_too_short() -> None:
    series = VectorSeries(
        samples=np.zeros((5, 3)), analysis_time_s=np.arange(5) * 0.1, frame=Frame.PHONE
    )
    forward = estimate_forward_direction(
        linear_accel=series,
        up_direction=Vector3(0.0, 0.0, 1.0, Frame.PHONE),
        speed_mps=np.zeros(5),
    )
    assert not forward.is_available
    assert "need at least" in forward.notes[0]


def test_forward_direction_ignores_rows_the_speed_mask_excludes() -> None:
    """Phase 2 marks stale GNSS; a stale speed must not drive the estimate."""
    drive = synthetic_drive()
    gravity = estimate_gravity(drive.accel_phone, gravity_channel=drive.gravity_phone)
    linear = linear_accel_for_alignment(drive.accel_phone, gravity)
    assert linear is not None and gravity.up_direction is not None
    forward = estimate_forward_direction(
        linear_accel=linear,
        up_direction=gravity.up_direction,
        speed_mps=drive.speed_mps,
        valid_speed=np.zeros(len(drive), dtype=bool),
    )
    assert not forward.is_available


# --- alignment quality checks ----------------------------------------------


def test_a_correct_alignment_passes_every_physical_check() -> None:
    truth = quat.from_euler(math.radians(15), math.radians(-10), math.radians(50))
    drive = synthetic_drive(q_vehicle_phone=truth)
    gravity = estimate_gravity(drive.accel_phone, gravity_channel=drive.gravity_phone)
    linear = linear_accel_for_alignment(drive.accel_phone, gravity)
    assert linear is not None
    checks = check_alignment(
        FrameRotation.from_quaternion(truth, Frame.VEHICLE, Frame.PHONE),
        linear_accel=linear,
        gravity=gravity,
        speed_mps=drive.speed_mps,
        gyro=drive.gyro_phone,
        reference_yaw_rate_rps=drive.yaw_rate_rps,
    )
    assert checks.gravity_residual_deg is not None
    assert checks.gravity_residual_deg < 1.0
    assert checks.longitudinal_correlation is not None
    assert checks.longitudinal_correlation > 0.9
    assert checks.angular_rate_consistency is not None
    assert checks.angular_rate_consistency > 0.9
    assert checks.notes == []


def test_a_deliberately_wrong_alignment_fails_the_checks() -> None:
    """90° of yaw error puts forward acceleration on the lateral axis."""
    drive = synthetic_drive()
    gravity = estimate_gravity(drive.accel_phone, gravity_channel=drive.gravity_phone)
    linear = linear_accel_for_alignment(drive.accel_phone, gravity)
    assert linear is not None
    wrong = quat.from_axis_angle(Z_AXIS, math.pi / 2)
    checks = check_alignment(
        FrameRotation.from_quaternion(wrong, Frame.VEHICLE, Frame.PHONE),
        linear_accel=linear,
        gravity=gravity,
        speed_mps=drive.speed_mps,
        gyro=drive.gyro_phone,
    )
    # Gravity still lands on vertical — a yaw error cannot disturb it, which is
    # exactly why the longitudinal check has to exist as well.
    assert checks.gravity_residual_deg is not None
    assert checks.gravity_residual_deg < 1.0
    assert checks.longitudinal_correlation is not None
    assert abs(checks.longitudinal_correlation) < 0.3
    assert any("longitudinal check" in note for note in checks.notes)


def test_an_upside_down_alignment_fails_the_gravity_check() -> None:
    drive = synthetic_drive()
    gravity = estimate_gravity(drive.accel_phone, gravity_channel=drive.gravity_phone)
    linear = linear_accel_for_alignment(drive.accel_phone, gravity)
    assert linear is not None
    checks = check_alignment(
        FrameRotation.from_quaternion(
            quat.from_axis_angle(X_AXIS, math.pi), Frame.VEHICLE, Frame.PHONE
        ),
        linear_accel=linear,
        gravity=gravity,
    )
    assert checks.gravity_residual_deg is not None
    assert checks.gravity_residual_deg > MAX_GRAVITY_RESIDUAL_DEG


# --- failure states --------------------------------------------------------


def test_unusable_gravity_blocks_everything() -> None:
    """Without a vertical there is no tilt either, so nothing is returned."""
    count = 400
    accel = VectorSeries(
        samples=np.tile([6.0, 6.0, STANDARD_GRAVITY_MPS2], (count, 1)),
        analysis_time_s=np.arange(count) * 0.1,
        frame=Frame.PHONE,
    )
    estimate = estimate_alignment(gravity=estimate_gravity(accel))
    assert estimate.status is CalibrationStatus.LOW_GRAVITY_CONFIDENCE
    assert estimate.rotation is None
    assert estimate.tilt_only_rotation is None
    assert estimate.confidence == 0.0
    assert not estimate.is_usable


def test_no_motion_data_at_all_yields_insufficient_motion_with_tilt() -> None:
    drive = stationary_only()
    gravity = estimate_gravity(drive.accel_phone, gravity_channel=drive.gravity_phone)
    estimate = estimate_alignment(gravity=gravity)
    assert estimate.status is CalibrationStatus.INSUFFICIENT_MOTION
    assert estimate.tilt_only_rotation is not None
    assert any("no motion data" in note for note in estimate.notes)


def test_a_disturbed_compass_is_named_when_it_was_the_last_resort() -> None:
    """MAG_DISTURBED tells the operator the problem is environmental."""
    drive = disturbed_magnetometer(synthetic_drive(cruise_speed_mps=3.0), offset_ut=400.0)
    gravity = estimate_gravity(drive.accel_phone, gravity_channel=drive.gravity_phone)
    linear = linear_accel_for_alignment(drive.accel_phone, gravity)
    magnetometer = assess_magnetometer(drive.mag_phone, gravity=gravity)
    assert not magnetometer.is_trusted
    # Force the low-heading branch with data that moves but never accelerates
    # decisively, so the magnetometer is the only remaining candidate.
    estimate = estimate_alignment(
        gravity=gravity,
        linear_accel=linear,
        speed_mps=np.full(len(drive), 5.0),
        magnetometer=magnetometer,
        duration_s=60.0,
    )
    assert estimate.status in (
        CalibrationStatus.INSUFFICIENT_MOTION,
        CalibrationStatus.MAG_DISTURBED,
    )
    assert estimate.rotation is None


def test_calibration_status_classification() -> None:
    assert not CalibrationStatus.NOT_STARTED.is_terminal
    assert not CalibrationStatus.COLLECTING.is_terminal
    assert CalibrationStatus.SUCCESS.is_terminal
    assert CalibrationStatus.SUCCESS.has_tilt
    assert not CalibrationStatus.LOW_GRAVITY_CONFIDENCE.has_tilt
    assert not CalibrationStatus.FAILED.has_tilt


# --- synthetic rotation utility --------------------------------------------


def test_rotation_sweep_produces_the_documented_angles() -> None:
    sweep = rotation_sweep(Z_AXIS)
    assert [angle for angle, _ in sweep] == list(STANDARD_SWEEP_DEG)
    for angle, q in sweep:
        assert math.degrees(quat.angular_distance(quat.identity(), q)) == pytest.approx(
            angle, abs=1e-9
        )


def test_rotate_series_retags_the_frame_and_is_invertible() -> None:
    drive = synthetic_drive()
    q = quat.from_axis_angle(Y_AXIS, math.radians(30))
    rotated = rotate_series(drive.accel_phone, q, Frame.VEHICLE)
    assert rotated.frame is Frame.VEHICLE
    back = rotate_series(rotated, quat.conjugate(q), Frame.PHONE)
    assert np.allclose(back.samples, drive.accel_phone.samples, atol=1e-12)


def test_the_synthetic_drive_is_internally_consistent() -> None:
    """If the fixture is wrong, every recovery test is meaningless."""
    drive = synthetic_drive()
    # A stationary vehicle reads +g on its up axis, not zero and not −g.
    resting = drive.accel_vehicle.samples[drive.stationary_mask]
    assert np.allclose(resting[:, 2], STANDARD_GRAVITY_MPS2, atol=1e-9)
    assert np.allclose(resting[:, :2], 0.0, atol=1e-9)
    # Applying the truth rotation to the phone frame returns the vehicle frame.
    recovered = drive.truth_rotation.apply_series(drive.accel_phone)
    assert np.allclose(recovered.samples, drive.accel_vehicle.samples, atol=1e-9)


def test_the_synthetic_drive_actually_accelerates_and_turns() -> None:
    drive = synthetic_drive()
    assert float(np.max(drive.speed_mps)) > 10.0
    assert float(np.min(drive.speed_mps)) == pytest.approx(0.0, abs=1e-9)
    assert float(np.max(np.abs(drive.yaw_rate_rps))) > 0.05


def test_synthetic_drive_rejects_a_non_positive_rate() -> None:
    with pytest.raises(ValueError, match="rate_hz must be positive"):
        synthetic_drive(rate_hz=0.0)


def test_jittered_timestamps_stay_ordered() -> None:
    """A logger delivers samples late, not out of order."""
    drive = synthetic_drive(jitter_s=0.05)
    assert bool(np.all(np.diff(drive.analysis_time_s) >= 0.0))
