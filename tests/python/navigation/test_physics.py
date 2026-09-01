"""The physics the mechanization has to get right, tested against closed forms.

These are the tests §32 of the phase brief asks for, and they are the only
place in Phase 4 where a correctness claim can be made — real IO-VNBD has no
ground truth for any of these quantities.

Two of them exist specifically to fail on a single-character change:
``test_gravity_sign_regression`` and ``test_linear_acceleration_adds_gravity``.
Phase 3 shipped both of those bugs at one point, neither produced an obviously
wrong number, and one produced a 180° heading error.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from idr.frames import quaternion as quat
from idr.frames.conventions import STANDARD_GRAVITY_MPS2, Frame
from idr.navigation import synthetic as syn
from idr.navigation.gravity import ConstantGravity, WGS84Gravity, linear_acceleration_nav
from idr.navigation.mechanization import (
    AlignmentPolicy,
    AttitudeMode,
    MechanizationConfig,
    navigation_from_vehicle,
    propagate_attitude,
)
from idr.navigation.reference_alignment import attitude_error_deg
from idr.navigation.trajectory import mechanize_series


def run(drive, mode=AttitudeMode.PHASE3_FILTER, **config_kwargs):
    """Propagate a synthetic drive from its exact initial state."""
    return mechanize_series(
        initial=drive.initial_state(),
        specific_force=drive.specific_force_phone,
        angular_rate=drive.angular_rate_phone,
        orientations=drive.orientation,
        config=MechanizationConfig(
            attitude_mode=mode,
            alignment_policy=AlignmentPolicy.PHONE_FRAME_DIAGNOSTIC,
            **config_kwargs,
        ),
    )


# --- Test 1 & 6: stationary and the gravity sign --------------------------


def test_stationary_stays_put():
    """Test 1. Velocity and position must not move over 30 s at rest."""
    drive = syn.stationary(duration_s=30.0)
    trajectory = run(drive)
    assert np.abs(trajectory.velocity_mps).max() == pytest.approx(0.0, abs=1e-12)
    assert np.abs(trajectory.position_m).max() == pytest.approx(0.0, abs=1e-12)


def test_gravity_sign_regression():
    """Test 6. ``f + g = 0`` at rest, so compensated acceleration is exactly zero.

    This is the test that must fail if ``+ g`` becomes ``- g``. A resting level
    device reads +9.80665 m/s² upward; adding gravity cancels it, subtracting it
    doubles it. The assertion below is tight enough that the difference is not a
    matter of tolerance.
    """
    drive = syn.stationary(duration_s=10.0)
    trajectory = run(drive)
    acceleration = trajectory.acceleration_nav[1:]  # sample 0 carries no acceleration
    assert np.abs(acceleration).max() < 1e-12, (
        "a resting device must compensate to zero acceleration; a non-zero result "
        "here means gravity was subtracted instead of added, or the model's sign is wrong"
    )
    # And state the failure mode explicitly, so the test documents what it guards.
    wrong = linear_acceleration_nav(drive.specific_force_phone.samples[0], -drive.gravity_nav)
    assert np.linalg.norm(wrong) == pytest.approx(2.0 * STANDARD_GRAVITY_MPS2, rel=1e-9)


def test_resting_specific_force_points_up():
    """A level resting phone reads +g on its up axis, not −g and not zero."""
    drive = syn.stationary(duration_s=2.0)
    sample = drive.specific_force_phone.samples[0]
    assert sample[2] == pytest.approx(STANDARD_GRAVITY_MPS2, rel=1e-12)
    assert np.linalg.norm(sample[:2]) == pytest.approx(0.0, abs=1e-12)


def test_linear_acceleration_adds_gravity():
    """``a = f + g``, verified against the one case a reader can check by hand."""
    gravity = ConstantGravity().acceleration_nav()
    assert gravity[2] == pytest.approx(-STANDARD_GRAVITY_MPS2)
    at_rest = np.array([0.0, 0.0, STANDARD_GRAVITY_MPS2])
    assert linear_acceleration_nav(at_rest, gravity) == pytest.approx(np.zeros(3), abs=1e-12)


def test_gravity_model_is_injectable():
    """A substituted model changes the compensation without touching mechanization."""
    constant = ConstantGravity(9.79)
    assert constant.acceleration_nav()[2] == pytest.approx(-9.79)
    somigliana = WGS84Gravity(default_latitude_deg=52.0)
    magnitude = abs(float(somigliana.acceleration_nav()[2]))
    # Normal gravity at 52°N is ~9.8127 m/s²; well inside the equator-to-pole span.
    assert 9.80 < magnitude < 9.82
    assert abs(WGS84Gravity().acceleration_nav()[2]) == pytest.approx(9.7803253359, rel=1e-9)


def test_gravity_model_rejects_nonsense():
    with pytest.raises(ValueError, match="finite and positive"):
        ConstantGravity(0.0)
    with pytest.raises(ValueError, match="finite and positive"):
        ConstantGravity(float("nan"))


# --- Test 2 & 3: the integration is exact ---------------------------------


def test_constant_acceleration_matches_closed_form():
    """Test 2. ``v = at`` and ``p = ½at²`` to machine precision."""
    drive = syn.constant_acceleration(acceleration_mps2=1.0, duration_s=20.0)
    trajectory = run(drive)
    elapsed = trajectory.analysis_time_s - trajectory.analysis_time_s[0]

    assert trajectory.velocity_mps[:, 0] == pytest.approx(1.0 * elapsed, abs=1e-12)
    assert trajectory.position_m[:, 0] == pytest.approx(0.5 * elapsed**2, abs=1e-9)
    # Trapezoidal integration is exact for a linearly varying rate, so this is
    # float64 rounding rather than a tolerance chosen to make the test pass.
    assert np.abs(trajectory.velocity_mps - drive.velocity_nav_mps).max() < 1e-12
    assert np.abs(trajectory.position_m - drive.position_nav_m).max() < 1e-9


def test_constant_velocity_matches_closed_form():
    """Test 3. ``p = vt``, and the sensors read gravity alone."""
    drive = syn.constant_velocity(speed_mps=15.0, duration_s=20.0)
    trajectory = run(drive)
    elapsed = trajectory.analysis_time_s - trajectory.analysis_time_s[0]

    assert trajectory.velocity_mps[:, 0] == pytest.approx(np.full(len(trajectory), 15.0))
    assert trajectory.position_m[:, 0] == pytest.approx(15.0 * elapsed, abs=1e-9)
    assert np.abs(trajectory.acceleration_nav[1:]).max() < 1e-12


def test_deceleration_to_rest():
    """A braking profile ends where the closed form says, not merely nearby."""
    drive = syn.analytic_trajectory(
        acceleration_nav_mps2=[-2.0, 0.0, 0.0],
        initial_velocity_nav_mps=[20.0, 0.0, 0.0],
        duration_s=10.0,
    )
    trajectory = run(drive)
    assert trajectory.velocity_mps[-1, 0] == pytest.approx(0.0, abs=1e-11)
    assert trajectory.position_m[-1, 0] == pytest.approx(100.0, abs=1e-8)


# --- Test 4: attitude ------------------------------------------------------


def test_pure_rotation_tracks_attitude_and_stays_put():
    """Test 4. Attitude follows the gyro; position and velocity do not move.

    The demanding part is the second half: the specific force sweeps through the
    whole phone frame as the device turns, and must still resolve to zero
    navigation acceleration at every instant.
    """
    drive = syn.pure_rotation(rate_rps=0.3, duration_s=20.0, rate_hz=50.0)
    trajectory = run(drive, mode=AttitudeMode.GYRO_PROPAGATION)

    errors = attitude_error_deg(trajectory.orientation, drive.orientation)
    assert errors.max() < 1e-9, "gyro propagation must reproduce a constant-rate rotation"
    assert np.abs(trajectory.velocity_mps).max() < 1e-9
    assert np.abs(trajectory.position_m).max() < 1e-8


def test_attitude_increment_is_right_multiplied():
    """A body-frame increment must not be applied about navigation axes.

    Starting from a non-trivial attitude, right- and left-multiplication give
    genuinely different rotations. Both are unit quaternions and both look fine;
    only one is correct, and this pins which.
    """
    start = quat.from_euler(0.3, -0.4, 1.2)
    rate = np.array([0.0, 0.0, 0.5])
    dt = 0.1
    propagated = propagate_attitude(start, rate, dt)

    increment = quat.from_axis_angle([0.0, 0.0, 1.0], 0.05)
    assert quat.allclose(propagated, quat.multiply(start, increment))
    left = quat.multiply(increment, start)
    assert not quat.allclose(propagated, left), (
        "right- and left-multiplication must differ here, or the test proves nothing"
    )


def test_attitude_propagation_handles_zero_rate():
    """A zero rate has no axis; the increment must be identity, not a NaN."""
    start = quat.from_euler(0.1, 0.2, 0.3)
    assert quat.allclose(propagate_attitude(start, np.zeros(3), 0.1), start)
    assert quat.allclose(propagate_attitude(start, np.array([1e-18, 0.0, 0.0]), 0.1), start)


def test_attitude_propagation_is_exact_for_large_angles():
    """The exponential map is not a small-angle approximation."""
    start = quat.identity()
    rate = np.array([0.0, 0.0, math.pi])  # 180°/s
    after = propagate_attitude(start, rate, 0.5)  # a quarter turn
    expected = quat.from_axis_angle([0.0, 0.0, 1.0], math.pi / 2.0)
    assert quat.allclose(after, expected)


def test_attitude_propagation_rejects_bad_input():
    start = quat.identity()
    with pytest.raises(ValueError, match="shape"):
        propagate_attitude(start, np.zeros(2), 0.1)
    with pytest.raises(ValueError, match="finite"):
        propagate_attitude(start, np.array([np.nan, 0.0, 0.0]), 0.1)
    with pytest.raises(ValueError, match="finite"):
        propagate_attitude(start, np.zeros(3), float("inf"))


def test_propagated_quaternion_stays_normalized():
    """Ten thousand steps must not let the norm drift."""
    q = quat.identity()
    rate = np.array([0.1, -0.2, 0.3])
    for _ in range(10_000):
        q = propagate_attitude(q, rate, 0.01)
    assert quat.norm(q) == pytest.approx(1.0, abs=1e-12)


# --- Test 5: the mounting rotation ----------------------------------------


def test_known_mounting_rotation_is_recovered():
    """Test 5. A known ``R_vehicle_phone`` yields the right vehicle-frame force."""
    mount = quat.from_euler(math.radians(20.0), math.radians(-35.0), math.radians(110.0))
    tilted = quat.from_euler(0.2, -0.1, 0.9)
    drive = syn.constant_acceleration(
        acceleration_mps2=1.2,
        duration_s=15.0,
        q_navigation_phone=tilted,
        q_vehicle_phone=mount,
    )
    trajectory = run(drive)
    assert np.abs(trajectory.velocity_mps - drive.velocity_nav_mps).max() < 1e-12
    assert np.abs(trajectory.position_m - drive.position_nav_m).max() < 1e-9

    # And the vehicle-frame force is the phone force rotated by the mount.
    expected = drive.specific_force_phone.samples @ drive.alignment_rotation.matrix.T
    assert drive.specific_force_vehicle() == pytest.approx(expected)


def test_navigation_from_vehicle_composes_through_the_phone_frame():
    """``R_nav_vehicle = R_nav_phone ∘ R_phone_vehicle``, with the frames checked."""
    q_nav_phone = quat.from_euler(0.2, -0.3, 0.5)
    mount = syn.analytic_trajectory(
        q_vehicle_phone=quat.from_euler(0.1, 0.2, -0.4)
    ).alignment_rotation

    composed = navigation_from_vehicle(q_nav_phone, mount)
    assert composed.to_frame is Frame.NAVIGATION
    assert composed.from_frame is Frame.VEHICLE

    expected = quat.to_rotation_matrix(q_nav_phone) @ mount.matrix.T
    assert composed.matrix == pytest.approx(expected)


def test_navigation_from_vehicle_rejects_the_wrong_rotation():
    """Handing it ``R_phone_vehicle`` must raise, not silently invert."""
    mount = syn.analytic_trajectory(
        q_vehicle_phone=quat.from_euler(0.1, 0.2, -0.4)
    ).alignment_rotation
    with pytest.raises(ValueError, match="expected R_vehicle_phone"):
        navigation_from_vehicle(quat.identity(), mount.inverse())


def test_a_transposed_mount_gives_a_different_answer():
    """Guards against a transpose that would pass every orthonormality check."""
    mount = quat.from_euler(math.radians(30.0), 0.0, math.radians(45.0))
    rotation = syn.analytic_trajectory(q_vehicle_phone=mount).alignment_rotation
    force = np.array([1.0, 2.0, 9.0])
    assert not np.allclose(rotation.matrix @ force, rotation.matrix.T @ force)


def test_wgs84_altitude_and_description_branches():
    """The substituted model's altitude term and its self-description.

    Free-air gravity falls by ~3.086e-6 m/s² per metre of altitude, so a
    kilometre of elevation is worth ~0.0031 m/s² — small, but the term exists so
    that a later phase working in terrain does not have to add it.
    """
    model = WGS84Gravity(default_latitude_deg=45.0)
    at_sea_level = abs(float(model.acceleration_nav()[2]))
    at_altitude = abs(float(model.acceleration_nav(altitude_m=1000.0)[2]))
    assert at_sea_level - at_altitude == pytest.approx(3.086e-3, rel=1e-6)

    assert "45.000" in model.description
    assert "equatorial fallback" in WGS84Gravity().description
    # A non-finite latitude falls back rather than producing a NaN gravity.
    assert abs(float(model.acceleration_nav(latitude_deg=float("nan"))[2])) == pytest.approx(
        9.7803253359, rel=1e-9
    )


def test_constant_gravity_describes_itself():
    assert "9.80665" in ConstantGravity().description
    assert "navigation −Z" in ConstantGravity().description
