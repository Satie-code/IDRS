"""Quaternion tests.

Almost every quaternion defect is a convention defect, so these tests pin the
conventions themselves — storage order, product order, active rotation, and the
frame-composition rule — rather than only checking that the arithmetic closes.
A library that is self-consistent under the wrong convention passes every
round-trip test and still rotates the data backwards.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from idr.frames import quaternion as quat
from idr.frames.conventions import (
    QUATERNION_NORM_TOLERANCE,
    QUATERNION_SCALAR_INDEX,
    ROUND_TRIP_TOLERANCE,
)

X_AXIS = np.array([1.0, 0.0, 0.0])
Y_AXIS = np.array([0.0, 1.0, 0.0])
Z_AXIS = np.array([0.0, 0.0, 1.0])


def _random_quaternions(count: int, seed: int = 7) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [quat.normalize(rng.normal(size=4)) for _ in range(count)]


# --- conventions -----------------------------------------------------------


def test_scalar_is_stored_last() -> None:
    """[x, y, z, w]. A library assuming [w, x, y, z] would read a 90° rotation
    about X as a 0° rotation about nothing."""
    assert QUATERNION_SCALAR_INDEX == 3
    assert quat.identity()[QUATERNION_SCALAR_INDEX] == 1.0
    assert np.allclose(quat.identity()[:3], 0.0)

    ninety_about_x = quat.from_axis_angle(X_AXIS, math.pi / 2)
    assert ninety_about_x[0] == pytest.approx(math.sin(math.pi / 4))
    assert ninety_about_x[3] == pytest.approx(math.cos(math.pi / 4))


def test_rotation_is_active_and_right_handed() -> None:
    """+90° about Z carries +X to +Y, not to −Y."""
    q = quat.from_axis_angle(Z_AXIS, math.pi / 2)
    assert np.allclose(quat.rotate_vector(q, X_AXIS), Y_AXIS, atol=1e-12)
    assert np.allclose(quat.rotate_vector(q, Y_AXIS), -X_AXIS, atol=1e-12)
    assert np.allclose(quat.rotate_vector(q, Z_AXIS), Z_AXIS, atol=1e-12)


def test_multiplication_matches_matrix_multiplication_in_the_same_order() -> None:
    """multiply(a, b) ↔ R(a) @ R(b). This is what makes q_A_C = q_A_B ⊗ q_B_C."""
    for a, b in zip(_random_quaternions(12, 1), _random_quaternions(12, 2), strict=True):
        product = quat.to_rotation_matrix(quat.multiply(a, b))
        matrices = quat.to_rotation_matrix(a) @ quat.to_rotation_matrix(b)
        assert np.allclose(product, matrices, atol=1e-12)


def test_composition_order_is_not_commutative() -> None:
    """A guard against a test suite that would pass under either order."""
    a = quat.from_axis_angle(X_AXIS, math.pi / 2)
    b = quat.from_axis_angle(Y_AXIS, math.pi / 2)
    assert not quat.allclose(quat.multiply(a, b), quat.multiply(b, a))


# --- identity, inverse, composition ----------------------------------------


def test_identity_leaves_every_vector_unchanged() -> None:
    rng = np.random.default_rng(3)
    for _ in range(20):
        v = rng.normal(size=3)
        assert np.allclose(quat.rotate_vector(quat.identity(), v), v, atol=1e-15)


def test_inverse_undoes_the_rotation() -> None:
    """q⁻¹(q v) = v."""
    rng = np.random.default_rng(4)
    for q in _random_quaternions(20, 5):
        v = rng.normal(size=3)
        round_tripped = quat.rotate_vector(quat.inverse(q), quat.rotate_vector(q, v))
        assert np.allclose(round_tripped, v, atol=1e-12)


def test_quaternion_times_its_inverse_is_identity() -> None:
    for q in _random_quaternions(20, 6):
        assert quat.allclose(quat.multiply(q, quat.inverse(q)), quat.identity())


def test_conjugate_equals_inverse_for_unit_quaternions() -> None:
    for q in _random_quaternions(10, 8):
        assert np.allclose(quat.conjugate(q), quat.inverse(q), atol=1e-12)


def test_inverse_handles_a_non_unit_quaternion() -> None:
    """conj(q)/‖q‖² — the conjugate alone would be wrong here."""
    q = np.array([1.0, 2.0, 3.0, 4.0])
    assert quat.allclose(quat.multiply(q, quat.inverse(q)), quat.identity())


# --- known rotations -------------------------------------------------------


@pytest.mark.parametrize(
    ("axis", "vector", "expected"),
    [
        (Z_AXIS, X_AXIS, Y_AXIS),
        (Z_AXIS, Y_AXIS, -X_AXIS),
        (X_AXIS, Y_AXIS, Z_AXIS),
        (X_AXIS, Z_AXIS, -Y_AXIS),
        (Y_AXIS, Z_AXIS, X_AXIS),
        (Y_AXIS, X_AXIS, -Z_AXIS),
    ],
)
def test_ninety_degree_rotations_give_known_results(
    axis: np.ndarray, vector: np.ndarray, expected: np.ndarray
) -> None:
    q = quat.from_axis_angle(axis, math.pi / 2)
    assert np.allclose(quat.rotate_vector(q, vector), expected, atol=1e-12)


def test_rotation_preserves_length_and_angles() -> None:
    rng = np.random.default_rng(11)
    for q in _random_quaternions(10, 12):
        a, b = rng.normal(size=3), rng.normal(size=3)
        ra, rb = quat.rotate_vector(q, a), quat.rotate_vector(q, b)
        assert np.linalg.norm(ra) == pytest.approx(np.linalg.norm(a))
        assert float(ra @ rb) == pytest.approx(float(a @ b), abs=1e-12)


def test_four_ninety_degree_turns_return_to_start() -> None:
    q = quat.from_axis_angle(Z_AXIS, math.pi / 2)
    total = quat.identity()
    for _ in range(4):
        total = quat.multiply(q, total)
    assert quat.allclose(total, quat.identity(), tolerance=1e-12)


# --- round trips -----------------------------------------------------------


def test_quaternion_matrix_quaternion_round_trip() -> None:
    for q in _random_quaternions(50, 13):
        recovered = quat.from_rotation_matrix(quat.to_rotation_matrix(q))
        assert quat.angular_distance(q, recovered) <= ROUND_TRIP_TOLERANCE


def test_round_trip_survives_the_180_degree_branch() -> None:
    """The naive √(1+trace) formula loses most of its precision here."""
    for axis in (X_AXIS, Y_AXIS, Z_AXIS, np.array([1.0, 1.0, 1.0])):
        q = quat.from_axis_angle(axis, math.pi)
        recovered = quat.from_rotation_matrix(quat.to_rotation_matrix(q))
        assert quat.angular_distance(q, recovered) <= ROUND_TRIP_TOLERANCE


def test_euler_quaternion_matrix_quaternion_round_trip() -> None:
    rng = np.random.default_rng(14)
    for _ in range(50):
        roll = rng.uniform(-math.pi, math.pi)
        # Stay clear of ±90° pitch, where the decomposition is degenerate.
        pitch = rng.uniform(-1.4, 1.4)
        yaw = rng.uniform(-math.pi, math.pi)
        q = quat.from_euler(roll, pitch, yaw)
        through_matrix = quat.from_rotation_matrix(quat.to_rotation_matrix(q))
        back_roll, back_pitch, back_yaw = quat.to_euler(through_matrix)
        assert quat.allclose(q, quat.from_euler(back_roll, back_pitch, back_yaw), 1e-8)


def test_axis_angle_round_trip() -> None:
    rng = np.random.default_rng(15)
    for _ in range(30):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(0.01, math.pi - 0.01)
        recovered_axis, recovered_angle = quat.to_axis_angle(quat.from_axis_angle(axis, angle))
        assert recovered_angle == pytest.approx(angle, abs=1e-9)
        assert np.allclose(recovered_axis, axis, atol=1e-9)


def test_zero_rotation_reports_a_zero_angle_without_inventing_an_axis() -> None:
    axis, angle = quat.to_axis_angle(quat.identity())
    assert angle == 0.0
    assert np.allclose(axis, X_AXIS)


# --- sign equivalence ------------------------------------------------------


def test_q_and_negative_q_are_the_same_rotation() -> None:
    """Required by the brief: a sign difference is never a failure."""
    rng = np.random.default_rng(16)
    for q in _random_quaternions(20, 17):
        v = rng.normal(size=3)
        assert np.allclose(quat.rotate_vector(q, v), quat.rotate_vector(-q, v), atol=1e-12)
        assert quat.angular_distance(q, -q) == pytest.approx(0.0, abs=1e-9)
        assert quat.allclose(q, -q)
        assert np.allclose(quat.to_rotation_matrix(q), quat.to_rotation_matrix(-q), atol=1e-12)


# --- normalization and numerical stability ---------------------------------


def test_normalize_produces_a_unit_quaternion() -> None:
    rng = np.random.default_rng(18)
    for scale in (1e-6, 1.0, 1e6):
        q = quat.normalize(rng.normal(size=4) * scale)
        assert quat.is_normalized(q)
        assert abs(quat.norm(q) - 1.0) <= QUATERNION_NORM_TOLERANCE


def test_repeated_multiplication_does_not_drift_off_the_unit_sphere() -> None:
    """A thousand compositions is a short drive at 10 Hz."""
    q = quat.from_axis_angle(np.array([1.0, 2.0, 3.0]), 0.01)
    total = quat.identity()
    for _ in range(1000):
        total = quat.multiply(total, q)
    assert abs(quat.norm(total) - 1.0) <= 1e-9
    expected = quat.from_axis_angle(np.array([1.0, 2.0, 3.0]), 0.01 * 1000)
    assert quat.angular_distance(total, expected) < 1e-9


def test_normalize_refuses_a_degenerate_quaternion() -> None:
    for bad in ([0.0, 0.0, 0.0, 0.0], [np.nan, 0.0, 0.0, 1.0]):
        with pytest.raises(ValueError, match="normalize"):
            quat.normalize(bad)


def test_malformed_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="4 components"):
        quat.normalize([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="3 components"):
        quat.from_axis_angle([1.0, 0.0], 1.0)
    with pytest.raises(ValueError, match="no direction"):
        quat.from_axis_angle([0.0, 0.0, 0.0], 1.0)
    with pytest.raises(ValueError, match="finite"):
        quat.from_axis_angle(Z_AXIS, math.nan)
    with pytest.raises(ValueError, match="3-vector"):
        quat.rotate_vector(quat.identity(), [1.0, 2.0])


# --- euler diagnostics -----------------------------------------------------


def test_euler_matches_the_documented_zyx_composition() -> None:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll), and nothing else."""
    roll, pitch, yaw = 0.3, -0.2, 1.1
    composed = quat.multiply(
        quat.from_axis_angle(Z_AXIS, yaw),
        quat.multiply(quat.from_axis_angle(Y_AXIS, pitch), quat.from_axis_angle(X_AXIS, roll)),
    )
    assert quat.allclose(quat.from_euler(roll, pitch, yaw), composed, 1e-12)


def test_euler_round_trip_away_from_gimbal_lock() -> None:
    rng = np.random.default_rng(19)
    for _ in range(50):
        roll, pitch, yaw = (
            rng.uniform(-math.pi, math.pi),
            rng.uniform(-1.4, 1.4),
            rng.uniform(-math.pi, math.pi),
        )
        back = quat.to_euler(quat.from_euler(roll, pitch, yaw))
        assert back[0] == pytest.approx(roll, abs=1e-9)
        assert back[1] == pytest.approx(pitch, abs=1e-9)
        assert back[2] == pytest.approx(yaw, abs=1e-9)


def test_gimbal_lock_is_reported_rather_than_silently_split() -> None:
    """At |pitch| = 90° only yaw ± roll is defined; the split must be flagged."""
    q = quat.from_euler(0.4, math.pi / 2, 1.2)
    assert not quat.euler_is_well_conditioned(q)
    _, pitch, _ = quat.to_euler(q)
    assert abs(pitch) == pytest.approx(math.pi / 2, abs=1e-6)
    # Even at the degeneracy the returned angles still rebuild the rotation.
    assert quat.allclose(quat.from_euler(*quat.to_euler(q)), q, 1e-7)


def test_ordinary_rotations_are_well_conditioned() -> None:
    assert quat.euler_is_well_conditioned(quat.from_euler(0.3, 0.2, 1.0))
    assert quat.euler_is_well_conditioned(quat.identity())


# --- distance, interpolation, alignment ------------------------------------


def test_angular_distance_matches_a_known_angle() -> None:
    for degrees in (0.0, 15.0, 30.0, 45.0, 90.0, 179.0):
        q = quat.from_axis_angle(np.array([1.0, 1.0, 0.0]), math.radians(degrees))
        assert math.degrees(quat.angular_distance(quat.identity(), q)) == pytest.approx(
            degrees, abs=1e-9
        )


def test_slerp_endpoints_and_midpoint() -> None:
    start = quat.identity()
    end = quat.from_axis_angle(Z_AXIS, math.pi / 2)
    assert quat.allclose(quat.slerp(start, end, 0.0), start)
    assert quat.allclose(quat.slerp(start, end, 1.0), end)
    middle = quat.slerp(start, end, 0.5)
    assert math.degrees(quat.angular_distance(start, middle)) == pytest.approx(45.0, abs=1e-9)


def test_slerp_takes_the_short_way_round() -> None:
    """With a negated endpoint the result must be identical, not the long arc."""
    start = quat.identity()
    end = quat.from_axis_angle(Z_AXIS, math.pi / 2)
    assert quat.allclose(quat.slerp(start, end, 0.5), quat.slerp(start, -end, 0.5))


def test_slerp_rejects_a_fraction_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        quat.slerp(quat.identity(), quat.identity(), 1.5)


def test_from_two_vectors_aligns_the_directions() -> None:
    rng = np.random.default_rng(21)
    for _ in range(30):
        a, b = rng.normal(size=3), rng.normal(size=3)
        q = quat.from_two_vectors(a, b)
        rotated = quat.rotate_vector(q, a / np.linalg.norm(a))
        assert np.allclose(rotated, b / np.linalg.norm(b), atol=1e-12)


def test_from_two_vectors_handles_the_antiparallel_case() -> None:
    """No unique answer exists; any 180° rotation is valid and must be one."""
    q = quat.from_two_vectors(Z_AXIS, -Z_AXIS)
    assert np.allclose(quat.rotate_vector(q, Z_AXIS), -Z_AXIS, atol=1e-12)


def test_from_two_vectors_on_identical_directions_is_identity() -> None:
    assert quat.allclose(quat.from_two_vectors(Z_AXIS, Z_AXIS * 3.0), quat.identity())


def test_from_two_vectors_refuses_a_zero_vector() -> None:
    with pytest.raises(ValueError, match="no direction"):
        quat.from_two_vectors(np.zeros(3), Z_AXIS)


def test_rotate_vectors_matches_rotate_vector_row_by_row() -> None:
    rng = np.random.default_rng(22)
    q = quat.normalize(rng.normal(size=4))
    block = rng.normal(size=(25, 3))
    rotated = quat.rotate_vectors(q, block)
    for index in range(block.shape[0]):
        assert np.allclose(rotated[index], quat.rotate_vector(q, block[index]), atol=1e-12)


def test_rotate_vectors_rejects_a_bad_shape() -> None:
    with pytest.raises(ValueError, match=r"\(n, 3\)"):
        quat.rotate_vectors(quat.identity(), np.zeros((4, 2)))
