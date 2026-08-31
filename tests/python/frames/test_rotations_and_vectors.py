"""Rotation-matrix and frame-tagged vector tests.

The rotation-matrix tests check SO(3) membership properly — orthonormality
*and* determinant — because an orthonormal matrix with det = −1 is a
reflection: it passes every ``RᵀR ≈ I`` check while mirroring the data, turning
a left turn into a right one.

The vector tests check that the frame tag is load-bearing rather than
decorative: operations across mismatched frames must raise, not compute.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from idr.frames import quaternion as quat
from idr.frames import rotations
from idr.frames.conventions import (
    DETERMINANT_TOLERANCE,
    ORTHONORMALITY_TOLERANCE,
    Frame,
)
from idr.frames.rotations import FrameRotation
from idr.frames.vectors import TimestampedVector3, Vector3, VectorSeries

Z_AXIS = np.array([0.0, 0.0, 1.0])


def _random_matrices(count: int, seed: int = 31) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [quat.to_rotation_matrix(quat.normalize(rng.normal(size=4))) for _ in range(count)]


# --- SO(3) validation ------------------------------------------------------


def test_valid_rotations_satisfy_orthonormality_and_determinant() -> None:
    for matrix in _random_matrices(30):
        assert rotations.orthonormality_error(matrix) <= ORTHONORMALITY_TOLERANCE
        assert abs(rotations.determinant(matrix) - 1.0) <= DETERMINANT_TOLERANCE
        assert rotations.is_rotation_matrix(matrix)


def test_a_reflection_is_rejected_despite_being_orthonormal() -> None:
    """det = −1. RᵀR = I holds, so orthonormality alone would let it through."""
    reflection = np.diag([1.0, 1.0, -1.0])
    assert rotations.orthonormality_error(reflection) == pytest.approx(0.0, abs=1e-15)
    assert rotations.determinant(reflection) == pytest.approx(-1.0)
    assert not rotations.is_rotation_matrix(reflection)
    with pytest.raises(ValueError, match="reflection"):
        rotations.require_rotation_matrix(reflection)


def test_a_scaled_rotation_is_rejected() -> None:
    scaled = 2.0 * np.eye(3)
    assert not rotations.is_rotation_matrix(scaled)
    with pytest.raises(ValueError, match="not orthonormal"):
        rotations.require_rotation_matrix(scaled)


def test_a_malformed_or_non_finite_matrix_is_rejected() -> None:
    assert not rotations.is_rotation_matrix(np.zeros((2, 2)))
    assert not rotations.is_rotation_matrix(np.full((3, 3), np.nan))
    with pytest.raises(ValueError, match="3x3"):
        rotations.require_rotation_matrix(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="non-finite"):
        rotations.require_rotation_matrix(np.full((3, 3), np.nan))


def test_identity_and_transpose_behave() -> None:
    assert rotations.is_rotation_matrix(rotations.identity())
    for matrix in _random_matrices(10, 32):
        assert np.allclose(rotations.transpose(matrix), matrix.T)
        assert np.allclose(rotations.inverse(matrix) @ matrix, np.eye(3), atol=1e-12)


def test_inverse_refuses_a_non_rotation_rather_than_transposing_it() -> None:
    """Transposing a non-rotation silently returns something that is not its
    inverse — a far more confusing failure than an exception."""
    with pytest.raises(ValueError):
        rotations.inverse(np.diag([1.0, 2.0, 3.0]))


# --- vector application ----------------------------------------------------


def test_apply_matches_the_quaternion_rotation() -> None:
    rng = np.random.default_rng(33)
    for _ in range(20):
        q = quat.normalize(rng.normal(size=4))
        v = rng.normal(size=3)
        matrix = quat.to_rotation_matrix(q)
        assert np.allclose(rotations.apply(matrix, v), quat.rotate_vector(q, v), atol=1e-12)


def test_apply_many_matches_apply_row_by_row() -> None:
    rng = np.random.default_rng(34)
    matrix = _random_matrices(1, 35)[0]
    block = rng.normal(size=(20, 3))
    rotated = rotations.apply_many(matrix, block)
    for index in range(block.shape[0]):
        assert np.allclose(rotated[index], rotations.apply(matrix, block[index]), atol=1e-12)


def test_apply_rejects_wrong_shapes() -> None:
    with pytest.raises(ValueError, match="3-vector"):
        rotations.apply(rotations.identity(), [1.0, 2.0])
    with pytest.raises(ValueError, match=r"\(n, 3\)"):
        rotations.apply_many(rotations.identity(), np.zeros((3, 2)))


# --- orthonormalization ----------------------------------------------------


def test_orthonormalize_repairs_a_slightly_corrupted_rotation() -> None:
    rng = np.random.default_rng(36)
    matrix = _random_matrices(1, 37)[0] + rng.normal(0.0, 1e-6, size=(3, 3))
    assert not rotations.is_rotation_matrix(matrix)
    repaired = rotations.orthonormalize(matrix)
    assert rotations.is_rotation_matrix(repaired)
    assert np.allclose(repaired, matrix, atol=1e-5)


def test_orthonormalize_returns_a_proper_rotation_from_a_reflection() -> None:
    repaired = rotations.orthonormalize(np.diag([1.0, 1.0, -1.0]))
    assert rotations.determinant(repaired) == pytest.approx(1.0)


def test_orthonormalize_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        rotations.orthonormalize(np.full((3, 3), np.nan))


# --- basis construction ----------------------------------------------------


def test_basis_from_up_and_forward_keeps_up_exact() -> None:
    """Up comes from gravity and is the better-measured axis, so a noisy
    heading hint must not be allowed to tilt it."""
    up = np.array([0.0, 0.0, 1.0])
    hint = np.array([1.0, 0.0, 0.7])  # deliberately not horizontal
    matrix = rotations.basis_from_up_and_forward(up, hint)
    assert rotations.is_rotation_matrix(matrix)
    assert np.allclose(matrix[:, 2], up, atol=1e-12)
    # Forward keeps the hint's azimuth but loses its vertical component.
    assert np.allclose(matrix[:, 0], [1.0, 0.0, 0.0], atol=1e-12)
    # Right-handed: with X forward and Z up, Y is left.
    assert np.allclose(matrix[:, 1], [0.0, 1.0, 0.0], atol=1e-12)


def test_basis_rejects_a_hint_parallel_to_up() -> None:
    """A vertical hint carries no azimuth, so yaw is genuinely unconstrained."""
    with pytest.raises(ValueError, match="constrains no heading"):
        rotations.basis_from_up_and_forward(Z_AXIS, Z_AXIS)


def test_basis_rejects_a_zero_up() -> None:
    with pytest.raises(ValueError, match="no direction"):
        rotations.basis_from_up_and_forward(np.zeros(3), np.array([1.0, 0.0, 0.0]))


def test_from_basis_columns_are_the_supplied_axes() -> None:
    matrix = rotations.from_basis(
        np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )
    assert np.allclose(matrix, np.eye(3))


# --- FrameRotation ---------------------------------------------------------


def test_frame_rotation_names_itself_by_its_frames() -> None:
    rotation = FrameRotation.identity_between(Frame.VEHICLE, Frame.PHONE)
    assert rotation.name == "R_vehicle_phone"


def test_frame_rotation_transforms_only_matching_frames() -> None:
    rotation = FrameRotation.identity_between(Frame.VEHICLE, Frame.PHONE)
    phone_vector = Vector3(1.0, 2.0, 3.0, Frame.PHONE)
    assert rotation.apply_vector(phone_vector).frame is Frame.VEHICLE
    with pytest.raises(ValueError, match="expects a phone vector"):
        rotation.apply_vector(Vector3(1.0, 2.0, 3.0, Frame.NAVIGATION))


def test_inverting_a_frame_rotation_swaps_the_labels() -> None:
    q = quat.from_axis_angle(Z_AXIS, 0.7)
    rotation = FrameRotation.from_quaternion(q, Frame.VEHICLE, Frame.PHONE)
    inverted = rotation.inverse()
    assert inverted.to_frame is Frame.PHONE
    assert inverted.from_frame is Frame.VEHICLE
    assert np.allclose(inverted.matrix @ rotation.matrix, np.eye(3), atol=1e-12)


def test_composition_chains_frames_and_matches_matrix_multiplication() -> None:
    """R_nav_phone = R_nav_vehicle @ R_vehicle_phone."""
    nav_vehicle = FrameRotation.from_quaternion(
        quat.from_axis_angle(Z_AXIS, 0.4), Frame.NAVIGATION, Frame.VEHICLE
    )
    vehicle_phone = FrameRotation.from_quaternion(
        quat.from_axis_angle(np.array([1.0, 1.0, 0.0]), 0.9), Frame.VEHICLE, Frame.PHONE
    )
    composed = nav_vehicle.compose(vehicle_phone)
    assert composed.name == "R_navigation_phone"
    assert np.allclose(composed.matrix, nav_vehicle.matrix @ vehicle_phone.matrix, atol=1e-12)


def test_composing_in_the_wrong_order_is_rejected_not_computed() -> None:
    """The whole point of naming rotations by their frames."""
    nav_vehicle = FrameRotation.identity_between(Frame.NAVIGATION, Frame.VEHICLE)
    vehicle_phone = FrameRotation.identity_between(Frame.VEHICLE, Frame.PHONE)
    with pytest.raises(ValueError, match="cannot compose"):
        vehicle_phone.compose(nav_vehicle)


def test_frame_rotation_rejects_a_non_rotation_matrix() -> None:
    with pytest.raises(ValueError, match="R_vehicle_phone"):
        FrameRotation(np.diag([1.0, 1.0, -1.0]), Frame.VEHICLE, Frame.PHONE)


def test_frame_rotation_round_trips_through_its_quaternion() -> None:
    q = quat.from_euler(0.2, -0.3, 1.1)
    rotation = FrameRotation.from_quaternion(q, Frame.VEHICLE, Frame.PHONE)
    assert quat.allclose(rotation.quaternion, q, 1e-12)


def test_angle_to_compares_only_matching_frame_pairs() -> None:
    a = FrameRotation.identity_between(Frame.VEHICLE, Frame.PHONE)
    b = FrameRotation.from_quaternion(
        quat.from_axis_angle(Z_AXIS, math.radians(30)), Frame.VEHICLE, Frame.PHONE
    )
    assert math.degrees(a.angle_to(b)) == pytest.approx(30.0, abs=1e-9)
    other = FrameRotation.identity_between(Frame.NAVIGATION, Frame.VEHICLE)
    with pytest.raises(ValueError, match="cannot compare"):
        a.angle_to(other)


def test_series_transformation_retags_the_frame_and_records_the_step() -> None:
    series = VectorSeries(
        samples=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        analysis_time_s=np.array([0.0, 0.1]),
        frame=Frame.PHONE,
    )
    rotation = FrameRotation.from_quaternion(
        quat.from_axis_angle(Z_AXIS, math.pi / 2), Frame.VEHICLE, Frame.PHONE
    )
    result = rotation.apply_series(series)
    assert result.frame is Frame.VEHICLE
    assert np.allclose(result.samples[0], [0.0, 1.0, 0.0], atol=1e-12)
    assert any("R_vehicle_phone" in note for note in result.notes)
    with pytest.raises(ValueError, match="expects a phone series"):
        rotation.apply_series(result)


# --- Vector3 / VectorSeries ------------------------------------------------


def test_vector_operations_require_a_matching_frame() -> None:
    phone = Vector3(1.0, 0.0, 0.0, Frame.PHONE)
    vehicle = Vector3(1.0, 0.0, 0.0, Frame.VEHICLE)
    for operation in ("dot", "cross", "angle_to"):
        with pytest.raises(ValueError, match=operation):
            getattr(phone, operation)(vehicle)


def test_vector_normalization_refuses_a_direction_that_does_not_exist() -> None:
    with pytest.raises(ValueError, match="no reliable direction"):
        Vector3(0.0, 0.0, 0.0, Frame.PHONE).normalized()
    with pytest.raises(ValueError, match="no reliable direction"):
        Vector3(math.nan, 0.0, 0.0, Frame.PHONE).normalized()


def test_vector_angle_is_precise_for_nearly_parallel_vectors() -> None:
    """acos would lose most of its precision here; atan2 does not."""
    a = Vector3(1.0, 0.0, 0.0, Frame.PHONE)
    b = Vector3(math.cos(1e-8), math.sin(1e-8), 0.0, Frame.PHONE)
    assert a.angle_to(b) == pytest.approx(1e-8, rel=1e-6)


def test_vector_basics() -> None:
    v = Vector3(3.0, 4.0, 0.0, Frame.PHONE)
    assert v.norm == pytest.approx(5.0)
    assert v.is_finite
    assert v.normalized().norm == pytest.approx(1.0)
    assert v.scaled(2.0).norm == pytest.approx(10.0)
    assert np.allclose(v.to_array(), [3.0, 4.0, 0.0])
    assert Vector3.from_array([3.0, 4.0, 0.0], Frame.PHONE) == v
    assert "phone" in repr(v)
    assert v.with_frame(Frame.VEHICLE).frame is Frame.VEHICLE


def test_vector_from_array_rejects_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="exactly 3 components"):
        Vector3.from_array([1.0, 2.0])


def test_cross_product_is_right_handed() -> None:
    x = Vector3(1.0, 0.0, 0.0, Frame.VEHICLE)
    y = Vector3(0.0, 1.0, 0.0, Frame.VEHICLE)
    assert np.allclose(x.cross(y).to_array(), [0.0, 0.0, 1.0])


def test_series_validates_its_shape_and_length() -> None:
    with pytest.raises(ValueError, match=r"\(n, 3\)"):
        VectorSeries(samples=np.zeros((4, 2)), analysis_time_s=np.zeros(4))
    with pytest.raises(ValueError, match="timestamps for"):
        VectorSeries(samples=np.zeros((4, 3)), analysis_time_s=np.zeros(3))


def test_series_subset_preserves_metadata() -> None:
    series = VectorSeries(
        samples=np.arange(12, dtype="float64").reshape(4, 3),
        analysis_time_s=np.array([0.0, 0.1, 0.2, 0.3]),
        frame=Frame.PHONE,
        source_time_s=np.array([10.0, 10.1, 10.2, 10.3]),
        quality_flags=np.array([0, 1, 2, 3]),
        source="test",
    )
    subset = series.subset(np.array([True, False, True, False]))
    assert len(subset) == 2
    assert subset.frame is Frame.PHONE
    assert subset.source == "test"
    assert subset.source_time_s is not None
    assert np.allclose(subset.source_time_s, [10.0, 10.2])
    assert subset.quality_flags is not None
    assert np.allclose(subset.quality_flags, [0, 2])


def test_series_at_returns_a_timestamped_vector_carrying_provenance() -> None:
    series = VectorSeries(
        samples=np.array([[1.0, 2.0, 3.0]]),
        analysis_time_s=np.array([5.0]),
        frame=Frame.PHONE,
        source_time_s=np.array([99.0]),
        quality_flags=np.array([4]),
        source="segment:accel",
    )
    sample = series.at(0)
    assert isinstance(sample, TimestampedVector3)
    assert sample.frame is Frame.PHONE
    assert sample.analysis_time_s == 5.0
    assert sample.source_time_s == 99.0
    assert sample.quality_flags == 4
    assert np.allclose(sample.to_array(), [1.0, 2.0, 3.0])


def test_series_mean_ignores_non_finite_rows() -> None:
    series = VectorSeries(
        samples=np.array([[1.0, 1.0, 1.0], [np.nan, 0.0, 0.0], [3.0, 3.0, 3.0]]),
        analysis_time_s=np.array([0.0, 0.1, 0.2]),
        frame=Frame.PHONE,
    )
    assert np.allclose(series.mean_vector().to_array(), [2.0, 2.0, 2.0])


def test_series_mean_refuses_an_empty_selection() -> None:
    series = VectorSeries(
        samples=np.full((2, 3), np.nan),
        analysis_time_s=np.array([0.0, 0.1]),
        frame=Frame.PHONE,
    )
    with pytest.raises(ValueError, match="no finite samples"):
        series.mean_vector()
