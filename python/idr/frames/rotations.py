"""Rotation matrices and the named frame transforms built from them.

The quaternion module owns the representation; this module owns SO(3)
validation and the *named* transforms that later phases consume. The naming
rule is enforced by construction: :class:`FrameRotation` carries the source and
destination frames, so applying a phone→vehicle rotation to navigation-frame
data raises instead of returning a plausible wrong answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from idr.frames import quaternion as quat
from idr.frames.conventions import (
    DETERMINANT_TOLERANCE,
    MIN_VECTOR_NORM,
    ORTHONORMALITY_TOLERANCE,
    Frame,
)
from idr.frames.vectors import Vector3, VectorSeries


def identity() -> np.ndarray:
    return np.eye(3, dtype="float64")


def transpose(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype="float64").T.copy()


def determinant(matrix: np.ndarray) -> float:
    return float(np.linalg.det(np.asarray(matrix, dtype="float64")))


def orthonormality_error(matrix: np.ndarray) -> float:
    """Largest absolute element of ``RᵀR − I``."""
    array = np.asarray(matrix, dtype="float64")
    if array.shape != (3, 3):
        raise ValueError(f"expected a 3x3 matrix, got {array.shape}")
    return float(np.max(np.abs(array.T @ array - np.eye(3))))


def is_rotation_matrix(
    matrix: np.ndarray,
    *,
    orthonormality_tolerance: float = ORTHONORMALITY_TOLERANCE,
    determinant_tolerance: float = DETERMINANT_TOLERANCE,
) -> bool:
    """Whether ``matrix`` is a proper rotation: orthonormal with det = +1.

    The determinant check is not redundant. An orthonormal matrix with
    det = −1 is a *reflection*: it satisfies ``RᵀR = I``, passes any
    orthonormality test, and quietly mirrors the data — turning a left turn
    into a right one.
    """
    array = np.asarray(matrix, dtype="float64")
    if array.shape != (3, 3) or not np.isfinite(array).all():
        return False
    if orthonormality_error(array) > orthonormality_tolerance:
        return False
    return abs(determinant(array) - 1.0) <= determinant_tolerance


def require_rotation_matrix(matrix: np.ndarray, name: str = "matrix") -> np.ndarray:
    """Return the matrix, or raise explaining which property failed."""
    array = np.asarray(matrix, dtype="float64")
    if array.shape != (3, 3):
        raise ValueError(f"{name} must be 3x3, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite entries")
    error = orthonormality_error(array)
    if error > ORTHONORMALITY_TOLERANCE:
        raise ValueError(
            f"{name} is not orthonormal: max|RᵀR − I| = {error:.3e} "
            f"exceeds {ORTHONORMALITY_TOLERANCE:.1e}"
        )
    det = determinant(array)
    if abs(det - 1.0) > DETERMINANT_TOLERANCE:
        kind = "a reflection" if det < 0.0 else "a scaling"
        raise ValueError(f"{name} has det = {det:.12f}, so it is {kind}, not a rotation")
    return array


def inverse(matrix: np.ndarray) -> np.ndarray:
    """Inverse of a rotation, computed as the transpose.

    Validated first: transposing a non-rotation returns something that is not
    its inverse, which is a far more confusing failure than an exception.
    """
    return require_rotation_matrix(matrix, "matrix").T.copy()


def apply(matrix: np.ndarray, vector: np.ndarray | list[float]) -> np.ndarray:
    array = np.asarray(vector, dtype="float64").reshape(-1)
    if array.size != 3:
        raise ValueError(f"can only rotate a 3-vector, got {array.size} components")
    return np.asarray(matrix, dtype="float64") @ array


def apply_many(matrix: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Apply one rotation to an ``(n, 3)`` block."""
    array = np.asarray(vectors, dtype="float64")
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"expected an (n, 3) array, got {array.shape}")
    return array @ np.asarray(matrix, dtype="float64").T


def orthonormalize(matrix: np.ndarray) -> np.ndarray:
    """Snap a nearly-orthonormal matrix onto SO(3).

    Uses the SVD (the closest proper rotation in the Frobenius sense) rather
    than Gram-Schmidt, which is order-dependent and privileges the first
    column. If the input is a reflection the sign of the last singular
    direction is flipped, so the result is always a proper rotation.
    """
    array = np.asarray(matrix, dtype="float64")
    if array.shape != (3, 3):
        raise ValueError(f"expected a 3x3 matrix, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("cannot orthonormalize a matrix with non-finite entries")
    u, _, vt = np.linalg.svd(array)
    result = u @ vt
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def from_basis(forward: np.ndarray, left: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Assemble a rotation whose **columns** are the given basis vectors.

    With columns ``(f, l, u)`` expressed in frame A, the result maps a vector's
    B-coordinates to A-coordinates, where B is the frame those three vectors
    define. That is ``R_A_B`` in this package's naming.
    """
    columns = [np.asarray(v, dtype="float64").reshape(-1) for v in (forward, left, up)]
    for index, column in enumerate(columns):
        if column.size != 3:
            raise ValueError(f"basis vector {index} needs 3 components, got {column.size}")
    matrix = np.column_stack(columns)
    return require_rotation_matrix(orthonormalize(matrix), "basis")


def basis_from_up_and_forward(up: np.ndarray, forward_hint: np.ndarray) -> np.ndarray:
    """Build an orthonormal (forward, left, up) basis from two rough directions.

    ``up`` is taken as exact — it comes from gravity, which is the better
    constrained of the two — and ``forward_hint`` supplies only the azimuth:
    its component along ``up`` is removed rather than averaged in. Returning a
    basis that tilted the vertical to accommodate a noisy heading would trade a
    well-measured axis for a poorly-measured one.

    Raises when the hint is parallel to ``up``, because then it carries no
    azimuth information at all.
    """
    up_vector = np.asarray(up, dtype="float64").reshape(-1)
    hint = np.asarray(forward_hint, dtype="float64").reshape(-1)
    if up_vector.size != 3 or hint.size != 3:
        raise ValueError("basis_from_up_and_forward needs two 3-vectors")
    up_norm = float(np.linalg.norm(up_vector))
    if up_norm < MIN_VECTOR_NORM:
        raise ValueError("up direction has no direction")
    unit_up = up_vector / up_norm

    projected = hint - float(hint @ unit_up) * unit_up
    projected_norm = float(np.linalg.norm(projected))
    if projected_norm < MIN_VECTOR_NORM:
        raise ValueError(
            "forward hint is parallel to up, so it constrains no heading; "
            "yaw cannot be resolved from it"
        )
    unit_forward = projected / projected_norm
    unit_left = np.cross(unit_up, unit_forward)
    return from_basis(unit_forward, unit_left, unit_up)


@dataclass(frozen=True)
class FrameRotation:
    """A rotation that knows which two frames it connects.

    ``FrameRotation(R, Frame.VEHICLE, Frame.PHONE)`` is ``R_vehicle_phone``: it
    takes phone-frame coordinates to vehicle-frame coordinates. Applying it to
    a :class:`~idr.frames.vectors.Vector3` tagged with any other frame raises.
    """

    matrix: np.ndarray
    to_frame: Frame
    from_frame: Frame

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "matrix",
            require_rotation_matrix(self.matrix, f"R_{self.to_frame}_{self.from_frame}"),
        )

    @classmethod
    def from_quaternion(cls, q: np.ndarray, to_frame: Frame, from_frame: Frame) -> FrameRotation:
        return cls(quat.to_rotation_matrix(q), to_frame, from_frame)

    @classmethod
    def identity_between(cls, to_frame: Frame, from_frame: Frame) -> FrameRotation:
        return cls(identity(), to_frame, from_frame)

    @property
    def quaternion(self) -> np.ndarray:
        return quat.from_rotation_matrix(self.matrix)

    @property
    def name(self) -> str:
        return f"R_{self.to_frame}_{self.from_frame}"

    def inverse(self) -> FrameRotation:
        """The reverse transform, with the frame labels swapped to match."""
        return FrameRotation(self.matrix.T.copy(), self.from_frame, self.to_frame)

    def compose(self, inner: FrameRotation) -> FrameRotation:
        """``self ∘ inner``. Requires ``inner.to_frame == self.from_frame``.

        This is where the naming earns its keep: composing ``R_nav_vehicle``
        with ``R_vehicle_phone`` yields ``R_nav_phone``, and composing them the
        other way round is rejected rather than producing nonsense.
        """
        if inner.to_frame != self.from_frame:
            raise ValueError(
                f"cannot compose {self.name} with {inner.name}: "
                f"{self.name} consumes {self.from_frame}, but {inner.name} produces "
                f"{inner.to_frame}"
            )
        return FrameRotation(self.matrix @ inner.matrix, self.to_frame, inner.from_frame)

    def apply_vector(self, vector: Vector3) -> Vector3:
        if vector.frame != self.from_frame:
            raise ValueError(f"{self.name} expects a {self.from_frame} vector, got {vector.frame}")
        return Vector3.from_array(apply(self.matrix, vector.to_array()), self.to_frame)

    def apply_series(self, series: VectorSeries) -> VectorSeries:
        if series.frame != self.from_frame:
            raise ValueError(f"{self.name} expects a {self.from_frame} series, got {series.frame}")
        return VectorSeries(
            samples=apply_many(self.matrix, series.samples),
            analysis_time_s=series.analysis_time_s,
            frame=self.to_frame,
            source_time_s=series.source_time_s,
            quality_flags=series.quality_flags,
            source=series.source,
            notes=[*series.notes, f"rotated by {self.name}"],
        )

    def angle_to(self, other: FrameRotation) -> float:
        """Rotation angle in radians between two transforms of the same pair."""
        if (self.to_frame, self.from_frame) != (other.to_frame, other.from_frame):
            raise ValueError(f"cannot compare {self.name} with {other.name}")
        return quat.angular_distance(self.quaternion, other.quaternion)

    def euler_deg(self) -> tuple[float, float, float]:
        """(roll, pitch, yaw) in degrees — for reporting only."""
        roll, pitch, yaw = quat.to_euler(self.quaternion)
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)
