"""Quaternion mathematics, with the conventions stated rather than implied.

Most quaternion bugs are convention bugs, not arithmetic bugs, so the four
choices that matter are fixed here and repeated in
``docs/mathematics/phase3_orientation.md``:

1. **Storage order is ``[x, y, z, w]``** — vector part first, scalar last.
   ``QUATERNION_SCALAR_INDEX`` names the scalar slot so no caller has to
   remember. (SciPy uses this order; Eigen and many textbooks use ``[w,x,y,z]``.
   Mixing the two silently swaps a rotation for a different one.)

2. **Hamilton product**, not JPL. ``ij = k``.

3. **Rotations are active.** ``rotate_vector(q, v)`` returns the vector ``v``
   rotated by ``q`` inside one fixed frame, by the right-hand rule about the
   axis. ``from_axis_angle(axis, θ)`` therefore rotates *vectors* by ``+θ``.

4. **Frame naming follows from (3).** If frame B's basis is frame A's basis
   actively rotated by ``q``, then a vector's coordinates convert as
   ``v_A = R(q) @ v_B``. So the quaternion carrying that rotation is named
   ``q_A_B`` and composition chains left to right::

       q_A_C = multiply(q_A_B, q_B_C)

   matching ``R_A_C = R_A_B @ R_B_C`` exactly. This is checked by a test rather
   than asserted here.

``q`` and ``-q`` denote the same rotation. Every comparison in this module is
sign-agnostic, and no function may report them as different.
"""

from __future__ import annotations

import math

import numpy as np

from idr.frames.conventions import (
    EULER_GIMBAL_LIMIT_RAD,
    MIN_VECTOR_NORM,
    QUATERNION_NORM_TOLERANCE,
)

#: The multiplicative identity: zero vector part, unit scalar.
IDENTITY = np.array([0.0, 0.0, 0.0, 1.0], dtype="float64")


def _as_quaternion(q: np.ndarray | list[float]) -> np.ndarray:
    array = np.asarray(q, dtype="float64").reshape(-1)
    if array.size != 4:
        raise ValueError(f"a quaternion needs exactly 4 components, got {array.size}")
    return array


def identity() -> np.ndarray:
    """A fresh identity quaternion (never the shared constant, so it is safe to
    mutate at the call site)."""
    return IDENTITY.copy()


def norm(q: np.ndarray | list[float]) -> float:
    return float(np.linalg.norm(_as_quaternion(q)))


def is_normalized(q: np.ndarray | list[float], tolerance: float | None = None) -> bool:
    limit = QUATERNION_NORM_TOLERANCE if tolerance is None else tolerance
    return abs(norm(q) - 1.0) <= limit


def normalize(q: np.ndarray | list[float]) -> np.ndarray:
    """Scale to unit length.

    Raises on a zero or non-finite quaternion instead of returning something
    that merely looks like a rotation.
    """
    array = _as_quaternion(q)
    magnitude = float(np.linalg.norm(array))
    if not math.isfinite(magnitude) or magnitude < MIN_VECTOR_NORM:
        raise ValueError(f"cannot normalize a quaternion of norm {magnitude!r}")
    return array / magnitude


def conjugate(q: np.ndarray | list[float]) -> np.ndarray:
    """Negate the vector part. For a unit quaternion this is the inverse."""
    array = _as_quaternion(q)
    return np.array([-array[0], -array[1], -array[2], array[3]], dtype="float64")


def inverse(q: np.ndarray | list[float]) -> np.ndarray:
    """True inverse, valid for non-unit quaternions too: ``conj(q)/‖q‖²``."""
    array = _as_quaternion(q)
    squared = float(array @ array)
    if not math.isfinite(squared) or squared < MIN_VECTOR_NORM:
        raise ValueError(f"cannot invert a quaternion of squared norm {squared!r}")
    return conjugate(array) / squared


def multiply(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> np.ndarray:
    """Hamilton product ``a ⊗ b``, in ``[x, y, z, w]`` storage.

    Order matters and is the same as matrix multiplication: ``multiply(a, b)``
    corresponds to ``to_rotation_matrix(a) @ to_rotation_matrix(b)``, so
    ``q_A_C = multiply(q_A_B, q_B_C)``.
    """
    ax, ay, az, aw = _as_quaternion(a)
    bx, by, bz, bw = _as_quaternion(b)
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype="float64",
    )


def rotate_vector(q: np.ndarray | list[float], v: np.ndarray | list[float]) -> np.ndarray:
    """Actively rotate ``v`` by ``q``.

    Uses the cross-product form rather than building a matrix: it costs fewer
    operations for a single vector and avoids the intermediate rounding of a
    matrix construction.
    """
    unit = normalize(q)
    vector = np.asarray(v, dtype="float64").reshape(-1)
    if vector.size != 3:
        raise ValueError(f"can only rotate a 3-vector, got {vector.size} components")
    axis = unit[:3]
    scalar = float(unit[3])
    t = 2.0 * np.cross(axis, vector)
    return vector + scalar * t + np.cross(axis, t)


def rotate_vectors(q: np.ndarray | list[float], vectors: np.ndarray) -> np.ndarray:
    """Rotate an ``(n, 3)`` block by one quaternion.

    Goes through the rotation matrix here because a single 3×3 matmul over n
    rows beats n independent cross-product pairs once n is more than a handful.
    """
    array = np.asarray(vectors, dtype="float64")
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"expected an (n, 3) array, got {array.shape}")
    return array @ to_rotation_matrix(q).T


def from_axis_angle(axis: np.ndarray | list[float], angle_rad: float) -> np.ndarray:
    """Unit quaternion for an active rotation of ``angle_rad`` about ``axis``."""
    vector = np.asarray(axis, dtype="float64").reshape(-1)
    if vector.size != 3:
        raise ValueError(f"an axis needs 3 components, got {vector.size}")
    magnitude = float(np.linalg.norm(vector))
    if not math.isfinite(magnitude) or magnitude < MIN_VECTOR_NORM:
        raise ValueError("rotation axis has no direction")
    if not math.isfinite(angle_rad):
        raise ValueError(f"rotation angle must be finite, got {angle_rad!r}")
    unit_axis = vector / magnitude
    half = 0.5 * angle_rad
    return np.array([*(unit_axis * math.sin(half)), math.cos(half)], dtype="float64")


def to_axis_angle(q: np.ndarray | list[float]) -> tuple[np.ndarray, float]:
    """Inverse of :func:`from_axis_angle`, with the angle in ``[0, π]``.

    For a rotation of (near) zero the axis is undefined; ``+X`` is returned with
    a zero angle, which composes correctly and is the only choice that does not
    invent a direction.
    """
    unit = normalize(q)
    # Fix the sign so the scalar part is non-negative: q and -q are the same
    # rotation, and this picks the representative with angle <= π.
    if unit[3] < 0.0:
        unit = -unit
    vector_norm = float(np.linalg.norm(unit[:3]))
    angle = 2.0 * math.atan2(vector_norm, float(unit[3]))
    if vector_norm < MIN_VECTOR_NORM:
        return np.array([1.0, 0.0, 0.0], dtype="float64"), 0.0
    return unit[:3] / vector_norm, angle


def to_rotation_matrix(q: np.ndarray | list[float]) -> np.ndarray:
    """The 3×3 active rotation matrix for ``q``."""
    x, y, z, w = normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype="float64",
    )


def from_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    """Recover a unit quaternion from a rotation matrix.

    Uses Shepperd's branch selection: pick whichever of the four components is
    largest and derive the rest from it. The naive ``w = √(1+trace)/2`` formula
    divides by a value that vanishes near a 180° rotation and loses most of its
    precision there; branching keeps the divisor bounded away from zero.
    """
    array = np.asarray(matrix, dtype="float64")
    if array.shape != (3, 3):
        raise ValueError(f"expected a 3x3 matrix, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("rotation matrix contains non-finite entries")

    m00, m01, m02 = array[0]
    m10, m11, m12 = array[1]
    m20, m21, m22 = array[2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (m21 - m12) / scale
        y = (m02 - m20) / scale
        z = (m10 - m01) / scale
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / scale
        x = 0.25 * scale
        y = (m01 + m10) / scale
        z = (m02 + m20) / scale
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / scale
        x = (m01 + m10) / scale
        y = 0.25 * scale
        z = (m12 + m21) / scale
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / scale
        x = (m02 + m20) / scale
        y = (m12 + m21) / scale
        z = 0.25 * scale
    return normalize(np.array([x, y, z, w], dtype="float64"))


def from_euler(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """Build a quaternion from intrinsic Z-Y-X Euler angles.

    The composition is ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``: roll is applied
    first, about the body X axis, then pitch about the new Y, then yaw about
    the new Z. This is the aerospace ordering, and it is the only Euler
    sequence this package uses.
    """
    for name, value in (("roll", roll_rad), ("pitch", pitch_rad), ("yaw", yaw_rad)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
    half_roll, half_pitch, half_yaw = 0.5 * roll_rad, 0.5 * pitch_rad, 0.5 * yaw_rad
    sr, cr = math.sin(half_roll), math.cos(half_roll)
    sp, cp = math.sin(half_pitch), math.cos(half_pitch)
    sy, cy = math.sin(half_yaw), math.cos(half_yaw)
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype="float64",
    )


def to_euler(q: np.ndarray | list[float]) -> tuple[float, float, float]:
    """Decompose into intrinsic Z-Y-X Euler angles ``(roll, pitch, yaw)`` in rad.

    **Diagnostics only.** Euler angles are not the internal representation:
    they are discontinuous, sequence-dependent, and degenerate at |pitch| = 90°.
    Near that degeneracy roll and yaw are not separable — only their sum is —
    and this function resolves it by assigning the whole rotation to yaw and
    setting roll to zero, which is a *choice*, not a measurement. Use
    :func:`euler_is_well_conditioned` before quoting the split.
    """
    unit = normalize(q)
    x, y, z, w = unit

    # sin(pitch), clamped: rounding can push it a few ulp outside [-1, 1] and
    # asin would then return NaN for a perfectly valid rotation.
    sin_pitch = 2.0 * (w * y - z * x)
    sin_pitch = min(1.0, max(-1.0, sin_pitch))
    pitch = math.asin(sin_pitch)

    if abs(abs(sin_pitch) - 1.0) < EULER_GIMBAL_LIMIT_RAD:
        # At pitch = +90° the rotation depends only on (yaw − roll); at −90°
        # only on (yaw + roll). Substituting sp = ±cp = ±√2/2 into from_euler
        # gives 2·atan2(x, w) = roll − yaw for +90° and roll + yaw for −90°,
        # so the invariant is recovered by negating in the +90° case only.
        # Getting this sign backwards returns roll − yaw as the yaw, which
        # rebuilds a *different* rotation — caught by the round-trip assertion.
        roll = 0.0
        yaw = 2.0 * math.atan2(x, w) * (-1.0 if sin_pitch > 0.0 else 1.0)
        return roll, pitch, math.atan2(math.sin(yaw), math.cos(yaw))

    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def euler_is_well_conditioned(q: np.ndarray | list[float]) -> bool:
    """Whether the Z-Y-X roll/yaw split of ``q`` is meaningful.

    False near |pitch| = 90°, where the decomposition is degenerate.
    """
    unit = normalize(q)
    x, y, z, w = unit
    sin_pitch = min(1.0, max(-1.0, 2.0 * (w * y - z * x)))
    return abs(abs(sin_pitch) - 1.0) >= EULER_GIMBAL_LIMIT_RAD


def angular_distance(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    """Smallest rotation angle in radians carrying ``a`` onto ``b``.

    Sign-agnostic: ``q`` and ``-q`` are the same rotation, so the result is
    always in ``[0, π]``.

    Computed as ``2·atan2(‖vec(a⁻¹b)‖, |w(a⁻¹b)|)`` rather than
    ``2·acos(|a·b|)``. The two are equal in exact arithmetic, but ``acos`` has
    an infinite derivative at 1, so for two rotations differing by one ulp it
    returns about 1e-8 instead of about 1e-16 — a floor that made a correct
    round trip look like a 3e-8 error. ``atan2`` is well conditioned there.
    """
    unit_a, unit_b = normalize(a), normalize(b)
    relative = multiply(conjugate(unit_a), unit_b)
    vector_norm = float(np.linalg.norm(relative[:3]))
    return 2.0 * math.atan2(vector_norm, abs(float(relative[3])))


def allclose(
    a: np.ndarray | list[float], b: np.ndarray | list[float], tolerance: float = 1e-9
) -> bool:
    """Whether two quaternions represent the same rotation, ignoring sign."""
    return angular_distance(a, b) <= tolerance


def slerp(a: np.ndarray | list[float], b: np.ndarray | list[float], fraction: float) -> np.ndarray:
    """Spherical linear interpolation, taking the short way round.

    Falls back to normalized linear interpolation when the two rotations are
    nearly identical, where ``sin(θ)`` in the denominator would be numerically
    unstable while the two results are indistinguishable anyway.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must lie in [0, 1], got {fraction!r}")
    unit_a, unit_b = normalize(a), normalize(b)
    dot = float(unit_a @ unit_b)
    if dot < 0.0:  # pick the shorter arc between the two equivalent signs
        unit_b = -unit_b
        dot = -dot
    if dot > 1.0 - 1e-9:
        return normalize(unit_a + fraction * (unit_b - unit_a))
    theta = math.acos(min(1.0, dot))
    sin_theta = math.sin(theta)
    weight_a = math.sin((1.0 - fraction) * theta) / sin_theta
    weight_b = math.sin(fraction * theta) / sin_theta
    return normalize(weight_a * unit_a + weight_b * unit_b)


def from_two_vectors(
    source: np.ndarray | list[float], target: np.ndarray | list[float]
) -> np.ndarray:
    """Shortest rotation carrying the direction ``source`` onto ``target``.

    Used to express "align this measured gravity direction with vehicle-down".
    The 180° case has no unique answer — any axis perpendicular to ``source``
    works — so one perpendicular is chosen deterministically and the ambiguity
    is documented rather than hidden.
    """
    a = np.asarray(source, dtype="float64").reshape(-1)
    b = np.asarray(target, dtype="float64").reshape(-1)
    if a.size != 3 or b.size != 3:
        raise ValueError("from_two_vectors needs two 3-vectors")
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if min(norm_a, norm_b) < MIN_VECTOR_NORM:
        raise ValueError("cannot align vectors when one has no direction")
    a, b = a / norm_a, b / norm_b

    dot = float(np.clip(a @ b, -1.0, 1.0))
    if dot > 1.0 - 1e-12:
        return identity()
    if dot < -1.0 + 1e-12:
        # Antiparallel: pick the axis least aligned with `a` so the cross
        # product is well conditioned.
        fallback = np.zeros(3)
        fallback[int(np.argmin(np.abs(a)))] = 1.0
        axis = np.cross(a, fallback)
        return from_axis_angle(axis, math.pi)
    axis = np.cross(a, b)
    return from_axis_angle(axis, math.acos(dot))
