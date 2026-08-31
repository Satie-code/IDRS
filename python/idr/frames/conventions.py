"""Frame, rotation and numerical conventions for Phase 3.

Everything in :mod:`idr.frames` obeys the conventions declared here, and this
module is the single place they are defined. The prose version, with the
reasoning, is ``docs/architecture/sensor_frame_conventions.md``.

The one rule worth stating twice: a rotation is *always* named for the two
frames it connects, source last. ``R_vehicle_phone`` maps phone-frame
coordinates to vehicle-frame coordinates::

    v_vehicle = R_vehicle_phone @ v_phone

There is no ``R1``/``R2`` anywhere in this package, because the direction of a
rotation is exactly the thing that gets silently inverted.
"""

from __future__ import annotations

from enum import StrEnum

#: Standard gravity, matching the constant Phase 2 used for its g→m/s²
#: conversion (``idr.pipeline.canonical.G_TO_MPS2``). Keeping one value across
#: phases means a vehicle acceleration expressed in g and a gravity magnitude
#: check cannot disagree by a rounding choice.
STANDARD_GRAVITY_MPS2 = 9.80665

#: Plausible magnitude of Earth's magnetic field at the surface, in µT. Used
#: only to *flag* a suspicious magnetometer reading, never to correct one.
EARTH_FIELD_MIN_UT = 25.0
EARTH_FIELD_MAX_UT = 65.0


class Frame(StrEnum):
    """The coordinate frames this project distinguishes.

    A value of this enum is carried alongside sensor data so that a triple of
    floats can never be silently reinterpreted in the wrong frame.
    """

    #: Local-tangent-plane navigation frame: X east, Y north, Z up.
    NAVIGATION = "navigation"
    #: Vehicle body frame: X forward, Y left, Z up.
    VEHICLE = "vehicle"
    #: Phone/device sensor frame as delivered by the logger.
    PHONE = "phone"
    #: Frame is genuinely unknown. Not a default — an explicit admission.
    UNKNOWN = "unknown"


#: Navigation frame axis order, documented as data so a reader never has to
#: guess whether index 0 is east or north.
NAVIGATION_AXES = ("east", "north", "up")

#: Vehicle frame axis order. Right-handed: with X forward and Z up, Y is left.
#: "Lateral" is ambiguous about sign, so the sign is fixed here and nowhere
#: else: a left turn produces positive lateral (Y) acceleration.
VEHICLE_AXES = ("forward", "left", "up")

#: Phone frame axis order. These are *labels for positions*, not a claim about
#: which way the device is physically pointing — see the module docstring of
#: :mod:`idr.frames.alignment` and section 3 of the conventions document.
PHONE_AXES = ("x", "y", "z")


class RotationConvention(StrEnum):
    """Which of the several mutually incompatible conventions we use."""

    #: Hamilton quaternion product (not JPL), scalar stored last.
    HAMILTON = "hamilton"
    #: Euler sequence: intrinsic Z-Y-X, i.e. R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    EULER_ZYX_INTRINSIC = "euler_zyx_intrinsic"


#: Index of the scalar component in the quaternion storage order [x, y, z, w].
QUATERNION_SCALAR_INDEX = 3

#: Human-readable storage order, for reports and error messages.
QUATERNION_ORDER = "xyzw"


# ---------------------------------------------------------------------------
# Numerical tolerances
# ---------------------------------------------------------------------------
#
# These are not decoration. Each is chosen for the operation it guards, in
# float64, and section 29 of the phase brief explicitly forbids scattering a
# single 1e-6 everywhere. The reasoning is recorded next to each value so a
# later phase can tighten or loosen one deliberately.

#: Deviation of ‖q‖ from 1 that still counts as normalized. A norm is four
#: squares, a sum and a square root: about 6 rounding steps, so ~1e-15 of
#: relative error. 1e-9 leaves eleven thousand-fold headroom while still
#: catching a quaternion that has drifted through repeated multiplication.
QUATERNION_NORM_TOLERANCE = 1e-9

#: Largest allowed element of |RᵀR − I|. RᵀR is nine dot products of
#: three terms each, so error accumulates to a few ulp (~1e-15).
ORTHONORMALITY_TOLERANCE = 1e-9

#: Allowed deviation of det(R) from +1. A 3×3 determinant is six products and
#: five additions; the same order of error applies.
DETERMINANT_TOLERANCE = 1e-9

#: Tolerance for a quaternion → matrix → quaternion round trip, as an angle in
#: radians. Shepperd's branch selection keeps the divisor bounded away from
#: zero even at a 180° rotation, and the measured worst case over 200 random
#: rotations is 8.7e-16 rad. 1e-12 therefore leaves three orders of magnitude
#: of headroom while still failing if the branch logic regresses.
#:
#: This was 1e-8 while ``angular_distance`` used ``acos``, whose infinite
#: derivative at 1 put a ~1.5e-8 floor under every comparison of near-identical
#: rotations. That floor was a property of the *measurement*, not of the round
#: trip, and it is gone now that the distance uses ``atan2``.
ROUND_TRIP_TOLERANCE = 1e-12

#: How close |pitch| may come to 90° before yaw and roll stop being separable.
#: At exactly ±90° the Z-Y-X decomposition is degenerate (gimbal lock): only
#: yaw ± roll is defined. Below this threshold the split is reported as
#: unreliable rather than silently returned.
EULER_GIMBAL_LIMIT_RAD = 1e-7

#: Below this magnitude a vector has no meaningful direction and normalizing it
#: would amplify noise into a confident-looking unit vector. Chosen well above
#: float64 denormals so that a genuinely tiny-but-real vector is still usable.
MIN_VECTOR_NORM = 1e-12

#: Angle below which two directions are treated as the same for reporting.
#: One millidegree is far finer than any sensor here resolves.
ANGLE_EQUALITY_TOLERANCE_RAD = 1.7453292519943296e-05
