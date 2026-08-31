"""Orientation estimation from gyroscope, gravity and (conditionally) magnetometer.

A complementary filter, not a Kalman filter. That is a scope decision as much
as a design one: Phase 3 is allowed to produce an orientation and a confidence,
but a state vector with a covariance is Phase 4/5 work, and building one here
would mean building it twice.

The physics the filter encodes:

* **The gyroscope is right in the short term and wrong in the long term.**
  Integrating it tracks fast rotation faithfully and accumulates bias
  indefinitely.
* **Gravity is right in the long term and wrong in the short term.** It pins
  roll and pitch absolutely, but only while the device is not accelerating.
* **The magnetometer is the only absolute yaw reference available**, and in a
  vehicle it is frequently wrong (see :mod:`idr.frames.magnetometer`).

So the filter integrates the gyro and pulls it gently back toward gravity, and
toward magnetic north only when the magnetometer has earned it. Which sensor
supports which axis is reported explicitly on the result, because the honest
answer for a phone in a car is often "roll and pitch are observed, yaw is
dead-reckoned from the gyro and is drifting".

Nothing here integrates into position or velocity. That is Phase 4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from idr.frames import quaternion as quat
from idr.frames.conventions import MIN_VECTOR_NORM, STANDARD_GRAVITY_MPS2
from idr.frames.gravity import (
    LOW_DYNAMIC_ACCEL_TOLERANCE_MPS2,
    GravityEstimate,
    GravityMethod,
    estimate_gravity,
)
from idr.frames.magnetometer import MagnetometerQuality, assess_magnetometer
from idr.frames.vectors import VectorSeries

#: Complementary-filter gain pulling the estimate toward measured gravity, in
#: units of 1/s. 0.5 gives a correction time constant of about two seconds:
#: fast enough to bound gyro drift over a long drive, slow enough that a few
#: seconds of hard braking cannot tip the estimate by more than a degree or so.
GRAVITY_GAIN_HZ = 0.5

#: Gain pulling yaw toward magnetic north, 1/s. An order of magnitude weaker
#: than the gravity gain, because the magnetometer's failure mode is a *steady*
#: bias, and a strong gain would track that bias faithfully into the estimate.
MAGNETIC_GAIN_HZ = 0.05

#: Longest Δt the filter will integrate across in one step. Beyond this the
#: small-angle propagation stops being a good approximation and, more
#: importantly, the gap means something happened that the gyro did not see.
MAX_INTEGRATION_STEP_S = 0.5

#: Angular-rate ceiling, rad/s. Above ~35 rad/s (2000°/s) a consumer MEMS gyro
#: is saturated and the reading is not a measurement.
MAX_PLAUSIBLE_RATE_RPS = 35.0


class AxisSupport(StrEnum):
    """How well-determined one orientation axis is."""

    #: Pinned by an absolute reference (gravity for roll/pitch, a trusted
    #: magnetometer for yaw).
    OBSERVED = "observed"
    #: Propagated from the gyroscope only; correct in relative terms, drifting
    #: in absolute terms.
    DEAD_RECKONED = "dead_reckoned"
    #: No information at all.
    UNRESOLVED = "unresolved"


@dataclass
class OrientationEstimate:
    """Orientation of the phone frame relative to the navigation frame.

    ``quaternion`` is ``q_navigation_phone``: it takes phone-frame coordinates
    to navigation-frame (ENU) coordinates.

    The Euler angles are diagnostics. ``euler_well_conditioned`` says whether
    the roll/yaw split is meaningful at this pitch; when it is False the
    individual roll and yaw numbers should not be quoted.
    """

    quaternion: np.ndarray
    analysis_time_s: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    euler_well_conditioned: bool
    confidence: float
    roll_support: AxisSupport
    pitch_support: AxisSupport
    yaw_support: AxisSupport
    gravity_residual_deg: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """``R_navigation_phone``."""
        return quat.to_rotation_matrix(self.quaternion)

    def axis_support(self) -> dict[str, AxisSupport]:
        return {
            "roll": self.roll_support,
            "pitch": self.pitch_support,
            "yaw": self.yaw_support,
        }


@dataclass
class OrientationTrack:
    """Orientation over a whole segment, plus how it was produced."""

    quaternions: np.ndarray
    analysis_time_s: np.ndarray
    confidence: np.ndarray
    gravity: GravityEstimate
    magnetometer: MagnetometerQuality
    roll_support: AxisSupport
    pitch_support: AxisSupport
    yaw_support: AxisSupport
    integrated_steps: int
    skipped_steps: int
    saturated_samples: int
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.quaternions.shape[0])

    def at(self, index: int) -> OrientationEstimate:
        q = self.quaternions[index]
        roll, pitch, yaw = quat.to_euler(q)
        return OrientationEstimate(
            quaternion=q.copy(),
            analysis_time_s=float(self.analysis_time_s[index]),
            roll_deg=math.degrees(roll),
            pitch_deg=math.degrees(pitch),
            yaw_deg=math.degrees(yaw),
            euler_well_conditioned=quat.euler_is_well_conditioned(q),
            confidence=float(self.confidence[index]),
            roll_support=self.roll_support,
            pitch_support=self.pitch_support,
            yaw_support=self.yaw_support,
        )

    def euler_deg(self) -> np.ndarray:
        """``(n, 3)`` array of (roll, pitch, yaw) in degrees, for plotting."""
        out = np.empty((len(self), 3), dtype="float64")
        for index in range(len(self)):
            roll, pitch, yaw = quat.to_euler(self.quaternions[index])
            out[index] = (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))
        return out

    def mean_confidence(self) -> float:
        finite = self.confidence[np.isfinite(self.confidence)]
        return float(finite.mean()) if finite.size else 0.0


def _initial_quaternion(gravity: GravityEstimate) -> tuple[np.ndarray, list[str]]:
    """Seed the filter so that measured down maps onto navigation −Z.

    This fixes roll and pitch immediately and leaves yaw at an arbitrary but
    fixed value, which is the honest starting point: gravity carries no
    heading information whatsoever.
    """
    if gravity.direction is None:
        return quat.identity(), [
            "no gravity estimate; the filter starts at identity and its absolute "
            "roll/pitch are meaningless until gravity becomes observable"
        ]
    nav_down = np.array([0.0, 0.0, -1.0], dtype="float64")
    q = quat.from_two_vectors(gravity.direction.to_array(), nav_down)
    return q, [
        "initial attitude set so measured gravity maps to navigation down; "
        "initial yaw is arbitrary because gravity gives no heading"
    ]


def _propagate(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """Advance attitude by a body-frame angular rate over ``dt``.

    Uses the exact axis-angle exponential of ω·dt rather than the first-order
    ``q + 0.5·Ω·q·dt`` update. The exact form costs one sin/cos and stays a unit
    quaternion by construction, whereas the first-order form has to be
    renormalized every step and still accumulates error proportional to the
    rotation rate — which is exactly when accuracy matters.
    """
    magnitude = float(np.linalg.norm(omega))
    if magnitude < MIN_VECTOR_NORM or dt <= 0.0:
        return q
    delta = quat.from_axis_angle(omega / magnitude, magnitude * dt)
    return quat.normalize(quat.multiply(q, delta))


def estimate_orientation(
    *,
    gyro: VectorSeries | None,
    accel: VectorSeries | None,
    mag: VectorSeries | None = None,
    gravity_channel: VectorSeries | None = None,
    magnetometer_available: bool = True,
    gravity_gain_hz: float = GRAVITY_GAIN_HZ,
    magnetic_gain_hz: float = MAGNETIC_GAIN_HZ,
) -> OrientationTrack:
    """Estimate ``q_navigation_phone`` across one segment.

    Δt comes from ``analysis_time_s`` sample by sample: no fixed rate is
    assumed anywhere, in line with the Phase 2 finding that the dataset spans
    roughly 2 Hz to 1000 Hz. A step whose Δt is non-positive (a duplicate
    timestamp, which Phase 2 retains and flags) or longer than
    ``MAX_INTEGRATION_STEP_S`` is *not* integrated across; it is counted and
    reported instead of being silently bridged.
    """
    if accel is None or len(accel) == 0:
        raise ValueError("orientation estimation needs an accelerometer series")

    gravity = estimate_gravity(accel, gravity_channel=gravity_channel, gyro=gyro)
    mag_quality = assess_magnetometer(mag, gravity=gravity, available=magnetometer_available)

    times = accel.analysis_time_s
    count = len(accel)
    quaternions = np.empty((count, 4), dtype="float64")
    confidence = np.zeros(count, dtype="float64")

    q, notes = _initial_quaternion(gravity)

    has_gyro = gyro is not None and len(gyro) == count
    if not has_gyro:
        notes.append(
            "no matching gyroscope series; attitude is taken from gravity alone and "
            "carries no rotation dynamics"
        )

    use_mag = mag is not None and len(mag) == count and mag_quality.is_trusted
    if mag is not None and len(mag) == count and not mag_quality.is_trusted:
        notes.append(f"magnetometer excluded: {mag_quality.describe()}")

    integrated = skipped = saturated = 0
    accel_finite = accel.finite_mask
    accel_norms = accel.norms

    for index in range(count):
        if index > 0:
            dt = float(times[index] - times[index - 1])
            if has_gyro and 0.0 < dt <= MAX_INTEGRATION_STEP_S:
                assert gyro is not None
                omega = gyro.samples[index - 1]
                if np.isfinite(omega).all():
                    rate = float(np.linalg.norm(omega))
                    if rate > MAX_PLAUSIBLE_RATE_RPS:
                        saturated += 1
                    else:
                        q = _propagate(q, omega, dt)
                        integrated += 1
                else:
                    skipped += 1
            elif has_gyro:
                skipped += 1

            # --- gravity correction, only when the reading is near 1 g ------
            #
            # Correcting toward an accelerometer that is currently measuring
            # braking would tilt the estimate toward the direction of braking.
            # The magnitude gate is the cheapest reliable way to tell the two
            # apart sample by sample.
            if accel_finite[index] and dt > 0.0:
                error = abs(accel_norms[index] - STANDARD_GRAVITY_MPS2)
                if error <= LOW_DYNAMIC_ACCEL_TOLERANCE_MPS2:
                    q = _apply_gravity_correction(
                        q, accel.samples[index], min(dt, MAX_INTEGRATION_STEP_S), gravity_gain_hz
                    )
                    if use_mag:
                        assert mag is not None
                        q = _apply_magnetic_correction(
                            q,
                            mag.samples[index],
                            min(dt, MAX_INTEGRATION_STEP_S),
                            magnetic_gain_hz * mag_quality.weight,
                        )

        quaternions[index] = q
        confidence[index] = _sample_confidence(gravity, mag_quality, has_gyro)

    roll_support = pitch_support = (
        AxisSupport.OBSERVED if gravity.is_available else AxisSupport.UNRESOLVED
    )
    if use_mag:
        yaw_support = AxisSupport.OBSERVED
    elif has_gyro:
        yaw_support = AxisSupport.DEAD_RECKONED
        notes.append(
            "yaw is propagated from the gyroscope with no absolute reference; it is "
            "correct in relative terms and drifts in absolute terms"
        )
    else:
        yaw_support = AxisSupport.UNRESOLVED
        notes.append("yaw is unresolved: no trusted magnetometer and no gyroscope")

    if skipped:
        notes.append(
            f"{skipped} step(s) were not integrated (non-positive Δt from a retained "
            f"duplicate timestamp, a gap over {MAX_INTEGRATION_STEP_S}s, or a non-finite rate)"
        )
    if saturated:
        notes.append(
            f"{saturated} sample(s) exceeded {MAX_PLAUSIBLE_RATE_RPS} rad/s and were "
            "treated as gyro saturation rather than real motion"
        )

    return OrientationTrack(
        quaternions=quaternions,
        analysis_time_s=times.copy(),
        confidence=confidence,
        gravity=gravity,
        magnetometer=mag_quality,
        roll_support=roll_support,
        pitch_support=pitch_support,
        yaw_support=yaw_support,
        integrated_steps=integrated,
        skipped_steps=skipped,
        saturated_samples=saturated,
        notes=notes,
    )


def _apply_gravity_correction(
    q: np.ndarray, accel_sample: np.ndarray, dt: float, gain_hz: float
) -> np.ndarray:
    """Rotate ``q`` a little so predicted down moves toward measured down."""
    magnitude = float(np.linalg.norm(accel_sample))
    if magnitude < MIN_VECTOR_NORM:
        return q
    measured_down_nav = quat.rotate_vector(q, accel_sample / magnitude) * -1.0
    expected_down_nav = np.array([0.0, 0.0, -1.0], dtype="float64")
    correction = quat.from_two_vectors(measured_down_nav, expected_down_nav)
    axis, angle = quat.to_axis_angle(correction)
    step = min(1.0, gain_hz * dt)
    return quat.normalize(quat.multiply(quat.from_axis_angle(axis, angle * step), q))


def _apply_magnetic_correction(
    q: np.ndarray, mag_sample: np.ndarray, dt: float, gain_hz: float
) -> np.ndarray:
    """Rotate ``q`` about navigation vertical toward magnetic north.

    Only the *horizontal* component of the field is used, and the correction is
    constrained to the vertical axis. Letting the magnetometer influence roll
    and pitch would let a disturbed field undo the gravity correction, which is
    the better-founded of the two.
    """
    if gain_hz <= 0.0:
        return q
    magnitude = float(np.linalg.norm(mag_sample))
    if magnitude < MIN_VECTOR_NORM:
        return q
    field_nav = quat.rotate_vector(q, mag_sample / magnitude)
    horizontal = np.array([field_nav[0], field_nav[1], 0.0], dtype="float64")
    if float(np.linalg.norm(horizontal)) < 1e-3:
        # Field is nearly vertical in the navigation frame: it carries no
        # heading. Happens at high magnetic latitude and under disturbance.
        return q
    # ENU: north is +Y. The heading error is the angle from measured horizontal
    # field to north, taken about the up axis.
    error = math.atan2(float(horizontal[0]), float(horizontal[1]))
    step = min(1.0, gain_hz * dt)
    correction = quat.from_axis_angle(np.array([0.0, 0.0, 1.0]), -error * step)
    return quat.normalize(quat.multiply(correction, q))


def _sample_confidence(
    gravity: GravityEstimate, mag_quality: MagnetometerQuality, has_gyro: bool
) -> float:
    """Confidence in the full 3-DOF attitude.

    Capped hard when yaw has no absolute reference: an estimate whose roll and
    pitch are excellent and whose yaw is a free-running integral is not a
    high-confidence orientation, and reporting one number above 0.5 for it
    would be the kind of unexplained magic score the brief rules out.
    """
    base = gravity.confidence if gravity.method is not GravityMethod.UNAVAILABLE else 0.0
    if not has_gyro:
        base *= 0.5
    if mag_quality.is_trusted:
        return float(min(1.0, base * (0.6 + 0.4 * mag_quality.weight)))
    return float(min(0.5, base))
