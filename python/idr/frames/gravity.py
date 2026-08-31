"""Gravity-direction estimation from accelerometer, gravity channel and gyro.

The point of this module is one distinction that is easy to lose:

    an accelerometer measures **specific force**, not gravity.

At rest the two coincide, which is why the naive "gravity is wherever the
accelerometer points" works well enough to be dangerous. Under braking, on a
bumpy road, or with the phone being handled, the accelerometer reading is
gravity *plus* whatever the vehicle and the road are doing, and treating it as
vertical tilts the entire downstream frame.

So every estimate here carries the method that produced it, the evidence that
supported it, and a confidence — and when the evidence is not there, the
estimate says so instead of returning the mean of the accelerometer.

The methods are ordinary physics. Nothing here is learned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from idr.frames.conventions import MIN_VECTOR_NORM, STANDARD_GRAVITY_MPS2, Frame
from idr.frames.vectors import Vector3, VectorSeries

#: How far ‖a‖ may sit from 1 g and still count as low-dynamic, in m/s².
#: 0.5 m/s² is about 0.05 g — comfortably above sensor noise and normal road
#: vibration, comfortably below the ~1–3 m/s² of ordinary acceleration and
#: braking that this filter exists to exclude.
LOW_DYNAMIC_ACCEL_TOLERANCE_MPS2 = 0.5

#: Angular-rate ceiling for a low-dynamic sample, rad/s. 0.1 rad/s ≈ 5.7°/s;
#: a vehicle taking a normal corner turns several times faster, and a phone
#: being picked up faster still.
LOW_DYNAMIC_GYRO_TOLERANCE_RPS = 0.1

#: Minimum number of qualifying samples before an estimate is offered at all.
#: Below this the mean direction is dominated by whatever the few samples
#: happened to be doing.
MIN_GRAVITY_SAMPLES = 20

#: Angular spread (radians) of the contributing samples at which confidence
#: from consistency reaches zero. 0.35 rad ≈ 20°: a spread that wide means the
#: device moved during the window and no single direction describes it.
MAX_DIRECTION_SPREAD_RAD = 0.35


class GravityMethod(StrEnum):
    """Where a gravity estimate came from. Recorded, never inferred later."""

    #: The logger's own gravity channel (Android's fused, gyro-aided estimate).
    GRAVITY_CHANNEL = "gravity_channel"
    #: Accelerometer averaged over detected low-dynamic samples.
    ACCEL_LOW_DYNAMIC = "accel_low_dynamic"
    #: No defensible estimate. Direction is None.
    UNAVAILABLE = "unavailable"


#: Both source channels measure specific force, so both point *away* from
#: gravity when the device is at rest, and both must be negated to yield the
#: gravity vector. This was verified against the dataset rather than assumed:
#: over 66 real segments with a quasi-static interval, the median cosine
#: between the accelerometer and the gravity channel is +0.999 and the gravity
#: channel's magnitude is exactly 9.807 m/s². The two agree in sign, and the
#: sign they agree on is Android's — the gravity sensor is documented to read
#: the same as the accelerometer at rest. Getting this backwards inverts the
#: vertical axis and flips every recovered yaw by 180°, which is exactly the
#: failure a synthetic round-trip test caught.
SPECIFIC_FORCE_OPPOSES_GRAVITY = True


@dataclass
class GravityEstimate:
    """A gravity direction with the evidence behind it.

    ``direction`` is a unit vector in the phone frame pointing the way the
    gravity *vector* points — i.e. downward, toward the centre of the Earth.
    Vehicle "up" is its negation.

    Note that this is the opposite of what either source channel reads: see
    :data:`SPECIFIC_FORCE_OPPOSES_GRAVITY`.
    """

    direction: Vector3 | None
    magnitude_mps2: float | None
    method: GravityMethod
    confidence: float
    sample_count: int
    candidate_count: int
    direction_spread_rad: float | None
    magnitude_error_mps2: float | None
    notes: list[str] = field(default_factory=list)

    @property
    def is_available(self) -> bool:
        return self.direction is not None

    @property
    def up_direction(self) -> Vector3 | None:
        """Vehicle/world "up" expressed in the phone frame."""
        return None if self.direction is None else self.direction.scaled(-1.0)

    def tilt_from_phone_z_deg(self) -> float | None:
        """Angle between measured down and the phone's −Z axis.

        Zero means the phone is lying flat, screen up, in the Android
        convention. Reported as a *description* of the device pose, never as
        proof that the convention holds for this dataset.
        """
        if self.direction is None:
            return None
        phone_down = Vector3(0.0, 0.0, -1.0, Frame.PHONE)
        return math.degrees(self.direction.angle_to(phone_down))


def low_dynamic_mask(
    accel: VectorSeries,
    gyro: VectorSeries | None = None,
    *,
    accel_tolerance_mps2: float = LOW_DYNAMIC_ACCEL_TOLERANCE_MPS2,
    gyro_tolerance_rps: float = LOW_DYNAMIC_GYRO_TOLERANCE_RPS,
) -> np.ndarray:
    """Boolean mask of samples where the device is close to quasi-static.

    Two independent conditions, both required when the data supports them:

    - the specific-force magnitude is within ``accel_tolerance_mps2`` of 1 g,
      so the sensor is measuring approximately gravity alone;
    - the angular rate is small, so the frame is not rotating underneath the
      average being taken.

    The magnitude test alone is not enough, for two distinct reasons:

    - a vehicle in a steady turn can hold ‖a‖ near 1 g while the direction
      sweeps round — this is what the gyro condition rejects, and when no gyro
      is available the caller is told the estimate rests on the weaker test;
    - the test is **insensitive to acceleration perpendicular to gravity**.
      Adding ``a`` at right angles to 1 g changes the magnitude by only
      ``√(g² + a²) − g``, so with the default 0.5 m/s² tolerance a horizontal
      acceleration up to about 3.1 m/s² passes unnoticed. Such a sample tilts
      the estimated vertical by up to ``atan(a/g)`` ≈ 18°. In practice the
      surviving error is far smaller, because horizontal accelerations average
      toward zero over a window while gravity does not — but a window that is
      *consistently* accelerating one way will bias the vertical, and that is
      what the reported direction spread and confidence exist to expose.
    """
    if len(accel) == 0:
        return np.zeros(0, dtype=bool)

    mask = accel.finite_mask & (np.abs(accel.norms - STANDARD_GRAVITY_MPS2) <= accel_tolerance_mps2)
    if gyro is not None and len(gyro) == len(accel):
        mask &= gyro.finite_mask & (gyro.norms <= gyro_tolerance_rps)
    return mask


def _direction_spread(samples: np.ndarray, mean_direction: np.ndarray) -> float:
    """RMS angle between each sample direction and the mean direction."""
    norms = np.linalg.norm(samples, axis=1)
    usable = norms > MIN_VECTOR_NORM
    if not usable.any():
        return float("nan")
    unit = samples[usable] / norms[usable, None]
    cosines = np.clip(unit @ mean_direction, -1.0, 1.0)
    angles = np.arccos(cosines)
    return float(np.sqrt(np.mean(angles**2)))


def _confidence(
    spread_rad: float, magnitude_error: float, sample_count: int, minimum_samples: int
) -> tuple[float, list[str]]:
    """Combine the three independent pieces of evidence into one score.

    Kept as a product of named factors rather than a tuned formula, so a low
    score can always be traced to the factor that caused it. The individual
    metrics stay on the estimate for exactly that reason.
    """
    notes: list[str] = []

    if math.isfinite(spread_rad):
        consistency = max(0.0, 1.0 - spread_rad / MAX_DIRECTION_SPREAD_RAD)
    else:
        consistency = 0.0
        notes.append("direction spread could not be measured")
    if consistency < 0.5:
        notes.append(
            f"contributing samples disagree by {math.degrees(spread_rad):.1f}° RMS; "
            "the device was probably moving"
        )

    # Relative magnitude error, saturating at 20% away from 1 g.
    magnitude_score = max(0.0, 1.0 - abs(magnitude_error) / (0.2 * STANDARD_GRAVITY_MPS2))
    if magnitude_score < 0.5:
        notes.append(
            f"mean magnitude is {magnitude_error:+.2f} m/s² from standard gravity; "
            "the sensor may be biased or the window is not quasi-static"
        )

    # Support saturates at 10x the minimum: more samples help, but only so far.
    support = min(1.0, sample_count / (10.0 * minimum_samples))
    if sample_count < 2 * minimum_samples:
        notes.append(f"only {sample_count} qualifying sample(s)")

    return float(consistency * magnitude_score * (0.5 + 0.5 * support)), notes


def estimate_gravity(
    accel: VectorSeries,
    *,
    gravity_channel: VectorSeries | None = None,
    gyro: VectorSeries | None = None,
    minimum_samples: int = MIN_GRAVITY_SAMPLES,
) -> GravityEstimate:
    """Estimate the gravity direction in the phone frame.

    Prefers the logger's gravity channel when it is present and finite, because
    that channel is already the product of an on-device gyro/accelerometer
    fusion and does not require the vehicle to be quasi-static. Falls back to
    averaging the accelerometer over detected low-dynamic samples.

    Returns an estimate with ``method = UNAVAILABLE`` rather than raising when
    neither path has enough evidence: an absent gravity estimate is a normal
    outcome for a segment that never stops moving, and callers need to handle
    it as data, not as an exception.
    """
    notes: list[str] = []

    if gravity_channel is not None and len(gravity_channel) > 0:
        finite = gravity_channel.finite_mask
        # The channel is only useful where it actually has a magnitude; some
        # files carry the column but leave it at zero.
        finite &= gravity_channel.norms > MIN_VECTOR_NORM
        if int(finite.sum()) >= minimum_samples:
            return _estimate_from(
                gravity_channel.subset(finite),
                method=GravityMethod.GRAVITY_CHANNEL,
                candidate_count=len(gravity_channel),
                minimum_samples=minimum_samples,
                notes=["used the logger's gravity channel; no quasi-static window required"],
            )
        notes.append(
            f"gravity channel present but only {int(finite.sum())} usable sample(s); "
            "fell back to the accelerometer"
        )

    if len(accel) == 0:
        return GravityEstimate(
            direction=None,
            magnitude_mps2=None,
            method=GravityMethod.UNAVAILABLE,
            confidence=0.0,
            sample_count=0,
            candidate_count=0,
            direction_spread_rad=None,
            magnitude_error_mps2=None,
            notes=[*notes, "no accelerometer samples"],
        )

    if gyro is None or len(gyro) != len(accel):
        notes.append(
            "no matching gyroscope series; low-dynamic detection used the "
            "magnitude test alone, which a steady turn can pass"
        )
    mask = low_dynamic_mask(accel, gyro)
    qualifying = int(mask.sum())
    if qualifying < minimum_samples:
        return GravityEstimate(
            direction=None,
            magnitude_mps2=None,
            method=GravityMethod.UNAVAILABLE,
            confidence=0.0,
            sample_count=qualifying,
            candidate_count=len(accel),
            direction_spread_rad=None,
            magnitude_error_mps2=None,
            notes=[
                *notes,
                f"{qualifying} low-dynamic sample(s) out of {len(accel)}, "
                f"below the minimum of {minimum_samples}; no gravity direction claimed",
            ],
        )

    return _estimate_from(
        accel.subset(mask),
        method=GravityMethod.ACCEL_LOW_DYNAMIC,
        candidate_count=len(accel),
        minimum_samples=minimum_samples,
        notes=[
            *notes,
            f"averaged {qualifying} low-dynamic sample(s) of {len(accel)} "
            f"({100.0 * qualifying / len(accel):.1f}%)",
        ],
    )


def _estimate_from(
    series: VectorSeries,
    *,
    method: GravityMethod,
    candidate_count: int,
    minimum_samples: int,
    notes: list[str],
) -> GravityEstimate:
    mean = series.samples.mean(axis=0)
    magnitude = float(np.linalg.norm(mean))
    if magnitude < MIN_VECTOR_NORM:
        return GravityEstimate(
            direction=None,
            magnitude_mps2=None,
            method=GravityMethod.UNAVAILABLE,
            confidence=0.0,
            sample_count=len(series),
            candidate_count=candidate_count,
            direction_spread_rad=None,
            magnitude_error_mps2=None,
            notes=[*notes, "contributing samples cancelled out; no mean direction exists"],
        )

    # Both channels read specific force, which at rest points *up*. The
    # gravity vector is its negation. Spread is measured against the
    # unnegated mean because it is a property of the samples, not of the sign.
    measured_unit = mean / magnitude
    unit = -measured_unit if SPECIFIC_FORCE_OPPOSES_GRAVITY else measured_unit
    spread = _direction_spread(series.samples, measured_unit)
    # Magnitude is judged on the per-sample mean norm, not the norm of the
    # mean: averaging vectors that point different ways shortens the result,
    # and that shortening is the spread's business, not the magnitude's.
    mean_norm = float(np.mean(series.norms))
    magnitude_error = mean_norm - STANDARD_GRAVITY_MPS2
    confidence, confidence_notes = _confidence(
        spread, magnitude_error, len(series), minimum_samples
    )

    return GravityEstimate(
        direction=Vector3.from_array(unit, Frame.PHONE),
        magnitude_mps2=mean_norm,
        method=method,
        confidence=confidence,
        sample_count=len(series),
        candidate_count=candidate_count,
        direction_spread_rad=spread,
        magnitude_error_mps2=magnitude_error,
        notes=[*notes, *confidence_notes],
    )


def linear_acceleration(accel: VectorSeries, gravity: GravityEstimate) -> VectorSeries:
    """Recover linear acceleration from specific force.

    An accelerometer measures ``f = a − g``, where ``g`` is the gravity vector
    (pointing down). So the linear acceleration is ``a = f + g`` — the gravity
    vector is **added**, not subtracted. Subtracting it doubles the vertical
    term instead of cancelling it, which is easy to write and easy to miss,
    because the result still looks like an acceleration.

    Requires an available estimate: an unavailable gravity would mean adding a
    guess to every sample.
    """
    if gravity.direction is None or gravity.magnitude_mps2 is None:
        raise ValueError(
            f"cannot remove gravity without an available estimate; method was {gravity.method}"
        )
    if accel.frame != gravity.direction.frame:
        raise ValueError(
            f"accelerometer is in {accel.frame} but gravity was estimated in "
            f"{gravity.direction.frame}"
        )
    offset = gravity.direction.to_array() * gravity.magnitude_mps2
    return VectorSeries(
        samples=accel.samples + offset,
        analysis_time_s=accel.analysis_time_s,
        frame=accel.frame,
        source_time_s=accel.source_time_s,
        quality_flags=accel.quality_flags,
        source=accel.source,
        notes=[*accel.notes, f"gravity removed using a {gravity.method} estimate"],
    )
