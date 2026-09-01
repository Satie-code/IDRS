"""IMU bias representation and stationary initialization.

An unaided inertial solution is dominated by bias. A gyroscope bias of
0.01 rad/s tilts the attitude by 0.6° in a second, which leaks gravity into the
horizontal channels at about 0.1 m/s² and grows the position error
*cubically*. An accelerometer bias of 0.05 m/s² is 90 m of position error in
60 s on its own. So bias gets a typed representation here rather than a float
passed around, and Phase 5's filter will estimate what this phase can only
accept as configuration.

Phase 4 supports a **fixed** bias supplied at initialization or estimated once
from a stationary window. It does not estimate bias adaptively — that is a
filter state, and it belongs to the fusion phase.

The uncomfortable part, stated up front: **an accelerometer bias is not fully
observable from a stationary window.** See :func:`estimate_bias_from_rest`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from idr.frames.conventions import MIN_VECTOR_NORM, Frame
from idr.frames.gravity import GravityEstimate
from idr.frames.vectors import VectorSeries

#: Earth's rotation rate, rad/s. Quoted only to justify ignoring it: a
#: consumer-grade MEMS gyroscope's noise floor is one to two orders of
#: magnitude above this, so the component of Earth rate a stationary window
#: would attribute to bias is unmeasurable here. A navigation-grade IMU would
#: require compensating it, and a later phase using one must revisit this.
EARTH_ROTATION_RATE_RPS = 7.292115e-5

#: Largest bias magnitude that is plausibly a bias rather than a broken sensor.
#: A consumer MEMS gyroscope's run-to-run bias is typically well under
#: 0.05 rad/s (≈3°/s). 0.35 rad/s is 20°/s — seven times that, and far more
#: plausibly a window that was not stationary than a real offset. Accepting it
#: as a calibration constant would corrupt every subsequent run.
MAX_PLAUSIBLE_GYRO_BIAS_RPS = 0.35
#: Likewise for the accelerometer. 2 m/s² is a fifth of gravity — far beyond
#: any real offset and a sign the "stationary" window was not stationary.
MAX_PLAUSIBLE_ACCEL_BIAS_MPS2 = 2.0


@dataclass(frozen=True)
class GyroscopeBias:
    """Additive gyroscope offset in the phone frame, rad/s.

    The convention, stated once: the sensor reports
    ``omega_measured = omega_true + bias``, so correction **subtracts**.
    """

    values_rps: np.ndarray
    #: How the bias was obtained, written into artifacts verbatim.
    source: str = "unspecified"
    #: 1σ uncertainty per axis where it is known, rad/s.
    sigma_rps: np.ndarray | None = None

    def __post_init__(self) -> None:
        _require_triple(self.values_rps, "gyroscope bias")

    @classmethod
    def zero(cls) -> GyroscopeBias:
        return cls(np.zeros(3, dtype="float64"), source="assumed zero")

    @property
    def magnitude_rps(self) -> float:
        return float(np.linalg.norm(self.values_rps))

    @property
    def is_plausible(self) -> bool:
        return self.magnitude_rps <= MAX_PLAUSIBLE_GYRO_BIAS_RPS

    def correct(self, measured: np.ndarray) -> np.ndarray:
        """``omega_true = omega_measured − bias``."""
        return np.asarray(measured, dtype="float64") - self.values_rps


@dataclass(frozen=True)
class AccelerometerBias:
    """Additive accelerometer offset in the phone frame, m/s².

    ``f_measured = f_true + bias``, so correction subtracts.

    ``observable_axes`` records which components were actually determined. A
    stationary estimate can only resolve the component parallel to gravity, so
    the other two are zero *by construction* rather than by measurement, and a
    consumer that treats all three as equally supported would be wrong.
    """

    values_mps2: np.ndarray
    source: str = "unspecified"
    sigma_mps2: np.ndarray | None = None
    observable_axes: str = "none"

    def __post_init__(self) -> None:
        _require_triple(self.values_mps2, "accelerometer bias")

    @classmethod
    def zero(cls) -> AccelerometerBias:
        return cls(np.zeros(3, dtype="float64"), source="assumed zero", observable_axes="none")

    @property
    def magnitude_mps2(self) -> float:
        return float(np.linalg.norm(self.values_mps2))

    @property
    def is_plausible(self) -> bool:
        return self.magnitude_mps2 <= MAX_PLAUSIBLE_ACCEL_BIAS_MPS2

    def correct(self, measured: np.ndarray) -> np.ndarray:
        """``f_true = f_measured − bias``."""
        return np.asarray(measured, dtype="float64") - self.values_mps2


@dataclass(frozen=True)
class ImuBias:
    """Both biases together, so a run carries one calibration object."""

    gyroscope: GyroscopeBias
    accelerometer: AccelerometerBias
    notes: list[str] = field(default_factory=list)

    @classmethod
    def zero(cls) -> ImuBias:
        return cls(
            gyroscope=GyroscopeBias.zero(),
            accelerometer=AccelerometerBias.zero(),
            notes=["no bias correction applied"],
        )

    @property
    def is_zero(self) -> bool:
        return self.gyroscope.magnitude_rps == 0.0 and self.accelerometer.magnitude_mps2 == 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "gyro_bias_rps": [float(value) for value in self.gyroscope.values_rps],
            "gyro_bias_magnitude_rps": self.gyroscope.magnitude_rps,
            "gyro_bias_source": self.gyroscope.source,
            "accel_bias_mps2": [float(value) for value in self.accelerometer.values_mps2],
            "accel_bias_magnitude_mps2": self.accelerometer.magnitude_mps2,
            "accel_bias_source": self.accelerometer.source,
            "accel_bias_observable_axes": self.accelerometer.observable_axes,
            "notes": list(self.notes),
        }


def _require_triple(values: np.ndarray, label: str) -> None:
    array = np.asarray(values, dtype="float64")
    if array.shape != (3,):
        raise ValueError(f"{label} must have shape (3,), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite, got {array!r}")


def estimate_bias_from_rest(
    *,
    gyro: VectorSeries | None,
    accel: VectorSeries | None,
    mask: np.ndarray,
    gravity: GravityEstimate | None = None,
    expected_gravity_mps2: float | None = None,
) -> ImuBias:
    """Estimate what a stationary window can actually determine — and only that.

    **Gyroscope.** At rest the true angular rate is zero (Earth rate is
    ignored; see :data:`EARTH_ROTATION_RATE_RPS`), so the mean measured rate
    *is* the bias. This estimate is sound.

    **Accelerometer.** This is where the obvious approach is wrong, and §21 of
    the phase brief warns about it specifically. At rest the sensor reads
    ``f = −g + b``. The mean of that window therefore confounds the bias with
    gravity, and the two cannot be separated by any amount of averaging:

    * The **direction** of gravity in the phone frame is unknown a priori — it
      is precisely what Phase 3 estimates *from this same mean*. Using that
      estimate here and then subtracting would give ``b ≡ 0`` identically: a
      number that looks like a measurement and contains none.
    * What *is* separable is the component along gravity, and only by appealing
      to an external fact: the local gravity magnitude. If the window's mean
      specific force has magnitude 9.85 m/s² where gravity is 9.80665, the
      0.043 m/s² excess is bias along that axis — assuming the magnitude is
      right, which is itself a model.
    * The two components **perpendicular** to gravity remain unobservable. They
      are indistinguishable from the device being tilted slightly differently
      than believed, and no stationary data resolves that.

    So this function returns a bias whose radial component is estimated and
    whose tangential components are exactly zero, and records
    ``observable_axes="gravity_parallel"`` so no caller mistakes the zeros for
    measurements. Resolving the other two axes needs motion with known
    dynamics, or a filter with an independent position reference — Phase 5.
    """
    notes: list[str] = []
    selection = np.asarray(mask, dtype=bool)
    count = int(selection.sum())

    gyro_bias = GyroscopeBias.zero()
    if gyro is None:
        notes.append("no gyroscope series; gyroscope bias left at zero")
    elif len(gyro) != selection.size:
        notes.append(
            f"gyroscope has {len(gyro)} samples against a {selection.size}-sample mask; "
            "bias left at zero rather than aligning two streams by position"
        )
    elif count == 0:
        notes.append("no stationary samples were identified; gyroscope bias left at zero")
    else:
        window = gyro.samples[selection]
        finite = np.isfinite(window).all(axis=1)
        if not finite.any():
            notes.append("every stationary gyroscope sample was non-finite")
        else:
            mean = window[finite].mean(axis=0)
            sigma = window[finite].std(axis=0, ddof=1) if int(finite.sum()) > 1 else None
            candidate = GyroscopeBias(
                values_rps=mean,
                source=f"mean of {int(finite.sum())} stationary samples",
                sigma_rps=sigma,
            )
            if candidate.is_plausible:
                gyro_bias = candidate
                notes.append(
                    f"gyroscope bias {candidate.magnitude_rps:.5f} rad/s from "
                    f"{int(finite.sum())} stationary samples"
                )
            else:
                notes.append(
                    f"rejected a gyroscope bias of {candidate.magnitude_rps:.4f} rad/s: "
                    f"beyond the {MAX_PLAUSIBLE_GYRO_BIAS_RPS} rad/s plausibility bound, "
                    "so the window was probably not stationary"
                )

    accel_bias = _accelerometer_bias_from_rest(
        accel=accel,
        selection=selection,
        gravity=gravity,
        expected_gravity_mps2=expected_gravity_mps2,
        notes=notes,
    )
    return ImuBias(gyroscope=gyro_bias, accelerometer=accel_bias, notes=notes)


def _accelerometer_bias_from_rest(
    *,
    accel: VectorSeries | None,
    selection: np.ndarray,
    gravity: GravityEstimate | None,
    expected_gravity_mps2: float | None,
    notes: list[str],
) -> AccelerometerBias:
    """The gravity-parallel component only. See :func:`estimate_bias_from_rest`."""
    if accel is None:
        notes.append("no accelerometer series; accelerometer bias left at zero")
        return AccelerometerBias.zero()
    if len(accel) != selection.size:
        notes.append(
            f"accelerometer has {len(accel)} samples against a {selection.size}-sample mask; "
            "accelerometer bias left at zero"
        )
        return AccelerometerBias.zero()
    if expected_gravity_mps2 is None:
        notes.append(
            "no expected gravity magnitude supplied, so no component of the "
            "accelerometer bias is observable; left at zero"
        )
        return AccelerometerBias.zero()
    if gravity is None or gravity.direction is None:
        notes.append(
            "no gravity direction available, so the observable axis of the "
            "accelerometer bias is undefined; left at zero"
        )
        return AccelerometerBias.zero()

    window = accel.samples[selection]
    finite = np.isfinite(window).all(axis=1)
    if int(finite.sum()) < 2:
        notes.append("fewer than two finite stationary accelerometer samples; bias left at zero")
        return AccelerometerBias.zero()

    mean = window[finite].mean(axis=0)
    measured_magnitude = float(np.linalg.norm(mean))
    if measured_magnitude < MIN_VECTOR_NORM:
        notes.append("the stationary accelerometer mean has no direction; bias left at zero")
        return AccelerometerBias.zero()

    # At rest f = −g + b. Along the measured "up" direction (−ĝ), the reading
    # should be exactly |g|; whatever it exceeds that by is the bias component
    # on this axis. Off-axis components are not recoverable — see the docstring.
    up = -gravity.direction.to_array()
    along = float(mean @ up)
    excess = along - expected_gravity_mps2
    values = excess * up

    candidate = AccelerometerBias(
        values_mps2=values,
        source=(
            f"gravity-parallel component from {int(finite.sum())} stationary samples "
            f"(|f|={measured_magnitude:.4f} vs expected {expected_gravity_mps2:.4f} m/s²)"
        ),
        observable_axes="gravity_parallel",
    )
    if not candidate.is_plausible:
        notes.append(
            f"rejected an accelerometer bias of {candidate.magnitude_mps2:.3f} m/s²: "
            f"beyond the {MAX_PLAUSIBLE_ACCEL_BIAS_MPS2} m/s² plausibility bound"
        )
        return AccelerometerBias.zero()

    notes.append(
        f"accelerometer bias {excess:+.4f} m/s² along the gravity axis only; the two "
        "perpendicular components are unobservable from a stationary window and are "
        "zero by construction, not by measurement"
    )
    return candidate


def perturbed(
    bias: ImuBias, *, gyro_rps: np.ndarray | None = None, accel_mps2: np.ndarray | None = None
) -> ImuBias:
    """A copy with an added perturbation, for the bias-sensitivity experiment.

    Kept here rather than in the experiment so that the perturbation obeys the
    same sign convention as the correction it will pass through: the value
    added is what the sensor is pretending to measure in addition to the truth.
    """
    gyro_delta = np.zeros(3) if gyro_rps is None else np.asarray(gyro_rps, dtype="float64")
    accel_delta = np.zeros(3) if accel_mps2 is None else np.asarray(accel_mps2, dtype="float64")
    labels = []
    if float(np.linalg.norm(gyro_delta)) > 0.0:
        labels.append(f"gyro {math.degrees(float(np.linalg.norm(gyro_delta))):.3f}°/s")
    if float(np.linalg.norm(accel_delta)) > 0.0:
        labels.append(f"accel {float(np.linalg.norm(accel_delta)):.4f} m/s²")
    label = " and ".join(labels) if labels else "no perturbation"
    return ImuBias(
        gyroscope=GyroscopeBias(
            values_rps=bias.gyroscope.values_rps + gyro_delta,
            source=f"{bias.gyroscope.source} + perturbation",
            sigma_rps=bias.gyroscope.sigma_rps,
        ),
        accelerometer=AccelerometerBias(
            values_mps2=bias.accelerometer.values_mps2 + accel_delta,
            source=f"{bias.accelerometer.source} + perturbation",
            sigma_mps2=bias.accelerometer.sigma_mps2,
            observable_axes=bias.accelerometer.observable_axes,
        ),
        notes=[*bias.notes, f"perturbed by {label} for sensitivity analysis"],
    )


def phone_frame_check(series: VectorSeries) -> None:
    """Guard that a series really is phone-frame before a bias is applied to it.

    Biases live in the sensor's own frame. Applying one after a rotation into
    the vehicle or navigation frame subtracts a constant from the wrong axes,
    and the result stays plausible.
    """
    if series.frame is not Frame.PHONE:
        raise ValueError(
            f"IMU biases are defined in the phone frame; this series is in {series.frame}. "
            "Apply the bias before any frame transformation, not after."
        )
