"""Strapdown inertial mechanization — the Phase 4 baseline.

The chain, in the order the code performs it:

.. code-block:: text

    f_phone (measured)
        − accelerometer bias                    idr.navigation.bias
        ↓
    R_vehicle_phone                             Phase 3 AlignmentEstimate
        ↓
    f_vehicle                                   diagnostics, and the chain below
        ↓
    R_navigation_vehicle                        Phase 3 orientation ∘ alignment⁻¹
        ↓
    f_navigation
        + g_navigation                          idr.navigation.gravity
        ↓
    a_navigation
        ↓ trapezoidal, on actual Δt             idr.navigation.integration
    velocity → position

**The correction budget, in full.** This is the number that makes a Phase 5
ablation meaningful, so it is stated in the code and not only in the report.

*Used:* the Phase 3 orientation estimate (or a pure gyro propagation, never
both — see :class:`AttitudeMode`); the Phase 3 phone→vehicle alignment; a
gravity model; a fixed IMU bias if one is configured; an initial position,
velocity and attitude.

*Not used:* GNSS position or velocity after initialization; the Phase 2
reference velocity in any form; map matching; non-holonomic constraints; a
zero-velocity update; a learned correction; any filter. Nothing in this module
reads a reference signal, and :mod:`idr.navigation.reference_alignment` — which
does — cannot write back into a trajectory.

**What is not compensated.** No coning correction, no sculling correction, no
midpoint attitude update. Each matters when the rotation rate and the specific
force vary appreciably *within* one sample interval; at the 10 Hz that dominates
this dataset, and against an accelerometer bias measured in tenths of a m/s²,
they are far from the leading error term. The state interface is arranged so
they can be added inside :meth:`StrapdownMechanization.step` without changing
anything a caller sees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from idr.frames import quaternion as quat
from idr.frames.alignment import AlignmentEstimate
from idr.frames.conventions import Frame
from idr.frames.rotations import FrameRotation
from idr.navigation.bias import ImuBias
from idr.navigation.gravity import ConstantGravity, GravityModel, linear_acceleration_nav
from idr.navigation.integration import (
    InvalidStepAction,
    InvalidTimeStepError,
    StepVerdict,
    TimeStepPolicy,
    trapezoidal_step,
)
from idr.navigation.state import InitialState, NavigationFlag, NavigationState

#: Largest specific-force magnitude treated as a real measurement, m/s².
#: 15 g. A vehicle crash peaks around 20–40 g and no legitimate driving sample
#: approaches this; a reading beyond it is a sensor fault or a decode error,
#: and integrating it would move the solution kilometres in a second.
MAX_PLAUSIBLE_SPECIFIC_FORCE_MPS2 = 147.0

#: Largest angular rate treated as a real measurement, rad/s. Matches
#: ``idr.frames.orientation.MAX_PLAUSIBLE_RATE_RPS`` — 35 rad/s is 2000°/s, the
#: full scale of a typical consumer MEMS gyroscope.
MAX_PLAUSIBLE_RATE_RPS = 35.0

#: Alignment confidence below which a sample is flagged
#: ``ALIGNMENT_LOW_CONFIDENCE``. Phase 3 measured a median confidence of 0.17
#: across accepted alignments, so this is not a rejection threshold — it is an
#: annotation, and setting it where most real segments fall is the point: the
#: reader should see that most of this dataset's alignments are weak.
LOW_ALIGNMENT_CONFIDENCE = 0.25


class AttitudeMode(StrEnum):
    """Where attitude comes from. Exactly one source, never a blend.

    This is the §10 requirement and it is enforced structurally rather than by
    convention: in :data:`PHASE3_FILTER` the propagator is never run, and in
    :data:`GYRO_PROPAGATION` a supplied orientation is ignored with a note. A
    mechanization that took the Phase 3 estimate *and* integrated the gyro *and*
    corrected with gravity would be counting the same measurements two or three
    times, and the resulting attitude would look better than either input while
    being answerable to neither.
    """

    #: Attitude is the Phase 3 complementary-filter output at each sample.
    #: The Phase 4 baseline. That filter already fuses gyro, gravity and — where
    #: trusted — the magnetometer, on actual Δt.
    PHASE3_FILTER = "phase3_filter"
    #: Attitude is integrated from the gyroscope alone, starting from the
    #: initial attitude and never corrected. Implemented for comparison: it is
    #: what an unaided attitude solution does, and the gap between the two is
    #: the value the Phase 3 filter adds.
    GYRO_PROPAGATION = "gyro_propagation"


class AlignmentPolicy(StrEnum):
    """What to do when ``R_vehicle_phone`` is unavailable.

    Never an identity fallback. Phase 3 refuses to fabricate yaw precisely so
    that the refusal survives into this phase; substituting identity here would
    quietly undo that and produce a trajectory heading off in a direction
    nothing measured.
    """

    #: Refuse to run. The default, and the right answer for a navigation result.
    REQUIRE = "require"
    #: Run without the vehicle frame, marking every sample
    #: ``ALIGNMENT_UNAVAILABLE``. Legitimate because a strapdown solution
    #: actually needs ``R_navigation_phone``, not the vehicle frame — but the
    #: output is a diagnostic, and :attr:`MechanizationRun.is_navigation_grade`
    #: says so, because the vehicle-frame quantities every later phase wants
    #: (longitudinal/lateral split, NHC, wheel-speed comparison) do not exist.
    PHONE_FRAME_DIAGNOSTIC = "phone_frame_diagnostic"


@dataclass(frozen=True)
class MechanizationConfig:
    """Everything that changes what the mechanization does, in one place."""

    gravity_model: GravityModel = field(default_factory=ConstantGravity)
    attitude_mode: AttitudeMode = AttitudeMode.PHASE3_FILTER
    alignment_policy: AlignmentPolicy = AlignmentPolicy.REQUIRE
    time_step: TimeStepPolicy = field(default_factory=TimeStepPolicy)
    bias: ImuBias = field(default_factory=ImuBias.zero)
    #: Latitude and altitude hints handed to the gravity model. The constant
    #: baseline ignores both; a substituted model may not.
    latitude_deg: float | None = None
    altitude_m: float | None = None
    #: Median Δt of the segment, for the irregularity check. Measured by the
    #: caller because the mechanization sees one sample at a time.
    median_dt_s: float | None = None

    def describe(self) -> str:
        return (
            f"attitude from {self.attitude_mode}; alignment policy "
            f"{self.alignment_policy}; gravity {self.gravity_model.description}; "
            f"{self.time_step.describe()}"
        )


class MechanizationError(RuntimeError):
    """Raised when a run cannot legitimately begin or continue."""


@dataclass
class StepCounters:
    """What happened over a run, counted rather than inferred afterwards."""

    integrated: int = 0
    skipped_non_positive: int = 0
    skipped_too_large: int = 0
    skipped_not_finite: int = 0
    irregular: int = 0
    invalid_sensor: int = 0
    attitude_updates: int = 0

    @property
    def skipped(self) -> int:
        return self.skipped_non_positive + self.skipped_too_large + self.skipped_not_finite

    def as_dict(self) -> dict[str, int]:
        return {
            "integrated_steps": self.integrated,
            "skipped_non_positive_dt": self.skipped_non_positive,
            "skipped_large_gap": self.skipped_too_large,
            "skipped_non_finite_dt": self.skipped_not_finite,
            "irregular_dt": self.irregular,
            "invalid_sensor_samples": self.invalid_sensor,
            "attitude_updates": self.attitude_updates,
        }


class StrapdownMechanization:
    """Propagate attitude, velocity and position from IMU samples.

    Stateful and sample-at-a-time, so that a caller — a later filter, most
    obviously — can interleave its own work between steps without this class
    needing to know about it. :func:`mechanize_series` wraps it for the batch
    case.

    The state is never corrected from outside once :meth:`initialize` has run.
    There is deliberately no ``update``, ``reset_velocity`` or ``apply_fix``
    method: adding one would make the Phase 5 comparison meaningless, and its
    absence is easier to verify than its disuse.
    """

    def __init__(self, config: MechanizationConfig | None = None) -> None:
        self.config = config or MechanizationConfig()
        self._state: NavigationState | None = None
        self._initial: InitialState | None = None
        self._alignment: AlignmentEstimate | None = None
        self._rotation_vehicle_phone: FrameRotation | None = None
        self._previous_acceleration: np.ndarray | None = None
        self._gravity_nav = self.config.gravity_model.acceleration_nav(
            latitude_deg=self.config.latitude_deg, altitude_m=self.config.altitude_m
        )
        self.counters = StepCounters()
        self.notes: list[str] = []

    # -- lifecycle ---------------------------------------------------------

    def initialize(
        self, initial: InitialState, *, alignment: AlignmentEstimate | None = None
    ) -> NavigationState:
        """Set the starting state and the mounting rotation for the run.

        ``alignment`` is taken once here rather than per step because a phone
        does not remount itself mid-segment. :meth:`step` accepts an override
        for the case where a caller genuinely has a time-varying estimate, but
        the ordinary path sets it once and the API says so.
        """
        self._alignment = alignment
        self._rotation_vehicle_phone = self._resolve_alignment(alignment)
        self._state = initial.to_state()
        self._initial = initial
        self._previous_acceleration = None
        self.counters = StepCounters()
        self.notes = list(initial.notes)
        return self._state

    def _resolve_alignment(self, alignment: AlignmentEstimate | None) -> FrameRotation | None:
        rotation = alignment.rotation if alignment is not None else None
        if rotation is not None:
            return rotation
        if self.config.alignment_policy is AlignmentPolicy.REQUIRE:
            status = "no alignment supplied" if alignment is None else f"status {alignment.status}"
            raise MechanizationError(
                f"no phone→vehicle rotation is available ({status}). The configured "
                f"policy is {AlignmentPolicy.REQUIRE}, and substituting identity would "
                "assert a mounting that was never measured. Either supply an alignment "
                f"whose yaw resolved, or select {AlignmentPolicy.PHONE_FRAME_DIAGNOSTIC} "
                "and read the result as a diagnostic rather than a navigation solution."
            )
        return None

    @property
    def state(self) -> NavigationState:
        if self._state is None:
            raise MechanizationError("initialize() must be called before the state is available")
        return self._state

    @property
    def is_initialized(self) -> bool:
        return self._state is not None

    @property
    def gravity_nav(self) -> np.ndarray:
        """``g`` in ENU — pointing down, so approximately ``[0, 0, -9.81]``."""
        return self._gravity_nav.copy()

    # -- the step ----------------------------------------------------------

    def step(
        self,
        *,
        timestamp: float,
        specific_force: np.ndarray,
        angular_rate: np.ndarray | None = None,
        orientation: np.ndarray | None = None,
        alignment: AlignmentEstimate | None = None,
        stationary: bool = False,
        index: int = -1,
    ) -> NavigationState:
        """Propagate one sample. ``specific_force`` and ``angular_rate`` are phone-frame.

        Both sensor arguments are the **measured** values; the configured bias
        is subtracted here, inside the phone frame, which is the only frame a
        bias is defined in.
        """
        if self._state is None:
            raise MechanizationError("initialize() must be called before step()")
        if alignment is not None:
            self._alignment = alignment
            self._rotation_vehicle_phone = self._resolve_alignment(alignment)

        previous = self._state
        dt = float(timestamp) - previous.analysis_time_s
        verdict = self.config.time_step.classify(dt)
        flags = NavigationFlag.GOOD
        if stationary:
            flags |= NavigationFlag.STATIONARY

        if verdict is not StepVerdict.VALID:
            return self._handle_invalid_step(
                verdict=verdict,
                dt=dt,
                timestamp=float(timestamp),
                orientation=orientation,
                flags=flags,
                index=index,
            )
        if self.config.time_step.is_irregular(dt, self.config.median_dt_s):
            flags |= NavigationFlag.IRREGULAR_DT
            self.counters.irregular += 1

        force = self.config.bias.accelerometer.correct(np.asarray(specific_force, dtype="float64"))
        rate = (
            None
            if angular_rate is None
            else self.config.bias.gyroscope.correct(np.asarray(angular_rate, dtype="float64"))
        )

        if not _is_plausible(force, MAX_PLAUSIBLE_SPECIFIC_FORCE_MPS2) or (
            rate is not None and not _is_plausible(rate, MAX_PLAUSIBLE_RATE_RPS)
        ):
            flags |= NavigationFlag.INVALID_SENSOR
            self.counters.invalid_sensor += 1
            # Hold the state rather than integrating a value known to be wrong.
            # Sanitizing it into something plausible would produce a trajectory
            # that no measurement supports and no flag reveals.
            #
            # The accumulator is dropped for the same reason an invalid Δt drops
            # it: the acceleration at this instant is unknown, so the next valid
            # step must integrate over its own interval alone rather than
            # trapezoiding against a rate measured before the bad sample and
            # treating it as though it had held across the gap.
            self._previous_acceleration = None
            self._state = NavigationState(
                analysis_time_s=float(timestamp),
                position_m=previous.position_m,
                velocity_mps=previous.velocity_mps,
                orientation=previous.orientation,
                flags=flags,
            )
            return self._state

        attitude = self._attitude(previous.orientation, rate, orientation, dt)
        rotation_nav_phone = quat.to_rotation_matrix(attitude)

        force_vehicle: np.ndarray | None = None
        if self._rotation_vehicle_phone is not None:
            force_vehicle = self._rotation_vehicle_phone.matrix @ force
            # R_nav_vehicle = R_nav_phone @ R_phone_vehicle, and R_phone_vehicle
            # is the transpose of the alignment. Routing f through the vehicle
            # frame this way is algebraically identical to R_nav_phone @ f, so
            # the documented chain costs nothing and f_vehicle falls out of it.
            rotation_nav_vehicle = rotation_nav_phone @ self._rotation_vehicle_phone.matrix.T
            force_nav = rotation_nav_vehicle @ force_vehicle
            weak_alignment = (
                self._alignment is not None
                and self._alignment.confidence < LOW_ALIGNMENT_CONFIDENCE
            )
            if weak_alignment:
                flags |= NavigationFlag.ALIGNMENT_LOW_CONFIDENCE
        else:
            flags |= NavigationFlag.ALIGNMENT_UNAVAILABLE
            force_nav = rotation_nav_phone @ force

        acceleration = linear_acceleration_nav(force_nav, self._gravity_nav)
        previous_acceleration = (
            acceleration if self._previous_acceleration is None else self._previous_acceleration
        )
        velocity = trapezoidal_step(previous.velocity_mps, previous_acceleration, acceleration, dt)
        position = trapezoidal_step(previous.position_m, previous.velocity_mps, velocity, dt)

        self._previous_acceleration = acceleration
        self.counters.integrated += 1
        self._state = NavigationState(
            analysis_time_s=float(timestamp),
            position_m=position,
            velocity_mps=velocity,
            orientation=attitude,
            specific_force_phone=force,
            acceleration_nav=acceleration,
            flags=flags,
        )
        return self._state

    def _handle_invalid_step(
        self,
        *,
        verdict: StepVerdict,
        dt: float,
        timestamp: float,
        orientation: np.ndarray | None,
        flags: NavigationFlag,
        index: int,
    ) -> NavigationState:
        """Apply the configured policy to a Δt that cannot be integrated across."""
        assert self._state is not None
        if verdict is StepVerdict.NON_POSITIVE:
            flags |= NavigationFlag.DUPLICATE_TIMESTAMP
            self.counters.skipped_non_positive += 1
        elif verdict is StepVerdict.TOO_LARGE:
            flags |= NavigationFlag.LARGE_TIME_GAP
            self.counters.skipped_too_large += 1
        else:
            flags |= NavigationFlag.INVALID_SENSOR
            self.counters.skipped_not_finite += 1

        action = self.config.time_step.action
        if action is InvalidStepAction.FAIL:
            raise InvalidTimeStepError(dt, verdict, index)
        if action is InvalidStepAction.SEGMENT:
            raise TrajectoryBreakError(dt, verdict, index)

        # SKIP: hold position and velocity, because nothing was propagated. In
        # PHASE3_FILTER mode the attitude is still adopted — it is a supplied
        # measurement at this instant, not something integrated across the gap,
        # and holding it would misreport an orientation the filter did resolve.
        attitude = self._state.orientation
        if self.config.attitude_mode is AttitudeMode.PHASE3_FILTER and orientation is not None:
            attitude = quat.normalize(orientation)
            self.counters.attitude_updates += 1

        # A held state advances in time but not in space; the previous
        # acceleration is dropped so the next valid step does not trapezoid
        # against a rate from before the gap.
        self._previous_acceleration = None
        self._state = NavigationState(
            analysis_time_s=timestamp if math.isfinite(timestamp) else self._state.analysis_time_s,
            position_m=self._state.position_m,
            velocity_mps=self._state.velocity_mps,
            orientation=attitude,
            flags=flags,
        )
        return self._state

    def _attitude(
        self,
        previous: np.ndarray,
        rate: np.ndarray | None,
        supplied: np.ndarray | None,
        dt: float,
    ) -> np.ndarray:
        """The single attitude source for this run. See :class:`AttitudeMode`."""
        if self.config.attitude_mode is AttitudeMode.PHASE3_FILTER:
            if supplied is None:
                raise MechanizationError(
                    f"{AttitudeMode.PHASE3_FILTER} needs an orientation at every step; "
                    "none was supplied. Pass the Phase 3 estimate, or select "
                    f"{AttitudeMode.GYRO_PROPAGATION} to integrate the gyroscope instead."
                )
            self.counters.attitude_updates += 1
            return quat.normalize(supplied)

        # GYRO_PROPAGATION: integrate, and ignore any supplied orientation
        # rather than blending it in. The blend is the double-counting §10 is
        # about, and it is much easier to avoid here than to detect later.
        if rate is None:
            return previous
        self.counters.attitude_updates += 1
        return propagate_attitude(previous, rate, dt)


class TrajectoryBreakError(RuntimeError):
    """Signals that :data:`InvalidStepAction.SEGMENT` ended the current trajectory."""

    def __init__(self, dt_s: float, verdict: StepVerdict, index: int) -> None:
        super().__init__(
            f"sample {index} has dt={dt_s!r} ({verdict}); the configured policy ends "
            "the trajectory here and requires re-initialization"
        )
        self.dt_s = dt_s
        self.verdict = verdict
        self.index = index


def propagate_attitude(q: np.ndarray, angular_rate: np.ndarray, dt_s: float) -> np.ndarray:
    """Advance ``q_navigation_phone`` by a body-frame rotation increment.

    ``δθ = ω·Δt`` is the rotation the *body* undergoes, so the increment
    multiplies on the **right**:

    .. code-block:: text

        q_nav_phone(k) = q_nav_phone(k−1) ⊗ δq(δθ)

    Left-multiplying instead would apply the increment about navigation-frame
    axes, which is a different rotation entirely whenever the device is not
    already level and facing north — and it produces an attitude that looks
    perfectly well-formed.

    The exponential map is exact for a constant rate over the interval; it is
    not a small-angle approximation, so a 90°-per-step rotation propagates
    correctly. What it does *not* capture is a rate that rotates within the
    interval — that is the coning error, and it is not compensated here.

    Below :data:`~idr.frames.conventions.MIN_VECTOR_NORM` of total angle the
    axis is undefined and the increment is the identity, which is returned
    directly rather than normalizing a near-zero vector into a confident
    direction.
    """
    rate = np.asarray(angular_rate, dtype="float64")
    if rate.shape != (3,):
        raise ValueError(f"angular rate must have shape (3,), got {rate.shape}")
    if not math.isfinite(dt_s):
        raise ValueError(f"dt must be finite, got {dt_s!r}")
    if not np.isfinite(rate).all():
        raise ValueError(f"angular rate must be finite, got {rate!r}")

    delta = rate * dt_s
    angle = float(np.linalg.norm(delta))
    if angle < 1e-12:
        return quat.normalize(q)
    increment = quat.from_axis_angle(delta / angle, angle)
    return quat.normalize(quat.multiply(quat.normalize(q), increment))


def navigation_from_vehicle(
    q_navigation_phone: np.ndarray, alignment_rotation: FrameRotation
) -> FrameRotation:
    """``R_navigation_vehicle`` from the two rotations that are actually estimated.

    Neither Phase 3 output is this rotation: the orientation filter produces
    ``R_navigation_phone`` and the alignment produces ``R_vehicle_phone``. The
    vehicle→navigation rotation is their composition through the phone frame,
    and :class:`~idr.frames.rotations.FrameRotation` checks that the inner
    frames meet rather than trusting the caller to have inverted the right one.
    """
    if alignment_rotation.to_frame is not Frame.VEHICLE:
        raise ValueError(f"expected R_vehicle_phone, got {alignment_rotation.name}")
    nav_phone = FrameRotation.from_quaternion(q_navigation_phone, Frame.NAVIGATION, Frame.PHONE)
    return nav_phone.compose(alignment_rotation.inverse())


def _is_plausible(values: np.ndarray, limit: float) -> bool:
    if not np.isfinite(values).all():
        return False
    return bool(np.linalg.norm(values) <= limit)
