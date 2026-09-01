"""Synthetic trajectories with closed-form solutions.

The only place in Phase 4 where accuracy can be measured. Real IO-VNBD has no
position ground truth, no attitude ground truth, and a velocity *reference*
carrying several seconds of residual synchronization error — so a real-data run
can show that the mechanization behaves plausibly and can never show that it is
right.

These fixtures can. Every one is generated from an analytical solution, run
*backwards* into sensor readings, and the mechanization must recover the
solution it came from:

.. code-block:: text

    p(t), v(t), a(t), q(t)          chosen analytically
              ↓
    f_nav = a_nav − g_nav           the definition of specific force
              ↓
    f_phone = R_nav_phone(t)ᵀ f_nav rotated into the sensor frame
              ↓
    [ mechanization under test ]
              ↓
    p̂(t), v̂(t)                     must equal p(t), v(t)

Constant acceleration is integrated *exactly* by the trapezoidal rule, so the
tests demand agreement at machine precision rather than at a tolerance chosen
until they passed.

The sign of ``f_nav = a_nav − g_nav`` here is the mirror of
``a_nav = f_nav + g_nav`` in the mechanization. Generating the fixture with the
same sign error as the code under test would make the pair self-consistently
wrong, so the two are written from the definition independently, and the
stationary case (:func:`stationary`) pins the result to a number a reader can
check by hand: a level, resting device reads ``+9.80665`` on its up axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from idr.frames import quaternion as quat
from idr.frames.conventions import STANDARD_GRAVITY_MPS2, Frame
from idr.frames.rotations import FrameRotation
from idr.frames.vectors import VectorSeries
from idr.navigation.state import InitialState


@dataclass(frozen=True)
class GenerationParams:
    """Everything :func:`analytic_trajectory` needs to rebuild a drive.

    Retained so that a fixture whose *timestamps* change — a duplicate injected,
    a gap opened — can be regenerated rather than merely relabelled. Without
    this the truth arrays would still describe the timestamps the drive was
    born with, and a test comparing against them would be measuring the
    fixture's staleness instead of the mechanization.
    """

    acceleration_nav_mps2: np.ndarray
    initial_velocity_nav_mps: np.ndarray
    initial_position_nav_m: np.ndarray
    q_navigation_phone: np.ndarray
    body_rate_rps: np.ndarray
    q_vehicle_phone: np.ndarray
    gravity_mps2: float


@dataclass
class SyntheticTrajectory:
    """A generated drive plus the exact answer the mechanization must reproduce."""

    analysis_time_s: np.ndarray
    #: Truth, navigation frame.
    position_nav_m: np.ndarray
    velocity_nav_mps: np.ndarray
    acceleration_nav_mps2: np.ndarray
    #: Truth ``q_navigation_phone`` per sample.
    orientation: np.ndarray
    #: What the sensors report, phone frame, including any injected bias/noise.
    specific_force_phone: VectorSeries
    angular_rate_phone: VectorSeries
    #: Truth ``q_vehicle_phone`` — the mounting rotation the estimator would
    #: have to recover, supplied directly here so mechanization can be tested
    #: without also testing Phase 3's estimator.
    q_vehicle_phone: np.ndarray
    gravity_mps2: float
    #: How this drive was generated, so it can be rebuilt on new timestamps.
    params: GenerationParams | None = None
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.analysis_time_s.size)

    @property
    def gravity_nav(self) -> np.ndarray:
        return np.array([0.0, 0.0, -self.gravity_mps2], dtype="float64")

    @property
    def alignment_rotation(self) -> FrameRotation:
        return FrameRotation.from_quaternion(self.q_vehicle_phone, Frame.VEHICLE, Frame.PHONE)

    def specific_force_vehicle(self) -> np.ndarray:
        """Truth ``f_vehicle``, for the mounting-rotation test."""
        return self.specific_force_phone.samples @ self.alignment_rotation.matrix.T

    def initial_state(self, *, heading_observed: bool = True) -> InitialState:
        """The exact initial state, so a test isolates propagation from initialization."""
        return InitialState(
            analysis_time_s=float(self.analysis_time_s[0]),
            position_m=self.position_nav_m[0].copy(),
            velocity_mps=self.velocity_nav_mps[0].copy(),
            orientation=self.orientation[0].copy(),
            heading_observed=heading_observed,
            velocity_source="synthetic truth",
            position_source="synthetic truth",
            orientation_source="synthetic truth",
            notes=["initialized from the analytical solution"],
        )


def analytic_trajectory(
    *,
    times: np.ndarray | None = None,
    duration_s: float = 20.0,
    rate_hz: float = 10.0,
    acceleration_nav_mps2: np.ndarray | list[float] | None = None,
    initial_velocity_nav_mps: np.ndarray | list[float] | None = None,
    initial_position_nav_m: np.ndarray | list[float] | None = None,
    q_navigation_phone: np.ndarray | None = None,
    body_rate_rps: np.ndarray | list[float] | None = None,
    q_vehicle_phone: np.ndarray | None = None,
    gravity_mps2: float = STANDARD_GRAVITY_MPS2,
    accel_bias_mps2: np.ndarray | list[float] | None = None,
    gyro_bias_rps: np.ndarray | list[float] | None = None,
    noise_mps2: float = 0.0,
    seed: int = 20260901,
) -> SyntheticTrajectory:
    """Generate a trajectory with a constant navigation-frame acceleration.

    Constant acceleration and a constant *body* rotation rate are the two
    profiles with exact closed forms:

    .. code-block:: text

        v(t) = v₀ + a·t
        p(t) = p₀ + v₀·t + ½·a·t²
        q(t) = q₀ ⊗ exp(½·ω·t)

    which covers Tests 1–4 and 6–9 of the phase brief between them (stationary
    is ``a = 0, v₀ = 0``; constant velocity is ``a = 0, v₀ ≠ 0``; pure rotation
    is ``a = 0`` with ``ω ≠ 0``). ``times`` may be passed directly to build the
    irregular-sampling and duplicate-timestamp cases.

    ``accel_bias_mps2`` and ``gyro_bias_rps`` are added to the *reported*
    sensor values, matching the convention in :mod:`idr.navigation.bias` that a
    sensor reads ``true + bias``. The truth arrays are unaffected, which is what
    makes the bias-sensitivity experiment measure a real divergence rather than
    a relabelled one.
    """
    if times is None:
        if rate_hz <= 0.0:
            raise ValueError(f"rate_hz must be positive, got {rate_hz!r}")
        count = max(2, int(round(duration_s * rate_hz)) + 1)
        stamps = np.arange(count, dtype="float64") / rate_hz
    else:
        stamps = np.asarray(times, dtype="float64")
        count = stamps.size
    if count < 2:
        raise ValueError("a trajectory needs at least two samples")

    acceleration = _triple(acceleration_nav_mps2)
    velocity0 = _triple(initial_velocity_nav_mps)
    position0 = _triple(initial_position_nav_m)
    rate = _triple(body_rate_rps)
    q0 = quat.identity() if q_navigation_phone is None else quat.normalize(q_navigation_phone)
    q_mount = quat.identity() if q_vehicle_phone is None else quat.normalize(q_vehicle_phone)

    elapsed = stamps - stamps[0]
    velocity = velocity0 + np.outer(elapsed, acceleration)
    position = (
        position0 + np.outer(elapsed, velocity0) + 0.5 * np.outer(elapsed * elapsed, acceleration)
    )
    accelerations = np.tile(acceleration, (count, 1))

    # Attitude: exact for a constant body rate. Built from the elapsed time at
    # each sample rather than stepped, so the truth carries no integration error
    # of its own for the mechanization to be measured against.
    orientation = np.empty((count, 4), dtype="float64")
    rate_magnitude = float(np.linalg.norm(rate))
    for index, moment in enumerate(elapsed):
        if rate_magnitude < 1e-15:
            orientation[index] = q0
        else:
            increment = quat.from_axis_angle(rate / rate_magnitude, rate_magnitude * moment)
            orientation[index] = quat.multiply(q0, increment)

    gravity_nav = np.array([0.0, 0.0, -gravity_mps2], dtype="float64")
    # f = a − g. Written from the definition, deliberately not by calling the
    # mechanization's helper, so a sign error there cannot cancel here.
    force_nav = accelerations - gravity_nav
    force_phone = np.empty((count, 3), dtype="float64")
    for index in range(count):
        rotation = quat.to_rotation_matrix(orientation[index])
        force_phone[index] = rotation.T @ force_nav[index]

    rates = np.tile(rate, (count, 1))
    if accel_bias_mps2 is not None:
        force_phone = force_phone + _triple(accel_bias_mps2)
    if gyro_bias_rps is not None:
        rates = rates + _triple(gyro_bias_rps)
    if noise_mps2 > 0.0:
        rng = np.random.default_rng(seed)
        force_phone = force_phone + rng.normal(0.0, noise_mps2, size=force_phone.shape)

    notes = [
        f"{count} samples over {float(elapsed[-1]):.2f}s",
        f"constant navigation acceleration {acceleration.tolist()} m/s²",
    ]
    if rate_magnitude > 0.0:
        notes.append(f"constant body rate {rate.tolist()} rad/s")

    return SyntheticTrajectory(
        analysis_time_s=stamps,
        position_nav_m=position,
        velocity_nav_mps=velocity,
        acceleration_nav_mps2=accelerations,
        orientation=orientation,
        specific_force_phone=VectorSeries(
            samples=force_phone,
            analysis_time_s=stamps,
            frame=Frame.PHONE,
            source="synthetic:specific_force",
        ),
        angular_rate_phone=VectorSeries(
            samples=rates,
            analysis_time_s=stamps,
            frame=Frame.PHONE,
            source="synthetic:angular_rate",
        ),
        q_vehicle_phone=q_mount,
        gravity_mps2=gravity_mps2,
        params=GenerationParams(
            acceleration_nav_mps2=acceleration,
            initial_velocity_nav_mps=velocity0,
            initial_position_nav_m=position0,
            q_navigation_phone=q0,
            body_rate_rps=rate,
            q_vehicle_phone=q_mount,
            gravity_mps2=gravity_mps2,
        ),
        notes=notes,
    )


def stationary(
    *, duration_s: float = 30.0, rate_hz: float = 10.0, **kwargs: object
) -> SyntheticTrajectory:
    """A device at rest. The gravity-sign regression case.

    At rest ``a = 0``, so ``f = −g`` and a level phone reads ``+9.80665`` on its
    up axis. The mechanization must return that to zero acceleration; a sign
    error turns it into ``2g`` upward and 300 m of climb in 5.5 s.
    """
    drive = analytic_trajectory(duration_s=duration_s, rate_hz=rate_hz, **kwargs)  # type: ignore[arg-type]
    drive.notes.append("stationary: velocity and position must not move")
    return drive


def constant_acceleration(
    *,
    acceleration_mps2: float = 1.0,
    duration_s: float = 20.0,
    rate_hz: float = 10.0,
    **kwargs: object,
) -> SyntheticTrajectory:
    """Straight-line acceleration along navigation east. ``v = at``, ``p = ½at²``."""
    drive = analytic_trajectory(
        acceleration_nav_mps2=[acceleration_mps2, 0.0, 0.0],
        duration_s=duration_s,
        rate_hz=rate_hz,
        **kwargs,  # type: ignore[arg-type]
    )
    drive.notes.append(f"constant {acceleration_mps2} m/s² east")
    return drive


def constant_velocity(
    *,
    speed_mps: float = 15.0,
    duration_s: float = 20.0,
    rate_hz: float = 10.0,
    **kwargs: object,
) -> SyntheticTrajectory:
    """Cruise at a fixed velocity. ``p = vt``, and the sensors read only gravity."""
    drive = analytic_trajectory(
        initial_velocity_nav_mps=[speed_mps, 0.0, 0.0],
        duration_s=duration_s,
        rate_hz=rate_hz,
        **kwargs,  # type: ignore[arg-type]
    )
    drive.notes.append(f"constant {speed_mps} m/s east; specific force is gravity alone")
    return drive


def pure_rotation(
    *,
    rate_rps: float = 0.3,
    axis: np.ndarray | list[float] | None = None,
    duration_s: float = 20.0,
    rate_hz: float = 20.0,
    **kwargs: object,
) -> SyntheticTrajectory:
    """Rotate in place. Attitude must track; velocity and position must not move.

    The demanding part is not the attitude — it is that the specific force,
    which swings through the whole phone frame as the device turns, must still
    resolve to zero navigation acceleration at every instant. A transposed
    rotation or a left-multiplied attitude increment leaves the residual
    sweeping in a circle, and the position walks off in a spiral.
    """
    direction = np.array([0.0, 0.0, 1.0]) if axis is None else np.asarray(axis, dtype="float64")
    unit = direction / float(np.linalg.norm(direction))
    drive = analytic_trajectory(
        body_rate_rps=unit * rate_rps,
        duration_s=duration_s,
        rate_hz=rate_hz,
        **kwargs,  # type: ignore[arg-type]
    )
    drive.notes.append(
        f"rotating at {math.degrees(rate_rps):.1f}°/s about {unit.tolist()} in the phone frame"
    )
    return drive


def irregular_times(
    *, duration_s: float = 20.0, rate_hz: float = 10.0, jitter_s: float = 0.03, seed: int = 7
) -> np.ndarray:
    """Non-uniform but strictly increasing timestamps.

    Jittered then sorted, and clipped away from zero spacing: a logger delivers
    samples late, not out of order, and Phase 2 would have sorted them anyway.
    Duplicates are a *separate* fixture — see :func:`with_duplicate_timestamps` —
    because they test a different branch.
    """
    count = max(2, int(round(duration_s * rate_hz)) + 1)
    rng = np.random.default_rng(seed)
    base = np.arange(count, dtype="float64") / rate_hz
    jittered = np.sort(base + rng.uniform(-jitter_s, jitter_s, size=count))
    # Guarantee strict increase without changing the character of the jitter.
    minimum = 1e-4
    for index in range(1, count):
        if jittered[index] - jittered[index - 1] < minimum:
            jittered[index] = jittered[index - 1] + minimum
    return jittered


def with_duplicate_timestamps(
    trajectory: SyntheticTrajectory, count: int = 5
) -> SyntheticTrajectory:
    """Repeat some timestamps, as Phase 2 leaves them.

    Phase 2 flags duplicates and keeps the rows — 147,946 of them across this
    dataset — so a Δt of exactly zero reaches the mechanization legitimately and
    has to be handled rather than assumed away.
    """
    stamps = trajectory.analysis_time_s.copy()
    for index in range(1, count + 1):
        target = index * 2
        if target < stamps.size:
            stamps[target] = stamps[target - 1]
    return retimed(trajectory, stamps, f"{count} duplicate timestamp(s) injected")


def with_time_gap(
    trajectory: SyntheticTrajectory, *, at_fraction: float = 0.5, gap_s: float = 5.0
) -> SyntheticTrajectory:
    """Open a gap in the middle, to exercise the large-Δt policy.

    Only the timestamps move; the truth arrays are left as they were, so a test
    can assert what the *policy* did without also having to model what the
    vehicle would have done during a gap nothing observed.
    """
    stamps = trajectory.analysis_time_s.copy()
    index = max(1, min(stamps.size - 1, int(stamps.size * at_fraction)))
    stamps[index:] += gap_s
    return retimed(trajectory, stamps, f"{gap_s:g}s gap opened at sample {index}")


def retimed(trajectory: SyntheticTrajectory, stamps: np.ndarray, note: str) -> SyntheticTrajectory:
    """Rebuild a drive on new timestamps, truth included.

    The truth arrays are *regenerated* from the analytical solution evaluated at
    the new stamps, not carried over. Carrying them over would leave the
    fixture describing the times it was born with — so a duplicated timestamp
    would show a one-step disagreement that belongs to the fixture, and a test
    could not tell that from a real integration error.
    """
    if trajectory.params is None:  # pragma: no cover - every drive records params
        raise ValueError("this trajectory cannot be retimed: no generation parameters")
    params = trajectory.params
    rebuilt = analytic_trajectory(
        times=stamps,
        acceleration_nav_mps2=params.acceleration_nav_mps2,
        initial_velocity_nav_mps=params.initial_velocity_nav_mps,
        initial_position_nav_m=params.initial_position_nav_m,
        q_navigation_phone=params.q_navigation_phone,
        body_rate_rps=params.body_rate_rps,
        q_vehicle_phone=params.q_vehicle_phone,
        gravity_mps2=params.gravity_mps2,
    )
    rebuilt.notes.extend([*trajectory.notes, note])
    return rebuilt


def corrupt_samples(
    trajectory: SyntheticTrajectory, indices: list[int], *, value: float = float("nan")
) -> SyntheticTrajectory:
    """Replace some specific-force samples with NaN or infinity."""
    forces = trajectory.specific_force_phone.samples.copy()
    for index in indices:
        forces[index] = value
    return SyntheticTrajectory(
        analysis_time_s=trajectory.analysis_time_s,
        position_nav_m=trajectory.position_nav_m,
        velocity_nav_mps=trajectory.velocity_nav_mps,
        acceleration_nav_mps2=trajectory.acceleration_nav_mps2,
        orientation=trajectory.orientation,
        specific_force_phone=VectorSeries(
            samples=forces,
            analysis_time_s=trajectory.analysis_time_s,
            frame=Frame.PHONE,
            source="synthetic:specific_force_corrupted",
        ),
        angular_rate_phone=trajectory.angular_rate_phone,
        q_vehicle_phone=trajectory.q_vehicle_phone,
        gravity_mps2=trajectory.gravity_mps2,
        params=trajectory.params,
        notes=[*trajectory.notes, f"samples {indices} set to {value}"],
    )


def _triple(values: np.ndarray | list[float] | None) -> np.ndarray:
    if values is None:
        return np.zeros(3, dtype="float64")
    array = np.asarray(values, dtype="float64").reshape(-1)
    if array.size != 3:
        raise ValueError(f"expected 3 components, got {array.size}")
    return array
