"""The trajectory container, and the batch driver that fills it.

A trajectory is the whole propagated solution plus the provenance needed to
interpret it. Two pieces of that provenance are not optional here:

* **the tangent-plane origin.** A position in metres is not a location until
  something says where the origin is. Carrying them separately is how a
  trajectory ends up plotted against a reference in a different frame.
* **whether heading was ever observed.** In this dataset it usually was not
  (Phase 3: the magnetometer is trusted on 86 of 154 segments, and where it is
  not, the yaw is dead-reckoned from an arbitrary origin). The east/north
  components are then correct only up to an unknown rotation about the vertical,
  and any position error computed against a geo-referenced track inherits it.
  :attr:`Trajectory.heading_observed` is how a consumer finds that out without
  having to know the Phase 3 report by heart.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from idr.frames import quaternion as quat
from idr.frames.alignment import AlignmentEstimate
from idr.frames.conventions import Frame
from idr.frames.vectors import VectorSeries
from idr.navigation.mechanization import (
    AlignmentPolicy,
    MechanizationConfig,
    StepCounters,
    StrapdownMechanization,
    TrajectoryBreakError,
)
from idr.navigation.state import InitialState, NavigationFlag, NavigationState

if TYPE_CHECKING:  # pragma: no cover - typing only
    from idr.navigation.geodesy import TangentPlane


@dataclass
class Trajectory:
    """A propagated solution over one segment.

    Arrays are parallel and all of length ``n``. Storing them column-wise
    rather than as a list of :class:`NavigationState` keeps the plotting and
    error analysis vectorized; :meth:`state_at` reconstructs one state when a
    caller wants the object.
    """

    analysis_time_s: np.ndarray
    position_m: np.ndarray
    velocity_mps: np.ndarray
    orientation: np.ndarray
    acceleration_nav: np.ndarray
    specific_force_phone: np.ndarray
    flags: np.ndarray
    #: Where the local tangent plane is anchored. ``None`` when the run was
    #: never geo-referenced, in which case position is relative to the start.
    origin: TangentPlane | None = None
    #: False when the initial yaw was an arbitrary origin rather than a
    #: measurement. See the module docstring — this is the common case here.
    heading_observed: bool = False
    #: Whether the vehicle frame was available throughout. False means the run
    #: was a phone-frame diagnostic, not a navigation solution.
    is_navigation_grade: bool = True
    counters: StepCounters = field(default_factory=StepCounters)
    initial: InitialState | None = None
    config_description: str = ""
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.analysis_time_s.size)

    @property
    def duration_s(self) -> float:
        if len(self) < 2:
            return 0.0
        return float(self.analysis_time_s[-1] - self.analysis_time_s[0])

    @property
    def speed_mps(self) -> np.ndarray:
        return np.linalg.norm(self.velocity_mps, axis=1)

    @property
    def horizontal_position_m(self) -> np.ndarray:
        """East and north only. The vertical channel of an unaided INS is the
        least trustworthy one — gravity compensation error goes straight into
        it — so horizontal error is reported separately rather than buried in a
        3-D norm."""
        return self.position_m[:, :2]

    def distance_travelled_m(self) -> float:
        """Path length of the propagated solution, not displacement."""
        if len(self) < 2:
            return 0.0
        steps = np.linalg.norm(np.diff(self.position_m, axis=0), axis=1)
        return float(steps[np.isfinite(steps)].sum())

    def displacement_m(self) -> float:
        if len(self) < 2:
            return 0.0
        return float(np.linalg.norm(self.position_m[-1] - self.position_m[0]))

    def state_at(self, index: int) -> NavigationState:
        return NavigationState(
            analysis_time_s=float(self.analysis_time_s[index]),
            position_m=self.position_m[index],
            velocity_mps=self.velocity_mps[index],
            orientation=self.orientation[index],
            specific_force_phone=self.specific_force_phone[index],
            acceleration_nav=self.acceleration_nav[index],
            flags=NavigationFlag(int(self.flags[index])),
        )

    def flag_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for flag in NavigationFlag:
            if flag is NavigationFlag.GOOD:
                continue
            hits = int(np.count_nonzero(self.flags & int(flag)))
            if hits:
                counts[(flag.name or str(flag)).lower()] = hits
        return counts

    def euler_deg(self) -> np.ndarray:
        """``(n, 3)`` roll/pitch/yaw in degrees, for plotting only."""
        out = np.empty((len(self), 3), dtype="float64")
        for index in range(len(self)):
            roll, pitch, yaw = quat.to_euler(self.orientation[index])
            out[index] = np.degrees([roll, pitch, yaw])
        return out

    def as_dict(self) -> dict[str, object]:
        """Scalar summary for artifacts. Never the arrays — those go to Parquet."""
        return {
            "samples": len(self),
            "duration_s": self.duration_s,
            "final_speed_mps": float(self.speed_mps[-1]) if len(self) else 0.0,
            "max_speed_mps": float(np.nanmax(self.speed_mps)) if len(self) else 0.0,
            "displacement_m": self.displacement_m(),
            "path_length_m": self.distance_travelled_m(),
            "heading_observed": self.heading_observed,
            "is_navigation_grade": self.is_navigation_grade,
            "flags": self.flag_counts(),
            "counters": self.counters.as_dict(),
            "config": self.config_description,
        }


def mechanize_series(
    *,
    initial: InitialState,
    specific_force: VectorSeries,
    angular_rate: VectorSeries | None = None,
    orientations: np.ndarray | None = None,
    alignment: AlignmentEstimate | None = None,
    config: MechanizationConfig | None = None,
    stationary_mask: np.ndarray | None = None,
    origin: TangentPlane | None = None,
) -> Trajectory:
    """Drive :class:`StrapdownMechanization` across a whole series.

    ``specific_force`` must be phone-frame: the bias is subtracted inside the
    mechanization and a bias is only defined in the sensor's own frame.

    The first sample is not integrated. It carries the initial state verbatim,
    which makes the analytical tests exact — with ``v₀`` and ``p₀`` given, a
    constant-acceleration trajectory must reproduce ``v = v₀ + at`` and
    ``p = p₀ + v₀t + ½at²`` to machine precision rather than to a tolerance
    that hides an off-by-one in the first step.
    """
    if specific_force.frame is not Frame.PHONE:
        raise ValueError(
            f"mechanization consumes phone-frame specific force, got {specific_force.frame}"
        )
    count = len(specific_force)
    if count == 0:
        raise ValueError("cannot mechanize an empty series")

    settings = config or MechanizationConfig()
    engine = StrapdownMechanization(settings)
    engine.initialize(initial, alignment=alignment)

    times = specific_force.analysis_time_s
    positions = np.full((count, 3), np.nan, dtype="float64")
    velocities = np.full((count, 3), np.nan, dtype="float64")
    attitudes = np.full((count, 4), np.nan, dtype="float64")
    accelerations = np.full((count, 3), np.nan, dtype="float64")
    forces = np.full((count, 3), np.nan, dtype="float64")
    flags = np.zeros(count, dtype="int64")

    def record(index: int, state: NavigationState) -> None:
        positions[index] = state.position_m
        velocities[index] = state.velocity_mps
        attitudes[index] = state.orientation
        if state.acceleration_nav is not None:
            accelerations[index] = state.acceleration_nav
        if state.specific_force_phone is not None:
            forces[index] = state.specific_force_phone
        flags[index] = int(state.flags)

    record(0, engine.state)
    notes: list[str] = []
    stopped_at: int | None = None

    # A gyroscope of the wrong length cannot be indexed alongside the
    # accelerometer, so it is dropped — but silently dropping it would leave a
    # GYRO_PROPAGATION run with a frozen attitude and nothing to say why.
    use_gyro = angular_rate is not None and len(angular_rate) == count
    if angular_rate is not None and not use_gyro:
        notes.append(
            f"the gyroscope series has {len(angular_rate)} samples against "
            f"{count} accelerometer samples, so it was not used; under "
            "GYRO_PROPAGATION the attitude will hold at its initial value"
        )

    for index in range(1, count):
        rate = angular_rate.samples[index] if use_gyro and angular_rate is not None else None
        supplied = None if orientations is None else orientations[index]
        try:
            state = engine.step(
                timestamp=float(times[index]),
                specific_force=specific_force.samples[index],
                angular_rate=rate,
                orientation=supplied,
                stationary=(bool(stationary_mask[index]) if stationary_mask is not None else False),
                index=index,
            )
        except TrajectoryBreakError as broke:
            notes.append(
                f"the trajectory was ended at sample {index} by the segment policy: {broke}"
            )
            stopped_at = index
            break
        record(index, state)

    if stopped_at is not None:
        selection = slice(0, stopped_at)
        times, positions, velocities = times[selection], positions[selection], velocities[selection]
        attitudes, accelerations = attitudes[selection], accelerations[selection]
        forces, flags = forces[selection], flags[selection]

    navigation_grade = (
        settings.alignment_policy is not AlignmentPolicy.PHONE_FRAME_DIAGNOSTIC
        and alignment is not None
        and alignment.rotation is not None
    )
    if not navigation_grade:
        notes.append(
            "no phone→vehicle rotation was available, so this run is a phone-frame "
            "diagnostic: the propagated position is a real inertial solution but the "
            "vehicle-frame quantities every later phase needs do not exist for it"
        )

    return Trajectory(
        analysis_time_s=np.asarray(times, dtype="float64"),
        position_m=positions,
        velocity_mps=velocities,
        orientation=attitudes,
        acceleration_nav=accelerations,
        specific_force_phone=forces,
        flags=flags,
        origin=origin,
        heading_observed=initial.heading_observed,
        is_navigation_grade=navigation_grade,
        counters=engine.counters,
        initial=initial,
        config_description=settings.describe(),
        notes=[*engine.notes, *notes],
    )


def concatenate(pieces: Sequence[Trajectory]) -> Trajectory:
    """Join trajectories that were segmented by the time-step policy.

    Used only for plotting and reporting. The pieces are *not* made continuous:
    each was propagated from its own initial state, so joining them produces a
    single array with a real discontinuity at each join, which is the honest
    representation of what the mechanization produced.
    """
    usable = [piece for piece in pieces if len(piece)]
    if not usable:
        raise ValueError("nothing to concatenate")
    if len(usable) == 1:
        return usable[0]
    counters = StepCounters()
    for piece in usable:
        counters.integrated += piece.counters.integrated
        counters.skipped_non_positive += piece.counters.skipped_non_positive
        counters.skipped_too_large += piece.counters.skipped_too_large
        counters.skipped_not_finite += piece.counters.skipped_not_finite
        counters.irregular += piece.counters.irregular
        counters.invalid_sensor += piece.counters.invalid_sensor
        counters.attitude_updates += piece.counters.attitude_updates
    return Trajectory(
        analysis_time_s=np.concatenate([piece.analysis_time_s for piece in usable]),
        position_m=np.vstack([piece.position_m for piece in usable]),
        velocity_mps=np.vstack([piece.velocity_mps for piece in usable]),
        orientation=np.vstack([piece.orientation for piece in usable]),
        acceleration_nav=np.vstack([piece.acceleration_nav for piece in usable]),
        specific_force_phone=np.vstack([piece.specific_force_phone for piece in usable]),
        flags=np.concatenate([piece.flags for piece in usable]),
        origin=usable[0].origin,
        heading_observed=all(piece.heading_observed for piece in usable),
        is_navigation_grade=all(piece.is_navigation_grade for piece in usable),
        counters=counters,
        initial=usable[0].initial,
        config_description=usable[0].config_description,
        notes=[f"concatenation of {len(usable)} segmented pieces; joins are discontinuous"],
    )
