"""Navigation state, per-sample quality flags, and the units they carry.

One state object, one set of units, declared once. Phase 3 established that
most orientation bugs are convention bugs; the equivalent hazard in a
mechanization is a unit or a frame silently changing meaning between two
functions that both look correct in isolation.

Units, fixed for the whole package:

===================  ==========  ===========================================
Quantity             Unit        Frame
===================  ==========  ===========================================
position             m           NAVIGATION (ENU), relative to a stated origin
velocity             m/s         NAVIGATION
acceleration         m/s²        NAVIGATION
specific force       m/s²        as tagged (PHONE, VEHICLE or NAVIGATION)
angular rate         rad/s       PHONE
orientation          quaternion  ``q_navigation_phone``, Hamilton, ``[x,y,z,w]``
time                 s           ``analysis_time_s`` from Phase 2
===================  ==========  ===========================================

Degrees appear only in diagnostics and reports. Nothing internal to this
package stores an angle in degrees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntFlag

import numpy as np

from idr.frames import quaternion as quat
from idr.frames.conventions import Frame
from idr.frames.vectors import Vector3


class NavigationFlag(IntFlag):
    """Conditions attached to one propagated navigation sample.

    Deliberately few. A flag exists here only where a downstream consumer
    would act differently because of it — the Phase 2 ``quality_flags`` bitmask
    already records the *data* conditions, and duplicating it would create two
    sources of truth for the same fact.
    """

    GOOD = 0
    #: A full alignment exists but its confidence is below the configured bar.
    ALIGNMENT_LOW_CONFIDENCE = 1
    #: No ``R_vehicle_phone``. Vehicle-frame quantities are unavailable.
    ALIGNMENT_UNAVAILABLE = 2
    #: Δt departs from the segment's median by more than the configured factor.
    IRREGULAR_DT = 4
    #: Δt ≤ 0 — a duplicate or non-monotonic timestamp Phase 2 retained.
    DUPLICATE_TIMESTAMP = 8
    #: Δt exceeds the maximum integration step; nothing was integrated across.
    LARGE_TIME_GAP = 16
    #: A sensor sample was NaN, infinite, or beyond a physical plausibility bound.
    INVALID_SENSOR = 32
    #: The stationarity detector considers the vehicle at rest here.
    STATIONARY = 64

    def describe(self) -> str:
        if self == NavigationFlag.GOOD:
            return "good"
        return ", ".join(
            (flag.name or str(flag)).lower().replace("_", " ")
            for flag in NavigationFlag
            if flag is not NavigationFlag.GOOD and flag & self
        )


#: Flags that mean the propagated state at this sample is not trustworthy as a
#: navigation solution, as opposed to merely annotated. Kept as one definition
#: so that "was this sample usable?" has a single answer everywhere.
UNTRUSTWORTHY_FLAGS = (
    NavigationFlag.ALIGNMENT_UNAVAILABLE
    | NavigationFlag.DUPLICATE_TIMESTAMP
    | NavigationFlag.LARGE_TIME_GAP
    | NavigationFlag.INVALID_SENSOR
)


@dataclass(frozen=True)
class NavigationState:
    """The propagated state at one instant.

    ``orientation`` is ``q_navigation_phone``: it takes phone-frame
    coordinates into the ENU navigation frame. It is stored as a quaternion and
    only ever a quaternion — Euler angles are produced on request for
    diagnostics and are never the state.

    ``position`` is metres in the local tangent plane whose origin the owning
    :class:`~idr.navigation.trajectory.Trajectory` records. A position without
    its origin is not a location, so the origin never travels separately.
    """

    analysis_time_s: float
    position_m: np.ndarray
    velocity_mps: np.ndarray
    orientation: np.ndarray
    #: Bias-corrected specific force in the phone frame, m/s². Retained because
    #: the whole chain downstream of it is a transformation of this vector, and
    #: a drift investigation that cannot see the input is guesswork.
    specific_force_phone: np.ndarray | None = None
    #: Linear acceleration in the navigation frame, m/s² — i.e. ``f_nav + g_nav``.
    acceleration_nav: np.ndarray | None = None
    flags: NavigationFlag = NavigationFlag.GOOD

    def __post_init__(self) -> None:
        for name in ("position_m", "velocity_mps", "orientation"):
            value = getattr(self, name)
            expected = 4 if name == "orientation" else 3
            if np.asarray(value).shape != (expected,):
                raise ValueError(
                    f"{name} must have shape ({expected},), got {np.asarray(value).shape}"
                )

    @property
    def speed_mps(self) -> float:
        return float(np.linalg.norm(self.velocity_mps))

    @property
    def position(self) -> Vector3:
        """The position as a frame-tagged vector, for interoperation with Phase 3."""
        return Vector3.from_array(self.position_m, Frame.NAVIGATION)

    @property
    def velocity(self) -> Vector3:
        return Vector3.from_array(self.velocity_mps, Frame.NAVIGATION)

    def euler_deg(self) -> tuple[float, float, float]:
        """(roll, pitch, yaw) in degrees. Diagnostic only — see the conventions doc."""
        roll, pitch, yaw = quat.to_euler(self.orientation)
        return (np.degrees(roll), np.degrees(pitch), np.degrees(yaw))

    def with_flags(self, flags: NavigationFlag) -> NavigationState:
        return NavigationState(
            analysis_time_s=self.analysis_time_s,
            position_m=self.position_m,
            velocity_mps=self.velocity_mps,
            orientation=self.orientation,
            specific_force_phone=self.specific_force_phone,
            acceleration_nav=self.acceleration_nav,
            flags=self.flags | flags,
        )


class InitialVelocitySource(str):
    """Marker strings for where an initial velocity came from.

    A string subclass rather than an enum because these are written verbatim
    into report tables and JSON artifacts, and the provenance of the initial
    velocity is one of the things a reader of those artifacts most needs.
    """


#: Initial velocity taken from the Phase 2 reference velocity, lag-corrected.
VELOCITY_FROM_REFERENCE = "reference_velocity_lag_corrected"
#: Initial velocity taken as zero because the window was detected stationary.
VELOCITY_ZERO_STATIONARY = "zero_stationary_detected"
#: Initial velocity supplied by the caller.
VELOCITY_SUPPLIED = "supplied_by_caller"


@dataclass
class InitialState:
    """Everything needed to start a mechanization run, with its provenance.

    Separated from :class:`NavigationState` because initialization carries
    facts that the propagated state does not: *where each number came from*,
    and whether heading was ever actually observed. A trajectory whose heading
    was never observed is a real trajectory in an unknown frame, and that is a
    materially different object from one anchored to magnetic north.
    """

    analysis_time_s: float
    position_m: np.ndarray
    velocity_mps: np.ndarray
    orientation: np.ndarray
    #: Whether the initial yaw is an actual measurement or an arbitrary origin.
    #: False is the common case in this dataset and is not an error.
    heading_observed: bool
    velocity_source: str
    position_source: str
    orientation_source: str
    notes: list[str] = field(default_factory=list)

    def to_state(self) -> NavigationState:
        return NavigationState(
            analysis_time_s=self.analysis_time_s,
            position_m=np.asarray(self.position_m, dtype="float64"),
            velocity_mps=np.asarray(self.velocity_mps, dtype="float64"),
            orientation=quat.normalize(self.orientation),
        )
