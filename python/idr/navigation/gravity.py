"""The gravity model the mechanization compensates with.

Separate from :mod:`idr.frames.gravity`, and the distinction matters. Phase 3's
module *estimates the direction of gravity in the phone frame from sensor
data*. This module *states what gravity is in the navigation frame* as a matter
of physics. One is a measurement; the other is a model. Conflating them is how
a gravity estimate ends up being compensated with itself.

The baseline model is a constant magnitude along navigation −Z. That is
deliberately crude: the point of Phase 4 is a correct mechanization, not an
accurate geoid, and the error a constant model introduces (see
:class:`WGS84Gravity` for the magnitude) is orders of magnitude below the
accelerometer bias that dominates this dataset. The interface exists so a
better model can be substituted without touching the mechanization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from idr.frames.conventions import STANDARD_GRAVITY_MPS2

#: WGS-84 normal gravity at the equator, m/s².
_SOMIGLIANA_GAMMA_E = 9.7803253359
#: Somigliana's dimensionless constant.
_SOMIGLIANA_K = 0.00193185265241
#: First eccentricity squared of the WGS-84 ellipsoid.
_WGS84_E2 = 0.00669437999013
#: Free-air gradient, s⁻². Used only for the altitude term.
_FREE_AIR_GRADIENT = 3.086e-6


@runtime_checkable
class GravityModel(Protocol):
    """A gravitational acceleration field, evaluated in the navigation frame.

    The returned vector is the **gravitational acceleration** ``g``, pointing
    down — approximately ``[0, 0, -9.81]`` in ENU. It is *not* the specific
    force a stationary accelerometer would report, which is ``-g`` and points
    up. Everything in this package that touches gravity is written against this
    sentence, and the sign regression test in the Phase 4 suite exists to keep
    it true.
    """

    def acceleration_nav(
        self, *, latitude_deg: float | None = None, altitude_m: float | None = None
    ) -> np.ndarray:
        """``g`` in ENU, m/s². Arguments are hints a model may ignore."""
        ...

    @property
    def description(self) -> str:
        """One line naming the model, for reports and artifacts."""
        ...


@dataclass(frozen=True)
class ConstantGravity:
    """A fixed magnitude along navigation −Z. The Phase 4 baseline.

    ``magnitude_mps2`` defaults to the same standard gravity Phase 2 used for
    its g→m/s² unit conversion and Phase 3 used for its magnitude checks. One
    value across all four phases means a vehicle acceleration expressed in g
    and a gravity compensation cannot disagree by a rounding choice.
    """

    magnitude_mps2: float = STANDARD_GRAVITY_MPS2

    def __post_init__(self) -> None:
        if not math.isfinite(self.magnitude_mps2) or self.magnitude_mps2 <= 0.0:
            raise ValueError(
                f"gravity magnitude must be finite and positive, got {self.magnitude_mps2!r}"
            )

    def acceleration_nav(
        self, *, latitude_deg: float | None = None, altitude_m: float | None = None
    ) -> np.ndarray:
        del latitude_deg, altitude_m  # A constant model has nothing to do with either.
        return np.array([0.0, 0.0, -self.magnitude_mps2], dtype="float64")

    @property
    def description(self) -> str:
        return f"constant {self.magnitude_mps2:.5f} m/s² along navigation −Z"


@dataclass(frozen=True)
class WGS84Gravity:
    """Somigliana normal gravity with a free-air altitude correction.

    Not used by the baseline. It exists to demonstrate that the interface is
    genuinely substitutable, and to put a number on what the constant model
    costs: normal gravity spans 9.780 m/s² at the equator to 9.832 m/s² at the
    poles, so choosing the wrong constant is worth up to ~0.026 m/s² — which
    integrates to roughly 47 m of vertical position error over 60 s. That is
    large in absolute terms and still small beside the accelerometer bias this
    dataset exhibits, which is the honest reason the baseline does not bother.

    Falls back to the equatorial value when no latitude is supplied, and says
    so through :attr:`description` rather than silently.
    """

    default_latitude_deg: float | None = None

    def acceleration_nav(
        self, *, latitude_deg: float | None = None, altitude_m: float | None = None
    ) -> np.ndarray:
        latitude = latitude_deg if latitude_deg is not None else self.default_latitude_deg
        if latitude is None or not math.isfinite(latitude):
            magnitude = _SOMIGLIANA_GAMMA_E
        else:
            sin2 = math.sin(math.radians(latitude)) ** 2
            magnitude = (
                _SOMIGLIANA_GAMMA_E
                * (1.0 + _SOMIGLIANA_K * sin2)
                / math.sqrt(1.0 - _WGS84_E2 * sin2)
            )
        if altitude_m is not None and math.isfinite(altitude_m):
            magnitude -= _FREE_AIR_GRADIENT * altitude_m
        return np.array([0.0, 0.0, -magnitude], dtype="float64")

    @property
    def description(self) -> str:
        if self.default_latitude_deg is None:
            return "WGS-84 Somigliana normal gravity (equatorial fallback without latitude)"
        return f"WGS-84 Somigliana normal gravity at {self.default_latitude_deg:.3f}°"


def linear_acceleration_nav(specific_force_nav: np.ndarray, gravity_nav: np.ndarray) -> np.ndarray:
    """``a = f + g``. The one equation this whole phase turns on.

    An accelerometer measures specific force ``f = a − g``, where ``g`` is the
    gravitational acceleration vector pointing down. Rearranging gives
    ``a = f + g``: the gravity vector is **added**.

    The failure this guards against is not subtle in its consequences and is
    very subtle on the page. Writing ``f − g`` instead leaves a stationary
    device reporting ``2g`` upward rather than zero, which integrates to about
    300 m of spurious climb in 5.5 s, and 1.2 km in 11 s — but the intermediate
    numbers all look
    like plausible accelerations, so nothing downstream complains.

    Phase 3 hit both halves of this: the sign of the gravity *direction* (a
    180° heading error) and the sign of this *sum* (a doubled vertical term).
    Both were caught by tests rather than by reading, which is why
    ``test_gravity_sign`` in the Phase 4 suite asserts the stationary case
    explicitly and would fail on a single character change here.
    """
    return np.asarray(specific_force_nav, dtype="float64") + np.asarray(
        gravity_nav, dtype="float64"
    )
