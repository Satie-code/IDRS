"""Latitude/longitude → local ENU metres, with the origin stated.

Phase 3 explicitly left this to Phase 4: "Phase 3 never converts latitude and
longitude into a navigation-frame position — that is Phase 4's job, and it will
need to state the origin and the projection it uses."

So: the projection is a **local tangent plane** anchored at a stated origin,
using the WGS-84 meridional and transverse radii of curvature evaluated at that
origin. East and north are exact to first order in the displacement, and the
second-order error over a 10 km excursion is under a metre — far below anything
else in this comparison. It is not a general map projection and must not be
used as one; over hundreds of kilometres the fixed radii stop being right.

This module exists to make a reference track comparable to a propagated one. It
never touches the propagated trajectory itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: WGS-84 semi-major axis, m.
WGS84_SEMI_MAJOR_M = 6378137.0
#: WGS-84 first eccentricity squared.
WGS84_E2 = 0.00669437999014


@dataclass(frozen=True)
class TangentPlane:
    """A local ENU frame anchored at one geodetic point.

    Instances are frozen and carry their own radii so that a position in metres
    can always be turned back into a coordinate. A trajectory that has lost its
    origin has lost its meaning, which is why
    :class:`~idr.navigation.trajectory.Trajectory` holds one rather than
    assuming the caller kept it.
    """

    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 0.0

    @property
    def meridional_radius_m(self) -> float:
        """Radius of curvature in the meridian (north–south) at this latitude."""
        sin2 = math.sin(math.radians(self.latitude_deg)) ** 2
        return WGS84_SEMI_MAJOR_M * (1.0 - WGS84_E2) / (1.0 - WGS84_E2 * sin2) ** 1.5

    @property
    def transverse_radius_m(self) -> float:
        """Radius of curvature in the prime vertical (east–west) at this latitude."""
        sin2 = math.sin(math.radians(self.latitude_deg)) ** 2
        return WGS84_SEMI_MAJOR_M / math.sqrt(1.0 - WGS84_E2 * sin2)

    def to_enu(
        self,
        latitude_deg: np.ndarray | float,
        longitude_deg: np.ndarray | float,
        altitude_m: np.ndarray | float | None = None,
    ) -> np.ndarray:
        """Geodetic coordinates → ENU metres relative to this origin.

        Returns an ``(n, 3)`` array. Non-finite inputs stay non-finite rather
        than being interpolated away: a missing fix is missing, and filling it
        would invent a position the receiver never reported.
        """
        lat = np.atleast_1d(np.asarray(latitude_deg, dtype="float64"))
        lon = np.atleast_1d(np.asarray(longitude_deg, dtype="float64"))
        if lat.shape != lon.shape:
            raise ValueError(f"latitude {lat.shape} and longitude {lon.shape} must match")

        d_lat = np.radians(lat - self.latitude_deg)
        d_lon = np.radians(lon - self.longitude_deg)
        cos_lat = math.cos(math.radians(self.latitude_deg))

        north = d_lat * (self.meridional_radius_m + self.altitude_m)
        east = d_lon * (self.transverse_radius_m + self.altitude_m) * cos_lat
        # A fix is a pair. If either half is missing the position is missing,
        # and letting the valid half through would give a row that looks
        # partially usable — a zero east offset reads as "at the origin"
        # rather than as "unknown".
        incomplete = ~(np.isfinite(lat) & np.isfinite(lon))
        north = np.where(incomplete, np.nan, north)
        east = np.where(incomplete, np.nan, east)
        if altitude_m is None:
            up = np.zeros_like(north)
        else:
            alt = np.atleast_1d(np.asarray(altitude_m, dtype="float64"))
            up = alt - self.altitude_m
        return np.column_stack([east, north, up])

    def describe(self) -> str:
        return (
            f"local tangent plane at {self.latitude_deg:.6f}°, "
            f"{self.longitude_deg:.6f}°, {self.altitude_m:.1f} m "
            "(WGS-84 radii of curvature at the origin)"
        )


def origin_from_fixes(
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    altitude_m: np.ndarray | None = None,
    valid: np.ndarray | None = None,
) -> TangentPlane | None:
    """Anchor a tangent plane at the **first valid** fix, not the mean.

    The first fix rather than the centroid because the origin has to coincide
    with where the trajectory starts: a propagated position begins at zero, and
    a reference whose origin is the centroid of the drive begins hundreds of
    metres away, producing a constant offset that looks exactly like an
    initialization error.
    """
    lat = np.asarray(latitude_deg, dtype="float64")
    lon = np.asarray(longitude_deg, dtype="float64")
    usable = np.isfinite(lat) & np.isfinite(lon)
    if valid is not None:
        usable &= np.asarray(valid, dtype=bool)
    # (0, 0) is in the Gulf of Guinea and is what a receiver writes when it has
    # no fix; treating it as an origin would put the whole track in the Atlantic.
    usable &= ~((np.abs(lat) < 1e-9) & (np.abs(lon) < 1e-9))
    if not usable.any():
        return None
    index = int(np.argmax(usable))
    altitude = 0.0
    if altitude_m is not None:
        candidate = float(np.asarray(altitude_m, dtype="float64")[index])
        if math.isfinite(candidate):
            altitude = candidate
    return TangentPlane(float(lat[index]), float(lon[index]), altitude)
