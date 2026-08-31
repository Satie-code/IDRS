"""Map IO-VNBD source columns onto the canonical schema.

Mapping is driven by the semantic roles Phase 1 established, with two
deliberate corrections made here after re-examining the source headers:

1. **Gyroscope axis naming is inconsistent.** 158 files label the three
   gyroscope columns ``Yaw/Pitch/Roll`` while 73 label the same three
   positions ``X/Y/Z``. Two files carrying the same recording under the two
   headers were verified to have byte-identical data rows, so these are the
   same channels under different names. Both are mapped to canonical
   ``gyro_x/y/z`` (device frame); the yaw/pitch/roll labels are recorded as
   unreliable aliases rather than trusted.

2. **Orientation yaw is also aliased** as ``ORIENTATION (Azimuth)``, which is
   the same quantity under the Android convention.

Anything not recognized is reported, never silently dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from idr.dataset.schema import SchemaReport
from idr.pipeline.canonical import G_TO_MPS2, KM_TO_M, KMH_TO_MPS

#: Canonical field <- (semantic role from Phase 1, conversion factor or None).
#: A factor of ``None`` means the source value is already in canonical units.
_ROLE_TO_CANONICAL: dict[str, tuple[str, float | None]] = {
    # Smartphone inertial
    "accel_x": ("accel_x", None),
    "accel_y": ("accel_y", None),
    "accel_z": ("accel_z", None),
    "gravity_x": ("gravity_x", None),
    "gravity_y": ("gravity_y", None),
    "gravity_z": ("gravity_z", None),
    "mag_x": ("mag_x", None),
    "mag_y": ("mag_y", None),
    "mag_z": ("mag_z", None),
    "orientation_yaw": ("orientation_yaw", None),
    "orientation_pitch": ("orientation_pitch", None),
    "orientation_roll": ("orientation_roll", None),
    # GNSS, shared by both streams
    "gnss_latitude": ("latitude", None),
    "gnss_longitude": ("longitude", None),
    "gnss_accuracy": ("gnss_accuracy_m", None),
    "gnss_heading": ("gnss_heading_deg", None),
    "gnss_speed": ("gnss_speed_mps", KMH_TO_MPS),
    # Vehicle reference
    "velocity": ("ref_speed_mps", KMH_TO_MPS),
    "vehicle_speed": ("ref_indicated_speed_mps", KMH_TO_MPS),
    "heading": ("ref_heading_deg", None),
    "vertical_velocity": ("ref_vertical_speed_mps", KMH_TO_MPS),
    "yaw_rate": ("ref_yaw_rate_dps", None),
    "accel_longitudinal": ("ref_accel_long_mps2", G_TO_MPS2),
    "accel_lateral": ("ref_accel_lat_mps2", G_TO_MPS2),
    "steering_angle": ("ref_steering_angle_deg", None),
    "wheel_speed_fl": ("ref_wheel_speed_fl", None),
    "wheel_speed_fr": ("ref_wheel_speed_fr", None),
    "wheel_speed_rl": ("ref_wheel_speed_rl", None),
    "wheel_speed_rr": ("ref_wheel_speed_rr", None),
    "engine_speed": ("ref_engine_speed_rpm", None),
    "gear": ("ref_gear", None),
    "gear_requested": ("ref_gear_requested", None),
    "brake_pressure": ("ref_brake_pressure_psi", None),
    "brake_position": ("ref_brake_position", None),
    "handbrake": ("ref_handbrake", None),
    "clutch_position": ("ref_clutch_position", None),
    "accelerator_pedal": ("ref_accelerator_pedal", None),
    "battery_voltage": ("ref_battery_voltage_v", None),
    "coolant_temperature": ("ref_coolant_temp_c", None),
    "air_temperature": ("ref_air_temp_c", None),
    "sample_period_s": ("ref_sample_period_s", None),
}

# Gyroscope and orientation-yaw columns that Phase 1's role table did not
# recognize, matched on the raw header text. Keyed by the normalized token.
_ALIAS_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^GYROSCOPE\s*(X|YAW)\b"), "gyro_x"),
    (re.compile(r"^GYROSCOPE\s*(Y|PITCH)\b"), "gyro_y"),
    (re.compile(r"^GYROSCOPE\s*(Z|ROLL)\b"), "gyro_z"),
    (re.compile(r"^ORIENTATION\s*\(\s*(YAW|AZIMUTH)"), "orientation_yaw"),
)

#: Roles handled by dedicated logic rather than a direct column copy.
_SPECIAL_ROLES = frozenset(
    {
        "time_since_start_ms",
        "time_of_day_s",
        "datetime_string",
        "gnss_satellites",
        "gnss_altitude",
        "altitude",
    }
)


@dataclass(frozen=True)
class ColumnMapping:
    """Resolved mapping for one source file."""

    #: canonical field -> (source column name, conversion factor or None)
    fields: dict[str, tuple[str, float | None]]
    #: Source columns that matched no canonical field.
    unmapped: list[str]
    #: Canonical fields resolved via an alias rather than a Phase 1 role.
    alias_resolved: dict[str, str]
    time_column: str | None
    time_role: str | None
    date_column: str | None
    satellites_column: str | None
    altitude_column: str | None
    altitude_factor: float


def _normalize(raw: str) -> str:
    return re.sub(r"\s+", " ", raw.strip()).upper()


def build_mapping(schema: SchemaReport, stream: str) -> ColumnMapping:
    """Resolve one file's header into a canonical column mapping."""
    fields: dict[str, tuple[str, float | None]] = {}
    alias_resolved: dict[str, str] = {}
    unmapped: list[str] = []
    time_column = time_role = date_column = satellites_column = None
    altitude_column = None
    altitude_factor = 1.0

    for column in schema.columns:
        role = column.semantic_role
        raw = column.raw_name
        normalized = _normalize(raw)

        if role == "time_since_start_ms":
            time_column, time_role = raw, role
            continue
        if role == "time_of_day_s":
            time_column, time_role = raw, role
            continue
        if role == "datetime_string":
            date_column = raw
            continue
        if role == "gnss_satellites":
            satellites_column = raw
            continue
        if role in ("gnss_altitude", "altitude"):
            altitude_column = raw
            # Vehicle files report "Height (km)"; smartphone files report metres.
            altitude_factor = KM_TO_M if "(KM)" in normalized else 1.0
            continue

        target = _ROLE_TO_CANONICAL.get(role)
        if target is not None:
            fields[target[0]] = (raw, target[1])
            continue

        # Role unknown to Phase 1: try the alias patterns before giving up.
        matched = False
        for pattern, canonical_name in _ALIAS_PATTERNS:
            if pattern.match(normalized):
                fields[canonical_name] = (raw, None)
                alias_resolved[canonical_name] = raw.strip()
                matched = True
                break
        if not matched and normalized:
            unmapped.append(raw.strip())

    # The vehicle stream's satellite column is a plain count, not "a / b".
    return ColumnMapping(
        fields=fields,
        unmapped=unmapped,
        alias_resolved=alias_resolved,
        time_column=time_column,
        time_role=time_role,
        date_column=date_column,
        satellites_column=satellites_column,
        altitude_column=altitude_column,
        altitude_factor=altitude_factor,
    )


_SATELLITE_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")


def parse_satellites(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Parse the smartphone satellite field into (in_use, in_view).

    The field is written ``"27 / 28"``. The two numbers are interpreted as
    satellites-in-use and satellites-in-view respectively — an interpretation
    supported by the left value never exceeding the right, but **not stated by
    any dataset documentation**. The raw string is always preserved alongside
    these derived values so the interpretation can be revisited.

    A plain integer (the vehicle stream's satellite count) parses as in_use
    with in_view left missing. Anything unparseable yields NaN for both rather
    than raising.
    """
    text = series.astype("string").fillna("")
    in_use = np.full(len(text), np.nan, dtype="float64")
    in_view = np.full(len(text), np.nan, dtype="float64")

    extracted = text.str.extract(_SATELLITE_RE)
    paired = extracted[0].notna()
    if paired.any():
        in_use[paired.to_numpy()] = extracted.loc[paired, 0].astype("float64").to_numpy()
        in_view[paired.to_numpy()] = extracted.loc[paired, 1].astype("float64").to_numpy()

    # Fall back to a bare number for rows that were not of the "a / b" form.
    remaining = ~paired
    if remaining.any():
        plain = pd.to_numeric(text[remaining], errors="coerce").to_numpy(
            dtype="float64", na_value=np.nan
        )
        in_use[remaining.to_numpy()] = plain

    return in_use, in_view


def satellites_interpretation_is_consistent(in_use: np.ndarray, in_view: np.ndarray) -> bool:
    """True if in_use <= in_view wherever both are present.

    Used to report evidence for (or against) the 'a / b' reading rather than
    asserting it.
    """
    both = np.isfinite(in_use) & np.isfinite(in_view)
    if not both.any():
        return True
    return bool(np.all(in_use[both] <= in_view[both]))
