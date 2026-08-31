"""Canonical schema definitions for IO-VNBD derived data (Phase 2).

This module is the single source of truth for what a canonical record *is*:
field names, dtypes, units, and the quality vocabulary. Everything downstream
(readers, segmentation, splits, sequence generation, and the Phase 3 contract)
refers to these definitions rather than restating column names.

Design rules enforced here:

- Raw data is never represented as "clean". Every canonical frame carries
  quality flags and explicit availability metadata alongside the values.
- A structurally absent channel (a reduced 18-column file has no magnetometer)
  is NaN with ``has_magnetometer = False`` — never zero, never fabricated.
- Units are normalized and documented; the source unit is recorded so a
  conversion can always be re-derived.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag

import numpy as np

#: Bumped whenever the canonical layout changes in a way that invalidates
#: previously written Parquet files.
CANONICAL_SCHEMA_VERSION = "2.0.0"


class QualityFlag(IntFlag):
    """Per-row quality vocabulary, stored as a single integer bitmask.

    A bitmask keeps the canonical frame narrow (one column instead of a dozen
    booleans) while remaining losslessly decomposable — see :func:`describe_flags`.
    The set is deliberately small and closed; new flags require updating the
    Phase 2 contract document.
    """

    VALID = 0
    #: One or more sensor channels are absent for this row.
    MISSING_SENSOR = 1 << 0
    #: The row's timestamp could not be parsed or is not finite.
    INVALID_TIMESTAMP = 1 << 1
    #: This timestamp value occurs more than once in the stream.
    DUPLICATE_TIMESTAMP = 1 << 2
    #: This row's timestamp is not greater than its predecessor's.
    NON_MONOTONIC_TIMESTAMP = 1 << 3
    #: The source file required a non-UTF-8 decode or contained mojibake.
    ENCODING_WARNING = 1 << 4
    #: The row came from a file whose columns were repaired (see repair.py).
    SCHEMA_REPAIR = 1 << 5
    #: The GNSS position is a repeat of the previous one beyond the freshness
    #: threshold — a held position, NOT a loss of fix.
    GNSS_STALE = 1 << 6
    #: The owning pair has no reliable smartphone/vehicle time alignment.
    SYNC_UNRESOLVED = 1 << 7
    #: The reference label for this row is low-confidence or derived.
    REFERENCE_LOW_CONFIDENCE = 1 << 8
    #: The row is the first sample of a new logical segment.
    SEGMENT_BOUNDARY = 1 << 9


def describe_flags(mask: int) -> list[str]:
    """Decompose a bitmask into its flag names, for human-readable reports."""
    if mask == 0:
        return ["VALID"]
    return [
        flag.name
        for flag in QualityFlag
        if flag.name is not None and flag.value and mask & flag.value
    ]


@dataclass(frozen=True)
class FieldSpec:
    """One canonical field: its type, unit, and provenance."""

    name: str
    dtype: str
    unit: str | None
    description: str
    #: Unit as it appears in the IO-VNBD source header, when different.
    source_unit: str | None = None


# ---------------------------------------------------------------------------
# Identity fields — present on every canonical frame.
# ---------------------------------------------------------------------------
IDENTITY_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("dataset_family", "string", None, "Source tree the file came from"),
    FieldSpec("source_file", "string", None, "Path relative to the raw dataset root"),
    FieldSpec("file_hash", "string", None, "SHA-256 of the whole source file"),
    FieldSpec(
        "payload_hash",
        "string",
        None,
        "SHA-256 of the file's data rows excluding the header line; groups "
        "recordings that are identical apart from header labels",
    ),
    FieldSpec("session_id", "string", None, "Session identifier parsed from the filename"),
    FieldSpec("segment_id", "string", None, "Logical trip within the file (see timeline.py)"),
    FieldSpec("stream", "string", None, "'smartphone' or 'vehicle'"),
)

# ---------------------------------------------------------------------------
# Time fields.
#
# IO-VNBD gives smartphone files an elapsed-ms clock plus a wall-clock DATE
# string, and vehicle files a seconds-since-midnight clock. Those are different
# semantics, so both are represented explicitly and neither is silently coerced
# into the other.
# ---------------------------------------------------------------------------
TIME_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        "elapsed_s",
        "float64",
        "s",
        "Seconds from the first sample of the segment, derived from "
        "analysis_time_s. The time axis sequence generation uses.",
    ),
    FieldSpec(
        "analysis_time_s",
        "float64",
        "s",
        "The continuous clock chosen for this stream: wall-clock seconds from "
        "the smartphone DATE column where present (it does not reset), else the "
        "elapsed counter; seconds-since-midnight for vehicle streams. "
        "Segmentation and synchronization operate on this axis.",
    ),
    FieldSpec(
        "analysis_time_source",
        "string",
        None,
        "Which clock analysis_time_s came from: 'wall_clock', 'elapsed', or 'time_of_day'.",
    ),
    FieldSpec(
        "source_time_s",
        "float64",
        "s",
        "The source clock in seconds, unmodified: elapsed-since-recording-start "
        "for smartphone files, seconds-since-local-midnight for vehicle files.",
        source_unit="ms (smartphone) / s (vehicle)",
    ),
    FieldSpec(
        "time_of_day_s",
        "float64",
        "s",
        "Seconds since local midnight — the only axis on which a smartphone and "
        "a vehicle stream can be compared. NaN where unavailable.",
    ),
    FieldSpec(
        "timestamp_utc",
        "datetime64[ns]",
        None,
        "Wall-clock timestamp from the smartphone DATE column. NaT for vehicle "
        "streams, which carry no date. NOT timezone-resolved — see the contract.",
    ),
)

# ---------------------------------------------------------------------------
# Smartphone sensor fields.
#
# Gyroscope axes are mapped by COLUMN POSITION, not by the source label: the
# dataset uses two mutually inconsistent namings ("Yaw/Pitch/Roll" and "X/Y/Z")
# for the same three columns, and two files carrying the same recording under
# the two headers were verified byte-identical in their data rows. The X/Y/Z
# device-frame reading is therefore used, and the yaw/pitch/roll labels are
# treated as unreliable aliases.
# ---------------------------------------------------------------------------
SMARTPHONE_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("accel_x", "float32", "m/s^2", "Accelerometer, device X axis"),
    FieldSpec("accel_y", "float32", "m/s^2", "Accelerometer, device Y axis"),
    FieldSpec("accel_z", "float32", "m/s^2", "Accelerometer, device Z axis"),
    FieldSpec("gyro_x", "float32", "rad/s", "Gyroscope, device X axis (source may label 'Yaw')"),
    FieldSpec("gyro_y", "float32", "rad/s", "Gyroscope, device Y axis (source may label 'Pitch')"),
    FieldSpec("gyro_z", "float32", "rad/s", "Gyroscope, device Z axis (source may label 'Roll')"),
    FieldSpec("mag_x", "float32", "uT", "Magnetometer, device X axis"),
    FieldSpec("mag_y", "float32", "uT", "Magnetometer, device Y axis"),
    FieldSpec("mag_z", "float32", "uT", "Magnetometer, device Z axis"),
    FieldSpec("gravity_x", "float32", "m/s^2", "Gravity vector, device X axis"),
    FieldSpec("gravity_y", "float32", "m/s^2", "Gravity vector, device Y axis"),
    FieldSpec("gravity_z", "float32", "m/s^2", "Gravity vector, device Z axis"),
    FieldSpec(
        "orientation_yaw",
        "float32",
        "deg",
        "Device orientation yaw/azimuth (source labels 'Yaw' or 'Azimuth')",
    ),
    FieldSpec("orientation_pitch", "float32", "deg", "Device orientation pitch"),
    FieldSpec("orientation_roll", "float32", "deg", "Device orientation roll"),
)

# ---------------------------------------------------------------------------
# GNSS fields (smartphone and vehicle both carry a position).
# ---------------------------------------------------------------------------
GNSS_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("latitude", "float64", "deg", "WGS-84 latitude (float64: 1e-7 deg matters)"),
    FieldSpec("longitude", "float64", "deg", "WGS-84 longitude"),
    FieldSpec("altitude_m", "float32", "m", "Altitude above sea level", source_unit="m / km"),
    FieldSpec("gnss_speed_mps", "float32", "m/s", "GNSS speed over ground", source_unit="km/h"),
    FieldSpec("gnss_accuracy_m", "float32", "m", "Reported horizontal accuracy"),
    FieldSpec("gnss_heading_deg", "float32", "deg", "GNSS course over ground"),
    FieldSpec(
        "gnss_satellites_raw",
        "string",
        None,
        "Satellite field exactly as written in the source (e.g. '27 / 28'). "
        "Preserved verbatim; see satellites parsing for the derived integers.",
    ),
    FieldSpec("gnss_satellites_in_use", "float32", None, "Parsed left value of 'a / b'"),
    FieldSpec("gnss_satellites_in_view", "float32", None, "Parsed right value of 'a / b'"),
    FieldSpec("gnss_fix_available", "boolean", None, "Position present, finite, and not (0,0)"),
    FieldSpec("gnss_position_fresh", "boolean", None, "Position differs from the previous sample"),
    FieldSpec(
        "gnss_stale_duration_s",
        "float32",
        "s",
        "Seconds since the position last changed. 0 on a fresh sample.",
    ),
)

# ---------------------------------------------------------------------------
# Vehicle / reference fields.
# ---------------------------------------------------------------------------
VEHICLE_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("ref_speed_mps", "float32", "m/s", "GNSS-derived vehicle velocity", "km/h"),
    FieldSpec("ref_indicated_speed_mps", "float32", "m/s", "Vehicle indicated speed", "km/h"),
    FieldSpec("ref_heading_deg", "float32", "deg", "Vehicle heading"),
    FieldSpec("ref_vertical_speed_mps", "float32", "m/s", "Vertical velocity", "km/h"),
    FieldSpec("ref_yaw_rate_dps", "float32", "deg/s", "Vehicle yaw rate"),
    FieldSpec("ref_accel_long_mps2", "float32", "m/s^2", "Longitudinal acceleration", "g"),
    FieldSpec("ref_accel_lat_mps2", "float32", "m/s^2", "Lateral acceleration", "g"),
    FieldSpec("ref_steering_angle_deg", "float32", "deg", "Steering wheel angle"),
    FieldSpec("ref_wheel_speed_fl", "float32", "rad/s", "Wheel speed front left"),
    FieldSpec("ref_wheel_speed_fr", "float32", "rad/s", "Wheel speed front right"),
    FieldSpec("ref_wheel_speed_rl", "float32", "rad/s", "Wheel speed rear left"),
    FieldSpec("ref_wheel_speed_rr", "float32", "rad/s", "Wheel speed rear right"),
    FieldSpec("ref_engine_speed_rpm", "float32", "rev/min", "Engine speed"),
    FieldSpec("ref_gear", "float32", None, "Engaged gear"),
    FieldSpec("ref_gear_requested", "float32", None, "Requested gear"),
    FieldSpec("ref_brake_pressure_psi", "float32", "psi", "Brake pressure"),
    FieldSpec("ref_brake_position", "float32", None, "Brake applied (0/1)"),
    FieldSpec("ref_handbrake", "float32", None, "Handbrake engaged (0/1)"),
    FieldSpec("ref_clutch_position", "float32", None, "Clutch engaged (0/1)"),
    FieldSpec("ref_accelerator_pedal", "float32", None, "Accelerator pedal position"),
    FieldSpec("ref_battery_voltage_v", "float32", "V", "Battery voltage"),
    FieldSpec("ref_coolant_temp_c", "float32", "degC", "Coolant temperature"),
    FieldSpec("ref_air_temp_c", "float32", "degC", "Air temperature"),
    FieldSpec("ref_sample_period_s", "float32", "s", "Sample period reported by the logger"),
)

QUALITY_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("quality_flags", "int32", None, "Bitmask of QualityFlag values"),
)

SMARTPHONE_SCHEMA: tuple[FieldSpec, ...] = (
    IDENTITY_FIELDS + TIME_FIELDS + SMARTPHONE_FIELDS + GNSS_FIELDS + QUALITY_FIELDS
)
VEHICLE_SCHEMA: tuple[FieldSpec, ...] = (
    IDENTITY_FIELDS + TIME_FIELDS + GNSS_FIELDS + VEHICLE_FIELDS + QUALITY_FIELDS
)

#: Availability flags recorded per source file, so a structurally absent
#: channel is never confused with a channel that happens to be NaN.
FEATURE_AVAILABILITY_KEYS: tuple[str, ...] = (
    "has_accelerometer",
    "has_gyroscope",
    "has_magnetometer",
    "has_gravity",
    "has_orientation",
    "has_gnss",
    "has_gnss_accuracy",
)

#: Unit conversions applied at canonicalization, kept here so the report and
#: the contract document can state them without duplicating magic numbers.
KMH_TO_MPS = 1.0 / 3.6
G_TO_MPS2 = 9.80665
KM_TO_M = 1000.0


def schema_for(stream: str) -> tuple[FieldSpec, ...]:
    if stream == "smartphone":
        return SMARTPHONE_SCHEMA
    if stream == "vehicle":
        return VEHICLE_SCHEMA
    raise ValueError(f"Unknown stream {stream!r}; expected 'smartphone' or 'vehicle'")


def pandas_dtypes(stream: str) -> dict[str, str]:
    """Canonical field -> pandas dtype, for constructing/validating frames."""
    return {spec.name: spec.dtype for spec in schema_for(stream)}


def empty_column(spec: FieldSpec, length: int) -> np.ndarray:
    """An all-missing column of the right dtype, for absent channels.

    Missing means NaN (or NaT / pd.NA), never 0 — a zero would be
    indistinguishable from a real stationary reading.
    """
    if spec.dtype.startswith("float"):
        return np.full(length, np.nan, dtype=spec.dtype)
    if spec.dtype == "int32":
        return np.zeros(length, dtype="int32")
    return np.full(length, None, dtype=object)
