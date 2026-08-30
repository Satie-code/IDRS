"""CSV schema extraction: encoding, column names, units, and semantic roles.

Header inspection is deliberately cheap (it reads only the first few KB), so
the fast inspection mode can classify the whole dataset without touching row
data. Semantic role assignment maps the dataset's own column names onto the
signal vocabulary later phases care about; anything not confidently
recognized is reported as ``unknown`` rather than force-fitted.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from idr.dataset.evidence import Evidence

#: Tried in order; ``latin-1`` cannot fail, so detection always terminates.
CANDIDATE_ENCODINGS = ("utf-8", "cp1252", "latin-1")

_HEADER_PROBE_BYTES = 16384
_UNIT_RE = re.compile(r"\(([^)]*)\)\s*$")

# Column-name substrings -> semantic role. Matched against the upper-cased,
# unit-stripped column name. Order matters: the first match wins, so more
# specific patterns are listed before generic ones.
_ROLE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("GPS LATITUDE", "gnss_latitude"),
    ("GPS LONGITUDE", "gnss_longitude"),
    ("GPS ALTITUDE", "gnss_altitude"),
    ("GPS SPEED", "gnss_speed"),
    ("GPS ACCURACY", "gnss_accuracy"),
    ("GPS ORIENTATION", "gnss_heading"),
    ("SATELLITES IN RANGE", "gnss_satellites"),
    ("NO OF GPS SATELLITES", "gnss_satellites"),
    ("TIME SINCE START OF DAY", "time_of_day_s"),
    ("TIME SINCE START", "time_since_start_ms"),
    ("SAMPLE PERIOD", "sample_period_s"),
    ("DATE", "datetime_string"),
    ("ACCELEROMETER X", "accel_x"),
    ("ACCELEROMETER Y", "accel_y"),
    ("ACCELEROMETER Z", "accel_z"),
    ("GRAVITY X", "gravity_x"),
    ("GRAVITY Y", "gravity_y"),
    ("GRAVITY Z", "gravity_z"),
    ("GYROSCOPE YAW", "gyro_yaw"),
    ("GYROSCOPE PITCH", "gyro_pitch"),
    ("GYROSCOPE ROLL", "gyro_roll"),
    ("MAGNETIC FIELD X", "mag_x"),
    ("MAGNETIC FIELD Y", "mag_y"),
    ("MAGNETIC FIELD Z", "mag_z"),
    ("ORIENTATION (YAW", "orientation_yaw"),
    ("ORIENTATION (PITCH", "orientation_pitch"),
    ("ORIENTATION (ROLL", "orientation_roll"),
    ("LATITUDE", "gnss_latitude"),
    ("LONGITUDE", "gnss_longitude"),
    ("VERTICAL VELOCITY", "vertical_velocity"),
    ("VELOCITY", "velocity"),
    ("HEADING", "heading"),
    ("HEIGHT", "altitude"),
    ("STEERING ANGLE", "steering_angle"),
    ("WHEEL SPEED FRONT LEFT", "wheel_speed_fl"),
    ("WHEEL SPEED FRONT RIGHT", "wheel_speed_fr"),
    ("WHEEL SPEED REAR LEFT", "wheel_speed_rl"),
    ("WHEEL SPEED REAR RIGHT", "wheel_speed_rr"),
    ("YAW RATE", "yaw_rate"),
    ("INDICATED VEHICLE SPEED", "vehicle_speed"),
    ("INDICATED LONGITUDINAL ACCELERATION", "accel_longitudinal"),
    ("INDICATED LATERAL ACCELERATION", "accel_lateral"),
    ("HANDBRAKE", "handbrake"),
    ("GEAR REQUESTED", "gear_requested"),
    ("GEAR", "gear"),
    ("ENGINE SPEED", "engine_speed"),
    ("COOLANT TEMPERATURE", "coolant_temperature"),
    ("CLUTCH POSITION", "clutch_position"),
    ("BRAKE PRESSURE", "brake_pressure"),
    ("BRAKE POSITION", "brake_position"),
    ("BATTERY VOLTAGE", "battery_voltage"),
    ("AIR TEMPERATURE", "air_temperature"),
    ("ACCELERATOR PEDAL POSITION", "accelerator_pedal"),
)


@dataclass(frozen=True)
class ColumnSchema:
    """One column as it actually appears in the file."""

    index: int
    raw_name: str
    normalized_name: str
    unit: str | None
    semantic_role: str
    role_evidence: str


@dataclass
class SchemaReport:
    """The header structure of one CSV file."""

    relative_path: str
    encoding: str
    encoding_is_utf8: bool
    encoding_note: str
    column_count: int
    columns: list[ColumnSchema] = field(default_factory=list)

    @property
    def schema_id(self) -> str:
        """Stable id for a distinct column layout, so files can be grouped."""
        joined = "|".join(column.normalized_name for column in self.columns)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]

    @property
    def roles(self) -> set[str]:
        return {column.semantic_role for column in self.columns}


def detect_encoding(path: Path) -> tuple[str, bool, str]:
    """Return ``(encoding, is_valid_utf8, note)`` for a CSV file.

    IO-VNBD smartphone files mix raw cp1252 bytes (``m/s²``) with
    double-encoded UTF-8 sequences (``Â°``) in the same header, so strict
    UTF-8 decoding fails and a note records the inconsistency rather than
    silently falling back.
    """
    head = path.read_bytes()[:_HEADER_PROBE_BYTES]
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as exc:
        note = f"not valid UTF-8 (byte {exc.start}); decoded with cp1252 fallback"
        return "cp1252", False, note

    text = head.decode("utf-8")
    if "Â" in text or "Î¼" in text:
        return (
            "utf-8",
            True,
            "valid UTF-8 but contains mojibake (e.g. 'Â°'), i.e. double-encoded source",
        )
    return "utf-8", True, "valid UTF-8"


def normalize_column_name(raw: str) -> str:
    """Strip whitespace and the trailing unit parenthetical from a column name."""
    stripped = raw.strip()
    without_unit = _UNIT_RE.sub("", stripped).strip()
    return without_unit or stripped


def extract_unit(raw: str) -> str | None:
    match = _UNIT_RE.search(raw.strip())
    return match.group(1).strip() if match else None


def assign_role(raw_name: str) -> tuple[str, str]:
    """Map a column name to a semantic role, or ``unknown`` if unrecognized."""
    haystack = raw_name.strip().upper()
    for needle, role in _ROLE_PATTERNS:
        if needle in haystack:
            return role, Evidence.VERIFIED_FROM_FILE
    return "unknown", Evidence.UNVERIFIED


def read_schema(path: Path, relative_path: str | None = None) -> SchemaReport:
    """Read only the header row of ``path`` and describe its columns."""
    encoding, is_utf8, note = detect_encoding(path)
    with path.open("r", encoding=encoding, newline="") as handle:
        header_line = handle.readline()
    if not header_line.strip():
        raise ValueError(f"{path} has an empty header row")

    raw_names = next(csv.reader(io.StringIO(header_line)))
    columns: list[ColumnSchema] = []
    for index, raw in enumerate(raw_names):
        role, role_evidence = assign_role(raw)
        columns.append(
            ColumnSchema(
                index=index,
                raw_name=raw,
                normalized_name=normalize_column_name(raw),
                unit=extract_unit(raw),
                semantic_role=role,
                role_evidence=role_evidence,
            )
        )

    return SchemaReport(
        relative_path=relative_path or path.name,
        encoding=encoding,
        encoding_is_utf8=is_utf8,
        encoding_note=note,
        column_count=len(columns),
        columns=columns,
    )


def schema_to_dict(report: SchemaReport) -> dict[str, object]:
    data = asdict(report)
    data["schema_id"] = report.schema_id
    return data
