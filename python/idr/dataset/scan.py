"""Single-pass, memory-bounded scan of one CSV stream.

Every per-file statistic Phase 1 reports is produced here in **one** chunked
read: row counts, per-column missingness and numeric ranges, timestamp deltas,
and GNSS availability. Downstream modules (timestamps, quality, outages,
synchronization) interpret this result rather than re-reading the file, which
keeps a full deep scan of the dataset to a single pass over each file.

Memory is bounded by ``chunk_size`` rows plus one float per row for the
timestamp series, which is what the interval and gap analyses need.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from idr.dataset.evidence import StatKind
from idr.dataset.schema import SchemaReport, read_schema

DEFAULT_CHUNK_SIZE = 50_000

#: Roles whose numeric distribution is worth recording for sanity checks.
_NUMERIC_ROLES_OF_INTEREST = (
    "gnss_latitude",
    "gnss_longitude",
    "gnss_altitude",
    "gnss_speed",
    "gnss_accuracy",
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_yaw",
    "gyro_pitch",
    "gyro_roll",
    "mag_x",
    "mag_y",
    "mag_z",
    "velocity",
    "vehicle_speed",
    "heading",
    "altitude",
    "yaw_rate",
    "steering_angle",
    "wheel_speed_fl",
    "wheel_speed_fr",
    "wheel_speed_rl",
    "wheel_speed_rr",
    "accel_longitudinal",
    "accel_lateral",
    "engine_speed",
)


@dataclass
class ColumnStats:
    """Missingness and numeric range for one column, accumulated over chunks."""

    raw_name: str
    normalized_name: str
    semantic_role: str
    total_rows: int = 0
    missing_rows: int = 0
    non_numeric_rows: int = 0
    numeric_count: int = 0
    minimum: float | None = None
    maximum: float | None = None
    _sum: float = 0.0
    _sum_sq: float = 0.0

    @property
    def missing_percentage(self) -> float:
        return 100.0 * self.missing_rows / self.total_rows if self.total_rows else 0.0

    @property
    def mean(self) -> float | None:
        return self._sum / self.numeric_count if self.numeric_count else None

    @property
    def std(self) -> float | None:
        if self.numeric_count < 2:
            return None
        mean = self._sum / self.numeric_count
        variance = max(self._sum_sq / self.numeric_count - mean * mean, 0.0)
        return float(np.sqrt(variance))

    @property
    def is_constant(self) -> bool:
        """True if every observed numeric value was identical."""
        return (
            self.numeric_count > 1
            and self.minimum is not None
            and self.maximum is not None
            and self.minimum == self.maximum
        )

    def update(self, series: pd.Series[str]) -> None:
        self.total_rows += len(series)
        as_string = series.astype("string")
        blank = as_string.isna() | (as_string.str.strip() == "")
        self.missing_rows += int(blank.sum())

        numeric = pd.to_numeric(series, errors="coerce")
        # A value that is present but not parseable as a number (IO-VNBD's
        # satellite field is literally "27 / 28") is a distinct finding from
        # a missing value, so the two are counted separately.
        self.non_numeric_rows += int((numeric.isna() & ~blank).sum())

        values = numeric.to_numpy(dtype="float64", na_value=np.nan)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        self.numeric_count += int(values.size)
        chunk_min = float(values.min())
        chunk_max = float(values.max())
        self.minimum = chunk_min if self.minimum is None else min(self.minimum, chunk_min)
        self.maximum = chunk_max if self.maximum is None else max(self.maximum, chunk_max)
        self._sum += float(values.sum())
        self._sum_sq += float(np.square(values).sum())


@dataclass
class StreamScan:
    """Complete single-pass scan result for one CSV file."""

    relative_path: str
    schema: SchemaReport
    row_count: int
    row_count_kind: str
    #: Absolute path the scan came from, so plots can re-read specific columns.
    source_path: str | None = None
    columns: dict[str, ColumnStats] = field(default_factory=dict)
    #: Timestamp series in seconds, or ``None`` if no timestamp column exists.
    time_seconds: np.ndarray | None = None
    time_role: str | None = None
    time_unit: str | None = None
    #: Seconds since local midnight, giving S and V streams a comparable clock.
    time_of_day_s: np.ndarray | None = None
    time_of_day_source: str | None = None
    #: First calendar date seen in a ``DATE`` column, if the schema has one.
    first_date: str | None = None
    #: Boolean per row: does this row carry a usable GNSS fix?
    gnss_valid: np.ndarray | None = None
    #: Boolean per row: did the reported position differ from the previous row?
    #: Distinguishes a genuinely fresh fix from a held/forward-filled one.
    gnss_position_changed: np.ndarray | None = None
    #: Subsampled latitude/longitude retained for trajectory plotting.
    trajectory: tuple[np.ndarray, np.ndarray] | None = None
    parse_error: str | None = None

    def column_by_role(self, role: str) -> ColumnStats | None:
        for stats in self.columns.values():
            if stats.semantic_role == role:
                return stats
        return None


def parse_iovnbd_datetime(series: pd.Series[str]) -> tuple[np.ndarray, str | None]:
    """Parse IO-VNBD ``DATE`` strings into seconds since local midnight.

    The format is ``YYYY-MM-DD HH:MM:SS:mmm`` — note the *colon* before the
    milliseconds, which no standard parser accepts, so the field is split
    manually. Returns the seconds array and the first date string seen.
    """
    text = series.astype("string").str.strip()
    first_date: str | None = None
    non_null = text.dropna()
    if not non_null.empty:
        first_date = str(non_null.iloc[0]).split(" ")[0]

    time_part = text.str.split(" ", n=1).str[-1]
    pieces = time_part.str.split(":", expand=True)
    if pieces.shape[1] < 3:
        return np.full(len(series), np.nan), first_date

    def numeric(index: int) -> np.ndarray:
        return pd.to_numeric(pieces[index], errors="coerce").to_numpy(
            dtype="float64", na_value=np.nan
        )

    hours, minutes, seconds = numeric(0), numeric(1), numeric(2)
    millis = numeric(3) if pieces.shape[1] > 3 else np.zeros(len(series))
    return hours * 3600.0 + minutes * 60.0 + seconds + np.nan_to_num(millis) / 1000.0, first_date


def _resolve_time_column(schema: SchemaReport) -> tuple[str, str, float] | None:
    """Pick the timestamp column. Returns ``(raw_name, role, seconds_per_unit)``.

    Prefers the numeric elapsed-time columns the dataset provides. The
    smartphone's ``DATE`` string column is not used as the primary clock
    because its non-standard ``HH-MI-SS_SSS`` millisecond separator makes it
    ambiguous to parse; it is reported separately instead.
    """
    for column in schema.columns:
        if column.semantic_role == "time_since_start_ms":
            return column.raw_name, column.semantic_role, 1e-3
    for column in schema.columns:
        if column.semantic_role == "time_of_day_s":
            return column.raw_name, column.semantic_role, 1.0
    return None


def scan_csv(
    path: Path,
    relative_path: str | None = None,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    trajectory_points: int = 4000,
) -> StreamScan:
    """Scan one CSV file, computing all Phase 1 per-stream statistics.

    Raises nothing for malformed data: a parse failure is recorded on the
    returned :class:`StreamScan` so one bad file cannot abort a dataset-wide
    scan, and the failure remains visible in the integrity report.
    """
    rel = relative_path or path.name
    schema = read_schema(path, rel)

    scan = StreamScan(
        relative_path=rel,
        schema=schema,
        row_count=0,
        row_count_kind=StatKind.EXACT,
        source_path=str(path),
    )
    for column in schema.columns:
        scan.columns[column.raw_name] = ColumnStats(
            raw_name=column.raw_name,
            normalized_name=column.normalized_name,
            semantic_role=column.semantic_role,
        )

    time_spec = _resolve_time_column(schema)
    latitude_col = next(
        (c.raw_name for c in schema.columns if c.semantic_role == "gnss_latitude"), None
    )
    longitude_col = next(
        (c.raw_name for c in schema.columns if c.semantic_role == "gnss_longitude"), None
    )

    date_col = next(
        (c.raw_name for c in schema.columns if c.semantic_role == "datetime_string"), None
    )
    time_of_day_col = next(
        (c.raw_name for c in schema.columns if c.semantic_role == "time_of_day_s"), None
    )

    time_parts: list[np.ndarray] = []
    tod_parts: list[np.ndarray] = []
    gnss_parts: list[np.ndarray] = []
    lat_parts: list[np.ndarray] = []
    lon_parts: list[np.ndarray] = []

    try:
        reader = pd.read_csv(
            path,
            encoding=schema.encoding,
            chunksize=chunk_size,
            dtype=str,
            keep_default_na=False,
            na_values=[""],
            on_bad_lines="warn",
            low_memory=False,
        )
        for chunk in reader:
            scan.row_count += len(chunk)
            for raw_name, stats in scan.columns.items():
                if raw_name in chunk.columns:
                    stats.update(chunk[raw_name])

            if time_spec is not None and time_spec[0] in chunk.columns:
                seconds = (
                    pd.to_numeric(chunk[time_spec[0]], errors="coerce").to_numpy(
                        dtype="float64", na_value=np.nan
                    )
                    * time_spec[2]
                )
                time_parts.append(seconds)

            if date_col is not None and date_col in chunk.columns:
                tod, first_date = parse_iovnbd_datetime(chunk[date_col])
                tod_parts.append(tod)
                if scan.first_date is None:
                    scan.first_date = first_date
                scan.time_of_day_source = "smartphone DATE column"
            elif time_of_day_col is not None and time_of_day_col in chunk.columns:
                tod_parts.append(
                    pd.to_numeric(chunk[time_of_day_col], errors="coerce").to_numpy(
                        dtype="float64", na_value=np.nan
                    )
                )
                scan.time_of_day_source = "Time Since Start of Day column"

            if latitude_col in chunk.columns and longitude_col in chunk.columns:
                lat = pd.to_numeric(chunk[latitude_col], errors="coerce").to_numpy(
                    dtype="float64", na_value=np.nan
                )
                lon = pd.to_numeric(chunk[longitude_col], errors="coerce").to_numpy(
                    dtype="float64", na_value=np.nan
                )
                # A fix at exactly (0, 0) is the classic "no fix" sentinel, not
                # a real position in the Gulf of Guinea.
                valid = np.isfinite(lat) & np.isfinite(lon) & ~((lat == 0.0) & (lon == 0.0))
                gnss_parts.append(valid)
                lat_parts.append(lat)
                lon_parts.append(lon)
    except (pd.errors.ParserError, UnicodeDecodeError, OSError, ValueError) as exc:
        scan.parse_error = f"{type(exc).__name__}: {exc}"
        return scan

    if time_spec is not None and time_parts:
        scan.time_seconds = np.concatenate(time_parts)
        scan.time_role = time_spec[1]
        scan.time_unit = "s"
    if tod_parts:
        scan.time_of_day_s = np.concatenate(tod_parts)
    if gnss_parts:
        scan.gnss_valid = np.concatenate(gnss_parts)
    if lat_parts:
        latitude = np.concatenate(lat_parts)
        longitude = np.concatenate(lon_parts)
        step = max(1, latitude.size // trajectory_points)
        scan.trajectory = (latitude[::step], longitude[::step])

        # A row whose position equals the previous row's is a *held* fix, not a
        # new measurement. On the smartphone stream the position is
        # forward-filled between far slower GNSS updates, so counting rows
        # would badly overstate the real fix rate.
        changed = np.zeros(latitude.size, dtype=bool)
        if latitude.size > 1:
            changed[1:] = (latitude[1:] != latitude[:-1]) | (longitude[1:] != longitude[:-1])
        scan.gnss_position_changed = changed

    return scan


def load_role_series(scan: StreamScan, role: str) -> np.ndarray | None:
    """Re-read one column of a scanned file by semantic role.

    The scan keeps only aggregates for most channels, so plotting a full
    waveform needs a targeted second read. Used only for the handful of
    sessions selected for diagnostic plots.
    """
    column = next((c for c in scan.schema.columns if c.semantic_role == role), None)
    if column is None or scan.source_path is None:
        return None
    path = Path(scan.source_path)
    if not path.exists():
        return None
    frame = pd.read_csv(
        path,
        encoding=scan.schema.encoding,
        usecols=[column.raw_name],
        low_memory=False,
    )
    return pd.to_numeric(frame[column.raw_name], errors="coerce").to_numpy(
        dtype="float64", na_value=np.nan
    )
