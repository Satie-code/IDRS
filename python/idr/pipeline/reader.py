"""Read one IO-VNBD source file into a canonical, quality-annotated frame.

Responsibilities, in order: decode the file (recording how), detect and repair
a column shift if present, map source columns onto the canonical schema,
convert units and types, parse the satellite field, and attach feature
availability. Reading is chunked so memory stays bounded regardless of file
size.

Nothing here drops rows, interpolates values, or resamples. Timing analysis
and segmentation happen in :mod:`idr.pipeline.timeline` on the frame this
module returns.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from idr.dataset.schema import SchemaReport, read_schema
from idr.pipeline.canonical import (
    FEATURE_AVAILABILITY_KEYS,
    QualityFlag,
    schema_for,
)
from idr.pipeline.mapping import (
    ColumnMapping,
    build_mapping,
    parse_satellites,
    satellites_interpretation_is_consistent,
)
from idr.pipeline.repair import (
    NO_REPAIR,
    RepairDecision,
    apply_column_shift,
    detect_column_shift,
)

DEFAULT_CHUNK_SIZE = 100_000


class ReadError(RuntimeError):
    """Raised when a source file cannot be turned into a canonical frame."""


@dataclass
class ReadResult:
    """A canonical frame plus everything worth recording about producing it."""

    frame: pd.DataFrame
    source_file: str
    stream: str
    file_hash: str
    payload_hash: str
    source_encoding: str
    decode_method: str
    encoding_warning: str | None
    repair: RepairDecision
    row_count: int
    availability: dict[str, bool]
    unmapped_columns: list[str] = field(default_factory=list)
    alias_resolved: dict[str, str] = field(default_factory=dict)
    satellites_consistent: bool = True
    warnings: list[str] = field(default_factory=list)


def hash_file_and_payload(path: Path) -> tuple[str, str]:
    """Return ``(file_sha256, payload_sha256)``.

    The payload hash covers everything after the first newline — the data rows
    without the header. Phase 2 found files whose bytes differ only in header
    labels but whose data rows are byte-identical; grouping on the file hash
    alone would place such recordings in different splits and leak. The payload
    hash is what the split logic groups on.
    """
    file_digest = hashlib.sha256()
    payload_digest = hashlib.sha256()
    header_consumed = False
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            file_digest.update(chunk)
            if header_consumed:
                payload_digest.update(chunk)
            else:
                newline = chunk.find(b"\n")
                if newline >= 0:
                    payload_digest.update(chunk[newline + 1 :])
                    header_consumed = True
    return file_digest.hexdigest(), payload_digest.hexdigest()


def _availability(mapping: ColumnMapping, stream: str) -> dict[str, bool]:
    present = set(mapping.fields)
    return {
        "has_accelerometer": {"accel_x", "accel_y", "accel_z"} <= present,
        "has_gyroscope": {"gyro_x", "gyro_y", "gyro_z"} <= present,
        "has_magnetometer": {"mag_x", "mag_y", "mag_z"} <= present,
        "has_gravity": {"gravity_x", "gravity_y", "gravity_z"} <= present,
        "has_orientation": {"orientation_pitch", "orientation_roll"} <= present,
        "has_gnss": {"latitude", "longitude"} <= present,
        "has_gnss_accuracy": "gnss_accuracy_m" in present,
    }


def _numeric(series: pd.Series, factor: float | None) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    return values if factor is None else values * factor


def _parse_datetime_seconds(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Parse IO-VNBD ``DATE`` strings to (seconds-of-day, epoch-nanoseconds).

    Format is ``YYYY-MM-DD HH:MM:SS:mmm`` — milliseconds follow a *colon*, which
    no standard parser accepts, so the field is split manually.
    """
    text = series.astype("string").str.strip()
    parts = text.str.split(" ", n=1, expand=True)
    if parts.shape[1] < 2:
        empty = np.full(len(series), np.nan)
        return empty, empty

    time_parts = parts[1].str.split(":", expand=True)
    if time_parts.shape[1] < 3:
        empty = np.full(len(series), np.nan)
        return empty, empty

    def numeric(index: int) -> np.ndarray:
        return pd.to_numeric(time_parts[index], errors="coerce").to_numpy(
            dtype="float64", na_value=np.nan
        )

    hours, minutes, seconds = numeric(0), numeric(1), numeric(2)
    millis = numeric(3) if time_parts.shape[1] > 3 else np.zeros(len(series))
    seconds_of_day = hours * 3600.0 + minutes * 60.0 + seconds + np.nan_to_num(millis) / 1000.0

    # Combine the date with the reconstructed time to get a wall-clock stamp.
    dates = pd.to_datetime(parts[0], errors="coerce", format="%Y-%m-%d")
    offsets = pd.to_timedelta(seconds_of_day, unit="s")
    stamps = (dates + offsets).to_numpy(dtype="datetime64[ns]")
    return seconds_of_day, stamps


def read_canonical(
    path: Path,
    relative_path: str,
    stream: str,
    *,
    dataset_family: str,
    session_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> ReadResult:
    """Read one source file into a canonical frame.

    Raises :class:`ReadError` on a failure the caller must record; it never
    returns a partially-populated frame silently.
    """
    try:
        schema: SchemaReport = read_schema(path, relative_path)
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise ReadError(f"header unreadable: {exc}") from exc

    encoding_warning = None if schema.encoding_is_utf8 else schema.encoding_note
    if schema.encoding_is_utf8 and "mojibake" in schema.encoding_note:
        encoding_warning = schema.encoding_note

    header_names = [column.raw_name for column in schema.columns]
    mapping = build_mapping(schema, stream)
    file_hash, payload_hash = hash_file_and_payload(path)

    warnings: list[str] = []
    repair = NO_REPAIR
    pieces: list[pd.DataFrame] = []

    try:
        reader = pd.read_csv(
            path,
            encoding=schema.encoding,
            chunksize=chunk_size,
            dtype=str,
            keep_default_na=False,
            na_values=[""],
            low_memory=False,
        )
        for chunk in reader:
            if repair is NO_REPAIR and not pieces:
                repair = detect_column_shift(chunk, header_names)
            if repair.applied:
                chunk = apply_column_shift(chunk, header_names)
            pieces.append(chunk)
    except (pd.errors.ParserError, UnicodeDecodeError, OSError, ValueError) as exc:
        raise ReadError(f"{type(exc).__name__}: {exc}") from exc

    # A header-only file yields one *empty* chunk rather than no chunks, so the
    # emptiness test has to be on the concatenated rows. Reporting it as a read
    # failure is deliberate: a zero-row canonical frame would otherwise flow on
    # and be counted as a successfully processed file.
    source = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    row_count = len(source)
    if row_count == 0:
        raise ReadError("file contains a header but no data rows")
    availability = _availability(mapping, stream)

    canonical: dict[str, np.ndarray | pd.Series] = {}
    flags = np.zeros(row_count, dtype="int64")

    # --- values -----------------------------------------------------------
    for canonical_name, (source_name, factor) in mapping.fields.items():
        if source_name not in source.columns:
            warnings.append(f"mapped column {source_name!r} absent after parse")
            continue
        canonical[canonical_name] = _numeric(source[source_name], factor)

    if mapping.altitude_column and mapping.altitude_column in source.columns:
        canonical["altitude_m"] = _numeric(source[mapping.altitude_column], mapping.altitude_factor)

    # --- satellites -------------------------------------------------------
    satellites_consistent = True
    if mapping.satellites_column and mapping.satellites_column in source.columns:
        raw_sat = source[mapping.satellites_column].astype("string")
        canonical["gnss_satellites_raw"] = raw_sat
        in_use, in_view = parse_satellites(raw_sat)
        canonical["gnss_satellites_in_use"] = in_use
        canonical["gnss_satellites_in_view"] = in_view
        satellites_consistent = satellites_interpretation_is_consistent(in_use, in_view)
        if not satellites_consistent:
            warnings.append(
                "satellite field has rows where in_use > in_view; the 'in use / in view' "
                "reading is not supported by this file"
            )

    # --- time -------------------------------------------------------------
    source_time = np.full(row_count, np.nan)
    if mapping.time_column and mapping.time_column in source.columns:
        raw_time = _numeric(source[mapping.time_column], None)
        # Smartphone elapsed time is milliseconds; vehicle time-of-day is seconds.
        source_time = raw_time / 1000.0 if mapping.time_role == "time_since_start_ms" else raw_time
    canonical["source_time_s"] = source_time

    time_of_day = np.full(row_count, np.nan)
    stamps = np.full(row_count, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    if mapping.date_column and mapping.date_column in source.columns:
        time_of_day, stamps = _parse_datetime_seconds(source[mapping.date_column])
    elif mapping.time_role == "time_of_day_s":
        time_of_day = source_time
    canonical["time_of_day_s"] = time_of_day
    canonical["timestamp_utc"] = stamps

    # Choose the continuous clock for analysis. The smartphone's elapsed
    # counter restarts mid-recording in 15 files (verified), while its
    # wall-clock DATE column runs straight through; segmenting on the counter
    # would fragment a single continuous drive, so the wall clock wins where
    # it exists.
    if mapping.date_column and np.isfinite(stamps.astype("float64")).any():
        analysis_time = stamps.astype("datetime64[ns]").astype("float64") / 1e9
        analysis_source = "wall_clock"
    elif mapping.time_role == "time_of_day_s":
        analysis_time = time_of_day
        analysis_source = "time_of_day"
    else:
        analysis_time = source_time
        analysis_source = "elapsed"
    canonical["analysis_time_s"] = analysis_time

    flags[~np.isfinite(analysis_time)] |= QualityFlag.INVALID_TIMESTAMP.value

    # --- identity ---------------------------------------------------------
    frame = pd.DataFrame(canonical)
    for name, value in (
        ("analysis_time_source", analysis_source),
        ("dataset_family", dataset_family),
        ("source_file", relative_path),
        ("file_hash", file_hash),
        ("payload_hash", payload_hash),
        ("session_id", session_id),
        ("stream", stream),
    ):
        frame[name] = value
    # segment_id is assigned by timeline.segment_frame once gaps are known.
    frame["segment_id"] = pd.NA

    # --- fill absent canonical channels with typed missing values ----------
    for spec in schema_for(stream):
        if spec.name in frame.columns:
            continue
        if spec.dtype.startswith("float"):
            frame[spec.name] = np.nan
        elif spec.dtype == "boolean":
            frame[spec.name] = pd.NA
        elif spec.dtype.startswith("datetime"):
            frame[spec.name] = pd.NaT
        elif spec.dtype == "int32":
            frame[spec.name] = 0
        else:
            frame[spec.name] = pd.NA

    if not all(availability[key] for key in ("has_accelerometer", "has_gnss")):
        flags |= QualityFlag.MISSING_SENSOR.value
    for key in FEATURE_AVAILABILITY_KEYS:
        if not availability.get(key, False):
            flags |= QualityFlag.MISSING_SENSOR.value
            break
    if encoding_warning:
        flags |= QualityFlag.ENCODING_WARNING.value
    if repair.applied:
        flags |= QualityFlag.SCHEMA_REPAIR.value

    frame["quality_flags"] = flags.astype("int32")

    if mapping.unmapped:
        warnings.append(f"unmapped source columns: {mapping.unmapped}")

    return ReadResult(
        frame=frame,
        source_file=relative_path,
        stream=stream,
        file_hash=file_hash,
        payload_hash=payload_hash,
        source_encoding=schema.encoding,
        decode_method="strict" if schema.encoding_is_utf8 else "cp1252-fallback",
        encoding_warning=encoding_warning,
        repair=repair,
        row_count=row_count,
        availability=availability,
        unmapped_columns=mapping.unmapped,
        alias_resolved=mapping.alias_resolved,
        satellites_consistent=satellites_consistent,
        warnings=warnings,
    )
