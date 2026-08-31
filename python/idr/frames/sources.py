"""Reading Phase 2 canonical segments into frame-tagged sensor series.

Phase 3 consumes the canonical Parquet written by Phase 2 and never touches a
raw CSV. That is not a stylistic preference: the Phase 2 reader resolves
encoding fallbacks, the S-A4 column shift, three separate clocks and a
positional gyroscope mapping, and a second reader would drift from those
decisions the first time one of them changed.

This module also owns the three timing rules the phase brief calls out, in one
place so an algorithm cannot quietly ignore them:

* **Clock rule.** Differences are taken on ``analysis_time_s``, the clock Phase
  2 established as continuous for that stream. ``source_time_s`` is carried
  through untouched for provenance; ``timestamp_utc`` is never used for
  arithmetic, because Phase 2 could not resolve its timezone.
* **Sampling rule.** Nothing assumes 10 Hz. Actual per-sample Δt is measured
  and reported, and an algorithm that needs regular sampling has to say so.
* **Duplicate rule.** Duplicate timestamps are flagged by Phase 2, not dropped.
  Where an algorithm needs strictly increasing time, a stated deterministic
  policy is applied to a *derived copy*; the canonical file is never rewritten.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from idr.frames.conventions import Frame
from idr.frames.vectors import VectorSeries
from idr.pipeline.canonical import QualityFlag

#: Canonical column triples this phase consumes, by logical channel.
CHANNEL_COLUMNS: dict[str, tuple[str, str, str]] = {
    "accel": ("accel_x", "accel_y", "accel_z"),
    "gyro": ("gyro_x", "gyro_y", "gyro_z"),
    "mag": ("mag_x", "mag_y", "mag_z"),
    "gravity": ("gravity_x", "gravity_y", "gravity_z"),
}

#: A stream is called irregular when the spread of its intervals exceeds this
#: fraction of the median interval. 0.5 is loose on purpose: smartphone logging
#: jitter of tens of percent is normal and harmless, and flagging it as a
#: problem would make the flag meaningless.
IRREGULARITY_THRESHOLD = 0.5


@dataclass
class SampleTiming:
    """Measured timing of one segment. Nothing here is assumed."""

    sample_count: int
    duration_s: float
    median_interval_s: float | None
    mean_interval_s: float | None
    measured_rate_hz: float | None
    min_interval_s: float | None
    max_interval_s: float | None
    interval_iqr_s: float | None
    #: IQR of the intervals divided by the median. 0 means perfectly regular.
    irregularity: float | None
    duplicate_timestamp_rows: int
    non_positive_intervals: int
    analysis_time_source: str
    notes: list[str] = field(default_factory=list)

    @property
    def is_regular(self) -> bool:
        return self.irregularity is not None and self.irregularity <= IRREGULARITY_THRESHOLD


@dataclass
class SegmentData:
    """One canonical segment, ready for frame work."""

    segment_id: str
    session_id: str
    source_file: str
    stream: str
    path: Path
    frame_data: pd.DataFrame
    timing: SampleTiming
    availability: dict[str, bool]
    notes: list[str] = field(default_factory=list)

    def series(self, channel: str) -> VectorSeries | None:
        """A frame-tagged series for one channel, or None when it is absent.

        Returns ``None`` rather than a NaN-filled series when the channel is
        structurally missing, so a caller cannot mistake "this file has no
        magnetometer" for "the magnetometer happened to read NaN".
        """
        columns = CHANNEL_COLUMNS.get(channel)
        if columns is None:
            raise KeyError(
                f"unknown channel {channel!r}; expected one of {sorted(CHANNEL_COLUMNS)}"
            )
        if not self.availability.get(_availability_key(channel), False):
            return None
        if not all(column in self.frame_data.columns for column in columns):
            return None

        samples = self.frame_data.loc[:, list(columns)].to_numpy(dtype="float64")
        if not np.isfinite(samples).any():
            return None
        return VectorSeries(
            samples=samples,
            analysis_time_s=self.frame_data["analysis_time_s"].to_numpy(dtype="float64"),
            frame=Frame.PHONE,
            source_time_s=(
                self.frame_data["source_time_s"].to_numpy(dtype="float64")
                if "source_time_s" in self.frame_data.columns
                else None
            ),
            quality_flags=(
                self.frame_data["quality_flags"].to_numpy(dtype="int64")
                if "quality_flags" in self.frame_data.columns
                else None
            ),
            source=f"{self.segment_id}:{channel}",
        )

    def column(self, name: str) -> np.ndarray | None:
        if name not in self.frame_data.columns:
            return None
        return self.frame_data[name].to_numpy(dtype="float64")


def _availability_key(channel: str) -> str:
    return {
        "accel": "has_accelerometer",
        "gyro": "has_gyroscope",
        "mag": "has_magnetometer",
        "gravity": "has_gravity",
    }[channel]


def measure_timing(frame: pd.DataFrame) -> SampleTiming:
    """Measure a segment's actual sampling behaviour.

    Every field is derived from the timestamps as written. The sampling rate is
    a *measurement* reported back to the caller, never a configured constant —
    Phase 2 found families at roughly 2, 10, 24 and 1000 Hz in the same dataset.
    """
    times = frame["analysis_time_s"].to_numpy(dtype="float64")
    finite = np.isfinite(times)
    count = int(times.size)
    source = (
        str(frame["analysis_time_source"].iloc[0])
        if "analysis_time_source" in frame.columns and count
        else "unknown"
    )
    notes: list[str] = []

    duplicates = 0
    if "quality_flags" in frame.columns:
        flags = frame["quality_flags"].to_numpy(dtype="int64")
        duplicates = int((flags & QualityFlag.DUPLICATE_TIMESTAMP.value).astype(bool).sum())

    if int(finite.sum()) < 2:
        return SampleTiming(
            sample_count=count,
            duration_s=0.0,
            median_interval_s=None,
            mean_interval_s=None,
            measured_rate_hz=None,
            min_interval_s=None,
            max_interval_s=None,
            interval_iqr_s=None,
            irregularity=None,
            duplicate_timestamp_rows=duplicates,
            non_positive_intervals=0,
            analysis_time_source=source,
            notes=["fewer than two finite timestamps; no interval can be measured"],
        )

    usable = times[finite]
    intervals = np.diff(usable)
    non_positive = int((intervals <= 0.0).sum())
    positive = intervals[intervals > 0.0]
    duration = float(usable[-1] - usable[0])

    if positive.size == 0:
        notes.append("every interval is zero or negative; the clock does not advance")
        return SampleTiming(
            sample_count=count,
            duration_s=duration,
            median_interval_s=None,
            mean_interval_s=None,
            measured_rate_hz=None,
            min_interval_s=None,
            max_interval_s=None,
            interval_iqr_s=None,
            irregularity=None,
            duplicate_timestamp_rows=duplicates,
            non_positive_intervals=non_positive,
            analysis_time_source=source,
            notes=notes,
        )

    median = float(np.median(positive))
    iqr = float(np.percentile(positive, 75) - np.percentile(positive, 25))
    irregularity = float(iqr / median) if median > 0.0 else None
    if irregularity is not None and irregularity > IRREGULARITY_THRESHOLD:
        notes.append(
            f"intervals vary by an IQR of {iqr * 1000:.1f} ms around a median of "
            f"{median * 1000:.1f} ms; treat this stream as irregularly sampled"
        )
    if non_positive:
        notes.append(
            f"{non_positive} interval(s) are zero or negative "
            "(duplicate or out-of-order timestamps, retained by Phase 2)"
        )

    return SampleTiming(
        sample_count=count,
        duration_s=duration,
        median_interval_s=median,
        mean_interval_s=float(np.mean(positive)),
        measured_rate_hz=float(1.0 / median),
        min_interval_s=float(np.min(positive)),
        max_interval_s=float(np.max(positive)),
        interval_iqr_s=iqr,
        irregularity=irregularity,
        duplicate_timestamp_rows=duplicates,
        non_positive_intervals=non_positive,
        analysis_time_source=source,
        notes=notes,
    )


@dataclass
class DeduplicationResult:
    """A derived, strictly-increasing view plus the record of what it changed."""

    frame_data: pd.DataFrame
    removed_rows: int
    policy: str


def strictly_increasing_view(frame: pd.DataFrame) -> DeduplicationResult:
    """A derived copy whose ``analysis_time_s`` strictly increases.

    **Policy, stated once and applied everywhere:** among rows sharing an
    ``analysis_time_s``, the *first in canonical row order* is kept and the rest
    are set aside. Rows whose timestamp is not finite are also set aside.

    First-wins is chosen over last-wins or averaging because Phase 2 already
    stably sorted rows within a segment, so "first" is a defined, reproducible
    row rather than an arbitrary one — and averaging two samples that claim the
    same instant would invent a measurement that no sensor produced.

    The input frame is never modified. This returns a new object, and the
    canonical Parquet on disk is untouched.
    """
    times = frame["analysis_time_s"].to_numpy(dtype="float64")
    keep = np.zeros(times.size, dtype=bool)
    previous = -np.inf
    for index, value in enumerate(times):
        if np.isfinite(value) and value > previous:
            keep[index] = True
            previous = value
    removed = int(times.size - keep.sum())
    return DeduplicationResult(
        frame_data=frame.loc[keep].reset_index(drop=True),
        removed_rows=removed,
        policy=(
            "keep the first row of each analysis_time_s in canonical order; "
            "set aside later duplicates and non-finite timestamps"
        ),
    )


def load_field_mapping(metadata_dir: Path) -> dict[str, dict[str, bool]]:
    """Per-source-file feature availability recorded by Phase 2.

    Returns an empty mapping when the artifact is absent, so a caller working
    from a bare processed directory still runs — with availability then taken
    from the data itself rather than from the manifest.
    """
    path = metadata_dir / "canonical_field_mapping.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, dict[str, bool]] = {}
    for entry in payload.get("per_file", []):
        source_file = entry.get("source_file")
        if not isinstance(source_file, str):
            continue
        mapping[source_file] = {
            key: bool(value) for key, value in entry.items() if key.startswith("has_")
        }
    return mapping


def _availability_from_data(frame: pd.DataFrame) -> dict[str, bool]:
    """Fallback availability, inferred from whether a channel has any value."""
    availability: dict[str, bool] = {}
    for channel, columns in CHANNEL_COLUMNS.items():
        present = all(column in frame.columns for column in columns)
        has_values = present and bool(
            np.isfinite(frame.loc[:, list(columns)].to_numpy(dtype="float64")).any()
        )
        availability[_availability_key(channel)] = has_values
    return availability


def load_segment(
    path: Path, *, field_mapping: dict[str, dict[str, bool]] | None = None
) -> SegmentData:
    """Load one canonical Parquet segment.

    Read-only. Phase 3 writes its outputs to ``reports/`` and its own metadata
    directory; it never writes back into ``data/processed/``.
    """
    frame = pd.read_parquet(path)
    if "analysis_time_s" not in frame.columns:
        raise ValueError(f"{path.name} has no analysis_time_s column; is it a canonical segment?")

    def first(column: str, default: str = "") -> str:
        if column not in frame.columns or frame.empty:
            return default
        value = frame[column].iloc[0]
        return default if pd.isna(value) else str(value)

    source_file = first("source_file")
    availability = _availability_from_data(frame)
    notes: list[str] = []
    if field_mapping and source_file in field_mapping:
        recorded = field_mapping[source_file]
        for key, value in recorded.items():
            if key in availability and availability[key] != value:
                notes.append(
                    f"{key}: manifest says {value}, data says {availability[key]}; "
                    "taking the stricter of the two"
                )
                availability[key] = availability[key] and value
            elif key not in availability:
                availability[key] = value

    return SegmentData(
        segment_id=first("segment_id", path.stem),
        session_id=first("session_id"),
        source_file=source_file,
        stream=first("stream", "unknown"),
        path=path,
        frame_data=frame,
        timing=measure_timing(frame),
        availability=availability,
        notes=notes,
    )


def find_segments(processed_dir: Path, stream: str = "smartphone") -> list[Path]:
    """Canonical segment files for one stream, in a stable order."""
    directory = processed_dir / stream
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.parquet"))
