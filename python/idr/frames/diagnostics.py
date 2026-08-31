"""Running the Phase 3 estimators over real canonical segments.

This is a *diagnostic* pass, and the distinction is load-bearing. It cannot
validate the alignment, because the true phone→vehicle rotation for IO-VNBD is
not documented and no independent measurement of it exists. What it can
establish is everything else: that the estimators run over real, irregularly
sampled, duplicate-bearing data without failing; that their confidences move in
the directions the physics predicts; that the recovered geometry is
self-consistent; and where the data simply cannot support an answer.

Anything a reader might mistake for ground truth is labelled as an estimate,
here and in the generated report.

The speed reference deserves a note. Yaw needs an independent scalar speed, and
Phase 2 offers two: ``ref_velocity_mps``, interpolated from a synchronized
vehicle stream, and ``gnss_speed_mps`` from the phone. The first is preferred
where Phase 2 marked it available, because the second updates only about every
9 s — the phone's position is forward-filled between fixes — and differentiating
a staircase produces acceleration spikes that are artefacts of the sampling, not
of the vehicle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from idr.frames.alignment import (
    AlignmentEstimate,
    CalibrationStatus,
    estimate_alignment,
    linear_accel_for_alignment,
)
from idr.frames.gravity import GravityEstimate, estimate_gravity
from idr.frames.magnetometer import MagnetometerQuality, assess_magnetometer
from idr.frames.orientation import OrientationTrack, estimate_orientation
from idr.frames.sources import SampleTiming, SegmentData, load_segment
from idr.frames.vectors import VectorSeries
from idr.logging import get_logger

logger = get_logger(__name__)

#: Speed references, best first.
SPEED_SOURCES = ("ref_velocity_mps", "gnss_speed_mps")


@dataclass
class SpeedReference:
    """Which speed signal was used for heading, and how good it is."""

    values: np.ndarray | None
    valid: np.ndarray | None
    source: str
    valid_fraction: float
    note: str = ""


@dataclass
class SegmentDiagnostic:
    """Everything Phase 3 measured about one canonical segment."""

    segment_id: str
    session_id: str
    source_file: str
    path: Path
    timing: SampleTiming
    gravity: GravityEstimate
    magnetometer: MagnetometerQuality
    orientation: OrientationTrack | None
    alignment: AlignmentEstimate | None
    speed_reference: SpeedReference
    #: Descriptive comparison against the dataset's own orientation columns.
    dataset_orientation_rmse_deg: dict[str, float] | None
    orientation_seconds: float
    alignment_seconds: float
    seconds_per_sample: float
    #: Residual phone/vehicle lag Phase 3 had to remove, seconds. ``None`` when
    #: no heading estimate was attempted.
    forward_lag_s: float | None = None
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def status(self) -> CalibrationStatus:
        if self.error is not None:
            return CalibrationStatus.FAILED
        return self.alignment.status if self.alignment else CalibrationStatus.FAILED

    @property
    def phone_accel(self) -> VectorSeries | None:
        """Re-read the phone-frame accelerometer for this segment.

        Reloaded on demand rather than retained: only a couple of segments are
        ever plotted, and holding every segment's raw samples would multiply
        the run's memory by the size of the dataset for no benefit.
        """
        return load_segment(self.path).series("accel")


@dataclass
class DiagnosticRun:
    """The whole Phase 3 diagnostic pass."""

    started_at: str
    processed_dir: str
    segments: list[SegmentDiagnostic] = field(default_factory=list)
    elapsed_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    def succeeded(self) -> list[SegmentDiagnostic]:
        return [s for s in self.segments if s.status is CalibrationStatus.SUCCESS]

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for segment in self.segments:
            counts[str(segment.status)] = counts.get(str(segment.status), 0) + 1
        return dict(sorted(counts.items()))


def choose_speed_reference(segment: SegmentData) -> SpeedReference:
    """Pick the best available scalar speed, and say which one it is.

    Prefers the Phase 2 reference velocity, falling back to GNSS speed on rows
    Phase 2 marked fresh. A stale GNSS row repeats the previous value, so
    including it would inject a false zero into ``dv/dt``.
    """
    reference = segment.column("ref_velocity_mps")
    if reference is not None and np.isfinite(reference).any():
        available = segment.frame_data.get("label_available")
        valid = (
            np.isfinite(reference)
            if available is None
            else np.isfinite(reference) & available.fillna(False).to_numpy(dtype=bool)
        )
        fraction = float(valid.mean())
        if fraction > 0.05:
            return SpeedReference(
                values=reference,
                valid=valid,
                source="ref_velocity_mps",
                valid_fraction=fraction,
                note="Phase 2 reference velocity from a synchronized vehicle stream",
            )

    gnss = segment.column("gnss_speed_mps")
    if gnss is not None and np.isfinite(gnss).any():
        fresh = segment.frame_data.get("gnss_position_fresh")
        valid = (
            np.isfinite(gnss)
            if fresh is None
            else np.isfinite(gnss) & fresh.fillna(False).to_numpy(dtype=bool)
        )
        return SpeedReference(
            values=gnss,
            valid=valid,
            source="gnss_speed_mps",
            valid_fraction=float(valid.mean()),
            note=(
                "smartphone GNSS speed, restricted to rows Phase 2 marked fresh; "
                "the underlying position refreshes only about every 9 s"
            ),
        )

    return SpeedReference(
        values=None,
        valid=None,
        source="none",
        valid_fraction=0.0,
        note="no usable speed reference, so heading cannot be estimated",
    )


def _dataset_orientation_rmse(
    segment: SegmentData, track: OrientationTrack
) -> dict[str, float] | None:
    """Compare our estimate against the dataset's own orientation columns.

    **Descriptive only.** Phase 2 established that this dataset's axis labels
    are untrustworthy aliases, and the columns' own convention is undocumented,
    so a disagreement here is not evidence that either side is wrong. It is
    reported because a large disagreement is worth knowing about, and a small
    one is mildly reassuring — nothing more.
    """
    columns = ("orientation_roll", "orientation_pitch", "orientation_yaw")
    if not all(name in segment.frame_data.columns for name in columns):
        return None
    ours = track.euler_deg()
    result: dict[str, float] = {}
    for index, column in enumerate(columns):
        theirs = segment.column(column)
        if theirs is None:
            continue
        finite = np.isfinite(theirs) & np.isfinite(ours[:, index])
        if int(finite.sum()) < 10:
            continue
        # Wrap to ±180° before differencing: 359° and 1° are 2° apart.
        difference = (ours[finite, index] - theirs[finite] + 180.0) % 360.0 - 180.0
        result[column] = float(np.sqrt(np.mean(difference**2)))
    return result or None


def analyze_segment(
    path: Path, *, field_mapping: dict[str, dict[str, bool]] | None = None
) -> SegmentDiagnostic:
    """Run the full Phase 3 chain over one canonical segment."""
    segment = load_segment(path, field_mapping=field_mapping)
    accel = segment.series("accel")
    gyro = segment.series("gyro")
    mag = segment.series("mag")
    gravity_channel = segment.series("gravity")
    speed = choose_speed_reference(segment)
    notes = list(segment.notes)

    if accel is None:
        empty = estimate_gravity(
            gyro if gyro is not None else _empty_series(),
        )
        return SegmentDiagnostic(
            segment_id=segment.segment_id,
            session_id=segment.session_id,
            source_file=segment.source_file,
            path=path,
            timing=segment.timing,
            gravity=empty,
            magnetometer=assess_magnetometer(None, available=False),
            orientation=None,
            alignment=None,
            speed_reference=speed,
            dataset_orientation_rmse_deg=None,
            orientation_seconds=0.0,
            alignment_seconds=0.0,
            seconds_per_sample=0.0,
            error="segment has no usable accelerometer channel",
            notes=notes,
        )

    started = time.perf_counter()
    track = estimate_orientation(
        gyro=gyro,
        accel=accel,
        mag=mag,
        gravity_channel=gravity_channel,
        magnetometer_available=segment.availability.get("has_magnetometer", False),
    )
    orientation_seconds = time.perf_counter() - started

    started = time.perf_counter()
    linear = linear_accel_for_alignment(accel, track.gravity)
    alignment = estimate_alignment(
        gravity=track.gravity,
        linear_accel=linear,
        speed_mps=speed.values,
        valid_speed=speed.valid,
        gyro=gyro,
        magnetometer=track.magnetometer,
        duration_s=segment.timing.duration_s,
    )
    alignment_seconds = time.perf_counter() - started

    sample_count = max(1, len(accel))
    return SegmentDiagnostic(
        segment_id=segment.segment_id,
        session_id=segment.session_id,
        source_file=segment.source_file,
        path=path,
        timing=segment.timing,
        gravity=track.gravity,
        magnetometer=track.magnetometer,
        orientation=track,
        alignment=alignment,
        speed_reference=speed,
        dataset_orientation_rmse_deg=_dataset_orientation_rmse(segment, track),
        orientation_seconds=orientation_seconds,
        alignment_seconds=alignment_seconds,
        seconds_per_sample=(orientation_seconds + alignment_seconds) / sample_count,
        forward_lag_s=alignment.quality.forward_lag_s,
        notes=notes,
    )


def _empty_series() -> VectorSeries:
    return VectorSeries(samples=np.zeros((0, 3)), analysis_time_s=np.zeros(0))


def run_diagnostics(
    processed_dir: Path,
    *,
    metadata_dir: Path | None = None,
    stream: str = "smartphone",
    limit: int | None = None,
) -> DiagnosticRun:
    """Run Phase 3 over canonical segments and collect the results.

    Segments are taken in sorted order so a limited run is reproducible rather
    than dependent on directory iteration order.
    """
    from datetime import UTC, datetime

    from idr.frames.sources import find_segments, load_field_mapping

    started = time.perf_counter()
    run = DiagnosticRun(started_at=datetime.now(UTC).isoformat(), processed_dir=str(processed_dir))

    field_mapping = load_field_mapping(metadata_dir) if metadata_dir else {}
    if metadata_dir and not field_mapping:
        run.notes.append(
            "no canonical_field_mapping.json found; sensor availability was inferred "
            "from the data rather than read from the Phase 2 manifest"
        )

    paths = find_segments(processed_dir, stream)
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        run.notes.append(f"no canonical {stream} segments under {processed_dir}")
        run.elapsed_s = time.perf_counter() - started
        return run

    for index, path in enumerate(paths, start=1):
        try:
            run.segments.append(analyze_segment(path, field_mapping=field_mapping))
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("Segment %s failed: %s", path.name, exc)
            run.notes.append(f"{path.name}: {type(exc).__name__}: {exc}")
        if index % 25 == 0 or index == len(paths):
            logger.info("Analyzed %d/%d segment(s)", index, len(paths))

    run.elapsed_s = time.perf_counter() - started
    return run


def performance_summary(run: DiagnosticRun) -> dict[str, float]:
    """Average and p95 per-sample processing cost, in microseconds.

    Reported because the engine must eventually run in real time, not because
    anything here has been optimized. Nothing is tuned for speed in Phase 3.
    """
    per_sample = np.array(
        [s.seconds_per_sample for s in run.segments if s.seconds_per_sample > 0.0]
    )
    if per_sample.size == 0:
        return {}
    total = np.array([s.orientation_seconds + s.alignment_seconds for s in run.segments])
    return {
        "mean_us_per_sample": float(per_sample.mean() * 1e6),
        "p95_us_per_sample": float(np.percentile(per_sample, 95) * 1e6),
        "mean_s_per_segment": float(total.mean()),
        "p95_s_per_segment": float(np.percentile(total, 95)),
        "segments": float(per_sample.size),
    }
