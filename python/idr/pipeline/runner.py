"""Phase 2 pipeline orchestrator and CLI.

Usage::

    python -m idr.pipeline.runner --raw-path data/raw/io_vnbd \\
        --processed data/processed/io_vnbd --metadata data/metadata/io_vnbd \\
        --report-output reports/io_vnbd

Stages: read/canonicalize -> diagnose timing and segment -> annotate GNSS
freshness -> write canonical Parquet -> pair and synchronize -> build reference
labels -> assign leakage-safe splits -> emit metadata and reports.

Failures are recorded per file and surfaced in ``processing_errors.csv``;
nothing is silently skipped. A structural failure (no canonical output at all)
exits non-zero.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from idr.dataset.inventory import build_file_inventory
from idr.logging import get_logger
from idr.pipeline.canonical import (
    CANONICAL_SCHEMA_VERSION,
    FEATURE_AVAILABILITY_KEYS,
    QualityFlag,
    describe_flags,
    schema_for,
)
from idr.pipeline.freshness import FreshnessReport, annotate_freshness
from idr.pipeline.reader import ReadError, ReadResult, read_canonical
from idr.pipeline.reference import LabelCoverage, build_velocity_labels
from idr.pipeline.splits import (
    DEFAULT_SEED,
    ContentAudit,
    SplitAssignment,
    SplitSummary,
    SplitUnit,
    assign_splits,
    audit_content_leakage,
    content_fingerprint,
    splits_to_json,
    verify_no_leakage,
)
from idr.pipeline.sync import SyncResult, SyncStatus, analyze_pair
from idr.pipeline.timeline import SegmentRecord, TimingDiagnosis, diagnose_and_segment
from idr.version import __version__

logger = get_logger(__name__)


@dataclass
class ProcessingError:
    source_file: str
    stage: str
    error_type: str
    message: str
    recoverable: bool
    status: str


@dataclass
class PipelineResult:
    """Everything the Phase 2 run produced."""

    run_id: str
    started_at: str
    raw_path: str
    reads: list[ReadResult] = field(default_factory=list)
    timing: list[TimingDiagnosis] = field(default_factory=list)
    segments: list[SegmentRecord] = field(default_factory=list)
    freshness: list[FreshnessReport] = field(default_factory=list)
    sync: list[SyncResult] = field(default_factory=list)
    labels: list[LabelCoverage] = field(default_factory=list)
    assignments: list[SplitAssignment] = field(default_factory=list)
    split_summary: SplitSummary | None = None
    errors: list[ProcessingError] = field(default_factory=list)
    canonical_paths: dict[str, str] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)
    elapsed_s: float = 0.0
    leakage_violations: list[str] = field(default_factory=list)
    content_audit: ContentAudit | None = None


def _stream_of(name: str) -> str | None:
    upper = Path(name).name.upper()
    if upper.startswith("S-"):
        return "smartphone"
    if upper.startswith("V-"):
        return "vehicle"
    return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # An explicit empty artifact beats a missing one: "zero errors" is a
        # result, not an absence of information.
        path.write_text("# no rows\n", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


class Phase2Pipeline:
    """Runs canonicalization end to end. Never writes into ``raw_path``."""

    def __init__(
        self,
        raw_path: Path,
        processed_path: Path,
        *,
        limit: int | None = None,
        seed: str = DEFAULT_SEED,
    ) -> None:
        if not raw_path.is_dir():
            raise NotADirectoryError(f"raw path is not a directory: {raw_path}")
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.limit = limit
        self.seed = seed

    # -- stage 1: canonicalize -------------------------------------------
    def _canonicalize(self, result: PipelineResult) -> dict[str, pd.DataFrame]:
        records = [
            record
            for record in build_file_inventory(self.raw_path, compute_hashes=False)
            if record.is_file and record.extension == ".csv"
        ]
        if self.limit:
            records = records[: self.limit]

        frames: dict[str, pd.DataFrame] = {}
        for index, record in enumerate(records, start=1):
            stream = _stream_of(record.filename)
            if stream is None:
                result.errors.append(
                    ProcessingError(
                        record.relative_path,
                        "canonicalize",
                        "UnrecognizedStream",
                        "filename does not start with 'S-' or 'V-'",
                        recoverable=True,
                        status="skipped",
                    )
                )
                continue

            try:
                read = read_canonical(
                    self.raw_path / record.relative_path,
                    record.relative_path,
                    stream,
                    dataset_family=record.dataset_family,
                    session_id=record.session_id or "unknown",
                )
            except ReadError as exc:
                result.errors.append(
                    ProcessingError(
                        record.relative_path,
                        "canonicalize",
                        type(exc).__name__,
                        str(exc)[:400],
                        recoverable=True,
                        status="failed",
                    )
                )
                continue

            try:
                frame, timing, segments = diagnose_and_segment(
                    read.frame,
                    source_file=record.relative_path,
                    session_id=record.session_id or "unknown",
                    stream=stream,
                    dataset_family=record.dataset_family,
                    payload_hash=read.payload_hash,
                )
                frame, freshness = annotate_freshness(
                    frame, source_file=record.relative_path, stream=stream
                )
            except (ValueError, KeyError, TypeError) as exc:
                result.errors.append(
                    ProcessingError(
                        record.relative_path,
                        "timeline/freshness",
                        type(exc).__name__,
                        str(exc)[:400],
                        recoverable=True,
                        status="failed",
                    )
                )
                continue

            result.reads.append(read)
            result.timing.append(timing)
            result.segments.extend(segments)
            result.freshness.append(freshness)
            frames[record.relative_path] = frame

            if index % 50 == 0 or index == len(records):
                logger.info("Canonicalized %d/%d file(s)", index, len(records))
        return frames

    # -- stage 2: synchronize --------------------------------------------
    def _synchronize(
        self, result: PipelineResult, frames: dict[str, pd.DataFrame]
    ) -> dict[str, SyncResult]:
        by_session: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(
            lambda: {"smartphone": [], "vehicle": []}
        )
        for path, frame in frames.items():
            if frame.empty:
                continue
            family = str(frame["dataset_family"].iloc[0])
            session = str(frame["session_id"].iloc[0])
            stream = str(frame["stream"].iloc[0])
            by_session[(family, session)][stream].append(path)

        results: dict[str, SyncResult] = {}
        for (family, session), streams in sorted(by_session.items()):
            if not streams["smartphone"] or not streams["vehicle"]:
                continue
            phone_path = sorted(streams["smartphone"])[0]
            vehicle_path = sorted(streams["vehicle"])[0]
            phone, vehicle = frames[phone_path], frames[vehicle_path]

            sync = analyze_pair(
                pair_id=f"{family}:{session}",
                session_id=session,
                dataset_family=family,
                phone={
                    "file": phone_path,
                    "segment": str(phone["segment_id"].iloc[0]),
                    "time": phone["time_of_day_s"].to_numpy(dtype="float64"),
                    "speed": phone["gnss_speed_mps"].to_numpy(dtype="float64"),
                },
                vehicle={
                    "file": vehicle_path,
                    "segment": str(vehicle["segment_id"].iloc[0]),
                    "time": vehicle["time_of_day_s"].to_numpy(dtype="float64"),
                    "speed": vehicle["ref_speed_mps"].to_numpy(dtype="float64"),
                },
            )
            result.sync.append(sync)
            results[phone_path] = sync
        return results

    # -- stage 3: labels --------------------------------------------------
    def _label(
        self,
        result: PipelineResult,
        frames: dict[str, pd.DataFrame],
        sync_by_phone: dict[str, SyncResult],
    ) -> None:
        for path, frame in frames.items():
            if frame.empty or str(frame["stream"].iloc[0]) != "smartphone":
                continue
            sync = sync_by_phone.get(path)
            vehicle_time = vehicle_speed = None
            offset = None
            confirmed = False

            if sync is not None and sync.status != SyncStatus.UNRESOLVED:
                vehicle_frame = frames.get(sync.vehicle_file)
                if vehicle_frame is not None:
                    vehicle_time = vehicle_frame["time_of_day_s"].to_numpy(dtype="float64")
                    vehicle_speed = vehicle_frame["ref_speed_mps"].to_numpy(dtype="float64")
                    offset = sync.estimated_offset_s
                    confirmed = sync.status == SyncStatus.CONFIRMED_BY_SIGNAL

            labelled, coverage = build_velocity_labels(
                frame,
                segment_id=str(frame["segment_id"].iloc[0]),
                session_id=str(frame["session_id"].iloc[0]),
                source_file=path,
                vehicle_time_s=vehicle_time,
                vehicle_speed_mps=vehicle_speed,
                sync_offset_s=offset,
                sync_confirmed=confirmed,
            )
            if sync is not None and sync.status == SyncStatus.UNRESOLVED:
                flags = labelled["quality_flags"].to_numpy(dtype="int64") | (
                    QualityFlag.SYNC_UNRESOLVED.value
                )
                labelled["quality_flags"] = flags.astype("int32")
            frames[path] = labelled
            result.labels.append(coverage)

    # -- stage 4: write canonical ----------------------------------------
    def _write_canonical(self, result: PipelineResult, frames: dict[str, pd.DataFrame]) -> None:
        for stream in ("smartphone", "vehicle"):
            subset = [f for f in frames.values() if not f.empty and f["stream"].iloc[0] == stream]
            if not subset:
                continue
            out_dir = self.processed_path / stream
            out_dir.mkdir(parents=True, exist_ok=True)
            written = 0
            rows = 0
            for frame in subset:
                segment = str(frame["segment_id"].iloc[0])
                safe = segment.replace(":", "_")
                target = out_dir / f"{safe}.parquet"
                ordered = [spec.name for spec in schema_for(stream) if spec.name in frame.columns]
                extra = [c for c in frame.columns if c not in ordered]
                try:
                    frame[ordered + extra].to_parquet(target, index=False, compression="snappy")
                except (OSError, ValueError) as exc:
                    result.errors.append(
                        ProcessingError(
                            str(frame["source_file"].iloc[0]),
                            "write_parquet",
                            type(exc).__name__,
                            str(exc)[:400],
                            recoverable=True,
                            status="failed",
                        )
                    )
                    continue
                written += 1
                rows += len(frame)
            result.canonical_paths[stream] = str(out_dir)
            result.row_counts[stream] = rows
            logger.info("Wrote %d canonical %s file(s), %d rows", written, stream, rows)

    # -- stage 5: splits --------------------------------------------------
    def _split(self, result: PipelineResult, frames: dict[str, pd.DataFrame]) -> None:
        units = [
            SplitUnit(
                segment_id=segment.segment_id,
                payload_hash=segment.payload_hash,
                session_id=segment.session_id,
                source_file=segment.source_file,
                dataset_family=segment.dataset_family,
                stream=segment.stream,
                row_count=segment.row_count,
                duration_s=segment.duration_s if np.isfinite(segment.duration_s) else 0.0,
            )
            for segment in result.segments
        ]
        if not units:
            return
        assignments, summary = assign_splits(units, seed=self.seed)
        result.assignments = assignments
        result.split_summary = summary
        result.leakage_violations = verify_no_leakage(assignments)

        # A second, stricter pass: the payload hash only sees byte-identical
        # data rows, so two copies of one recording written with different
        # numeric formatting would look distinct to it. Fingerprint the parsed
        # sensor values instead and confirm no such pair straddles a split.
        fingerprints = {
            str(frame["segment_id"].iloc[0]): content_fingerprint(frame)
            for frame in frames.values()
            if not frame.empty
        }
        audit = audit_content_leakage(assignments, fingerprints)
        result.content_audit = audit
        result.leakage_violations.extend(audit.violations)
        logger.info(
            "Content audit: %d segment(s), %d distinct content group(s), "
            "%d duplicated beyond the payload hash, %d violation(s)",
            audit.segments_fingerprinted,
            audit.distinct_content_groups,
            audit.duplicate_content_groups,
            len(audit.violations),
        )
        if result.leakage_violations:
            logger.error("Split leakage detected: %d violation(s)", len(result.leakage_violations))

    def run(self) -> PipelineResult:
        started = time.time()
        result = PipelineResult(
            run_id=uuid.uuid4().hex[:16],
            started_at=datetime.now(UTC).isoformat(),
            raw_path=str(self.raw_path),
        )
        logger.info("Phase 2 pipeline starting on %s", self.raw_path)

        frames = self._canonicalize(result)
        if not frames:
            logger.error("No file produced a canonical frame")
            result.elapsed_s = time.time() - started
            return result

        sync_by_phone = self._synchronize(result, frames)
        self._label(result, frames, sync_by_phone)
        self._write_canonical(result, frames)
        self._split(result, frames)

        result.elapsed_s = time.time() - started
        logger.info("Phase 2 pipeline finished in %.1fs", result.elapsed_s)
        return result


# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------


def write_artifacts(
    result: PipelineResult, metadata_dir: Path, config: dict[str, Any]
) -> dict[str, str]:
    """Emit every Phase 2 machine-readable artifact."""
    metadata_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    def record(name: str, path: Path) -> None:
        written[name] = str(path)

    # canonical schema + field mapping
    schema_payload = {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "quality_flags": {flag.name: int(flag.value) for flag in QualityFlag if flag.name},
        "feature_availability_keys": list(FEATURE_AVAILABILITY_KEYS),
        "streams": {
            stream: [asdict(spec) for spec in schema_for(stream)]
            for stream in ("smartphone", "vehicle")
        },
    }
    path = metadata_dir / "canonical_schema.json"
    _write_json(path, schema_payload)
    record("canonical_schema.json", path)

    mapping_rows = []
    alias_note: dict[str, str] = {}
    for read in result.reads:
        for canonical_name, source_name in read.alias_resolved.items():
            alias_note.setdefault(canonical_name, source_name)
        mapping_rows.append(
            {
                "source_file": read.source_file,
                "stream": read.stream,
                "encoding": read.source_encoding,
                "decode_method": read.decode_method,
                "encoding_warning": read.encoding_warning or "",
                "repair_applied": read.repair.applied,
                "unmapped_columns": ";".join(read.unmapped_columns),
                "alias_resolved": ";".join(f"{k}<-{v}" for k, v in read.alias_resolved.items()),
                **{key: read.availability.get(key, False) for key in FEATURE_AVAILABILITY_KEYS},
            }
        )
    path = metadata_dir / "canonical_field_mapping.json"
    _write_json(
        path,
        {
            "alias_resolutions": alias_note,
            "note": (
                "Gyroscope axes are mapped by column position. 158 files label them "
                "Yaw/Pitch/Roll and 73 label the same positions X/Y/Z; two files holding "
                "the same recording under both headers were verified to have byte-identical "
                "data rows, so the labels are aliases. Device-frame X/Y/Z is canonical and "
                "the yaw/pitch/roll naming is NOT trusted as an axis semantic."
            ),
            "per_file": mapping_rows,
        },
    )
    record("canonical_field_mapping.json", path)

    # segments
    path = metadata_dir / "segment_inventory.csv"
    _write_csv(path, [asdict(segment) for segment in result.segments])
    record("segment_inventory.csv", path)

    # synchronization
    path = metadata_dir / "synchronization_results.csv"
    _write_csv(path, [asdict(sync) for sync in result.sync])
    record("synchronization_results.csv", path)

    # labels
    path = metadata_dir / "reference_label_inventory.csv"
    _write_csv(path, [asdict(label) for label in result.labels])
    record("reference_label_inventory.csv", path)

    # freshness
    path = metadata_dir / "gnss_freshness_report.csv"
    _write_csv(path, [asdict(report) for report in result.freshness])
    record("gnss_freshness_report.csv", path)

    # quality summary
    quality_rows = []
    for timing, read in zip(result.timing, result.reads, strict=False):
        quality_rows.append(
            {
                "source_file": timing.source_file,
                "stream": read.stream,
                "row_count": timing.row_count,
                "valid_timestamps": timing.valid_timestamps,
                "invalid_timestamps": timing.invalid_timestamps,
                "duplicate_timestamp_rows": timing.duplicate_timestamp_rows,
                "non_monotonic_rows": timing.non_monotonic_rows,
                "reordered": timing.reordered,
                "measured_rate_hz": timing.measured_rate_hz,
                "segment_count": timing.segment_count,
                "encoding": read.source_encoding,
                "encoding_warning": read.encoding_warning or "",
                "repair_applied": read.repair.applied,
                "satellites_consistent": read.satellites_consistent,
            }
        )
    path = metadata_dir / "quality_summary.csv"
    _write_csv(path, quality_rows)
    record("quality_summary.csv", path)

    # splits
    if result.split_summary:
        path = metadata_dir / "split_manifest.csv"
        _write_csv(path, [asdict(a) for a in result.assignments])
        record("split_manifest.csv", path)

        path = metadata_dir / "splits.json"
        path.write_text(splits_to_json(result.assignments, result.split_summary), encoding="utf-8")
        record("splits.json", path)

    # errors — always written, even when empty
    path = metadata_dir / "processing_errors.csv"
    _write_csv(path, [asdict(error) for error in result.errors])
    record("processing_errors.csv", path)

    # canonical dataset manifest
    manifest = {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "generator": f"idr {__version__} Phase 2",
        "raw_path": result.raw_path,
        "source_file_count": len(result.reads),
        "segment_count": len(result.segments),
        "row_counts": result.row_counts,
        "canonical_paths": result.canonical_paths,
        "storage_format": "parquet (snappy)",
        "encodings": _count(read.source_encoding for read in result.reads),
        "repairs_applied": sum(1 for read in result.reads if read.repair.applied),
        "files_with_encoding_warning": sum(1 for read in result.reads if read.encoding_warning),
    }
    path = metadata_dir / "canonical_dataset_manifest.json"
    _write_json(path, manifest)
    record("canonical_dataset_manifest.json", path)

    # processing run — the only artifact that carries wall-clock identity
    run = {
        "run_id": result.run_id,
        "timestamp": result.started_at,
        "code_version": __version__,
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dataset_source": result.raw_path,
        "configuration": config,
        "output_paths": result.canonical_paths,
        "row_counts": result.row_counts,
        "segment_counts": len(result.segments),
        "error_counts": len(result.errors),
        "warning_counts": sum(len(read.warnings) for read in result.reads),
        "elapsed_seconds": round(result.elapsed_s, 2),
        "leakage_violations": result.leakage_violations,
    }
    path = metadata_dir / "processing_run.json"
    _write_json(path, run)
    record("processing_run.json", path)

    # preprocessing report (machine-readable companion to the markdown report)
    sync_status = _count(sync.status for sync in result.sync)
    label_sources = _count(label.label_source for label in result.labels)
    flag_counts: dict[str, int] = defaultdict(int)
    for read in result.reads:
        for name in describe_flags(int(read.frame["quality_flags"].max() or 0)):
            flag_counts[name] += 0  # ensure key presence without overcounting
    preprocessing = {
        "files_processed": len(result.reads),
        "files_failed": len([e for e in result.errors if e.status == "failed"]),
        "segments": len(result.segments),
        "multi_segment_files": len({t.source_file for t in result.timing if t.segment_count > 1}),
        "reordered_files": len([t for t in result.timing if t.reordered]),
        "sync_status": sync_status,
        "label_sources": label_sources,
        "split_summary": asdict(result.split_summary) if result.split_summary else None,
        "leakage_violations": result.leakage_violations,
        "content_audit": asdict(result.content_audit) if result.content_audit else None,
    }
    path = metadata_dir / "preprocessing_report.json"
    _write_json(path, preprocessing)
    record("preprocessing_report.json", path)

    return written


def _count(values: Any) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[str(value)] += 1
    return dict(sorted(counts.items()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m idr.pipeline.runner",
        description="Phase 2 IO-VNBD canonicalization pipeline. Never modifies raw data.",
    )
    parser.add_argument("--raw-path", type=Path, default=Path("data/raw/io_vnbd"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed/io_vnbd"))
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/io_vnbd"))
    parser.add_argument("--report-output", type=Path, default=Path("reports/io_vnbd"))
    parser.add_argument("--seed", default=DEFAULT_SEED, help="Split assignment seed")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N files")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        pipeline = Phase2Pipeline(args.raw_path, args.processed, limit=args.limit, seed=args.seed)
    except NotADirectoryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    result = pipeline.run()
    config = {
        "raw_path": str(args.raw_path),
        "processed": str(args.processed),
        "seed": args.seed,
        "limit": args.limit,
    }
    written = write_artifacts(result, args.metadata, config)

    plots: list[Path] = []
    if not args.no_plots and result.reads:
        from idr.pipeline.plots import generate_phase2_plots

        plots = generate_phase2_plots(result, args.report_output / "phase2_plots")
        logger.info("Generated %d Phase 2 plot(s)", len(plots))

    if not args.no_report and result.reads:
        from idr.pipeline.report import generate_phase2_report

        report_path = generate_phase2_report(
            result, args.report_output / "IO_VNBD_Phase2_Preprocessing_Report.md", plots
        )
        logger.info("Wrote Phase 2 report to %s", report_path)

    failed = len([e for e in result.errors if e.status == "failed"])
    print(f"Phase 2 pipeline: {len(result.reads)} file(s) canonicalized")
    print(f"  segments={len(result.segments)}  errors={len(result.errors)} (failed={failed})")
    if result.split_summary:
        print(f"  splits={result.split_summary.counts}")
    print(f"  leakage violations={len(result.leakage_violations)}")
    print(f"  wrote {len(written)} artifact(s) to {args.metadata}")

    if not result.reads:
        return 1
    return 1 if result.leakage_violations else 0


if __name__ == "__main__":
    sys.exit(main())
