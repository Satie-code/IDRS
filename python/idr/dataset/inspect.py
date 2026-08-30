"""Dataset inspection orchestrator and CLI.

Usage::

    python -m idr.dataset.inspect --dataset-path data/raw/io_vnbd \\
        --output data/metadata/io_vnbd --report-output reports/io_vnbd --deep

Fast mode inventories files and extracts schemas from headers only. Deep mode
additionally scans every CSV once, computing timing, missingness, range, GNSS
and synchronization statistics from that single pass per file.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from idr.dataset.evidence import Evidence
from idr.dataset.inventory import (
    FileRecord,
    SessionRecord,
    build_file_inventory,
    canonical_paths,
    discover_sessions,
    find_duplicate_groups,
)
from idr.dataset.outages import OutageInterval, OutageReport, analyze_outages
from idr.dataset.quality import (
    MissingnessRecord,
    RangeFinding,
    SessionQuality,
    analyze_missingness,
    analyze_ranges,
    build_session_quality,
    quality_to_dict,
)
from idr.dataset.scan import StreamScan, scan_csv
from idr.dataset.schema import SchemaReport, read_schema
from idr.dataset.synchronization import SynchronizationReport, analyze_pair
from idr.dataset.timestamps import (
    SAMPLING_THRESHOLDS,
    TimestampReport,
    analyze_timestamps,
)
from idr.logging import get_logger
from idr.version import __version__

logger = get_logger(__name__)


@dataclass
class InspectionResult:
    """Everything Phase 1 established about the dataset."""

    dataset_path: str
    scan_timestamp: str
    deep: bool
    files: list[FileRecord] = field(default_factory=list)
    sessions: list[SessionRecord] = field(default_factory=list)
    schemas: dict[str, SchemaReport] = field(default_factory=dict)
    scans: dict[str, StreamScan] = field(default_factory=dict)
    timestamps: list[TimestampReport] = field(default_factory=list)
    missingness: list[MissingnessRecord] = field(default_factory=list)
    range_findings: list[RangeFinding] = field(default_factory=list)
    synchronization: list[SynchronizationReport] = field(default_factory=list)
    outage_reports: list[OutageReport] = field(default_factory=list)
    outage_intervals: list[OutageInterval] = field(default_factory=list)
    quality: list[SessionQuality] = field(default_factory=list)
    unreadable_files: list[str] = field(default_factory=list)
    parse_failures: list[str] = field(default_factory=list)
    #: {sha256: [relative_path, ...]} for content duplicated across the tree.
    duplicate_groups: dict[str, list[str]] = field(default_factory=dict)
    #: One representative path per distinct content hash.
    canonical_files: set[str] = field(default_factory=set)


class DatasetInspector:
    """Runs the Phase 1 inspection pipeline over a dataset tree.

    Read-only: this class never writes into ``dataset_path``.
    """

    def __init__(self, dataset_path: Path, *, deep: bool = False, chunk_size: int = 50_000):
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        if not dataset_path.is_dir():
            raise NotADirectoryError(f"Dataset path is not a directory: {dataset_path}")
        self.dataset_path = dataset_path
        self.deep = deep
        self.chunk_size = chunk_size

    def run(self) -> InspectionResult:
        result = InspectionResult(
            dataset_path=str(self.dataset_path),
            scan_timestamp=datetime.now(UTC).isoformat(),
            deep=self.deep,
        )

        logger.info("Building file inventory for %s", self.dataset_path)
        result.files = build_file_inventory(self.dataset_path)
        result.sessions = discover_sessions(result.files)
        result.duplicate_groups = find_duplicate_groups(result.files)
        result.canonical_files = canonical_paths(result.files)
        if result.duplicate_groups:
            redundant = sum(len(p) - 1 for p in result.duplicate_groups.values())
            logger.warning(
                "%d duplicate content group(s), %d redundant copy/copies — "
                "a file-level split would leak data between train and test",
                len(result.duplicate_groups),
                redundant,
            )
        logger.info(
            "Found %d file(s), %d session(s)",
            sum(1 for f in result.files if f.is_file),
            len(result.sessions),
        )

        csv_records = [
            record
            for record in result.files
            if record.is_file and record.extension == ".csv" and not record.is_lfs_pointer
        ]

        for index, record in enumerate(csv_records, start=1):
            path = self.dataset_path / record.relative_path
            try:
                if self.deep:
                    scan = scan_csv(path, record.relative_path, chunk_size=self.chunk_size)
                    result.scans[record.relative_path] = scan
                    schema = scan.schema
                    if scan.parse_error:
                        result.parse_failures.append(f"{record.relative_path}: {scan.parse_error}")
                else:
                    schema = read_schema(path, record.relative_path)
            except (OSError, ValueError, UnicodeDecodeError) as exc:
                result.unreadable_files.append(f"{record.relative_path}: {exc}")
                continue

            result.schemas[record.relative_path] = schema
            if index % 50 == 0 or index == len(csv_records):
                logger.info("Processed %d/%d CSV file(s)", index, len(csv_records))

        if self.deep:
            self._analyze_deep(result)
        return result

    def _analyze_deep(self, result: InspectionResult) -> None:
        stream_of: dict[str, str] = {}
        session_of: dict[str, str] = {}
        for record in result.files:
            if record.is_file and record.stream and record.session_id:
                stream_of[record.relative_path] = record.stream
                session_of[record.relative_path] = record.session_id

        for relative_path, scan in result.scans.items():
            timestamps = analyze_timestamps(scan)
            result.timestamps.append(timestamps)
            result.missingness.extend(analyze_missingness(scan))

            findings = analyze_ranges(scan)
            result.range_findings.extend(findings)

            session_id = session_of.get(relative_path, "unknown")
            stream = stream_of.get(relative_path, "unknown")

            outage_report, intervals = analyze_outages(scan, session_id, stream)
            result.outage_reports.append(outage_report)
            result.outage_intervals.extend(intervals)

            result.quality.append(
                build_session_quality(scan, timestamps, session_id, stream, findings)
            )

        for session in result.sessions:
            if not session.is_paired:
                continue
            phone = result.scans.get(session.smartphone_files[0])
            vehicle = result.scans.get(session.vehicle_files[0])
            if phone is None or vehicle is None:
                continue
            result.synchronization.append(
                analyze_pair(session.session_id, session.dataset_family, phone, vehicle)
            )


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # An explanatory stub beats a zero-byte file that looks like a bug.
        path.write_text("# no rows produced by this analysis\n", encoding="utf-8")
        return
    names = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _integrity_status(result: InspectionResult) -> tuple[str, list[str]]:
    reasons: list[str] = []
    pointer_count = sum(1 for f in result.files if f.is_lfs_pointer)
    data_files = [f for f in result.files if f.is_file and f.extension == ".csv"]

    if not data_files:
        return "FAIL", ["no CSV data files found under the dataset path"]
    if pointer_count == len(data_files):
        return "FAIL", ["every CSV is an unresolved Git LFS pointer; no actual data present"]
    if pointer_count:
        reasons.append(f"{pointer_count} unresolved Git LFS pointer file(s)")
    if result.unreadable_files:
        reasons.append(f"{len(result.unreadable_files)} unreadable file(s)")
    if result.parse_failures:
        reasons.append(f"{len(result.parse_failures)} CSV parse failure(s)")

    non_utf8 = sum(1 for s in result.schemas.values() if not s.encoding_is_utf8)
    if non_utf8:
        reasons.append(f"{non_utf8} file(s) are not valid UTF-8 (cp1252 fallback used)")

    status = "PASS_WITH_WARNINGS" if reasons else "PASS"
    return status, reasons or ["no integrity problems detected"]


def write_artifacts(
    result: InspectionResult, metadata_dir: Path, report_dir: Path
) -> dict[str, str]:
    """Write every machine-readable artifact. Returns ``{name: path}``."""
    metadata_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    def record(name: str, path: Path) -> None:
        written[name] = str(path)

    # --- dataset manifest -------------------------------------------------
    families: dict[str, int] = defaultdict(int)
    for file_record in result.files:
        if file_record.is_file:
            families[file_record.dataset_family] += 1

    manifest = {
        "generator": f"idr {__version__} (Phase 1 dataset inspection)",
        "dataset_path": result.dataset_path,
        "scan_timestamp": result.scan_timestamp,
        "deep_scan": result.deep,
        "file_count": sum(1 for f in result.files if f.is_file),
        "directory_count": sum(1 for f in result.files if f.is_directory),
        "csv_count": sum(1 for f in result.files if f.is_file and f.extension == ".csv"),
        "lfs_pointer_count": sum(1 for f in result.files if f.is_lfs_pointer),
        "dataset_families": dict(sorted(families.items())),
        "distinct_csv_content_count": len(result.canonical_files),
        "duplicate_group_count": len(result.duplicate_groups),
        "redundant_copy_count": sum(len(p) - 1 for p in result.duplicate_groups.values()),
        "session_count": len(result.sessions),
        "paired_session_count": sum(1 for s in result.sessions if s.is_paired),
        "sampling_thresholds": SAMPLING_THRESHOLDS,
        "files": [asdict(f) for f in result.files if f.is_file],
    }
    path = metadata_dir / "dataset_manifest.json"
    _write_json(path, manifest)
    record("dataset_manifest.json", path)

    # --- integrity --------------------------------------------------------
    status, reasons = _integrity_status(result)
    schema_ids = {s.schema_id for s in result.schemas.values()}
    integrity = {
        "dataset_path": result.dataset_path,
        "scan_timestamp": result.scan_timestamp,
        "file_count": sum(1 for f in result.files if f.is_file),
        "data_file_count": sum(
            1 for f in result.files if f.is_file and f.extension == ".csv" and not f.is_lfs_pointer
        ),
        "lfs_pointer_count": sum(1 for f in result.files if f.is_lfs_pointer),
        "unreadable_files": result.unreadable_files,
        "parse_failures": result.parse_failures,
        "schema_variants": len(schema_ids),
        "missing_expected_files": (
            "UNKNOWN — the dataset publishes no manifest of expected files, so "
            "absence cannot be distinguished from never having existed"
        ),
        "integrity_status": status,
        "reasons": reasons,
    }
    path = metadata_dir / "integrity_report.json"
    _write_json(path, integrity)
    record("integrity_report.json", path)

    # --- sessions ---------------------------------------------------------
    session_rows = [
        {
            "session_id": s.session_id,
            "dataset_family": s.dataset_family,
            "driver": s.driver or "",
            "driver_evidence": s.driver_evidence,
            "category": s.category or "",
            "country": s.country or "",
            "country_evidence": s.country_evidence,
            "smartphone_files": ";".join(s.smartphone_files),
            "vehicle_files": ";".join(s.vehicle_files),
            "is_paired": s.is_paired,
            "pairing_evidence": s.pairing_evidence,
        }
        for s in result.sessions
    ]
    path = metadata_dir / "session_inventory.csv"
    _write_csv(path, session_rows)
    record("session_inventory.csv", path)

    # --- schemas ----------------------------------------------------------
    by_schema: dict[str, list[str]] = defaultdict(list)
    for relative_path, schema in result.schemas.items():
        by_schema[schema.schema_id].append(relative_path)

    # Keep the file list in a parallel, properly typed map: the JSON payload
    # itself is dict[str, object], which loses the element type needed below.
    variants: list[dict[str, Any]] = []
    files_by_variant: dict[str, list[str]] = {}
    for schema_id, paths in sorted(by_schema.items()):
        files_by_variant[schema_id] = sorted(paths)
        example = result.schemas[paths[0]]
        variants.append(
            {
                "schema_id": schema_id,
                "file_count": len(paths),
                "column_count": example.column_count,
                "encoding": example.encoding,
                "encoding_is_utf8": example.encoding_is_utf8,
                "encoding_note": example.encoding_note,
                "example_file": paths[0],
                "files": sorted(paths),
                "columns": [asdict(c) for c in example.columns],
            }
        )
    path = metadata_dir / "schema_inventory.json"
    _write_json(
        path,
        {"schema_variant_count": len(variants), "variants": variants},
    )
    record("schema_inventory.json", path)

    # Split by stream so Phase 2 can read one file per stream type.
    for stream, filename in (("S-", "smartphone_schema.json"), ("V-", "vehicle_schema.json")):
        subset = [
            variant
            for variant in variants
            if any(
                Path(name).name.upper().startswith(stream)
                for name in files_by_variant[variant["schema_id"]]
            )
        ]
        path = metadata_dir / filename
        _write_json(
            path,
            {
                "stream": "smartphone" if stream == "S-" else "vehicle",
                "schema_variant_count": len(subset),
                "variants": subset,
            },
        )
        record(filename, path)

    # --- deep-only artifacts ---------------------------------------------
    path = metadata_dir / "timestamp_report.csv"
    _write_csv(path, [asdict(t) for t in result.timestamps])
    record("timestamp_report.csv", path)

    sampling_rows = [
        {
            "relative_path": t.relative_path,
            "time_column_role": t.time_column_role,
            "measured_rate_hz": t.estimated_rate_hz,
            "rate_kind": t.rate_kind,
            "delta_median_s": t.delta_median_s,
            "delta_mean_s": t.delta_mean_s,
            "delta_std_s": t.delta_std_s,
            "delta_min_s": t.delta_min_s,
            "delta_max_s": t.delta_max_s,
            "coefficient_of_variation": t.coefficient_of_variation,
            "sampling_class": t.sampling_class,
            "evidence": t.evidence,
        }
        for t in result.timestamps
    ]
    path = metadata_dir / "sampling_report.csv"
    _write_csv(path, sampling_rows)
    record("sampling_report.csv", path)

    path = metadata_dir / "synchronization_report.csv"
    _write_csv(path, [asdict(s) for s in result.synchronization])
    record("synchronization_report.csv", path)

    path = metadata_dir / "gnss_outages.csv"
    _write_csv(path, [asdict(i) for i in result.outage_intervals])
    record("gnss_outages.csv", path)

    path = metadata_dir / "gnss_availability.csv"
    _write_csv(path, [asdict(o) for o in result.outage_reports])
    record("gnss_availability.csv", path)

    path = metadata_dir / "missingness_report.csv"
    _write_csv(path, [asdict(m) for m in result.missingness])
    record("missingness_report.csv", path)

    path = metadata_dir / "range_findings.csv"
    _write_csv(path, [asdict(f) for f in result.range_findings])
    record("range_findings.csv", path)

    path = metadata_dir / "session_quality.csv"
    _write_csv(path, [quality_to_dict(q) for q in result.quality])
    record("session_quality.csv", path)

    # --- duplicates -------------------------------------------------------
    duplicate_rows = []
    for digest, paths in result.duplicate_groups.items():
        for index, relative in enumerate(paths):
            duplicate_rows.append(
                {
                    "sha256": digest,
                    "copy_count": len(paths),
                    "relative_path": relative,
                    "is_canonical": relative in result.canonical_files,
                    "duplicate_index": index,
                }
            )
    path = metadata_dir / "duplicate_files.csv"
    _write_csv(path, duplicate_rows)
    record("duplicate_files.csv", path)

    # --- scenarios --------------------------------------------------------
    scenario_rows = []
    for session in result.sessions:
        scenario_rows.append(
            {
                "scenario_id": session.category or "uncategorized",
                "name": session.category or "",
                "session": session.session_id,
                "driver": session.driver or "",
                "country": session.country or "",
                "dataset_family": session.dataset_family,
                "source": "directory_structure",
                "verification_status": (
                    Evidence.VERIFIED_FROM_FILE if session.category else Evidence.UNVERIFIED
                ),
            }
        )
    path = metadata_dir / "scenario_inventory.csv"
    _write_csv(path, scenario_rows)
    record("scenario_inventory.csv", path)

    report_dir.mkdir(parents=True, exist_ok=True)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m idr.dataset.inspect",
        description=(
            "Inspect an IO-VNBD dataset tree and emit forensic metadata. "
            "Read-only: never modifies the dataset."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Root of the dataset tree to inspect (e.g. data/raw/io_vnbd)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/metadata/io_vnbd"),
        help="Directory for machine-readable metadata (default: %(default)s)",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("reports/io_vnbd"),
        help="Directory for the human-readable report and plots (default: %(default)s)",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Scan row data (timing, missingness, ranges, GNSS, sync), not just headers",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="Rows per CSV read chunk, bounding memory use (default: %(default)s)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip diagnostic plot generation (deep mode only)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        inspector = DatasetInspector(args.dataset_path, deep=args.deep, chunk_size=args.chunk_size)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    result = inspector.run()
    written = write_artifacts(result, args.output, args.report_output)

    plots: list[Path] = []
    if args.deep and not args.no_plots:
        from idr.dataset.report import generate_plots

        plots = generate_plots(result, args.report_output / "plots")
        logger.info("Generated %d diagnostic plot(s)", len(plots))

    if args.deep:
        from idr.dataset.forensic_report import generate_forensic_report

        report_path = generate_forensic_report(
            result, args.report_output / "IO_VNBD_Forensic_Report.md", plots
        )
        logger.info("Wrote forensic report to %s", report_path)

    status, reasons = _integrity_status(result)
    print(f"Inspected {result.dataset_path}")
    print(f"  files={sum(1 for f in result.files if f.is_file)} sessions={len(result.sessions)}")
    print(f"  integrity={status}: {'; '.join(reasons)}")
    print(f"  wrote {len(written)} metadata artifact(s) to {args.output}")

    return 0 if status != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
