"""Phase 3 diagnostic CLI.

Usage::

    python -m idr.frames.runner --processed data/processed/io_vnbd \\
        --metadata data/metadata/io_vnbd \\
        --report-output reports/sensor_alignment

Reads Phase 2 canonical Parquet, runs the Phase 3 estimators over it, and
writes a validation report, diagnostic figures and machine-readable artifacts.
It writes **nothing** into ``data/raw/`` or ``data/processed/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from idr.config import LoggingConfig
from idr.frames.diagnostics import DiagnosticRun, performance_summary, run_diagnostics
from idr.frames.plots import generate_phase3_plots, sweep_recovery_errors
from idr.frames.report import generate_phase3_report, summarize_run
from idr.logging import configure_logging, get_logger

logger = get_logger(__name__)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # An explicit empty artifact beats a missing one.
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


def write_artifacts(run: DiagnosticRun, metadata_dir: Path) -> list[Path]:
    """Emit the machine-readable Phase 3 artifacts."""
    written: list[Path] = []

    rows: list[dict[str, Any]] = []
    for segment in run.segments:
        alignment = segment.alignment
        quality = alignment.quality if alignment else None
        orientation = segment.orientation
        rows.append(
            {
                "segment_id": segment.segment_id,
                "session_id": segment.session_id,
                "source_file": segment.source_file,
                "status": str(segment.status),
                "sample_count": segment.timing.sample_count,
                "duration_s": round(segment.timing.duration_s, 3),
                "measured_rate_hz": segment.timing.measured_rate_hz,
                "is_regular": segment.timing.is_regular,
                "duplicate_timestamp_rows": segment.timing.duplicate_timestamp_rows,
                "analysis_time_source": segment.timing.analysis_time_source,
                "gravity_method": str(segment.gravity.method),
                "gravity_confidence": round(segment.gravity.confidence, 4),
                "gravity_tilt_from_phone_z_deg": segment.gravity.tilt_from_phone_z_deg(),
                "magnetometer_weight": round(segment.magnetometer.weight, 4),
                "magnetometer_issues": segment.magnetometer.describe(),
                "yaw_support": str(orientation.yaw_support) if orientation else "",
                "orientation_mean_confidence": (
                    round(orientation.mean_confidence(), 4) if orientation else None
                ),
                "orientation_skipped_steps": orientation.skipped_steps if orientation else None,
                "speed_reference": segment.speed_reference.source,
                "speed_valid_fraction": round(segment.speed_reference.valid_fraction, 4),
                "alignment_confidence": round(alignment.confidence, 4) if alignment else 0.0,
                "yaw_resolved": alignment.yaw_resolved if alignment else False,
                "roll_deg": alignment.roll_deg if alignment else None,
                "pitch_deg": alignment.pitch_deg if alignment else None,
                "yaw_deg": alignment.yaw_deg if alignment else None,
                "forward_correlation": quality.forward_correlation if quality else None,
                "forward_lag_s": segment.forward_lag_s,
                "gravity_residual_deg": quality.gravity_residual_deg if quality else None,
                "longitudinal_correlation": (quality.longitudinal_correlation if quality else None),
                "lateral_correlation": quality.lateral_correlation if quality else None,
                "dynamic_sample_count": quality.dynamic_sample_count if quality else 0,
                "seconds_per_sample": segment.seconds_per_sample,
                "error": segment.error or "",
            }
        )

    path = metadata_dir / "sensor_frame_diagnostics.csv"
    _write_csv(path, rows)
    written.append(path)

    path = metadata_dir / "alignment_estimates.json"
    _write_json(
        path,
        {
            "note": (
                "Estimated phone->vehicle alignments. R_vehicle_phone maps phone-frame "
                "coordinates to vehicle-frame coordinates. rotation is null wherever yaw "
                "could not be resolved; there is no ground truth for these values."
            ),
            "quaternion_order": "xyzw",
            "alignments": [
                {
                    "segment_id": segment.segment_id,
                    "status": str(segment.status),
                    "q_vehicle_phone": (
                        segment.alignment.rotation.quaternion.tolist()
                        if segment.alignment and segment.alignment.rotation
                        else None
                    ),
                    "confidence": segment.alignment.confidence if segment.alignment else 0.0,
                    "quality": (segment.alignment.quality.as_dict() if segment.alignment else None),
                    "notes": segment.alignment.notes[:6] if segment.alignment else [],
                }
                for segment in run.segments
            ],
        },
    )
    written.append(path)

    path = metadata_dir / "phase3_run.json"
    _write_json(
        path,
        {
            "started_at": run.started_at,
            "processed_dir": run.processed_dir,
            "summary": summarize_run(run),
            "status_counts": run.status_counts(),
            "performance": performance_summary(run),
            "notes": run.notes,
        },
    )
    written.append(path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m idr.frames.runner",
        description="Phase 3 sensor-frame, orientation and alignment diagnostics.",
    )
    parser.add_argument(
        "--processed",
        type=Path,
        default=Path("data/processed/io_vnbd"),
        help="Phase 2 canonical Parquet root (read-only).",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/metadata/io_vnbd"),
        help="Where Phase 3 artifacts are written; also read for the Phase 2 manifest.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("reports/sensor_alignment"),
        help="Directory for the validation report and figures.",
    )
    parser.add_argument("--stream", default="smartphone", help="Canonical stream to analyse.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Analyse only the first N segments."
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation.")
    parser.add_argument("--no-report", action="store_true", help="Skip the markdown report.")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(LoggingConfig(level=args.log_level), force=True)

    if not args.processed.is_dir():
        logger.error(
            "Canonical data not found at %s; run the Phase 2 pipeline first", args.processed
        )
        return 2

    run = run_diagnostics(
        args.processed,
        metadata_dir=args.metadata,
        stream=args.stream,
        limit=args.limit,
    )
    if not run.segments:
        logger.error("No segments analysed under %s", args.processed)
        for note in run.notes:
            logger.error("  %s", note)
        return 2

    artifacts = write_artifacts(run, args.metadata)

    plots: list[Path] = []
    synthetic: list[tuple[float, float]] = []
    if not args.no_plots:
        synthetic = sweep_recovery_errors()
        plots = generate_phase3_plots(
            run, args.report_output / "plots", synthetic_results=synthetic
        )
        logger.info("Generated %d Phase 3 figure(s)", len(plots))

    if not args.no_report:
        if not synthetic:
            synthetic = sweep_recovery_errors()
        report = generate_phase3_report(
            run,
            args.report_output / "Phase3_Sensor_Frame_Validation.md",
            plots,
            synthetic_results=synthetic,
        )
        logger.info("Wrote Phase 3 validation report to %s", report)

    summary = summarize_run(run)
    print(f"Phase 3 diagnostics: {summary['segments_analyzed']} segment(s) analysed")
    print(f"  alignment outcomes={run.status_counts()}")
    print(
        f"  gravity available={summary['gravity_available']}  "
        f"magnetometer trusted={summary['magnetometer_trusted']}"
    )
    print(
        f"  residual lag corrected on {summary['segments_needing_lag_correction']} "
        f"segment(s), median |lag|={summary['median_abs_lag_s']:.1f}s"
    )
    print(f"  wrote {len(artifacts)} artifact(s) and {len(plots)} figure(s)")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())


__all__ = ["build_parser", "main", "write_artifacts"]
