"""Phase 4 baseline CLI.

Usage::

    python -m idr.navigation.runner --processed data/processed/io_vnbd \\
        --metadata data/metadata/io_vnbd \\
        --report-output reports/navigation

Reads Phase 2 canonical Parquet, runs the Phase 3 chain and the Phase 4
mechanization over it, and writes a validation report, diagnostic figures and
machine-readable artifacts. It writes **nothing** into ``data/raw/`` or
``data/processed/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from idr.config import LoggingConfig
from idr.frames.alignment import CalibrationStatus
from idr.logging import configure_logging, get_logger
from idr.navigation.diagnostics import (
    DiagnosticRun,
    SegmentEvaluation,
    performance_summary,
    run_diagnostics,
)
from idr.navigation.plots import (
    generate_run_figures,
    generate_segment_figures,
    plot_synthetic_recovery,
)
from idr.navigation.report import generate_phase4_report, summarize_run

logger = get_logger(__name__)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
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


def synthetic_recovery() -> dict[str, float]:
    """Run every analytical fixture and return the worst position error of each.

    Executed by the CLI so the report's accuracy claim is regenerated from the
    current code on every run, rather than quoted from a number that was true
    once.
    """
    from idr.frames import quaternion as quat
    from idr.navigation import synthetic as syn
    from idr.navigation.mechanization import (
        AlignmentPolicy,
        AttitudeMode,
        MechanizationConfig,
    )
    from idr.navigation.trajectory import mechanize_series

    def worst(drive: syn.SyntheticTrajectory, mode: AttitudeMode) -> float:
        trajectory = mechanize_series(
            initial=drive.initial_state(),
            specific_force=drive.specific_force_phone,
            angular_rate=drive.angular_rate_phone,
            orientations=drive.orientation,
            config=MechanizationConfig(
                attitude_mode=mode,
                alignment_policy=AlignmentPolicy.PHONE_FRAME_DIAGNOSTIC,
            ),
        )
        count = min(len(trajectory), len(drive))
        error = np.linalg.norm(trajectory.position_m[:count] - drive.position_nav_m[:count], axis=1)
        finite = error[np.isfinite(error)]
        return float(finite.max()) if finite.size else float("nan")

    tilted = quat.from_euler(0.2, -0.35, 1.1)
    mount = quat.from_euler(np.radians(20.0), np.radians(-35.0), np.radians(110.0))
    filtered = AttitudeMode.PHASE3_FILTER

    return {
        "stationary (gravity-sign regression)": worst(syn.stationary(duration_s=60.0), filtered),
        "constant 1 m/s² acceleration": worst(
            syn.constant_acceleration(acceleration_mps2=1.0, duration_s=60.0), filtered
        ),
        "constant 15 m/s velocity": worst(
            syn.constant_velocity(speed_mps=15.0, duration_s=60.0), filtered
        ),
        "pure rotation, gyro propagated": worst(
            syn.pure_rotation(rate_rps=0.3, duration_s=60.0, rate_hz=50.0),
            AttitudeMode.GYRO_PROPAGATION,
        ),
        "tilted phone, known mounting": worst(
            syn.constant_acceleration(
                acceleration_mps2=1.2,
                duration_s=60.0,
                q_navigation_phone=tilted,
                q_vehicle_phone=mount,
            ),
            filtered,
        ),
        "irregular sampling": worst(
            syn.analytic_trajectory(
                times=syn.irregular_times(duration_s=60.0, rate_hz=10.0, jitter_s=0.03),
                acceleration_nav_mps2=[0.8, 0.3, 0.0],
            ),
            filtered,
        ),
    }


def write_artifacts(run: DiagnosticRun, metadata_dir: Path) -> list[Path]:
    """Emit the machine-readable Phase 4 artifacts."""
    written: list[Path] = []

    rows = [evaluation.row() for evaluation in run.evaluations]
    path = metadata_dir / "navigation_baseline_diagnostics.csv"
    _write_csv(path, rows)
    written.append(path)

    sensitivity_rows: list[dict[str, Any]] = []
    for result in run.bias_sensitivity:
        for point in result.points:
            sensitivity_rows.append(
                {
                    "segment_id": result.segment_id,
                    "duration_s": round(result.duration_s, 2),
                    "axis": point.axis,
                    "attitude_mode": point.attitude_mode,
                    "magnitude": point.magnitude,
                    "divergence_from_baseline_m": round(point.divergence_from_baseline_m, 3),
                    "final_horizontal_error_m": (
                        round(point.final_horizontal_error_m, 2)
                        if point.final_horizontal_error_m is not None
                        else ""
                    ),
                    "path_length_m": round(point.path_length_m, 1),
                }
            )
    path = metadata_dir / "navigation_bias_sensitivity.csv"
    _write_csv(path, sensitivity_rows)
    written.append(path)

    path = metadata_dir / "phase4_run.json"
    _write_json(path, summarize_run(run))
    written.append(path)
    return written


def choose_examples(run: DiagnosticRun, count: int = 3) -> list[SegmentEvaluation]:
    """Pick per-segment figure subjects to span the outcomes, not to flatter.

    One well-aligned segment, one weakly aligned, one with no alignment at all —
    selected by alignment status and duration, both fixed before any error
    metric is looked at.
    """
    usable = [item for item in run.evaluations if item.trajectory is not None]
    if not usable:
        return []
    chosen: list[SegmentEvaluation] = []

    def longest(predicate) -> SegmentEvaluation | None:  # type: ignore[no-untyped-def]
        candidates = [item for item in usable if predicate(item) and item not in chosen]
        return max(candidates, key=lambda item: item.duration_s) if candidates else None

    for predicate in (
        lambda item: (
            item.alignment_status is CalibrationStatus.SUCCESS and item.alignment_confidence >= 0.25
        ),
        lambda item: item.alignment_available and item.alignment_confidence < 0.25,
        lambda item: not item.alignment_available,
    ):
        pick = longest(predicate)
        if pick is not None:
            chosen.append(pick)
    while len(chosen) < count and len(chosen) < len(usable):
        remaining = [item for item in usable if item not in chosen]
        chosen.append(max(remaining, key=lambda item: item.duration_s))
    return chosen[:count]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m idr.navigation.runner",
        description="Run the Phase 4 inertial mechanization baseline over canonical data.",
    )
    parser.add_argument(
        "--processed",
        type=Path,
        default=Path("data/processed/io_vnbd"),
        help="Directory of Phase 2 canonical Parquet segments.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/metadata/io_vnbd"),
        help="Directory for machine-readable artifacts.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("reports/navigation"),
        help="Directory for the validation report and figures.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Process only the first N segments."
    )
    parser.add_argument(
        "--sensitivity-segments",
        type=int,
        default=3,
        help="How many segments to run the bias-sensitivity experiment on.",
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
        args.metadata,
        limit=args.limit,
        sensitivity_segments=args.sensitivity_segments,
    )
    if not run.evaluations:
        logger.error("No segments were propagated under %s", args.processed)
        for note in run.notes:
            logger.error("  %s", note)
        return 2

    artifacts = write_artifacts(run, args.metadata)
    examples = choose_examples(run)

    figures: list[Path] = []
    synthetic: dict[str, float] = {}
    if not args.no_plots:
        directory = args.report_output / "plots"
        figures = generate_run_figures(run, directory)
        for index, evaluation in enumerate(examples, start=1):
            figures.extend(generate_segment_figures(evaluation, directory, f"1{index}"))
        synthetic_cases = _synthetic_cases()
        recovered = plot_synthetic_recovery(
            synthetic_cases, directory / "07_synthetic_recovery.png"
        )
        if recovered is not None:
            figures.append(recovered)
        logger.info("Generated %d Phase 4 figure(s)", len(figures))

    if not args.no_report:
        synthetic = synthetic_recovery()
        report = generate_phase4_report(
            run,
            args.report_output / "Phase4_Inertial_Mechanization_Validation.md",
            figures=figures,
            synthetic_results=synthetic,
            example_segments=examples,
        )
        logger.info("Wrote Phase 4 validation report to %s", report)

    summary = run.summary()
    performance = performance_summary(run)
    print(f"Phase 4 baseline: {summary['segments_processed']} segment(s) propagated")
    print(
        f"  with alignment={summary['segments_with_alignment']}  "
        f"without={summary['segments_without_alignment']}  "
        f"compared={summary['segments_compared']}"
    )
    print(
        f"  median speed RMSE: zero-lag={_show(summary['median_zero_lag_speed_rmse_mps'])} m/s  "
        f"lag-corrected={_show(summary['median_best_lag_speed_rmse_mps'])} m/s"
    )
    print(
        f"  median horizontal drift={_show(summary['median_position_drift_mps'])} m/s  "
        f"lag applied on {summary['segments_with_accepted_lag']} segment(s), "
        f"median |lag|={_show(summary['median_abs_lag_s'])}s"
    )
    if performance.get("segments"):
        print(
            f"  performance: mean {performance['mean_us_per_sample']:.0f} µs/sample, "
            f"p95 {performance['p95_us_per_sample']:.0f} µs/sample"
        )
    print(f"  wrote {len(artifacts)} artifact(s) and {len(figures)} figure(s)")
    return 0


def _synthetic_cases():  # type: ignore[no-untyped-def]
    """Analytical fixtures paired with their propagated result, for the figure."""
    from idr.navigation import synthetic as syn
    from idr.navigation.mechanization import (
        AlignmentPolicy,
        AttitudeMode,
        MechanizationConfig,
    )
    from idr.navigation.trajectory import mechanize_series

    cases = {
        "stationary": (syn.stationary(duration_s=60.0), AttitudeMode.PHASE3_FILTER),
        "constant acceleration": (
            syn.constant_acceleration(acceleration_mps2=1.0, duration_s=60.0),
            AttitudeMode.PHASE3_FILTER,
        ),
        "constant velocity": (
            syn.constant_velocity(speed_mps=15.0, duration_s=60.0),
            AttitudeMode.PHASE3_FILTER,
        ),
        "pure rotation": (
            syn.pure_rotation(rate_rps=0.3, duration_s=60.0, rate_hz=50.0),
            AttitudeMode.GYRO_PROPAGATION,
        ),
        "irregular sampling": (
            syn.analytic_trajectory(
                times=syn.irregular_times(duration_s=60.0, rate_hz=10.0, jitter_s=0.03),
                acceleration_nav_mps2=[0.8, 0.3, 0.0],
            ),
            AttitudeMode.PHASE3_FILTER,
        ),
    }
    out = {}
    for name, (drive, mode) in cases.items():
        out[name] = (
            drive,
            mechanize_series(
                initial=drive.initial_state(),
                specific_force=drive.specific_force_phone,
                angular_rate=drive.angular_rate_phone,
                orientations=drive.orientation,
                config=MechanizationConfig(
                    attitude_mode=mode,
                    alignment_policy=AlignmentPolicy.PHONE_FRAME_DIAGNOSTIC,
                ),
            ),
        )
    return out


def _show(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return "—" if value is None else str(value)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
