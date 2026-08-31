"""End-to-end tests for the Phase 3 diagnostic pass, figures, report and CLI.

Built on a synthetic canonical tree that mirrors what Phase 2 actually writes,
so the whole chain runs without the real 344 MB dataset present. The safety
assertion that matters most is the last one: a full run must leave every input
file byte-identical.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from idr.frames import quaternion as quat
from idr.frames.alignment import CalibrationStatus
from idr.frames.conventions import STANDARD_GRAVITY_MPS2
from idr.frames.diagnostics import (
    analyze_segment,
    choose_speed_reference,
    performance_summary,
    run_diagnostics,
)
from idr.frames.plots import generate_phase3_plots, sweep_recovery_errors
from idr.frames.report import generate_phase3_report, summarize_run
from idr.frames.runner import build_parser, main, write_artifacts
from idr.frames.sources import load_segment
from idr.frames.synthetic import stationary_only, synthetic_drive


def _canonical_frame(drive, *, segment_id: str, with_reference: bool = True) -> pd.DataFrame:
    """A frame with the columns Phase 2 actually writes."""
    count = len(drive)
    frame = pd.DataFrame(
        {
            "dataset_family": ["synthetic"] * count,
            "source_file": [f"synthetic/{segment_id}.csv"] * count,
            "session_id": [segment_id[:4]] * count,
            "segment_id": [segment_id] * count,
            "stream": ["smartphone"] * count,
            "analysis_time_s": drive.analysis_time_s,
            "analysis_time_source": ["wall_clock"] * count,
            "source_time_s": drive.analysis_time_s * 1000.0,
            "elapsed_s": drive.analysis_time_s - drive.analysis_time_s[0],
            "quality_flags": np.zeros(count, dtype="int32"),
            "gnss_speed_mps": drive.speed_mps,
            "gnss_position_fresh": np.ones(count, dtype=bool),
            "orientation_roll": np.zeros(count),
            "orientation_pitch": np.zeros(count),
            "orientation_yaw": np.zeros(count),
        }
    )
    for prefix, series in (
        ("accel", drive.accel_phone),
        ("gyro", drive.gyro_phone),
        ("mag", drive.mag_phone),
        ("gravity", drive.gravity_phone),
    ):
        for index, axis in enumerate("xyz"):
            frame[f"{prefix}_{axis}"] = series.samples[:, index]
    if with_reference:
        frame["ref_velocity_mps"] = drive.speed_mps
        frame["label_available"] = np.ones(count, dtype=bool)
        frame["label_source"] = ["TIER_1_VEHICLE_SYNCED"] * count
    return frame


@pytest.fixture
def canonical_tree(tmp_path: Path) -> Path:
    """A miniature Phase 2 output: two moving drives and one stationary."""
    processed = tmp_path / "processed"
    smartphone = processed / "smartphone"
    smartphone.mkdir(parents=True)

    truth = quat.from_euler(math.radians(12), math.radians(-8), math.radians(48))
    for name, drive, reference in (
        ("aaa111222333_000", synthetic_drive(q_vehicle_phone=truth), True),
        ("bbb444555666_000", synthetic_drive(q_vehicle_phone=truth, noise_mps2=0.1), True),
        ("ccc777888999_000", stationary_only(q_vehicle_phone=truth), False),
    ):
        _canonical_frame(
            drive, segment_id=name.replace("_000", ":000"), with_reference=reference
        ).to_parquet(smartphone / f"{name}.parquet", index=False)
    return processed


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


# --- the diagnostic pass ---------------------------------------------------


def test_the_diagnostic_pass_runs_over_a_canonical_tree(canonical_tree: Path) -> None:
    run = run_diagnostics(canonical_tree)
    assert len(run.segments) == 3
    assert all(segment.error is None for segment in run.segments)
    assert run.elapsed_s > 0.0


def test_moving_segments_align_and_stationary_ones_decline(canonical_tree: Path) -> None:
    run = run_diagnostics(canonical_tree)
    by_id = {segment.segment_id: segment for segment in run.segments}
    assert by_id["aaa111222333:000"].status is CalibrationStatus.SUCCESS
    assert by_id["ccc777888999:000"].status is CalibrationStatus.INSUFFICIENT_MOTION
    # Even the declining segment has a vertical.
    assert by_id["ccc777888999:000"].gravity.is_available


def test_the_recovered_alignment_matches_the_generated_truth(canonical_tree: Path) -> None:
    """The only accuracy assertion possible: the fixture's rotation is known."""
    truth = quat.from_euler(math.radians(12), math.radians(-8), math.radians(48))
    run = run_diagnostics(canonical_tree)
    successes = run.succeeded()
    assert successes
    for segment in successes:
        assert segment.alignment is not None and segment.alignment.rotation is not None
        error = math.degrees(quat.angular_distance(segment.alignment.rotation.quaternion, truth))
        assert error < 2.0, f"{segment.segment_id} recovered {error:.2f}° from truth"


def test_the_reference_velocity_is_preferred_over_gnss_speed(canonical_tree: Path) -> None:
    """GNSS speed refreshes only ~every 9 s; the vehicle reference is better."""
    segment = load_segment(canonical_tree / "smartphone" / "aaa111222333_000.parquet")
    reference = choose_speed_reference(segment)
    assert reference.source == "ref_velocity_mps"
    assert reference.valid_fraction > 0.9


def test_gnss_speed_is_the_fallback_when_no_reference_exists(canonical_tree: Path) -> None:
    segment = load_segment(canonical_tree / "smartphone" / "ccc777888999_000.parquet")
    reference = choose_speed_reference(segment)
    assert reference.source == "gnss_speed_mps"
    assert "refreshes only about every 9 s" in reference.note


def test_a_segment_with_no_usable_speed_reports_it(tmp_path: Path) -> None:
    drive = synthetic_drive()
    frame = _canonical_frame(drive, segment_id="ddd:000", with_reference=False)
    frame["gnss_speed_mps"] = np.nan
    path = tmp_path / "ddd_000.parquet"
    frame.to_parquet(path, index=False)
    segment = load_segment(path)
    reference = choose_speed_reference(segment)
    assert reference.source == "none"
    assert reference.values is None


def test_a_segment_without_an_accelerometer_is_reported_not_crashed(tmp_path: Path) -> None:
    drive = synthetic_drive()
    frame = _canonical_frame(drive, segment_id="eee:000")
    frame[["accel_x", "accel_y", "accel_z"]] = np.nan
    path = tmp_path / "eee_000.parquet"
    frame.to_parquet(path, index=False)
    diagnostic = analyze_segment(path)
    assert diagnostic.error is not None
    assert diagnostic.status is CalibrationStatus.FAILED


def test_the_dataset_orientation_comparison_is_computed(canonical_tree: Path) -> None:
    """Descriptive only — see the report. It must still be produced."""
    run = run_diagnostics(canonical_tree)
    with_rmse = [s for s in run.segments if s.dataset_orientation_rmse_deg]
    assert with_rmse
    rmse = with_rmse[0].dataset_orientation_rmse_deg
    assert rmse is not None
    assert "orientation_roll" in rmse


def test_an_empty_processed_directory_is_reported_not_crashed(tmp_path: Path) -> None:
    run = run_diagnostics(tmp_path)
    assert run.segments == []
    assert any("no canonical" in note for note in run.notes)


def test_a_missing_field_mapping_is_noted(canonical_tree: Path, tmp_path: Path) -> None:
    run = run_diagnostics(canonical_tree, metadata_dir=tmp_path / "absent")
    assert any("canonical_field_mapping" in note for note in run.notes)


def test_the_limit_is_applied_deterministically(canonical_tree: Path) -> None:
    first = run_diagnostics(canonical_tree, limit=2)
    second = run_diagnostics(canonical_tree, limit=2)
    assert len(first.segments) == 2
    assert [s.segment_id for s in first.segments] == [s.segment_id for s in second.segments]


def test_performance_is_measured(canonical_tree: Path) -> None:
    performance = performance_summary(run_diagnostics(canonical_tree))
    assert performance["mean_us_per_sample"] > 0.0
    assert performance["p95_us_per_sample"] >= performance["mean_us_per_sample"] * 0.5


def test_performance_summary_of_an_empty_run(tmp_path: Path) -> None:
    assert performance_summary(run_diagnostics(tmp_path)) == {}


# --- figures ---------------------------------------------------------------


def test_figures_are_produced_from_a_real_run(canonical_tree: Path, tmp_path: Path) -> None:
    run = run_diagnostics(canonical_tree)
    plots = generate_phase3_plots(
        run, tmp_path / "plots", synthetic_results=[(0.0, 0.1), (45.0, 0.2)]
    )
    assert plots
    for path in plots:
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_figures_decline_an_empty_run(tmp_path: Path) -> None:
    run = run_diagnostics(tmp_path)
    assert generate_phase3_plots(run, tmp_path / "plots") == []


def test_the_synthetic_sweep_recovers_every_rotation(tmp_path: Path) -> None:
    """The sweep the report quotes; run here so the numbers are checked."""
    results = sweep_recovery_errors()
    assert len(results) >= 5
    for angle, error in results:
        assert error < 3.0, f"{angle}° recovered with {error:.2f}° of error"


# --- report ----------------------------------------------------------------


def test_the_report_is_generated_with_every_section(canonical_tree: Path, tmp_path: Path) -> None:
    run = run_diagnostics(canonical_tree)
    output = generate_phase3_report(
        run,
        tmp_path / "Phase3_Sensor_Frame_Validation.md",
        [],
        synthetic_results=[(0.0, 0.05), (45.0, 0.12)],
    )
    text = output.read_text(encoding="utf-8")
    for heading in (
        "## 1. What this phase established",
        "## 2. Frame conventions",
        "## 3. Numerical tolerances",
        "## 4. Gravity estimation",
        "## 5. Orientation and the magnetometer",
        "## 6. Synthetic rotation recovery",
        "## 7. Phone→vehicle alignment on real data",
        "## 8. A finding: residual Phase 2 synchronization error",
        "## 9. Timing, sampling and duplicate handling",
        "## 10. Comparison with the dataset",
        "## 11. Performance",
        "## 12. Figures",
        "## 13. Limitations",
        "## 14. What Phase 4 may rely on",
    ):
        assert heading in text, f"missing {heading}"


def test_the_report_refuses_to_claim_ground_truth(canonical_tree: Path, tmp_path: Path) -> None:
    """The central honesty requirement of this phase."""
    run = run_diagnostics(canonical_tree)
    text = generate_phase3_report(
        run, tmp_path / "report.md", [], synthetic_results=[(0.0, 0.05)]
    ).read_text(encoding="utf-8")
    assert "no ground truth exists" in text
    assert "Neither series is ground truth for the other" in text
    assert "nothing in the real-data sections below is an accuracy claim" in text
    assert "Accuracy is established only against" in text


def test_the_run_summary_is_machine_readable(canonical_tree: Path) -> None:
    summary = summarize_run(run_diagnostics(canonical_tree))
    assert summary["segments_analyzed"] == 3
    assert summary["gravity_available"] == 3
    assert isinstance(summary["median_forward_correlation"], float)


# --- artifacts and CLI -----------------------------------------------------


def test_artifacts_are_written(canonical_tree: Path, tmp_path: Path) -> None:
    run = run_diagnostics(canonical_tree)
    written = write_artifacts(run, tmp_path / "metadata")
    assert {path.name for path in written} == {
        "sensor_frame_diagnostics.csv",
        "alignment_estimates.json",
        "phase3_run.json",
    }
    for path in written:
        assert path.stat().st_size > 0


def test_the_alignment_artifact_marks_unresolved_yaw_as_null(
    canonical_tree: Path, tmp_path: Path
) -> None:
    """Phase 4 must be able to see that no rotation exists, not a placeholder."""
    run = run_diagnostics(canonical_tree)
    write_artifacts(run, tmp_path / "metadata")
    payload = json.loads(
        (tmp_path / "metadata" / "alignment_estimates.json").read_text(encoding="utf-8")
    )
    assert payload["quaternion_order"] == "xyzw"
    by_status = {entry["segment_id"]: entry for entry in payload["alignments"]}
    stationary = by_status["ccc777888999:000"]
    assert stationary["q_vehicle_phone"] is None
    assert stationary["status"] == "INSUFFICIENT_MOTION"
    moving = by_status["aaa111222333:000"]
    assert moving["q_vehicle_phone"] is not None
    assert len(moving["q_vehicle_phone"]) == 4


def test_the_cli_runs_end_to_end(canonical_tree: Path, tmp_path: Path) -> None:
    exit_code = main(
        [
            "--processed",
            str(canonical_tree),
            "--metadata",
            str(tmp_path / "metadata"),
            "--report-output",
            str(tmp_path / "reports"),
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "metadata" / "phase3_run.json").is_file()
    assert (tmp_path / "reports" / "Phase3_Sensor_Frame_Validation.md").is_file()
    assert list((tmp_path / "reports" / "plots").glob("*.png"))


def test_the_cli_can_skip_plots_and_report(canonical_tree: Path, tmp_path: Path) -> None:
    exit_code = main(
        [
            "--processed",
            str(canonical_tree),
            "--metadata",
            str(tmp_path / "metadata"),
            "--report-output",
            str(tmp_path / "reports"),
            "--no-plots",
            "--no-report",
        ]
    )
    assert exit_code == 0
    assert not (tmp_path / "reports" / "Phase3_Sensor_Frame_Validation.md").exists()


def test_the_cli_fails_cleanly_without_canonical_data(tmp_path: Path) -> None:
    assert main(["--processed", str(tmp_path / "absent")]) == 2


def test_the_cli_fails_cleanly_on_an_empty_tree(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    assert main(["--processed", str(tmp_path / "empty")]) == 2


def test_the_parser_exposes_the_documented_options() -> None:
    args = build_parser().parse_args([])
    assert args.stream == "smartphone"
    assert args.limit is None
    assert args.processed == Path("data/processed/io_vnbd")


# --- data safety -----------------------------------------------------------


def test_phase3_does_not_modify_canonical_data(canonical_tree: Path, tmp_path: Path) -> None:
    """The central safety guarantee: Phase 3 is read-only over Phase 2 output."""
    before = _hash_tree(canonical_tree)
    main(
        [
            "--processed",
            str(canonical_tree),
            "--metadata",
            str(tmp_path / "metadata"),
            "--report-output",
            str(tmp_path / "reports"),
        ]
    )
    assert _hash_tree(canonical_tree) == before


def test_analysis_does_not_mutate_a_loaded_frame(canonical_tree: Path) -> None:
    path = canonical_tree / "smartphone" / "aaa111222333_000.parquet"
    segment = load_segment(path)
    before = segment.frame_data.copy(deep=True)
    analyze_segment(path)
    pd.testing.assert_frame_equal(load_segment(path).frame_data, before)


def test_gravity_is_available_on_every_generated_segment(canonical_tree: Path) -> None:
    """A guard on the fixture itself: if gravity failed here the alignment
    assertions above would pass vacuously."""
    run = run_diagnostics(canonical_tree)
    assert all(segment.gravity.is_available for segment in run.segments)
    assert all(
        segment.gravity.magnitude_mps2 == pytest.approx(STANDARD_GRAVITY_MPS2, abs=0.5)
        for segment in run.segments
    )
