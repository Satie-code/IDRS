"""End-to-end tests for the Phase 4 pass, artifacts, figures, report and CLI.

Built on a miniature canonical tree that mirrors what Phase 2 actually writes,
so the whole chain runs without the real 344 MB dataset present.

The assertion that matters most is :func:`test_a_full_run_leaves_every_input_untouched`.
§47 requires that raw and canonical data are never modified, and a hash of every
input before and after a full run is the only way to know that rather than
believe it.
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
from idr.frames.synthetic import stationary_only, synthetic_drive
from idr.navigation.diagnostics import (
    bias_sensitivity,
    build_reference,
    evaluate_segment,
    initial_state_for,
    performance_summary,
    prepare_segment,
    run_diagnostics,
)
from idr.navigation.geodesy import TangentPlane
from idr.navigation.mechanization import AttitudeMode
from idr.navigation.plots import generate_run_figures, generate_segment_figures
from idr.navigation.report import generate_phase4_report, summarize_run
from idr.navigation.runner import (
    build_parser,
    choose_examples,
    main,
    synthetic_recovery,
    write_artifacts,
)
from idr.navigation.state import (
    VELOCITY_FROM_REFERENCE,
    VELOCITY_ZERO_STATIONARY,
    NavigationFlag,
)

# A small tangent-plane origin so the fixture carries a position reference.
_ORIGIN_LAT = 51.5074
_ORIGIN_LON = -0.1278


def _canonical_frame(
    drive, *, segment_id: str, with_reference: bool = True, with_gnss: bool = True
) -> pd.DataFrame:
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
        frame["ref_heading_deg"] = np.zeros(count)
    if with_gnss:
        # A track heading north at the drive's own speed, so the position
        # reference is at least self-consistent with the speed reference.
        northing = np.cumsum(drive.speed_mps) * 0.1
        origin = TangentPlane(_ORIGIN_LAT, _ORIGIN_LON)
        frame["latitude"] = _ORIGIN_LAT + np.degrees(northing / origin.meridional_radius_m)
        frame["longitude"] = np.full(count, _ORIGIN_LON)
        frame["altitude_m"] = np.zeros(count)
    return frame


@pytest.fixture
def canonical_tree(tmp_path: Path) -> Path:
    """A miniature Phase 2 output: two moving drives and one stationary."""
    processed = tmp_path / "processed"
    smartphone = processed / "smartphone"
    smartphone.mkdir(parents=True)

    truth = quat.from_euler(math.radians(12), math.radians(-8), math.radians(48))
    for name, drive, reference in (
        (
            "aaa111222333_000",
            synthetic_drive(q_vehicle_phone=truth, driving_s=120.0, stationary_s=30.0),
            True,
        ),
        (
            "bbb444555666_000",
            synthetic_drive(
                q_vehicle_phone=truth, noise_mps2=0.1, driving_s=100.0, stationary_s=25.0
            ),
            True,
        ),
        ("ccc777888999_000", stationary_only(q_vehicle_phone=truth, stationary_s=60.0), False),
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


# --- the pass ---------------------------------------------------------------


def test_the_pass_runs_over_a_canonical_tree(canonical_tree: Path) -> None:
    run = run_diagnostics(canonical_tree, sensitivity_segments=1)
    assert len(run.evaluations) == 3
    assert all(item.error is None for item in run.evaluations)
    assert run.elapsed_s > 0.0
    summary = run.summary()
    assert summary["segments_processed"] == 3
    assert summary["segments_propagated"] == 3


def test_moving_segments_get_an_alignment_and_stationary_ones_do_not(
    canonical_tree: Path,
) -> None:
    """The Phase 3 refusal survives into Phase 4 rather than being papered over."""
    run = run_diagnostics(canonical_tree, sensitivity_segments=0)
    by_id = {item.segment_id: item for item in run.evaluations}
    assert by_id["aaa111222333:000"].alignment_available
    assert not by_id["ccc777888999:000"].alignment_available

    stationary = by_id["ccc777888999:000"]
    assert stationary.trajectory is not None
    assert not stationary.trajectory.is_navigation_grade
    assert (stationary.trajectory.flags[1:] & int(NavigationFlag.ALIGNMENT_UNAVAILABLE)).all()


def test_a_stationary_segment_does_not_wander(canonical_tree: Path) -> None:
    """The end-to-end gravity-sign check on data that went through Parquet.

    A segment that never moves must produce a trajectory that barely moves. The
    bound is loose because the fixture's orientation filter is doing real work,
    but a gravity-sign error would be off by kilometres, not metres.
    """
    inputs = prepare_segment(canonical_tree / "smartphone" / "ccc777888999_000.parquet")
    assert inputs is not None
    evaluation = evaluate_segment(inputs)
    assert evaluation.trajectory is not None
    assert evaluation.trajectory.displacement_m() < 50.0


def test_a_stationary_segment_initializes_velocity_from_evidence(
    canonical_tree: Path,
) -> None:
    inputs = prepare_segment(canonical_tree / "smartphone" / "ccc777888999_000.parquet")
    assert inputs is not None
    initial = initial_state_for(inputs)
    assert initial.velocity_source == VELOCITY_ZERO_STATIONARY
    assert float(np.linalg.norm(initial.velocity_mps)) == 0.0
    assert any("detected stationary" in note for note in initial.notes)


def test_a_moving_segment_initializes_from_the_reference(canonical_tree: Path) -> None:
    """§15: the reference velocity is read at the lag-corrected instant."""
    inputs = prepare_segment(canonical_tree / "smartphone" / "aaa111222333_000.parquet")
    assert inputs is not None
    # Force the non-stationary branch: a segment that never comes to rest has
    # no detected run for the opening check to find.
    inputs.stationarity.mask[:] = False
    inputs.stationarity.longest_span = None
    inputs.stationarity.longest_interval_s = 0.0
    initial = initial_state_for(inputs)
    assert initial.velocity_source == VELOCITY_FROM_REFERENCE
    assert any("residual synchronization offset" in note for note in initial.notes)


def test_gnss_is_used_for_initialization_only(canonical_tree: Path) -> None:
    """§14. The origin comes from GNSS; nothing else in the run reads it."""
    inputs = prepare_segment(canonical_tree / "smartphone" / "aaa111222333_000.parquet")
    assert inputs is not None
    assert inputs.origin is not None
    initial = initial_state_for(inputs)
    assert float(np.linalg.norm(initial.position_m)) == 0.0
    assert "GNSS" in initial.position_source

    evaluation = evaluate_segment(inputs)
    assert evaluation.trajectory is not None
    assert evaluation.trajectory.origin == inputs.origin


def test_the_reference_track_records_its_provenance(canonical_tree: Path) -> None:
    inputs = prepare_segment(canonical_tree / "smartphone" / "aaa111222333_000.parquet")
    assert inputs is not None
    reference = build_reference(inputs.segment, inputs.origin)
    assert reference.speed_source == "ref_velocity_mps"
    assert reference.has_position
    assert reference.position_source == "gnss_fresh_fixes"
    assert not reference.is_ground_truth
    assert reference.heading_source.startswith("ref_heading_deg")


def test_a_reference_without_gnss_still_compares_on_speed(tmp_path: Path) -> None:
    processed = tmp_path / "processed" / "smartphone"
    processed.mkdir(parents=True)
    drive = synthetic_drive(driving_s=80.0, stationary_s=20.0)
    _canonical_frame(drive, segment_id="ddd:000", with_gnss=False).to_parquet(
        processed / "ddd_000.parquet", index=False
    )
    inputs = prepare_segment(processed / "ddd_000.parquet")
    assert inputs is not None
    assert inputs.origin is None
    assert not inputs.reference.has_position
    assert inputs.reference.has_speed


def test_short_segments_are_skipped_on_a_stated_criterion(tmp_path: Path) -> None:
    processed = tmp_path / "processed" / "smartphone"
    processed.mkdir(parents=True)
    drive = synthetic_drive(driving_s=5.0, stationary_s=1.0)
    _canonical_frame(drive, segment_id="eee:000").to_parquet(
        processed / "eee_000.parquet", index=False
    )
    assert prepare_segment(processed / "eee_000.parquet") is None
    run = run_diagnostics(tmp_path / "processed", sensitivity_segments=0)
    assert run.evaluations == []
    assert any("fewer than" in note for note in run.notes)


def test_the_comparison_reports_both_lags_on_real_shaped_data(
    canonical_tree: Path,
) -> None:
    run = run_diagnostics(canonical_tree, sensitivity_segments=0)
    for evaluation in run.evaluations:
        assert evaluation.comparison is not None
        assert evaluation.comparison.zero_lag.lag_s == 0.0
        assert "zero_lag" in evaluation.comparison.as_dict()
        assert "best_lag" in evaluation.comparison.as_dict()


def test_gyro_propagation_is_selectable_and_differs(canonical_tree: Path) -> None:
    """The two attitude sources are separate, and choosing one changes the answer."""
    inputs = prepare_segment(canonical_tree / "smartphone" / "aaa111222333_000.parquet")
    assert inputs is not None
    filtered = evaluate_segment(inputs, attitude_mode=AttitudeMode.PHASE3_FILTER)
    propagated = evaluate_segment(inputs, attitude_mode=AttitudeMode.GYRO_PROPAGATION)
    assert filtered.trajectory is not None and propagated.trajectory is not None
    assert not np.allclose(filtered.trajectory.position_m[-1], propagated.trajectory.position_m[-1])


def test_bias_sensitivity_produces_a_monotone_effect(canonical_tree: Path) -> None:
    """A larger perturbation must move the trajectory further, on every axis."""
    inputs = prepare_segment(canonical_tree / "smartphone" / "aaa111222333_000.parquet")
    assert inputs is not None
    result = bias_sensitivity(inputs, gyro_magnitudes=(0.002, 0.02), accel_magnitudes=(0.01, 0.1))
    assert result.points
    by_axis: dict[str, list[tuple[float, float]]] = {}
    for point in result.points:
        by_axis.setdefault(point.axis, []).append(
            (point.magnitude, point.divergence_from_baseline_m)
        )
    for axis, values in by_axis.items():
        ordered = sorted(values)
        assert ordered[-1][1] >= ordered[0][1], f"{axis} divergence must not shrink"


def test_performance_summary_reports_per_sample_cost(canonical_tree: Path) -> None:
    run = run_diagnostics(canonical_tree, sensitivity_segments=0)
    numbers = performance_summary(run)
    assert numbers["segments"] == 3.0
    assert numbers["mean_us_per_sample"] > 0.0
    assert numbers["p95_us_per_sample"] >= numbers["mean_us_per_sample"] * 0.5
    assert performance_summary(
        type(run)(evaluations=[], processed_dir=Path("."), started_at="", elapsed_s=0.0)
    ) == {"segments": 0.0}


# --- artifacts, figures, report ---------------------------------------------


def test_artifacts_are_written_and_parse(canonical_tree: Path, tmp_path: Path) -> None:
    run = run_diagnostics(canonical_tree, sensitivity_segments=1)
    metadata = tmp_path / "metadata"
    written = write_artifacts(run, metadata)
    assert len(written) == 3
    for path in written:
        assert path.is_file() and path.stat().st_size > 0

    payload = json.loads((metadata / "phase4_run.json").read_text(encoding="utf-8"))
    assert payload["summary"]["segments_processed"] == 3
    assert "alignment_status_counts" in payload
    assert "performance" in payload

    rows = (metadata / "navigation_baseline_diagnostics.csv").read_text(encoding="utf-8")
    assert "segment_id" in rows and "alignment_available" in rows
    assert "yaw_offset_removed_deg" in rows


def test_empty_artifacts_are_explicit_not_missing(tmp_path: Path) -> None:
    from idr.navigation.diagnostics import DiagnosticRun

    run = DiagnosticRun(evaluations=[], processed_dir=tmp_path, started_at="", elapsed_s=0.0)
    written = write_artifacts(run, tmp_path / "metadata")
    assert (tmp_path / "metadata" / "navigation_bias_sensitivity.csv").read_text(
        encoding="utf-8"
    ) == "# no rows\n"
    assert len(written) == 3


def test_figures_are_generated(canonical_tree: Path, tmp_path: Path) -> None:
    run = run_diagnostics(canonical_tree, sensitivity_segments=1)
    figures = generate_run_figures(run, tmp_path / "plots")
    assert figures
    for path in figures:
        assert path.is_file() and path.stat().st_size > 1000

    examples = choose_examples(run)
    assert examples
    per_segment = generate_segment_figures(examples[0], tmp_path / "plots", "11")
    assert per_segment


def test_examples_span_the_outcomes(canonical_tree: Path) -> None:
    """Figure subjects are chosen by status and duration, before any error is seen."""
    run = run_diagnostics(canonical_tree, sensitivity_segments=0)
    examples = choose_examples(run)
    assert len(examples) == 3
    assert any(not item.alignment_available for item in examples), (
        "a segment with no alignment must be shown, not filtered out"
    )


def test_the_report_is_written_and_says_the_honest_things(
    canonical_tree: Path, tmp_path: Path
) -> None:
    run = run_diagnostics(canonical_tree, sensitivity_segments=1)
    output = tmp_path / "Phase4.md"
    generate_phase4_report(
        run,
        output,
        figures=[tmp_path / "plots" / "01_dt_distribution.png"],
        synthetic_results={"stationary": 0.0, "constant acceleration": 1e-12},
        example_segments=run.evaluations[:1],
    )
    text = output.read_text(encoding="utf-8")

    for heading in (
        "## 1. Objective",
        "## 4. Synthetic validation",
        "## 6. Reference synchronization methodology",
        "## 7. Drift analysis",
        "## 8. Bias sensitivity",
        "## 10. Failure cases",
        "## 11. Limitations",
        "## 12. What Phase 5 requires",
    ):
        assert heading in text, f"missing {heading}"

    # The claims the phase brief insists on.
    assert "a = f + g" in text
    assert "reference rather than ground truth" in text or "reference" in text
    assert "lower bound" in text
    assert "gravity_parallel" in text
    assert "Not used" in text


def test_the_report_survives_an_empty_run(tmp_path: Path) -> None:
    from idr.navigation.diagnostics import DiagnosticRun

    run = DiagnosticRun(evaluations=[], processed_dir=tmp_path, started_at="", elapsed_s=0.0)
    output = tmp_path / "Phase4.md"
    generate_phase4_report(run, output)
    text = output.read_text(encoding="utf-8")
    assert "No synthetic results" in text
    assert "No figures" in text


def test_summarize_run_is_json_serializable(canonical_tree: Path) -> None:
    run = run_diagnostics(canonical_tree, sensitivity_segments=0)
    json.dumps(summarize_run(run), default=str)


def test_synthetic_recovery_is_regenerated_from_the_code() -> None:
    """The report's accuracy claim comes from a live run, not a quoted number."""
    results = synthetic_recovery()
    assert set(results) >= {
        "stationary (gravity-sign regression)",
        "constant 1 m/s² acceleration",
        "pure rotation, gyro propagated",
    }
    for name, error in results.items():
        assert error < 1e-6, f"{name} recovered {error:.3e} m of error"


# --- the CLI ----------------------------------------------------------------


def test_the_cli_runs_end_to_end(canonical_tree: Path, tmp_path: Path) -> None:
    code = main(
        [
            "--processed",
            str(canonical_tree),
            "--metadata",
            str(tmp_path / "metadata"),
            "--report-output",
            str(tmp_path / "reports"),
            "--sensitivity-segments",
            "1",
        ]
    )
    assert code == 0
    assert (tmp_path / "metadata" / "phase4_run.json").is_file()
    assert (tmp_path / "reports" / "Phase4_Inertial_Mechanization_Validation.md").is_file()
    assert list((tmp_path / "reports" / "plots").glob("*.png"))


def test_the_cli_can_skip_plots_and_report(canonical_tree: Path, tmp_path: Path) -> None:
    code = main(
        [
            "--processed",
            str(canonical_tree),
            "--metadata",
            str(tmp_path / "metadata"),
            "--report-output",
            str(tmp_path / "reports"),
            "--no-plots",
            "--no-report",
            "--sensitivity-segments",
            "0",
        ]
    )
    assert code == 0
    assert not (tmp_path / "reports").exists()


def test_the_cli_refuses_a_missing_directory(tmp_path: Path) -> None:
    assert main(["--processed", str(tmp_path / "absent")]) == 2


def test_the_cli_reports_an_empty_tree(tmp_path: Path) -> None:
    empty = tmp_path / "processed" / "smartphone"
    empty.mkdir(parents=True)
    assert main(["--processed", str(tmp_path / "processed")]) == 2


def test_the_parser_defaults_are_the_documented_paths() -> None:
    args = build_parser().parse_args([])
    assert args.processed == Path("data/processed/io_vnbd")
    assert args.metadata == Path("data/metadata/io_vnbd")
    assert args.report_output == Path("reports/navigation")


def test_the_limit_flag_is_respected(canonical_tree: Path, tmp_path: Path) -> None:
    run = run_diagnostics(canonical_tree, limit=1, sensitivity_segments=0)
    assert len(run.evaluations) == 1


# --- data safety (§47) ------------------------------------------------------


def test_a_full_run_leaves_every_input_untouched(canonical_tree: Path, tmp_path: Path) -> None:
    """The guarantee §47 demands, verified by hash rather than by intent.

    Phase 4 writes to ``reports/`` and its own metadata directory. It must never
    write into the canonical tree — not a modified Parquet, not a sidecar, not a
    cache file.
    """
    before = _hash_tree(canonical_tree)
    main(
        [
            "--processed",
            str(canonical_tree),
            "--metadata",
            str(tmp_path / "metadata"),
            "--report-output",
            str(tmp_path / "reports"),
            "--sensitivity-segments",
            "1",
        ]
    )
    after = _hash_tree(canonical_tree)
    assert before == after, "the canonical tree was modified by a Phase 4 run"


def test_no_new_files_appear_in_the_canonical_tree(canonical_tree: Path, tmp_path: Path) -> None:
    before = {path for path in canonical_tree.rglob("*")}
    run_diagnostics(canonical_tree, sensitivity_segments=1)
    assert {path for path in canonical_tree.rglob("*")} == before


def test_canonical_timestamps_are_never_shifted(canonical_tree: Path) -> None:
    """§26: the lag is measured, never written back into the data or the clock."""
    path = canonical_tree / "smartphone" / "aaa111222333_000.parquet"
    original = pd.read_parquet(path)["analysis_time_s"].to_numpy()

    inputs = prepare_segment(path)
    assert inputs is not None
    evaluation = evaluate_segment(inputs)
    assert evaluation.trajectory is not None

    # The trajectory runs on the canonical clock, unshifted...
    assert evaluation.trajectory.analysis_time_s == pytest.approx(original)
    # ...and the file itself is unchanged.
    assert pd.read_parquet(path)["analysis_time_s"].to_numpy() == pytest.approx(original)


def test_gyro_sensitivity_is_measured_under_gyro_propagation(canonical_tree: Path) -> None:
    """A gyroscope bias is inert in the baseline's attitude mode, so it is not
    measured there.

    In `PHASE3_FILTER` the attitude comes from the Phase 3 filter, computed
    upstream from the unperturbed gyroscope; Phase 4's bias is applied to the
    angular rate afterwards, where nothing consumes it. Reporting the resulting
    zero as "gyroscope bias does not matter" would be badly misleading, so
    gyroscope perturbations are run under `GYRO_PROPAGATION` instead. This test
    pins both halves: the mode label, and that the divergence is non-zero.
    """
    inputs = prepare_segment(canonical_tree / "smartphone" / "aaa111222333_000.parquet")
    assert inputs is not None
    result = bias_sensitivity(inputs, gyro_magnitudes=(0.02,), accel_magnitudes=(0.05,))

    gyro = [point for point in result.points if point.axis.startswith("gyro")]
    accel = [point for point in result.points if point.axis.startswith("accel")]
    assert gyro and accel
    assert all(point.attitude_mode == str(AttitudeMode.GYRO_PROPAGATION) for point in gyro)
    assert all(point.attitude_mode == str(AttitudeMode.PHASE3_FILTER) for point in accel)
    assert max(point.divergence_from_baseline_m for point in gyro) > 0.0
    assert any("structurally insensitive" in note for note in result.notes)


def test_a_gyro_bias_is_genuinely_inert_in_the_filtered_mode(canonical_tree: Path) -> None:
    """The claim the previous test is built on, asserted directly."""
    from idr.navigation.bias import GyroscopeBias, ImuBias, perturbed

    inputs = prepare_segment(canonical_tree / "smartphone" / "aaa111222333_000.parquet")
    assert inputs is not None
    baseline = evaluate_segment(inputs, estimate_bias=False, bias_override=ImuBias.zero())
    disturbed = evaluate_segment(
        inputs,
        estimate_bias=False,
        bias_override=perturbed(ImuBias.zero(), gyro_rps=np.array([0.05, 0.0, 0.0])),
    )
    assert baseline.trajectory is not None and disturbed.trajectory is not None
    assert disturbed.trajectory.position_m[-1] == pytest.approx(baseline.trajectory.position_m[-1])
    assert isinstance(GyroscopeBias.zero(), GyroscopeBias)


def test_sensitivity_segments_are_typical_not_longest(canonical_tree: Path) -> None:
    """Selection is by closeness to the median duration, and the run says so."""
    run = run_diagnostics(canonical_tree, sensitivity_segments=1)
    assert run.bias_sensitivity
    assert any("median duration" in note for note in run.notes)
