from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from idr.pipeline.runner import Phase2Pipeline, main, write_artifacts


@pytest.fixture
def dataset_tree(
    tmp_path: Path, smartphone_csv: Path, vehicle_csv: Path, smartphone_with_gap: Path
) -> Path:
    """A miniature raw tree mirroring the real directory conventions."""
    root = tmp_path / "raw"
    session = (
        root / "Synchronised V abd S datasets" / "Categorised IOVNB Dataset" / "S (Driver A)" / "S1"
    )
    session.mkdir(parents=True)
    (session / "S-S1.csv").write_bytes(smartphone_csv.read_bytes())
    (session / "V-S1.csv").write_bytes(vehicle_csv.read_bytes())

    other = root / "Unsynchronised V and S Dataset" / "Uncategorised IOVNB (V and S) Dataset"
    other.mkdir(parents=True)
    (other / "S-G1.csv").write_bytes(smartphone_with_gap.read_bytes())
    return root


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_pipeline_runs_end_to_end(dataset_tree: Path, tmp_path: Path) -> None:
    result = Phase2Pipeline(dataset_tree, tmp_path / "processed").run()
    assert len(result.reads) == 3
    assert result.segments
    assert result.labels
    assert not [e for e in result.errors if e.status == "failed"]


def test_pipeline_does_not_modify_raw_data(dataset_tree: Path, tmp_path: Path) -> None:
    """The central safety guarantee: raw bytes are identical after a full run."""
    before = _hash_tree(dataset_tree)
    result = Phase2Pipeline(dataset_tree, tmp_path / "processed").run()
    write_artifacts(result, tmp_path / "metadata", {})
    assert _hash_tree(dataset_tree) == before


def test_canonical_parquet_is_written_and_readable(dataset_tree: Path, tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    Phase2Pipeline(dataset_tree, processed).run()
    files = list(processed.rglob("*.parquet"))
    assert files
    frame = pd.read_parquet(files[0])
    assert len(frame) > 0
    for column in ("source_time_s", "analysis_time_s", "elapsed_s", "segment_id", "quality_flags"):
        assert column in frame.columns


def test_parquet_round_trip_preserves_values(dataset_tree: Path, tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    result = Phase2Pipeline(dataset_tree, processed).run()
    written = sorted(processed.rglob("*.parquet"))
    assert written
    frame = pd.read_parquet(written[0])
    # Timestamps must survive as float64: rounding them would corrupt timing.
    assert frame["source_time_s"].dtype == "float64"
    assert frame["latitude"].dtype == "float64"
    assert result.row_counts


def test_all_expected_artifacts_are_written(dataset_tree: Path, tmp_path: Path) -> None:
    result = Phase2Pipeline(dataset_tree, tmp_path / "processed").run()
    metadata = tmp_path / "metadata"
    written = write_artifacts(result, metadata, {"seed": "t"})
    for name in (
        "canonical_schema.json",
        "canonical_field_mapping.json",
        "canonical_dataset_manifest.json",
        "preprocessing_report.json",
        "split_manifest.csv",
        "splits.json",
        "segment_inventory.csv",
        "synchronization_results.csv",
        "reference_label_inventory.csv",
        "gnss_freshness_report.csv",
        "quality_summary.csv",
        "processing_run.json",
        "processing_errors.csv",
    ):
        assert name in written, f"{name} missing"
        assert Path(written[name]).exists()


def test_error_artifact_exists_even_with_zero_errors(dataset_tree: Path, tmp_path: Path) -> None:
    result = Phase2Pipeline(dataset_tree, tmp_path / "processed").run()
    written = write_artifacts(result, tmp_path / "metadata", {})
    text = Path(written["processing_errors.csv"]).read_text(encoding="utf-8")
    assert text.strip()  # a stub, not an empty file


def test_split_has_no_leakage_on_real_tree(dataset_tree: Path, tmp_path: Path) -> None:
    result = Phase2Pipeline(dataset_tree, tmp_path / "processed").run()
    assert result.leakage_violations == []


def test_pipeline_is_deterministic(dataset_tree: Path, tmp_path: Path) -> None:
    """Same input and seed must give identical segments and split assignment."""
    first = Phase2Pipeline(dataset_tree, tmp_path / "p1", seed="fixed").run()
    second = Phase2Pipeline(dataset_tree, tmp_path / "p2", seed="fixed").run()

    assert [s.segment_id for s in first.segments] == [s.segment_id for s in second.segments]
    assert {a.segment_id: a.split for a in first.assignments} == {
        a.segment_id: a.split for a in second.assignments
    }


def test_run_id_is_not_part_of_data_identity(dataset_tree: Path, tmp_path: Path) -> None:
    """Two runs differ in run_id but must agree on every data-bearing value."""
    first = Phase2Pipeline(dataset_tree, tmp_path / "p1", seed="fixed").run()
    second = Phase2Pipeline(dataset_tree, tmp_path / "p2", seed="fixed").run()
    assert first.run_id != second.run_id
    assert [s.payload_hash for s in first.segments] == [s.payload_hash for s in second.segments]


def test_unreadable_file_is_recorded_not_skipped_silently(
    dataset_tree: Path, tmp_path: Path
) -> None:
    (dataset_tree / "S-BROKEN.csv").write_text("", encoding="utf-8")
    result = Phase2Pipeline(dataset_tree, tmp_path / "processed").run()
    assert any("S-BROKEN" in error.source_file for error in result.errors)


def test_unrecognized_filename_is_recorded(dataset_tree: Path, tmp_path: Path) -> None:
    (dataset_tree / "notes.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    result = Phase2Pipeline(dataset_tree, tmp_path / "processed").run()
    assert any(e.error_type == "UnrecognizedStream" for e in result.errors)


def test_cli_rejects_missing_raw_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--raw-path", str(tmp_path / "nope")])
    assert code == 2
    assert "not a directory" in capsys.readouterr().err


def test_cli_succeeds_on_valid_tree(dataset_tree: Path, tmp_path: Path) -> None:
    code = main(
        [
            "--raw-path",
            str(dataset_tree),
            "--processed",
            str(tmp_path / "proc"),
            "--metadata",
            str(tmp_path / "meta"),
            "--report-output",
            str(tmp_path / "rep"),
            "--no-plots",
            "--no-report",
        ]
    )
    assert code == 0
    manifest = json.loads(
        (tmp_path / "meta" / "canonical_dataset_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["source_file_count"] == 3


def test_gapped_file_produces_multiple_segments_in_pipeline(
    dataset_tree: Path, tmp_path: Path
) -> None:
    result = Phase2Pipeline(dataset_tree, tmp_path / "processed").run()
    gapped = [s for s in result.segments if "S-G1" in s.source_file]
    assert len(gapped) == 2


def test_segments_of_one_file_share_a_split(dataset_tree: Path, tmp_path: Path) -> None:
    """Both trips of a multi-segment file must land in the same partition."""
    result = Phase2Pipeline(dataset_tree, tmp_path / "processed").run()
    gapped = {s.segment_id for s in result.segments if "S-G1" in s.source_file}
    splits = {a.split for a in result.assignments if a.segment_id in gapped}
    assert len(splits) == 1


def test_plots_are_generated_from_a_real_result(dataset_tree: Path, tmp_path: Path) -> None:
    """Plot generation is exercised end to end, not just in the real run."""
    from idr.pipeline.plots import generate_phase2_plots

    result = Phase2Pipeline(dataset_tree, tmp_path / "processed").run()
    plots = generate_phase2_plots(result, tmp_path / "plots")
    assert plots, "expected at least one figure"
    for plot in plots:
        assert plot.exists() and plot.stat().st_size > 0


def test_report_is_generated_and_states_the_key_guarantees(
    dataset_tree: Path, tmp_path: Path
) -> None:
    from idr.pipeline.report import generate_phase2_report

    result = Phase2Pipeline(dataset_tree, tmp_path / "processed").run()
    output = generate_phase2_report(result, tmp_path / "report.md", [])
    text = output.read_text(encoding="utf-8")

    # The report must carry the claims Phase 3 depends on.
    assert "Canonical schema" in text
    assert "payload hash" in text.lower()
    assert "causal" in text.lower()
    assert "GNSS_STALE" in text
    # It must not claim outages exist.
    assert "zero confirmed" in text.lower() or "no confirmed" in text.lower()


def test_report_handles_a_result_with_no_sync_pairs(tmp_path: Path, smartphone_csv: Path) -> None:
    """A smartphone-only tree still produces a coherent report."""
    from idr.pipeline.report import generate_phase2_report

    root = tmp_path / "solo"
    root.mkdir()
    (root / "S-X1.csv").write_bytes(smartphone_csv.read_bytes())
    result = Phase2Pipeline(root, tmp_path / "processed").run()
    output = generate_phase2_report(result, tmp_path / "solo.md", [])
    assert output.exists()
    assert "No paired sessions" in output.read_text(encoding="utf-8")
