from __future__ import annotations

import json
from pathlib import Path

import pytest

from idr.dataset.inspect import DatasetInspector, main, write_artifacts
from idr.dataset.inventory import (
    build_file_inventory,
    classify_family,
    discover_sessions,
)


@pytest.fixture
def dataset_tree(tmp_path: Path, smartphone_csv: Path, vehicle_csv: Path) -> Path:
    """A miniature dataset tree mirroring the real directory conventions."""
    root = tmp_path / "tree"
    session = (
        root / "Synchronised V abd S datasets" / "Categorised IOVNB Dataset" / "S (Driver A)" / "S1"
    )
    session.mkdir(parents=True)
    (session / "S-S1.csv").write_bytes(smartphone_csv.read_bytes())
    (session / "V-S1.csv").write_bytes(vehicle_csv.read_bytes())
    (root / "README.md").write_text("dataset docs\n", encoding="utf-8")
    return root


def test_inventory_discovers_all_files(dataset_tree: Path) -> None:
    records = build_file_inventory(dataset_tree)
    files = {r.filename for r in records if r.is_file}
    assert files == {"S-S1.csv", "V-S1.csv", "README.md"}
    assert any(r.is_directory for r in records)


def test_inventory_computes_full_file_hashes(dataset_tree: Path) -> None:
    records = [r for r in build_file_inventory(dataset_tree) if r.filename == "S-S1.csv"]
    assert records[0].sha256 is not None
    assert records[0].sha256_scope == "full_file"


def test_inventory_flags_pointer_without_hashing_as_data(tmp_path: Path, lfs_pointer: Path) -> None:
    root = tmp_path / "ptr"
    root.mkdir()
    (root / "S-P.csv").write_bytes(lfs_pointer.read_bytes())
    record = next(r for r in build_file_inventory(root) if r.is_file)
    assert record.is_lfs_pointer
    assert record.lfs_size == 9631499
    assert record.lfs_oid


def test_family_classification_from_path() -> None:
    assert classify_family("Synchronised V abd S datasets/Categorised IOVNB Dataset/x.csv") == (
        "synchronized_categorized"
    )
    assert classify_family("Unsynchronised V and S Dataset/Uncategorised x/y.csv") == (
        "unsynchronized_uncategorized"
    )
    assert classify_family("README.md") == "documentation"
    assert classify_family("photo.jpg") == "imagery"


def test_session_discovery_pairs_streams(dataset_tree: Path) -> None:
    sessions = discover_sessions(build_file_inventory(dataset_tree))
    assert len(sessions) == 1
    session = sessions[0]
    assert session.session_id == "S1"
    assert session.driver == "A"
    assert session.is_paired
    # Filename convention alone justifies only an inference, not a verified fact.
    assert session.pairing_evidence == "INFERRED"


def test_session_ids_are_case_normalized(tmp_path: Path, smartphone_csv: Path) -> None:
    root = tmp_path / "case"
    root.mkdir()
    (root / "S-Vta11.csv").write_bytes(smartphone_csv.read_bytes())
    (root / "V-vta11.csv").write_bytes(smartphone_csv.read_bytes())
    sessions = discover_sessions(build_file_inventory(root))
    assert len(sessions) == 1
    assert sessions[0].is_paired


def test_inspector_deep_run_produces_reports(dataset_tree: Path) -> None:
    result = DatasetInspector(dataset_tree, deep=True).run()
    assert result.scans
    assert result.timestamps
    assert result.synchronization
    assert result.quality
    assert not result.parse_failures


def test_manifest_is_valid_and_deterministic(dataset_tree: Path, tmp_path: Path) -> None:
    result = DatasetInspector(dataset_tree, deep=True).run()

    first = tmp_path / "m1"
    second = tmp_path / "m2"
    write_artifacts(result, first, tmp_path / "r1")
    write_artifacts(result, second, tmp_path / "r2")

    left = json.loads((first / "dataset_manifest.json").read_text(encoding="utf-8"))
    right = json.loads((second / "dataset_manifest.json").read_text(encoding="utf-8"))
    # scan_timestamp is captured once per run, so identical input gives identical output.
    assert left == right
    assert left["csv_count"] == 2
    assert left["session_count"] == 1


def test_all_expected_artifacts_are_written(dataset_tree: Path, tmp_path: Path) -> None:
    result = DatasetInspector(dataset_tree, deep=True).run()
    metadata = tmp_path / "meta"
    written = write_artifacts(result, metadata, tmp_path / "rep")

    for name in (
        "dataset_manifest.json",
        "integrity_report.json",
        "session_inventory.csv",
        "schema_inventory.json",
        "smartphone_schema.json",
        "vehicle_schema.json",
        "sampling_report.csv",
        "timestamp_report.csv",
        "synchronization_report.csv",
        "gnss_outages.csv",
        "scenario_inventory.csv",
        "session_quality.csv",
    ):
        assert name in written, f"{name} not written"
        assert (metadata / name).exists()


def test_integrity_fails_when_everything_is_a_pointer(tmp_path: Path, lfs_pointer: Path) -> None:
    root = tmp_path / "allptr"
    root.mkdir()
    (root / "S-A.csv").write_bytes(lfs_pointer.read_bytes())
    result = DatasetInspector(root, deep=True).run()
    written = write_artifacts(result, tmp_path / "m", tmp_path / "r")
    report = json.loads(Path(written["integrity_report.json"]).read_text(encoding="utf-8"))
    assert report["integrity_status"] == "FAIL"
    assert "unresolved Git LFS pointer" in report["reasons"][0]


def test_cli_rejects_missing_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--dataset-path", str(tmp_path / "nope")])
    assert code == 2
    assert "does not exist" in capsys.readouterr().err


def test_cli_rejects_file_as_dataset_path(
    smartphone_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--dataset-path", str(smartphone_csv)])
    assert code == 2
    assert "not a directory" in capsys.readouterr().err


def test_cli_succeeds_on_valid_tree(dataset_tree: Path, tmp_path: Path) -> None:
    code = main(
        [
            "--dataset-path",
            str(dataset_tree),
            "--output",
            str(tmp_path / "out"),
            "--report-output",
            str(tmp_path / "rep"),
            "--deep",
            "--no-plots",
        ]
    )
    assert code == 0
    assert (tmp_path / "out" / "dataset_manifest.json").exists()


def test_inspection_does_not_modify_the_dataset(dataset_tree: Path, tmp_path: Path) -> None:
    before = {
        path.relative_to(dataset_tree).as_posix(): path.read_bytes()
        for path in sorted(dataset_tree.rglob("*"))
        if path.is_file()
    }
    result = DatasetInspector(dataset_tree, deep=True).run()
    write_artifacts(result, tmp_path / "out", tmp_path / "rep")
    after = {
        path.relative_to(dataset_tree).as_posix(): path.read_bytes()
        for path in sorted(dataset_tree.rglob("*"))
        if path.is_file()
    }
    assert before == after


def test_detects_byte_identical_duplicates(tmp_path: Path, smartphone_csv: Path) -> None:
    """The same content filed in two places must be reported, not counted twice."""
    from idr.dataset.inventory import canonical_paths, find_duplicate_groups

    root = tmp_path / "dupes"
    (root / "Categorised").mkdir(parents=True)
    (root / "Uncategorised").mkdir(parents=True)
    payload = smartphone_csv.read_bytes()
    (root / "Categorised" / "S-D1.csv").write_bytes(payload)
    (root / "Uncategorised" / "S-D1.csv").write_bytes(payload)
    (root / "Categorised" / "S-D2.csv").write_bytes(payload[:-20])

    records = build_file_inventory(root)
    groups = find_duplicate_groups(records)

    assert len(groups) == 1
    assert sorted(next(iter(groups.values()))) == [
        "Categorised/S-D1.csv",
        "Uncategorised/S-D1.csv",
    ]
    # Three files, two distinct contents.
    assert len(canonical_paths(records)) == 2


def test_unique_files_produce_no_duplicate_groups(dataset_tree: Path) -> None:
    from idr.dataset.inventory import find_duplicate_groups

    assert find_duplicate_groups(build_file_inventory(dataset_tree)) == {}


def test_duplicate_artifact_is_written(tmp_path: Path, smartphone_csv: Path) -> None:
    root = tmp_path / "d"
    root.mkdir()
    payload = smartphone_csv.read_bytes()
    (root / "S-A.csv").write_bytes(payload)
    (root / "S-B.csv").write_bytes(payload)

    result = DatasetInspector(root, deep=True).run()
    written = write_artifacts(result, tmp_path / "m", tmp_path / "r")

    assert "duplicate_files.csv" in written
    text = Path(written["duplicate_files.csv"]).read_text(encoding="utf-8")
    assert "S-A.csv" in text and "S-B.csv" in text
    assert len(result.canonical_files) == 1
