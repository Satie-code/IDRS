from __future__ import annotations

from pathlib import Path

from idr.dataset.lfs import is_pointer, read_pointer


def test_recognizes_lfs_pointer(lfs_pointer: Path) -> None:
    pointer = read_pointer(lfs_pointer)
    assert pointer is not None
    assert pointer.oid == "e79a2eea18143b825438f9a11c7d724a97c9e18329776defd4d9291d03d4aaca"
    assert pointer.size == 9631499


def test_real_csv_is_not_a_pointer(smartphone_csv: Path) -> None:
    assert read_pointer(smartphone_csv) is None
    assert not is_pointer(smartphone_csv)


def test_truncated_pointer_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.csv"
    path.write_text("version https://git-lfs.github.com/spec/v1\n", encoding="utf-8")
    assert read_pointer(path) is None


def test_missing_file_is_not_a_pointer(tmp_path: Path) -> None:
    assert read_pointer(tmp_path / "does-not-exist.csv") is None
