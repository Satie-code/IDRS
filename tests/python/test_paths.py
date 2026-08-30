from __future__ import annotations

from idr.utils import find_repo_root


def test_find_repo_root_locates_pyproject() -> None:
    root = find_repo_root()
    assert (root / "pyproject.toml").exists()
    assert (root / "python" / "idr").is_dir()
