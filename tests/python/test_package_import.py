"""The idr package must be importable without manual PYTHONPATH edits, and
must expose a single, non-empty version string."""

from __future__ import annotations

import idr


def test_package_imports() -> None:
    assert idr.__version__


def test_version_is_dotted_triplet() -> None:
    parts = idr.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
