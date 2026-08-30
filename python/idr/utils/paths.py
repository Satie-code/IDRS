"""Repository path resolution.

Rule 2 (see docs/development/coding_standards.md): never hardcode
environment-specific paths. Everything that needs to locate files relative
to the repository root should go through :func:`find_repo_root`.
"""

from __future__ import annotations

from pathlib import Path

_MARKERS = ("pyproject.toml", ".git")


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward from ``start`` (default: this file) to find the repo root.

    The repo root is identified by the presence of ``pyproject.toml`` or a
    ``.git`` directory. Raises :class:`FileNotFoundError` if neither marker
    is found, rather than silently falling back to an arbitrary directory.
    """
    current = (start or Path(__file__)).resolve()
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in _MARKERS):
            return candidate
    raise FileNotFoundError(
        f"Could not locate repository root above {current!s} (looked for one of {_MARKERS!r})"
    )
