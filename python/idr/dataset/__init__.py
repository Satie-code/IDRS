"""IO-VNBD dataset inspection and forensic-analysis layer (Phase 1).

Read-only by design: nothing in this package modifies raw dataset files.
The public API below is what Phase 2 preprocessing should build on.
"""

from __future__ import annotations

from idr.dataset.evidence import Evidence, StatKind
from idr.dataset.lfs import LfsPointer, is_pointer, read_pointer

__all__ = [
    "Evidence",
    "LfsPointer",
    "StatKind",
    "is_pointer",
    "read_pointer",
]
