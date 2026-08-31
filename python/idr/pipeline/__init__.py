"""Phase 2: IO-VNBD canonicalization, synchronization and ML-ready pipeline.

Read-only with respect to raw data. The public surface below is what Phase 3
and later should build on; see
``docs/datasets/io_vnbd_phase2_contract.md`` for the data contract.
"""

from __future__ import annotations

from idr.pipeline.canonical import (
    CANONICAL_SCHEMA_VERSION,
    QualityFlag,
    describe_flags,
    schema_for,
)
from idr.pipeline.reference import LabelSource
from idr.pipeline.sequences import WindowConfig, generate_windows
from idr.pipeline.splits import SplitUnit, assign_splits, verify_no_leakage
from idr.pipeline.sync import SyncStatus

__all__ = [
    "CANONICAL_SCHEMA_VERSION",
    "LabelSource",
    "QualityFlag",
    "SplitUnit",
    "SyncStatus",
    "WindowConfig",
    "assign_splits",
    "describe_flags",
    "generate_windows",
    "schema_for",
    "verify_no_leakage",
]
