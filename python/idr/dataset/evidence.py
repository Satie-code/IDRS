"""Provenance and confidence vocabulary shared by every dataset report.

Phase 1 is a forensic exercise: a claim about the dataset is only useful if
the reader can tell *how it was established*. Every fact recorded by this
package therefore carries an :class:`Evidence` level, and every numeric
statistic carries a :class:`StatKind` saying whether it was computed over
all rows or estimated from a sample.
"""

from __future__ import annotations

from enum import StrEnum


class Evidence(StrEnum):
    """How a recorded claim was established."""

    #: Read directly out of the actual data files in this repository.
    VERIFIED_FROM_FILE = "VERIFIED_FROM_FILE"
    #: Stated by documentation shipped inside the dataset itself.
    VERIFIED_FROM_DATASET_DOCUMENTATION = "VERIFIED_FROM_DATASET_DOCUMENTATION"
    #: Stated by the official upstream dataset source (e.g. its Git host).
    VERIFIED_FROM_OFFICIAL_SOURCE = "VERIFIED_FROM_OFFICIAL_SOURCE"
    #: Reasoned from other verified facts, but not directly observed.
    INFERRED = "INFERRED"
    #: Not established. Recorded so the gap is visible rather than silently absent.
    UNVERIFIED = "UNVERIFIED"


class StatKind(StrEnum):
    """Whether a statistic covers all data or only part of it."""

    #: Computed over every row of the stream.
    EXACT = "exact"
    #: Computed over a documented subset (see ``sample_size``/``sampling_method``).
    SAMPLED = "sampled"
    #: Derived rather than measured (e.g. duration from first/last timestamp).
    ESTIMATED = "estimated"
