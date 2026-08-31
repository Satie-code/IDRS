"""Detection and derived repair of column-shifted source files.

Phase 1 found exactly one malformed file, ``S-A4.csv``: its header names 24
columns but every data row carries 25 fields, because an empty field is
inserted at index 6. Every value from that point on therefore sits one column
to the right of the header label that describes it, which is silent
corruption — the file parses cleanly and only the *meaning* is wrong.

Rules this module follows:

- The raw file is never touched. Repair produces a corrected view in memory
  that flows only into derived (canonical) output.
- The repair is **detected**, not assumed: it applies only to files whose
  structure actually matches the defect signature, so a generic parser never
  silently reshapes a healthy file.
- Every repaired row is tagged ``QualityFlag.SCHEMA_REPAIR`` and the decision
  is recorded with its reason and confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

#: Column index where the spurious empty field appears in the affected file.
SHIFT_INDEX = 6


@dataclass(frozen=True)
class RepairDecision:
    """Outcome of evaluating one file for the column-shift defect."""

    applied: bool
    rule: str
    reason: str
    confidence: str
    shift_index: int | None = None
    dropped_column: str | None = None


NO_REPAIR = RepairDecision(
    applied=False,
    rule="none",
    reason="file structure does not match the column-shift signature",
    confidence="n/a",
)


def meaningful_header_names(header_names: list[str]) -> list[str]:
    """Header names with a trailing blank field removed.

    The affected file ends its header line with a delimiter, so the last
    "name" is empty. That blank is a symptom of the defect, not a real column.
    """
    names = list(header_names)
    while names and not str(names[-1]).strip():
        names.pop()
    return names


def detect_column_shift(frame: pd.DataFrame, header_names: list[str]) -> RepairDecision:
    """Decide whether ``frame`` exhibits the one-column shift defect.

    The signature, all of which must hold:

    1. the parsed frame has exactly one more column than the header *names*
       (ignoring a trailing blank), with the surplus materialized by pandas as
       an ``Unnamed:`` column;
    2. the column at :data:`SHIFT_INDEX` is entirely empty;
    3. the trailing surplus column is *not* entirely empty — real data has been
       pushed into it.

    Requiring all three keeps a file with a merely-empty channel, or a stray
    trailing comma with no data behind it, from being reshaped.
    """
    names = meaningful_header_names(header_names)
    if frame.shape[1] != len(names) + 1:
        return NO_REPAIR

    trailing = frame.columns[-1]
    if not str(trailing).startswith("Unnamed:"):
        return NO_REPAIR

    shifted_col = frame.columns[SHIFT_INDEX]
    shifted_is_empty = (
        frame[shifted_col].isna().all()
        or (frame[shifted_col].astype("string").str.strip() == "").all()
    )
    trailing_has_data = not (
        frame[trailing].isna().all()
        or (frame[trailing].astype("string").fillna("").str.strip() == "").all()
    )

    if not (shifted_is_empty and trailing_has_data):
        return NO_REPAIR

    return RepairDecision(
        applied=True,
        rule=f"drop empty column at index {SHIFT_INDEX}, shift remaining columns left by one",
        reason=(
            f"data rows carry one field more than the {len(names)}-column header; "
            f"column {SHIFT_INDEX} is empty in every row while the surplus trailing column "
            f"holds data, so every value past index {SHIFT_INDEX} is offset by one"
        ),
        confidence="high",
        shift_index=SHIFT_INDEX,
        dropped_column=str(shifted_col),
    )


def apply_column_shift(frame: pd.DataFrame, header_names: list[str]) -> pd.DataFrame:
    """Return a repaired copy: drop the spurious column and re-label.

    The input frame is not modified. Callers must have confirmed the defect
    with :func:`detect_column_shift` first.
    """
    names = meaningful_header_names(header_names)
    repaired = frame.drop(columns=[frame.columns[SHIFT_INDEX]])
    if repaired.shape[1] != len(names):
        raise ValueError(
            f"repair produced {repaired.shape[1]} columns but the header names "
            f"{len(names)}; refusing to emit a mislabelled frame"
        )
    repaired.columns = pd.Index(names)
    return repaired
