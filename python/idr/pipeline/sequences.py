"""Causal sequence-window generation for future ML training.

Phase 2 builds the *generator*, not a training corpus. Windows are produced on
demand from canonical Parquet so the dataset is never duplicated on disk in a
fixed windowing that a later experiment would have to redo.

The one inviolable property is **causality**: for a target at time ``t``, the
input window contains only samples with time ``<= t``. Nothing from after the
target may enter the window. This is enforced here and asserted by tests.

Window geometry is configuration, not a constant. The 5 s / 0.1 s default is a
starting point, not a finding.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowConfig:
    """Geometry of a generated sequence window."""

    duration_seconds: float = 5.0
    stride_seconds: float = 0.1
    #: Reject a window whose samples cover less than this fraction of its span,
    #: which keeps sparse or gap-ridden stretches out of the training set.
    min_coverage: float = 0.8
    #: Require a finite target value at the window's end.
    require_label: bool = True

    def __post_init__(self) -> None:
        if self.duration_seconds <= 0:
            raise ValueError("window duration_seconds must be positive")
        if self.stride_seconds <= 0:
            raise ValueError("window stride_seconds must be positive")
        if not 0.0 <= self.min_coverage <= 1.0:
            raise ValueError("min_coverage must be within [0, 1]")


@dataclass(frozen=True)
class Window:
    """One causal window: indices into the segment plus its target."""

    segment_id: str
    session_id: str
    split: str
    #: Row indices of the input samples, all with time <= target_time_s.
    input_indices: np.ndarray
    target_index: int
    target_time_s: float
    target_value: float
    label_source: str
    label_quality: str
    coverage: float
    quality_flags: int

    @property
    def sample_count(self) -> int:
        return int(self.input_indices.size)


def generate_windows(
    frame: pd.DataFrame,
    config: WindowConfig,
    *,
    segment_id: str,
    session_id: str,
    split: str = "unassigned",
    target_column: str = "ref_velocity_mps",
    time_column: str = "elapsed_s",
) -> Iterator[Window]:
    """Yield causal windows over one segment.

    The frame must be a single segment, ordered by ``time_column``. Targets are
    placed on a uniform stride grid; for each target the window spans
    ``[t - duration, t]`` **inclusive of t and exclusive of nothing after it**.
    """
    if frame.empty:
        return

    times = frame[time_column].to_numpy(dtype="float64")
    finite = np.isfinite(times)
    if finite.sum() < 2:
        return

    targets = (
        frame[target_column].to_numpy(dtype="float64")
        if target_column in frame.columns
        else np.full(len(frame), np.nan)
    )
    flags = (
        frame["quality_flags"].to_numpy(dtype="int64")
        if "quality_flags" in frame.columns
        else np.zeros(len(frame), dtype="int64")
    )
    label_source = str(frame["label_source"].iloc[0]) if "label_source" in frame.columns else ""
    label_quality = str(frame["label_quality"].iloc[0]) if "label_quality" in frame.columns else ""

    start_time = float(times[finite][0])
    end_time = float(times[finite][-1])
    first_target = start_time + config.duration_seconds
    if first_target > end_time:
        return

    # Expected sample count if the segment ran at its own median cadence.
    deltas = np.diff(times[finite])
    positive = deltas[deltas > 0]
    median_interval = float(np.median(positive)) if positive.size else None

    grid = np.arange(first_target, end_time + config.stride_seconds, config.stride_seconds)
    for target_time in grid:
        if target_time > end_time:
            break
        window_start = target_time - config.duration_seconds

        # Causality: strictly no sample after the target may be included.
        in_window = finite & (times >= window_start) & (times <= target_time)
        indices = np.flatnonzero(in_window)
        if indices.size < 2:
            continue

        if median_interval:
            expected = max(1.0, config.duration_seconds / median_interval)
            coverage = min(1.0, indices.size / expected)
        else:
            coverage = 1.0
        if coverage < config.min_coverage:
            continue

        target_index = int(indices[-1])
        target_value = float(targets[target_index])
        if config.require_label and not np.isfinite(target_value):
            continue

        yield Window(
            segment_id=segment_id,
            session_id=session_id,
            split=split,
            input_indices=indices,
            target_index=target_index,
            target_time_s=float(times[target_index]),
            target_value=target_value,
            label_source=label_source,
            label_quality=label_quality,
            coverage=float(coverage),
            quality_flags=int(np.bitwise_or.reduce(flags[indices])) if indices.size else 0,
        )


def count_windows(
    frame: pd.DataFrame, config: WindowConfig, *, segment_id: str, session_id: str
) -> int:
    """Number of windows a segment would yield, without materializing them."""
    return sum(
        1 for _ in generate_windows(frame, config, segment_id=segment_id, session_id=session_id)
    )
