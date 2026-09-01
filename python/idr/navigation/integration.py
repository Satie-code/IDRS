"""Numerical integration and the time-step policy that guards it.

Two responsibilities, kept apart from the mechanization so each can be tested
against an analytical answer without a sensor anywhere in sight:

* :func:`trapezoidal_step` — one integration step on an actual Δt.
* :class:`TimeStepPolicy` — deciding whether a Δt may be integrated across at
  all, and what to do when it may not.

The second is not defensive boilerplate. Phase 2 retains 147,946 duplicate
timestamps (Δt = 0) and Phase 3 measured segment rates from 2 Hz to 1000 Hz, so
invalid and wildly varying steps are the normal case in this dataset rather
than an edge case. A mechanization that assumes otherwise integrates a
zero-second interval or bridges a ten-second gap, and reports a trajectory
either way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

#: Longest interval that may be integrated across by default, seconds. Matches
#: ``idr.frames.orientation.MAX_INTEGRATION_STEP_S`` so that attitude and
#: velocity break at the same places — a trajectory whose attitude was held
#: constant across a gap its velocity integrated through would be internally
#: inconsistent in a way nothing downstream could detect.
MAX_INTEGRATION_STEP_S = 0.5

#: A Δt this far from the segment's median counts as irregular. Flagging only;
#: an irregular step is still integrated, because irregular sampling is a
#: property of the logger rather than an error, and refusing it would discard
#: whole segments. 5× is loose enough to ignore ordinary jitter.
IRREGULAR_DT_FACTOR = 5.0


class InvalidStepAction(StrEnum):
    """What to do with a Δt that cannot be integrated across."""

    #: Carry the state forward unchanged, flag the sample, continue. The
    #: default: it keeps the trajectory contiguous in time and marks exactly
    #: where nothing was propagated.
    SKIP = "skip"
    #: End the current trajectory and require re-initialization. Appropriate
    #: when a downstream consumer must not see two sides of a gap as one
    #: continuous solution.
    SEGMENT = "segment"
    #: Raise. For tests and for callers who would rather not get a number.
    FAIL = "fail"


class StepVerdict(StrEnum):
    """The classification of one candidate integration step."""

    VALID = "valid"
    #: Δt ≤ 0 — a duplicate or non-monotonic timestamp.
    NON_POSITIVE = "non_positive"
    #: Δt exceeds the configured maximum.
    TOO_LARGE = "too_large"
    #: Δt is NaN or infinite.
    NOT_FINITE = "not_finite"


@dataclass(frozen=True)
class TimeStepPolicy:
    """How Δt is validated, and what happens when validation fails.

    ``max_step_s`` is a physical judgement, not a numerical one: across a gap
    longer than this, an inertial solution has no information about what
    happened, and trapezoidal integration between the two endpoints is an
    assumption about the interior dressed as a measurement.
    """

    max_step_s: float = MAX_INTEGRATION_STEP_S
    action: InvalidStepAction = InvalidStepAction.SKIP
    irregular_factor: float = IRREGULAR_DT_FACTOR

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_step_s) or self.max_step_s <= 0.0:
            raise ValueError(f"max_step_s must be finite and positive, got {self.max_step_s!r}")
        if self.irregular_factor <= 1.0:
            raise ValueError(
                f"irregular_factor must exceed 1, got {self.irregular_factor!r}; "
                "a factor of 1 would flag every step that is not exactly the median"
            )

    def classify(self, dt_s: float) -> StepVerdict:
        if not math.isfinite(dt_s):
            return StepVerdict.NOT_FINITE
        if dt_s <= 0.0:
            return StepVerdict.NON_POSITIVE
        if dt_s > self.max_step_s:
            return StepVerdict.TOO_LARGE
        return StepVerdict.VALID

    def is_irregular(self, dt_s: float, median_dt_s: float | None) -> bool:
        """Whether this step is unusual *for this segment*, not in absolute terms.

        A 0.4 s step is ordinary at 2 Hz and a forty-fold outlier at 1000 Hz,
        so the comparison is always against the segment's own median.
        """
        if median_dt_s is None or not math.isfinite(median_dt_s) or median_dt_s <= 0.0:
            return False
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            return False
        ratio = dt_s / median_dt_s
        return ratio > self.irregular_factor or ratio < 1.0 / self.irregular_factor

    def describe(self) -> str:
        return (
            f"steps longer than {self.max_step_s:g}s or not strictly positive are "
            f"handled by '{self.action}'; steps beyond {self.irregular_factor:g}× the "
            "segment median are flagged as irregular but still integrated"
        )


class InvalidTimeStepError(ValueError):
    """Raised when :data:`InvalidStepAction.FAIL` meets an invalid Δt."""

    def __init__(self, dt_s: float, verdict: StepVerdict, index: int) -> None:
        super().__init__(
            f"sample {index} has dt={dt_s!r} ({verdict}); the configured policy is to fail"
        )
        self.dt_s = dt_s
        self.verdict = verdict
        self.index = index


def trapezoidal_step(
    previous: np.ndarray,
    rate_previous: np.ndarray,
    rate_current: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    """One trapezoidal integration step on an actual Δt.

    ``x_k = x_{k−1} + ½(ẋ_{k−1} + ẋ_k)·Δt``

    Trapezoidal rather than rectangular because the rectangular rule's error is
    first order in Δt and biased in one direction — it under-integrates a
    rising signal at every step, so a vehicle accelerating from rest
    accumulates a systematic velocity deficit rather than a zero-mean error.
    Trapezoidal is second order and exact for a linearly varying rate, which
    covers constant acceleration exactly. That is what makes the analytical
    tests in the Phase 4 suite able to demand agreement to machine precision
    rather than to a hand-chosen tolerance.

    Not implemented here, deliberately: coning and sculling compensation, and a
    midpoint attitude update. See ``docs/mathematics/phase4_mechanization.md``
    for what that costs and when it starts to matter.
    """
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError(f"trapezoidal_step needs a positive finite dt, got {dt_s!r}")
    mean_rate = 0.5 * (
        np.asarray(rate_previous, dtype="float64") + np.asarray(rate_current, dtype="float64")
    )
    return np.asarray(previous, dtype="float64") + mean_rate * dt_s


def integrate_series(
    rates: np.ndarray, times: np.ndarray, initial: np.ndarray | None = None
) -> np.ndarray:
    """Trapezoidal integration of a whole series, for tests and diagnostics.

    The mechanization does not use this — it integrates sample by sample so
    that a state can be inspected and flagged mid-run — but having the vectorized
    form lets a test compare the loop against an independent implementation
    rather than against itself.
    """
    values = np.asarray(rates, dtype="float64")
    stamps = np.asarray(times, dtype="float64")
    if values.shape[0] != stamps.shape[0]:
        raise ValueError(f"{values.shape[0]} rates against {stamps.shape[0]} timestamps")
    start = np.zeros(values.shape[1:]) if initial is None else np.asarray(initial, dtype="float64")
    out = np.empty_like(values)
    out[0] = start
    dt = np.diff(stamps)
    increments = 0.5 * (values[:-1] + values[1:]) * dt.reshape(-1, *([1] * (values.ndim - 1)))
    out[1:] = start + np.cumsum(increments, axis=0)
    return out


def median_interval(times: np.ndarray) -> float | None:
    """The segment's median positive Δt, or None when there is none.

    Used as the reference for irregularity. Positive intervals only: with
    147,946 duplicate timestamps across this dataset, including the zeros would
    drag the median toward zero on exactly the segments that need the check.
    """
    stamps = np.asarray(times, dtype="float64")
    if stamps.size < 2:
        return None
    intervals = np.diff(stamps)
    positive = intervals[np.isfinite(intervals) & (intervals > 0.0)]
    return float(np.median(positive)) if positive.size else None
