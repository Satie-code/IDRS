"""Deterministic stationarity detection, for initialization and diagnostics only.

Scope, stated first because the temptation is real: this module **detects**
stationary intervals. It does not correct anything. A zero-velocity update is a
state correction, it belongs with the filter, and Phase 4 deliberately does not
have one — the whole point of the baseline is to show what an unaided solution
does, and a ZUPT is aiding.

What the detection is used for here:

* choosing the window a bias estimate is taken from (:mod:`idr.navigation.bias`);
* justifying an initial velocity of zero;
* annotating samples in the trajectory, so a drift plot can show that velocity
  wandered while the vehicle demonstrably was not moving.

The detector is thresholds on physically meaningful quantities. No classifier
is trained; a learned detector would need this one to label its data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from idr.frames.conventions import STANDARD_GRAVITY_MPS2
from idr.frames.vectors import VectorSeries

#: Standard deviation of accelerometer magnitude, over a window, below which
#: the device is not experiencing translational dynamics, m/s². Road vibration
#: through a parked car's suspension is well under this; Phase 3 measured a
#: median horizontal specific force of ~1.2 m/s² while *driving*, so 0.25 is
#: comfortably separating.
MAX_STATIONARY_ACCEL_STD_MPS2 = 0.25

#: Mean gyroscope magnitude below which the device is not rotating, rad/s.
#: 0.03 rad/s ≈ 1.7°/s: above a MEMS noise floor, far below a vehicle turning.
MAX_STATIONARY_GYRO_RPS = 0.03

#: How far the accelerometer magnitude may sit from standard gravity, m/s².
#: At rest the sensor reads exactly |g|; a persistent departure means either
#: real acceleration or a bias large enough that "stationary" is the wrong
#: label for what the sensor is doing.
#:
#: **The value matters more than it looks, because horizontal acceleration adds
#: to gravity in quadrature.** A horizontal acceleration ``a`` raises the
#: measured magnitude only to ``√(g² + a²)``, so a tolerance ``τ`` leaves the
#: detector blind to accelerations below ``√(2gτ)``. At τ = 0.5 that is
#: 3.1 m/s² — brisk acceleration in a car, wrongly called rest. At τ = 0.15 it
#: is 1.7 m/s², which is the blind spot this implementation actually has and
#: which :func:`detect_stationarity` documents rather than papers over.
#:
#: 0.15 is not arbitrary: across the real segments where a stationary interval
#: was found, the mean specific-force magnitude during rest sits between 9.80
#: and 9.90 m/s², a worst-case departure of 0.09. The threshold accommodates
#: what the data actually does, with margin, and no more.
MAX_STATIONARY_GRAVITY_ERROR_MPS2 = 0.15

#: Shortest window that can support a stationarity claim, seconds. Below this a
#: window can sit inside the quiet part of an ordinary drive — the instant
#: between lifting off the accelerator and touching the brake — and calling
#: that stationary would zero a real velocity.
MIN_STATIONARY_WINDOW_S = 1.0


@dataclass
class StationarityResult:
    """Per-sample stationarity, with the metrics that produced it."""

    #: Boolean per sample. True means "at rest at this instant".
    mask: np.ndarray
    #: Fraction of samples judged stationary, 0–1.
    fraction: float
    confidence: float
    accel_std_mps2: np.ndarray
    gyro_magnitude_rps: np.ndarray
    gravity_error_mps2: np.ndarray
    window_s: float
    #: Longest contiguous stationary run, seconds.
    longest_interval_s: float
    #: Slice bounds of that run, or None when there is no run of two samples.
    longest_span: tuple[int, int] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def any_stationary(self) -> bool:
        return bool(self.mask.any())

    def longest_window_mask(self, minimum_s: float = MIN_STATIONARY_WINDOW_S) -> np.ndarray:
        """A mask selecting only the longest stationary run, if it is long enough.

        Bias estimation wants one contiguous interval rather than every
        scattered quiet sample: samples drawn from a dozen different moments in
        a drive have a dozen different orientations, and averaging across them
        smears the gravity direction the accelerometer bias is measured along.
        """
        if self.longest_span is None or self.longest_interval_s < minimum_s:
            return np.zeros_like(self.mask, dtype=bool)
        start, stop = self.longest_span
        out = np.zeros_like(self.mask, dtype=bool)
        out[start:stop] = True
        return out


def _rolling_std(values: np.ndarray, times: np.ndarray, window_s: float) -> np.ndarray:
    """Trailing standard deviation over a *time* window.

    Time-based rather than sample-count-based for the same reason Phase 3's
    smoothing is: a 20-sample window is 10 seconds at 2 Hz and 20 milliseconds
    at 1000 Hz, and this dataset contains both.
    """
    array = np.asarray(values, dtype="float64")
    finite = np.isfinite(array)
    filled = np.where(finite, array, 0.0)

    total = np.concatenate([[0.0], np.cumsum(filled)])
    total_sq = np.concatenate([[0.0], np.cumsum(filled * filled)])
    count = np.concatenate([[0.0], np.cumsum(finite)])

    start = np.searchsorted(times, times - window_s, side="left")
    stop = np.arange(1, array.size + 1)

    n = count[stop] - count[start]
    s1 = total[stop] - total[start]
    s2 = total_sq[stop] - total_sq[start]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / np.maximum(n, 1.0)
        variance = s2 / np.maximum(n, 1.0) - mean * mean
    variance = np.where(n >= 2, np.maximum(variance, 0.0), np.nan)
    return np.sqrt(variance)


def _rolling_mean(values: np.ndarray, times: np.ndarray, window_s: float) -> np.ndarray:
    array = np.asarray(values, dtype="float64")
    finite = np.isfinite(array)
    filled = np.where(finite, array, 0.0)
    total = np.concatenate([[0.0], np.cumsum(filled)])
    count = np.concatenate([[0.0], np.cumsum(finite)])
    start = np.searchsorted(times, times - window_s, side="left")
    stop = np.arange(1, array.size + 1)
    n = count[stop] - count[start]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[stop] - total[start]) / np.maximum(n, 1.0), np.nan)


def detect_stationarity(
    *,
    accel: VectorSeries,
    gyro: VectorSeries | None = None,
    window_s: float = 2.0,
    expected_gravity_mps2: float = STANDARD_GRAVITY_MPS2,
    accel_std_tolerance: float = MAX_STATIONARY_ACCEL_STD_MPS2,
    gyro_tolerance: float = MAX_STATIONARY_GYRO_RPS,
    gravity_error_tolerance: float = MAX_STATIONARY_GRAVITY_ERROR_MPS2,
) -> StationarityResult:
    """Classify each sample as at rest or not, and say how confidently.

    Three independent conditions, all of which must hold:

    1. the accelerometer magnitude is *steady* over the window (no dynamics);
    2. the gyroscope magnitude is small (no rotation);
    3. the accelerometer magnitude sits near gravity (what a resting sensor
       reads, and a check the first condition alone does not provide — a
       constant 3 m/s² acceleration is perfectly steady).

    Condition 3 is the one that earns its place. Without it a vehicle in smooth
    cruise on a straight road passes conditions 1 and 2 comfortably, and would
    have its velocity declared zero.

    **The known blind spot.** Condition 3 discriminates only in quadrature: a
    horizontal acceleration ``a`` raises the measured magnitude to
    ``√(g² + a²)``, so accelerations below about 1.7 m/s² (see
    :data:`MAX_STATIONARY_GRAVITY_ERROR_MPS2`) do not move it past the
    threshold. A vehicle holding a gentle, perfectly constant acceleration in a
    straight line can therefore be classified as stationary by an IMU-only
    detector — not by this one specifically, but by the class of them. Rest and
    constant acceleration are genuinely indistinguishable from specific force
    alone, and separating them needs an independent velocity observation.

    Phase 4 uses the result only for initialization, bias windows and
    annotation, so the consequence of the blind spot is bounded: a mis-detected
    window yields a bias estimate that is rejected by the plausibility check, or
    an initial velocity of zero that is recorded as an assumption. It would not
    be bounded if this fed a zero-velocity *correction*, which is one reason
    Phase 4 does not have one.
    """
    if len(accel) == 0:
        raise ValueError("stationarity detection needs at least one accelerometer sample")
    if window_s <= 0.0:
        raise ValueError(f"window_s must be positive, got {window_s!r}")

    times = accel.analysis_time_s
    magnitudes = accel.norms
    notes: list[str] = []

    accel_std = _rolling_std(magnitudes, times, window_s)
    gravity_error = np.abs(_rolling_mean(magnitudes, times, window_s) - expected_gravity_mps2)

    if gyro is not None and len(gyro) == len(accel):
        gyro_magnitude = _rolling_mean(gyro.norms, times, window_s)
    else:
        gyro_magnitude = np.full(len(accel), np.nan)
        if gyro is None:
            notes.append("no gyroscope; the rotation condition was not applied")
        else:
            notes.append(
                f"gyroscope has {len(gyro)} samples against {len(accel)} accelerometer "
                "samples; the rotation condition was not applied"
            )

    quiet = np.isfinite(accel_std) & (accel_std <= accel_std_tolerance)
    near_gravity = np.isfinite(gravity_error) & (gravity_error <= gravity_error_tolerance)
    still = np.where(np.isfinite(gyro_magnitude), gyro_magnitude <= gyro_tolerance, True)
    mask = quiet & near_gravity & still

    span, duration = _longest_run(mask, times)
    fraction = float(mask.mean())
    confidence = (
        _confidence(accel_std[mask], gyro_magnitude[mask], accel_std_tolerance, gyro_tolerance)
        if mask.any()
        else 0.0
    )

    if not mask.any():
        notes.append(
            "no stationary interval found: the segment either never stops or the "
            "accelerometer magnitude never settles near gravity"
        )
    elif duration < MIN_STATIONARY_WINDOW_S:
        notes.append(
            f"the longest stationary run is {duration:.2f}s, below the "
            f"{MIN_STATIONARY_WINDOW_S:g}s minimum for a defensible bias estimate"
        )

    return StationarityResult(
        mask=mask,
        fraction=fraction,
        confidence=confidence,
        accel_std_mps2=accel_std,
        gyro_magnitude_rps=gyro_magnitude,
        gravity_error_mps2=gravity_error,
        window_s=window_s,
        longest_interval_s=duration,
        longest_span=span,
        notes=notes,
    )


def _longest_run(mask: np.ndarray, times: np.ndarray) -> tuple[tuple[int, int] | None, float]:
    """The longest contiguous True run, as a (start, stop) slice and a duration."""
    best_span: tuple[int, int] | None = None
    best_duration = 0.0
    index = 0
    size = mask.size
    while index < size:
        if not mask[index]:
            index += 1
            continue
        start = index
        while index < size and mask[index]:
            index += 1
        stop = index
        if stop - start >= 2:
            duration = float(times[stop - 1] - times[start])
        else:
            duration = 0.0
        if duration > best_duration:
            best_duration = duration
            best_span = (start, stop)
    return best_span, best_duration


def _confidence(
    accel_std: np.ndarray,
    gyro_magnitude: np.ndarray,
    accel_tolerance: float,
    gyro_tolerance: float,
) -> float:
    """How far inside the thresholds the stationary samples sit.

    A window that scrapes past both thresholds and one that is an order of
    magnitude quieter are both "stationary", and the difference matters when
    the answer is used to justify zeroing a velocity.
    """
    scores: list[float] = []
    finite_accel = accel_std[np.isfinite(accel_std)]
    if finite_accel.size:
        scores.append(1.0 - min(1.0, float(finite_accel.mean()) / accel_tolerance))
    finite_gyro = gyro_magnitude[np.isfinite(gyro_magnitude)]
    if finite_gyro.size:
        scores.append(1.0 - min(1.0, float(finite_gyro.mean()) / gyro_tolerance))
    if not scores:
        return 0.0
    return float(max(0.0, min(1.0, sum(scores) / len(scores))))


def stationary_duration_s(mask: np.ndarray, times: np.ndarray) -> float:
    """Total time spent stationary, summing only genuinely contiguous intervals."""
    stamps = np.asarray(times, dtype="float64")
    selection = np.asarray(mask, dtype=bool)
    if selection.size < 2:
        return 0.0
    both = selection[:-1] & selection[1:]
    intervals = np.diff(stamps)
    valid = both & np.isfinite(intervals) & (intervals > 0.0)
    total = float(intervals[valid].sum()) if valid.any() else 0.0
    return total if math.isfinite(total) else 0.0
