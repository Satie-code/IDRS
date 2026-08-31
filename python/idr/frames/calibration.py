"""The calibration session: an incremental, stateful front end to alignment.

:mod:`idr.frames.alignment` answers "given this block of data, what is the
phone→vehicle rotation?". This module answers the operational question around
it: "the driver just started calibration — what should we tell them, and when
do we have enough?"

The distinction matters because the two have different shapes. Alignment is a
pure function over a window. A calibration is a *conversation*: it starts, it
accumulates, it can tell you what it is still waiting for, and it ends in a
named outcome. Later phases will drive it from a live sensor stream and from
recorded data alike, so it is written to consume samples one at a time and to
be queryable at any point.

The protocol it is built around, from the phase brief:

    stationary phase → begin moving → estimate forward direction → refine yaw
    → validate alignment

The stationary phase is not required — a session started mid-drive works — but
it is useful, because tilt converges quickly and cheaply when the vehicle is
still, and the session can then report roll and pitch while it is still waiting
for the motion that yaw needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from idr.frames.alignment import (
    MIN_FORWARD_SAMPLES,
    MIN_HEADING_SPEED_MPS,
    MIN_LONGITUDINAL_ACCEL_MPS2,
    AlignmentEstimate,
    CalibrationStatus,
    estimate_alignment,
)
from idr.frames.conventions import Frame
from idr.frames.gravity import (
    LOW_DYNAMIC_ACCEL_TOLERANCE_MPS2,
    MIN_GRAVITY_SAMPLES,
    STANDARD_GRAVITY_MPS2,
    GravityEstimate,
    estimate_gravity,
    linear_acceleration,
)
from idr.frames.magnetometer import MagnetometerQuality, assess_magnetometer
from idr.frames.vectors import VectorSeries

#: Default minimum wall-clock duration before a session will declare success.
#: Shorter windows can produce a numerically fine answer from a single
#: acceleration event, which is precisely the case the split-half check exists
#: to distrust.
MIN_CALIBRATION_DURATION_S = 20.0

#: Hard cap on retained samples, so a long drive cannot grow the session
#: without bound. At 100 Hz this is ten minutes, far more than calibration
#: needs, and it keeps the memory profile flat for eventual on-device use.
MAX_RETAINED_SAMPLES = 60_000


@dataclass
class CalibrationProgress:
    """What the session has, and what it is still waiting for.

    Intended to be surfaced to a user: every field maps to something they can
    act on ("keep driving", "drive straight", "move the phone away from the
    speaker") rather than to an internal counter.
    """

    status: CalibrationStatus
    sample_count: int
    duration_s: float
    stationary_samples: int
    dynamic_samples: int
    has_gravity: bool
    gravity_confidence: float
    waiting_for: list[str] = field(default_factory=list)

    def describe(self) -> str:
        if not self.waiting_for:
            return f"{self.status}: ready"
        return f"{self.status}: waiting for {'; '.join(self.waiting_for)}"


class CalibrationSession:
    """Accumulate samples and estimate a phone→vehicle alignment.

    Usage::

        session = CalibrationSession()
        session.start()
        for row in stream:
            session.consume_sample(
                analysis_time_s=row.t, accel=row.accel, gyro=row.gyro,
                mag=row.mag, speed_mps=row.speed,
            )
        result = session.finish()

    :meth:`status`, :meth:`progress` and :meth:`estimate` may be called at any
    point; none of them consumes or resets anything.
    """

    def __init__(
        self,
        *,
        minimum_duration_s: float = MIN_CALIBRATION_DURATION_S,
        max_samples: int = MAX_RETAINED_SAMPLES,
        magnetometer_available: bool = True,
    ) -> None:
        self.minimum_duration_s = minimum_duration_s
        self.max_samples = max_samples
        self.magnetometer_available = magnetometer_available
        self._status = CalibrationStatus.NOT_STARTED
        self._result: AlignmentEstimate | None = None
        self._reset_buffers()

    def _reset_buffers(self) -> None:
        self._times: list[float] = []
        self._accel: list[tuple[float, float, float]] = []
        self._gyro: list[tuple[float, float, float] | None] = []
        self._mag: list[tuple[float, float, float] | None] = []
        self._gravity_channel: list[tuple[float, float, float] | None] = []
        self._speed: list[float] = []
        self._speed_valid: list[bool] = []
        self._reference_yaw_rate: list[float] = []
        self._dropped_samples = 0
        self._notes: list[str] = []

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        """Begin (or restart) a session, discarding anything already held."""
        self._reset_buffers()
        self._status = CalibrationStatus.COLLECTING
        self._result = None

    def consume_sample(
        self,
        *,
        analysis_time_s: float,
        accel: tuple[float, float, float] | np.ndarray,
        gyro: tuple[float, float, float] | np.ndarray | None = None,
        mag: tuple[float, float, float] | np.ndarray | None = None,
        gravity: tuple[float, float, float] | np.ndarray | None = None,
        speed_mps: float | None = None,
        speed_is_valid: bool = True,
        reference_yaw_rate_rps: float | None = None,
    ) -> None:
        """Add one sample.

        ``analysis_time_s`` must be the Phase 2 analysis clock for the stream,
        not ``timestamp_utc`` and not a sample index — the estimator
        differentiates it, and a sample index would silently assert a fixed
        rate that the dataset does not have.

        A sample with a non-finite time or accelerometer is counted and
        discarded rather than stored: it cannot contribute, and keeping it
        would force every downstream loop to re-check.
        """
        if self._status is CalibrationStatus.NOT_STARTED:
            raise RuntimeError("call start() before consuming samples")
        if self._status.is_terminal:
            raise RuntimeError(f"session already finished with status {self._status}")

        accel_tuple = _triple(accel)
        if not math.isfinite(analysis_time_s) or accel_tuple is None:
            self._dropped_samples += 1
            return
        if len(self._times) >= self.max_samples:
            self._dropped_samples += 1
            return

        self._times.append(float(analysis_time_s))
        self._accel.append(accel_tuple)
        self._gyro.append(_triple(gyro))
        self._mag.append(_triple(mag))
        self._gravity_channel.append(_triple(gravity))
        self._speed.append(float(speed_mps) if speed_mps is not None else math.nan)
        self._speed_valid.append(bool(speed_is_valid and speed_mps is not None))
        self._reference_yaw_rate.append(
            float(reference_yaw_rate_rps) if reference_yaw_rate_rps is not None else math.nan
        )

    def consume_series(
        self,
        *,
        accel: VectorSeries,
        gyro: VectorSeries | None = None,
        mag: VectorSeries | None = None,
        gravity: VectorSeries | None = None,
        speed_mps: np.ndarray | None = None,
        speed_is_valid: np.ndarray | None = None,
        reference_yaw_rate_rps: np.ndarray | None = None,
    ) -> None:
        """Add a whole block at once — the offline path.

        Equivalent to calling :meth:`consume_sample` for each row, and kept
        deliberately equivalent: a test asserts the two paths agree, so the
        offline diagnostics exercise the same code the live path would.
        """
        for index in range(len(accel)):
            self.consume_sample(
                analysis_time_s=float(accel.analysis_time_s[index]),
                accel=accel.samples[index],
                gyro=None if gyro is None or len(gyro) != len(accel) else gyro.samples[index],
                mag=None if mag is None or len(mag) != len(accel) else mag.samples[index],
                gravity=(
                    None
                    if gravity is None or len(gravity) != len(accel)
                    else gravity.samples[index]
                ),
                speed_mps=(
                    None
                    if speed_mps is None or len(speed_mps) != len(accel)
                    else float(speed_mps[index])
                ),
                speed_is_valid=(
                    True
                    if speed_is_valid is None or len(speed_is_valid) != len(accel)
                    else bool(speed_is_valid[index])
                ),
                reference_yaw_rate_rps=(
                    None
                    if reference_yaw_rate_rps is None or len(reference_yaw_rate_rps) != len(accel)
                    else float(reference_yaw_rate_rps[index])
                ),
            )

    def status(self) -> CalibrationStatus:
        return self._status

    def confidence(self) -> float:
        """Confidence of the current estimate, 0 while nothing is estimable."""
        estimate = self._result if self._result is not None else self.estimate()
        return estimate.confidence

    # -- inspection -------------------------------------------------------
    def progress(self) -> CalibrationProgress:
        """A snapshot of what is held and what is still missing."""
        count = len(self._times)
        duration = (self._times[-1] - self._times[0]) if count >= 2 else 0.0
        accel = self._accel_series()
        stationary = dynamic = 0
        gravity_estimate: GravityEstimate | None = None

        if accel is not None:
            norms = accel.norms
            stationary = int(
                (np.abs(norms - STANDARD_GRAVITY_MPS2) <= LOW_DYNAMIC_ACCEL_TOLERANCE_MPS2).sum()
            )
            dynamic = self._dynamic_sample_count()
            gravity_estimate = estimate_gravity(
                accel, gravity_channel=self._gravity_series(), gyro=self._gyro_series()
            )

        waiting: list[str] = []
        if count < MIN_GRAVITY_SAMPLES:
            waiting.append(f"more samples ({count}/{MIN_GRAVITY_SAMPLES} minimum)")
        if gravity_estimate is not None and not gravity_estimate.is_available:
            waiting.append("a quasi-static interval to fix the vertical")
        if duration < self.minimum_duration_s:
            waiting.append(f"{self.minimum_duration_s - duration:.0f}s more data")
        if dynamic < MIN_FORWARD_SAMPLES:
            waiting.append(
                f"forward acceleration above {MIN_LONGITUDINAL_ACCEL_MPS2} m/s² while moving "
                f"faster than {MIN_HEADING_SPEED_MPS} m/s ({dynamic}/{MIN_FORWARD_SAMPLES})"
            )

        return CalibrationProgress(
            status=self._status,
            sample_count=count,
            duration_s=duration,
            stationary_samples=stationary,
            dynamic_samples=dynamic,
            has_gravity=bool(gravity_estimate and gravity_estimate.is_available),
            gravity_confidence=gravity_estimate.confidence if gravity_estimate else 0.0,
            waiting_for=waiting,
        )

    def estimate(self) -> AlignmentEstimate:
        """Estimate from everything held so far, without ending the session."""
        accel = self._accel_series()
        if accel is None or len(accel) == 0:
            return _empty_estimate(
                CalibrationStatus.FAILED,
                "no usable samples were collected",
                self._dropped_samples,
            )

        gravity = estimate_gravity(
            accel, gravity_channel=self._gravity_series(), gyro=self._gyro_series()
        )
        magnetometer = self._magnetometer_quality(gravity)

        linear = None
        if gravity.is_available:
            linear = linear_acceleration(accel, gravity)

        duration = float(accel.analysis_time_s[-1] - accel.analysis_time_s[0])
        speed = np.asarray(self._speed, dtype="float64")
        has_speed = bool(np.isfinite(speed).any())

        estimate = estimate_alignment(
            gravity=gravity,
            linear_accel=linear,
            speed_mps=speed if has_speed else None,
            valid_speed=np.asarray(self._speed_valid, dtype=bool) if has_speed else None,
            gyro=self._gyro_series(),
            reference_yaw_rate_rps=self._reference_yaw_rate_array(),
            magnetometer=magnetometer,
            duration_s=duration,
        )

        notes = list(estimate.notes)
        if self._dropped_samples:
            notes.append(
                f"{self._dropped_samples} sample(s) were rejected on arrival "
                "(non-finite time or accelerometer, or the retention cap)"
            )
        if estimate.status is CalibrationStatus.SUCCESS and duration < self.minimum_duration_s:
            # Numerically fine, operationally thin. Downgrade rather than
            # accept an alignment fitted to one acceleration event.
            notes.append(
                f"only {duration:.1f}s of data, below the {self.minimum_duration_s:.0f}s "
                "minimum; the alignment is reported but not accepted"
            )
            estimate = _downgrade(estimate, CalibrationStatus.LOW_HEADING_CONFIDENCE, notes)
        else:
            estimate.notes = notes
        return estimate

    def finish(self) -> AlignmentEstimate:
        """End the session and return the final estimate."""
        if self._status is CalibrationStatus.NOT_STARTED:
            raise RuntimeError("cannot finish a session that was never started")
        if self._result is None:
            self._result = self.estimate()
            self._status = self._result.status
        return self._result

    # -- internals --------------------------------------------------------
    def _series(self, values: list[Any], frame: Frame = Frame.PHONE) -> VectorSeries | None:
        if not self._times or all(value is None for value in values):
            return None
        samples = np.array(
            [(math.nan, math.nan, math.nan) if v is None else v for v in values], dtype="float64"
        )
        return VectorSeries(
            samples=samples,
            analysis_time_s=np.asarray(self._times, dtype="float64"),
            frame=frame,
            source="calibration_session",
        )

    def _accel_series(self) -> VectorSeries | None:
        return self._series(list(self._accel))

    def _gyro_series(self) -> VectorSeries | None:
        return self._series(self._gyro)

    def _mag_series(self) -> VectorSeries | None:
        return self._series(self._mag)

    def _gravity_series(self) -> VectorSeries | None:
        return self._series(self._gravity_channel)

    def _reference_yaw_rate_array(self) -> np.ndarray | None:
        array = np.asarray(self._reference_yaw_rate, dtype="float64")
        return array if np.isfinite(array).any() else None

    def _magnetometer_quality(self, gravity: GravityEstimate) -> MagnetometerQuality:
        return assess_magnetometer(
            self._mag_series(), gravity=gravity, available=self.magnetometer_available
        )

    def _dynamic_sample_count(self) -> int:
        """Samples that could contribute to the forward-direction estimate."""
        if len(self._times) < 3:
            return 0
        speed = np.asarray(self._speed, dtype="float64")
        times = np.asarray(self._times, dtype="float64")
        if not np.isfinite(speed).any():
            return 0
        rate = np.full(speed.size, np.nan, dtype="float64")
        dt = times[2:] - times[:-2]
        usable = np.isfinite(dt) & (dt > 0.0)
        rate[1:-1] = np.where(usable, (speed[2:] - speed[:-2]) / np.where(usable, dt, 1.0), np.nan)
        qualifying = (
            np.isfinite(rate)
            & (np.abs(rate) >= MIN_LONGITUDINAL_ACCEL_MPS2)
            & (speed >= MIN_HEADING_SPEED_MPS)
            & np.asarray(self._speed_valid, dtype=bool)
        )
        return int(qualifying.sum())


def _triple(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype="float64").reshape(-1)
    if array.size != 3 or not np.isfinite(array).all():
        return None
    return float(array[0]), float(array[1]), float(array[2])


def _downgrade(
    estimate: AlignmentEstimate, status: CalibrationStatus, notes: list[str]
) -> AlignmentEstimate:
    """Re-report an estimate at a lower status, withdrawing the yaw claim."""
    estimate.status = status
    estimate.rotation = None
    estimate.yaw_resolved = False
    estimate.yaw_deg = None
    estimate.notes = notes
    return estimate


def _empty_estimate(status: CalibrationStatus, reason: str, dropped: int) -> AlignmentEstimate:
    from idr.frames.alignment import AlignmentQuality

    notes = [reason]
    if dropped:
        notes.append(f"{dropped} sample(s) were rejected on arrival")
    return AlignmentEstimate(
        status=status,
        rotation=None,
        tilt_only_rotation=None,
        up_direction_phone=None,
        forward_direction_phone=None,
        roll_deg=None,
        pitch_deg=None,
        yaw_deg=None,
        yaw_resolved=False,
        confidence=0.0,
        quality=AlignmentQuality(
            gravity_confidence=0.0,
            gravity_residual_deg=None,
            forward_confidence=0.0,
            forward_correlation=None,
            longitudinal_correlation=None,
            lateral_correlation=None,
            angular_rate_consistency=None,
            magnetometer_weight=0.0,
            dynamic_sample_count=0,
            total_sample_count=0,
            duration_s=0.0,
            forward_lag_s=None,
        ),
        sample_count=0,
        duration_s=0.0,
        notes=notes,
    )
