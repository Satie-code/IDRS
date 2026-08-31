"""Synthetic rotation and drive generation, for tests and robustness work.

Real IO-VNBD data cannot validate an alignment estimator, because the true
phone→vehicle rotation for those recordings is not documented anywhere. Only
synthetic data can: generate a drive in the *vehicle* frame where forward is
known to be +X by construction, rotate it into a phone frame by a rotation you
chose, and check that the estimator recovers the rotation you applied.

Two facilities:

* :func:`rotate_series` and :func:`rotation_sweep` — apply a controlled
  rotation to any sequence. This is the augmentation primitive the phase brief
  asks for, and later robustness work will reuse it directly.
* :func:`synthetic_drive` — a physically consistent vehicle-frame drive with
  acceleration, braking, cornering and a stationary interval, so alignment
  tests exercise the same code paths real data does.

Nothing here generates a training corpus. It builds fixtures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from idr.frames import quaternion as quat
from idr.frames.conventions import STANDARD_GRAVITY_MPS2, Frame
from idr.frames.rotations import FrameRotation
from idr.frames.vectors import VectorSeries

#: Rotation magnitudes the phase brief names, in degrees.
STANDARD_SWEEP_DEG = (0.0, 15.0, 30.0, 45.0, 90.0)


def rotate_series(series: VectorSeries, q: np.ndarray, to_frame: Frame) -> VectorSeries:
    """Apply one rotation to a whole series, retagging the frame.

    ``q`` is read as ``q_to_from``, consistent with the rest of the package:
    the result holds the same physical vectors expressed in ``to_frame``.
    """
    rotation = FrameRotation.from_quaternion(q, to_frame, series.frame)
    return rotation.apply_series(series)


def rotation_sweep(
    axis: np.ndarray | list[float], angles_deg: tuple[float, ...] = STANDARD_SWEEP_DEG
) -> list[tuple[float, np.ndarray]]:
    """(angle in degrees, quaternion) pairs for a controlled rotation sweep."""
    return [(angle, quat.from_axis_angle(axis, math.radians(angle))) for angle in angles_deg]


@dataclass
class SyntheticDrive:
    """A generated drive with every quantity known exactly.

    ``truth_q_vehicle_phone`` is the rotation the estimator is expected to
    recover. The phone-frame series are the vehicle-frame ones expressed in the
    phone frame, so applying the truth rotation to them returns the vehicle
    frame exactly.
    """

    analysis_time_s: np.ndarray
    speed_mps: np.ndarray
    yaw_rate_rps: np.ndarray
    accel_phone: VectorSeries
    gyro_phone: VectorSeries
    mag_phone: VectorSeries
    gravity_phone: VectorSeries
    accel_vehicle: VectorSeries
    truth_q_vehicle_phone: np.ndarray
    stationary_mask: np.ndarray
    notes: list[str] = field(default_factory=list)

    @property
    def truth_rotation(self) -> FrameRotation:
        return FrameRotation.from_quaternion(self.truth_q_vehicle_phone, Frame.VEHICLE, Frame.PHONE)

    def __len__(self) -> int:
        return int(self.analysis_time_s.size)


def synthetic_drive(
    *,
    q_vehicle_phone: np.ndarray | None = None,
    rate_hz: float = 10.0,
    stationary_s: float = 15.0,
    driving_s: float = 60.0,
    cruise_speed_mps: float = 15.0,
    jitter_s: float = 0.0,
    noise_mps2: float = 0.0,
    magnetic_field_ut: float = 48.0,
    seed: int = 20260831,
) -> SyntheticDrive:
    """Build a drive whose phone→vehicle rotation is known by construction.

    The profile is deliberately ordinary: a stationary interval so tilt is
    observable, then repeated acceleration and braking so the forward axis is
    observable, with a sinusoidal yaw rate so lateral behaviour is exercised.

    ``jitter_s`` perturbs the sample times, which is how the irregular-sampling
    path gets tested: the estimator must use actual Δt, so a drive whose
    timestamps wobble should still be solved correctly.

    The vehicle frame is X forward, Y left, Z up, so an accelerating vehicle
    shows specific force ``a_x = dv/dt``, gravity contributes ``−g`` on Z, and
    a left turn at yaw rate ω gives lateral specific force ``+v·ω`` on Y.
    """
    if rate_hz <= 0.0:
        raise ValueError(f"rate_hz must be positive, got {rate_hz!r}")
    rng = np.random.default_rng(seed)
    q_truth = quat.identity() if q_vehicle_phone is None else quat.normalize(q_vehicle_phone)

    step = 1.0 / rate_hz
    n_still = max(1, int(round(stationary_s * rate_hz)))
    n_drive = max(1, int(round(driving_s * rate_hz)))
    count = n_still + n_drive

    times = np.arange(count, dtype="float64") * step
    if jitter_s > 0.0:
        # Jitter, then re-sort: a logger delivers late samples, it does not
        # deliver them out of order, and Phase 2 would have sorted them anyway.
        times = np.sort(times + rng.uniform(-jitter_s, jitter_s, size=count))

    # --- speed profile ----------------------------------------------------
    speed = np.zeros(count, dtype="float64")
    drive_t = times[n_still:] - times[n_still]
    # Three accelerate/cruise/brake cycles, so the forward axis is supported by
    # several independent events rather than one.
    period = max(driving_s / 3.0, 1e-6)
    phase = (drive_t % period) / period
    profile = np.where(
        phase < 0.35,
        phase / 0.35,
        np.where(phase < 0.65, 1.0, np.clip(1.0 - (phase - 0.65) / 0.35, 0.0, 1.0)),
    )
    speed[n_still:] = cruise_speed_mps * profile

    # --- yaw rate ---------------------------------------------------------
    yaw_rate = np.zeros(count, dtype="float64")
    yaw_rate[n_still:] = 0.15 * np.sin(2.0 * math.pi * drive_t / max(driving_s / 2.0, 1e-6))
    yaw_rate[speed < 1.0] = 0.0

    # --- vehicle-frame specific force -------------------------------------
    longitudinal = np.gradient(speed, times)
    lateral = speed * yaw_rate
    accel_vehicle = np.column_stack([longitudinal, lateral, np.full(count, STANDARD_GRAVITY_MPS2)])
    # Specific force includes the reaction to gravity: a vehicle at rest reads
    # +g on its up axis, not zero. Getting this sign wrong would make every
    # gravity test pass against an upside-down convention.
    if noise_mps2 > 0.0:
        accel_vehicle = accel_vehicle + rng.normal(0.0, noise_mps2, size=accel_vehicle.shape)

    gyro_vehicle = np.column_stack([np.zeros(count), np.zeros(count), yaw_rate])
    # The gravity *channel* follows the Android convention, reading the same as
    # the accelerometer at rest — i.e. +g on the up axis, pointing away from
    # gravity. Verified against the real dataset (see
    # idr.frames.gravity.SPECIFIC_FORCE_OPPOSES_GRAVITY); generating it with
    # the opposite sign would make the fixture disagree with the data it stands
    # in for, and would hide a sign error rather than expose one.
    gravity_vehicle = np.tile(np.array([0.0, 0.0, STANDARD_GRAVITY_MPS2]), (count, 1))

    # Earth field: dip 60° below horizontal, pointing magnetic north (+Y).
    dip = math.radians(60.0)
    field_vehicle = np.tile(
        np.array([0.0, magnetic_field_ut * math.cos(dip), -magnetic_field_ut * math.sin(dip)]),
        (count, 1),
    )

    # --- express in the phone frame ---------------------------------------
    # q_truth is q_vehicle_phone, so its inverse takes vehicle → phone.
    q_phone_vehicle = quat.conjugate(q_truth)

    def to_phone(values: np.ndarray, name: str) -> VectorSeries:
        return VectorSeries(
            samples=quat.rotate_vectors(q_phone_vehicle, values),
            analysis_time_s=times,
            frame=Frame.PHONE,
            source=f"synthetic:{name}",
        )

    stationary = np.zeros(count, dtype=bool)
    stationary[:n_still] = True

    return SyntheticDrive(
        analysis_time_s=times,
        speed_mps=speed,
        yaw_rate_rps=yaw_rate,
        accel_phone=to_phone(accel_vehicle, "accel"),
        gyro_phone=to_phone(gyro_vehicle, "gyro"),
        mag_phone=to_phone(field_vehicle, "mag"),
        gravity_phone=to_phone(gravity_vehicle, "gravity"),
        accel_vehicle=VectorSeries(
            samples=accel_vehicle,
            analysis_time_s=times,
            frame=Frame.VEHICLE,
            source="synthetic:accel_vehicle",
        ),
        truth_q_vehicle_phone=q_truth,
        stationary_mask=stationary,
        notes=[
            f"{count} samples at a nominal {rate_hz} Hz "
            f"({'jittered' if jitter_s > 0 else 'regular'})",
            f"{stationary_s:.0f}s stationary then {driving_s:.0f}s driving",
        ],
    )


def stationary_only(
    *,
    q_vehicle_phone: np.ndarray | None = None,
    stationary_s: float = 30.0,
    rate_hz: float = 10.0,
    **kwargs: float,
) -> SyntheticDrive:
    """A drive that never moves — the case where yaw must stay unresolved."""
    drive = synthetic_drive(
        q_vehicle_phone=q_vehicle_phone,
        rate_hz=rate_hz,
        stationary_s=stationary_s,
        driving_s=0.1,
        cruise_speed_mps=0.0,
        **kwargs,  # type: ignore[arg-type]
    )
    drive.notes.append("stationary throughout: no heading information exists")
    return drive


def with_duplicate_timestamps(drive: SyntheticDrive, count: int = 5) -> SyntheticDrive:
    """Copy a drive with some timestamps repeated, as Phase 2 leaves them.

    Phase 2 flags duplicate timestamps and keeps the rows. Anything that
    differentiates time must therefore cope with a Δt of exactly zero, and this
    fixture is how that is tested rather than assumed.
    """
    times = drive.analysis_time_s.copy()
    if times.size > count + 1:
        for index in range(1, count + 1):
            times[index * 2] = times[index * 2 - 1]

    def retimed(series: VectorSeries) -> VectorSeries:
        return VectorSeries(
            samples=series.samples,
            analysis_time_s=times,
            frame=series.frame,
            source=series.source,
            notes=[*series.notes, "timestamps duplicated for testing"],
        )

    return SyntheticDrive(
        analysis_time_s=times,
        speed_mps=drive.speed_mps,
        yaw_rate_rps=drive.yaw_rate_rps,
        accel_phone=retimed(drive.accel_phone),
        gyro_phone=retimed(drive.gyro_phone),
        mag_phone=retimed(drive.mag_phone),
        gravity_phone=retimed(drive.gravity_phone),
        accel_vehicle=retimed(drive.accel_vehicle),
        truth_q_vehicle_phone=drive.truth_q_vehicle_phone,
        stationary_mask=drive.stationary_mask,
        notes=[*drive.notes, f"{count} duplicate timestamp(s) injected"],
    )


def disturbed_magnetometer(drive: SyntheticDrive, offset_ut: float = 80.0) -> SyntheticDrive:
    """Copy a drive whose magnetometer carries a large steady bias.

    This is the realistic in-vehicle failure: a *constant* field added in the
    phone frame. It leaves the field looking perfectly stable while making the
    heading it implies wrong, which is why the quality checks look at magnitude
    and dip rather than at stability alone.
    """
    biased = drive.mag_phone.samples + np.array([offset_ut, 0.0, 0.0])
    return SyntheticDrive(
        analysis_time_s=drive.analysis_time_s,
        speed_mps=drive.speed_mps,
        yaw_rate_rps=drive.yaw_rate_rps,
        accel_phone=drive.accel_phone,
        gyro_phone=drive.gyro_phone,
        mag_phone=VectorSeries(
            samples=biased,
            analysis_time_s=drive.analysis_time_s,
            frame=Frame.PHONE,
            source="synthetic:mag_disturbed",
        ),
        gravity_phone=drive.gravity_phone,
        accel_vehicle=drive.accel_vehicle,
        truth_q_vehicle_phone=drive.truth_q_vehicle_phone,
        stationary_mask=drive.stationary_mask,
        notes=[*drive.notes, f"magnetometer biased by {offset_ut} µT along phone X"],
    )
