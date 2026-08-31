"""Calibration-session and canonical-data-access tests.

Two concerns:

* the session behaves as a state machine, reports honestly what it is still
  waiting for, and never returns an arbitrary rotation when the evidence is
  missing;
* Phase 3 reads Phase 2 canonical Parquet and respects the three timing rules
  — actual timestamps, measured (not assumed) rates, and duplicates handled by
  a stated policy applied to a derived copy.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from idr.frames import quaternion as quat
from idr.frames.alignment import CalibrationStatus
from idr.frames.calibration import CalibrationSession
from idr.frames.conventions import STANDARD_GRAVITY_MPS2, Frame
from idr.frames.sources import (
    IRREGULARITY_THRESHOLD,
    load_field_mapping,
    load_segment,
    measure_timing,
    strictly_increasing_view,
)
from idr.frames.synthetic import stationary_only, synthetic_drive
from idr.pipeline.canonical import QualityFlag

Z_AXIS = np.array([0.0, 0.0, 1.0])


def _feed(session: CalibrationSession, drive, with_speed: bool = True) -> None:
    session.consume_series(
        accel=drive.accel_phone,
        gyro=drive.gyro_phone,
        mag=drive.mag_phone,
        gravity=drive.gravity_phone,
        speed_mps=drive.speed_mps if with_speed else None,
        reference_yaw_rate_rps=drive.yaw_rate_rps,
    )


# --- session lifecycle -----------------------------------------------------


def test_a_session_starts_not_started_and_must_be_started() -> None:
    session = CalibrationSession()
    assert session.status() is CalibrationStatus.NOT_STARTED
    with pytest.raises(RuntimeError, match="call start"):
        session.consume_sample(analysis_time_s=0.0, accel=(0.0, 0.0, 9.8))
    with pytest.raises(RuntimeError, match="never started"):
        session.finish()


def test_a_finished_session_refuses_further_samples() -> None:
    session = CalibrationSession()
    session.start()
    _feed(session, synthetic_drive())
    session.finish()
    with pytest.raises(RuntimeError, match="already finished"):
        session.consume_sample(analysis_time_s=999.0, accel=(0.0, 0.0, 9.8))


def test_finish_is_idempotent() -> None:
    session = CalibrationSession()
    session.start()
    _feed(session, synthetic_drive())
    first, second = session.finish(), session.finish()
    assert first is second


def test_restarting_discards_the_previous_data() -> None:
    session = CalibrationSession()
    session.start()
    _feed(session, synthetic_drive())
    assert session.progress().sample_count > 0
    session.start()
    assert session.progress().sample_count == 0
    assert session.status() is CalibrationStatus.COLLECTING


# --- success path ----------------------------------------------------------


def test_a_full_drive_calibrates_successfully() -> None:
    truth = quat.from_euler(math.radians(12), math.radians(-8), math.radians(55))
    session = CalibrationSession()
    session.start()
    _feed(session, synthetic_drive(q_vehicle_phone=truth))
    result = session.finish()
    assert result.status is CalibrationStatus.SUCCESS
    assert result.rotation is not None
    assert math.degrees(quat.angular_distance(result.rotation.quaternion, truth)) < 1.0
    assert session.confidence() > 0.0


def test_the_streaming_and_block_paths_agree() -> None:
    """The offline diagnostics must exercise the same code the live path would."""
    drive = synthetic_drive(q_vehicle_phone=quat.from_axis_angle(Z_AXIS, 0.6))

    block = CalibrationSession()
    block.start()
    _feed(block, drive)
    block_result = block.finish()

    streamed = CalibrationSession()
    streamed.start()
    for index in range(len(drive)):
        streamed.consume_sample(
            analysis_time_s=float(drive.analysis_time_s[index]),
            accel=drive.accel_phone.samples[index],
            gyro=drive.gyro_phone.samples[index],
            mag=drive.mag_phone.samples[index],
            gravity=drive.gravity_phone.samples[index],
            speed_mps=float(drive.speed_mps[index]),
            reference_yaw_rate_rps=float(drive.yaw_rate_rps[index]),
        )
    stream_result = streamed.finish()

    assert stream_result.status is block_result.status
    assert stream_result.rotation is not None and block_result.rotation is not None
    assert quat.allclose(stream_result.rotation.quaternion, block_result.rotation.quaternion, 1e-12)


def test_estimate_does_not_end_the_session() -> None:
    session = CalibrationSession()
    session.start()
    _feed(session, synthetic_drive())
    session.estimate()
    assert session.status() is CalibrationStatus.COLLECTING
    session.consume_sample(analysis_time_s=1e6, accel=(0.0, 0.0, 9.8))


# --- progress reporting ----------------------------------------------------


def test_progress_reports_what_is_still_missing() -> None:
    session = CalibrationSession()
    session.start()
    _feed(session, stationary_only())
    progress = session.progress()
    assert progress.has_gravity
    assert progress.stationary_samples > 0
    assert progress.dynamic_samples == 0
    assert any("forward acceleration" in item for item in progress.waiting_for)
    assert "waiting for" in progress.describe()


def test_progress_is_clear_on_an_empty_session() -> None:
    session = CalibrationSession()
    session.start()
    progress = session.progress()
    assert progress.sample_count == 0
    assert not progress.has_gravity
    assert progress.waiting_for


def test_progress_counts_dynamic_samples_once_the_vehicle_moves() -> None:
    session = CalibrationSession()
    session.start()
    _feed(session, synthetic_drive())
    progress = session.progress()
    assert progress.dynamic_samples > 0
    assert progress.stationary_samples > 0
    assert progress.duration_s > 60.0


# --- failure and degenerate paths ------------------------------------------


def test_a_session_with_no_samples_fails_rather_than_returning_a_rotation() -> None:
    session = CalibrationSession()
    session.start()
    result = session.finish()
    assert result.status is CalibrationStatus.FAILED
    assert result.rotation is None
    assert result.confidence == 0.0


def test_a_stationary_session_reports_tilt_and_withholds_yaw() -> None:
    session = CalibrationSession()
    session.start()
    _feed(session, stationary_only())
    result = session.finish()
    assert result.status is CalibrationStatus.INSUFFICIENT_MOTION
    assert result.roll_deg is not None
    assert not result.yaw_resolved
    assert result.rotation is None


def test_a_session_without_speed_cannot_resolve_yaw() -> None:
    session = CalibrationSession()
    session.start()
    _feed(session, synthetic_drive(), with_speed=False)
    result = session.finish()
    assert not result.yaw_resolved
    assert result.rotation is None


def test_non_finite_samples_are_rejected_on_arrival_and_counted() -> None:
    session = CalibrationSession()
    session.start()
    session.consume_sample(analysis_time_s=math.nan, accel=(0.0, 0.0, 9.8))
    session.consume_sample(analysis_time_s=0.0, accel=(math.nan, 0.0, 9.8))
    _feed(session, synthetic_drive())
    result = session.finish()
    assert any("rejected on arrival" in note for note in result.notes)


def test_the_retention_cap_is_enforced() -> None:
    session = CalibrationSession(max_samples=50)
    session.start()
    _feed(session, synthetic_drive())
    assert session.progress().sample_count == 50


def test_a_short_but_clean_drive_is_reported_and_not_accepted() -> None:
    """Numerically fine, operationally thin: one acceleration event is not a
    calibration, so the yaw claim is withdrawn."""
    session = CalibrationSession(minimum_duration_s=600.0)
    session.start()
    _feed(session, synthetic_drive())
    result = session.finish()
    assert result.status is CalibrationStatus.LOW_HEADING_CONFIDENCE
    assert result.rotation is None
    assert not result.yaw_resolved
    assert any("below the" in note and "minimum" in note for note in result.notes)


def test_a_missing_magnetometer_is_declared_not_inferred() -> None:
    session = CalibrationSession(magnetometer_available=False)
    session.start()
    _feed(session, synthetic_drive())
    result = session.finish()
    assert result.quality.magnetometer_weight == 0.0


# --- timing measurement ----------------------------------------------------


def _frame(times: np.ndarray, flags: np.ndarray | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "analysis_time_s": times,
            "analysis_time_source": ["wall_clock"] * times.size,
            "quality_flags": np.zeros(times.size, dtype="int32") if flags is None else flags,
        }
    )


def test_the_measured_rate_comes_from_the_data_not_a_constant() -> None:
    """Phase 2 found families near 2, 10, 24 and 1000 Hz."""
    for rate in (2.0, 10.0, 24.0, 1000.0):
        timing = measure_timing(_frame(np.arange(500) / rate))
        assert timing.measured_rate_hz == pytest.approx(rate, rel=1e-9)
        assert timing.is_regular


def test_irregular_sampling_is_detected_and_reported() -> None:
    rng = np.random.default_rng(3)
    times = np.cumsum(rng.uniform(0.01, 0.5, size=400))
    timing = measure_timing(_frame(times))
    assert timing.irregularity is not None
    assert timing.irregularity > IRREGULARITY_THRESHOLD
    assert not timing.is_regular
    assert any("irregularly sampled" in note for note in timing.notes)


def test_duplicate_and_backward_timestamps_are_counted_not_hidden() -> None:
    times = np.array([0.0, 0.1, 0.1, 0.2, 0.15, 0.3])
    flags = np.array([0, 0, QualityFlag.DUPLICATE_TIMESTAMP.value, 0, 0, 0], dtype="int32")
    timing = measure_timing(_frame(times, flags))
    assert timing.non_positive_intervals == 2
    assert timing.duplicate_timestamp_rows == 1
    assert any("zero or negative" in note for note in timing.notes)


def test_timing_of_a_degenerate_segment() -> None:
    single = measure_timing(_frame(np.array([1.0])))
    assert single.median_interval_s is None
    assert "fewer than two" in single.notes[0]

    frozen = measure_timing(_frame(np.zeros(50)))
    assert frozen.measured_rate_hz is None
    assert any("does not advance" in note for note in frozen.notes)

    nan_times = measure_timing(_frame(np.full(10, np.nan)))
    assert nan_times.median_interval_s is None


def test_timing_records_which_clock_phase_two_chose() -> None:
    timing = measure_timing(_frame(np.arange(100) * 0.1))
    assert timing.analysis_time_source == "wall_clock"


# --- duplicate-timestamp policy --------------------------------------------


def test_the_deduplication_policy_keeps_the_first_row_and_says_so() -> None:
    frame = pd.DataFrame(
        {"analysis_time_s": [0.0, 0.1, 0.1, 0.2], "value": [10.0, 11.0, 99.0, 12.0]}
    )
    result = strictly_increasing_view(frame)
    assert result.removed_rows == 1
    assert list(result.frame_data["value"]) == [10.0, 11.0, 12.0]
    assert "first row" in result.policy
    # The input is untouched: the canonical data is never rewritten.
    assert len(frame) == 4


def test_deduplication_drops_backward_steps_and_non_finite_times() -> None:
    frame = pd.DataFrame({"analysis_time_s": [0.0, 0.2, 0.1, np.nan, 0.3]})
    result = strictly_increasing_view(frame)
    assert list(result.frame_data["analysis_time_s"]) == [0.0, 0.2, 0.3]
    assert result.removed_rows == 2


def test_deduplication_of_an_already_clean_frame_changes_nothing() -> None:
    frame = pd.DataFrame({"analysis_time_s": [0.0, 0.1, 0.2]})
    result = strictly_increasing_view(frame)
    assert result.removed_rows == 0
    assert len(result.frame_data) == 3


# --- canonical segment loading ---------------------------------------------


def _canonical_frame(count: int = 300) -> pd.DataFrame:
    times = np.arange(count, dtype="float64") * 0.1
    return pd.DataFrame(
        {
            "segment_id": ["abc123def456:000"] * count,
            "session_id": ["S1"] * count,
            "source_file": ["synth/S-S1.csv"] * count,
            "stream": ["smartphone"] * count,
            "analysis_time_s": times,
            "analysis_time_source": ["wall_clock"] * count,
            "source_time_s": times * 1000.0,
            "quality_flags": np.zeros(count, dtype="int32"),
            "accel_x": np.zeros(count),
            "accel_y": np.zeros(count),
            "accel_z": np.full(count, STANDARD_GRAVITY_MPS2),
            "gyro_x": np.zeros(count),
            "gyro_y": np.zeros(count),
            "gyro_z": np.zeros(count),
            "mag_x": np.zeros(count),
            "mag_y": np.full(count, 24.0),
            "mag_z": np.full(count, -41.6),
            "gravity_x": np.zeros(count),
            "gravity_y": np.zeros(count),
            "gravity_z": np.full(count, STANDARD_GRAVITY_MPS2),
            "gnss_speed_mps": np.zeros(count),
        }
    )


@pytest.fixture
def canonical_segment(tmp_path: Path) -> Path:
    path = tmp_path / "abc123def456_000.parquet"
    _canonical_frame().to_parquet(path, index=False)
    return path


def test_loading_a_canonical_segment_recovers_its_identity_and_timing(
    canonical_segment: Path,
) -> None:
    segment = load_segment(canonical_segment)
    assert segment.segment_id == "abc123def456:000"
    assert segment.session_id == "S1"
    assert segment.stream == "smartphone"
    assert segment.timing.measured_rate_hz == pytest.approx(10.0)
    assert segment.timing.analysis_time_source == "wall_clock"


def test_channels_come_back_tagged_with_the_phone_frame(canonical_segment: Path) -> None:
    segment = load_segment(canonical_segment)
    accel = segment.series("accel")
    assert accel is not None
    assert accel.frame is Frame.PHONE
    assert len(accel) == 300
    assert accel.source_time_s is not None, "provenance clock must survive"
    assert accel.quality_flags is not None


def test_an_absent_channel_returns_none_rather_than_a_nan_series(tmp_path: Path) -> None:
    """ "This file has no magnetometer" must not look like "it read NaN"."""
    frame = _canonical_frame()
    frame[["mag_x", "mag_y", "mag_z"]] = np.nan
    path = tmp_path / "nomag_000.parquet"
    frame.to_parquet(path, index=False)
    segment = load_segment(path)
    assert segment.series("mag") is None
    assert segment.availability["has_magnetometer"] is False
    assert segment.series("accel") is not None


def test_an_unknown_channel_is_a_programming_error(canonical_segment: Path) -> None:
    with pytest.raises(KeyError, match="unknown channel"):
        load_segment(canonical_segment).series("barometer")


def test_a_non_canonical_parquet_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "wrong.parquet"
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="no analysis_time_s"):
        load_segment(path)


def test_column_access_returns_none_for_a_missing_column(canonical_segment: Path) -> None:
    segment = load_segment(canonical_segment)
    assert segment.column("gnss_speed_mps") is not None
    assert segment.column("ref_velocity_mps") is None


def test_the_manifest_can_only_make_availability_stricter(
    canonical_segment: Path, tmp_path: Path
) -> None:
    """A manifest saying a sensor exists cannot resurrect an all-NaN channel."""
    mapping = {"synth/S-S1.csv": {"has_magnetometer": False, "has_accelerometer": True}}
    segment = load_segment(canonical_segment, field_mapping=mapping)
    assert segment.availability["has_magnetometer"] is False
    assert segment.series("mag") is None
    assert any("manifest says" in note for note in segment.notes)


def test_a_missing_field_mapping_artifact_is_not_fatal(tmp_path: Path) -> None:
    assert load_field_mapping(tmp_path) == {}


def test_field_mapping_parses_the_phase_two_artifact(tmp_path: Path) -> None:
    (tmp_path / "canonical_field_mapping.json").write_text(
        '{"per_file": [{"source_file": "a/S-1.csv", "stream": "smartphone", '
        '"has_magnetometer": false, "has_gravity": true}]}',
        encoding="utf-8",
    )
    mapping = load_field_mapping(tmp_path)
    assert mapping == {"a/S-1.csv": {"has_magnetometer": False, "has_gravity": True}}
