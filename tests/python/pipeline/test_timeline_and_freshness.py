from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from idr.pipeline.canonical import QualityFlag
from idr.pipeline.freshness import annotate_freshness
from idr.pipeline.reader import read_canonical
from idr.pipeline.timeline import diagnose_and_segment, gap_threshold_for


def _prepare(path: Path, stream: str = "smartphone"):
    read = read_canonical(path, path.name, stream, dataset_family="f", session_id="T1")
    return diagnose_and_segment(
        read.frame,
        source_file=path.name,
        session_id="T1",
        stream=stream,
        dataset_family="f",
        payload_hash=read.payload_hash,
    )


# --- segmentation ----------------------------------------------------------


def test_clean_file_is_one_segment(smartphone_csv: Path) -> None:
    _, timing, segments = _prepare(smartphone_csv)
    assert timing.segment_count == 1
    assert len(segments) == 1
    assert segments[0].segmentation_reason == "first segment of file"


def test_large_gap_splits_into_two_segments(smartphone_with_gap: Path) -> None:
    frame, timing, segments = _prepare(smartphone_with_gap)
    assert timing.segment_count == 2
    assert timing.large_gap_count == 1
    assert segments[1].gap_before_s == pytest.approx(600.0, abs=1.0)
    assert "forward gap" in segments[1].segmentation_reason
    # Rows are partitioned, never lost.
    assert sum(s.row_count for s in segments) == len(frame)


def test_segments_get_distinct_ids(smartphone_with_gap: Path) -> None:
    frame, _, segments = _prepare(smartphone_with_gap)
    ids = {s.segment_id for s in segments}
    assert len(ids) == 2
    assert set(frame["segment_id"].unique()) == ids


def test_counter_reset_does_not_split_a_continuous_drive(
    smartphone_with_counter_reset: Path,
) -> None:
    """The elapsed counter restarts but the wall clock runs on: one trip."""
    _, timing, segments = _prepare(smartphone_with_counter_reset)
    assert timing.source_counter_reset_count == 1
    assert timing.clock_reset_count == 0
    assert timing.segment_count == 1
    assert timing.reordered is False


def test_counter_reset_is_still_reported(smartphone_with_counter_reset: Path) -> None:
    _, timing, _ = _prepare(smartphone_with_counter_reset)
    assert any("source counter" in note for note in timing.notes)


def test_elapsed_is_measured_from_each_segment_start(smartphone_with_gap: Path) -> None:
    frame, _, segments = _prepare(smartphone_with_gap)
    for segment in segments:
        span = frame.iloc[segment.start_row : segment.end_row + 1]
        assert span["elapsed_s"].iloc[0] == pytest.approx(0.0, abs=1e-6)
        assert (span["elapsed_s"].diff().dropna() >= -1e-9).all()


def test_segment_boundary_rows_are_flagged(smartphone_with_gap: Path) -> None:
    frame, _, segments = _prepare(smartphone_with_gap)
    flags = frame["quality_flags"].to_numpy()
    for segment in segments:
        assert flags[segment.start_row] & QualityFlag.SEGMENT_BOUNDARY.value


def test_gap_threshold_scales_with_cadence() -> None:
    # A 10 Hz stream: 100 x 0.1 s = 10 s, which meets the floor.
    assert gap_threshold_for(0.1) == pytest.approx(10.0)
    # A 2 Hz stream needs a proportionally larger gap.
    assert gap_threshold_for(0.5) == pytest.approx(50.0)
    # A very fast stream is protected by the absolute floor.
    assert gap_threshold_for(0.001) == pytest.approx(10.0)
    assert gap_threshold_for(None) is None


def test_segmentation_is_deterministic(smartphone_with_gap: Path) -> None:
    first = _prepare(smartphone_with_gap)[2]
    second = _prepare(smartphone_with_gap)[2]
    assert [s.segment_id for s in first] == [s.segment_id for s in second]
    assert [s.start_row for s in first] == [s.start_row for s in second]


# --- duplicates and ordering ----------------------------------------------


def test_duplicate_timestamps_are_flagged_not_dropped(tmp_path: Path) -> None:
    from tests.python.pipeline.conftest import SMARTPHONE_HEADER, smartphone_row

    rows = [smartphone_row(0), smartphone_row(0), smartphone_row(1)]
    path = tmp_path / "S-DUP.csv"
    path.write_text("\n".join([SMARTPHONE_HEADER, *rows]) + "\n", encoding="utf-8")

    frame, timing, _ = _prepare(path)
    assert len(frame) == 3, "no row may be dropped"
    assert timing.duplicate_timestamp_rows == 2
    assert timing.duplicate_groups == 1
    flags = frame["quality_flags"].to_numpy()
    assert int((flags & QualityFlag.DUPLICATE_TIMESTAMP.value).astype(bool).sum()) == 2


def test_row_count_is_preserved_through_segmentation(smartphone_with_gap: Path) -> None:
    read = read_canonical(
        smartphone_with_gap, "S-GAP.csv", "smartphone", dataset_family="f", session_id="T1"
    )
    frame, _, _ = _prepare(smartphone_with_gap)
    assert len(frame) == read.row_count


# --- freshness -------------------------------------------------------------


def test_every_row_has_a_fix_but_not_every_row_is_fresh(smartphone_stale_gnss: Path) -> None:
    frame, _, _ = _prepare(smartphone_stale_gnss)
    annotated, report = annotate_freshness(frame, source_file="S-STALE.csv", stream="smartphone")
    assert report.availability_fraction == pytest.approx(1.0)
    # The position was held for the first 200 rows, so far fewer updates than rows.
    assert report.position_update_count < report.row_count
    assert annotated["gnss_position_fresh"].sum() == report.position_update_count


def test_stale_rows_are_flagged(smartphone_stale_gnss: Path) -> None:
    frame, _, _ = _prepare(smartphone_stale_gnss)
    annotated, report = annotate_freshness(frame, source_file="S-STALE.csv", stream="smartphone")
    assert report.stale_rows > 0
    flags = annotated["quality_flags"].to_numpy()
    assert int((flags & QualityFlag.GNSS_STALE.value).astype(bool).sum()) == report.stale_rows


def test_stale_duration_resets_on_a_fresh_fix(smartphone_stale_gnss: Path) -> None:
    frame, _, _ = _prepare(smartphone_stale_gnss)
    annotated, _ = annotate_freshness(frame, source_file="S-STALE.csv", stream="smartphone")
    fresh = annotated["gnss_position_fresh"].fillna(False).to_numpy(dtype=bool)
    durations = annotated["gnss_stale_duration_s"].to_numpy(dtype="float64")
    assert np.allclose(durations[fresh], 0.0, atol=1e-6)


def test_freshness_never_reports_a_confirmed_outage(smartphone_stale_gnss: Path) -> None:
    frame, _, _ = _prepare(smartphone_stale_gnss)
    _, report = annotate_freshness(frame, source_file="S-STALE.csv", stream="smartphone")
    assert report.confirmed_outages == 0


def test_zero_zero_position_is_not_a_fix(tmp_path: Path) -> None:
    from tests.python.pipeline.conftest import SMARTPHONE_HEADER, smartphone_row

    rows = [smartphone_row(0), smartphone_row(1, lat=0.0, lon=0.0), smartphone_row(2)]
    path = tmp_path / "S-ZERO.csv"
    path.write_text("\n".join([SMARTPHONE_HEADER, *rows]) + "\n", encoding="utf-8")
    frame, _, _ = _prepare(path)
    annotated, _ = annotate_freshness(frame, source_file="S-ZERO.csv", stream="smartphone")
    assert list(annotated["gnss_fix_available"]) == [True, False, True]


def test_counter_reset_is_a_trip_boundary_when_no_wall_clock_disproves_it(
    smartphone_no_date_reset: Path,
) -> None:
    """The mirror of the S-S2 case.

    ``smartphone_with_counter_reset`` must NOT split, because its wall clock
    proves the recording ran on. Remove that wall clock and the same reset is
    the only evidence there is, so it must split — and the two trips must never
    be sorted into each other.
    """
    frame, timing, segments = _prepare(smartphone_no_date_reset)
    assert timing.clock_reset_count == 1
    assert timing.segment_count == 2
    assert len(segments) == 2
    assert "clock reset" in segments[1].segmentation_reason
    assert timing.reordered is False
    assert sum(s.row_count for s in segments) == len(frame)
    # Each trip keeps its own rising clock; the trips are not interleaved.
    for segment in segments:
        span = frame["analysis_time_s"].to_numpy()[segment.start_row : segment.end_row + 1]
        assert np.all(np.diff(span) > 0)


def test_a_single_row_yields_no_measurable_cadence(smartphone_single_row: Path) -> None:
    """One sample cannot imply an interval, and none is invented."""
    _, timing, segments = _prepare(smartphone_single_row)
    assert timing.median_interval_s is None
    assert timing.measured_rate_hz is None
    assert timing.gap_threshold_s is None
    assert len(segments) == 1


def test_a_frozen_clock_disables_gap_segmentation_rather_than_guessing(
    smartphone_frozen_clock: Path,
) -> None:
    """No forward motion in time means no cadence, so no threshold may be guessed."""
    _, timing, segments = _prepare(smartphone_frozen_clock)
    assert timing.median_interval_s is None
    assert timing.gap_threshold_s is None
    assert timing.large_gap_count == 0
    assert len(segments) == 1
    assert any("no usable cadence" in note for note in timing.notes)
    # Every row shares a timestamp, so every row is flagged as duplicated.
    assert timing.duplicate_timestamp_rows == timing.row_count


def test_locally_swapped_rows_are_sorted_within_the_segment_not_split(
    smartphone_out_of_order: Path,
) -> None:
    """A sub-second backward step is logger noise, not a new recording."""
    frame, timing, segments = _prepare(smartphone_out_of_order)
    assert timing.clock_reset_count == 0
    assert timing.segment_count == 1
    assert timing.non_monotonic_rows == 1
    assert timing.reordered is True
    assert "WITHIN each segment only" in timing.reorder_reason
    times = frame["analysis_time_s"].to_numpy()
    assert np.all(np.diff(times) > 0)
    assert len(frame) == segments[0].row_count


def test_reordering_flags_the_offending_row_rather_than_dropping_it(
    smartphone_out_of_order: Path,
) -> None:
    frame, _, _ = _prepare(smartphone_out_of_order)
    flags = frame["quality_flags"].to_numpy()
    flagged = int((flags & QualityFlag.NON_MONOTONIC_TIMESTAMP.value).astype(bool).sum())
    assert flagged == 1
    assert len(frame) == 60
