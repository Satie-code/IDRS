"""Diagnostic-figure tests.

The figures are how a reader checks the claims made in the Phase 2 report, so
two properties matter and are asserted here: a figure is produced whenever the
run actually supports it, and no figure is invented from data that does not
exist. A histogram drawn from one sync pair or one segment would look like
evidence while showing none, so those cases must decline rather than emit an
empty axes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from idr.pipeline.freshness import FreshnessReport
from idr.pipeline.plots import (
    generate_phase2_plots,
    plot_gnss_freshness,
    plot_label_coverage,
    plot_multi_segment_files,
    plot_sampling_rates,
    plot_segment_sizes,
    plot_split_balance,
    plot_sync_confidence,
    plot_sync_offsets,
)
from idr.pipeline.reader import ReadResult
from idr.pipeline.reference import LabelCoverage
from idr.pipeline.repair import NO_REPAIR
from idr.pipeline.runner import PipelineResult
from idr.pipeline.splits import SplitSummary
from idr.pipeline.sync import SyncResult
from idr.pipeline.timeline import SegmentRecord, TimingDiagnosis

ALL_BUILDERS = (
    plot_sampling_rates,
    plot_segment_sizes,
    plot_multi_segment_files,
    plot_sync_offsets,
    plot_sync_confidence,
    plot_gnss_freshness,
    plot_label_coverage,
    plot_split_balance,
)


def _read(name: str, stream: str) -> ReadResult:
    return ReadResult(
        frame=pd.DataFrame(),
        source_file=name,
        stream=stream,
        file_hash="f" * 64,
        payload_hash="p" * 64,
        source_encoding="utf-8",
        decode_method="utf-8",
        encoding_warning=None,
        repair=NO_REPAIR,
        row_count=100,
        availability={},
    )


def _timing(name: str, *, rate: float | None, segments: int = 1) -> TimingDiagnosis:
    return TimingDiagnosis(
        source_file=name,
        row_count=100,
        valid_timestamps=100,
        invalid_timestamps=0,
        duplicate_timestamp_rows=0,
        duplicate_groups=0,
        non_monotonic_rows=0,
        reordered=False,
        reorder_reason="input order preserved",
        median_interval_s=(1.0 / rate) if rate else None,
        measured_rate_hz=rate,
        segment_count=segments,
        gap_threshold_s=10.0,
    )


def _segment(index: int, duration: float) -> SegmentRecord:
    return SegmentRecord(
        source_file=f"S-{index}.csv",
        session_id="S1",
        stream="smartphone",
        dataset_family="synchronised",
        payload_hash="p" * 64,
        segment_id=f"seg:{index:03d}",
        segment_index=0,
        start_row=0,
        end_row=99,
        row_count=100,
        start_time_s=0.0,
        end_time_s=duration,
        duration_s=duration,
        gap_before_s=None,
        median_interval_s=0.1,
        measured_rate_hz=10.0,
        segmentation_reason="first segment of file",
        is_short=False,
    )


def _freshness(name: str, stream: str, interval: float | None) -> FreshnessReport:
    return FreshnessReport(
        source_file=name,
        stream=stream,
        row_count=100,
        rows_with_fix=100,
        availability_fraction=1.0,
        position_update_count=10,
        median_update_interval_s=interval,
        p90_update_interval_s=interval,
        max_update_interval_s=interval,
        measured_update_rate_hz=(1.0 / interval) if interval else None,
        stale_threshold_s=2.0,
        stale_rows=0,
        stale_fraction=0.0,
        max_stale_duration_s=0.0,
    )


def _sync(index: int, offset: float, correlation: float, status: str) -> SyncResult:
    return SyncResult(
        pair_id=f"pair-{index}",
        session_id="S1",
        dataset_family="synchronised",
        smartphone_file=f"S-{index}.csv",
        vehicle_file=f"V-{index}.csv",
        smartphone_segment=f"s:{index}",
        vehicle_segment=f"v:{index}",
        smartphone_start_s=0.0,
        smartphone_end_s=600.0,
        vehicle_start_s=0.0,
        vehicle_end_s=600.0,
        overlap_start_s=0.0,
        overlap_end_s=600.0,
        overlap_duration_s=600.0,
        clock_offset_s=offset,
        estimated_offset_s=offset,
        offset_method="speed_cross_correlation",
        offset_confidence="high",
        correlation=correlation,
        peak_margin=0.2,
        drift_detected=False,
        drift_estimate_s=None,
        drift_ppm=None,
        status=status,
    )


def _label(index: int, source: str, coverage: float) -> LabelCoverage:
    return LabelCoverage(
        segment_id=f"seg:{index:03d}",
        session_id="S1",
        source_file=f"S-{index}.csv",
        label_source=source,
        label_quality="high",
        row_count=100,
        labelled_rows=int(100 * coverage),
        coverage_fraction=coverage,
        mean_label_mps=12.0,
        max_label_mps=25.0,
        stationary_fraction=0.1,
    )


@pytest.fixture
def empty_result() -> PipelineResult:
    return PipelineResult(run_id="empty", started_at="2026-01-01T00:00:00Z", raw_path="raw")


@pytest.fixture
def full_result() -> PipelineResult:
    result = PipelineResult(run_id="full", started_at="2026-01-01T00:00:00Z", raw_path="raw")
    result.reads = [_read("S-1.csv", "smartphone"), _read("V-1.csv", "vehicle")]
    result.timing = [
        _timing("S-1.csv", rate=10.0, segments=3),
        _timing("V-1.csv", rate=100.0, segments=1),
    ]
    result.segments = [_segment(index, 120.0 * (index + 1)) for index in range(4)]
    result.freshness = [
        _freshness("S-1.csv", "smartphone", 9.0),
        _freshness("V-1.csv", "vehicle", 0.1),
    ]
    result.sync = [
        _sync(0, 0.0, 0.95, "CONFIRMED_BY_SIGNAL"),
        _sync(1, 3600.0, 0.62, "CONFIRMED_BY_SIGNAL"),
        _sync(2, -3600.0, 0.30, "UNRESOLVED"),
    ]
    result.labels = [
        _label(0, "TIER_1_VEHICLE_SYNCED", 0.98),
        _label(1, "TIER_1_VEHICLE_SYNCED", 0.91),
        _label(2, "TIER_3_SMARTPHONE_GNSS", 0.44),
    ]
    result.split_summary = SplitSummary(
        seed="sih26168-phase2",
        ratios={"train": 0.7, "validation": 0.15, "test": 0.15},
        group_count=3,
        segment_count=4,
        counts={"train": 2, "validation": 1, "test": 1},
        rows={"train": 700, "validation": 150, "test": 150},
    )
    return result


def test_every_plot_declines_an_empty_result(empty_result: PipelineResult, tmp_path: Path) -> None:
    """No data means no figure — never an empty axes that reads as evidence."""
    for builder in ALL_BUILDERS:
        assert builder(empty_result, tmp_path / f"{builder.__name__}.png") is None
    assert generate_phase2_plots(empty_result, tmp_path / "plots") == []
    assert not list((tmp_path / "plots").glob("*.png"))


def test_every_plot_is_produced_from_a_complete_result(
    full_result: PipelineResult, tmp_path: Path
) -> None:
    produced = generate_phase2_plots(full_result, tmp_path / "plots")
    assert len(produced) == len(ALL_BUILDERS)
    for path in produced:
        assert path.exists()
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_a_single_observation_is_not_plotted_as_a_distribution(
    full_result: PipelineResult, tmp_path: Path
) -> None:
    """One segment or one pair is not a distribution; those figures must decline."""
    full_result.segments = full_result.segments[:1]
    full_result.sync = full_result.sync[:1]
    assert plot_segment_sizes(full_result, tmp_path / "seg.png") is None
    assert plot_sync_offsets(full_result, tmp_path / "off.png") is None
    assert plot_sync_confidence(full_result, tmp_path / "conf.png") is None


def test_single_stream_runs_still_plot(full_result: PipelineResult, tmp_path: Path) -> None:
    """A vehicle-only or smartphone-only run still gets its own figure."""
    full_result.reads = [_read("V-1.csv", "vehicle")]
    full_result.timing = [_timing("V-1.csv", rate=100.0)]
    full_result.freshness = [_freshness("V-1.csv", "vehicle", 0.1)]
    assert plot_sampling_rates(full_result, tmp_path / "rates.png") is not None
    assert plot_gnss_freshness(full_result, tmp_path / "fresh.png") is not None

    full_result.reads = [_read("S-1.csv", "smartphone")]
    full_result.timing = [_timing("S-1.csv", rate=10.0)]
    full_result.freshness = [_freshness("S-1.csv", "smartphone", 9.0)]
    assert plot_sampling_rates(full_result, tmp_path / "rates2.png") is not None
    assert plot_gnss_freshness(full_result, tmp_path / "fresh2.png") is not None


def test_streams_without_a_measured_rate_are_skipped_not_guessed(
    full_result: PipelineResult, tmp_path: Path
) -> None:
    """A stream whose cadence could not be measured contributes no bar."""
    full_result.reads = [_read("S-1.csv", "smartphone")]
    full_result.timing = [_timing("S-1.csv", rate=None)]
    assert plot_sampling_rates(full_result, tmp_path / "rates.png") is None

    full_result.freshness = [_freshness("S-1.csv", "smartphone", None)]
    assert plot_gnss_freshness(full_result, tmp_path / "fresh.png") is None


def test_multi_segment_figure_lists_only_split_files(
    full_result: PipelineResult, tmp_path: Path
) -> None:
    assert plot_multi_segment_files(full_result, tmp_path / "multi.png") is not None
    full_result.timing = [_timing("S-1.csv", rate=10.0, segments=1)]
    assert plot_multi_segment_files(full_result, tmp_path / "multi2.png") is None


def test_label_coverage_declines_when_nothing_was_labelled(
    full_result: PipelineResult, tmp_path: Path
) -> None:
    full_result.labels = []
    assert plot_label_coverage(full_result, tmp_path / "labels.png") is None


def test_split_balance_declines_a_summary_with_no_rows(
    full_result: PipelineResult, tmp_path: Path
) -> None:
    assert full_result.split_summary is not None
    full_result.split_summary.rows = {}
    assert plot_split_balance(full_result, tmp_path / "split.png") is None
