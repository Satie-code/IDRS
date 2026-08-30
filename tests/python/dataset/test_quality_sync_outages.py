from __future__ import annotations

from pathlib import Path

import pytest

from idr.dataset.outages import analyze_outages
from idr.dataset.quality import analyze_missingness, analyze_ranges, build_session_quality
from idr.dataset.scan import scan_csv
from idr.dataset.synchronization import analyze_pair
from idr.dataset.timestamps import analyze_timestamps


def test_missingness_counts_empty_fields(gnss_gap_csv: Path) -> None:
    scan = scan_csv(gnss_gap_csv)
    records = {r.semantic_role: r for r in analyze_missingness(scan)}
    latitude = records["gnss_latitude"]
    assert latitude.total_rows == 30
    assert latitude.missing_rows == 10
    assert latitude.missing_percentage == pytest.approx(100 * 10 / 30)
    assert latitude.kind == "exact"


def test_missingness_does_not_alter_data(gnss_gap_csv: Path) -> None:
    before = gnss_gap_csv.read_bytes()
    scan = scan_csv(gnss_gap_csv)
    analyze_missingness(scan)
    analyze_ranges(scan)
    assert gnss_gap_csv.read_bytes() == before


def test_range_finding_reports_without_deleting(tmp_path: Path) -> None:
    header = "TIME SINCE START (ms), GPS LATITUDE (degrees), GPS LONGITUDE (degrees)"
    rows = ["0,52.4,-1.5", "100,999.0,-1.5"]  # 999 degrees is impossible
    path = tmp_path / "S-RANGE.csv"
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")

    scan = scan_csv(path)
    findings = analyze_ranges(scan)
    latitude = [f for f in findings if f.semantic_role == "gnss_latitude"]
    assert latitude, "expected an out-of-range latitude finding"
    assert latitude[0].interpretation == "possible_data_error"
    assert latitude[0].observed_max == 999.0
    # Row count is untouched: reporting never removes data.
    assert scan.row_count == 2


def test_high_acceleration_is_flagged_as_possible_valid_extreme(tmp_path: Path) -> None:
    header = "TIME SINCE START (ms), Indicated Longitudinal Acceleration (g)"
    rows = ["0,0.1", "100,2.5"]
    path = tmp_path / "V-BRAKE.csv"
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")

    findings = analyze_ranges(scan_csv(path))
    accel = [f for f in findings if f.semantic_role == "accel_longitudinal"]
    assert accel
    # Hard braking is plausible physics, not automatically corrupt data.
    assert accel[0].interpretation == "possible_valid_extreme"


def test_constant_channel_flagged_as_possible_artifact(tmp_path: Path) -> None:
    header = "TIME SINCE START (ms), GPS SPEED (Kmh)"
    rows = [f"{i * 100},5.0" for i in range(10)]
    path = tmp_path / "S-STUCK.csv"
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")

    findings = analyze_ranges(scan_csv(path))
    stuck = [f for f in findings if "constant value" in f.detail]
    assert stuck
    assert stuck[0].interpretation == "possible_sensor_artifact"


def test_detects_controlled_gnss_gap(gnss_gap_csv: Path) -> None:
    scan = scan_csv(gnss_gap_csv)
    report, intervals = analyze_outages(scan, "GAP", "smartphone")

    assert report.total_rows == 30
    assert report.rows_with_fix == 20
    assert report.availability_fraction == pytest.approx(20 / 30)
    assert len(intervals) == 1

    gap = intervals[0]
    assert gap.start_index == 10
    assert gap.end_index == 19
    # Rows 10..19 have no fix; the fix returns at row 20, so the outage spans
    # 1.0 s (t=1.0 s to t=2.0 s), not the 0.9 s between the first and last bad row.
    assert gap.duration_s == pytest.approx(1.0, abs=1e-6)
    assert gap.source == "observed_missingness"
    assert gap.evidence == "VERIFIED_FROM_FILE"


def test_outage_is_never_labelled_documented(gnss_gap_csv: Path) -> None:
    report, intervals = analyze_outages(
        scan_csv(gnss_gap_csv), "GAP", "smartphone", min_duration_s=0.0
    )
    assert report.documented_outages_found is False
    assert all(i.source != "documented" for i in intervals)


def test_short_gaps_below_threshold_are_not_reported(gnss_gap_csv: Path) -> None:
    _, intervals = analyze_outages(scan_csv(gnss_gap_csv), "GAP", "smartphone", min_duration_s=5.0)
    assert intervals == []


def test_synchronization_measures_one_hour_offset(smartphone_csv: Path, vehicle_csv: Path) -> None:
    phone = scan_csv(smartphone_csv)
    vehicle = scan_csv(vehicle_csv)
    report = analyze_pair("T1", "synchronized_categorized", phone, vehicle)

    assert report.row_counts_match
    assert report.mean_offset_s == pytest.approx(3600.0, abs=0.01)
    assert report.whole_hour_offset == 1
    assert report.residual_offset_s == pytest.approx(0.0, abs=0.01)
    assert report.drift_s == pytest.approx(0.0, abs=0.01)
    assert report.interpretation == "constant_offset_whole_hour_likely_timezone"
    # Measured from files, but the timezone conclusion is our inference.
    assert report.evidence == "INFERRED"


def test_synchronization_overlap_after_alignment(smartphone_csv: Path, vehicle_csv: Path) -> None:
    report = analyze_pair(
        "T1", "synchronized_categorized", scan_csv(smartphone_csv), scan_csv(vehicle_csv)
    )
    # Raw clocks are an hour apart, so they do not overlap as recorded...
    assert report.raw_overlap_s == pytest.approx(0.0)
    # ...but they cover the same 1.9 s once the offset is removed.
    assert report.aligned_overlap_s == pytest.approx(1.9, abs=0.01)


def test_synchronization_handles_missing_clock(tmp_path: Path, smartphone_csv: Path) -> None:
    path = tmp_path / "V-NOCLOCK.csv"
    path.write_text("Latitude (degrees), Longitude (degrees)\n52.4,-1.5\n", encoding="utf-8")
    report = analyze_pair("X", "f", scan_csv(smartphone_csv), scan_csv(path))
    assert report.interpretation == "unknown"
    assert report.evidence == "UNVERIFIED"
    assert "lack a usable time-of-day clock" in report.note


def test_quality_score_is_transparent_and_never_drops(smartphone_csv: Path) -> None:
    scan = scan_csv(smartphone_csv)
    timestamps = analyze_timestamps(scan)
    findings = analyze_ranges(scan)
    quality = build_session_quality(scan, timestamps, "T1", "smartphone", findings)

    assert 0.0 <= quality.quality_score <= 1.0
    # Every sub-metric survives alongside the score.
    assert quality.timestamp_quality == 1.0
    assert quality.sampling_quality == 1.0
    assert quality.gnss_availability == 1.0
    assert "never used to drop a session" in quality.score_formula


def test_measures_position_update_rate_not_row_rate(tmp_path: Path) -> None:
    """A forward-filled position must not be counted as a fresh fix per row."""
    header = "GPS LATITUDE (degrees), GPS LONGITUDE (degrees), TIME SINCE START (ms)"
    rows = []
    for index in range(100):
        # Position refreshes only every 10th row (every 1.0 s at 10 Hz).
        lat = 52.4 + (index // 10) * 0.001
        rows.append(f"{lat:.5f},-1.5,{index * 100}")
    path = tmp_path / "S-HELD.csv"
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")

    scan = scan_csv(path)
    report, _ = analyze_outages(scan, "HELD", "smartphone")

    # Every row has a valid fix...
    assert report.availability_fraction == pytest.approx(1.0)
    # ...but the position only refreshes once a second: 10 update events
    # (the initial sample plus 9 changes) at a measured 1 Hz, not 10 Hz.
    assert report.position_update_count == 10
    assert report.position_update_rate_hz == pytest.approx(1.0, rel=1e-6)
    assert report.median_update_interval_s == pytest.approx(1.0)


def test_long_position_hold_is_reported_as_interval(tmp_path: Path) -> None:
    header = "GPS LATITUDE (degrees), GPS LONGITUDE (degrees), TIME SINCE START (ms)"
    rows = ["52.40000,-1.5,0"]
    # Hold the same position for 60 s, then move.
    for index in range(1, 600):
        rows.append(f"52.40000,-1.5,{index * 100}")
    rows.append("52.41000,-1.5,60000")
    path = tmp_path / "S-HOLD.csv"
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")

    _, intervals = analyze_outages(scan_csv(path), "HOLD", "smartphone")
    holds = [i for i in intervals if i.source == "observed_position_hold"]
    assert len(holds) == 1
    assert holds[0].duration_s == pytest.approx(60.0, abs=0.1)
    assert holds[0].evidence == "VERIFIED_FROM_FILE"


def test_short_position_holds_are_not_reported(tmp_path: Path) -> None:
    header = "GPS LATITUDE (degrees), GPS LONGITUDE (degrees), TIME SINCE START (ms)"
    rows = [f"{52.4 + i * 0.001:.5f},-1.5,{i * 100}" for i in range(50)]
    path = tmp_path / "S-FAST.csv"
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")

    _, intervals = analyze_outages(scan_csv(path), "FAST", "smartphone")
    assert [i for i in intervals if i.source == "observed_position_hold"] == []
