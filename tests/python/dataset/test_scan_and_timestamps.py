from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from idr.dataset.scan import parse_iovnbd_datetime, scan_csv
from idr.dataset.timestamps import analyze_timestamps, classify_sampling


def test_scan_counts_rows_exactly(smartphone_csv: Path) -> None:
    scan = scan_csv(smartphone_csv)
    assert scan.row_count == 20
    assert scan.row_count_kind == "exact"
    assert scan.parse_error is None


def test_measured_rate_is_10hz(smartphone_csv: Path) -> None:
    report = analyze_timestamps(scan_csv(smartphone_csv))
    assert report.estimated_rate_hz == pytest.approx(10.0)
    assert report.delta_median_s == pytest.approx(0.1)
    assert report.sampling_class == "stable"
    assert report.is_monotonic


def test_duration_matches_span(smartphone_csv: Path) -> None:
    report = analyze_timestamps(scan_csv(smartphone_csv))
    # 20 samples at 100 ms spans 19 intervals = 1.9 s.
    assert report.duration_s == pytest.approx(1.9)


def test_detects_non_monotonic_and_duplicate_timestamps(tmp_path: Path) -> None:
    header = "TIME SINCE START (ms), ACCELEROMETER X (m/s2)"
    # 0, 100, 100 (duplicate), 50 (backwards), 300
    rows = ["0,1.0", "100,1.0", "100,1.0", "50,1.0", "300,1.0"]
    path = tmp_path / "S-BAD.csv"
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")

    report = analyze_timestamps(scan_csv(path))
    assert report.is_monotonic is False
    assert report.duplicate_timestamp_count == 1
    assert report.zero_interval_count == 1
    assert report.negative_interval_count == 1


def test_irregular_sampling_is_classified(tmp_path: Path) -> None:
    header = "TIME SINCE START (ms), ACCELEROMETER X (m/s2)"
    # Wildly uneven intervals: 100, 900, 100, 900 ms.
    rows = ["0,1.0", "100,1.0", "1000,1.0", "1100,1.0", "2000,1.0"]
    path = tmp_path / "S-IRREG.csv"
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")

    report = analyze_timestamps(scan_csv(path))
    assert report.sampling_class in ("irregular", "severely_irregular")
    assert report.coefficient_of_variation is not None
    assert report.coefficient_of_variation > 0.2


def test_classify_sampling_boundaries() -> None:
    assert classify_sampling(None) == "unknown"
    assert classify_sampling(0.01) == "stable"
    assert classify_sampling(0.10) == "mostly_stable"
    assert classify_sampling(0.50) == "irregular"
    assert classify_sampling(2.00) == "severely_irregular"


def test_missing_timestamp_column_yields_unknown(tmp_path: Path) -> None:
    path = tmp_path / "S-NOTIME.csv"
    path.write_text("Foo (x), Bar (y)\n1,2\n3,4\n", encoding="utf-8")
    report = analyze_timestamps(scan_csv(path))
    assert report.sampling_class == "unknown"
    assert report.estimated_rate_hz is None
    assert "no recognized timestamp column" in report.note


def test_parses_colon_millisecond_datetime() -> None:
    import pandas as pd

    series = pd.Series(["2019-09-08 10:07:49:546", "2019-09-08 10:07:49:646"])
    seconds, first_date = parse_iovnbd_datetime(series)
    assert first_date == "2019-09-08"
    expected = 10 * 3600 + 7 * 60 + 49 + 0.546
    assert seconds[0] == pytest.approx(expected)
    assert seconds[1] - seconds[0] == pytest.approx(0.1)


def test_non_numeric_field_counted_separately_from_missing(smartphone_csv: Path) -> None:
    scan = scan_csv(smartphone_csv)
    satellites = scan.column_by_role("gnss_satellites")
    assert satellites is not None
    # "27 / 28" is present but unparseable: not missing, but not numeric either.
    assert satellites.missing_rows == 0
    assert satellites.non_numeric_rows == 20


def test_scan_records_parse_error_instead_of_raising(tmp_path: Path) -> None:
    path = tmp_path / "S-EMPTY.csv"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty header"):
        scan_csv(path)


def test_gnss_zero_zero_is_treated_as_no_fix(tmp_path: Path) -> None:
    header = "GPS LATITUDE (degrees), GPS LONGITUDE (degrees), TIME SINCE START (ms)"
    rows = ["52.4,-1.5,0", "0,0,100", "52.4,-1.5,200"]
    path = tmp_path / "S-ZERO.csv"
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    scan = scan_csv(path)
    assert scan.gnss_valid is not None
    assert list(scan.gnss_valid) == [True, False, True]


def test_column_statistics_are_accurate_across_chunks(tmp_path: Path) -> None:
    header = "TIME SINCE START (ms), GPS SPEED (Kmh)"
    values = list(range(100))
    rows = [f"{i * 100},{v}" for i, v in enumerate(values)]
    path = tmp_path / "S-STATS.csv"
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")

    # A chunk size smaller than the file forces the incremental path.
    scan = scan_csv(path, chunk_size=7)
    speed = scan.column_by_role("gnss_speed")
    assert speed is not None
    assert speed.minimum == 0.0
    assert speed.maximum == 99.0
    assert speed.mean == pytest.approx(float(np.mean(values)))
    assert speed.std == pytest.approx(float(np.std(values)))


def test_flags_column_misalignment(tmp_path: Path) -> None:
    """A shifted row puts a non-numeric value in the timestamp column."""
    header = "GPS LATITUDE (degrees),GPS SATELLITES IN RANGE,TIME SINCE START (ms),GPS SPEED (Kmh)"
    # Each row carries an extra empty field, shifting everything right by one,
    # so "13 / 24" lands in the TIME SINCE START column.
    rows = [f"52.4,,13 / 24,{i * 100}" for i in range(20)]
    path = tmp_path / "S-SHIFTED.csv"
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")

    report = analyze_timestamps(scan_csv(path))
    assert report.suspected_column_misalignment is True
    assert report.valid_timestamps == 0
    assert report.sampling_class == "unknown"
    assert "misalignment suspected" in report.note


def test_healthy_file_is_not_flagged_as_misaligned(smartphone_csv: Path) -> None:
    assert analyze_timestamps(scan_csv(smartphone_csv)).suspected_column_misalignment is False


def test_tiny_file_is_not_flagged_as_misaligned(tmp_path: Path) -> None:
    """One valid row is too little evidence to call a file shifted."""
    path = tmp_path / "S-TINY.csv"
    path.write_text("TIME SINCE START (ms),GPS SPEED (Kmh)\n100,5\n", encoding="utf-8")
    report = analyze_timestamps(scan_csv(path))
    assert report.suspected_column_misalignment is False
    assert report.sampling_class == "unknown"
