from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from idr.dataset.schema import read_schema
from idr.pipeline.canonical import QualityFlag
from idr.pipeline.mapping import build_mapping, parse_satellites
from idr.pipeline.reader import hash_file_and_payload, read_canonical
from idr.pipeline.repair import detect_column_shift


def _read(path: Path, stream: str = "smartphone"):
    return read_canonical(path, path.name, stream, dataset_family="test_family", session_id="T1")


# --- encoding --------------------------------------------------------------


def test_reads_utf8_file_and_records_encoding(smartphone_csv: Path) -> None:
    result = _read(smartphone_csv)
    assert result.source_encoding == "utf-8"
    assert result.decode_method == "strict"
    assert result.encoding_warning is None
    assert result.row_count == 60


def test_reads_cp1252_file_and_records_fallback(smartphone_cp1252: Path) -> None:
    result = _read(smartphone_cp1252)
    assert result.source_encoding == "cp1252"
    assert result.decode_method == "cp1252-fallback"
    assert result.encoding_warning is not None
    # The rows still parse correctly despite the header encoding.
    assert result.row_count == 30
    assert np.isfinite(result.frame["accel_x"]).all()


def test_encoding_warning_sets_quality_flag(smartphone_cp1252: Path) -> None:
    result = _read(smartphone_cp1252)
    flags = result.frame["quality_flags"].to_numpy()
    assert bool(np.all(flags & QualityFlag.ENCODING_WARNING.value))


# --- schema variants -------------------------------------------------------


def test_full_schema_reports_all_channels_available(smartphone_csv: Path) -> None:
    result = _read(smartphone_csv)
    assert result.availability["has_magnetometer"] is True
    assert result.availability["has_orientation"] is True
    assert result.availability["has_gyroscope"] is True


def test_reduced_schema_marks_missing_channels_not_zero(smartphone_csv_18: Path) -> None:
    result = _read(smartphone_csv_18)
    assert result.availability["has_magnetometer"] is False
    assert result.availability["has_orientation"] is False
    # Absent channels must be NaN — a zero would read as a real measurement.
    assert result.frame["mag_x"].isna().all()
    assert result.frame["orientation_pitch"].isna().all()
    assert not (result.frame["mag_x"].fillna(-1) == 0).any()


def test_reduced_schema_flags_missing_sensor(smartphone_csv_18: Path) -> None:
    flags = _read(smartphone_csv_18).frame["quality_flags"].to_numpy()
    assert bool(np.all(flags & QualityFlag.MISSING_SENSOR.value))


def test_gyro_xyz_and_yawpitchroll_map_to_same_canonical_fields(
    smartphone_csv: Path, smartphone_csv_xyz: Path
) -> None:
    """The two namings label identical column positions; both must canonicalize alike."""
    named = _read(smartphone_csv).frame
    xyz = _read(smartphone_csv_xyz).frame
    for column in ("gyro_x", "gyro_y", "gyro_z", "orientation_yaw"):
        assert column in named.columns and column in xyz.columns
        pd.testing.assert_series_equal(
            named[column], xyz[column], check_names=False, check_dtype=False
        )


def test_alias_resolution_is_recorded(smartphone_csv_xyz: Path) -> None:
    result = _read(smartphone_csv_xyz)
    assert "gyro_x" in result.alias_resolved
    assert "GYROSCOPE X" in result.alias_resolved["gyro_x"]


def test_no_unmapped_columns_for_known_schemas(smartphone_csv: Path, vehicle_csv: Path) -> None:
    assert _read(smartphone_csv).unmapped_columns == []
    assert _read(vehicle_csv, "vehicle").unmapped_columns == []


# --- units -----------------------------------------------------------------


def test_speed_converted_from_kmh_to_mps(smartphone_csv: Path) -> None:
    frame = _read(smartphone_csv).frame
    # Fixture rows carry 36 km/h == 10 m/s exactly.
    assert frame["gnss_speed_mps"].iloc[0] == pytest.approx(10.0)


def test_vehicle_units_converted(vehicle_csv: Path) -> None:
    frame = _read(vehicle_csv, "vehicle").frame
    assert frame["ref_speed_mps"].iloc[0] == pytest.approx(10.0)
    # Height is km in the source; canonical altitude is metres.
    assert frame["altitude_m"].iloc[0] == pytest.approx(110.0)
    # Acceleration is g in the source; canonical is m/s^2.
    assert frame["ref_accel_long_mps2"].iloc[0] == pytest.approx(0.05 * 9.80665)


# --- satellites ------------------------------------------------------------


def test_satellite_field_parsed_without_crashing(smartphone_csv: Path) -> None:
    frame = _read(smartphone_csv).frame
    assert frame["gnss_satellites_raw"].iloc[0] == "27 / 28"
    assert frame["gnss_satellites_in_use"].iloc[0] == pytest.approx(27.0)
    assert frame["gnss_satellites_in_view"].iloc[0] == pytest.approx(28.0)


def test_satellite_parser_handles_plain_and_malformed_values() -> None:
    in_use, in_view = parse_satellites(pd.Series(["27 / 28", "11", "", "junk", None]))
    assert in_use[0] == 27 and in_view[0] == 28
    assert in_use[1] == 11 and np.isnan(in_view[1])
    assert np.isnan(in_use[2]) and np.isnan(in_use[3]) and np.isnan(in_use[4])


def test_raw_satellite_string_is_always_preserved(smartphone_csv: Path) -> None:
    frame = _read(smartphone_csv).frame
    assert (frame["gnss_satellites_raw"] == "27 / 28").all()


# --- repair ----------------------------------------------------------------


def test_detects_and_repairs_column_shift(smartphone_shifted: Path) -> None:
    result = _read(smartphone_shifted)
    assert result.repair.applied is True
    assert result.repair.shift_index == 6
    assert result.repair.confidence == "high"
    # After repair the timestamp column holds numbers again, not "27 / 28".
    assert np.isfinite(result.frame["source_time_s"]).all()
    assert result.frame["gnss_satellites_raw"].iloc[0] == "27 / 28"


def test_repair_sets_quality_flag(smartphone_shifted: Path) -> None:
    flags = _read(smartphone_shifted).frame["quality_flags"].to_numpy()
    assert bool(np.all(flags & QualityFlag.SCHEMA_REPAIR.value))


def test_healthy_file_is_never_repaired(smartphone_csv: Path) -> None:
    assert _read(smartphone_csv).repair.applied is False


def test_repair_is_deterministic(smartphone_shifted: Path) -> None:
    first = _read(smartphone_shifted).frame
    second = _read(smartphone_shifted).frame
    pd.testing.assert_frame_equal(first, second)


def test_repair_leaves_raw_file_untouched(smartphone_shifted: Path) -> None:
    before = smartphone_shifted.read_bytes()
    _read(smartphone_shifted)
    assert smartphone_shifted.read_bytes() == before


def test_trailing_comma_without_data_is_not_repaired(tmp_path: Path) -> None:
    """A stray trailing delimiter must not trigger a column shift."""
    header = "TIME SINCE START (ms), GPS SPEED (Kmh)"
    frame = pd.DataFrame({"TIME SINCE START (ms)": ["0", "100"], " GPS SPEED (Kmh)": ["1", "2"]})
    assert detect_column_shift(frame, header.split(",")).applied is False


# --- hashing ---------------------------------------------------------------


def test_payload_hash_ignores_header_differences(smartphone_csv: Path, tmp_path: Path) -> None:
    """Files differing only in header labels must share a payload hash.

    This is the real defect that makes file-hash splitting leak.
    """
    text = smartphone_csv.read_text(encoding="utf-8").splitlines()
    relabelled = tmp_path / "S-RELABEL.csv"
    relabelled.write_text(
        "\n".join([text[0].replace("GYROSCOPE Yaw", "GYROSCOPE X"), *text[1:]]) + "\n",
        encoding="utf-8",
    )
    file_a, payload_a = hash_file_and_payload(smartphone_csv)
    file_b, payload_b = hash_file_and_payload(relabelled)
    assert file_a != file_b
    assert payload_a == payload_b


def test_payload_hash_differs_when_data_differs(
    smartphone_csv: Path, smartphone_csv_18: Path
) -> None:
    assert hash_file_and_payload(smartphone_csv)[1] != hash_file_and_payload(smartphone_csv_18)[1]


# --- mapping ---------------------------------------------------------------


def test_mapping_identifies_time_and_date_columns(smartphone_csv: Path) -> None:
    mapping = build_mapping(read_schema(smartphone_csv), "smartphone")
    assert mapping.time_role == "time_since_start_ms"
    assert mapping.date_column is not None
    assert mapping.satellites_column is not None


def test_analysis_clock_prefers_wall_clock_for_smartphone(smartphone_csv: Path) -> None:
    frame = _read(smartphone_csv).frame
    assert frame["analysis_time_source"].iloc[0] == "wall_clock"


def test_analysis_clock_uses_time_of_day_for_vehicle(vehicle_csv: Path) -> None:
    frame = _read(vehicle_csv, "vehicle").frame
    assert frame["analysis_time_source"].iloc[0] == "time_of_day"


def test_analysis_clock_falls_back_to_elapsed_without_a_date_column(
    smartphone_no_date_reset: Path,
) -> None:
    """With no wall clock, the elapsed counter is the only clock available."""
    frame = _read(smartphone_no_date_reset).frame
    assert frame["analysis_time_source"].iloc[0] == "elapsed"
    assert frame["timestamp_utc"].isna().all()


def test_date_without_a_time_of_day_yields_no_wall_clock(
    smartphone_date_without_time: Path,
) -> None:
    """A date with no time is not a clock; it must not be guessed at midnight."""
    frame = _read(smartphone_date_without_time).frame
    assert frame["time_of_day_s"].isna().all()
    assert frame["analysis_time_source"].iloc[0] == "elapsed"


def test_date_without_seconds_yields_no_wall_clock(
    smartphone_date_without_seconds: Path,
) -> None:
    frame = _read(smartphone_date_without_seconds).frame
    assert frame["time_of_day_s"].isna().all()
    assert frame["analysis_time_source"].iloc[0] == "elapsed"


def test_header_only_file_is_rejected_not_silently_empty(
    smartphone_header_only: Path,
) -> None:
    from idr.pipeline.reader import ReadError

    with pytest.raises(ReadError, match="no data rows"):
        _read(smartphone_header_only)


def test_impossible_satellite_counts_are_flagged_not_reinterpreted(
    smartphone_satellites_inverted: Path,
) -> None:
    """``28 / 27`` cannot be 'in use / in view'; the reading is recorded as unsupported."""
    result = _read(smartphone_satellites_inverted)
    assert result.satellites_consistent is False
    assert any("in use" in warning for warning in result.warnings)
    # The values are still carried through verbatim — nothing is swapped.
    assert result.frame["gnss_satellites_in_use"].iloc[0] == 28
    assert result.frame["gnss_satellites_in_view"].iloc[0] == 27


# --- repair: the negative cases --------------------------------------------


def _shift_candidate() -> tuple[pd.DataFrame, list[str]]:
    names = [f"c{i}" for i in range(24)]
    data: dict[str, list[object]] = {name: [1.0, 2.0] for name in names}
    data["c6"] = ["", ""]
    frame = pd.DataFrame(data)
    frame["Unnamed: 24"] = ["9.0", "9.0"]
    return frame, names


def test_repair_requires_a_surplus_trailing_column() -> None:
    """A frame whose width already matches the header is not reshaped."""
    frame, names = _shift_candidate()
    frame = frame.drop(columns=["Unnamed: 24"])
    assert detect_column_shift(frame, names).applied is False


def test_repair_requires_the_surplus_column_to_be_unnamed() -> None:
    """A genuine 25th labelled channel is data, not a shift artefact."""
    frame, names = _shift_candidate()
    frame = frame.rename(columns={"Unnamed: 24": "REAL EXTRA CHANNEL"})
    assert detect_column_shift(frame, names).applied is False


def test_repair_declines_when_the_trailing_column_holds_nothing() -> None:
    """A stray trailing comma with no data behind it is not a shift."""
    frame, names = _shift_candidate()
    frame["Unnamed: 24"] = ["", ""]
    assert detect_column_shift(frame, names).applied is False


def test_repair_declines_when_the_shifted_column_holds_data() -> None:
    """If column 6 has values, nothing was pushed out of it."""
    frame, names = _shift_candidate()
    frame["c6"] = ["3.0", "4.0"]
    assert detect_column_shift(frame, names).applied is False


def test_repair_refuses_to_emit_a_mislabelled_frame() -> None:
    """Applying the shift to a frame it does not fit must raise, not mislabel."""
    from idr.pipeline.repair import apply_column_shift

    frame, names = _shift_candidate()
    with pytest.raises(ValueError, match="refusing to emit a mislabelled frame"):
        apply_column_shift(frame, [*names, "c24", "c25"])
