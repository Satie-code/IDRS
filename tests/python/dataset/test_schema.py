from __future__ import annotations

from pathlib import Path

from idr.dataset.schema import (
    assign_role,
    extract_unit,
    normalize_column_name,
    read_schema,
)


def test_extracts_smartphone_columns(smartphone_csv: Path) -> None:
    schema = read_schema(smartphone_csv)
    assert schema.column_count == 15
    roles = schema.roles
    assert {"gnss_latitude", "gnss_longitude", "accel_x", "gyro_yaw"} <= roles


def test_extracts_vehicle_columns(vehicle_csv: Path) -> None:
    schema = read_schema(vehicle_csv)
    assert schema.column_count == 7
    assert {"gnss_latitude", "velocity", "yaw_rate", "time_of_day_s"} <= schema.roles


def test_column_names_are_stripped_and_unit_separated() -> None:
    assert normalize_column_name(" GPS ALTITUDE (m)") == "GPS ALTITUDE"
    assert extract_unit(" GPS ALTITUDE (m)") == "m"


def test_unrecognized_column_is_marked_unknown() -> None:
    role, evidence = assign_role("Some Vendor Specific Channel (x)")
    assert role == "unknown"
    assert evidence == "UNVERIFIED"


def test_schema_id_is_stable_and_layout_sensitive(smartphone_csv: Path, vehicle_csv: Path) -> None:
    first = read_schema(smartphone_csv).schema_id
    assert first == read_schema(smartphone_csv).schema_id
    assert first != read_schema(vehicle_csv).schema_id


def test_detects_non_utf8_encoding(tmp_path: Path) -> None:
    path = tmp_path / "S-cp1252.csv"
    # 0xB2 is 'superscript two' in cp1252 but invalid on its own in UTF-8.
    path.write_bytes(b"ACCELEROMETER X (m/s\xb2), TIME SINCE START (ms)\n1.0,100\n")
    schema = read_schema(path)
    assert schema.encoding == "cp1252"
    assert schema.encoding_is_utf8 is False
