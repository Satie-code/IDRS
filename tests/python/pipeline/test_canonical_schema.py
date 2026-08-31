from __future__ import annotations

import numpy as np
import pytest

from idr.pipeline.canonical import (
    CANONICAL_SCHEMA_VERSION,
    FEATURE_AVAILABILITY_KEYS,
    QualityFlag,
    describe_flags,
    empty_column,
    pandas_dtypes,
    schema_for,
)


def test_schema_version_is_set() -> None:
    assert CANONICAL_SCHEMA_VERSION.count(".") == 2


def test_streams_have_distinct_schemas() -> None:
    phone = {spec.name for spec in schema_for("smartphone")}
    vehicle = {spec.name for spec in schema_for("vehicle")}
    assert "accel_x" in phone and "accel_x" not in vehicle
    assert "ref_speed_mps" in vehicle and "ref_speed_mps" not in phone
    # Identity, time, GNSS and quality fields are shared by both.
    assert {"segment_id", "elapsed_s", "latitude", "quality_flags"} <= phone & vehicle


def test_unknown_stream_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown stream"):
        schema_for("satellite")


def test_field_names_are_unique_within_a_schema() -> None:
    for stream in ("smartphone", "vehicle"):
        names = [spec.name for spec in schema_for(stream)]
        assert len(names) == len(set(names))


def test_every_numeric_field_declares_a_unit_or_is_dimensionless() -> None:
    for stream in ("smartphone", "vehicle"):
        for spec in schema_for(stream):
            if spec.dtype.startswith("float") and spec.unit is None:
                # Dimensionless numerics are allowed but must say what they are.
                assert spec.description, f"{spec.name} has neither a unit nor a description"


def test_pandas_dtypes_cover_the_schema() -> None:
    dtypes = pandas_dtypes("smartphone")
    assert dtypes["latitude"] == "float64"
    assert dtypes["accel_x"] == "float32"
    assert dtypes["quality_flags"] == "int32"


def test_latitude_is_float64_not_float32() -> None:
    """float32 would quantize position to roughly metre scale — unacceptable."""
    for stream in ("smartphone", "vehicle"):
        spec = next(s for s in schema_for(stream) if s.name == "latitude")
        assert spec.dtype == "float64"


def test_describe_flags_round_trips() -> None:
    assert describe_flags(0) == ["VALID"]
    mask = QualityFlag.GNSS_STALE.value | QualityFlag.DUPLICATE_TIMESTAMP.value
    names = describe_flags(mask)
    assert set(names) == {"GNSS_STALE", "DUPLICATE_TIMESTAMP"}


def test_flag_values_are_distinct_powers_of_two() -> None:
    values = [flag.value for flag in QualityFlag if flag.value]
    assert len(values) == len(set(values))
    assert all(value & (value - 1) == 0 for value in values)


def test_empty_column_is_nan_never_zero() -> None:
    spec = next(s for s in schema_for("smartphone") if s.name == "accel_x")
    column = empty_column(spec, 5)
    assert column.size == 5
    assert np.isnan(column).all(), "a missing sensor must not read as 0.0"


def test_empty_int_column_is_zero_flags() -> None:
    spec = next(s for s in schema_for("smartphone") if s.name == "quality_flags")
    assert (empty_column(spec, 3) == 0).all()


def test_feature_availability_keys_are_documented() -> None:
    assert "has_magnetometer" in FEATURE_AVAILABILITY_KEYS
    assert "has_gyroscope" in FEATURE_AVAILABILITY_KEYS
