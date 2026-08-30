"""Synthetic fixtures for dataset-inspection tests.

Deliberately tiny and hand-built: the unit tests must never need the real
1.7 GB IO-VNBD dataset, and controlled inputs are the only way to assert that
a statistic is *correct* rather than merely reproducible.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SMARTPHONE_HEADER = (
    "GPS LATITUDE (degrees), GPS LONGITUDE (degrees), GPS ALTITUDE (m), "
    "GPS SPEED (Kmh), GPS ACCURACY (m), GPS ORIENTATION (deg),"
    "GPS SATELLITES IN RANGE, TIME SINCE START (ms), "
    "DATE (YYYY-MO-DD HH-MI-SS_SSS), ACCELEROMETER X (m/s2) , "
    "ACCELEROMETER Y (m/s2), ACCELEROMETER Z (m/s2), GYROSCOPE Yaw (rad/s), "
    "GYROSCOPE Pitch (rad/s), GYROSCOPE Roll (rad/s)"
)

VEHICLE_HEADER = (
    "No of GPS Satellites Available, Time Since Start of Day (seconds), "
    "Latitude (degrees), Longitude (degrees), Velocity (km/hr), "
    "Heading (degrees), Yaw Rate (deg/sec)"
)


def _smartphone_row(index: int, *, lat: float = 52.4, lon: float = -1.5) -> str:
    millis = index * 100
    whole_seconds = 49 + millis // 1000
    remainder_ms = millis % 1000
    # IO-VNBD writes milliseconds after a colon, not a decimal point.
    timestamp = f"2019-09-08 10:07:{whole_seconds:02d}:{remainder_ms:03d}"
    return (
        f"{lat},{lon},147.5,10.0,3,241.6,27 / 28,{millis},{timestamp},0.1,1.5,9.8,0.01,-0.02,0.03"
    )


@pytest.fixture
def smartphone_csv(tmp_path: Path) -> Path:
    """20 rows at an exact 10 Hz with a valid fix throughout."""
    rows = [SMARTPHONE_HEADER] + [_smartphone_row(i) for i in range(20)]
    path = tmp_path / "S-T1.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def vehicle_csv(tmp_path: Path) -> Path:
    """20 rows at 10 Hz starting one hour earlier than ``smartphone_csv``.

    The one-hour difference reproduces the UTC/local-time relationship
    measured in the real dataset so synchronization logic can be asserted.
    """
    start = 10 * 3600 + 7 * 60 + 49.0 - 3600.0
    rows = [VEHICLE_HEADER]
    for index in range(20):
        rows.append(f"11.0,{start + index * 0.1:.1f},52.4,-1.5,36.0,241.7,-5.1")
    path = tmp_path / "V-T1.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def gnss_gap_csv(tmp_path: Path) -> Path:
    """30 rows where rows 10-19 have an empty GNSS fix (a 1.0 s gap)."""
    rows = [SMARTPHONE_HEADER]
    for index in range(30):
        millis = index * 100
        if 10 <= index < 20:
            rows.append(
                f",,147.5,10.0,3,241.6,0 / 0,{millis},"
                f"2019-09-08 10:07:49:000,0.1,1.5,9.8,0.01,-0.02,0.03"
            )
        else:
            rows.append(_smartphone_row(index))
    path = tmp_path / "S-GAP.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def lfs_pointer(tmp_path: Path) -> Path:
    path = tmp_path / "S-POINTER.csv"
    path.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:"
        "e79a2eea18143b825438f9a11c7d724a97c9e18329776defd4d9291d03d4aaca\n"
        "size 9631499\n",
        encoding="utf-8",
    )
    return path
