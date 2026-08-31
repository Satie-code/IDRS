"""Synthetic fixtures for Phase 2 pipeline tests.

Every fixture is built in ``tmp_path`` so the suite runs on a clone with no
dataset downloaded, and so assertions can be *exact*: a fixture with a known
600 s gap lets a test assert the segmenter finds exactly that boundary.

The header text mirrors the real IO-VNBD layout, including the leading spaces
and the ``a / b`` satellite field, so the tests exercise the same parsing path
the real data does.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SMARTPHONE_HEADER = (
    "GPS LATITUDE (degrees), GPS LONGITUDE (degrees), GPS ALTITUDE (m), "
    "GPS SPEED (Kmh), GPS ACCURACY (m), GPS ORIENTATION (deg),"
    "GPS SATELLITES IN RANGE, TIME SINCE START (ms), "
    "DATE (YYYY-MO-DD HH-MI-SS_SSS), ACCELEROMETER X (m/s2), "
    "ACCELEROMETER Y (m/s2), ACCELEROMETER Z (m/s2), GRAVITY X (m/s2), "
    "GRAVITY Y (m/s2), GRAVITY Z (m/s2), GYROSCOPE Yaw (rad/s), "
    "GYROSCOPE Pitch (rad/s), GYROSCOPE Roll (rad/s), MAGNETIC FIELD X (uT), "
    "MAGNETIC FIELD Y (uT), MAGNETIC FIELD Z (uT), ORIENTATION (Yaw) (deg), "
    "ORIENTATION (Pitch) (deg), ORIENTATION (Roll ) (deg)"
)

#: The reduced variant: no magnetometer, no orientation.
SMARTPHONE_HEADER_18 = (
    "GPS LATITUDE (degrees), GPS LONGITUDE (degrees), GPS ALTITUDE (m), "
    "GPS SPEED (Kmh), GPS ACCURACY (m), GPS ORIENTATION (deg),"
    "SATELLITES IN RANGE, TIME SINCE START (ms), "
    "DATE (YYYY-MO-DD HH-MI-SS_SSS), ACCELEROMETER X (m/s2), "
    "ACCELEROMETER Y (m/s2), ACCELEROMETER Z (m/s2), GRAVITY X (m/s2), "
    "GRAVITY Y (m/s2), GRAVITY Z (m/s2), GYROSCOPE Yaw (rad/s), "
    "GYROSCOPE Pitch (rad/s), GYROSCOPE Roll (rad/s)"
)

#: Same columns as SMARTPHONE_HEADER but with no ``DATE`` column, so the only
#: clock available for analysis is the elapsed counter.
SMARTPHONE_HEADER_NO_DATE = SMARTPHONE_HEADER.replace("DATE (YYYY-MO-DD HH-MI-SS_SSS), ", "")

#: Same columns as SMARTPHONE_HEADER but with the X/Y/Z gyroscope naming.
SMARTPHONE_HEADER_XYZ = SMARTPHONE_HEADER.replace(
    "GYROSCOPE Yaw (rad/s), GYROSCOPE Pitch (rad/s), GYROSCOPE Roll (rad/s)",
    "GYROSCOPE X (rad/s), GYROSCOPE Y (rad/s), GYROSCOPE Z (rad/s)",
).replace("ORIENTATION (Yaw) (deg)", "ORIENTATION (Azimuth) (deg)")

VEHICLE_HEADER = (
    "No of GPS Satellites Available, Time Since Start of Day (seconds), "
    "Latitude (degrees), Longitude (degrees), Velocity (km/hr), "
    "Heading (degrees), Height (km), Vertical velocity (km/hr), "
    "Sample period (seconds), Yaw Rate (deg/sec), "
    "Indicated Vehicle Speed (km/hr), Indicated Longitudinal Acceleration (g)"
)

#: Wall-clock start used by the smartphone fixtures: 10:07:49.000.
BASE_TIME_OF_DAY_S = 10 * 3600 + 7 * 60 + 49


def smartphone_row(
    index: int,
    *,
    elapsed_ms: int | None = None,
    wall_offset_s: float | None = None,
    lat: float = 52.4,
    lon: float = -1.5,
    speed_kmh: float = 36.0,
    header_18: bool = False,
) -> str:
    """One smartphone data row at a 10 Hz cadence."""
    millis = index * 100 if elapsed_ms is None else elapsed_ms
    offset = (millis / 1000.0) if wall_offset_s is None else wall_offset_s
    total = BASE_TIME_OF_DAY_S + offset
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    seconds = int(total % 60)
    ms = int(round((total - int(total)) * 1000)) % 1000
    stamp = f"2019-09-08 {hours:02d}:{minutes:02d}:{seconds:02d}:{ms:03d}"

    common = (
        f"{lat},{lon},147.5,{speed_kmh},3,241.6,27 / 28,{millis},{stamp},"
        f"0.1,1.5,9.8,0.0,0.0,9.81,0.01,-0.02,0.03"
    )
    if header_18:
        return common
    return common + ",-15.4,1.4,-37.9,185.7,0.7,-0.8"


def vehicle_row(index: int, *, start_s: float, speed_kmh: float = 36.0) -> str:
    return (
        f"11.0,{start_s + index * 0.1:.4f},52.4,-1.5,{speed_kmh},241.7,"
        f"0.110,0.2,0.1,-5.1,{speed_kmh},0.05"
    )


#: Field index of the ``DATE`` value inside a smartphone row.
DATE_FIELD_INDEX = 8


def with_date(row: str, value: str) -> str:
    """Replace a smartphone row's ``DATE`` field, keeping every other field."""
    parts = row.split(",")
    parts[DATE_FIELD_INDEX] = value
    return ",".join(parts)


def without_date(row: str) -> str:
    """Drop a smartphone row's ``DATE`` field, for the no-wall-clock schema."""
    parts = row.split(",")
    del parts[DATE_FIELD_INDEX]
    return ",".join(parts)


def _write(path: Path, header: str, rows: list[str], encoding: str = "utf-8") -> Path:
    path.write_text("\n".join([header, *rows]) + "\n", encoding=encoding)
    return path


@pytest.fixture
def smartphone_csv(tmp_path: Path) -> Path:
    """60 rows, clean, 10 Hz, full 24-column schema."""
    rows = [smartphone_row(i) for i in range(60)]
    return _write(tmp_path / "S-T1.csv", SMARTPHONE_HEADER, rows)


@pytest.fixture
def smartphone_csv_18(tmp_path: Path) -> Path:
    """Reduced schema: no magnetometer or orientation channels."""
    rows = [smartphone_row(i, header_18=True) for i in range(40)]
    return _write(tmp_path / "S-T18.csv", SMARTPHONE_HEADER_18, rows)


@pytest.fixture
def smartphone_csv_xyz(tmp_path: Path) -> Path:
    """Same data as ``smartphone_csv`` under the X/Y/Z gyroscope naming."""
    rows = [smartphone_row(i) for i in range(60)]
    return _write(tmp_path / "S-TXYZ.csv", SMARTPHONE_HEADER_XYZ, rows)


@pytest.fixture
def vehicle_csv(tmp_path: Path) -> Path:
    """60 rows aligned to the smartphone fixture's wall clock."""
    rows = [vehicle_row(i, start_s=BASE_TIME_OF_DAY_S) for i in range(60)]
    return _write(tmp_path / "V-T1.csv", VEHICLE_HEADER, rows)


@pytest.fixture
def vehicle_csv_offset(tmp_path: Path) -> Path:
    """Vehicle stream running exactly 120 s behind the smartphone clock."""
    rows = [vehicle_row(i, start_s=BASE_TIME_OF_DAY_S - 120.0) for i in range(600)]
    return _write(tmp_path / "V-OFF.csv", VEHICLE_HEADER, rows)


@pytest.fixture
def smartphone_with_gap(tmp_path: Path) -> Path:
    """Two trips separated by a 600 s wall-clock gap: 0-30 s, then 630-660 s."""
    rows = [smartphone_row(i) for i in range(301)]
    rows += [
        smartphone_row(i, elapsed_ms=630_000 + i * 100, wall_offset_s=630.0 + i * 0.1)
        for i in range(301)
    ]
    return _write(tmp_path / "S-GAP.csv", SMARTPHONE_HEADER, rows)


@pytest.fixture
def smartphone_with_counter_reset(tmp_path: Path) -> Path:
    """Elapsed counter restarts at 0 while the wall clock runs continuously.

    Reproduces the real ``S-A4``/``S-S2`` class of defect: the recording is one
    continuous drive, so the pipeline must NOT split or sort it.
    """
    rows = [smartphone_row(i) for i in range(200)]
    rows += [
        smartphone_row(i, elapsed_ms=i * 100, wall_offset_s=20.0 + i * 0.1) for i in range(200)
    ]
    return _write(tmp_path / "S-RESET.csv", SMARTPHONE_HEADER, rows)


@pytest.fixture
def smartphone_stale_gnss(tmp_path: Path) -> Path:
    """Position held for 20 s, then moving again — the real staleness pattern."""
    rows = [smartphone_row(i, lat=52.4) for i in range(200)]
    rows += [smartphone_row(200 + i, lat=52.4 + i * 0.0001) for i in range(200)]
    return _write(tmp_path / "S-STALE.csv", SMARTPHONE_HEADER, rows)


@pytest.fixture
def smartphone_shifted(tmp_path: Path) -> Path:
    """The S-A4 defect, reproduced faithfully.

    The real file pairs a header carrying a trailing empty 25th field with data
    rows that also carry 25 fields, the extra one being an empty at index 6.
    Both halves matter: without the trailing header field pandas silently
    consumes column 0 as an index instead of surfacing a surplus column, which
    is a different failure mode from the one this repair targets.
    """
    rows = []
    for index in range(50):
        parts = smartphone_row(index).split(",")
        parts.insert(6, "")
        rows.append(",".join(parts))
    return _write(tmp_path / "S-SHIFT.csv", SMARTPHONE_HEADER + ",", rows)


@pytest.fixture
def smartphone_no_date_reset(tmp_path: Path) -> Path:
    """No wall clock, and the elapsed counter restarts.

    The mirror image of ``smartphone_with_counter_reset``: with no ``DATE``
    column there is no continuous clock to prove the recording ran on, so the
    reset is the only evidence available and *must* be treated as a real trip
    boundary rather than sorted away.
    """
    rows = [without_date(smartphone_row(i)) for i in range(200)]
    rows += [without_date(smartphone_row(i)) for i in range(200)]
    return _write(tmp_path / "S-NODATE.csv", SMARTPHONE_HEADER_NO_DATE, rows)


@pytest.fixture
def smartphone_date_without_time(tmp_path: Path) -> Path:
    """``DATE`` carries a calendar date but no time of day."""
    rows = [with_date(smartphone_row(i), "2019-09-08") for i in range(30)]
    return _write(tmp_path / "S-DATEONLY.csv", SMARTPHONE_HEADER, rows)


@pytest.fixture
def smartphone_date_without_seconds(tmp_path: Path) -> Path:
    """``DATE`` carries hours and minutes only — too coarse to use."""
    rows = [with_date(smartphone_row(i), "2019-09-08 10:07") for i in range(30)]
    return _write(tmp_path / "S-NOSEC.csv", SMARTPHONE_HEADER, rows)


@pytest.fixture
def smartphone_header_only(tmp_path: Path) -> Path:
    """A header with no data rows behind it."""
    return _write(tmp_path / "S-EMPTY.csv", SMARTPHONE_HEADER, [])


@pytest.fixture
def smartphone_satellites_inverted(tmp_path: Path) -> Path:
    """``a / b`` with more satellites 'in use' than 'in view' — impossible."""
    rows = [smartphone_row(i).replace("27 / 28", "28 / 27") for i in range(30)]
    return _write(tmp_path / "S-SAT.csv", SMARTPHONE_HEADER, rows)


@pytest.fixture
def smartphone_single_row(tmp_path: Path) -> Path:
    """One data row: no interval can be measured from it."""
    return _write(tmp_path / "S-ONE.csv", SMARTPHONE_HEADER, [smartphone_row(0)])


@pytest.fixture
def smartphone_frozen_clock(tmp_path: Path) -> Path:
    """Every row carries the same timestamp, so no cadence can be measured."""
    rows = [smartphone_row(0, lat=52.4 + i * 0.0001) for i in range(40)]
    return _write(tmp_path / "S-FROZEN.csv", SMARTPHONE_HEADER, rows)


@pytest.fixture
def smartphone_out_of_order(tmp_path: Path) -> Path:
    """Two adjacent rows written in the wrong order inside a single trip.

    A sub-second backward step is logger noise, not a new recording, so it must
    be sorted locally rather than treated as a trip boundary.
    """
    rows = [smartphone_row(i) for i in range(60)]
    rows[20], rows[21] = rows[21], rows[20]
    return _write(tmp_path / "S-ORDER.csv", SMARTPHONE_HEADER, rows)


@pytest.fixture
def smartphone_cp1252(tmp_path: Path) -> Path:
    """A header carrying raw cp1252 bytes, as the real smartphone files do."""
    header = SMARTPHONE_HEADER.replace("(m/s2)", "(m/s\xb2)").replace("(deg)", "(\xb0)")
    rows = [smartphone_row(i) for i in range(30)]
    path = tmp_path / "S-CP.csv"
    path.write_bytes(("\n".join([header, *rows]) + "\n").encode("cp1252"))
    return path
