# SIH26168 — Intelligent Dead Reckoning

## Problem summary

ISRO Smart India Hackathon problem **SIH26168**: smartphones lose reliable
positioning whenever GNSS signal is degraded or unavailable (urban canyons,
tunnels, indoor transitions). This project builds an AI-ML-based dead
reckoning system that keeps a smartphone's navigation solution continuous
and accurate through GNSS outages, using only onboard IMU sensors
(accelerometer, gyroscope, magnetometer) plus GNSS when it is available.

## High-level objective

Maintain smartphone navigation continuity during GNSS outages by combining:

- automatic phone-to-vehicle alignment (no fixed mounting assumed),
- AI/ML-based vehicle motion estimation,
- strapdown inertial navigation,
- GNSS+INS fusion with outage handling and reacquisition,
- non-holonomic vehicle motion constraints,
- offline road-map matching,
- on-device inference,

targeting roughly **10 Hz** navigation updates on a smartphone, with a
navigation core portable enough to also run on an external-IMU edge device.

## Current project phase

```text
Phase 1 — Dataset Acquisition & Forensic Analysis (complete)
```

Phase 0 established the engineering foundation. Phase 1 added the IO-VNBD
dataset acquisition and forensic-inspection layer under
[`python/idr/dataset/`](python/idr/dataset/): LFS-aware acquisition with
SHA-256 verification, file/session inventory, schema extraction, timestamp
and sampling measurement, missingness and range diagnostics, S/V
synchronization analysis, and GNSS availability/staleness analysis.

**Still no navigation, sensor-fusion, or ML functionality** — Phase 1 is
observational only and never modifies raw data. See
[`docs/sih/phase1_scope.md`](docs/sih/phase1_scope.md) for the exact
boundary and [`docs/development/engineering_rules.md`](docs/development/engineering_rules.md)
for the rules (Rule 10: don't start future phases early) that keep it that
way until the next phase deliberately begins.

The forensic findings — including several that materially constrain Phase 2 —
are in [`reports/io_vnbd/IO_VNBD_Forensic_Report.md`](reports/io_vnbd/IO_VNBD_Forensic_Report.md).

## Architecture overview

```text
Data
 ↓
ML / Research   (Python: python/idr/)
 ↓
Navigation Core (C++: core/)
 ↓
Android / Edge  (future)
```

See [`docs/architecture/system_architecture.md`](docs/architecture/system_architecture.md)
and [`docs/architecture/software_architecture.md`](docs/architecture/software_architecture.md)
for the full breakdown.

## Repository structure

```text
├── python/idr/       # ML / research Python package
│   └── dataset/      #   IO-VNBD acquisition + forensic inspection (Phase 1)
├── core/             # C++ navigation core (CMake: include/, src/, tests/)
├── tools/health_check/  # Foundation health check (python -m tools.health_check)
├── tests/            # Python unit tests (tests/python) + integration tests
├── configs/          # Environment configuration (development.yaml, ...)
├── data/             # Dataset layout — datasets themselves are gitignored
├── models/           # Checkpoints / exported artifacts / metadata
├── docs/             # Architecture, development, dataset, and SIH docs
├── benchmarks/       # Reproducible benchmarks (empty until a later phase)
├── reports/          # Generated reports (gitignored contents)
├── scripts/          # Operational scripts
└── .github/workflows/  # CI
```

## Development setup

Commands below are verified against this repository's environment — see
[`docs/development/environment.md`](docs/development/environment.md) for
exact tool versions and machine-specific notes (e.g. why plain `make`/`g++`
on PATH aren't used for the C++ build here).

### Python

```bash
python -m venv .venv
source .venv/Scripts/activate      # Git Bash on Windows; use .venv\Scripts\Activate.ps1 in PowerShell
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
pytest --cov          # with coverage
```

Lint:

```bash
ruff check .
ruff format --check .
```

Type check:

```bash
mypy
```

Health check:

```bash
python -m tools.health_check
```

### C++

```bash
cmake -S . -B build
cmake --build build --config Debug
ctest --test-dir build --output-on-failure -C Debug
```

### Dataset inspection (Phase 1)

The dataset itself is never committed. To materialize it locally from the
official upstream source (~1.8 GB, every object SHA-256 verified) and then
inspect it:

```bash
# 1. Acquire real content for the Git LFS pointers shipped in the repo copy
python -c "from pathlib import Path; from idr.dataset.acquire import acquire_pointers; acquire_pointers(Path('data/IO-VNBD_dataset/IO-VNBD-master'), Path('data/raw/io_vnbd'))"

# 2. Fast inspection (inventory + schemas only)
python -m idr.dataset.inspect --dataset-path data/raw/io_vnbd

# 3. Full forensic scan (timing, missingness, ranges, GNSS, sync, plots, report)
python -m idr.dataset.inspect --dataset-path data/raw/io_vnbd \
    --output data/metadata/io_vnbd --report-output reports/io_vnbd --deep
```

Full details, including pre-commit setup, in
[`docs/development/setup.md`](docs/development/setup.md).

## Documentation

- [`docs/development/`](docs/development/) — setup, coding standards,
  testing strategy, engineering rules, environment snapshot.
- [`docs/architecture/`](docs/architecture/) — system and software
  architecture.
- [`docs/sih/`](docs/sih/) — SIH-specific scope notes.
- [`docs/datasets/`](docs/datasets/), [`data/README.md`](data/README.md) —
  dataset handling rules (raw data is never committed to Git).

## License

MIT — see [`LICENSE`](LICENSE).
