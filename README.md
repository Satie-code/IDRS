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
Phase 0 — Foundation
```

Only the engineering foundation exists right now: repo structure, Python
and C++ scaffolding, testing, configuration, logging, and CI. **No
navigation, sensor-fusion, or ML functionality has been implemented yet.**
See [`docs/sih/phase0_scope.md`](docs/sih/phase0_scope.md) for the exact
boundary and [`docs/development/engineering_rules.md`](docs/development/engineering_rules.md)
for the rules (Rule 10: don't start future phases early) that keep it that
way until the next phase deliberately begins.

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
├── python/idr/       # ML / research Python package (config, logging, utils)
├── core/             # C++ navigation core (CMake: include/, src/, tests/)
├── tools/health_check/  # Foundation health check (python -m tools.health_check)
├── tests/            # Python unit tests (tests/python) + integration tests
├── configs/          # Environment configuration (development.yaml, ...)
├── data/             # Dataset layout — datasets themselves are gitignored
├── models/           # Checkpoints / exported artifacts / metadata
├── docs/             # Architecture, development, dataset, and SIH docs
├── benchmarks/        # Reproducible benchmarks (empty until Phase 1+)
├── reports/          # Generated reports (gitignored contents)
├── scripts/          # Operational scripts (empty in Phase 0)
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
