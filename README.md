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
Phase 3 — Sensor Frames, Orientation & Vehicle Alignment (complete)
```

Phase 0 established the engineering foundation. Phase 1 acquired and
forensically analysed IO-VNBD ([`python/idr/dataset/`](python/idr/dataset/)).
Phase 2 added the preprocessing pipeline
([`python/idr/pipeline/`](python/idr/pipeline/)): a canonical schema, an
encoding-aware reader with derived schema repair, trip segmentation, per-pair
smartphone/vehicle synchronization, a tiered reference-label hierarchy, GNSS
freshness, leakage-safe splits, and a causal sequence generator.

Phase 3 added the sensor-frame foundation
([`python/idr/frames/`](python/idr/frames/)): frame conventions, quaternion and
rotation-matrix mathematics, gravity estimation, a complementary orientation
filter, magnetometer quality assessment, and a phone→vehicle alignment engine
that reports a confidence and refuses to fabricate yaw when the data cannot
support it.

Phase 4 added the **baseline dead-reckoning engine**
([`python/idr/navigation/`](python/idr/navigation/)): a strapdown inertial
mechanization that turns those orientation and alignment estimates into a
continuously propagated attitude, velocity and position. It is deliberately
**unaided** — no GNSS after initialization, no reference-velocity feedback, no
filter — because its purpose is to be the honest baseline that Phase 5's fusion
is measured against. An unaided solution on consumer MEMS sensors drifts
quickly, and the validation report says by how much rather than hiding it.

**Still no model, training, EKF, UKF, fusion, non-holonomic constraints, map
matching, or Android code.** See
[`docs/sih/phase4_scope.md`](docs/sih/phase4_scope.md) for the exact boundary and
[`docs/development/engineering_rules.md`](docs/development/engineering_rules.md)
for the rules (Rule 10: don't start future phases early).

Key documents:

- [`reports/io_vnbd/IO_VNBD_Forensic_Report.md`](reports/io_vnbd/IO_VNBD_Forensic_Report.md) — Phase 1 findings
- [`reports/io_vnbd/IO_VNBD_Phase2_Preprocessing_Report.md`](reports/io_vnbd/IO_VNBD_Phase2_Preprocessing_Report.md) — Phase 2 run report
- [`docs/datasets/io_vnbd_phase2_contract.md`](docs/datasets/io_vnbd_phase2_contract.md) — **the canonical data contract**
- [`docs/architecture/sensor_frame_conventions.md`](docs/architecture/sensor_frame_conventions.md) — **the binding frame conventions**
- [`docs/mathematics/phase3_orientation.md`](docs/mathematics/phase3_orientation.md) — orientation and alignment equations
- [`docs/mathematics/phase4_mechanization.md`](docs/mathematics/phase4_mechanization.md) — **the mechanization equations Phase 5 builds on**
- [`reports/sensor_alignment/Phase3_Sensor_Frame_Validation.md`](reports/sensor_alignment/Phase3_Sensor_Frame_Validation.md) — Phase 3 validation report
- [`reports/navigation/Phase4_Inertial_Mechanization_Validation.md`](reports/navigation/Phase4_Inertial_Mechanization_Validation.md) — Phase 4 baseline results and drift analysis

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
│   ├── dataset/      #   IO-VNBD acquisition + forensic inspection (Phase 1)
│   ├── pipeline/     #   Canonicalization + ML-ready pipeline (Phase 2)
│   └── frames/       #   Sensor frames, orientation, alignment (Phase 3)
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

### Canonical preprocessing pipeline (Phase 2)

Turns the raw tree into canonical Parquet plus split/quality metadata
(~4.5 minutes for 564 files). Read-only with respect to raw data:

```bash
python -m idr.pipeline.runner --raw-path data/raw/io_vnbd \
    --processed data/processed/io_vnbd --metadata data/metadata/io_vnbd \
    --report-output reports/io_vnbd
```

### Sensor-frame diagnostics (Phase 3)

Runs the orientation and alignment estimators over the canonical Parquet and
writes a validation report, figures and metadata. Read-only with respect to
both raw and canonical data:

```bash
python -m idr.frames.runner --processed data/processed/io_vnbd \
    --metadata data/metadata/io_vnbd --report-output reports/sensor_alignment
```

### Inertial baseline (Phase 4)

Propagates the unaided dead-reckoning trajectory over the canonical Parquet and
writes the validation report, drift figures and metadata. Read-only with
respect to both raw and canonical data:

```bash
python -m idr.navigation.runner --processed data/processed/io_vnbd \
    --metadata data/metadata/io_vnbd --report-output reports/navigation
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
