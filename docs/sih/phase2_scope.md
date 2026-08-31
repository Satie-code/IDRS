# SIH26168 — Phase 2 Scope Record

Phase 2 turned the raw IO-VNBD tree into a **canonical, quality-annotated,
leakage-safe dataset** plus the infrastructure to generate ML-ready sequences
from it. It is a data-engineering phase: it trains nothing and implements no
navigation algorithm.

## What Phase 2 delivered

Under `python/idr/pipeline/`:

- **`canonical.py`** — the canonical schema: field names, dtypes, units, the
  closed `QualityFlag` vocabulary, and feature-availability keys.
- **`mapping.py`** — source→canonical column mapping, unit conversion, and
  parsing of the `"27 / 28"` satellite field.
- **`repair.py`** — detection and *derived* repair of the one-column shift in
  `S-A4.csv`. The raw file is never touched.
- **`reader.py`** — encoding-aware chunked reading into canonical frames, with
  payload hashing.
- **`timeline.py`** — timestamp normalization, duplicate/ordering diagnosis,
  and trip segmentation on clock resets and large gaps.
- **`freshness.py`** — GNSS fix availability, position freshness, staleness.
- **`sync.py`** — per-pair smartphone/vehicle alignment by speed
  cross-correlation, with method, confidence, and drift.
- **`reference.py`** — the tiered reference-label hierarchy.
- **`splits.py`** — payload-hash-grouped, size-balanced, deterministic splits,
  plus a content-level audit that re-checks the result against parsed sensor
  values rather than raw bytes.
- **`sequences.py`** — the causal window generator.
- **`runner.py`**, **`plots.py`**, **`report.py`** — orchestration, diagnostics,
  and the run report.

Outputs: canonical Parquet in `data/processed/io_vnbd/`, thirteen metadata
artifacts in `data/metadata/io_vnbd/`, eight plots and the preprocessing
report in `reports/io_vnbd/`.

## What Phase 2 explicitly did NOT implement

No neural-network architecture, PyTorch, training, optimization, or inference.
No INS, EKF, UKF, non-holonomic constraints, map matching, GNSS fusion, or
navigation state estimation. No Android, ONNX, or C++ navigation code. No
synthetic GNSS outage simulator. No training normalization statistics — those
must be derived from the training partition alone, in the phase that trains.
No device→vehicle rotation estimation (Phase 3).

## Findings that changed the design

Three defects were discovered during Phase 2 and drove real design decisions:

1. **Header-only duplicates.** Files whose bytes differ only in header labels
   carry byte-identical data rows. A file-hash split would leak, so splits
   group on a **payload hash** instead (314 distinct payloads vs 329 file
   hashes).
2. **Elapsed-clock resets.** 15 smartphone files restart the elapsed counter
   mid-recording while their wall clock runs on. Sorting by the counter
   interleaves two trips into a trajectory that never happened, so
   segmentation uses the continuous wall clock and never sorts across a
   segment boundary.
3. **Gyroscope axis aliasing.** 158 files label the gyroscope columns
   `Yaw/Pitch/Roll` and 73 label the same positions `X/Y/Z`; the data rows are
   identical. Mapping is positional and the labels are recorded as untrusted
   aliases.

## Data safety

Raw data is never modified. A regression test hashes every input before and
after a full pipeline run and asserts equality, and the raw tree was
re-verified against the Phase 1 acquisition manifest after the real-data run.

## Why this boundary

Rule 10 in
[`../development/engineering_rules.md`](../development/engineering_rules.md):
do not start future phases early. The contract in
[`../datasets/io_vnbd_phase2_contract.md`](../datasets/io_vnbd_phase2_contract.md)
states exactly what Phase 3 may and may not assume — building a model before
that was settled would have meant rebuilding it.
