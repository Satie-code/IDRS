# SIH26168 — Phase 1 Scope Record

Phase 1 built the **IO-VNBD dataset acquisition and forensic-analysis layer**
that Phase 2 preprocessing will depend on. It is an observational phase: it
establishes what the dataset actually contains, and changes nothing about it.

## What Phase 1 delivered

- **Acquisition** (`idr.dataset.acquire`) — materializes Git LFS objects from
  the official upstream repository, verifying every object's SHA-256 against
  the oid in its pointer.
- **LFS detection** (`idr.dataset.lfs`) — so a pointer stub is never parsed as
  if it were data.
- **Inventory & session discovery** (`idr.dataset.inventory`) — files, dataset
  families, sessions, S/V pairing, driver/category extraction.
- **Schema forensics** (`idr.dataset.schema`) — encoding detection, column
  names/units, semantic role assignment.
- **Single-pass scanning** (`idr.dataset.scan`) — chunked, memory-bounded reads
  producing every per-stream statistic in one pass per file.
- **Timing** (`idr.dataset.timestamps`) — measured sampling rate, interval
  statistics, monotonicity, duplicates, stability classification.
- **Quality** (`idr.dataset.quality`) — missingness, range/sanity diagnostics,
  transparent session quality profiles.
- **Synchronization** (`idr.dataset.synchronization`) — S/V clock offset,
  drift, and overlap, diagnosed but never corrected.
- **GNSS** (`idr.dataset.outages`) — availability, gap intervals, and
  position-update-rate/staleness analysis.
- **Reporting** (`idr.dataset.report`, `idr.dataset.forensic_report`,
  `idr.dataset.plots`) — representative-session selection, diagnostic plots,
  and a generated forensic report.
- **CLI** (`python -m idr.dataset.inspect`) with fast and `--deep` modes.

Machine-readable outputs live in `data/metadata/io_vnbd/`; the human-readable
report and plots in `reports/io_vnbd/`.

## What Phase 1 explicitly did NOT implement

No ML models, PyTorch/TensorFlow/ONNX, training, window generation, label
construction, or ML feature normalization. No INS, dead reckoning, EKF/UKF,
non-holonomic constraints, map matching, GNSS fusion, or navigation state
estimation. No Android, JNI, or C++ navigation algorithms. The C++ core is
unchanged from Phase 0.

## Data safety

Raw data is never modified. The original LFS-pointer tree under
`data/IO-VNBD_dataset/` was left byte-for-byte untouched and remains the
provenance record; acquired content was written to the separate, gitignored
`data/raw/io_vnbd/`. A regression test
(`test_inspection_does_not_modify_the_dataset`) asserts that a full deep
inspection leaves every input byte unchanged.

## Why this boundary

Rule 10 in
[`../development/engineering_rules.md`](../development/engineering_rules.md):
do not start future phases early. Phase 1 surfaced several findings that
directly constrain how Phase 2 must be built (see §19–20 of the forensic
report) — building preprocessing before knowing them would have meant
rewriting it.
