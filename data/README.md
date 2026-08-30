# Data

This directory is a **layout**, not a data store. Actual datasets — most
notably IO-VNBD — are never committed to Git.

```text
data/
├── raw/            # Original, unmodified source data. NEVER edit in place (Rule 1).
├── intermediate/   # Partially processed data (regenerable from raw + code).
├── processed/      # Final processed datasets ready for training/evaluation.
├── sequences/      # Extracted/windowed sequences derived from processed data.
└── metadata/       # Small, hand-written docs *about* datasets (schemas, splits,
                     # provenance notes). This is the only subfolder tracked in Git.
```

## Why raw/intermediate/processed/sequences are gitignored

- Datasets like IO-VNBD are large binary/CSV corpora that don't belong in
  Git history — cloning the repo should never require pulling gigabytes of
  sensor logs.
- `intermediate/`, `processed/`, and `sequences/` are all regenerable from
  `raw/` plus code in this repository, so committing them would just be a
  redundant, driftable cache.
- `.gitignore` ignores the contents of these four directories (keeping a
  `.gitkeep` so the empty folders still exist after cloning) and ignores
  `data/IO-VNBD_dataset/` by name specifically, since a local copy of the
  dataset already exists there on some development machines.

## Local dataset note

A local copy of the IO-VNBD dataset may exist under
`data/IO-VNBD_dataset/IO-VNBD-master/` on this machine. It predates Phase 0
and is left in place rather than reorganized — dataset ingestion,
inspection, and preprocessing are explicitly out of scope for Phase 0 (see
`docs/sih/phase0_scope.md`) and are handled in a later phase.

## What IS tracked

Only `data/metadata/` and this README. If you add a genuinely small,
hand-curated artifact that should live in Git (e.g. a manually verified
label file), put it under `data/metadata/` and document why in a commit
message — don't rely on force-adding an ignored path.
