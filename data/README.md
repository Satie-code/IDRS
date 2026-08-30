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

## Acquired IO-VNBD content

`data/raw/io_vnbd/` holds the real IO-VNBD CSV content, materialized from the
Git LFS pointers in `data/IO-VNBD_dataset/` and SHA-256 verified against the
upstream objects. It is gitignored (`data/raw/*`) and regenerable — see
`docs/datasets/io_vnbd.md` for the exact command. Provenance for every file
is recorded in `data/metadata/io_vnbd/acquisition_manifest.csv`, which *is*
tracked.

## Local dataset note

`data/IO-VNBD_dataset/IO-VNBD-master/` predates Phase 0 and is left in place
rather than reorganized. Phase 1 established that **every CSV and JPG in it is
a Git LFS pointer stub, not data** — it is kept, untouched, as the provenance
record that acquisition verifies against. Never parse a file from this tree as
CSV; see `docs/datasets/io_vnbd.md`.

## What IS tracked

Only `data/metadata/` and this README. If you add a genuinely small,
hand-curated artifact that should live in Git (e.g. a manually verified
label file), put it under `data/metadata/` and document why in a commit
message — don't rely on force-adding an ignored path.
