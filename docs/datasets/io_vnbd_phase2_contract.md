# IO-VNBD Phase 2 — Canonical Data Contract

This is the contract Phase 3 and later build on. It defines what a canonical
record contains, what it guarantees, and — just as importantly — what it does
**not** guarantee.

Generated artifacts live in `data/metadata/io_vnbd/`; the run report is
`reports/io_vnbd/IO_VNBD_Phase2_Preprocessing_Report.md`. Regenerate everything
with:

```bash
python -m idr.pipeline.runner --raw-path data/raw/io_vnbd \
    --processed data/processed/io_vnbd --metadata data/metadata/io_vnbd \
    --report-output reports/io_vnbd
```

## 1. Data model

```text
RAW  (data/raw/io_vnbd/)          immutable; never written to
 ↓
CANONICAL (data/processed/io_vnbd/<stream>/<segment>.parquet)
 ↓
ML-READY  (generated on demand by idr.pipeline.sequences)
```

Canonical data is **Parquet, snappy-compressed, one file per segment**.

Parquet was chosen because a model needing six channels should not pay to read
sixty (columnar projection), numeric dtypes survive a round trip exactly,
compression is good on this volume, chunked reads are supported, and
pandas/pyarrow work identically on Windows. Pickle was rejected: it is unsafe
to load, version-fragile, and unreadable outside Python.

ML-ready sequences are **not materialized**. A generator produces windows from
canonical Parquet on demand, so changing the window geometry does not require
regenerating a corpus.

## 2. Storage layout

```text
data/processed/io_vnbd/
├── smartphone/<segment_id>.parquet
└── vehicle/<segment_id>.parquet
```

`segment_id` is `<payload_hash[:12]>:<segment_index:03d>`, with `:` replaced by
`_` in filenames.

## 3. Identity fields

| Field | Type | Meaning |
|---|---|---|
| `dataset_family` | string | Which source tree the file came from |
| `source_file` | string | Path relative to the raw root |
| `file_hash` | string | SHA-256 of the whole source file |
| `payload_hash` | string | SHA-256 of the data rows **excluding the header** |
| `session_id` | string | Session parsed from the filename |
| `segment_id` | string | Logical trip within the file |
| `stream` | string | `smartphone` or `vehicle` |

**Use `payload_hash`, not `file_hash` or `source_file`, for any grouping that
must prevent leakage.** Phase 1 found 235 byte-identical duplicate files; Phase
2 found 15 further groups whose files differ *only in header labels* while
their data rows are byte-identical (314 distinct payloads behind 329 distinct
file hashes). A file-hash split would still leak.

Grouping is the **transitive closure** over shared `payload_hash` and shared
`(dataset_family, session_id)`, which matters because the same recording is
filed under several family directories: session `VW14B`, for example, spans all
four families across seven files and three payloads, and the closure collapses
them into one group. Whole groups are then assigned to a split, so a segment
never straddles one.

An independent audit at the end of the Phase 2 run went further than the
payload hash and hashed the *canonical sensor matrix* of every written segment
(rounded to 1e-6). It found **58 groups of segments whose canonical content is
identical despite differing payload hashes** — files that differ only in raw
numeric formatting — and **none of those 58 groups spans a split**. Phase 3 may
therefore rely on the split boundary at content level, not merely at byte
level. This is a measured property of the current run, not a proof: a future
re-split should re-run the same audit.

## 4. Time fields

| Field | Type | Unit | Meaning |
|---|---|---|---|
| `source_time_s` | float64 | s | The source clock, unmodified |
| `analysis_time_s` | float64 | s | The continuous clock used for analysis |
| `analysis_time_source` | string | — | `wall_clock`, `elapsed`, or `time_of_day` |
| `time_of_day_s` | float64 | s | Seconds since local midnight |
| `elapsed_s` | float64 | s | Seconds from this segment's first sample |
| `timestamp_utc` | datetime64[ns] | — | Smartphone wall clock; `NaT` for vehicle |

Three clocks exist in the source and they are not interchangeable:

- smartphone `TIME SINCE START (ms)` — an elapsed counter that **restarts
  mid-recording in 15 files**;
- smartphone `DATE` — `YYYY-MM-DD HH:MM:SS:mmm`, milliseconds after a *colon*,
  which no standard parser accepts. It runs continuously;
- vehicle `Time Since Start of Day (seconds)` — no date component.

`analysis_time_s` prefers the smartphone wall clock precisely because the
elapsed counter resets. Segmentation and synchronization use `analysis_time_s`;
`source_time_s` is preserved so nothing is lost.

`timestamp_utc` is **not timezone-resolved**. The name reflects the column's
dtype, not a verified UTC offset — the dataset never states its timezone.

## 5. Smartphone channels

`accel_x/y/z` (m/s²), `gyro_x/y/z` (rad/s), `mag_x/y/z` (µT),
`gravity_x/y/z` (m/s²), `orientation_yaw/pitch/roll` (deg).

**Gyroscope axes are mapped by column position.** 158 files label these columns
`Yaw/Pitch/Roll` and 73 label the same positions `X/Y/Z`; two files carrying the
same recording under the two headers were verified to have byte-identical data
rows, so the labels are aliases for the same channels. Canonical names are
`gyro_x/y/z` in the **device frame**. The yaw/pitch/roll labels are recorded as
aliases and are **not** trustworthy axis semantics — establishing the
device→vehicle rotation is Phase 3's job.

## 6. GNSS channels

`latitude`, `longitude` (float64 — 1e-7 deg matters), `altitude_m`,
`gnss_speed_mps`, `gnss_accuracy_m`, `gnss_heading_deg`,
`gnss_satellites_raw` / `_in_use` / `_in_view`, `gnss_fix_available`,
`gnss_position_fresh`, `gnss_stale_duration_s`.

- `gnss_satellites_raw` preserves the source string verbatim (e.g. `"27 / 28"`).
  The derived integers interpret it as *in use / in view*; that reading is
  supported by the left value never exceeding the right, but **is not stated by
  any dataset documentation**. Never call `float()` on the raw field.
- **A fix on every row is not a fresh fix on every row.** The smartphone
  position is forward-filled, refreshing roughly every 9 s while rows arrive at
  ~10 Hz. Check `gnss_position_fresh` before treating a position as a
  measurement.
- `gnss_stale_duration_s` is seconds since the position last changed.

## 7. Vehicle / reference channels

`ref_speed_mps`, `ref_indicated_speed_mps`, `ref_heading_deg`,
`ref_vertical_speed_mps`, `ref_yaw_rate_dps`, `ref_accel_long_mps2`,
`ref_accel_lat_mps2`, `ref_steering_angle_deg`, `ref_wheel_speed_fl/fr/rl/rr`,
`ref_engine_speed_rpm`, `ref_gear`, `ref_gear_requested`,
`ref_brake_pressure_psi`, `ref_brake_position`, `ref_handbrake`,
`ref_clutch_position`, `ref_accelerator_pedal`, `ref_battery_voltage_v`,
`ref_coolant_temp_c`, `ref_air_temp_c`, `ref_sample_period_s`.

Units are normalized on ingest: km/h→m/s, g→m/s² (9.80665), km→m. Both the
canonical and the source unit are recorded in `canonical_schema.json`.

## 8. Reference labels

| Field | Meaning |
|---|---|
| `ref_velocity_mps` | The candidate training target |
| `label_source` | Which tier produced it |
| `label_quality` | `high` / `medium` / `low` / `none` |
| `label_available` | Whether this row has a usable target |

Hierarchy, highest first:

1. `TIER_1_VEHICLE_SYNCED` — vehicle velocity, alignment confirmed by speed
   cross-correlation. **Quality: high.**
2. `TIER_2_VEHICLE_CLOCK` — vehicle velocity, clock-only alignment. Medium.
3. `TIER_3_SMARTPHONE_GNSS` — smartphone GNSS speed, fresh rows only. **Low**,
   and flagged `REFERENCE_LOW_CONFIDENCE`, because the underlying position
   updates at ~0.1 Hz.
4. `UNAVAILABLE` — no defensible target. Reported, never guessed.

Vehicle velocity is interpolated onto smartphone sample times only where a
reference sample lies within 0.5 s; beyond that the row is left unlabelled
rather than extrapolated.

## 9. Quality flags

`quality_flags` is an integer bitmask (`idr.pipeline.canonical.QualityFlag`);
decode with `describe_flags`.

| Flag | Meaning |
|---|---|
| `MISSING_SENSOR` | One or more channels absent for this row |
| `INVALID_TIMESTAMP` | Timestamp unparseable or non-finite |
| `DUPLICATE_TIMESTAMP` | This timestamp occurs more than once |
| `NON_MONOTONIC_TIMESTAMP` | Not greater than its predecessor |
| `ENCODING_WARNING` | Source needed a non-UTF-8 decode |
| `SCHEMA_REPAIR` | Row came from a column-repaired file |
| `GNSS_STALE` | Position held beyond the freshness threshold |
| `SYNC_UNRESOLVED` | The owning pair has no reliable alignment |
| `REFERENCE_LOW_CONFIDENCE` | Label is low-confidence or derived |
| `SEGMENT_BOUNDARY` | First sample of a segment |

**No row is ever dropped for being suspicious.** Problems become flags.

## 10. Missing values and feature availability

Missing numerics are `NaN`, never `0` — a zero is indistinguishable from a real
stationary reading. Structurally absent channels (the reduced 18-column
smartphone files have no magnetometer or orientation) are additionally recorded
per file in `canonical_field_mapping.json` under `has_magnetometer`,
`has_orientation`, `has_gravity`, `has_gyroscope`, `has_accelerometer`,
`has_gnss`, `has_gnss_accuracy`.

Check availability before using a channel, or a structurally missing sensor
will look like a sensor that merely happened to be NaN.

## 11. Segment semantics

A segment is one logical trip. A new segment starts at:

- a backward jump on `analysis_time_s` of ≥ 1.0 s (a clock reset), or
- a forward gap exceeding `max(100 × the stream's median interval, 10 s)`.

The threshold is relative because the dataset spans ~2 Hz to ~1000 Hz streams;
a single absolute value would over-segment slow streams and under-segment fast
ones. Every segment records its exact reason in `segment_inventory.csv`.

Within a segment, rows may be stably sorted by time to fix local inversions —
**never across a boundary**, so distinct trips can never merge.

## 12. Split semantics

`splits.json` and `split_manifest.csv` assign every segment to exactly one of
train/validation/test.

Grouping is the transitive closure over shared `payload_hash` and shared
`(dataset_family, session_id)`. Each group is assigned whole via
`sha256(seed:group_key)` mapped onto cumulative ratio boundaries — deterministic
and independent of iteration order or wall-clock time.

Guarantees, each covered by a test:

- identical content never appears in two splits;
- a session never spans splits;
- every segment of a file shares one split, so overlapping windows from one trip
  cannot straddle a boundary;
- the same configuration always reproduces the same assignment.

**Rows and windows are never split randomly.**

## 13. Sequence generation

`idr.pipeline.sequences.generate_windows` yields causal windows. For a target
at time *t*, the window contains only samples with time ≤ *t*. Nothing after
the target can enter. A test asserts this directly.

`WindowConfig(duration_seconds=5.0, stride_seconds=0.1, min_coverage=0.8)` are
defaults, **not findings** — later experiments are expected to change them.

## 14. What Phase 3 may assume

- Every row has `source_time_s` as written, plus `analysis_time_s` and
  `elapsed_s`.
- Within a segment, `elapsed_s` is non-decreasing and starts at 0.
- A segment never spans a clock reset or a large gap.
- Units are as documented; conversions are recorded.
- Raw data is unmodified and re-derivable.

## 15. What Phase 3 must NOT assume

- **Not** 10 Hz smartphone sampling. Measured rates include ~2 Hz, ~10 Hz,
  ~24 Hz and ~1000 Hz families. Read the measured rate per segment.
- **Not** perfect synchronization. Check `status` and `offset_confidence` per
  pair; some pairs drift, and some are `UNRESOLVED`.
- **Not** a fresh GNSS fix on every row. Check `gnss_position_fresh`.
- **Not** that gyroscope yaw/pitch/roll labels denote vehicle axes.
- **Not** that one file is one trip. Use `segment_id`.
- **Not** that file identity is content identity. Group on `payload_hash`.
- **Not** that any GNSS gap here is a real outage. **There are none.** Any
  GNSS-denial scenario must be synthesized in a later phase and labelled
  synthetic.
- **Not** that `timestamp_utc` is UTC.
