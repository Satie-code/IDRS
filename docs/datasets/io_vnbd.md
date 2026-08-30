# IO-VNBD

Inertial Odometry Vehicle Navigation Benchmark Dataset — the primary dataset
for this project. This page is the durable summary; the full measured
analysis lives in
[`reports/io_vnbd/IO_VNBD_Forensic_Report.md`](../../reports/io_vnbd/IO_VNBD_Forensic_Report.md)
and the machine-readable artifacts in `data/metadata/io_vnbd/`.

## Source and provenance

- Official upstream: `https://github.com/onyekpeu/IO-VNBD`
- Recorded with a research vehicle on public roads in the **United Kingdom,
  Nigeria and France** (VERIFIED_FROM_DATASET_DOCUMENTATION).
- Vehicle sensors (GPS, inertial, wheel-speed, powertrain/chassis) plus an
  Android smartphone's inertial sensors and GPS receiver.
- Headline totals claimed by the dataset README: ~40 h / 1,300 km vehicle,
  ~58 h / 4,400 km smartphone.

## Acquisition — read this before touching the data

The copy committed to this repository under `data/IO-VNBD_dataset/` is a ZIP
export of the upstream GitHub repo, so **every CSV and JPG in it is a Git LFS
pointer stub containing no data**. Parsing those as CSV yields nonsense.

Real content is fetched with `idr.dataset.acquire`, which resolves each
pointer against the upstream LFS endpoint and verifies the download against
the SHA-256 in the pointer (an LFS `oid` *is* the content hash):

```bash
python -c "from pathlib import Path; from idr.dataset.acquire import acquire_pointers; acquire_pointers(Path('data/IO-VNBD_dataset/IO-VNBD-master'), Path('data/raw/io_vnbd'))"
```

The pointer tree is **never overwritten** — it stays as the provenance record.
Content lands in `data/raw/io_vnbd/`, which is gitignored.

## Structure

```text
IO-VNBD-master/
├── README.md, README_1.pdf, Data Checker Table 2.py
├── Synchronised V abd S datasets/          (note upstream typo: "abd")
│   ├── Categorised IOVNB Dataset/<Label> (Driver X)/<Session>/{S,V}-<id>.csv
│   └── Uncategorised IOVNB Dataset/{S,V}-Dataset/
└── Unsynchronised V and S Dataset/
    ├── Categorised IOVNB (V) Dataset/V Dataset/
    └── Uncategorised IOVNB (V and S) Dataset/{S,V}-Dataset/
```

`S-` files are smartphone streams (24 columns); `V-` files are
vehicle/reference streams (29 columns).

## Findings that constrain how the data may be used

Each is measured, not assumed — see the forensic report for the numbers.

1. **Massive duplication.** The 564 CSVs contain only ~329 distinct contents;
   the same recordings appear in multiple trees. **Split on content hash, not
   file path**, or train/test leakage is guaranteed (Rule 9).
2. **The smartphone GNSS position is ~0.1 Hz, forward-filled.** Availability
   is 100%, but ~99% of rows repeat the previous position. The vehicle stream
   is the genuine ~10 Hz position reference.
3. **No GNSS outages exist.** Not one gap in a fix anywhere, and no outage
   annotation. Any outage scenario Phase 2 needs must be **synthesized**, and
   labelled as synthetic.
4. **Sampling rate is not uniformly 10 Hz.** The vehicle stream is; the
   smartphone stream also contains ~2 Hz, ~24 Hz and ~1000 Hz families.
5. **Mixed character encoding.** Smartphone headers combine raw cp1252 bytes
   with double-encoded UTF-8; strict UTF-8 reads fail.
6. **Two incompatible clocks**, with a recurring ~1 h smartphone↔vehicle
   offset consistent with local-time vs UTC (INFERRED — nothing documents it).
7. **`GPS SATELLITES IN RANGE` is not numeric** — values look like `"27 / 28"`.
8. **Some files concatenate several trips**, showing huge timestamp jumps
   inside an otherwise 10 Hz stream.

## Regenerating the analysis

```bash
python -m idr.dataset.inspect --dataset-path data/raw/io_vnbd \
    --output data/metadata/io_vnbd --report-output reports/io_vnbd --deep
```

Read-only: a regression test asserts a full deep scan leaves every input byte
unchanged.
