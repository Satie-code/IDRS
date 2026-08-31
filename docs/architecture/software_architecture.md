# Software Architecture — Repository Structure

Maps the layers in [`system_architecture.md`](system_architecture.md) onto
the actual repository layout. The skeleton was established in Phase 0; the
`dataset`, `pipeline` and `frames` packages were added by Phases 1–3.

```text
intelligent-dead-reckoning/
├── python/idr/         # ML / research layer (Python package, importable as `idr`)
│   ├── config/          # Typed configuration loading (YAML + env overrides)
│   ├── logging/          # Project-wide logging setup (stdlib logging)
│   ├── utils/            # Repo-relative path resolution, etc.
│   ├── dataset/          # Phase 1: IO-VNBD acquisition + forensic inspection
│   ├── pipeline/         # Phase 2: canonicalization + ML-ready pipeline
│   ├── frames/           # Phase 3: sensor frames, orientation, alignment
│   └── version.py        # Single source of truth for the project version
│
├── core/                # Navigation Core layer (C++ library)
│   ├── include/idr/       # Public headers
│   ├── src/                # Implementation
│   └── tests/               # CTest-driven unit tests
│
├── tools/health_check/  # Cross-cutting: verifies the foundation is intact
├── tests/               # Python unit + integration tests
├── configs/             # Environment configuration files
├── data/                # Dataset layout (raw data itself is gitignored)
├── models/              # Checkpoints / exported artifacts / metadata
├── docs/                # This documentation tree
└── .github/workflows/   # CI
```

## Package boundaries

- `python/idr` never imports from `core/` directly — the two communicate
  only through exported model artifacts (`models/exported/`) and, in later
  phases, a defined data interchange format. No FFI/bindings exist yet.
- `core/idr_core` has zero dependencies on `python/idr`, Android, or any
  UI layer (Rule 5). It links against nothing beyond the C++ standard
  library in Phase 0.
- `tools/` depends on `python/idr` (health check imports it) but nothing
  depends on `tools/` — it's a leaf, not a library other code builds on.

## Layering inside `python/idr`

The research packages form a one-way chain, and the direction is deliberate:

```text
dataset  (Phase 1)  →  pipeline  (Phase 2)  →  frames  (Phase 3)
```

- `pipeline` imports `dataset` for inventory and schema work; `dataset` knows
  nothing about `pipeline`.
- `frames` imports `pipeline` for the canonical schema and quality flags, and
  reads canonical Parquet through `idr.frames.sources`. It never reads a raw
  CSV and never re-implements the Phase 2 reader — that reader resolves
  encoding fallbacks, a column-shift repair, three clocks and a positional
  gyroscope mapping, and a second copy would drift from those decisions.
- Nothing imports `frames`. Phase 4 will.

Each phase writes only to its own outputs. `frames` treats both `data/raw/` and
`data/processed/` as read-only, and a regression test asserts that a full run
leaves every input byte-identical.

## Build systems

- Python: `pyproject.toml` + `setuptools`, package discovered under
  `python/` (see `[tool.setuptools.packages.find]`).
- C++: CMake, root `CMakeLists.txt` → `core/src` (library target
  `idr_core`) → `core/tests` (test executable, registered with CTest).

The two build systems are independent in Phase 0 — there is no Python
extension module wrapping the C++ core yet. That binding is future work,
not established here.
