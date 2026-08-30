# Software Architecture — Repository Structure

Maps the layers in [`system_architecture.md`](system_architecture.md) onto
the actual repository layout established in Phase 0.

```text
intelligent-dead-reckoning/
├── python/idr/         # ML / research layer (Python package, importable as `idr`)
│   ├── config/          # Typed configuration loading (YAML + env overrides)
│   ├── logging/          # Project-wide logging setup (stdlib logging)
│   ├── utils/            # Repo-relative path resolution, etc.
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

## Build systems

- Python: `pyproject.toml` + `setuptools`, package discovered under
  `python/` (see `[tool.setuptools.packages.find]`).
- C++: CMake, root `CMakeLists.txt` → `core/src` (library target
  `idr_core`) → `core/tests` (test executable, registered with CTest).

The two build systems are independent in Phase 0 — there is no Python
extension module wrapping the C++ core yet. That binding is future work,
not established here.
