# Testing Strategy

## Test hierarchy

```text
tests/
├── python/       # Unit tests for the idr Python package and tools/.
│   └── dataset/  #   Dataset inspection layer, on synthetic fixtures only.
├── integration/  # Cross-module tests: does the foundation initialize
│                 # end-to-end with no errors? No mocking across the
│                 # boundary being tested.
└── fixtures/     # Small, hand-crafted shared fixture files (not real
                  # sensor data — see data/README.md).

core/tests/       # C++ unit tests for the idr_core library, run via CTest.
```

## Dataset tests never touch the real dataset

`tests/python/dataset/` builds every input it needs in `tmp_path` from
`conftest.py` fixtures — a 20-row smartphone CSV, a vehicle CSV offset by one
hour, a file with a controlled GNSS gap, an LFS pointer stub. The suite runs
in about a second and works on a clone with no data downloaded.

This is deliberate: a test that needs 1.8 GB of external data is a test that
stops being run. It also means assertions can be *exact* — a fixture with a
known 1.0 s gap lets the test assert the measured duration is 1.0 s, which
real data could never support.

One test (`test_inspection_does_not_modify_the_dataset`) hashes every input
file before and after a full deep scan, enforcing the read-only guarantee that
Rule 1 depends on.

## Unit tests (`tests/python/`, `core/tests/`)

- One behavior per test; test names describe the behavior
  (`test_load_config_env_override`, not `test_config_2`).
- Python: pytest, no test classes unless grouping genuinely helps —
  free functions are the default in this repo.
- C++: Phase 0 uses a minimal hand-rolled assertion harness
  (`core/tests/test_health_check.cpp`) with no external framework
  dependency. Adopt a real framework (Catch2/GoogleTest) only once there is
  enough real navigation logic to justify the dependency — don't add it
  speculatively.

## Integration tests (`tests/integration/`)

Prove that independently-tested pieces work *together*: package import →
config load → logging configured → a log record actually reaches a
handler. See `tests/integration/test_foundation_bootstrap.py`.

Do not put algorithm/navigation correctness tests here — those get their
own layer once navigation code exists.

## Regression testing philosophy (for later phases)

This section documents intent for phases that don't exist yet — nothing
here is implemented in Phase 0.

- Every algorithmic change lands with a test that would have caught the bug
  or would fail without the change (Rule 6 in
  [`engineering_rules.md`](engineering_rules.md)).
- Navigation accuracy regressions are caught by replaying fixed, versioned
  reference sequences and comparing against previously recorded baselines —
  not by eyeballing plots.
- Benchmarks (`benchmarks/`) are reproducible: pinned inputs, recorded
  environment, no "trust me" numbers (Rule 7 and Rule 8).

## Running tests

```bash
pytest                              # Python: unit + integration
pytest --cov                        # with coverage (see pyproject.toml)
ctest --test-dir build --output-on-failure -C Debug   # C++
```
