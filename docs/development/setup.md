# Development Setup

All commands below were run and verified during Phase 0 on the environment
documented in [`environment.md`](environment.md). Run from the repository
root unless noted otherwise.

## 1. Python environment

```bash
python -m venv .venv
# Git Bash / macOS / Linux:
source .venv/Scripts/activate    # Git Bash on Windows
# source .venv/bin/activate      # macOS / Linux
# PowerShell:
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[dev]"
```

This installs the `idr` package in editable mode plus dev tooling
(pytest, pytest-cov, ruff, mypy, pre-commit), and makes `import idr` work
from anywhere without manual `PYTHONPATH` changes.

## 2. Run Python tests

```bash
pytest
```

With coverage:

```bash
pytest --cov
```

## 3. Lint

```bash
ruff check .
ruff format --check .
```

## 4. Type check

```bash
mypy
```

## 5. Python health check

```bash
python -m tools.health_check
```

Exits `0` and prints `RESULT: PASS` when the foundation is healthy; exits
non-zero on any failed check.

## 6. C++ build

```bash
cmake -S . -B build
cmake --build build --config Debug
```

`--config Debug` is required on this machine because the default CMake
generator here (Visual Studio) is multi-config — the build type isn't fixed
at configure time the way it is with Makefiles/Ninja.

## 7. C++ tests

```bash
ctest --test-dir build --output-on-failure -C Debug
```

## 8. Pre-commit hooks (optional but recommended)

```bash
pre-commit install
pre-commit run --all-files
```

## One-shot local CI equivalent

```bash
pytest --cov && ruff check . && ruff format --check . && mypy \
  && python -m tools.health_check \
  && cmake -S . -B build && cmake --build build --config Debug \
  && ctest --test-dir build --output-on-failure -C Debug
```
