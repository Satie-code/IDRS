# Development Environment (as discovered during Phase 0)

Snapshot taken 2026-08-30 on the primary development machine. This is a
record of what was verified to work, not a requirement spec — later
contributors on other machines should re-verify rather than assume this
file is still accurate for their setup.

## OS / Architecture

- OS: Microsoft Windows 11 Home Single Language, version 10.0.26200 (build 26200)
- Architecture: x86_64
- Shells available: PowerShell 7 (pwsh) and Git Bash / MSYS2 (`MINGW64_NT-10.0-26200`)

## Python

- `python` → Python 3.12.0 (`C:\Users\<user>\AppData\Local\Programs\Python\Python312\python.exe`)
- `pip` → 23.2.1 (tied to the 3.12 install)
- **`requires-python` was raised to `>=3.12` in Phase 1.** The dataset layer
  uses `enum.StrEnum` and `datetime.UTC` (3.11+), and numpy's bundled type
  stubs need 3.12 syntax for mypy to parse them at all. Ruff's `target-version`
  and mypy's `python_version` were aligned to `py312`/`3.12` to match.
- Python 3.11 and 3.14 installations are also present on PATH but are **not**
  the default `python` — always invoke via `python`/`.venv` activation, never
  a hardcoded version-specific path.
- `python3` is NOT usable directly in Git Bash on this machine — it resolves
  to the Windows Store app-execution alias stub and fails. Use `python`.

## Git

- `git version 2.55.0.windows.3`

## C++ toolchain

- **CMake**: 4.3.1. Default generator on this machine: `Visual Studio 18 2026`.
- **Primary compiler**: MSVC from Visual Studio 18 Community
  (`C:\Program Files\Microsoft Visual Studio\18\Community`, VC toolset
  14.50.35717), invoked through CMake's Visual Studio generator. This is
  the toolchain Phase 0 was built and verified against.
- **Legacy toolchain present but unusable**: `gcc`/`g++` on PATH resolve to
  a very old `MinGW.org GCC-6.3.0-1` (2016) install. It does not ship a
  working `<optional>` header and cannot compile the C++17 test program used
  during environment verification. `mingw32-make` is present but plain
  `make` is not. **Do not rely on this toolchain** — if a MinGW/GCC build is
  needed later, install a current MinGW-w64 (e.g. via MSYS2) rather than
  fixing this one.

## Java / Android tooling

- `java -version` → OpenJDK 25.0.2 (`java.exe` under
  `Common Files\Oracle\Java\javapath` and under `Program Files\Java\jdk-21.0.10`).
- `JAVA_HOME` is currently set to `C:\Program Files\Java\jdk-21`, which does
  **not exist** (the actual install is `jdk-21.0.10`). This breaks `gradle
  --version`. Not fixed in Phase 0 — Android tooling is out of scope until
  the Android phase, but note it here so it isn't re-discovered from scratch.
- `ANDROID_HOME` is set (`...\AppData\Local\Android\Sdk\`) but `adb` is not
  on PATH; SDK contents were not inspected (out of scope for Phase 0).

## Phase 1 runtime dependencies

Added in Phase 1 for dataset forensics (no ML or navigation libraries):
`numpy`, `pandas`, `matplotlib`, plus `pandas-stubs` for type checking.
Versions verified at the time of writing: numpy 2.5.2, pandas 3.0.5,
matplotlib 3.11.1, mypy 2.3.1, ruff 0.16.5, pytest 9.1.1.

## GPU

- NVIDIA GeForce RTX 4050 Laptop GPU (driver 32.0.16.1088)
- AMD Radeon 740M Graphics (integrated, driver 32.0.13058.2)
- Not used by anything in Phase 0. Recorded for later phases (on-device /
  training acceleration) that may care whether CUDA is available.

## What this means for Phase 0 build commands

- Python commands: use `python` (via the project virtualenv), not `python3`.
- C++ commands: `cmake -S . -B build` uses the Visual Studio generator by
  default and needs no extra flags on this machine. `cmake --build build`
  and `ctest --test-dir build` both need `--config Debug` (or `-C Debug`)
  because the Visual Studio generator is multi-config. See
  `docs/development/setup.md` for exact commands.
