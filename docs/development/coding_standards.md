# Coding Standards

## Python

- Format/lint with **Ruff** (`pyproject.toml` → `[tool.ruff]`); it also acts
  as the formatter (`ruff format`). Don't hand-format against its rules.
- Type-check with **mypy**. New code under `python/idr/` must be fully
  typed (`disallow_untyped_defs = true`); test code is exempt.
- Naming: `snake_case` for functions/variables/modules, `PascalCase` for
  classes, `UPPER_SNAKE_CASE` for module-level constants.
- Prefer `dataclasses` for plain data containers (see `idr.config.schema`)
  over hand-written `__init__`/getters.
- `from __future__ import annotations` at the top of every module (keeps
  type hints cheap and forward-reference-friendly on Python 3.10+).
- Docstrings: module- and public-API-level only, and only when they explain
  *why*, not *what* — see the project-wide comment policy below.

## C++

- Standard: C++17 (see root `CMakeLists.txt`).
- Naming: `snake_case` for functions/variables/files, `PascalCase` for
  types/classes, `kPascalCase` for constants (see `idr::kVersion`).
- Headers use `#pragma once` (supported by every compiler this project
  targets — MSVC, GCC, Clang).
- Public API lives under `core/include/idr/`; implementation under
  `core/src/`. A `.cpp` file's matching header is included first.
- No exceptions-as-control-flow in hot navigation paths once those exist;
  Phase 0 code has no hot paths yet, so this is a forward note, not a
  current constraint.

## Comments and documentation

- Default to **no comments**. Only add one when the *why* is genuinely
  non-obvious — a hidden constraint, a workaround, a subtle invariant.
  Never restate what well-named code already says.
- Don't reference the current task/ticket/PR in comments; they rot. Put
  that context in the commit message instead.

## Error handling

- Fail loudly and specifically. No bare `except Exception: pass`.
- If you must catch broadly, either re-raise with added context or return a
  typed failure result (see `tools/health_check/checks.py::CheckResult` for
  the pattern used in this repo: a check function never raises, it returns
  `ok=False` with a `detail` message).
- C++: prefer returning/propagating errors over throwing in library code
  that later needs to run predictably on embedded/mobile targets; Phase 0's
  `idr_core` has no fallible operations yet, so this is establishing intent,
  not something exercised today.

## Logging

- Python: use `idr.logging.get_logger(__name__)` — never `print()` for
  anything other than a CLI's user-facing output (health check reports,
  `--help` text). Never hand-roll a second logging setup.
- Log levels: `DEBUG` for internal detail, `INFO` for normal lifecycle
  events, `WARNING` for recoverable anomalies, `ERROR` for failures that
  abort the current operation.

## See also

- [`docs/development/engineering_rules.md`](engineering_rules.md) — the
  10 project-wide rules that apply to *all* phases, not just style.
- [`docs/development/testing.md`](testing.md)
