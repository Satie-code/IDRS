"""CLI entry point: ``python -m tools.health_check``."""

from __future__ import annotations

import sys

from tools.health_check.checks import run_checks


def main() -> int:
    results = run_checks()

    print("IDR Phase 0 Health Check")
    print("=" * 40)
    all_ok = True
    for result in results:
        status = "OK  " if result.ok else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")
        all_ok = all_ok and result.ok
    print("=" * 40)
    print("RESULT: PASS" if all_ok else "RESULT: FAIL")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
