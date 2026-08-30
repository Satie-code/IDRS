from __future__ import annotations

from tools.health_check import run_checks


def test_all_health_checks_pass_on_a_correctly_configured_repo() -> None:
    results = run_checks()
    failed = [r for r in results if not r.ok]
    assert not failed, f"health check(s) failed: {failed}"


def test_health_check_covers_expected_areas() -> None:
    results = run_checks()
    names = {r.name for r in results}
    assert names == {
        "python_version",
        "package_import",
        "repo_root",
        "configuration",
        "logging",
    }
