"""Individual health checks. Each check is self-contained and never raises —
failures are captured as a failed :class:`CheckResult` so one broken check
cannot prevent the others from reporting, and so the caller decides how to
present/aggregate failures rather than the checks swallowing errors."""

from __future__ import annotations

import platform
import sys
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def _check_python_version() -> CheckResult:
    required = (3, 10)
    actual = sys.version_info[:2]
    if actual < required:
        return CheckResult(
            "python_version",
            False,
            f"Python {actual[0]}.{actual[1]} < required {required[0]}.{required[1]}",
        )
    return CheckResult(
        "python_version",
        True,
        f"Python {platform.python_version()} ({sys.executable})",
    )


def _check_package_import() -> CheckResult:
    try:
        import idr
    except ImportError as exc:
        return CheckResult("package_import", False, f"failed to import idr: {exc}")
    return CheckResult("package_import", True, f"idr package version {idr.__version__}")


def _check_repo_root() -> CheckResult:
    try:
        from idr.utils import find_repo_root

        root = find_repo_root()
    except FileNotFoundError as exc:
        return CheckResult("repo_root", False, str(exc))
    if not (root / "pyproject.toml").exists():
        return CheckResult("repo_root", False, f"resolved {root} but pyproject.toml is missing")
    return CheckResult("repo_root", True, f"repo root resolved to {root}")


def _check_configuration() -> CheckResult:
    try:
        from idr.config import load_config

        config = load_config(environment="development")
    except Exception as exc:  # noqa: BLE001 — surfaced as a failed check, not swallowed
        return CheckResult("configuration", False, f"failed to load config: {exc}")
    if config.environment != "development":
        return CheckResult(
            "configuration", False, f"expected environment=development, got {config.environment}"
        )
    return CheckResult(
        "configuration",
        True,
        f"loaded config for environment={config.environment!r}, "
        f"logging.level={config.logging.level!r}",
    )


def _check_logging() -> CheckResult:
    try:
        from idr.logging import configure_logging, get_logger
        from idr.logging.setup import ROOT_LOGGER_NAME

        configure_logging(force=True)
        logger = get_logger("health_check")
        logger.info("idr logging health check probe")
    except Exception as exc:  # noqa: BLE001 — surfaced as a failed check, not swallowed
        return CheckResult("logging", False, f"failed to configure logging: {exc}")

    import logging as stdlib_logging

    root_logger = stdlib_logging.getLogger(ROOT_LOGGER_NAME)
    if not root_logger.handlers:
        return CheckResult("logging", False, "idr root logger has no handlers after configure")
    return CheckResult(
        "logging",
        True,
        f"idr logger configured at level {stdlib_logging.getLevelName(root_logger.level)}",
    )


_CHECKS: tuple[Callable[[], CheckResult], ...] = (
    _check_python_version,
    _check_package_import,
    _check_repo_root,
    _check_configuration,
    _check_logging,
)


def run_checks() -> list[CheckResult]:
    return [check() for check in _CHECKS]
