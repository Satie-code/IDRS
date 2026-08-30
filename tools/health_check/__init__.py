"""Foundation health check for the Intelligent Dead Reckoning project.

Run with:

    python -m tools.health_check

Exits with status 0 if every foundation component is healthy, non-zero
otherwise. See CheckResult / run_checks for programmatic use (e.g. from
tests/integration).
"""

from __future__ import annotations

from tools.health_check.checks import CheckResult, run_checks

__all__ = ["CheckResult", "run_checks"]
