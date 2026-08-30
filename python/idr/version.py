"""Single source of truth for the ``idr`` package version.

Every other place that needs the project version (``pyproject.toml``,
docs, CLI ``--version`` flags, health checks) should read it from here
rather than maintaining a second copy.
"""

from __future__ import annotations

__version__ = "0.1.0"
