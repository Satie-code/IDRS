"""Project-wide logging utility, built on Python's standard ``logging`` module.

No custom logging framework — just a thin, consistent setup:
console output, levels, timestamps, and module names.
"""

from __future__ import annotations

from idr.logging.setup import configure_logging, get_logger

__all__ = ["configure_logging", "get_logger"]
