"""Central configuration mechanism for the ``idr`` package.

Phase 0 establishes the mechanism only: a strongly typed config object,
YAML-file loading per environment, and environment-variable overrides.
Later phases add domain-specific sections (sensors, model paths, map
settings, ...) to :class:`AppConfig` — this module intentionally does not
define any navigation/algorithm parameters yet.
"""

from __future__ import annotations

from idr.config.loader import load_config
from idr.config.schema import AppConfig, LoggingConfig

__all__ = ["AppConfig", "LoggingConfig", "load_config"]
