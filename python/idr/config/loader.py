"""Loads :class:`~idr.config.schema.AppConfig` from YAML + environment overrides.

Resolution order (later steps override earlier ones):

1. Dataclass defaults in ``idr.config.schema``.
2. ``configs/<environment>.yaml`` if it exists.
3. Environment variables prefixed with ``IDR_``, using ``__`` to separate
   nested keys, e.g. ``IDR_LOGGING__LEVEL=DEBUG`` overrides
   ``logging.level``.

The active environment itself is chosen by (in priority order): an explicit
``environment`` argument, the ``IDR_ENVIRONMENT`` environment variable, or
the ``development`` default.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from idr.config.schema import AppConfig, LoggingConfig
from idr.utils.paths import find_repo_root

ENV_PREFIX = "IDR_"
ENV_NESTING_SEPARATOR = "__"


def _default_config_dir() -> Path:
    return find_repo_root() / "configs"


def _load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at the top level of {path}, got {type(data)!r}")
    return data


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Apply ``IDR_SECTION__KEY=value`` environment variable overrides."""
    result = {k: (dict(v) if isinstance(v, dict) else v) for k, v in data.items()}
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(ENV_PREFIX) or env_key == f"{ENV_PREFIX}ENVIRONMENT":
            continue
        path = env_key[len(ENV_PREFIX) :].lower().split(ENV_NESTING_SEPARATOR)
        target = result
        for part in path[:-1]:
            target = target.setdefault(part, {})
        target[path[-1]] = env_value
    return result


def _build_config(environment: str, data: dict[str, Any]) -> AppConfig:
    logging_data = data.get("logging", {}) or {}
    logging_config = LoggingConfig(**logging_data)
    known_fields = {k: v for k, v in data.items() if k != "logging"}
    return AppConfig(environment=environment, logging=logging_config, **known_fields)


def load_config(
    environment: str | None = None,
    config_dir: Path | None = None,
) -> AppConfig:
    """Load the application configuration for a given environment.

    Raises FileNotFoundError only if the repo root cannot be located at all;
    a missing per-environment YAML file is not an error — dataclass defaults
    apply instead, so the health check has something sane to run against.
    """
    resolved_environment = environment or os.environ.get("IDR_ENVIRONMENT", "development")
    resolved_dir = config_dir or _default_config_dir()

    data = _load_yaml_file(resolved_dir / f"{resolved_environment}.yaml")
    data = _apply_env_overrides(data)
    return _build_config(resolved_environment, data)
