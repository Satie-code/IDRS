"""Typed configuration schema.

Kept intentionally minimal in Phase 0: only the settings the foundation
itself needs (which environment is active, how logging behaves). Future
phases extend :class:`AppConfig` with additional frozen dataclasses
(e.g. ``SensorConfig``, ``ModelConfig``) rather than loosening the typing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

VALID_ENVIRONMENTS = ("development", "testing", "production")


@dataclass(frozen=True)
class LoggingConfig:
    """Logging behaviour. See idr.logging for the consumer of this config."""

    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    environment: str = "development"
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self) -> None:
        if self.environment not in VALID_ENVIRONMENTS:
            raise ValueError(
                f"Unknown environment {self.environment!r}; expected one of {VALID_ENVIRONMENTS!r}"
            )
