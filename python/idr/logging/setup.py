"""Configure and retrieve loggers for the ``idr`` project."""

from __future__ import annotations

import logging
import sys

from idr.config.schema import LoggingConfig

ROOT_LOGGER_NAME = "idr"
_configured = False


def configure_logging(config: LoggingConfig | None = None, *, force: bool = False) -> None:
    """Configure the ``idr`` root logger with a console handler.

    Idempotent by default: calling this more than once is a no-op unless
    ``force=True``, so importing modules that call it doesn't duplicate
    handlers or clobber a caller's own logging setup.
    """
    global _configured
    if _configured and not force:
        return

    cfg = config or LoggingConfig()
    try:
        level = getattr(logging, cfg.level.upper())
    except AttributeError as exc:
        raise ValueError(f"Invalid log level: {cfg.level!r}") from exc

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=cfg.format, datefmt=cfg.datefmt))
    logger.addHandler(handler)
    # Deliberately leave propagate at its default (True): the "idr" logger
    # already emits once via the handler above, and leaving propagation on
    # means tools that attach to the root logger — pytest's caplog fixture,
    # in particular — still observe idr's log records.

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under the ``idr`` root logger.

    Callers normally pass ``__name__``. A leading ``idr.`` is stripped so an
    in-package module does not end up as ``idr.idr.dataset.foo``, and a module
    executed via ``python -m`` (where ``__name__`` is ``__main__``) is given a
    neutral name rather than logging as ``idr.__main__``.

    Calls :func:`configure_logging` with defaults if logging has not been
    configured yet, so callers get sane output without a separate setup step.
    """
    if not _configured:
        configure_logging()

    suffix = name
    if suffix == "__main__":
        suffix = "cli"
    elif suffix == ROOT_LOGGER_NAME:
        suffix = ""
    elif suffix.startswith(f"{ROOT_LOGGER_NAME}."):
        suffix = suffix[len(ROOT_LOGGER_NAME) + 1 :]

    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{suffix}" if suffix else ROOT_LOGGER_NAME)
