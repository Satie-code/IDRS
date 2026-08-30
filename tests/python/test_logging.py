from __future__ import annotations

import logging

from idr.config.schema import LoggingConfig
from idr.logging import configure_logging, get_logger
from idr.logging.setup import ROOT_LOGGER_NAME


def test_configure_logging_attaches_console_handler() -> None:
    configure_logging(force=True)
    root_logger = logging.getLogger(ROOT_LOGGER_NAME)
    assert root_logger.handlers
    assert isinstance(root_logger.handlers[0], logging.StreamHandler)


def test_configure_logging_is_idempotent_without_force() -> None:
    configure_logging(force=True)
    root_logger = logging.getLogger(ROOT_LOGGER_NAME)
    handler_count_before = len(root_logger.handlers)
    configure_logging()
    assert len(root_logger.handlers) == handler_count_before


def test_configure_logging_respects_level() -> None:
    configure_logging(LoggingConfig(level="DEBUG"), force=True)
    root_logger = logging.getLogger(ROOT_LOGGER_NAME)
    assert root_logger.level == logging.DEBUG


def test_get_logger_is_namespaced_under_idr() -> None:
    logger = get_logger("some_module")
    assert logger.name == "idr.some_module"
