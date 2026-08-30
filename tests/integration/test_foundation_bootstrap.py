"""Proves the Phase 0 foundation initializes end to end with no errors:
package import -> config load -> logging configure -> a log call actually
reaching a handler. This is deliberately NOT a navigation/algorithm test —
see docs/development/testing.md for the test hierarchy this belongs to."""

from __future__ import annotations

import logging

import pytest

import idr
from idr.config import load_config
from idr.logging import configure_logging, get_logger


def test_foundation_bootstraps_without_errors(caplog: pytest.LogCaptureFixture) -> None:
    assert idr.__version__

    config = load_config(environment="testing")
    assert config.environment == "testing"

    configure_logging(config.logging, force=True)
    logger = get_logger("integration_test")

    with caplog.at_level(logging.INFO, logger="idr"):
        logger.info("foundation bootstrap probe")

    assert any("foundation bootstrap probe" in record.message for record in caplog.records)
