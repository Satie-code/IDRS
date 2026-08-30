from __future__ import annotations

from pathlib import Path

import pytest

from idr.config import AppConfig, LoggingConfig, load_config
from idr.config.schema import VALID_ENVIRONMENTS


def test_default_app_config_is_development() -> None:
    config = AppConfig()
    assert config.environment == "development"
    assert isinstance(config.logging, LoggingConfig)


def test_app_config_rejects_unknown_environment() -> None:
    with pytest.raises(ValueError, match="Unknown environment"):
        AppConfig(environment="not-a-real-environment")


def test_load_config_defaults_when_no_file_present(tmp_path: Path) -> None:
    config = load_config(environment="testing", config_dir=tmp_path)
    assert config.environment == "testing"
    assert config.logging.level == "INFO"


def test_load_config_reads_yaml_file(tmp_path: Path) -> None:
    (tmp_path / "development.yaml").write_text("logging:\n  level: DEBUG\n", encoding="utf-8")
    config = load_config(environment="development", config_dir=tmp_path)
    assert config.logging.level == "DEBUG"


def test_load_config_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IDR_LOGGING__LEVEL", "WARNING")
    config = load_config(environment="testing", config_dir=tmp_path)
    assert config.logging.level == "WARNING"


def test_repo_development_yaml_loads_cleanly() -> None:
    from idr.utils import find_repo_root

    config = load_config(environment="development", config_dir=find_repo_root() / "configs")
    assert config.environment in VALID_ENVIRONMENTS
