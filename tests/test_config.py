from pathlib import Path

import pytest
from pydantic import ValidationError

from arogyaflow.config import Settings


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings()
    assert settings.environment == "development"
    assert settings.database_url is None


def test_settings_reject_invalid_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AROGYAFLOW_ENVIRONMENT", "invalid")
    with pytest.raises(ValidationError):
        Settings()


def test_settings_reject_invalid_pool_size(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValidationError):
        Settings(database_pool_min_size=3, database_pool_max_size=2)


def test_production_requires_api_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValidationError):
        Settings(environment="production")
