from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, Field, PostgresDsn, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AROGYAFLOW_", extra="ignore")

    app_name: str = "ArogyaFlow AI"
    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    database_url: PostgresDsn | None = None
    database_pool_min_size: int = Field(default=1, ge=1)
    database_pool_max_size: int = Field(default=5, ge=1)
    database_timeout_seconds: float = Field(default=10, gt=0)
    database_statement_timeout_ms: int = Field(default=5_000, gt=0)
    api_base_url: AnyHttpUrl = AnyHttpUrl("http://localhost:8000")
    api_timeout_seconds: float = Field(default=10, gt=0)
    api_key: SecretStr | None = None
    max_request_bytes: int = Field(default=1_048_576, ge=1_024)
    rate_limit_requests_per_minute: int = Field(default=120, ge=1)
    wait_model_path: Path | None = None
    arrival_model_path: Path | None = None
    no_show_model_path: Path | None = None
    occupancy_model_path: Path | None = None
    monitoring_report_path: Path | None = None
    wait_model_version: str | None = None
    arrival_model_version: str | None = None
    no_show_model_version: str | None = None
    occupancy_model_version: str | None = None

    @field_validator("api_key", mode="before")
    @classmethod
    def empty_api_key_is_unset(cls, value: object) -> object:
        return None if value == "" else value

    @model_validator(mode="after")
    def validate_pool_size(self) -> "Settings":
        if self.database_pool_max_size < self.database_pool_min_size:
            raise ValueError("database_pool_max_size must be >= database_pool_min_size")
        if self.environment == "production" and (
            self.api_key is None or not self.api_key.get_secret_value()
        ):
            raise ValueError("api_key is required in production")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
