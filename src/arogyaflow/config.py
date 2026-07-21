from functools import lru_cache
from typing import Literal

from pydantic import PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AROGYAFLOW_", extra="ignore")

    app_name: str = "ArogyaFlow AI"
    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    database_url: PostgresDsn | None = None
    wait_model_version: str | None = None
    arrival_model_version: str | None = None
    no_show_model_version: str | None = None
    occupancy_model_version: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
