import argparse
from datetime import datetime
from enum import StrEnum
from importlib import resources
from typing import Any, Protocol

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, ConfigDict

from arogyaflow.config import Settings, get_settings
from arogyaflow.exceptions import ConfigurationError, PersistenceError


class PredictionType(StrEnum):
    WAIT_TIME = "wait_time"
    ARRIVALS = "arrivals"
    NO_SHOW = "no_show"
    OCCUPANCY = "occupancy"


class PredictionRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    request_id: str
    prediction_type: PredictionType
    model_version: str
    schema_version: str
    response_payload: dict[str, Any]
    created_at: datetime


class PredictionStore(Protocol):
    def insert_prediction(
        self,
        *,
        request_id: str,
        prediction_type: PredictionType,
        model_version: str,
        schema_version: str,
        request_payload: dict[str, Any],
        response_payload: dict[str, Any],
        created_at: datetime,
    ) -> PredictionRecord: ...

    def recent_predictions(self, limit: int) -> list[PredictionRecord]: ...


class PostgresDatabase:
    def __init__(self, settings: Settings) -> None:
        if settings.database_url is None:
            raise ConfigurationError("database_url is required")
        self._timeout = settings.database_timeout_seconds
        self._pool = ConnectionPool(
            conninfo=str(settings.database_url),
            min_size=settings.database_pool_min_size,
            max_size=settings.database_pool_max_size,
            timeout=settings.database_timeout_seconds,
            kwargs={
                "row_factory": dict_row,
                "autocommit": True,
                "options": f"-c statement_timeout={settings.database_statement_timeout_ms}",
            },
            open=False,
        )

    def open(self) -> None:
        try:
            self._pool.open(wait=True, timeout=self._timeout)
            self.apply_migrations()
        except (psycopg.Error, OSError) as exc:
            self._pool.close()
            raise PersistenceError("PostgreSQL startup failed") from exc

    def close(self) -> None:
        self._pool.close()

    def apply_migrations(self) -> None:
        migration_root = resources.files("arogyaflow.migrations")
        migrations = sorted(
            (path for path in migration_root.iterdir() if path.name.endswith(".sql")),
            key=lambda path: path.name,
        )
        try:
            with self._pool.connection() as connection:
                connection.execute(
                    """
                    create table if not exists schema_migrations (
                        version text primary key,
                        applied_at timestamptz not null default now()
                    )
                    """
                )
                for migration in migrations:
                    exists = connection.execute(
                        "select 1 from schema_migrations where version = %s",
                        (migration.name,),
                    ).fetchone()
                    if exists:
                        continue
                    with connection.transaction():
                        connection.execute(migration.read_text(encoding="utf-8"))
                        connection.execute(
                            "insert into schema_migrations (version) values (%s)",
                            (migration.name,),
                        )
        except (psycopg.Error, OSError) as exc:
            raise PersistenceError("PostgreSQL migration failed") from exc

    def ping(self) -> None:
        try:
            with self._pool.connection() as connection:
                connection.execute("select 1").fetchone()
        except psycopg.Error as exc:
            raise PersistenceError("PostgreSQL readiness check failed") from exc

    def insert_prediction(
        self,
        *,
        request_id: str,
        prediction_type: PredictionType,
        model_version: str,
        schema_version: str,
        request_payload: dict[str, Any],
        response_payload: dict[str, Any],
        created_at: datetime,
    ) -> PredictionRecord:
        try:
            with self._pool.connection() as connection:
                row = connection.execute(
                    """
                    insert into prediction_records (
                        request_id, prediction_type, model_version, schema_version,
                        request_payload, response_payload, created_at
                    ) values (%s, %s, %s, %s, %s, %s, %s)
                    returning id, request_id, prediction_type, model_version,
                              schema_version, response_payload, created_at
                    """,
                    (
                        request_id,
                        prediction_type.value,
                        model_version,
                        schema_version,
                        Jsonb(request_payload),
                        Jsonb(response_payload),
                        created_at,
                    ),
                ).fetchone()
        except psycopg.Error as exc:
            raise PersistenceError("Prediction persistence failed") from exc
        if row is None:
            raise PersistenceError("Prediction persistence returned no record")
        return PredictionRecord.model_validate(row)

    def recent_predictions(self, limit: int) -> list[PredictionRecord]:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        try:
            with self._pool.connection() as connection:
                rows = connection.execute(
                    """
                    select id, request_id, prediction_type, model_version,
                           schema_version, response_payload, created_at
                    from prediction_records
                    order by created_at desc, id desc
                    limit %s
                    """,
                    (limit,),
                ).fetchall()
        except psycopg.Error as exc:
            raise PersistenceError("Recent predictions query failed") from exc
        return [PredictionRecord.model_validate(row) for row in rows]


def main() -> None:
    argparse.ArgumentParser().parse_args()
    database = PostgresDatabase(get_settings())
    database.open()
    database.close()


if __name__ == "__main__":
    main()
