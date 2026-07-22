# ArogyaFlow AI

[![CI](https://github.com/priyanshu141ai/ArogyaFlow-AI-/actions/workflows/ci.yml/badge.svg)](https://github.com/priyanshu141ai/ArogyaFlow-AI-/actions/workflows/ci.yml)

Hospital operations decision-support platform for forecasting demand, predicting
operational risks, and comparing capacity scenarios. It is an end-to-end Data
Science, Machine Learning, and MLOps portfolio project built with synthetic or
de-identified data.

> ArogyaFlow supports operational planning only. It does not diagnose patients,
> recommend treatment, or make autonomous hospital decisions.

## What it does

- Predicts outpatient waiting time with uncertainty intervals.
- Forecasts patient arrivals for 6, 12, and 24-hour horizons.
- Estimates calibrated no-show probability and reminder priority.
- Forecasts bed occupancy and capacity alerts.
- Compares staffing scenarios through discrete-event simulation.
- Validates and quarantines invalid data using strict contracts.
- Imports de-identified CSV/Parquet hospital extracts into versioned datasets.
- Compares and tunes multiple models using temporal validation.
- Produces SHAP feature-importance explanations.
- Detects data drift, quality failures, and model degradation.
- Gates automatic wait-time retraining and safely archives promoted models.
- Tracks experiments and model versions with MLflow.

## Architecture

```text
Synthetic or de-identified data
            |
   contracts + quarantine
            |
 temporal features and models ----> MLflow
            |
 prediction + simulation layer
            |
 FastAPI ---- PostgreSQL ---- Streamlit
            |
 monitoring ---- retraining gate
```

The application is a modular monolith: data, ML, simulation, serving, and UI
remain separate modules without premature microservices.

## Quick start

Prerequisites: Docker Desktop and
[uv](https://docs.astral.sh/uv/getting-started/installation/).

```powershell
uv sync --all-groups
uv run --group ml demo-stack
```

The first run creates local secrets, generates synthetic data, trains four models,
registers them in MLflow, creates a monitoring report, and starts the stack.

| Service | URL |
|---|---|
| Dashboard | http://localhost:18501 |
| API documentation | http://localhost:8000/docs |
| API readiness | http://localhost:8000/health/ready |
| MLflow | http://localhost:5000 |

Stop the stack:

```powershell
docker compose --env-file .arogyaflow-demo.env down
```

## Data workflow

Generate reproducible synthetic data:

```powershell
uv run generate-data --scenario normal_week --seed 31 `
  --start 2026-01-05T00:00:00+00:00 --days 10 --output data
```

Import a de-identified hospital extract:

```powershell
uv run ingest-external-data --source hospital-export `
  --dataset-name hospital-a --output data
```

The source directory must contain one CSV or Parquet file for each canonical
table: `departments`, `doctors`, `beds`, `staffing_rosters`, `appointments`,
`encounters`, `queue_events`, `admissions`, and `bed_events`. Direct identifiers
such as patient names, phone numbers, addresses, Aadhaar, or ABHA numbers are
rejected.

## Monitoring and retraining

Monitoring uses feature drift, missing/duplicate rates, and relative metric
degradation. Data-quality failures block retraining; drift or performance alerts
can trigger a candidate run.

```powershell
uv run --group ml retrain-wait-time `
  --bronze data/bronze/hospital-a/<dataset-hash> `
  --monitoring-report reports/monitoring/latest.json `
  --model-version wait-v2 --seed 11 --minimum-improvement 0.02 `
  --skip-tracking
```

A candidate is promoted only when it passes the configured improvement gate.
The previous production artifact and evaluation report are archived first.

## Technology

- Python, pandas, Pandera, scikit-learn, SHAP and SimPy
- FastAPI, Streamlit and PostgreSQL
- MLflow model tracking and registry
- Docker Compose, pytest, Ruff, mypy and GitHub Actions

## Quality checks

```powershell
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
```

The current suite contains 37 passing tests covering data generation, contracts,
model training, forecasting, simulation, monitoring, retraining, API security,
persistence, and end-to-end behavior.

## Project structure

```text
src/arogyaflow/data/   Data contracts, generation, ingestion and validation
src/arogyaflow/        Models, simulation, monitoring, API and dashboard
config/                Versioned demo configuration
tests/                 Unit, integration and end-to-end tests
docker-compose.yml     Local API, dashboard, PostgreSQL and MLflow stack
```

## Limitations

- Synthetic performance does not prove real hospital performance.
- External data must already be mapped to the canonical ArogyaFlow tables.
- Real deployment requires hospital-specific validation, governance, privacy
  review, and prospective workflow evaluation.
