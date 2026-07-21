.PHONY: install format lint type-check test api dashboard

install:
	uv sync --all-groups

format:
	uv run ruff format .

lint:
	uv run ruff check .

type-check:
	uv run mypy

test:
	uv run pytest

api:
	uv run uvicorn arogyaflow.api:app --reload

dashboard:
	uv run streamlit run src/arogyaflow/dashboard.py

