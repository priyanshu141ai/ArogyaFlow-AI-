.PHONY: install format lint type-check test api dashboard generate-data baseline-analysis train-wait-time train-arrivals train-no-show

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

generate-data:
	uv run generate-data --scenario $(SCENARIO) --seed $(SEED) --start $(START)

baseline-analysis:
	uv run baseline-analysis --bronze $(BRONZE)

train-wait-time:
	uv run --group ml train-wait-time --bronze $(BRONZE) --model-version $(VERSION) --seed $(SEED)

train-arrivals:
	uv run --group ml train-arrivals --bronze $(BRONZE) --model-version $(VERSION) --seed $(SEED)

train-no-show:
	uv run --group ml train-no-show --bronze $(BRONZE) --model-version $(VERSION) --seed $(SEED) --capacity-fraction $(CAPACITY) --reminder-effectiveness $(EFFECT)
