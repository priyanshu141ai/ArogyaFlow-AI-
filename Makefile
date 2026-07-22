.PHONY: install format lint type-check test audit api dashboard demo migrate generate-data baseline-analysis train-wait-time train-arrivals train-no-show train-occupancy simulate monitor

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

audit:
	uv audit --locked

api:
	uv run uvicorn arogyaflow.api:app --reload

dashboard:
	uv run streamlit run src/arogyaflow/dashboard.py

demo:
	uv run --group ml demo-stack

migrate:
	uv run migrate-db

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

train-occupancy:
	uv run --group ml train-occupancy --bronze $(BRONZE) --model-version $(VERSION) --seed $(SEED) --alert-threshold $(THRESHOLD)

simulate:
	uv run simulate-flow --seed $(SEED) --duration $(DURATION) --arrivals-per-hour $(ARRIVALS) --doctors $(DOCTORS) --rooms $(ROOMS) --service-minutes $(SERVICE_MINUTES) --max-doctors $(MAX_DOCTORS) --max-rooms $(MAX_ROOMS) --minimum-improvement $(MIN_IMPROVEMENT)

monitor:
	uv run monitor-model --reference $(REFERENCE) --current $(CURRENT) --reference-metrics $(REFERENCE_METRICS) --current-metrics $(CURRENT_METRICS) --output $(OUTPUT) $(REQUIRED_COLUMNS)
