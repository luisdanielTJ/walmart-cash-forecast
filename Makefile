.PHONY: install train predict test lint type-check

install:
	uv sync --all-extras

train:
	uv run walmart-forecast train --data-dir data/raw --model-dir models/v1

predict:
	uv run walmart-forecast predict --model-dir models/v1 --future-csv data/future_march2024.csv --stores-csv data/raw/stores.csv --out data/predictions_march2024.csv

test:
	uv run pytest

lint:
	uv run ruff check src/ tests/

type-check:
	uv run mypy src/
