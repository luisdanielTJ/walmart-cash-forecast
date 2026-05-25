.PHONY: install train predict test lint type-check

install:
	uv sync --all-extras

train:
	uv run walmart-forecast train

predict:
	uv run walmart-forecast predict --days 7

test:
	uv run pytest

lint:
	uv run ruff check src/ tests/

type-check:
	uv run mypy src/
