FROM python:3.11-slim

WORKDIR /app

# Install uv for fast, reproducible dependency management
RUN pip install uv

COPY pyproject.toml .
COPY src/ src/
COPY config.yaml .
COPY data/ data/
COPY artifacts/ artifacts/

# Install production dependencies only (no dev extras)
RUN uv sync --no-dev

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "walmart_cash_forecast.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
