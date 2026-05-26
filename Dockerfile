FROM python:3.11-slim

WORKDIR /app

# Install uv for fast, reproducible dependency management
RUN pip install uv

COPY pyproject.toml .
COPY uv.lock .
COPY src/ src/
COPY config.yaml .
COPY data/ data/

# Install production dependencies only (no dev extras)
RUN uv sync --no-dev

EXPOSE 8000
CMD ["uv", "run", "walmart-forecast", "serve", "--model-dir", "/app/models/v1", "--port", "8000"]
