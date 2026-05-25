"""Tests for FastAPI endpoints.

The pipeline is mocked so no real artefacts are required — these tests focus
on HTTP routing, schema validation, and response structure.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from walmart_cash_forecast.api.app import create_app
from walmart_cash_forecast.config import Config


def _make_mock_pipeline(n_rows: int = 2):
    """Return a MagicMock that mimics PredictionPipeline.predict output."""
    mock = MagicMock()
    mock.predict.return_value = pd.DataFrame({
        "store_id": [f"STR_00{i+1}" for i in range(n_rows)],
        "date": pd.to_datetime(["2024-01-15"] * n_rows),
        "forecast_blend": np.full(n_rows, 50_000.0),
        "lower": np.full(n_rows, 40_000.0),
        "upper": np.full(n_rows, 60_000.0),
        "q_star": np.full(n_rows, 55_000.0),
    })
    return mock


@pytest.fixture()
def client():
    cfg = Config()
    app = create_app(cfg, model_dir="/fake/model_dir")

    mock_pipeline = _make_mock_pipeline()
    with patch(
        "walmart_cash_forecast.api.app.PredictionPipeline",
        return_value=mock_pipeline,
    ):
        with TestClient(app) as c:
            yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_endpoint_returns_predictions(client):
    payload = {
        "observations": [
            {
                "store_id": "STR_001",
                "date": "2024-01-15",
                "day_of_week": 0,
                "is_payday": True,
                "is_holiday": False,
                "is_buen_fin": False,
                "is_navidad_season": False,
            },
            {
                "store_id": "STR_002",
                "date": "2024-01-15",
                "day_of_week": 0,
            },
        ],
        "stores": [
            {"store_id": "STR_001", "region": "Norte", "store_format": "Supercenter"},
            {"store_id": "STR_002", "region": "Sur", "store_format": "Bodega"},
        ],
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2
    for pred in data["predictions"]:
        assert "forecast_blend" in pred
        assert "lower" in pred
        assert "upper" in pred
        assert "q_star" in pred
        assert pred["lower"] >= 0
        assert pred["upper"] >= pred["lower"]


def test_predict_schema_validation_rejects_bad_day(client):
    """day_of_week must be 0-6; 7 should fail Pydantic validation."""
    payload = {
        "observations": [
            {"store_id": "STR_001", "date": "2024-01-15", "day_of_week": 7},
        ],
        "stores": [{"store_id": "STR_001", "region": "Norte", "store_format": "Supercenter"}],
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422  # Pydantic validation error
