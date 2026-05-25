"""Integration tests for training and prediction pipelines.

These tests wire all components together without the Bayesian MCMC step
(already tested in test_bayesian.py) by mocking BayesianForecaster.fit/predict
to avoid slow sampling in CI.  Everything else (ML, conformal, blender,
newsvendor, ILP) runs for real.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from walmart_cash_forecast.config import Config, MLConfig, BayesianConfig
from walmart_cash_forecast.pipelines.training import TrainingPipeline
from walmart_cash_forecast.pipelines.prediction import PredictionPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic_csvs(tmp_path: Path, n_stores: int = 3, n_days: int = 90) -> Path:
    """Write minimal synthetic CSVs to a temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=n_days)
    store_ids = [f"STR_{i:03d}" for i in range(1, n_stores + 1)]
    categories = ["Abarrotes", "Electrónica"]

    rows = []
    for store in store_ids:
        for date in dates:
            for cat in categories:
                total = int(rng.integers(100, 1000))
                cash_tx = int(rng.integers(50, total))
                rows.append({
                    "store_id": store,
                    "date": date,
                    "category": cat,
                    "amount_cash": float(rng.uniform(5_000, 50_000)),
                    "amount_card": float(rng.uniform(5_000, 50_000)),
                    "amount_total": float(rng.uniform(10_000, 100_000)),
                    "cash_transactions": cash_tx,
                    "card_transactions": total - cash_tx,
                    "total_transactions": total,
                    "is_payday": date.day in (15, 28),
                    "has_promotion": bool(rng.integers(0, 2)),
                })
    pd.DataFrame(rows).to_csv(data_dir / "transactions.csv", index=False)

    pd.DataFrame({
        "store_id": store_ids,
        "region": ["Norte", "Sur", "Centro"][:n_stores],
        "store_format": ["Supercenter", "Bodega", "Express"][:n_stores],
        "socioeconomic_level": ["C", "D", "B"][:n_stores],
        "size_sqm": [8_000, 3_000, 1_200][:n_stores],
    }).to_csv(data_dir / "stores.csv", index=False)

    # Calendar: required columns per DataValidator._CALENDAR_COLS
    cal_rows = []
    for date in dates:
        cal_rows.append({
            "date": date,
            "day_of_week": date.dayofweek,
            "is_payday": date.day in (15, 28),
            "is_holiday": date.month == 1 and date.day == 1,
            "is_weekend": date.dayofweek >= 5,
            "is_buen_fin": date.month == 11 and 15 <= date.day <= 21,
            "is_navidad_season": date.month == 12 and date.day >= 15,
            "is_semana_santa": False,
        })
    pd.DataFrame(cal_rows).to_csv(data_dir / "calendar.csv", index=False)

    return data_dir


def _fake_bayes_predict(future_df: pd.DataFrame) -> np.ndarray:
    """Return random samples as if from a posterior predictive distribution."""
    rng = np.random.default_rng(42)
    n_samples = 200
    n_obs = len(future_df)
    return rng.uniform(10_000, 80_000, size=(n_samples, n_obs))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fast_config():
    cfg = Config()
    cfg.bayesian = BayesianConfig(n_chains=1, n_draws=100, n_tune=50)
    cfg.ml = MLConfig(n_optuna_trials=1)
    cfg.holdout_days = 20
    return cfg


def test_training_pipeline_creates_artefacts(fast_config, tmp_path):
    """Full training pipeline (Bayesian mocked) should write all artefacts."""
    data_dir = make_synthetic_csvs(tmp_path)
    model_dir = tmp_path / "models"

    with patch(
        "walmart_cash_forecast.pipelines.training.BayesianForecaster.fit",
        return_value=None,
    ), patch(
        "walmart_cash_forecast.pipelines.training.BayesianForecaster.save",
        return_value=None,
    ), patch(
        "walmart_cash_forecast.pipelines.training.BayesianForecaster.predict",
        side_effect=_fake_bayes_predict,
    ):
        pipeline = TrainingPipeline(fast_config, data_dir, model_dir)
        metadata = pipeline.run()

    assert metadata["train_rows"] > 0
    assert metadata["calib_rows"] > 0
    assert (model_dir / "ml").exists()
    assert (model_dir / "conformal").exists()
    assert (model_dir / "blender").exists()
    assert (model_dir / "training_metadata.json").exists()


def test_prediction_pipeline_produces_output(fast_config, tmp_path):
    """Prediction pipeline (Bayesian mocked) should return a tidy DataFrame."""
    data_dir = make_synthetic_csvs(tmp_path)
    model_dir = tmp_path / "models"

    # Train first
    with patch(
        "walmart_cash_forecast.pipelines.training.BayesianForecaster.fit",
        return_value=None,
    ), patch(
        "walmart_cash_forecast.pipelines.training.BayesianForecaster.save",
        return_value=None,
    ), patch(
        "walmart_cash_forecast.pipelines.training.BayesianForecaster.predict",
        side_effect=_fake_bayes_predict,
    ):
        TrainingPipeline(fast_config, data_dir, model_dir).run()

    # Create minimal future feature DataFrame (ML features only; Bayesian is mocked)
    from walmart_cash_forecast.data.loader import DataLoader
    from walmart_cash_forecast.features.aggregator import StoreAggregator
    from walmart_cash_forecast.features.imputer import CashImputer
    from walmart_cash_forecast.features.engineer import FeatureEngine

    transactions, stores, calendar = DataLoader(data_dir).load()
    panel = StoreAggregator().aggregate(transactions)
    panel["day_of_week"] = panel["date"].dt.dayofweek
    panel["is_payday"] = panel["date"].dt.day.isin([15, 28])
    panel = panel.merge(stores[["store_id", "region", "store_format"]], on="store_id", how="left")
    for col in ["is_holiday", "is_buen_fin", "is_navidad_season"]:
        panel[col] = False
    panel = CashImputer().fit(panel).transform(panel)
    panel = FeatureEngine(fast_config).fit_transform(panel)
    future_df = panel.dropna().tail(30).copy()

    with patch(
        "walmart_cash_forecast.pipelines.prediction.BayesianForecaster.load",
        return_value=None,
    ), patch(
        "walmart_cash_forecast.pipelines.prediction.BayesianForecaster.predict",
        side_effect=_fake_bayes_predict,
    ):
        pred_pipeline = PredictionPipeline(fast_config, model_dir)
        pred_pipeline.load()
        result = pred_pipeline.predict(future_df, stores)

    assert len(result) == len(future_df)
    assert "forecast_blend" in result.columns
    assert "lower" in result.columns
    assert "upper" in result.columns
    assert "q_star" in result.columns
    assert (result["lower"] >= 0).all()
    assert (result["upper"] >= result["lower"]).all()
