"""Tests for MLForecaster.

Uses a very small Optuna budget (1 trial) so tests run in seconds.
"""
import numpy as np
import pandas as pd
import pytest

from walmart_cash_forecast.config import Config, MLConfig
from walmart_cash_forecast.models.ml.model import MLForecaster


@pytest.fixture(scope="module")
def fast_config():
    cfg = Config()
    cfg.ml = MLConfig(n_optuna_trials=1)
    return cfg


def make_feature_panel(n_stores=3, n_days=60):
    """Minimal feature-engineered panel for ML model tests."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=n_days)
    rows = []
    for s in range(1, n_stores + 1):
        for date in dates:
            rows.append({
                "store_id": f"STR_{s:03d}",
                "date": date,
                "amount_cash": float(rng.uniform(10_000, 100_000)),
                "day_of_week": date.dayofweek,
                "is_payday": float(date.day in (15, 28)),
                "is_holiday": 0.0,
                "is_buen_fin": 0.0,
                "is_navidad_season": 0.0,
                "lag_1": float(rng.uniform(10_000, 100_000)),
                "lag_7": float(rng.uniform(10_000, 100_000)),
                "lag_14": float(rng.uniform(10_000, 100_000)),
                "roll_mean_7": float(rng.uniform(10_000, 100_000)),
                "roll_std_7": float(rng.uniform(1_000, 10_000)),
                "roll_mean_28": float(rng.uniform(10_000, 100_000)),
                "cash_ratio": float(rng.uniform(0.1, 0.9)),
                "days_since_payday": float(rng.integers(0, 15)),
                "days_until_payday": float(rng.integers(0, 15)),
            })
    return pd.DataFrame(rows)


def test_ml_forecaster_fits_and_predicts(fast_config):
    df = make_feature_panel()
    forecaster = MLForecaster(fast_config)
    forecaster.fit(df)

    preds = forecaster.predict(df)
    assert preds.shape == (len(df),)
    assert (preds >= 0).all()


def test_ml_forecaster_predict_interval(fast_config):
    df = make_feature_panel()
    forecaster = MLForecaster(fast_config)
    forecaster.fit(df)

    interval = forecaster.predict_interval(df)
    assert set(interval.columns) == {"q10", "q50", "q90"}
    assert len(interval) == len(df)
    # q10 ≤ q90 on average (not guaranteed row-by-row with separate models)
    assert interval["q10"].mean() <= interval["q90"].mean()


def test_ml_forecaster_save_load(fast_config, tmp_path):
    df = make_feature_panel()
    forecaster = MLForecaster(fast_config)
    forecaster.fit(df)
    forecaster.save(tmp_path / "ml_model")

    loaded = MLForecaster(fast_config)
    loaded.load(tmp_path / "ml_model")
    preds_orig = forecaster.predict(df)
    preds_load = loaded.predict(df)
    np.testing.assert_allclose(preds_orig, preds_load, rtol=1e-5)
