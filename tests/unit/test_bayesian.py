"""Tests for BayesianForecaster.

Uses minimal MCMC settings (1 chain, 100 draws) to keep test execution fast
while still exercising the full model build, sampling, and prediction code paths.
"""
import numpy as np
import pandas as pd
import pytest

from walmart_cash_forecast.config import Config, BayesianConfig
from walmart_cash_forecast.models.bayesian.model import BayesianForecaster


@pytest.fixture(scope="module")
def fast_config():
    """Minimal MCMC config for fast testing — not for production use."""
    cfg = Config()
    cfg.bayesian = BayesianConfig(n_chains=1, n_draws=100, n_tune=100)
    return cfg


def make_mini_panel(n_stores=3, n_days=40):
    """Small panel: 3 stores × 40 days for fast MCMC sampling."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=n_days)
    store_ids = [f"STR_{i:03d}" for i in range(1, n_stores + 1)]
    regions = ["Norte", "Sur", "Centro"]
    formats = ["Supercenter", "Bodega", "Express"]
    rows = []
    for i, store in enumerate(store_ids):
        for date in dates:
            rows.append({
                "store_id": store,
                "date": date,
                "amount_cash": float(rng.uniform(10_000, 100_000)),
                "is_payday": date.day in (15, 28),
                "is_holiday": False,
                "is_buen_fin": False,
                "is_navidad_season": False,
                "day_of_week": date.dayofweek,
                "region": regions[i % len(regions)],
                "store_format": formats[i % len(formats)],
            })
    panel = pd.DataFrame(rows)
    stores_df = pd.DataFrame({
        "store_id": store_ids,
        "region": regions,
        "store_format": formats,
    })
    return panel, stores_df


def test_bayesian_forecaster_fits_and_predicts(fast_config):
    panel, stores_df = make_mini_panel()
    forecaster = BayesianForecaster(fast_config)
    forecaster.fit(panel, stores_df)

    # Predict for 7 future days across all stores
    future_dates = pd.date_range("2023-02-10", periods=7)
    future_rows = []
    for store in panel["store_id"].unique():
        for date in future_dates:
            future_rows.append({
                "store_id": store,
                "date": date,
                "is_payday": date.day in (15, 28),
                "is_holiday": False,
                "is_buen_fin": False,
                "is_navidad_season": False,
                "day_of_week": date.dayofweek,
            })
    future_df = pd.DataFrame(future_rows)
    samples = forecaster.predict(future_df)

    # Output shape: (n_posterior_samples, n_predictions)
    assert samples.ndim == 2
    assert samples.shape[1] == len(future_df)
    # All cash amounts must be positive (expm1 of any real number ≥ -1)
    assert (samples >= 0).all()


def test_bayesian_forecaster_save_load(fast_config, tmp_path):
    panel, stores_df = make_mini_panel()
    forecaster = BayesianForecaster(fast_config)
    forecaster.fit(panel, stores_df)
    forecaster.save(tmp_path / "bayesian_model")

    loaded = BayesianForecaster(fast_config)
    loaded.load(tmp_path / "bayesian_model")
    assert loaded._trace is not None
    assert len(loaded._store_to_idx) == 3
