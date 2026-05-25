"""Tests for FeatureEngine."""
import numpy as np
import pandas as pd

from walmart_cash_forecast.config import Config
from walmart_cash_forecast.features.engineer import FeatureEngine


def make_store_panel(n_stores=3, n_days=60):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=n_days)
    stores = [f"STR_{i:03d}" for i in range(1, n_stores + 1)]
    rows = []
    for store in stores:
        for date in dates:
            total = int(rng.integers(200, 2000))
            cash = int(rng.integers(50, total))  # cash is always a subset of total
            rows.append({
                "store_id": store, "date": date,
                "amount_cash": rng.uniform(10_000, 200_000),
                "cash_transactions": cash,
                "total_transactions": total,
                "amount_total": rng.uniform(50_000, 500_000),
                "is_payday": date.day in (15, 28, 29, 30, 31),
                "is_weekend": date.dayofweek >= 5,
                "is_holiday": False,
                "is_buen_fin": False,
                "is_navidad_season": False,
                "is_semana_santa": False,
                "day_of_week": date.dayofweek,
                "week_of_year": date.isocalendar()[1],
                "month": date.month,
                "store_format": "Supercenter",
                "region": "Norte",
                "socioeconomic_level": "B",
                "size_sqm": 10000,
                "num_checkouts": 20,
                "has_pharmacy": True,
                "has_fuel_station": False,
            })
    return pd.DataFrame(rows)


def test_feature_engine_adds_lag_columns():
    panel = make_store_panel()
    result = FeatureEngine(Config()).fit_transform(panel)
    assert "amount_cash_lag_7" in result.columns
    assert "amount_cash_lag_1" in result.columns


def test_feature_engine_adds_rolling_columns():
    panel = make_store_panel()
    result = FeatureEngine(Config()).fit_transform(panel)
    assert "amount_cash_roll7_mean" in result.columns
    assert "amount_cash_roll28_std" in result.columns


def test_feature_engine_adds_cash_ratio():
    panel = make_store_panel()
    result = FeatureEngine(Config()).fit_transform(panel)
    assert "cash_ratio" in result.columns
    assert (result["cash_ratio"].dropna() <= 1.0).all()


def test_feature_engine_no_future_leakage():
    """Lag-1 value on row i must equal amount_cash on row i-1 within same store."""
    panel = make_store_panel(n_stores=1, n_days=30)
    result = FeatureEngine(Config()).fit_transform(panel)
    store_result = result.sort_values("date").reset_index(drop=True)
    expected_lag1 = store_result["amount_cash"].iloc[0]
    actual_lag1 = store_result["amount_cash_lag_1"].iloc[1]
    assert abs(expected_lag1 - actual_lag1) < 1e-6


def test_feature_engine_requires_fit_before_transform():
    import pytest
    panel = make_store_panel(n_stores=1, n_days=10)
    fe = FeatureEngine(Config())
    with pytest.raises(RuntimeError, match="fit"):
        fe.transform(panel)
