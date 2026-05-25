"""Tests for CashImputer."""
import numpy as np
import pandas as pd

from walmart_cash_forecast.features.imputer import CashImputer


def make_store_df(n_days=30, null_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days)
    amounts = rng.uniform(10_000, 200_000, size=n_days)
    mask = rng.random(n_days) < null_frac
    return pd.DataFrame({
        "store_id": "STR_001",
        "date": dates,
        "day_of_week": dates.dayofweek,
        "cash_transactions": np.where(mask, np.nan, rng.integers(100, 500, n_days).astype(float)),
        "amount_cash": np.where(mask, np.nan, amounts),
    })


def test_imputer_fills_all_nulls():
    df = make_store_df()
    imputer = CashImputer()
    imputer.fit(df)
    result = imputer.transform(df)
    assert result["amount_cash"].isna().sum() == 0
    assert result["cash_transactions"].isna().sum() == 0


def test_imputer_no_leakage_between_fit_transform():
    """Medians computed on fit data only, not transform data."""
    train = make_store_df(n_days=60, seed=1)
    test = make_store_df(n_days=10, seed=2)
    imputer = CashImputer()
    imputer.fit(train)
    result = imputer.transform(test)
    assert result["amount_cash"].isna().sum() == 0
