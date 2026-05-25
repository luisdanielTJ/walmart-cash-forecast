"""Shared pytest fixtures: synthetic data matching the real CSV schemas.

All fixtures use a fixed random seed so test results are deterministic.
The synthetic dataset covers 10 stores × 6 categories × 90 days to keep
test execution fast while exercising all code paths.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from walmart_cash_forecast.config import Config

SEED = 42
N_STORES = 10
N_DAYS = 90   # 60 train + 30 holdout
CATEGORIES = ["Abarrotes", "Bebidas", "Cuidado_Personal", "Hogar", "Electronica", "Ropa"]
REGIONS = ["Norte", "Centro", "Sur", "Occidente", "Oriente"]
FORMATS = ["Supercenter", "Bodega", "Express"]
SOC_LEVELS = ["C", "C+", "B", "A/B"]


@pytest.fixture(scope="session")
def synthetic_stores() -> pd.DataFrame:
    """Static store metadata for 10 synthetic stores."""
    rng = np.random.default_rng(SEED)
    store_ids = [f"STR_{i:03d}" for i in range(1, N_STORES + 1)]
    return pd.DataFrame({
        "store_id": store_ids,
        "store_format": rng.choice(FORMATS, size=N_STORES),
        "region": rng.choice(REGIONS, size=N_STORES),
        "size_sqm": rng.integers(500, 15000, size=N_STORES),
        "num_checkouts": rng.integers(4, 40, size=N_STORES),
        "opening_year": rng.integers(2005, 2023, size=N_STORES),
        "socioeconomic_level": rng.choice(SOC_LEVELS, size=N_STORES),
        "has_pharmacy": rng.choice([True, False], size=N_STORES),
        "has_fuel_station": rng.choice([True, False], size=N_STORES),
    })


@pytest.fixture(scope="session")
def synthetic_calendar() -> pd.DataFrame:
    """Calendar with Mexican event flags for 90 days starting 2023-01-01."""
    dates = pd.date_range("2023-01-01", periods=N_DAYS)
    rng = np.random.default_rng(SEED)
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "day_of_week": dates.dayofweek,
        "day_name": dates.day_name(),
        "week_of_year": dates.isocalendar().week.values,
        "month": dates.month,
        "year": dates.year,
        "quarter": dates.quarter,
        "season": "Invierno",
        "is_holiday": rng.choice([True, False], p=[0.05, 0.95], size=N_DAYS),
        "holiday_name": None,
        "is_payday": [d.day in (15, 28, 29, 30, 31) for d in dates],
        "is_weekend": dates.dayofweek >= 5,
        "is_navidad_season": False,
        "is_buen_fin": False,
        "is_semana_santa": False,
    })


@pytest.fixture(scope="session")
def synthetic_transactions(synthetic_stores) -> pd.DataFrame:
    """Daily transactions per store × category for 90 days, with realistic nulls."""
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2023-01-01", periods=N_DAYS)
    store_ids = synthetic_stores["store_id"].tolist()
    rows = []
    for store in store_ids:
        for cat in CATEGORIES:
            for date in dates:
                total = int(rng.integers(200, 2000))
                cash_frac = rng.uniform(0.3, 0.7)
                cash_tx = int(total * cash_frac)
                amount_total = float(rng.uniform(50_000, 500_000))
                amount_cash = amount_total * cash_frac
                rows.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "store_id": store,
                    "category": cat,
                    "total_transactions": total,
                    # ~5% null rate simulates POS/connectivity failures
                    "cash_transactions": cash_tx if rng.random() > 0.05 else None,
                    "card_transactions": total - cash_tx,
                    "amount_total": amount_total,
                    "amount_cash": amount_cash if rng.random() > 0.05 else None,
                    "amount_card": amount_total - amount_cash,
                    "units_sold": float(rng.integers(100, 5000)) if rng.random() > 0.05 else None,
                    "avg_ticket": amount_total / total if rng.random() > 0.05 else None,
                    "has_promotion": int(rng.random() > 0.8),
                    "replenishment_signal": float(rng.uniform(100, 2000)) if rng.random() > 0.1 else None,
                })
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def default_config() -> Config:
    """Default Config with standard hyperparameters."""
    return Config()
