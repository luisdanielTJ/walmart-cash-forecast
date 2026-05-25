"""Tests for StationarityTester."""
import numpy as np
import pandas as pd

from walmart_cash_forecast.stats.stationarity import StationarityResult, StationarityTester


def test_stationary_series_detected():
    """White noise is stationary — ADF should reject the unit root (p < 0.05)."""
    rng = np.random.default_rng(42)
    series = pd.Series(rng.normal(0, 1, 300))
    result = StationarityTester().test(series)
    assert isinstance(result, StationarityResult)
    assert result.adf_pvalue < 0.05


def test_random_walk_is_nonstationary():
    """Random walk has a unit root — ADF should fail to reject (p > 0.05)."""
    rng = np.random.default_rng(42)
    series = pd.Series(np.cumsum(rng.normal(0, 1, 300)))
    result = StationarityTester().test(series)
    assert result.adf_pvalue > 0.05
