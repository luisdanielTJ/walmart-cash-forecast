"""Tests for NewsvendorOptimizer."""
import numpy as np
import pandas as pd
import pytest

from walmart_cash_forecast.optimization.newsvendor import NewsvendorOptimizer


def make_samples(n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.lognormal(mean=11.0, sigma=0.4, size=n).astype(np.float64)


def test_critical_ratio():
    opt = NewsvendorOptimizer(cost_underage=3.0, cost_overage=1.0)
    assert abs(opt.critical_ratio - 0.75) < 1e-9


def test_optimise_returns_correct_quantile():
    """q* must match the empirical CR-th quantile of the samples."""
    opt = NewsvendorOptimizer(cost_underage=3.0, cost_overage=1.0)
    samples = make_samples()
    result = opt.optimise("STR_001", pd.Timestamp("2024-01-15"), samples)

    expected_q = float(np.quantile(samples, opt.critical_ratio))
    assert abs(result.q_star - expected_q) < 1.0  # tolerance: MXN rounding
    assert result.critical_ratio == pytest.approx(0.75, abs=1e-9)


def test_expected_cost_at_q_star_less_than_extremes():
    """q* must yield lower expected cost than either q10 or q90."""
    opt = NewsvendorOptimizer(cost_underage=3.0, cost_overage=1.0)
    samples = make_samples()
    result = opt.optimise("STR_001", pd.Timestamp("2024-01-15"), samples)

    cost_q10 = opt._expected_cost(samples, result.q10)
    cost_q90 = opt._expected_cost(samples, result.q90)
    assert result.expected_cost <= cost_q10
    assert result.expected_cost <= cost_q90


def test_sensitivity_returns_dataframe():
    opt = NewsvendorOptimizer(cost_underage=3.0, cost_overage=1.0)
    samples = make_samples()
    df = opt.sensitivity(samples, cost_underage_range=(1.0, 9.0), n_points=10)

    assert len(df) == 10
    assert set(df.columns) == {"cost_underage", "critical_ratio", "q_star", "expected_cost"}
    # Higher underage cost → higher critical ratio → higher q*
    assert df["q_star"].is_monotonic_increasing


def test_invalid_costs_raise():
    with pytest.raises(ValueError):
        NewsvendorOptimizer(cost_underage=-1.0, cost_overage=1.0)
    with pytest.raises(ValueError):
        NewsvendorOptimizer(cost_underage=1.0, cost_overage=0.0)
