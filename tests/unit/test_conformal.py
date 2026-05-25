"""Tests for ConformalWrapper."""
import numpy as np
import pytest

from walmart_cash_forecast.models.conformal import ConformalWrapper


def test_calibrate_and_predict_interval():
    rng = np.random.default_rng(0)
    y_true = rng.uniform(10_000, 100_000, size=200).astype(np.float64)
    y_pred = y_true + rng.normal(0, 5_000, size=200)

    wrapper = ConformalWrapper(alpha=0.1)
    wrapper.calibrate(y_true, y_pred)

    assert wrapper.q_hat > 0
    intervals = wrapper.predict_interval(y_pred)
    assert set(intervals.columns) == {"lower", "upper"}
    # lower ≤ point prediction ≤ upper (q_hat ≥ 0 ensures this)
    assert (intervals["lower"] <= np.maximum(y_pred, 0)).all()
    assert (intervals["upper"] >= np.maximum(y_pred, 0)).all()


def test_empirical_coverage():
    """Marginal coverage must be ≥ 1−alpha on the calibration set itself."""
    rng = np.random.default_rng(42)
    n = 500
    y_true = rng.uniform(0, 100_000, size=n).astype(np.float64)
    y_pred = y_true + rng.normal(0, 8_000, size=n)

    wrapper = ConformalWrapper(alpha=0.1)
    wrapper.calibrate(y_true, y_pred)
    intervals = wrapper.predict_interval(y_pred)

    covered = (
        (y_true >= intervals["lower"].values) & (y_true <= intervals["upper"].values)
    ).mean()
    # Conformal guarantee: coverage ≥ 1−alpha
    assert covered >= 0.90 - 1e-6


def test_lower_bound_non_negative():
    """Lower bound should never be negative."""
    wrapper = ConformalWrapper(alpha=0.1)
    y_true = np.array([100.0, 200.0, 50.0])
    y_pred = np.array([80.0, 210.0, 60.0])
    wrapper.calibrate(y_true, y_pred)
    # Predict with a very small value — lower would go negative without clip
    intervals = wrapper.predict_interval(np.array([5.0]))
    assert (intervals["lower"] >= 0).all()


def test_save_load(tmp_path):
    rng = np.random.default_rng(7)
    y_true = rng.uniform(10_000, 50_000, size=100).astype(np.float64)
    y_pred = y_true + rng.normal(0, 3_000, size=100)

    wrapper = ConformalWrapper(alpha=0.1)
    wrapper.calibrate(y_true, y_pred)

    wrapper.save(tmp_path / "conformal")
    loaded = ConformalWrapper(alpha=0.1)
    loaded.load(tmp_path / "conformal")

    assert abs(loaded.q_hat - wrapper.q_hat) < 1e-6


def test_calibrate_empty_raises():
    wrapper = ConformalWrapper(alpha=0.1)
    with pytest.raises(ValueError, match="empty"):
        wrapper.calibrate(np.array([]), np.array([]))


def test_predict_before_calibrate_raises():
    wrapper = ConformalWrapper(alpha=0.1)
    with pytest.raises(RuntimeError, match="calibrate"):
        wrapper.predict_interval(np.array([1.0, 2.0]))
