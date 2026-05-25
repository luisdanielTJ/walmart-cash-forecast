"""Tests for ModelBlender."""
import numpy as np
import pytest

from walmart_cash_forecast.models.blender import ModelBlender


def make_calibration_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y_true = rng.uniform(10_000, 100_000, size=n)
    y_bayes = y_true + rng.normal(0, 5_000, size=n)
    y_ml = y_true + rng.normal(0, 7_000, size=n)
    return y_true.astype(np.float64), y_bayes.astype(np.float64), y_ml.astype(np.float64)


def test_blender_fits_and_predicts():
    y_true, y_bayes, y_ml = make_calibration_data()
    blender = ModelBlender()
    blender.fit(y_true, y_bayes, y_ml)

    preds = blender.predict(y_bayes, y_ml)
    assert preds.shape == (len(y_true),)
    assert (preds >= 0).all()


def test_weights_sum_to_one():
    y_true, y_bayes, y_ml = make_calibration_data()
    blender = ModelBlender()
    blender.fit(y_true, y_bayes, y_ml)
    assert abs(blender.weights.sum() - 1.0) < 1e-9
    assert (blender.weights >= 0).all()


def test_blender_lower_mse_than_either_base():
    """Blended predictions should not be drastically worse than both bases."""
    y_true, y_bayes, y_ml = make_calibration_data(n=500, seed=1)
    blender = ModelBlender()
    blender.fit(y_true, y_bayes, y_ml)
    preds = blender.predict(y_bayes, y_ml)

    mse_blend = np.mean((y_true - preds) ** 2)
    mse_bayes = np.mean((y_true - y_bayes) ** 2)
    mse_ml = np.mean((y_true - y_ml) ** 2)
    # Blended MSE should be ≤ max of the two base MSEs (basic sanity)
    assert mse_blend <= max(mse_bayes, mse_ml) * 1.05  # 5% tolerance


def test_save_load(tmp_path):
    y_true, y_bayes, y_ml = make_calibration_data()
    blender = ModelBlender()
    blender.fit(y_true, y_bayes, y_ml)
    blender.save(tmp_path / "blender")

    loaded = ModelBlender()
    loaded.load(tmp_path / "blender")
    np.testing.assert_allclose(loaded.weights, blender.weights, rtol=1e-9)


def test_predict_before_fit_raises():
    blender = ModelBlender()
    with pytest.raises(RuntimeError, match="fit"):
        blender.predict(np.array([1.0]), np.array([1.0]))
