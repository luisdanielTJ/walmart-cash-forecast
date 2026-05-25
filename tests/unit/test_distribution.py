"""Tests for DistributionFitter."""
import numpy as np

from walmart_cash_forecast.stats.distribution import DistributionFitter, DistributionResult


def test_fitter_returns_result_object():
    rng = np.random.default_rng(42)
    counts = rng.negative_binomial(n=5, p=0.4, size=200)
    result = DistributionFitter().fit(counts)
    assert isinstance(result, DistributionResult)
    assert result.best_model in ("poisson", "negbin", "zinb")


def test_fitter_detects_overdispersion():
    """Strongly overdispersed NegBin data should select negbin over Poisson."""
    rng = np.random.default_rng(42)
    counts = rng.negative_binomial(n=3, p=0.3, size=500)
    result = DistributionFitter().fit(counts)
    assert result.is_overdispersed
    assert result.aic_negbin < result.aic_poisson


def test_fitter_result_has_negbin_params():
    rng = np.random.default_rng(42)
    counts = rng.negative_binomial(n=5, p=0.4, size=300)
    result = DistributionFitter().fit(counts)
    assert result.negbin_mu > 0
    assert result.negbin_alpha > 0
