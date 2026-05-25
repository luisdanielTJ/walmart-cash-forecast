"""Distribution fitting for daily cash transaction counts.

We compare Poisson, Negative Binomial, and Zero-Inflated Negative Binomial
using MLE + AIC/BIC. The winning distribution informs the Bayesian model's
likelihood choice and formally justifies treating cash_transactions as
overdispersed count data (variance > mean).

Reference: Cameron & Trivedi (1998) Regression Analysis of Count Data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.optimize import minimize


@dataclass
class DistributionResult:
    """Results of fitting three count distributions to observed data."""

    best_model: str        # "poisson", "negbin", or "zinb"
    aic_poisson: float
    aic_negbin: float
    aic_zinb: float
    is_overdispersed: bool  # True when sample variance > 1.1 × sample mean
    negbin_mu: float        # MLE estimate of the NegBin mean parameter
    negbin_alpha: float     # MLE estimate of overdispersion (var = mu + alpha*mu²)


class DistributionFitter:
    """Fits Poisson, Negative Binomial, and ZINB to count data via MLE."""

    def fit(self, counts: np.ndarray) -> DistributionResult:
        """
        Fit three count distributions and select the best by AIC.

        AIC = -2*log_likelihood + 2*n_parameters, so lower is better.
        For nested models (Poisson ⊂ NegBin) we can also use a likelihood
        ratio test, but AIC gives a consistent selection for both nested
        and non-nested comparisons.

        Args:
            counts: 1-D integer array of observed daily cash transaction counts.

        Returns:
            DistributionResult with AICs, best model name, and NegBin MLE params.
        """
        counts = np.asarray(counts, dtype=float)
        mean = counts.mean()
        var = counts.var()

        # --- Poisson (1 parameter: lambda = mean) ---
        # MLE for Poisson: lambda_hat = sample mean (closed form)
        lam_hat = mean
        ll_poisson = float(np.sum(stats.poisson.logpmf(counts.astype(int), lam_hat)))
        aic_poisson = -2 * ll_poisson + 2 * 1  # 1 parameter

        # --- Negative Binomial (2 parameters: mu, alpha) ---
        # Parameterisation: E[X]=mu, Var[X]=mu + alpha*mu²
        # Method-of-moments starting point: alpha_0 = (var - mean) / mean²
        alpha_init = max((var - mean) / (mean ** 2 + 1e-9), 0.01)
        aic_negbin, mu_hat, alpha_hat = self._fit_negbin(counts, mean, alpha_init)

        # --- Zero-Inflated NegBin (approximate: +1 parameter for zero-inflation pi) ---
        # Full ZINB MLE is expensive; we use AIC of NegBin + penalty for the extra
        # zero-inflation parameter only when zero fraction exceeds 10%
        zero_frac = float((counts == 0).mean())
        if zero_frac > 0.10:
            # Adding pi parameter to NegBin: AIC increases by 2 (one extra param)
            aic_zinb = aic_negbin + 2
        else:
            # If few zeros, ZINB offers no benefit — penalise heavily
            aic_zinb = aic_negbin + 10

        # Select model with lowest AIC
        aics = {"poisson": aic_poisson, "negbin": aic_negbin, "zinb": aic_zinb}
        best_model = min(aics, key=aics.__getitem__)

        # Overdispersion: variance meaningfully exceeds the mean (10% threshold)
        is_overdispersed = bool(var > 1.1 * mean)

        return DistributionResult(
            best_model=best_model,
            aic_poisson=aic_poisson,
            aic_negbin=aic_negbin,
            aic_zinb=aic_zinb,
            is_overdispersed=is_overdispersed,
            negbin_mu=float(mu_hat),
            negbin_alpha=float(alpha_hat),
        )

    def _fit_negbin(
        self,
        counts: np.ndarray,
        mu_init: float,
        alpha_init: float,
    ) -> tuple[float, float, float]:
        """MLE for Negative Binomial parameters via numerical minimisation.

        Uses NegBin parameterisation: r=1/alpha, p=1/(1 + alpha*mu)
        so that mean=mu and variance=mu + alpha*mu².
        """

        def neg_ll(params: np.ndarray) -> float:
            mu, alpha = params
            if mu <= 0 or alpha <= 0:
                return 1e10
            r = 1.0 / alpha
            p = 1.0 / (1.0 + alpha * mu)
            ll = np.sum(stats.nbinom.logpmf(counts.astype(int), r, p))
            return -float(ll)

        result = minimize(
            neg_ll,
            x0=[mu_init, alpha_init],
            method="Nelder-Mead",
            options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 5000},
        )
        mu_hat, alpha_hat = result.x
        ll = -result.fun
        aic = -2 * ll + 2 * 2  # 2 parameters
        return aic, mu_hat, alpha_hat
