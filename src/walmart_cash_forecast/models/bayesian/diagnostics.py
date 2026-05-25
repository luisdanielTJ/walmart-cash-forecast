"""MCMC convergence diagnostics.

R-hat (Gelman-Rubin statistic) < 1.01 and ESS (Effective Sample Size) > 400
are the standard convergence thresholds for NUTS sampling. R-hat measures
whether multiple chains have converged to the same distribution; ESS measures
how many independent samples the correlated chain draws are equivalent to.

Severity tiers used here:
  R-hat <= 1.01 and ESS >= 400  → converged (pass)
  1.01 < R-hat <= 1.05          → borderline: WARNING logged, training continues
  R-hat > 1.05 or ESS < 100    → hard failure: ConvergenceError raised

Reference: Gelman et al. (2013) Bayesian Data Analysis, 3rd ed., Ch. 11.
           Vehtari et al. (2021) "Rank-normalization, folding, and localization:
           An improved R-hat for assessing convergence of MCMC." Bayesian Analysis.
"""
from __future__ import annotations

import logging

import arviz as az

logger = logging.getLogger(__name__)


class ConvergenceError(Exception):
    """Raised when MCMC chains fail the R-hat or ESS convergence diagnostics."""


def check_convergence(trace: az.InferenceData) -> None:
    """
    Verify that sampled parameters satisfy convergence diagnostics.

    Borderline cases (1.01 < R-hat <= 1.05 or 100 <= ESS < 400) emit a WARNING
    so training can continue and produce artefacts. Hard failures (R-hat > 1.05
    or ESS < 100) raise ConvergenceError because the posterior is unreliable.

    Args:
        trace: ArviZ InferenceData object returned by pm.sample().

    Raises:
        ConvergenceError: If R-hat > 1.05 or ESS < 100.
    """
    # ArviZ >= 1.0 returns a DataTree; .ds exposes the root node's Dataset
    rhat_dt = az.rhat(trace)
    rhat_values = rhat_dt.ds.to_array().values.flatten()
    max_rhat = float(rhat_values.max())

    if max_rhat > 1.05:
        raise ConvergenceError(
            f"R-hat > 1.05 detected (max R-hat = {max_rhat:.4f}). "
            "Chains have not converged — increase n_tune or reparameterize."
        )
    if max_rhat > 1.01:
        logger.warning(
            "Borderline R-hat (max = %.4f, threshold 1.01). "
            "Consider increasing n_tune or max_treedepth for stricter convergence.",
            max_rhat,
        )

    ess_dt = az.ess(trace)
    ess_values = ess_dt.ds.to_array().values.flatten()
    min_ess = float(ess_values.min())

    if min_ess < 100:
        raise ConvergenceError(
            f"ESS < 100 detected (min ESS = {min_ess:.0f}). "
            "Increase n_draws or reduce model complexity."
        )
    if min_ess < 400:
        logger.warning(
            "Borderline ESS (min = %.0f, threshold 400). "
            "Consider increasing n_draws for more reliable posterior estimates.",
            min_ess,
        )
