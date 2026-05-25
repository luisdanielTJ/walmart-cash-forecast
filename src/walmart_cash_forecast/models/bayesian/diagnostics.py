"""MCMC convergence diagnostics.

R-hat (Gelman-Rubin statistic) < 1.01 and ESS (Effective Sample Size) > 400
are the standard convergence thresholds for NUTS sampling. R-hat measures
whether multiple chains have converged to the same distribution; ESS measures
how many independent samples the correlated chain draws are equivalent to.

Reference: Gelman et al. (2013) Bayesian Data Analysis, 3rd ed., Ch. 11.
           Vehtari et al. (2021) "Rank-normalization, folding, and localization:
           An improved R-hat for assessing convergence of MCMC." Bayesian Analysis.
"""
from __future__ import annotations

import arviz as az


class ConvergenceError(Exception):
    """Raised when MCMC chains fail the R-hat or ESS convergence diagnostics."""


def check_convergence(trace: az.InferenceData) -> None:
    """
    Verify that all sampled parameters satisfy R-hat < 1.01 and ESS > 400.

    Args:
        trace: ArviZ InferenceData object returned by pm.sample().

    Raises:
        ConvergenceError: If any parameter violates either diagnostic threshold.
    """
    rhat_values = az.rhat(trace).to_array().values.flatten()
    max_rhat = float(rhat_values.max())
    if max_rhat > 1.01:
        raise ConvergenceError(
            f"R-hat > 1.01 detected (max R-hat = {max_rhat:.4f}). "
            "Chains have not converged — try increasing n_tune or simplifying the model."
        )

    ess_values = az.ess(trace).to_array().values.flatten()
    min_ess = float(ess_values.min())
    if min_ess < 400:
        raise ConvergenceError(
            f"ESS < 400 detected (min ESS = {min_ess:.0f}). "
            "Increase n_draws or reduce model complexity."
        )
