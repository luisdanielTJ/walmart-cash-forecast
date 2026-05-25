"""ADF and KPSS stationarity tests for store-level time series.

ADF (null: unit root present) and KPSS (null: stationary) test opposite
null hypotheses. Using both together avoids ambiguity: if ADF rejects AND
KPSS fails to reject, we have strong evidence of stationarity. If they
disagree, the series may be fractionally integrated.

Reference: Kwiatkowski, Phillips, Schmidt & Shin (1992) for KPSS;
           Dickey & Fuller (1979) for ADF.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


@dataclass
class StationarityResult:
    """Results from joint ADF + KPSS testing."""

    adf_statistic: float
    adf_pvalue: float
    kpss_statistic: float
    kpss_pvalue: float
    # Joint verdict: True when ADF rejects unit root AND KPSS fails to reject stationarity
    is_stationary: bool


class StationarityTester:
    """Runs ADF and KPSS tests on a univariate time series."""

    def test(self, series: pd.Series) -> StationarityResult:
        """
        Test stationarity using ADF and KPSS.

        Args:
            series: Univariate time series (e.g. amount_cash for one store).

        Returns:
            StationarityResult with both test statistics and a joint verdict.
        """
        series = series.dropna()

        # ADF: autolag='AIC' selects optimal lag order to control autocorrelation
        adf_stat, adf_p, *_ = adfuller(series, autolag="AIC")

        # KPSS: suppress the SpecificationWarning about lags that statsmodels emits
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, kpss_p, *_ = kpss(series, regression="c", nlags="auto")

        # Stationarity: ADF rejects unit root (p<0.05) AND KPSS fails to reject (p>0.05)
        is_stationary = bool((adf_p < 0.05) and (kpss_p > 0.05))

        return StationarityResult(
            adf_statistic=float(adf_stat),
            adf_pvalue=float(adf_p),
            kpss_statistic=float(kpss_stat),
            kpss_pvalue=float(kpss_p),
            is_stationary=is_stationary,
        )
