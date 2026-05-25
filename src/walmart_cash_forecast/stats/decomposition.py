"""STL decomposition and Ljung-Box residual autocorrelation test.

STL (Seasonal-Trend decomposition via LOESS) decomposes a time series into
trend + seasonal + residual components. Using period=7 captures the dominant
weekly retail cycle. A Ljung-Box test on the residuals verifies that the
assumed seasonal structure is sufficient: white-noise residuals confirm the
model has captured all systematic patterns.

Reference: Cleveland, Cleveland, McRae & Terpenning (1990) for STL;
           Ljung & Box (1978) for the portmanteau test.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.diagnostic import acorr_ljungbox


@dataclass
class DecompositionResult:
    """Results of STL decomposition and residual testing."""

    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series
    ljungbox_pvalue: float           # p-value of Ljung-Box test on residuals
    residuals_are_white_noise: bool  # True if Ljung-Box fails to reject (p > 0.05)


class SeriesDecomposer:
    """Applies STL decomposition to a single store's amount_cash time series."""

    def __init__(self, period: int = 7) -> None:
        # period=7 captures the weekly retail seasonality pattern
        self.period = period

    def decompose(self, series: pd.Series) -> DecompositionResult:
        """
        Decompose the series and test residuals for remaining autocorrelation.

        Args:
            series: Daily amount_cash for one store, indexed by date.

        Returns:
            DecompositionResult with trend, seasonal, residual, and Ljung-Box p-value.
        """
        series = series.dropna()
        stl = STL(series, period=self.period, robust=True)
        fit = stl.fit()

        # Ljung-Box test on residuals at lag 10: H₀ = no autocorrelation (white noise)
        lb_result = acorr_ljungbox(fit.resid, lags=[10], return_df=True)
        lb_pvalue = float(lb_result["lb_pvalue"].iloc[0])

        return DecompositionResult(
            trend=fit.trend,
            seasonal=fit.seasonal,
            residual=fit.resid,
            ljungbox_pvalue=lb_pvalue,
            residuals_are_white_noise=lb_pvalue > 0.05,
        )
