"""Feature engineering pipeline with fit/transform to prevent data leakage.

All lag and rolling features are computed within each store's time series,
sorted chronologically, so feature values at time T use only data from T-k.
The fit/transform pattern mirrors scikit-learn's API, making it safe to
apply the same fitted engine to both training and test data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from walmart_cash_forecast.config import Config


class FeatureEngine:
    """Builds the full feature matrix from the store-level daily panel."""

    # Number of days to lag (amount_cash and cash_transactions)
    _LAG_WINDOWS = [1, 7, 14]
    # Rolling statistic windows in days — computed on amount_cash
    _ROLL_WINDOWS = [7, 28]

    def __init__(self, config: Config) -> None:
        self.config = config
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "FeatureEngine":
        """Nothing to learn — fit exists for API consistency and pipeline safety."""
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add lag, rolling, ratio, and payday-proximity features.

        All transformations are applied per store to respect the panel structure
        and prevent information leakage across stores.

        Args:
            df: Store-level daily panel, must contain store_id, date, amount_cash,
                cash_transactions, total_transactions, is_payday columns.

        Returns:
            DataFrame with all original columns plus engineered features.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        df = df.copy().sort_values(["store_id", "date"]).reset_index(drop=True)

        # --- Lag features: yesterday, last week, two weeks ago ---
        for lag in self._LAG_WINDOWS:
            df[f"amount_cash_lag_{lag}"] = (
                df.groupby("store_id")["amount_cash"].shift(lag)
            )
            df[f"cash_tx_lag_{lag}"] = (
                df.groupby("store_id")["cash_transactions"].shift(lag)
            )

        # --- Rolling statistics on amount_cash ---
        # shift(1) ensures we only look at past data, never the current day
        for window in self._ROLL_WINDOWS:
            grouped = df.groupby("store_id")["amount_cash"]
            df[f"amount_cash_roll{window}_mean"] = grouped.transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean()
            )
            df[f"amount_cash_roll{window}_std"] = grouped.transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).std()
            )
            df[f"amount_cash_roll{window}_max"] = grouped.transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).max()
            )

        # --- Cash ratio: fraction of all transactions that are cash ---
        # Higher ratio → more change events per customer → more change needed
        df["cash_ratio"] = (
            df["cash_transactions"] / df["total_transactions"].replace(0, np.nan)
        )

        # --- Payday proximity features ---
        # Days since / until the next quincena captures the spending momentum
        # around Mexico's bimonthly payday cycle (day 15 and last day of month)
        df["days_since_payday"] = (
            df.groupby("store_id")["is_payday"]
            .transform(lambda s: _days_since_true(s))
        )
        df["days_until_payday"] = (
            df.groupby("store_id")["is_payday"]
            .transform(lambda s: _days_until_true(s))
        )

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience method: fit then transform in one call."""
        return self.fit(df).transform(df)


def _days_since_true(series: pd.Series) -> pd.Series:
    """Return the number of days since the last True value in a boolean series."""
    result = []
    count = 0
    for val in series:
        if val:
            count = 0
        else:
            count += 1
        result.append(count)
    return pd.Series(result, index=series.index)


def _days_until_true(series: pd.Series) -> pd.Series:
    """Return the number of days until the next True value in a boolean series."""
    # Reverse the series, compute days_since, then reverse back
    return _days_since_true(series.iloc[::-1]).iloc[::-1]
