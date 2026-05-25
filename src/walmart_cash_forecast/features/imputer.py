"""Median-by-day-of-week imputation for cash columns.

Using day-of-week medians preserves the weekly seasonality pattern rather
than collapsing to a flat global mean, which would distort weekday/weekend
differences that are predictive for cash demand.

fit() computes medians from training data only; transform() applies them —
this prevents any data leakage from future periods into the imputed values.
"""
from __future__ import annotations

import pandas as pd

# Columns that may have nulls due to POS/connectivity failures
_CASH_COLS = ["cash_transactions", "amount_cash"]


class CashImputer:
    """Fills nulls in cash columns using store×day-of-week medians from training data."""

    def __init__(self) -> None:
        # Maps (store_id, day_of_week) → {column: median_value}
        self._medians: dict[tuple[str, int], dict[str, float]] = {}
        # Global fallback for store-dow combinations not seen during fit
        self._global_medians: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "CashImputer":
        """Compute day-of-week medians per store from the training DataFrame."""
        for (store_id, dow), group in df.groupby(["store_id", "day_of_week"]):
            self._medians[(str(store_id), int(dow))] = {  # type: ignore[call-overload]
                col: float(group[col].median())
                for col in _CASH_COLS
                if col in df.columns
            }
        # Global fallback used when a store-dow key is unseen at transform time
        self._global_medians = {
            col: float(df[col].median())
            for col in _CASH_COLS
            if col in df.columns
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill nulls using fitted medians; unseen store-dow pairs use global median."""
        df = df.copy()
        for col in _CASH_COLS:
            if col not in df.columns:
                continue
            null_mask = df[col].isna()
            if not null_mask.any():
                continue

            def _fill(row: pd.Series) -> float:
                key = (row["store_id"], int(row["day_of_week"]))
                return self._medians.get(key, {}).get(
                    col, self._global_medians.get(col, 0.0)
                )

            df.loc[null_mask, col] = df[null_mask].apply(_fill, axis=1)
        return df
