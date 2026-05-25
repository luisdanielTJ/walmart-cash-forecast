"""Aggregates category-level transactions to store-level daily totals.

Cash management is a store-level decision: cashiers share a single float,
not a per-category float. We therefore sum all 6 categories before modeling.
Category information is retained as derived features (e.g., category mix
affects average ticket size) but the forecast target is always store-level.
"""
from __future__ import annotations

import pandas as pd


class StoreAggregator:
    """Collapses the store×category panel to a store×date panel."""

    # Numeric columns to sum across all categories for a given store-date
    _SUM_COLS = [
        "cash_transactions", "card_transactions", "total_transactions",
        "amount_cash", "amount_card", "amount_total",
        "units_sold",
    ]
    # Binary flags: if ANY category had a promotion, the store had a promotion
    _MAX_COLS = ["has_promotion"]

    def aggregate(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Sum numeric columns across categories, grouped by store + date.

        Args:
            transactions: Raw transactions DataFrame (store × category × date).

        Returns:
            Store-level daily DataFrame — one row per (store_id, date) pair.
        """
        agg_dict = {col: "sum" for col in self._SUM_COLS if col in transactions.columns}
        agg_dict.update({col: "max" for col in self._MAX_COLS if col in transactions.columns})

        return (
            transactions
            .groupby(["store_id", "date"], as_index=False)
            .agg(agg_dict)
        )
