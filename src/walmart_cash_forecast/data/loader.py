"""Loads the three raw CSVs and returns typed DataFrames.

Kept intentionally simple: no joins, no transformations — just reading
and type-casting. Downstream modules handle all data manipulation.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


class DataLoader:
    """Reads transactions, stores, and calendar CSVs from a directory."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all three source files.

        Returns:
            (transactions, stores, calendar) — date columns parsed as datetime.
        """
        transactions = pd.read_csv(self.data_dir / "transactions.csv", parse_dates=["date"])
        stores = pd.read_csv(self.data_dir / "stores.csv")
        calendar = pd.read_csv(self.data_dir / "calendar.csv", parse_dates=["date"])
        return transactions, stores, calendar
