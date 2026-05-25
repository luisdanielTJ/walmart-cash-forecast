"""Schema and range validation for raw CSVs.

Runs immediately after loading so data issues surface with a clear error
message rather than causing cryptic failures deep in the pipeline.
"""
from __future__ import annotations

import pandas as pd


class ValidationError(Exception):
    """Raised when a DataFrame fails schema validation."""


# Minimum required columns per CSV — not all columns are required so we
# remain robust to schema additions without breaking the pipeline
_TRANSACTION_COLS = {
    "date", "store_id", "category", "total_transactions",
    "cash_transactions", "card_transactions", "amount_total",
    "amount_cash", "amount_card", "has_promotion",
}
_STORE_COLS = {"store_id", "store_format", "region", "size_sqm", "socioeconomic_level"}
_CALENDAR_COLS = {
    "date", "day_of_week", "is_payday", "is_weekend",
    "is_holiday", "is_buen_fin", "is_navidad_season", "is_semana_santa",
}


class DataValidator:
    """Validates that DataFrames conform to the expected schema."""

    def validate_transactions(
        self,
        df: pd.DataFrame,
        known_stores: list[str] | None = None,
    ) -> None:
        """Check required columns and optionally that all store_ids are known.

        Args:
            df: Transactions DataFrame to validate.
            known_stores: If provided, any store_id not in this list raises an error.
        """
        missing = _TRANSACTION_COLS - set(df.columns)
        if missing:
            raise ValidationError(f"transactions missing columns: {sorted(missing)}")

        if known_stores is not None:
            unknown = set(df["store_id"].unique()) - set(known_stores)
            if unknown:
                raise ValidationError(
                    f"store_id values not found in stores.csv: {sorted(unknown)}"
                )

    def validate_stores(self, df: pd.DataFrame) -> None:
        """Check required columns exist in stores.csv."""
        missing = _STORE_COLS - set(df.columns)
        if missing:
            raise ValidationError(f"stores missing columns: {sorted(missing)}")

    def validate_calendar(self, df: pd.DataFrame) -> None:
        """Check required columns exist in calendar.csv."""
        missing = _CALENDAR_COLS - set(df.columns)
        if missing:
            raise ValidationError(f"calendar missing columns: {sorted(missing)}")
