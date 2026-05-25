"""Tests for DataLoader."""
import pandas as pd

from walmart_cash_forecast.data.loader import DataLoader


def test_loader_returns_three_dataframes(tmp_path, synthetic_transactions,
                                         synthetic_stores, synthetic_calendar):
    synthetic_transactions.to_csv(tmp_path / "transactions.csv", index=False)
    synthetic_stores.to_csv(tmp_path / "stores.csv", index=False)
    synthetic_calendar.to_csv(tmp_path / "calendar.csv", index=False)

    loader = DataLoader(tmp_path)
    transactions, stores, calendar = loader.load()

    assert isinstance(transactions, pd.DataFrame)
    assert isinstance(stores, pd.DataFrame)
    assert isinstance(calendar, pd.DataFrame)
    assert "date" in transactions.columns
    assert "store_id" in transactions.columns
    assert "amount_cash" in transactions.columns


def test_loader_parses_dates(tmp_path, synthetic_transactions,
                              synthetic_stores, synthetic_calendar):
    synthetic_transactions.to_csv(tmp_path / "transactions.csv", index=False)
    synthetic_stores.to_csv(tmp_path / "stores.csv", index=False)
    synthetic_calendar.to_csv(tmp_path / "calendar.csv", index=False)

    loader = DataLoader(tmp_path)
    transactions, _, calendar = loader.load()

    assert pd.api.types.is_datetime64_any_dtype(transactions["date"])
    assert pd.api.types.is_datetime64_any_dtype(calendar["date"])
