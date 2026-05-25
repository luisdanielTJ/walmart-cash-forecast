"""Tests for StoreAggregator."""
import pandas as pd

from walmart_cash_forecast.features.aggregator import StoreAggregator

CATEGORIES = ["Abarrotes", "Bebidas"]


def make_tx(n_stores=2, n_days=3):
    rows = []
    for s in range(1, n_stores + 1):
        for cat in CATEGORIES:
            for d in range(n_days):
                rows.append({
                    "date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=d),
                    "store_id": f"STR_{s:03d}",
                    "category": cat,
                    "cash_transactions": 100,
                    "amount_cash": 50_000.0,
                    "total_transactions": 200,
                    "amount_total": 100_000.0,
                    "has_promotion": 0,
                })
    return pd.DataFrame(rows)


def test_aggregator_sums_across_categories():
    """2 categories × 100 cash_tx each → 200 per store-day."""
    result = StoreAggregator().aggregate(make_tx())
    assert (result["cash_transactions"] == 200).all()
    assert (result["amount_cash"] == 100_000.0).all()


def test_aggregator_output_shape():
    """2 stores × 3 days = 6 rows."""
    result = StoreAggregator().aggregate(make_tx(n_stores=2, n_days=3))
    assert len(result) == 6
    assert "store_id" in result.columns
    assert "date" in result.columns
