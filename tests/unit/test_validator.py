"""Tests for DataValidator."""
import pytest

from walmart_cash_forecast.data.validator import DataValidator, ValidationError


def test_validator_accepts_valid_transactions(synthetic_transactions):
    DataValidator().validate_transactions(synthetic_transactions)


def test_validator_rejects_missing_required_column(synthetic_transactions):
    bad = synthetic_transactions.drop(columns=["amount_cash"])
    with pytest.raises(ValidationError, match="amount_cash"):
        DataValidator().validate_transactions(bad)


def test_validator_rejects_unknown_store_in_transactions(
        synthetic_transactions, synthetic_stores):
    bad = synthetic_transactions.copy()
    bad.loc[0, "store_id"] = "STR_GHOST"
    with pytest.raises(ValidationError, match="store_id"):
        DataValidator().validate_transactions(
            bad, known_stores=synthetic_stores["store_id"].tolist()
        )


def test_validator_accepts_valid_stores(synthetic_stores):
    DataValidator().validate_stores(synthetic_stores)


def test_validator_accepts_valid_calendar(synthetic_calendar):
    DataValidator().validate_calendar(synthetic_calendar)
