"""Tests for DenominationSolver (proportional change-fund allocator)."""
import pandas as pd

from walmart_cash_forecast.optimization.denomination import (
    DENOMINATIONS,
    DenominationSolver,
    _COIN_MINIMUMS,
)


def test_coin_minimums_always_present():
    """Fixed coin counts appear regardless of target size."""
    solver = DenominationSolver()
    result = solver.solve("STR_001", pd.Timestamp("2024-01-15"), target=50_000.0)
    for d, minimum in _COIN_MINIMUMS.items():
        assert result.mix.get(d, 0) == minimum, f"coin {d} should be {minimum}"


def test_bill_quantities_scale_with_target():
    """Larger daily forecast → larger bill counts (proportional)."""
    solver = DenominationSolver()
    low = solver.solve("STR_001", pd.Timestamp("2024-01-15"), target=300_000.0)
    high = solver.solve("STR_001", pd.Timestamp("2024-01-15"), target=900_000.0)
    # $100 and $200 bill counts should be strictly larger for the higher forecast
    assert high.mix.get(100.0, 0) > low.mix.get(100.0, 0)
    assert high.mix.get(200.0, 0) > low.mix.get(200.0, 0)


def test_large_bills_excluded():
    """$500 and $1000 bills are never stocked in the register float."""
    solver = DenominationSolver()
    result = solver.solve("STR_001", pd.Timestamp("2024-01-15"), target=1_000_000.0)
    assert result.mix.get(500.0, 0) == 0
    assert result.mix.get(1_000.0, 0) == 0


def test_status_is_proportional():
    """Allocator always returns 'Proportional' status (not ILP)."""
    solver = DenominationSolver()
    result = solver.solve("STR_001", pd.Timestamp("2024-01-15"), target=500_000.0)
    assert result.status == "Proportional"


def test_solve_batch():
    """Batch solve returns one result per row; higher targets give higher counts."""
    solver = DenominationSolver()
    targets = pd.DataFrame([
        {"store_id": "STR_001", "date": pd.Timestamp("2024-01-15"), "q_star": 300_000.0},
        {"store_id": "STR_002", "date": pd.Timestamp("2024-01-15"), "q_star": 900_000.0},
        {"store_id": "STR_003", "date": pd.Timestamp("2024-01-15"), "q_star": 600_000.0},
    ])
    results = solver.solve_batch(targets)
    assert len(results) == 3
    # $100 bill count should increase monotonically with target
    counts_100 = [r.mix.get(100.0, 0) for r in results]
    assert counts_100[1] > counts_100[0]   # STR_002 > STR_001
    assert counts_100[1] > counts_100[2]   # STR_002 > STR_003


def test_to_dataframe():
    """DataFrame output contains all denomination columns."""
    solver = DenominationSolver()
    targets = pd.DataFrame([
        {"store_id": "STR_001", "date": pd.Timestamp("2024-01-15"), "q_star": 500_000.0},
    ])
    results = solver.solve_batch(targets)
    df = solver.to_dataframe(results)
    assert "store_id" in df.columns
    assert "total_pieces" in df.columns
    for d in DENOMINATIONS:
        assert f"denom_{d:.2f}" in df.columns


def test_fund_ratio_controls_scale():
    """A higher fund_ratio produces proportionally more bills."""
    low_ratio = DenominationSolver(fund_ratio=0.02)
    high_ratio = DenominationSolver(fund_ratio=0.06)
    target = 500_000.0
    low_result = low_ratio.solve("STR_001", pd.Timestamp("2024-01-15"), target)
    high_result = high_ratio.solve("STR_001", pd.Timestamp("2024-01-15"), target)
    assert high_result.mix.get(100.0, 0) > low_result.mix.get(100.0, 0)
