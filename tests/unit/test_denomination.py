"""Tests for DenominationSolver."""
import pandas as pd
import pytest

from walmart_cash_forecast.optimization.denomination import DENOMINATIONS, DenominationSolver


def test_solve_covers_target():
    solver = DenominationSolver()
    result = solver.solve("STR_001", pd.Timestamp("2024-01-15"), target=5_000.0)
    assert result.total_value >= 5_000.0
    assert result.status == "Optimal"


def test_solve_minimises_pieces():
    """Naive strategy (all small coins) would use far more pieces."""
    solver = DenominationSolver()
    result = solver.solve("STR_001", pd.Timestamp("2024-01-15"), target=1_000.0)
    # With $1000 bills available, optimal mix needs at most 1 piece
    assert result.total_pieces <= 10


def test_solve_all_pieces_positive():
    solver = DenominationSolver()
    result = solver.solve("STR_001", pd.Timestamp("2024-01-15"), target=3_500.0)
    for d, n in result.mix.items():
        assert n > 0  # mix should only contain denominations actually used


def test_solve_batch():
    solver = DenominationSolver()
    targets = pd.DataFrame([
        {"store_id": "STR_001", "date": pd.Timestamp("2024-01-15"), "q_star": 2_000.0},
        {"store_id": "STR_002", "date": pd.Timestamp("2024-01-15"), "q_star": 8_000.0},
        {"store_id": "STR_003", "date": pd.Timestamp("2024-01-15"), "q_star": 500.0},
    ])
    results = solver.solve_batch(targets)
    assert len(results) == 3
    for r in results:
        assert r.total_value >= r.target


def test_to_dataframe():
    solver = DenominationSolver()
    targets = pd.DataFrame([
        {"store_id": "STR_001", "date": pd.Timestamp("2024-01-15"), "q_star": 1_000.0},
    ])
    results = solver.solve_batch(targets)
    df = solver.to_dataframe(results)
    assert "store_id" in df.columns
    assert "total_pieces" in df.columns
    # All denomination columns should be present
    for d in DENOMINATIONS:
        assert f"denom_{d:.2f}" in df.columns


def test_infeasible_raises():
    """Target far exceeding all capacity limits should raise RuntimeError."""
    tiny_limits = {d: 0 for d in DENOMINATIONS}
    solver = DenominationSolver(limits=tiny_limits)
    with pytest.raises(RuntimeError, match="infeasible"):
        solver.solve("STR_001", pd.Timestamp("2024-01-15"), target=100.0)
