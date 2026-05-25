"""Tests for StatAnalyzer."""
import json
import numpy as np
import pandas as pd
from walmart_cash_forecast.stats.reporter import StatAnalyzer


def make_panel(n_stores=3, n_days=60):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=n_days)
    rows = []
    for s in range(1, n_stores + 1):
        for date in dates:
            rows.append({
                "store_id": f"STR_{s:03d}",
                "date": date,
                "amount_cash": float(rng.uniform(10_000, 200_000)),
                "cash_transactions": int(rng.integers(100, 1000)),
                "total_transactions": int(rng.integers(200, 2000)),
                "amount_total": float(rng.uniform(50_000, 500_000)),
                "is_payday": date.day in (15, 28, 29, 30, 31),
                "day_of_week": date.dayofweek,
                "socioeconomic_level": "C",
            })
    return pd.DataFrame(rows)


def test_stat_analyzer_produces_json_report(tmp_path):
    panel = make_panel()
    report = StatAnalyzer().run(panel, output_dir=tmp_path)
    report_file = tmp_path / "stats_summary.json"
    assert report_file.exists()
    loaded = json.loads(report_file.read_text())
    assert "distribution" in loaded
    assert "payday_effect" in loaded
    assert "stationarity" in loaded
    assert "decomposition" in loaded


def test_stat_analyzer_returns_dict(tmp_path):
    panel = make_panel()
    result = StatAnalyzer().run(panel, output_dir=tmp_path)
    assert isinstance(result, dict)
    assert result["distribution"]["best_model"] in ("poisson", "negbin", "zinb")
