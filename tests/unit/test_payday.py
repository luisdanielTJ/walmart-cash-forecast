"""Tests for PaydayEffectTester."""
import numpy as np
import pandas as pd

from walmart_cash_forecast.stats.payday import PaydayEffectTester, PaydayResult


def test_payday_effect_detected_when_present():
    rng = np.random.default_rng(42)
    n = 200
    is_payday = rng.choice([True, False], size=n, p=[0.1, 0.9])
    # Payday days have clearly higher cash amounts
    amount_cash = np.where(is_payday,
                           rng.uniform(80_000, 120_000, n),
                           rng.uniform(10_000, 30_000, n))
    df = pd.DataFrame({"is_payday": is_payday, "amount_cash": amount_cash,
                       "socioeconomic_level": "C"})
    result = PaydayEffectTester().test(df)
    assert isinstance(result, PaydayResult)
    assert result.pvalue < 0.05
    assert result.effect_size > 0


def test_payday_no_effect_when_absent():
    rng = np.random.default_rng(42)
    n = 200
    is_payday = rng.choice([True, False], size=n, p=[0.1, 0.9])
    # Same distribution on both days
    amount_cash = rng.uniform(30_000, 80_000, n)
    df = pd.DataFrame({"is_payday": is_payday, "amount_cash": amount_cash,
                       "socioeconomic_level": "B"})
    result = PaydayEffectTester().test(df)
    assert result.pvalue > 0.05
