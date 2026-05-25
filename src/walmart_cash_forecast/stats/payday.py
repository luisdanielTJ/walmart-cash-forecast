"""Mann-Whitney U test for payday effect on cash demand.

The quincena (payday on day 15 and last day of month) is the strongest
external driver of cash demand spikes in Mexican retail. We test it
formally rather than assuming it using a non-parametric Mann-Whitney U test,
which makes no normality assumption on the cash amount distribution.

Effect size is reported as the proportional difference in medians:
(median_payday - median_nonpayday) / median_nonpayday, stratified by
socioeconomic level because the effect is significantly stronger in lower-
income areas (C, C+) where cash usage rates are higher.

Reference: Mann & Whitney (1947); non-parametric two-sample location test.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy import stats


@dataclass
class PaydayResult:
    """Results of Mann-Whitney U test comparing payday vs. non-payday days."""

    statistic: float
    pvalue: float
    # Proportional effect size: (median_payday / median_nonpayday) - 1
    effect_size: float
    payday_median: float
    nonpayday_median: float
    socioeconomic_level: str


class PaydayEffectTester:
    """Tests whether payday days have significantly higher cash demand."""

    def test(self, df: pd.DataFrame) -> PaydayResult:
        """
        Run Mann-Whitney U test comparing amount_cash on payday vs. non-payday days.

        Args:
            df: DataFrame with columns: is_payday, amount_cash, socioeconomic_level.

        Returns:
            PaydayResult with test statistic, p-value, and effect size.
        """
        soc_level = (
            df["socioeconomic_level"].iloc[0]
            if "socioeconomic_level" in df.columns
            else "unknown"
        )
        payday_cash = df.loc[df["is_payday"], "amount_cash"].dropna()
        nonpayday_cash = df.loc[~df["is_payday"], "amount_cash"].dropna()

        if len(payday_cash) < 2 or len(nonpayday_cash) < 2:
            return PaydayResult(0.0, 1.0, 0.0, 0.0, 0.0, soc_level)

        stat, pvalue = stats.mannwhitneyu(payday_cash, nonpayday_cash, alternative="two-sided")
        payday_med = float(payday_cash.median())
        nonpayday_med = float(nonpayday_cash.median())
        # Effect size: relative difference in medians; positive = payday > non-payday
        effect_size = (payday_med - nonpayday_med) / (nonpayday_med + 1e-9)

        return PaydayResult(
            statistic=float(stat),
            pvalue=float(pvalue),
            effect_size=effect_size,
            payday_median=payday_med,
            nonpayday_median=nonpayday_med,
            socioeconomic_level=str(soc_level),
        )
