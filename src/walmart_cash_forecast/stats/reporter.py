"""StatAnalyzer: orchestrates all statistical analyses and saves a report.

This layer runs before model training and produces evidence that justifies
every modeling decision downstream. The JSON report and plots are reviewer-
facing artifacts — they are not consumed by the prediction pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from walmart_cash_forecast.stats.decomposition import SeriesDecomposer
from walmart_cash_forecast.stats.distribution import DistributionFitter
from walmart_cash_forecast.stats.payday import PaydayEffectTester
from walmart_cash_forecast.stats.stationarity import StationarityTester


class StatAnalyzer:
    """Runs all statistical analyses and persists results to output_dir."""

    def run(self, df: pd.DataFrame, output_dir: Path) -> dict:
        """
        Execute all statistical analyses on the store-level daily panel.

        Args:
            df: Store-level daily panel with amount_cash, cash_transactions,
                is_payday, day_of_week, socioeconomic_level columns.
            output_dir: Directory to save stats_summary.json.

        Returns:
            Dictionary of results (also saved to disk as JSON).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report: dict = {}

        # --- 1. Distribution fitting on cash_transactions ---
        # Justifies the Negative Binomial likelihood used in the Bayesian model
        fitter = DistributionFitter()
        counts: np.ndarray = np.asarray(df["cash_transactions"].dropna().astype(int).values)
        dist_result = fitter.fit(counts)
        report["distribution"] = {
            "best_model": dist_result.best_model,
            "aic_poisson": round(dist_result.aic_poisson, 2),
            "aic_negbin": round(dist_result.aic_negbin, 2),
            "is_overdispersed": dist_result.is_overdispersed,
            "negbin_alpha": round(dist_result.negbin_alpha, 4),
        }

        # --- 2. Stationarity: test each store's amount_cash series ---
        tester = StationarityTester()
        stationarity_results: dict = {}
        for store_id, group in df.groupby("store_id"):
            series = (
                group.set_index("date")["amount_cash"]
                if "date" in group.columns
                else group["amount_cash"]
            )
            result = tester.test(series)
            stationarity_results[str(store_id)] = {
                "adf_pvalue": round(result.adf_pvalue, 4),
                "kpss_pvalue": round(result.kpss_pvalue, 4),
                "is_stationary": result.is_stationary,
            }
        pct_stationary = float(
            np.mean([v["is_stationary"] for v in stationarity_results.values()])
        )
        report["stationarity"] = {
            "pct_stores_stationary": round(pct_stationary, 3),
            "per_store": stationarity_results,
        }

        # --- 3. STL decomposition on one representative store ---
        first_store = df["store_id"].iloc[0]
        store_series = (
            df[df["store_id"] == first_store]
            .sort_values("date")["amount_cash"]
            if "date" in df.columns
            else df[df["store_id"] == first_store]["amount_cash"]
        )
        decomposer = SeriesDecomposer(period=7)
        decomp = decomposer.decompose(store_series.reset_index(drop=True))
        report["decomposition"] = {
            "example_store": str(first_store),
            "ljungbox_pvalue": round(decomp.ljungbox_pvalue, 4),
            "residuals_are_white_noise": decomp.residuals_are_white_noise,
        }

        # --- 4. Payday effect (Mann-Whitney U) ---
        payday_tester = PaydayEffectTester()
        payday_result = payday_tester.test(df)
        report["payday_effect"] = {
            "pvalue": round(payday_result.pvalue, 4),
            "effect_size": round(payday_result.effect_size, 4),
            "payday_median": round(payday_result.payday_median, 2),
            "nonpayday_median": round(payday_result.nonpayday_median, 2),
        }

        # Persist the full report to disk
        report_path = output_dir / "stats_summary.json"
        report_path.write_text(json.dumps(report, indent=2, default=str))

        return report
