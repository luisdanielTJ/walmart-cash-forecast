"""End-to-end training pipeline.

Orchestrates: load → validate → aggregate → impute → feature engineer →
statistical analysis → Bayesian fit → ML fit → conformal calibration →
blender calibration → save all artefacts.

The pipeline is intentionally stateless: it reads from data_dir and writes to
model_dir, with no global mutable state so it is safe to run in parallel or
inside a Docker container.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from walmart_cash_forecast.config import Config
from walmart_cash_forecast.data.loader import DataLoader
from walmart_cash_forecast.data.validator import DataValidator
from walmart_cash_forecast.features.aggregator import StoreAggregator
from walmart_cash_forecast.features.engineer import FeatureEngine
from walmart_cash_forecast.features.imputer import CashImputer
from walmart_cash_forecast.models.bayesian.model import BayesianForecaster
from walmart_cash_forecast.models.blender import ModelBlender
from walmart_cash_forecast.models.conformal import ConformalWrapper
from walmart_cash_forecast.models.ml.model import MLForecaster
from walmart_cash_forecast.stats.reporter import StatAnalyzer

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Full training pipeline: data → models → artefacts on disk.

    Args:
        config: Project configuration (random seeds, hyperparameters, etc.).
        data_dir: Directory containing transactions.csv, stores.csv, calendar.csv.
        model_dir: Output directory for trained model artefacts.
    """

    def __init__(self, config: Config, data_dir: Path, model_dir: Path) -> None:
        self.config = config
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)

    def run(self) -> dict:
        """
        Execute the full training pipeline.

        Steps:
          1. Load and validate raw CSVs.
          2. Aggregate category-level rows to store-day level.
          3. Impute missing cash values (store × day-of-week medians).
          4. Run statistical analyses and save report.
          5. Engineer lag / rolling / calendar features.
          6. Chronological train / calibration split (holdout_days config).
          7. Fit BayesianForecaster on train split.
          8. Fit MLForecaster on train split.
          9. Calibrate ConformalWrapper on calibration split residuals (ML q50).
          10. Calibrate ModelBlender on calibration split.
          11. Persist all artefacts to model_dir.

        Returns:
            Dict with split sizes, stat report, and training metadata.
        """
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # --- Step 1: Load + validate ---
        logger.info("Loading data from %s", self.data_dir)
        loader = DataLoader(self.data_dir)
        transactions, stores, calendar = loader.load()

        validator = DataValidator()
        validator.validate_transactions(transactions)
        validator.validate_stores(stores)
        if calendar is not None:
            validator.validate_calendar(calendar)

        # --- Step 2: Aggregate to store-day level ---
        logger.info("Aggregating to store-day level")
        aggregator = StoreAggregator()
        panel = aggregator.aggregate(transactions)

        # Derive day_of_week from date — needed by CashImputer and FeatureEngine
        panel["day_of_week"] = panel["date"].dt.dayofweek
        # Derive is_payday: day 15 and last calendar day of month (quincena)
        panel["is_payday"] = panel["date"].dt.day.isin([15]) | (
            panel["date"] == panel["date"].dt.to_period("M").dt.to_timestamp("M")
        )

        # Merge store metadata and calendar features
        panel = panel.merge(stores[["store_id", "region", "store_format"]], on="store_id", how="left")
        if calendar is not None:
            panel = panel.merge(
                calendar[["date", "is_holiday", "is_buen_fin", "is_navidad_season"]],
                on="date",
                how="left",
            )
        # Fill any unmatched calendar flags with False
        for col in ["is_holiday", "is_buen_fin", "is_navidad_season"]:
            if col not in panel.columns:
                panel[col] = False
            panel[col] = panel[col].fillna(False)

        # --- Step 3: Impute missing cash ---
        logger.info("Imputing missing cash values")
        imputer = CashImputer()
        panel = imputer.fit(panel).transform(panel)

        # --- Step 4: Statistical analysis ---
        logger.info("Running statistical analyses")
        stat_report = StatAnalyzer().run(panel, self.model_dir / "stats")

        # --- Step 5: Feature engineering ---
        logger.info("Engineering features")
        engine = FeatureEngine(self.config)
        panel = engine.fit_transform(panel)
        panel = panel.dropna()  # lag features create NaN for early rows

        # --- Step 6: Chronological split ---
        panel = panel.sort_values(["store_id", "date"])
        cutoff = panel["date"].max() - pd.Timedelta(days=self.config.holdout_days)
        train = panel[panel["date"] <= cutoff].copy()
        calib = panel[panel["date"] > cutoff].copy()
        logger.info("Train: %d rows, Calibration: %d rows", len(train), len(calib))

        # --- Step 7: Bayesian forecaster ---
        logger.info("Fitting BayesianForecaster")
        bayesian = BayesianForecaster(self.config)
        bayesian.fit(train, stores)
        bayesian.save(self.model_dir / "bayesian")

        # --- Step 8: ML forecaster ---
        logger.info("Fitting MLForecaster")
        ml = MLForecaster(self.config)
        ml.fit(train)
        ml.save(self.model_dir / "ml")

        # --- Step 9: Conformal calibration on ML q50 residuals ---
        logger.info("Calibrating ConformalWrapper")
        ml_calib_preds = ml.predict(calib, quantile=0.5)
        y_calib = calib["amount_cash"].to_numpy(dtype=np.float64)
        conformal = ConformalWrapper(alpha=self.config.conformal.alpha)
        conformal.calibrate(y_calib, ml_calib_preds)
        conformal.save(self.model_dir / "conformal")

        # --- Step 10: Blender calibration ---
        logger.info("Calibrating ModelBlender")
        # Bayesian q50: median of posterior samples
        bayes_samples = bayesian.predict(calib)
        bayes_calib_preds = np.median(bayes_samples, axis=0)
        blender = ModelBlender()
        blender.fit(y_calib, bayes_calib_preds, ml_calib_preds)
        blender.save(self.model_dir / "blender")

        # --- Step 11: Save metadata ---
        metadata = {
            "train_rows": len(train),
            "calib_rows": len(calib),
            "train_cutoff": str(cutoff.date()),
            "n_stores": int(panel["store_id"].nunique()),
            "blend_weights": blender.weights.tolist(),
            "conformal_q_hat": conformal.q_hat,
            "stat_report": stat_report,
        }
        (self.model_dir / "training_metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str)
        )
        logger.info("Training complete. Artefacts saved to %s", self.model_dir)
        return metadata
