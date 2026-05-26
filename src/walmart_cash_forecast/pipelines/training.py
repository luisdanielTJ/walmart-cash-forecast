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

import mlflow
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

        # Use SQLite so the tracking store is a single portable file.
        # The filesystem backend (mlruns/) is deprecated in MLflow >= 2.20.
        db_path = (self.model_dir.parent / "mlflow.db").resolve()
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        mlflow.set_experiment("walmart-cash-forecast")
        with mlflow.start_run():
            return self._run()

    def _run(self) -> dict:
        """Inner implementation — called inside an active MLflow run."""
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

        # Derive day_of_week — always available from date
        panel["day_of_week"] = panel["date"].dt.dayofweek

        # Merge store metadata
        panel = panel.merge(
            stores[["store_id", "region", "store_format"]], on="store_id", how="left"
        )

        # Merge calendar features — prefer calendar's is_payday over naive formula
        # because the calendar already handles edge cases (e.g. payday on weekend)
        cal_cols = ["date", "is_holiday", "is_buen_fin", "is_navidad_season"]
        if calendar is not None:
            if "is_payday" in calendar.columns:
                cal_cols.append("is_payday")
            panel = panel.merge(calendar[cal_cols], on="date", how="left")

        # Fallback: derive is_payday from date when calendar is absent or doesn't cover all dates
        if "is_payday" not in panel.columns or panel["is_payday"].isna().any():
            derived = panel["date"].dt.day.isin([15]) | (
                panel["date"] == panel["date"].dt.to_period("M").dt.to_timestamp("M")
            )
            if "is_payday" in panel.columns:
                panel["is_payday"] = panel["is_payday"].fillna(derived)
            else:
                panel["is_payday"] = derived

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

        # --- MLflow: log params, metrics, and artefact directory ---
        mlflow.log_params({
            # Sampling
            "n_draws": self.config.bayesian.n_draws,
            "n_tune": self.config.bayesian.n_tune,
            "n_chains": self.config.bayesian.n_chains,
            "target_accept": self.config.bayesian.target_accept,
            # Data split
            "holdout_days": self.config.holdout_days,
            "random_seed": self.config.random_seed,
            # Cost model
            "cost_underage": self.config.newsvendor.cost_underage,
            "cost_overage": self.config.newsvendor.cost_overage,
            # Conformal
            "conformal_alpha": self.config.conformal.alpha,
            # Optuna
            "n_optuna_trials": self.config.ml.n_optuna_trials,
        })
        mlflow.log_metrics({
            "train_rows": metadata["train_rows"],
            "calib_rows": metadata["calib_rows"],
            "n_stores": metadata["n_stores"],
            "blend_weight_bayes": blender.weights[0],
            "blend_weight_ml": blender.weights[1],
            "conformal_q_hat": conformal.q_hat,
            # Statistical analysis findings
            "payday_pvalue": stat_report["payday_effect"]["pvalue"],
            "payday_effect_size": stat_report["payday_effect"]["effect_size"],
            "pct_stores_stationary": stat_report["stationarity"]["pct_stores_stationary"],
            "negbin_alpha": stat_report["distribution"]["negbin_alpha"],
            "aic_negbin": stat_report["distribution"]["aic_negbin"],
            "aic_poisson": stat_report["distribution"]["aic_poisson"],
        })
        mlflow.log_param("dist_best_model", stat_report["distribution"]["best_model"])
        mlflow.log_artifacts(str(self.model_dir), artifact_path="model_artefacts")

        # Register in Model Registry so the Models tab shows a versioned entry
        run_id = mlflow.active_run().info.run_id  # type: ignore[union-attr]
        mlflow.register_model(f"runs:/{run_id}/model_artefacts", "walmart-cash-forecast")

        logger.info("Training complete. Artefacts saved to %s", self.model_dir)
        return metadata
