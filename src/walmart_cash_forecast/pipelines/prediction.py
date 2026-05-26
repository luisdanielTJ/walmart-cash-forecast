"""Prediction pipeline.

Loads all trained artefacts and produces the full output table:
  - Blended point forecast (MXN)
  - Conformal prediction interval [lower, upper]
  - Newsvendor optimal cash buffer q* (MXN)
  - Denomination mix (one column per denomination)

The output is a single wide DataFrame ready for downstream consumption
(API response, CSV export, dashboard).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from walmart_cash_forecast.config import Config
from walmart_cash_forecast.models.bayesian.model import BayesianForecaster
from walmart_cash_forecast.models.blender import ModelBlender
from walmart_cash_forecast.models.conformal import ConformalWrapper
from walmart_cash_forecast.models.ml.model import MLForecaster
from walmart_cash_forecast.optimization.denomination import DenominationSolver
from walmart_cash_forecast.optimization.newsvendor import NewsvendorOptimizer

logger = logging.getLogger(__name__)


class PredictionPipeline:
    """Loads trained artefacts and generates store-level daily cash recommendations.

    Args:
        config: Project configuration.
        model_dir: Directory containing training artefacts (from TrainingPipeline).
    """

    def __init__(self, config: Config, model_dir: Path) -> None:
        self.config = config
        self.model_dir = Path(model_dir)
        self._loaded = False

        # Artefacts — populated by load()
        self._bayesian: BayesianForecaster | None = None
        self._ml: MLForecaster | None = None
        self._conformal: ConformalWrapper | None = None
        self._blender: ModelBlender | None = None
        self._newsvendor: NewsvendorOptimizer | None = None
        self._denomination: DenominationSolver | None = None

    def load(self) -> "PredictionPipeline":
        """Load all trained artefacts from model_dir."""
        self._bayesian = BayesianForecaster(self.config)
        self._bayesian.load(self.model_dir / "bayesian")

        self._ml = MLForecaster(self.config)
        self._ml.load(self.model_dir / "ml")

        self._conformal = ConformalWrapper(alpha=self.config.conformal.alpha)
        self._conformal.load(self.model_dir / "conformal")

        self._blender = ModelBlender()
        self._blender.load(self.model_dir / "blender")

        self._newsvendor = NewsvendorOptimizer(
            cost_underage=self.config.newsvendor.cost_underage,
            cost_overage=self.config.newsvendor.cost_overage,
        )
        self._denomination = DenominationSolver(
            limits=self.config.denomination_limits or None
        )
        self._loaded = True
        return self

    def predict(self, future_df: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
        """
        Generate cash recommendations for a set of future store-dates.

        Args:
            future_df: Feature-engineered DataFrame with store_id, date, and all
                feature columns (must have been processed by FeatureEngine).
            stores: Store metadata with store_id, region, store_format.

        Returns:
            Wide DataFrame indexed by (store_id, date) with columns:
                - forecast_blend: Blended point forecast (MXN)
                - lower, upper: 90% conformal prediction interval
                - q_star: Newsvendor optimal buffer (MXN)
                - denom_*: Piece counts per denomination
        """
        if not self._loaded:
            raise RuntimeError("Call load() before predict().")

        assert self._bayesian and self._ml and self._conformal
        assert self._blender and self._newsvendor and self._denomination

        # --- Point forecasts from both models ---
        logger.info("Generating Bayesian posterior predictive samples")
        bayes_samples = self._bayesian.predict(future_df)  # (n_samples, n_rows)
        bayes_pred = np.median(bayes_samples, axis=0)

        logger.info("Generating ML forecasts")
        ml_pred = self._ml.predict(future_df, quantile=0.5)

        # --- Blended point forecast ---
        blend_pred = self._blender.predict(bayes_pred, ml_pred)

        # --- Conformal prediction interval ---
        interval_df = self._conformal.predict_interval(blend_pred)

        # --- Newsvendor optimal buffer ---
        # Use full Bayesian posterior samples for the demand distribution:
        # these provide a richer uncertainty estimate than a single quantile.
        q_stars = []
        for i, (_, row) in enumerate(future_df.iterrows()):
            samples_i = bayes_samples[:, i]
            result = self._newsvendor.optimise(
                store_id=str(row["store_id"]),
                date=pd.Timestamp(row["date"]),
                demand_samples=samples_i.astype(np.float64),
            )
            q_stars.append(result.q_star)

        # --- Denomination mix ---
        # The denomination mix covers the registers' change fund, not the full
        # cash buffer (which sits mostly in the safe). Cap the ILP target so
        # bill counts stay operationally realistic.
        fund_cap = self.config.denomination_fund_cap
        targets = pd.DataFrame({
            "store_id": future_df["store_id"].values,
            "date": future_df["date"].values,
            "q_star": [min(q, fund_cap) for q in q_stars],
        })
        denom_results = self._denomination.solve_batch(targets)
        denom_df = self._denomination.to_dataframe(denom_results)

        # --- Assemble output ---
        out = future_df[["store_id", "date"]].copy().reset_index(drop=True)
        out["forecast_blend"] = blend_pred
        out["lower"] = interval_df["lower"].values
        out["upper"] = interval_df["upper"].values
        out["q_star"] = q_stars
        denom_cols = [c for c in denom_df.columns if c.startswith("denom_")]
        denom_df = denom_df.reset_index(drop=True)
        out[denom_cols] = denom_df[denom_cols]

        return out

    @property
    def metadata(self) -> dict:
        """Load training metadata from model_dir (for logging / audit)."""
        meta_path = self.model_dir / "training_metadata.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text())
        return {}
