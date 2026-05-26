"""LightGBM quantile-regression forecaster with Optuna hyperparameter search.

Model: three separate GBDT models (q=0.1, q=0.5, q=0.9) each optimised for
pinball (quantile) loss, giving a calibrated prediction interval alongside the
point estimate.  Optuna finds the best tree structure / regularisation
hyperparameters via time-series cross-validation (expanding window).

Design choices:
- Quantile regression (not standard MSE) avoids symmetric-loss bias: the
  distribution of daily cash is right-skewed, so mean ≠ median.
- Separate models per quantile guarantees that the objective aligns with the
  evaluation metric (no post-hoc quantile estimation).
- LightGBM is chosen over XGBoost for faster training on tabular features and
  native categorical support (not used here, but available).
- Optuna TPE sampler is seeded for reproducibility; n_trials is configurable.
- _models stores raw lgb.Booster objects (not the sklearn wrapper) so that
  save/load works without re-fitting and without injecting private attributes.

Reference: Ke et al. (2017) "LightGBM: A Highly Efficient Gradient Boosting
Decision Tree", NeurIPS.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from walmart_cash_forecast.config import Config

logger = logging.getLogger(__name__)

# Quantile levels: lower, median, upper
QUANTILES = (0.1, 0.5, 0.9)

# LightGBM feature columns — names must match FeatureEngine output exactly.
# Lag/rolling columns use the full FeatureEngine naming convention so the
# training DataFrame and the prediction DataFrame share identical column names.
_FEATURE_COLS = [
    "day_of_week",
    "is_payday",
    "is_holiday",
    "is_buen_fin",
    "is_navidad_season",
    "amount_cash_lag_1",
    "amount_cash_lag_7",
    "amount_cash_lag_14",
    "amount_cash_roll7_mean",
    "amount_cash_roll7_std",
    "amount_cash_roll28_mean",
    "cash_ratio",
    "days_since_payday",
    "days_until_payday",
]

# Target column
_TARGET = "amount_cash"

optuna.logging.set_verbosity(optuna.logging.WARNING)


class MLForecaster:
    """LightGBM quantile forecaster: trains one model per quantile level.

    Attributes:
        config: Global project configuration.
        _models: Mapping from quantile → raw lgb.Booster (extracted via
            model.booster_ after sklearn fit). Raw Booster objects support
            save_model / Booster(model_file=...) round-trips cleanly.
        _feature_cols: Feature column names resolved at fit time (present
            features only — some lag columns may be absent for short series).
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._models: dict[float, lgb.Booster] = {}
        self._feature_cols: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "MLForecaster":
        """
        Tune and train one LightGBM model per quantile on the full training set.

        Args:
            df: Feature-engineered store-level daily panel.  Must contain
                all columns listed in _FEATURE_COLS (or a subset if lag
                columns were not created) plus amount_cash as the target.

        Returns:
            self (for chaining).
        """
        # Resolve available features — lag columns may be absent for very short
        # series passed during unit tests
        self._feature_cols = [c for c in _FEATURE_COLS if c in df.columns]
        x_mat: npt.NDArray[np.float32] = df[self._feature_cols].to_numpy(dtype=np.float32)
        y: npt.NDArray[np.float32] = df[_TARGET].to_numpy(dtype=np.float32)

        for q in QUANTILES:
            logger.info(
                "Tuning q=%.2f  (%d Optuna trials, %d rows)",
                q, self.config.ml.n_optuna_trials, len(y),
            )
            best_params = self._tune(x_mat, y, q)
            logger.info("Fitting final q=%.2f model with best params", q)
            reg = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                random_state=self.config.random_seed,
                n_jobs=-1,
                verbose=-1,
                **best_params,
            )
            reg.fit(x_mat, y)
            # Store the underlying Booster — avoids sklearn fitted-state issues on load
            self._models[q] = reg.booster_

        return self

    def predict(self, future_df: pd.DataFrame, quantile: float = 0.5) -> npt.NDArray[np.float64]:
        """
        Return point or quantile predictions for future observations.

        Args:
            future_df: Feature-engineered DataFrame with the same columns used
                during training.
            quantile: Which quantile model to use (0.1, 0.5, or 0.9).

        Returns:
            1-D array of length len(future_df) with non-negative predictions.
        """
        if quantile not in self._models:
            raise ValueError(f"No model for quantile={quantile}. Choose from {list(self._models)}")

        x_mat: npt.NDArray[np.float32] = future_df[self._feature_cols].to_numpy(dtype=np.float32)
        # np.asarray ensures a dense ndarray regardless of LightGBM's return type union
        preds = np.asarray(self._models[quantile].predict(x_mat), dtype=np.float64)
        # Quantile regression can theoretically predict negative values; clip
        # to zero since cash demand is strictly non-negative
        return np.maximum(preds, 0.0)

    def predict_interval(self, future_df: pd.DataFrame) -> pd.DataFrame:
        """
        Return all three quantile predictions as a DataFrame.

        Returns:
            DataFrame with columns q10, q50, q90 and the same index as future_df.
        """
        return pd.DataFrame(
            {
                "q10": self.predict(future_df, 0.1),
                "q50": self.predict(future_df, 0.5),
                "q90": self.predict(future_df, 0.9),
            },
            index=future_df.index,
        )

    def save(self, path: Path) -> None:
        """Persist all quantile models to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for q, booster in self._models.items():
            # LightGBM text format is human-readable and version-stable
            booster.save_model(str(path / f"lgbm_q{int(q * 100):02d}.txt"))
        (path / "feature_cols.txt").write_text("\n".join(self._feature_cols))

    def load(self, path: Path) -> "MLForecaster":
        """Restore quantile models saved with save()."""
        path = Path(path)
        self._feature_cols = (path / "feature_cols.txt").read_text().splitlines()
        for q in QUANTILES:
            self._models[q] = lgb.Booster(model_file=str(path / f"lgbm_q{int(q * 100):02d}.txt"))
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tune(
        self,
        x_mat: npt.NDArray[np.float32],
        y: npt.NDArray[np.float32],
        q: float,
    ) -> dict:
        """
        Run Optuna TPE search for LightGBM hyperparameters.

        Uses expanding-window time-series CV (not random k-fold) to avoid
        data leakage: earlier folds train on past data and validate on future.

        Args:
            x_mat: Feature matrix, shape (n, p).
            y: Target vector, shape (n,).
            q: Quantile level for this model.

        Returns:
            Dict of LightGBM constructor kwargs for the best trial.
        """
        tscv = TimeSeriesSplit(n_splits=min(3, max(len(y) // 20, 1)))

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            }
            pinball_losses = []
            for train_idx, val_idx in tscv.split(x_mat):
                model = lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=q,
                    random_state=self.config.random_seed,
                    n_jobs=-1,
                    verbose=-1,
                    **params,  # type: ignore[arg-type]
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(x_mat[train_idx], y[train_idx])
                    preds: npt.NDArray[np.float64] = model.predict(x_mat[val_idx])
                # Pinball loss: L_q(y, ŷ) = q·max(y-ŷ, 0) + (1-q)·max(ŷ-y, 0)
                errors: npt.NDArray[np.float64] = y[val_idx].astype(np.float64) - preds
                loss = float(np.mean(np.where(errors >= 0, q * errors, (q - 1.0) * errors)))
                pinball_losses.append(loss)
            return float(np.mean(pinball_losses))

        sampler = optuna.samplers.TPESampler(seed=self.config.random_seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=self.config.ml.n_optuna_trials, show_progress_bar=True)
        return study.best_params
