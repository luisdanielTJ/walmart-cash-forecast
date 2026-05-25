"""Ridge stacking meta-model: blends Bayesian and ML point predictions.

Stacking (Wolpert 1992) trains a second-level model on out-of-fold predictions
from the base models.  Here we use Ridge regression (L2-penalised OLS) as the
meta-learner:

    ŷ_blend = w_bayes · ŷ_bayes + w_ml · ŷ_ml

where the weights (w_bayes, w_ml) are fit by Ridge regression on the
calibration split.  To ensure the blended prediction lies in the convex hull
of the two base predictions (a natural constraint for positive quantities), the
weights are softmax-normalised before returning.

Design rationale:
- Bayesian model: stronger on stores with sparse data (shrinkage); intervals
  are posterior samples rather than point estimates.
- ML model: stronger when feature interactions dominate; faster to retrain.
- Ridge meta-learner avoids overfitting the blend weights on small calibration
  sets; its alpha hyperparameter provides implicit regularisation.
- Softmax normalisation ensures both weights are positive and sum to 1, so the
  blended prediction is a weighted average rather than an extrapolation.

Reference: Wolpert, D.H. (1992) "Stacked generalization." Neural Networks 5(2).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import Ridge


class ModelBlender:
    """Ridge stacking meta-model that combines Bayesian + ML point predictions.

    Attributes:
        ridge_alpha: L2 regularisation strength for the Ridge meta-learner.
        _weights: Softmax-normalised blend weights [w_bayes, w_ml] after fit.
        _ridge: Fitted Ridge estimator (kept for diagnostics).
    """

    def __init__(self, ridge_alpha: float = 1.0) -> None:
        self.ridge_alpha = ridge_alpha
        self._weights: npt.NDArray[np.float64] | None = None
        self._ridge: Ridge | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        y_true: npt.NDArray[np.float64],
        y_bayes: npt.NDArray[np.float64],
        y_ml: npt.NDArray[np.float64],
    ) -> "ModelBlender":
        """
        Learn blend weights from calibration-split predictions.

        Args:
            y_true: Observed cash amounts on the calibration split, shape (n,).
            y_bayes: Bayesian model point predictions (posterior median), shape (n,).
            y_ml: ML model point predictions (q50), shape (n,).

        Returns:
            self (for chaining).
        """
        X = np.column_stack([y_bayes, y_ml])
        ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        ridge.fit(X, y_true)
        self._ridge = ridge

        # Softmax-normalise the raw coefficients so weights are ≥ 0 and sum to 1.
        # This keeps blended predictions as a convex combination, which is
        # appropriate for positive-valued targets (no sign flip possible).
        raw: npt.NDArray[np.float64] = ridge.coef_.astype(np.float64)
        exp_w = np.exp(raw - raw.max())
        self._weights = exp_w / exp_w.sum()
        return self

    def predict(
        self,
        y_bayes: npt.NDArray[np.float64] | pd.Series,
        y_ml: npt.NDArray[np.float64] | pd.Series,
    ) -> npt.NDArray[np.float64]:
        """
        Blend base-model predictions using the learned weights.

        Args:
            y_bayes: Bayesian posterior median predictions, shape (n,).
            y_ml: ML q50 predictions, shape (n,).

        Returns:
            Blended predictions, shape (n,), all ≥ 0.

        Raises:
            RuntimeError: If fit() has not been called first.
        """
        if self._weights is None:
            raise RuntimeError("Call fit() before predict().")

        b = np.asarray(y_bayes, dtype=np.float64)
        m = np.asarray(y_ml, dtype=np.float64)
        blended = self._weights[0] * b + self._weights[1] * m
        return np.maximum(blended, 0.0)

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Softmax-normalised blend weights [w_bayes, w_ml]."""
        if self._weights is None:
            return np.array([0.5, 0.5])
        return self._weights

    def save(self, path: Path) -> None:
        """Persist blend weights to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self._weights is not None:
            np.save(path / "blend_weights.npy", self._weights)

    def load(self, path: Path) -> "ModelBlender":
        """Restore blend weights saved with save()."""
        self._weights = np.load(Path(path) / "blend_weights.npy")
        return self
