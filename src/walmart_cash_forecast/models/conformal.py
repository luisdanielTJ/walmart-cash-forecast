"""Split conformal prediction wrapper for distribution-free coverage guarantees.

Split conformal prediction (Papadopoulos et al. 2002; Angelopoulos & Bates 2023)
works by:
  1. Calibrating on a held-out set: compute nonconformity scores
     s_i = |y_i − ŷ_i| for each calibration point i = 1 … n.
  2. Computing the (1−α) quantile of those scores with finite-sample correction:
     q̂ = the ⌈(1−α)(1 + 1/n)⌉-th order statistic of {s_i}.
  3. At prediction time: return [ŷ − q̂, ŷ + q̂] (or [0, ŷ + q̂] for positive targets).

Guarantee: regardless of the true distribution, at least 1−α of future
intervals will cover the true value in marginal probability.

Key advantage over parametric intervals (e.g., Normal ± 2σ): no distributional
assumptions — valid even when the residuals are non-Gaussian.

Reference: Angelopoulos, A.N. & Bates, S. (2023) "A Gentle Introduction to
Conformal Prediction and Distribution-Free Uncertainty Quantification."
arXiv:2107.07511.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd


class ConformalWrapper:
    """Adds distribution-free prediction intervals to any point-forecast model.

    This is a post-hoc calibration layer: the underlying model is already
    trained before ConformalWrapper is instantiated; calibration only requires
    a forward pass to obtain residuals on the calibration split.

    Attributes:
        alpha: Miscoverage level. The resulting intervals cover ≥ 1−alpha of
            future observations in expectation.
        _q_hat: Calibrated nonconformity quantile (MXN).  None until calibrate()
            is called.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self._q_hat: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate(
        self,
        y_true: npt.NDArray[np.float64],
        y_pred: npt.NDArray[np.float64],
    ) -> "ConformalWrapper":
        """
        Compute and store the conformal quantile from calibration residuals.

        Args:
            y_true: Observed cash amounts on the calibration split, shape (n,).
            y_pred: Point predictions for the same rows, shape (n,).

        Returns:
            self (for chaining).

        Raises:
            ValueError: If calibration set is empty.
        """
        n = len(y_true)
        if n == 0:
            raise ValueError("Calibration set is empty.")

        # Absolute residuals are the nonconformity scores for regression
        scores: npt.NDArray[np.float64] = np.abs(y_true - y_pred)

        # Finite-sample corrected quantile: use the ceil((1-alpha)(n+1))-th order
        # statistic.  Equivalently, the quantile level is ceil((1-alpha)(n+1))/n.
        # This is the standard conformal guarantee from Angelopoulos & Bates (2023).
        level = min(float(np.ceil((1 - self.alpha) * (n + 1))) / n, 1.0)
        self._q_hat = float(np.quantile(scores, level))
        return self

    def predict_interval(
        self,
        y_pred: npt.NDArray[np.float64] | pd.Series,
    ) -> pd.DataFrame:
        """
        Wrap point predictions with the calibrated conformal margin.

        Args:
            y_pred: Point predictions from the underlying model, shape (n,).

        Returns:
            DataFrame with columns lower and upper (both ≥ 0).

        Raises:
            RuntimeError: If calibrate() has not been called first.
        """
        if self._q_hat is None:
            raise RuntimeError("Call calibrate() before predict_interval().")

        preds = np.asarray(y_pred, dtype=np.float64)
        # Lower bound clipped to 0 — cash demand cannot be negative
        lower = np.maximum(preds - self._q_hat, 0.0)
        upper = preds + self._q_hat
        return pd.DataFrame({"lower": lower, "upper": upper})

    @property
    def q_hat(self) -> float:
        """Calibrated margin in MXN (NaN until calibrate() is called)."""
        return self._q_hat if self._q_hat is not None else float("nan")

    def save(self, path: Path) -> None:
        """Persist the calibrated quantile to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "q_hat.npy", np.array([self._q_hat if self._q_hat is not None else np.nan]))

    def load(self, path: Path) -> "ConformalWrapper":
        """Restore a previously calibrated conformal wrapper."""
        arr = np.load(Path(path) / "q_hat.npy")
        value = float(arr[0])
        self._q_hat = None if np.isnan(value) else value
        return self
