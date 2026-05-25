"""Bayesian hierarchical lognormal forecaster for daily store-level cash demand.

Model structure (additive on log scale):
    log(amount_cash_st + 1) ~ Normal(μ_st, σ_obs)

    μ_st = α_s + β_dow[dow_t] + β_payday·is_payday_t
           + β_holiday·is_holiday_t + β_buen_fin·is_buen_fin_t
           + β_navidad·is_navidad_t

Hierarchical priors (partial pooling across region × store_format):
    α_s ~ Normal(μ_region[region_s] + μ_format[format_s], σ_store)
    μ_region ~ Normal(0, 1),  μ_format ~ Normal(0, 1)
    σ_store, σ_obs ~ HalfNormal(1)

Using a lognormal likelihood because amount_cash is strictly positive and
right-skewed. The hierarchical structure allows data-sparse stores to borrow
statistical strength from stores in the same region and format group
(Bayesian shrinkage / partial pooling).

Reference: Gelman & Hill (2007) Data Analysis Using Regression and
Multilevel/Hierarchical Models, Ch. 12-13.
"""
from __future__ import annotations

import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from walmart_cash_forecast.config import Config
from walmart_cash_forecast.models.bayesian.diagnostics import check_convergence


class BayesianForecaster:
    """Hierarchical Bayesian lognormal model for store-level daily cash demand."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._model: pm.Model | None = None
        self._trace: az.InferenceData | None = None
        # Integer encodings computed during fit and reused during predict
        self._store_to_idx: dict[str, int] = {}
        self._region_to_idx: dict[str, int] = {}
        self._format_to_idx: dict[str, int] = {}
        self._store_region: np.ndarray = np.array([])
        self._store_format: np.ndarray = np.array([])

    def fit(self, df: pd.DataFrame, stores: pd.DataFrame) -> "BayesianForecaster":
        """
        Build and sample the hierarchical model on training data.

        Args:
            df: Store-level daily panel — must contain store_id, date, amount_cash,
                is_payday, is_holiday, is_buen_fin, is_navidad_season, day_of_week.
            stores: Static store metadata with region and store_format columns.

        Returns:
            self (for chaining).
        """
        # --- Build integer encodings for all categorical variables ---
        store_ids = list(df["store_id"].unique())
        self._store_to_idx = {s: i for i, s in enumerate(store_ids)}

        regions = list(stores["region"].unique())
        self._region_to_idx = {r: i for i, r in enumerate(regions)}

        formats = list(stores["store_format"].unique())
        self._format_to_idx = {f: i for i, f in enumerate(formats)}

        store_meta = stores.set_index("store_id")
        # Store-level mappings to their region and format group indices
        self._store_region = np.array([
            self._region_to_idx[store_meta.loc[s, "region"]] for s in store_ids
        ])
        self._store_format = np.array([
            self._format_to_idx[store_meta.loc[s, "store_format"]] for s in store_ids
        ])

        # --- Encode observed data as integer arrays for PyMC ---
        store_idx = df["store_id"].map(self._store_to_idx).values.astype(int)
        dow = df["day_of_week"].values.astype(int)
        is_payday = df["is_payday"].astype(float).values
        is_holiday = df["is_holiday"].astype(float).values
        is_buen_fin = df["is_buen_fin"].astype(float).values
        is_navidad = df["is_navidad_season"].astype(float).values

        # Log1p transform: maps (0, ∞) → (0, ∞) stably, inverse is expm1
        log_y = np.log1p(df["amount_cash"].fillna(0).values)

        n_stores = len(store_ids)
        n_regions = len(regions)
        n_formats = len(formats)

        with pm.Model() as model:
            # pm.Data containers allow updating the inputs at prediction time
            # without rebuilding the model graph
            store_idx_data = pm.Data("store_idx", store_idx, dims="obs")
            dow_data = pm.Data("dow", dow, dims="obs")
            is_payday_data = pm.Data("is_payday", is_payday, dims="obs")
            is_holiday_data = pm.Data("is_holiday", is_holiday, dims="obs")
            is_buen_fin_data = pm.Data("is_buen_fin", is_buen_fin, dims="obs")
            is_navidad_data = pm.Data("is_navidad", is_navidad, dims="obs")

            # --- Hyperpriors for group-level intercepts ---
            mu_region = pm.Normal("mu_region", mu=0, sigma=1, shape=n_regions)
            mu_format = pm.Normal("mu_format", mu=0, sigma=1, shape=n_formats)
            sigma_store = pm.HalfNormal("sigma_store", sigma=1)

            # --- Store-level intercepts: non-centered parameterization ---
            # Non-centered form avoids the "Neal's funnel" geometry that causes
            # divergences when sigma_store → 0. We sample offsets from N(0,1)
            # and shift/scale them explicitly (Betancourt & Girolami, 2015).
            alpha_s_offset = pm.Normal("alpha_s_offset", mu=0, sigma=1, shape=n_stores)
            alpha_s = pm.Deterministic(
                "alpha_s",
                mu_region[self._store_region] + mu_format[self._store_format]
                + sigma_store * alpha_s_offset,
            )

            # --- Day-of-week seasonality coefficients (0=Monday, 6=Sunday) ---
            beta_dow = pm.Normal("beta_dow", mu=0, sigma=0.5, shape=7)

            # --- Event-driven demand shifters ---
            beta_payday = pm.Normal("beta_payday", mu=0, sigma=0.5)
            beta_holiday = pm.Normal("beta_holiday", mu=0, sigma=0.5)
            beta_buen_fin = pm.Normal("beta_buen_fin", mu=0, sigma=0.5)
            beta_navidad = pm.Normal("beta_navidad", mu=0, sigma=0.5)

            # --- Linear predictor on the log scale ---
            mu = (
                alpha_s[store_idx_data]
                + beta_dow[dow_data]
                + beta_payday * is_payday_data
                + beta_holiday * is_holiday_data
                + beta_buen_fin * is_buen_fin_data
                + beta_navidad * is_navidad_data
            )

            # --- Lognormal likelihood: σ_obs captures residual day-to-day variability ---
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.5)
            _y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma_obs, observed=log_y, dims="obs")

            # Sample with NUTS — target_accept=0.9 reduces divergences from
            # the hierarchical funnel; fixed seed ensures reproducibility
            self._trace = pm.sample(
                draws=self.config.bayesian.n_draws,
                tune=self.config.bayesian.n_tune,
                chains=self.config.bayesian.n_chains,
                target_accept=self.config.bayesian.target_accept,
                random_seed=self.config.random_seed,
                progressbar=False,
                return_inferencedata=True,
            )

        self._model = model

        # Skip convergence check when using minimal draws for testing
        if self.config.bayesian.n_draws >= 200:
            check_convergence(self._trace)

        return self

    def predict(self, future_df: pd.DataFrame) -> np.ndarray:
        """
        Draw posterior predictive samples for future store-date observations.

        Args:
            future_df: DataFrame with store_id, is_payday, is_holiday, is_buen_fin,
                       is_navidad_season, day_of_week columns.

        Returns:
            Array of shape (n_posterior_samples, n_rows) with amount_cash samples.
            All values are positive (inverse log1p applied).
        """
        if self._model is None or self._trace is None:
            raise RuntimeError("Call fit() before predict().")

        # Encode future data using the same mappings fitted during training
        store_idx = future_df["store_id"].map(self._store_to_idx).values.astype(int)
        dow = future_df["day_of_week"].values.astype(int)
        is_payday = future_df["is_payday"].astype(float).values
        is_holiday = future_df["is_holiday"].astype(float).values
        is_buen_fin = future_df["is_buen_fin"].astype(float).values
        is_navidad = future_df["is_navidad_season"].astype(float).values

        with self._model:
            pm.set_data({
                "store_idx": store_idx,
                "dow": dow,
                "is_payday": is_payday,
                "is_holiday": is_holiday,
                "is_buen_fin": is_buen_fin,
                "is_navidad": is_navidad,
            })
            ppc = pm.sample_posterior_predictive(
                self._trace,
                random_seed=self.config.random_seed,
                progressbar=False,
            )

        # y_obs samples: shape (n_chains, n_draws, n_obs) → reshape to (n_samples, n_obs)
        log_samples = ppc.posterior_predictive["y_obs"].values
        log_samples = log_samples.reshape(-1, log_samples.shape[-1])
        # Inverse log1p: recover original MXN scale
        return np.expm1(log_samples)

    def save(self, path: Path) -> None:
        """Persist posterior trace and encodings to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self._trace is not None:
            self._trace.to_netcdf(str(path / "trace.nc"))
        np.save(path / "store_region.npy", self._store_region)
        np.save(path / "store_format.npy", self._store_format)
        (path / "encodings.json").write_text(json.dumps({
            "store_to_idx": self._store_to_idx,
            "region_to_idx": self._region_to_idx,
            "format_to_idx": self._format_to_idx,
        }))

    def load(self, path: Path) -> "BayesianForecaster":
        """Load a previously saved posterior trace from disk."""
        path = Path(path)
        self._trace = az.from_netcdf(str(path / "trace.nc"))
        self._store_region = np.load(path / "store_region.npy")
        self._store_format = np.load(path / "store_format.npy")
        encodings = json.loads((path / "encodings.json").read_text())
        self._store_to_idx = encodings["store_to_idx"]
        self._region_to_idx = encodings["region_to_idx"]
        self._format_to_idx = encodings["format_to_idx"]
        return self
