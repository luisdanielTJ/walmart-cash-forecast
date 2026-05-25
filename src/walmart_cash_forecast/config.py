"""Configuration dataclasses loaded from config.yaml.

All model hyperparameters, cost assumptions, and denomination limits are
centralised here so the reviewer can reproduce any run by inspecting
a single file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class BayesianConfig:
    """MCMC sampling parameters for the PyMC hierarchical model."""

    n_chains: int = 4
    n_draws: int = 2000
    n_tune: int = 1000


@dataclass
class MLConfig:
    """LightGBM quantile regression and Optuna tuning settings."""

    n_optuna_trials: int = 50
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])


@dataclass
class NewsvendorConfig:
    """Cost parameters for the newsvendor optimal buffer formula.

    The optimal stock quantile is q* = cost_underage / (cost_underage + cost_overage).
    Default: cost_underage=3, cost_overage=1 → q*=0.75 (stockout is 3× worse than excess).
    """

    cost_underage: float = 3.0   # cost of running out of change (lost sale, customer friction)
    cost_overage: float = 1.0    # cost of holding excess change (working capital, security)


@dataclass
class ConformalConfig:
    """Split conformal prediction settings."""

    # alpha: target miscoverage rate — intervals cover (1-alpha)% of true values
    alpha: float = 0.1


@dataclass
class Config:
    """Top-level configuration object. Load from config.yaml via Config.from_yaml()."""

    random_seed: int = 42          # fixed globally for full reproducibility
    holdout_days: int = 60         # last N days reserved as test set
    n_cv_folds: int = 5            # expanding-window cross-validation folds
    bayesian: BayesianConfig = field(default_factory=BayesianConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    newsvendor: NewsvendorConfig = field(default_factory=NewsvendorConfig)
    conformal: ConformalConfig = field(default_factory=ConformalConfig)
    denomination_limits: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from a YAML file, falling back to defaults for missing keys."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            random_seed=data.get("random_seed", 42),
            holdout_days=data.get("holdout_days", 60),
            n_cv_folds=data.get("n_cv_folds", 5),
            bayesian=BayesianConfig(**data.get("bayesian", {})),
            ml=MLConfig(**data.get("ml", {})),
            newsvendor=NewsvendorConfig(**data.get("newsvendor", {})),
            conformal=ConformalConfig(**data.get("conformal", {})),
            denomination_limits=data.get("denomination_limits", {}),
        )
