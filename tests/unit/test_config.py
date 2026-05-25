"""Tests for the Config module."""
from pathlib import Path

from walmart_cash_forecast.config import Config


def test_config_loads_from_yaml(tmp_path):
    yaml_content = """
random_seed: 99
holdout_days: 30
n_cv_folds: 3
bayesian:
  n_chains: 2
  n_draws: 100
  n_tune: 50
ml:
  n_optuna_trials: 5
  quantiles: [0.1, 0.5, 0.9]
newsvendor:
  cost_underage: 3.0
  cost_overage: 1.0
conformal:
  alpha: 0.1
denomination_limits: {}
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)
    cfg = Config.from_yaml(config_file)
    assert cfg.random_seed == 99
    assert cfg.holdout_days == 30
    assert cfg.bayesian.n_chains == 2
    assert cfg.newsvendor.cost_underage == 3.0


def test_config_defaults():
    cfg = Config()
    assert cfg.random_seed == 42
    assert cfg.holdout_days == 60
