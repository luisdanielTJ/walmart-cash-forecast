# Walmart Mexico — Daily Cash Change Forecasting

A production-grade system for forecasting how much change (coins + bills) each
Walmart Mexico store should prepare each day, minimising both stockouts and
excess cash held overnight.

## Problem

Stores frequently run out of change mid-shift or hold far too much overnight —
both costly. This system forecasts daily cash demand at the store level and
uses the newsvendor model to compute the optimal cash buffer, then solves an
ILP to determine the exact coin/bill denomination mix.

## Solution Architecture

```
Raw CSVs
  └─ DataLoader → DataValidator
        └─ StoreAggregator (category → store-day)
              └─ CashImputer (store × dow medians)
                    └─ FeatureEngine (lags, rolling, calendar)
                          ├─ StatAnalyzer (distribution, stationarity, STL, payday)
                          ├─ BayesianForecaster (PyMC hierarchical lognormal)
                          ├─ MLForecaster (LightGBM quantile, Optuna-tuned)
                          ├─ ConformalWrapper (split conformal intervals)
                          ├─ ModelBlender (Ridge stacking, softmax weights)
                          ├─ NewsvendorOptimizer (critical fractile q*)
                          └─ DenominationSolver (ILP, PuLP + CBC)
```

**Statistical depth:**
- Distribution fitting (Poisson / NegBin / ZINB) justifies the lognormal likelihood
- ADF + KPSS joint stationarity test on each store's series
- STL decomposition + Ljung-Box residual test (period = 7 days)
- Mann-Whitney U payday effect with effect size
- R-hat and ESS convergence checks on MCMC chains (tiered: warn at 1.01–1.05, fail at > 1.05)
- Split conformal prediction (Angelopoulos & Bates 2023) for valid coverage

## Results on Real Data

Trained on 203,958 transaction rows · 80 stores · 425 calendar days (Jan 2023 – Feb 2024).

| Metric | Value |
|--------|-------|
| Train rows | 28,076 |
| Calibration rows | 4,797 |
| Blend weights (Bayes / ML) | 64.5% / 35.5% |
| Conformal q̂ (90% interval half-width) | 294,462 MXN |
| Distribution best fit | Negative Binomial (AIC 554k vs Poisson 22.5M) |
| NegBin overdispersion α | 0.39 |
| Payday effect (Mann-Whitney U p-value) | < 0.0001 |
| Payday median lift | +33.5% (MXN 425k vs MXN 318k on non-paydays) |
| Stores with stationary series (ADF + KPSS) | 0 / 80 (motivates trend-aware features) |
| STL residuals white noise (Ljung-Box) | No (p ≈ 0, confirms weekly seasonality) |

**March 2024 forecast summary (80 stores × 30 days = 2,400 predictions):**

| | Mean | Min | Max |
|-|------|-----|-----|
| Blended point forecast | 367k MXN | 167k | 1,080k |
| Conformal lower bound (90%) | 96k MXN | 0 | 785k |
| Conformal upper bound (90%) | 662k MXN | 462k | 1,374k |
| Newsvendor q* buffer (75th pct) | 448k MXN | 111k | 1,691k |
| Denomination pieces per store-day | 448 pieces | — | — |

## Quick Start

### Option 1 — Local (uv)

```bash
# Install uv (https://docs.astral.sh/uv/getting-started/installation/)
pip install uv

# Install all dependencies
make install          # equivalent: uv sync --extra dev

# Run the full test suite
make test

# Train on the provided dataset
walmart-forecast train \
  --data-dir Prueba_Tecnica_DS/Prueba_Tecnica_DS/data \
  --model-dir models/v1

# Generate predictions for a CSV of future store-dates
walmart-forecast predict \
  --model-dir models/v1 \
  --future-csv data/future_march2024.csv \
  --stores-csv Prueba_Tecnica_DS/Prueba_Tecnica_DS/data/stores.csv \
  --out data/predictions_march2024.csv

# Start the REST API
walmart-forecast serve --model-dir models/v1 --port 8000
```

### Option 2 — Docker

```bash
docker build -t walmart-forecast .
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  walmart-forecast
```

## API

```
GET  /health            → {"status": "ok"}
POST /predict           → batch store-date forecast + denomination mix
```

Example request:
```json
{
  "observations": [
    {"store_id": "STR_001", "date": "2024-01-15", "day_of_week": 0, "is_payday": true}
  ],
  "stores": [
    {"store_id": "STR_001", "region": "Norte", "store_format": "Supercenter"}
  ]
}
```

Interactive docs: `http://localhost:8000/docs`

## MLflow Experiment Tracking

Every training run is automatically logged to a local MLflow tracking store.

```bash
# After running walmart-forecast train ..., open the UI:
mlflow ui
# → http://localhost:5000
```

Logged per run:

| Category | Items |
|----------|-------|
| **Params** | `n_draws`, `n_tune`, `n_chains`, `target_accept`, `holdout_days`, `random_seed`, cost model, conformal α, Optuna trials |
| **Metrics** | `blend_weight_bayes/ml`, `conformal_q_hat`, `payday_pvalue`, `payday_effect_size`, `pct_stores_stationary`, `negbin_alpha`, AIC scores |
| **Artefacts** | All model artefacts (Bayesian trace, LightGBM boosters, conformal + blender weights, metadata JSON) |

The `mlruns/` directory is created locally and is not committed to git.

## Development

```bash
make lint          # ruff
make type-check    # mypy
make test          # pytest with coverage
make train         # full training pipeline
```

## Project Structure

```
src/walmart_cash_forecast/
  data/           loader, validator
  features/       aggregator, imputer, engineer
  stats/          distribution, stationarity, decomposition, payday, reporter
  models/
    bayesian/     PyMC hierarchical model + diagnostics
    ml/           LightGBM quantile model + Optuna tuning
    conformal.py  Split conformal wrapper
    blender.py    Ridge stacking meta-model
  optimization/
    newsvendor.py Critical fractile optimizer
    denomination.py ILP denomination solver (PuLP + CBC)
  pipelines/      training.py, prediction.py
  api/            FastAPI app
  cli.py          Typer CLI
tests/
  unit/           Per-module fast tests (< 1 min total, excl. Bayesian MCMC)
  integration/    End-to-end pipeline tests (Bayesian mocked)
notebooks/
  01_eda.ipynb    Exploratory data analysis
```

## Key References

- Gelman & Hill (2007) *Data Analysis Using Regression and Multilevel/Hierarchical Models*, Ch. 12–13
- Ke et al. (2017) "LightGBM", NeurIPS
- Angelopoulos & Bates (2023) "A Gentle Introduction to Conformal Prediction", arXiv:2107.07511
- Vehtari et al. (2021) "Rank-normalization, folding, and localization: An improved R-hat", *Bayesian Analysis*
- Arrow et al. (1951); Scarf (1958) — newsvendor critical fractile
- Wolpert (1992) "Stacked generalization", *Neural Networks*
