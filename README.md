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
- R-hat < 1.01 and ESS > 400 convergence checks on MCMC chains
- Split conformal prediction (Angelopoulos & Bates 2023) for valid coverage

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
  --data-dir Prueba_Tecnica_DS \
  --model-dir models/v1

# Generate predictions for a CSV of future store-dates
walmart-forecast predict \
  --model-dir models/v1 \
  --future-csv data/future_features.csv \
  --stores-csv Prueba_Tecnica_DS/stores.csv \
  --out predictions.csv

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
