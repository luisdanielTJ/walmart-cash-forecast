# Process — Walmart Mexico Cash Change Forecasting

This document records the reasoning, decisions, and tradeoffs made during the
development of the cash-change forecasting system.

---

## 1. Problem Framing

**Business pain point** (confirmed by Walmart Mexico Director of Data Science):
stores either run out of change mid-shift, halting transactions, or hold excess
cash overnight creating a security and capital inefficiency risk.

**Formalisation**: This is a **newsvendor problem** — a single-period inventory
problem where the demand distribution is uncertain. The optimal stocking level
(cash buffer) q* satisfies the critical-fractile condition:

    q* = F⁻¹(C_u / (C_u + C_o))

where C_u = underage cost, C_o = overage cost, and F is the demand CDF.
With C_u = 3, C_o = 1 (assumed 3× cost of stockout vs. excess), the optimal
service level is CR = 0.75.

---

## 2. Data Understanding

### 2.1 Grain and aggregation

The raw data is at the **store × category × day** level. However, cashiers share
one float per store — not per category — so cash management is a store-level
decision. We aggregate to **store × day** before any modelling.

### 2.2 Distribution of cash demand

Fitted Poisson, Negative Binomial, and Zero-Inflated NB to the cash transaction
counts per store-day. The Negative Binomial dominated in AIC (mean variance
ratio >> 1 across stores), which is consistent with retail over-dispersion
from burst-event days (paydays, holidays).

### 2.3 Stationarity

ADF + KPSS joint test (opposite null hypotheses give a definitive verdict):
- 0% of store series are stationary in levels — all 80 stores show some
  non-stationarity (mild trend or structural shift over 425 days).
- STL trend component is small relative to seasonal amplitude, so differencing
  is not necessary.
- Decision: model in levels; hierarchical intercepts in the Bayesian model
  absorb store-level baselines; lag and rolling features in the ML model
  capture local trend implicitly.

### 2.4 Weekly seasonality

STL decomposition (period = 7) shows a strong day-of-week pattern (weekends
≈ 40% higher cash demand). Ljung-Box on STL residuals confirms residual
autocorrelation is near white noise after extracting the weekly component.

### 2.5 Payday (quincena) effect

Mann-Whitney U test (non-parametric, appropriate for non-Normal distributions):
- p < 0.001 across all stores
- Effect size = (payday_median − nonpayday_median) / nonpayday_median ≈ 0.35–0.60

This justified adding `is_payday` as a model feature and a dedicated beta
coefficient in the Bayesian model.

---

## 3. Modeling Decisions

### 3.1 Bayesian hierarchical lognormal model

**Why Bayesian?**
- Provides full posterior predictive distribution — exactly what the newsvendor
  needs; we can compute q* from posterior samples without approximation.
- Partial pooling (hierarchical priors) allows data-sparse stores to borrow
  statistical strength from stores in the same region × format group. This is
  critical for new stores or stores with short histories.
- Native uncertainty quantification; no need for a separate conformal wrapper
  for the Bayesian output.

**Model structure:**
```
log(amount_cash_st + 1) ~ Normal(μ_st, σ_obs)

μ_st = α_s + β_dow[dow_t] + β_payday·is_payday_t
       + β_holiday·is_holiday_t + β_buen_fin·is_buen_fin_t + β_navidad·is_navidad_t

α_s ~ Normal(μ_region[r_s] + μ_format[f_s], σ_store)
```

The log1p transform maps the right-skewed positive-valued target to the real
line while handling zero observations (no log(0) issues).

**Convergence:** R-hat < 1.01 and ESS > 400 enforced programmatically after
sampling (skipped for n_draws < 200 in unit tests for speed).

### 3.2 LightGBM quantile regression

**Why LightGBM?**
- Captures complex feature interactions (lag × day-of-week, lag × payday)
  that the additive Bayesian model does not model explicitly.
- Quantile regression (separate models for q=0.1, 0.5, 0.9) provides
  asymmetric uncertainty — better calibrated than Normal intervals on
  right-skewed data.
- Optuna TPE search with expanding-window time-series CV prevents leakage.

### 3.3 Stacking (Ridge blender)

A Ridge regression meta-learner fits on calibration-split predictions from
both models. Softmax normalisation of the Ridge coefficients ensures non-negative
weights summing to 1 (convex combination). In practice the blender upweights
whichever model generalised better on the specific calibration split.

### 3.4 Conformal prediction intervals

Split conformal prediction (Angelopoulos & Bates 2023) wraps the blended point
forecast with distribution-free intervals that guarantee ≥ 90% marginal coverage
regardless of the true demand distribution — no normality assumption required.

The calibrated margin q̂ = the ⌈(1−α)(n+1)⌉-th order statistic of calibration
residuals. This is added/subtracted from the point forecast at inference time.

---

## 4. Optimization Layer

### 4.1 Newsvendor optimal buffer

Given the posterior predictive samples from the Bayesian model, we estimate
the demand CDF empirically and apply the critical-fractile formula. Using full
posterior samples (not just the point estimate) propagates forecasting
uncertainty into the stocking decision — a key advantage of Bayesian modelling.

### 4.2 Denomination Mix (Proportional Allocator)

Once q* is computed, the change fund for the registers is sized at **3% of the
daily forecast** (industry standard for retail change fund sizing). That fund is
distributed across denominations using circulation shares from the **Banco de
México 2023 Annual Report on Banknote and Coin Circulation**:

| Denomination | Share | Rationale |
|---|---|---|
| $200 | 42% | Most circulated bill in MX retail |
| $100 | 33% | Second most circulated |
| $50  | 13% | Mid-size change |
| $20  | 8%  | Petty change |
| $500 | 4%  | Occasional large-purchase change |
| Coins ($0.10–$10) | Fixed counts | Operational minimum per register, not volume-driven |
| $1,000 | 0% | Not stocked — most MX retailers refuse them |

**Why not an ILP?** An ILP minimising total piece count always selects the
largest available denomination (all $200 bills), which is mathematically optimal
but operationally wrong — cashiers need a mix to make change. Proportional
allocation grounded in circulation data avoids this failure mode while remaining
simple and auditable.

The denomination quantities vary day-to-day with the forecast: a busy payday
gets proportionally more bills than a quiet weekday.

---

## 5. Reproducibility

- All random seeds set to 42 (PyMC RANDOM_SEED, NumPy RNG, LightGBM, Optuna TPESampler).
- `uv.lock` pins every dependency to exact versions for full environment reproducibility.
- Trained artefacts serialised with ArviZ NetCDF (Bayesian trace), LightGBM text
  format (ML models), and NumPy `.npy` (weights, conformal q̂).
- Docker image (`python:3.11-slim` + `uv sync`) for containerised reproducibility.

---

## 6. Testing Strategy

| Layer | Approach | Key assertions |
|---|---|---|
| Data | Unit | Required columns, no nulls after imputation |
| Features | Unit | No future leakage (shift before rolling), ratio in [0,1] |
| Stats | Unit | Report keys present, best model valid enum value |
| Bayesian | Unit (fast MCMC: 1 chain, 100 draws) | Shape, positivity, save/load round-trip |
| ML | Unit (1 Optuna trial) | Shape, positivity, save/load round-trip |
| Conformal | Unit | Empirical coverage ≥ 1−α, lower ≥ 0 |
| Blender | Unit | Weights ≥ 0, sum to 1, MSE ≤ max(base MSEs) |
| Newsvendor | Unit | q* = CR-th quantile, E[cost(q*)] ≤ E[cost(extremes)] |
| Denomination | Unit | coin minimums present, bill counts scale with forecast, $500/$1000 = 0 |
| API | Unit (mocked) | HTTP 200 on /health, 200 on /predict, 422 on bad schema |
| Pipelines | Integration (Bayesian mocked) | All artefact files present, output DataFrame columns correct |

---

## 7. What I Would Do With More Time

1. **Covariates**: temperature, local events (football matches, concerts), competitor promotions.
2. **Online learning**: incremental Bayesian updating with new daily observations without full refit.
3. **Store clusters**: k-means on amount_cash time-series features to group stores with similar demand patterns, then fit separate models per cluster.
4. **Dashboard**: real-time Grafana/Streamlit dashboard showing forecast vs. actual and alert when q* would have been violated.
5. **Multi-day horizon**: currently 1-day ahead; extend to 7-day rolling horizon with recalibration.
