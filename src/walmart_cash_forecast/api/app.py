"""FastAPI application exposing the cash-forecast prediction pipeline.

Endpoints:
  GET  /health   — liveness probe (Docker / Kubernetes health check)
  POST /predict  — generate store-level cash recommendations for future dates

The app loads the trained artefacts once at startup (lifespan context) and
reuses them across requests — MCMC traces are large and loading them per
request would add seconds of latency.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from walmart_cash_forecast.config import Config
from walmart_cash_forecast.pipelines.prediction import PredictionPipeline

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class StoreDateRequest(BaseModel):
    """A single store-date prediction request.

    Lag and rolling feature names match FeatureEngine output so they can be
    passed directly to the ML model without renaming.
    """
    store_id: str
    date: str = Field(..., description="ISO-8601 date string, e.g. '2024-01-15'")
    day_of_week: int = Field(..., ge=0, le=6, description="0=Monday, 6=Sunday")
    is_payday: bool = False
    is_holiday: bool = False
    is_buen_fin: bool = False
    is_navidad_season: bool = False
    # Lag / rolling features — match FeatureEngine column names exactly
    amount_cash_lag_1: float = 0.0
    amount_cash_lag_7: float = 0.0
    amount_cash_lag_14: float = 0.0
    amount_cash_roll7_mean: float = 0.0
    amount_cash_roll7_std: float = 0.0
    amount_cash_roll28_mean: float = 0.0
    cash_ratio: float = 0.5
    days_since_payday: float = 7.0
    days_until_payday: float = 7.0


class PredictRequest(BaseModel):
    """Batch prediction request."""
    observations: list[StoreDateRequest]
    stores: list[dict] = Field(
        ...,
        description="Store metadata rows: store_id, region, store_format",
    )


class StorePrediction(BaseModel):
    """Prediction output for one store-date."""
    store_id: str
    date: str
    forecast_blend: float
    lower: float
    upper: float
    q_star: float
    # Denomination mix for the register change fund (pieces per denomination)
    denom_0_10: int = 0
    denom_0_20: int = 0
    denom_0_50: int = 0
    denom_1_00: int = 0
    denom_2_00: int = 0
    denom_5_00: int = 0
    denom_10_00: int = 0
    denom_20_00: int = 0
    denom_50_00: int = 0
    denom_100_00: int = 0
    denom_200_00: int = 0
    denom_500_00: int = 0
    denom_1000_00: int = 0


class PredictResponse(BaseModel):
    """Batch prediction response."""
    predictions: list[StorePrediction]


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

_pipeline: PredictionPipeline | None = None


def _get_pipeline() -> PredictionPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load trained artefacts once at startup; release on shutdown."""
    global _pipeline  # noqa: PLW0603
    model_dir = Path(app.state.model_dir)
    config = app.state.config
    _pipeline = PredictionPipeline(config, model_dir)
    _pipeline.load()
    yield
    _pipeline = None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(config: Config, model_dir: str | Path) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Separating creation from module-level instantiation allows the app to be
    created with different configs in tests without importing a singleton.

    Args:
        config: Project configuration.
        model_dir: Path to trained model artefacts.

    Returns:
        Configured FastAPI instance.
    """
    app = FastAPI(
        title="Walmart Cash Forecast API",
        description=(
            "Daily store-level cash demand forecasting with newsvendor "
            "optimisation and denomination mix recommendation."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.model_dir = str(model_dir)

    @app.get("/health", tags=["ops"])
    def health():
        """Liveness probe — always returns 200 if the server is running."""
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse, tags=["forecast"])
    def predict(request: PredictRequest):
        """
        Generate cash recommendations for a batch of future store-dates.

        The response includes:
        - forecast_blend: Bayesian + ML blended point estimate (MXN)
        - lower / upper: 90% conformal prediction interval (MXN)
        - q_star: Newsvendor optimal cash buffer (MXN)
        """
        pipeline = _get_pipeline()

        rows = [obs.model_dump() for obs in request.observations]
        future_df = pd.DataFrame(rows)
        future_df["date"] = pd.to_datetime(future_df["date"])
        future_df["is_payday"] = future_df["is_payday"].astype(float)
        future_df["is_holiday"] = future_df["is_holiday"].astype(float)
        future_df["is_buen_fin"] = future_df["is_buen_fin"].astype(float)
        future_df["is_navidad_season"] = future_df["is_navidad_season"].astype(float)

        stores_df = pd.DataFrame(request.stores)

        try:
            result_df = pipeline.predict(future_df, stores_df)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        predictions = [
            StorePrediction(
                store_id=str(row["store_id"]),
                date=str(row["date"].date()),
                forecast_blend=round(float(row["forecast_blend"]), 2),
                lower=round(float(row["lower"]), 2),
                upper=round(float(row["upper"]), 2),
                q_star=round(float(row["q_star"]), 2),
                denom_0_10=int(row.get("denom_0.10", 0)),
                denom_0_20=int(row.get("denom_0.20", 0)),
                denom_0_50=int(row.get("denom_0.50", 0)),
                denom_1_00=int(row.get("denom_1.00", 0)),
                denom_2_00=int(row.get("denom_2.00", 0)),
                denom_5_00=int(row.get("denom_5.00", 0)),
                denom_10_00=int(row.get("denom_10.00", 0)),
                denom_20_00=int(row.get("denom_20.00", 0)),
                denom_50_00=int(row.get("denom_50.00", 0)),
                denom_100_00=int(row.get("denom_100.00", 0)),
                denom_200_00=int(row.get("denom_200.00", 0)),
                denom_500_00=int(row.get("denom_500.00", 0)),
                denom_1000_00=int(row.get("denom_1000.00", 0)),
            )
            for _, row in result_df.iterrows()
        ]
        return PredictResponse(predictions=predictions)

    return app
