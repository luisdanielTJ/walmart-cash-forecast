"""Typer CLI for the Walmart Cash Forecast system.

Commands:
  train    — run the full training pipeline on raw CSVs
  predict  — run the prediction pipeline and save a CSV output
  serve    — start the FastAPI server with uvicorn
  stats    — run statistical analyses only (no model training)

Usage examples:
  walmart-forecast train --data-dir data/raw --model-dir models/v1
  walmart-forecast predict --model-dir models/v1 --future-csv data/future.csv --out predictions.csv
  walmart-forecast serve --model-dir models/v1 --port 8000
  walmart-forecast stats --data-dir data/raw --out-dir reports/stats
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="walmart-forecast",
    help="Daily store-level cash change forecasting for Walmart Mexico.",
    add_completion=False,
)


@app.command()
def train(
    data_dir: Path = typer.Option(..., help="Dir with transactions.csv, stores.csv, calendar.csv"),
    model_dir: Path = typer.Option(..., help="Output directory for trained model artefacts"),
    config_path: Optional[Path] = typer.Option(None, help="Path to config.yaml (optional)"),
) -> None:
    """Run the full training pipeline: load data -> fit models -> save artefacts."""
    import logging

    import mlflow

    from walmart_cash_forecast.config import Config
    from walmart_cash_forecast.pipelines.training import TrainingPipeline

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = Config.from_yaml(config_path) if config_path else Config()
    typer.echo(f"Training with data from {data_dir} -> artefacts to {model_dir}")
    typer.echo(f"Config: {cfg.bayesian.n_chains} chains x {cfg.bayesian.n_draws} draws, "
               f"holdout={cfg.bayesian.n_tune} tune steps")
    metadata = TrainingPipeline(cfg, data_dir, model_dir).run()
    typer.echo("")
    typer.echo(f"Train rows: {metadata['train_rows']}, Calib rows: {metadata['calib_rows']}")
    typer.echo(f"Blend weights (Bayes / ML): "
               f"{metadata['blend_weights'][0]:.1%} / {metadata['blend_weights'][1]:.1%}")
    typer.echo(f"Conformal q_hat: {metadata['conformal_q_hat']:,.0f} MXN")
    stat = metadata["stat_report"]
    typer.echo(f"Distribution: {stat['distribution']['best_model']} "
               f"(NegBin AIC {stat['distribution']['aic_negbin']:,.0f})")
    typer.echo(f"Payday lift: {stat['payday_effect']['effect_size']:.1%} "
               f"(p={stat['payday_effect']['pvalue']:.4f})")
    run = mlflow.last_active_run()
    if run:
        typer.echo(f"MLflow run: {run.info.run_id}  ->  run `mlflow ui` to explore")
    typer.echo("Training complete.")


@app.command()
def predict(
    model_dir: Path = typer.Option(..., help="Directory with trained artefacts"),
    future_csv: Path = typer.Option(..., help="CSV with future store-date feature rows"),
    stores_csv: Path = typer.Option(..., help="stores.csv with region and store_format"),
    out: Path = typer.Option(..., help="Output CSV path for predictions"),
    config_path: Optional[Path] = typer.Option(None, help="Path to config.yaml"),
) -> None:
    """Generate cash recommendations for future store-dates."""
    import pandas as pd

    from walmart_cash_forecast.config import Config
    from walmart_cash_forecast.pipelines.prediction import PredictionPipeline

    cfg = Config.from_yaml(config_path) if config_path else Config()
    future_df = pd.read_csv(future_csv, parse_dates=["date"])
    stores_df = pd.read_csv(stores_csv)

    typer.echo(f"Loading model from {model_dir}")
    pipeline = PredictionPipeline(cfg, model_dir)
    pipeline.load()

    typer.echo(f"Predicting for {len(future_df)} store-dates")
    result = pipeline.predict(future_df, stores_df)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    typer.echo(f"Predictions saved to {out} ({len(result)} rows)")


@app.command()
def serve(
    model_dir: Path = typer.Option(..., help="Directory with trained artefacts"),
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
    config_path: Optional[Path] = typer.Option(None, help="Path to config.yaml"),
) -> None:
    """Start the FastAPI prediction server with uvicorn."""
    import uvicorn

    from walmart_cash_forecast.api.app import create_app
    from walmart_cash_forecast.config import Config

    cfg = Config.from_yaml(config_path) if config_path else Config()
    application = create_app(cfg, model_dir)
    typer.echo(f"Starting server on {host}:{port} (model: {model_dir})")
    uvicorn.run(application, host=host, port=port)


@app.command()
def stats(
    data_dir: Path = typer.Option(..., help="Directory with raw CSVs"),
    out_dir: Path = typer.Option(..., help="Output directory for stats_summary.json"),
    config_path: Optional[Path] = typer.Option(None, help="Path to config.yaml"),
) -> None:
    """Run statistical analyses (distribution, stationarity, STL, payday effect)."""
    from walmart_cash_forecast.config import Config
    from walmart_cash_forecast.data.loader import DataLoader
    from walmart_cash_forecast.features.aggregator import StoreAggregator
    from walmart_cash_forecast.stats.reporter import StatAnalyzer

    cfg = Config.from_yaml(config_path) if config_path else Config()  # noqa: F841
    transactions, _, calendar = DataLoader(data_dir).load()
    panel = StoreAggregator().aggregate(transactions)

    # Add is_payday — prefer calendar if available, fall back to date formula
    if calendar is not None and "is_payday" in calendar.columns:
        panel = panel.merge(calendar[["date", "is_payday"]], on="date", how="left")
        panel["is_payday"] = panel["is_payday"].fillna(
            panel["date"].dt.day.isin([15]) | (
                panel["date"] == panel["date"].dt.to_period("M").dt.to_timestamp("M")
            )
        )
    else:
        panel["is_payday"] = panel["date"].dt.day.isin([15]) | (
            panel["date"] == panel["date"].dt.to_period("M").dt.to_timestamp("M")
        )

    report = StatAnalyzer().run(panel, out_dir)
    typer.echo(f"Stats report saved to {out_dir / 'stats_summary.json'}")
    typer.echo(f"Distribution best model: {report['distribution']['best_model']}")
    typer.echo(f"Payday effect p-value: {report['payday_effect']['pvalue']:.4f}")
    pct = report['stationarity']['pct_stores_stationary']
    typer.echo(f"Stores with stationary series: {pct:.1%}")


if __name__ == "__main__":
    app()
