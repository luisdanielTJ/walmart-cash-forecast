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
    data_dir: Path = typer.Option(..., help="Directory with transactions.csv, stores.csv, calendar.csv"),
    model_dir: Path = typer.Option(..., help="Output directory for trained model artefacts"),
    config_path: Optional[Path] = typer.Option(None, help="Path to config.yaml (uses defaults if omitted)"),
) -> None:
    """Run the full training pipeline: load data → fit models → save artefacts."""
    from walmart_cash_forecast.config import Config
    from walmart_cash_forecast.pipelines.training import TrainingPipeline

    cfg = Config.from_yaml(config_path) if config_path else Config()
    typer.echo(f"Training with data from {data_dir} → artefacts to {model_dir}")
    metadata = TrainingPipeline(cfg, data_dir, model_dir).run()
    typer.echo(f"Train rows: {metadata['train_rows']}, Calib rows: {metadata['calib_rows']}")
    typer.echo(f"Blend weights (Bayes, ML): {metadata['blend_weights']}")
    typer.echo(f"Conformal q̂: {metadata['conformal_q_hat']:.2f} MXN")
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
    from walmart_cash_forecast.config import Config
    from walmart_cash_forecast.api.app import create_app

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
    import pandas as pd
    from walmart_cash_forecast.config import Config
    from walmart_cash_forecast.data.loader import DataLoader
    from walmart_cash_forecast.features.aggregator import StoreAggregator
    from walmart_cash_forecast.stats.reporter import StatAnalyzer

    cfg = Config.from_yaml(config_path) if config_path else Config()  # noqa: F841
    transactions, _, _ = DataLoader(data_dir).load()
    panel = StoreAggregator().aggregate(transactions)

    report = StatAnalyzer().run(panel, out_dir)
    typer.echo(f"Stats report saved to {out_dir / 'stats_summary.json'}")
    typer.echo(f"Distribution best model: {report['distribution']['best_model']}")
    typer.echo(f"Payday effect p-value: {report['payday_effect']['pvalue']:.4f}")
    pct = report['stationarity']['pct_stores_stationary']
    typer.echo(f"Stores with stationary series: {pct:.1%}")


if __name__ == "__main__":
    app()
