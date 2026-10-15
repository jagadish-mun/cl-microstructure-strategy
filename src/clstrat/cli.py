"""Command-line entry point: ``clstrat <command>``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
import yaml

from clstrat.backtest.engine import Engine, backtest_inputs
from clstrat.backtest.metrics import summarize
from clstrat.config import Config, data_path, load_config, reports_path
from clstrat.data.adapters import make_source
from clstrat.data.synthetic import generate_dataset
from clstrat.data.validate import calibration_stats, clean_day, format_calibration, quality_table
from clstrat.features.pipeline import build_dataset, ic_decay, research_days
from clstrat.models.evaluate import (
    coefficient_table,
    fold_metrics,
    model_comparison,
    run_walk_forward,
    save_results,
    sign_flips,
)
from clstrat.report.build import plot_coef_stability, plot_equity, plot_ic_decay

app = typer.Typer(add_completion=False, help="CL futures microstructure research pipeline.")

ConfigOpt = typer.Option(None, "--config", "-c", help="YAML config (default: configs/default.yaml)")


def _synthetic_dir(cfg: Config) -> Path:
    return data_path(cfg, "synthetic")


def dataset_path(cfg: Config) -> Path:
    """Decision-time feature/target dataset written by ``features``."""
    return data_path(cfg, "features", "dataset.parquet")


@app.callback()
def main() -> None:
    """CL futures microstructure research pipeline."""


@app.command()
def show_config(config: Path | None = ConfigOpt) -> None:
    """Print the resolved configuration."""
    typer.echo(yaml.safe_dump(load_config(config), sort_keys=False))


@app.command()
def generate(config: Path | None = ConfigOpt) -> None:
    """Simulate the synthetic CL market and write parquet files under data/synthetic."""
    cfg = load_config(config)
    days = generate_dataset(cfg, _synthetic_dir(cfg))
    typer.echo(f"wrote {len(days)} days to {_synthetic_dir(cfg)}")


@app.command()
def validate(config: Path | None = ConfigOpt) -> None:
    """Run data-quality checks and print the calibration report."""
    cfg = load_config(config)
    store = make_source(cfg)
    summaries = []

    def sessions():
        for day in store.days():
            q, t, summary = clean_day(day, store.load_quotes(day), store.load_trades(day), cfg)
            summaries.append(summary)
            yield day, q, t

    report = format_calibration(calibration_stats(sessions(), cfg), cfg)
    quality = quality_table(summaries)
    out = reports_path(cfg)
    quality.to_csv(out / "data_quality.csv")
    (out / "calibration.txt").write_text(report + "\n")
    totals = quality.sum(numeric_only=True)
    totals["max_gap_s"] = quality["max_gap_s"].max()
    typer.echo(totals.to_string())
    typer.echo(report)


@app.command()
def features(config: Path | None = ConfigOpt) -> None:
    """Build bars, features and targets; write the IC decay table and plot (research days)."""
    cfg = load_config(config)
    df, summaries = build_dataset(cfg, make_source(cfg), data_path(cfg, "bars"))
    path = dataset_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    out = reports_path(cfg)
    quality_table(summaries).to_csv(out / "data_quality.csv")
    ic = ic_decay(research_days(df, cfg["cv"]["holdout_days"]), cfg["target"]["horizons_report"])
    ic.to_csv(out / "ic_decay.csv")
    plot_ic_decay(ic, out / "ic_decay.png", cfg["target"]["horizon"])
    primary = ic[cfg["target"]["horizon"]]
    typer.echo(f"{len(df)} decision rows, {ic.shape[0]} features -> {path}")
    typer.echo(ic.loc[primary.abs().sort_values(ascending=False).index[:10]].round(3).to_string())


@app.command()
def train(config: Path | None = ConfigOpt) -> None:
    """Purged walk-forward evaluation of every model and horizon (holdout untouched)."""
    cfg = load_config(config)
    result = run_walk_forward(pd.read_parquet(dataset_path(cfg)), cfg)
    out = reports_path(cfg)
    h = cfg["target"]["horizon"]
    comparison = model_comparison(result, cfg["models"])
    comparison.to_csv(out / "model_comparison.csv")
    fold_metrics(result, h, cfg["models"]).to_csv(out / "fold_metrics.csv")
    result.selection[h].to_csv(out / "selection_frequency.csv")
    choice = save_results(result, cfg, data_path(cfg, "models"))
    for name in (m for m in cfg["models"] if m not in ("zero", "ar1")):
        coefs = coefficient_table(result, h, name)
        coefs.join(sign_flips(coefs)).to_csv(out / f"coefficients_{name}.csv")
    coefs = coefficient_table(result, h, choice["best_model"])
    title = f"{choice['best_model']} coefficients across folds ({h}s target)"
    plot_coef_stability(coefs, sign_flips(coefs), out / "coef_stability.png", title)
    typer.echo(comparison[["oos_r2", "ic_spearman", "hit_rate", "folds_r2_pos"]].round(4))
    typer.echo(f"best model at {h}s: {choice['best_model']}")


@app.command()
def backtest(config: Path | None = ConfigOpt) -> None:
    """Event-driven backtest of the chosen model on the walk-forward test days."""
    cfg = load_config(config)
    preds, quantiles, load_bars = backtest_inputs(cfg)
    result = Engine(cfg).run(preds, load_bars, quantiles)
    stats = summarize(result)
    result.trades.to_csv(reports_path(cfg, "trade_log.csv"), index=False)
    result.daily.to_csv(reports_path(cfg, "daily_pnl.csv"))
    pd.Series(stats).to_csv(reports_path(cfg, "backtest_summary.csv"), header=["value"])
    plot_equity(result.daily, result.equity, reports_path(cfg, "equity_curve.png"),
                "Walk-forward test days: cumulative P&L")  # fmt: skip
    typer.echo(pd.Series(stats).round(3).to_string())


if __name__ == "__main__":
    app()
