"""Command-line entry point: ``clstrat <command>``."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from clstrat.config import Config, load_config
from clstrat.data.adapters import make_source
from clstrat.data.synthetic import generate_dataset
from clstrat.data.validate import calibration_stats, clean_day, format_calibration, quality_table
from clstrat.features.pipeline import build_dataset, ic_decay, research_days
from clstrat.report.build import plot_ic_decay

app = typer.Typer(add_completion=False, help="CL futures microstructure research pipeline.")

ConfigOpt = typer.Option(None, "--config", "-c", help="YAML config (default: configs/default.yaml)")


def _synthetic_dir(cfg: Config) -> Path:
    return Path(cfg["paths"]["data_dir"]) / "synthetic"


def dataset_path(cfg: Config) -> Path:
    """Decision-time feature/target dataset written by ``features``."""
    return Path(cfg["paths"]["data_dir"]) / "features" / "dataset.parquet"


def bars_dir(cfg: Config) -> Path:
    """Per-day 1 s bars written by ``features`` and read by the backtester."""
    return Path(cfg["paths"]["data_dir"]) / "bars"


def _reports_dir(cfg: Config) -> Path:
    path = Path(cfg["paths"]["reports_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


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
    out = _reports_dir(cfg)
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
    df, summaries = build_dataset(cfg, make_source(cfg), bars_dir(cfg))
    path = dataset_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    out = _reports_dir(cfg)
    quality_table(summaries).to_csv(out / "data_quality.csv")
    ic = ic_decay(research_days(df, cfg["cv"]["holdout_days"]), cfg["target"]["horizons_report"])
    ic.to_csv(out / "ic_decay.csv")
    plot_ic_decay(ic, out / "ic_decay.png", cfg["target"]["horizon"])
    primary = ic[cfg["target"]["horizon"]]
    typer.echo(f"{len(df)} decision rows, {ic.shape[0]} features -> {path}")
    typer.echo(ic.loc[primary.abs().sort_values(ascending=False).index[:10]].round(3).to_string())


if __name__ == "__main__":
    app()
