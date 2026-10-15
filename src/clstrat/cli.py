"""Command-line entry point: ``clstrat <command>``."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from clstrat.config import Config, load_config
from clstrat.data.adapters import make_source
from clstrat.data.synthetic import generate_dataset
from clstrat.data.validate import calibration_stats, clean_day, format_calibration, quality_table

app = typer.Typer(add_completion=False, help="CL futures microstructure research pipeline.")

ConfigOpt = typer.Option(None, "--config", "-c", help="YAML config (default: configs/default.yaml)")


def _synthetic_dir(cfg: Config) -> Path:
    return Path(cfg["paths"]["data_dir"]) / "synthetic"


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


if __name__ == "__main__":
    app()
