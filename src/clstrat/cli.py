"""Command-line entry point: ``clstrat <command>``."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from clstrat.config import load_config

app = typer.Typer(add_completion=False, help="CL futures microstructure research pipeline.")

ConfigOpt = typer.Option(None, "--config", "-c", help="YAML config (default: configs/default.yaml)")


@app.callback()
def main() -> None:
    """CL futures microstructure research pipeline."""


@app.command()
def show_config(config: Path | None = ConfigOpt) -> None:
    """Print the resolved configuration."""
    typer.echo(yaml.safe_dump(load_config(config), sort_keys=False))


if __name__ == "__main__":
    app()
