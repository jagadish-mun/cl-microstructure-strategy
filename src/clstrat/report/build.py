"""Plots and the research report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

FIGSIZE = (8, 4.5)


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_ic_decay(ic: pd.DataFrame, path: Path, primary: int, top_n: int = 12) -> Path:
    """Spearman IC vs horizon for the ``top_n`` features by |IC| at the primary horizon."""
    top = ic[primary].abs().sort_values(ascending=False).index[:top_n]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_prop_cycle(color=plt.cm.tab20.colors)
    horizons = [int(h) for h in ic.columns]
    for name in top:
        ax.plot(horizons, ic.loc[name].to_numpy(), marker="o", label=name)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(horizons, [f"{h}s" for h in horizons])
    ax.set_xlabel("forward horizon")
    ax.set_ylabel("Spearman IC")
    ax.set_title(f"IC decay: top {top_n} features by |IC| at {primary}s (research days)")
    ax.legend(fontsize=7, ncol=2, loc="best")
    return _save(fig, path)
