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


def plot_coef_stability(coefs: pd.DataFrame, flips: pd.Series, path: Path, title: str) -> Path:
    """Standardised coefficient of each ever-selected feature in every fold.

    Features whose coefficient changes sign across folds are labelled in red."""
    order = coefs.abs().mean(axis=1).sort_values().index
    coefs = coefs.loc[order]
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.28 * len(coefs) + 1)))
    for k, fold in enumerate(coefs.columns):
        ax.scatter(coefs[fold], range(len(coefs)), s=22, label=f"fold {fold}", zorder=3,
                   color=plt.cm.viridis(k / max(1, len(coefs.columns) - 1)))  # fmt: skip
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(range(len(coefs)), coefs.index, fontsize=7)
    for label in ax.get_yticklabels():
        if flips.get(label.get_text(), False):
            label.set_color("tab:red")
    ax.set_xlabel("standardised coefficient (ticks per s.d.)")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower right")
    return _save(fig, path)
