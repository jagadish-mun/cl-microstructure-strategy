"""Plots and the research report."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

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


def plot_equity(daily: pd.DataFrame, equity: pd.Series, path: Path, title: str) -> Path:
    """Cumulative net P&L (gross overlaid) by day, and the net drawdown from intraday marks."""
    fig, (ax, dd_ax) = plt.subplots(
        2, 1, figsize=(8, 5.5), sharex=False, gridspec_kw={"height_ratios": [3, 1.2]}
    )
    x = np.arange(1, len(daily) + 1)
    ax.plot(x, daily["gross"].cumsum(), ls="--", color="tab:grey", label="gross (before costs)")
    ax.plot(x, daily["net"].cumsum(), color="tab:blue", lw=2, label="net (after all costs)")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("cumulative P&L ($)")
    ax.set_xlabel("trading day")
    ax.set_title(title)
    ax.legend(loc="upper left")
    curve = np.concatenate([[0.0], equity.to_numpy()])
    drawdown = curve - np.maximum.accumulate(curve)
    dd_ax.fill_between(np.arange(len(curve)), drawdown, color="tab:red", alpha=0.5)
    dd_ax.set_ylabel("net drawdown ($)")
    dd_ax.set_xlabel("decision time (all test days)")
    return _save(fig, path)


def plot_heatmaps(
    table: pd.DataFrame, rows: str, cols: str, panels: list[tuple[str, str]], path: Path, title: str
) -> Path:
    """One annotated heatmap per ``(value_column, label)`` over a two-parameter grid."""
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.2))
    for ax, (value, label) in zip(np.atleast_1d(axes), panels, strict=True):
        grid = table.pivot(index=rows, columns=cols, values=value)
        lim = np.nanmax(np.abs(grid.to_numpy())) or 1.0
        im = ax.imshow(grid.to_numpy(), cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(grid.shape[1]), [f"{v:g}" for v in grid.columns])
        ax.set_yticks(range(grid.shape[0]), [f"{v:g}" for v in grid.index])
        ax.set_xlabel(cols.split(".")[-1])
        ax.set_ylabel(rows.split(".")[-1])
        ax.set_title(label)
        for (i, j), v in np.ndenumerate(grid.to_numpy()):
            if np.isfinite(v):
                ax.text(j, i, f"{v:,.0f}" if abs(v) >= 100 else f"{v:.2f}", ha="center",
                        va="center", fontsize=7)  # fmt: skip
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(title)
    return _save(fig, path)


def plot_cost_sensitivity(curves: dict[str, pd.DataFrame], break_even: float, path: Path) -> Path:
    """Net and gross P&L as fees and latency rise (chosen strategy, walk-forward days)."""
    fig, axes = plt.subplots(1, len(curves), figsize=(5.2 * len(curves), 4))
    for ax, (key, table) in zip(np.atleast_1d(axes), curves.items(), strict=True):
        x = table[f"execution.{key}"]
        ax.plot(x, table["net_pnl"], marker="o", lw=2, label="net")
        ax.plot(x, table["gross_pnl"], marker=".", ls="--", color="tab:grey", label="gross")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xlabel(key)
        ax.set_ylabel("P&L ($)")
        ax.legend()
        if key == "fee_per_contract_rt" and np.isfinite(break_even):
            ax.axvline(break_even, color="tab:red", ls=":", lw=1.5)
            ax.set_title(f"break-even round-trip fee ≈ ${break_even:.2f}")
        else:
            ax.set_title(f"net P&L vs {key}")
    return _save(fig, path)


def plot_firewall_comparison(results: dict, path: Path) -> Path:
    """Cumulative daily net P&L with and without firewalls."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for label, result in results.items():
        cum = result.daily["net"].cumsum()
        ax.plot(np.arange(1, len(cum) + 1), cum, lw=2, label=label)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("trading day")
    ax.set_ylabel("cumulative net P&L ($)")
    ax.set_title("Firewalls on vs off (walk-forward test days)")
    ax.legend()
    return _save(fig, path)


def plot_placebo(placebo_pnl: pd.Series, real: float, p_value: float, path: Path) -> Path:
    """Histogram of net P&L after shuffling predictions within each day, vs the real run."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(placebo_pnl, bins=30, color="tab:grey", alpha=0.8, label="shuffled predictions")
    ax.axvline(real, color="tab:blue", lw=2.5, label=f"real predictions (p = {p_value:.3f})")
    ax.set_xlabel("net P&L over the walk-forward test days ($)")
    ax.set_ylabel("count")
    ax.set_title(f"Placebo test: {len(placebo_pnl)} within-day shuffles (firewalls off)")
    ax.legend()
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

SAMPLE_FILES = (
    "report.md", "ic_decay.png", "coef_stability.png", "strategy_equity.png",
    "threshold_heatmap.png", "percentile_heatmap.png", "cost_sensitivity.png",
    "firewalls_on_off.png", "penalty_sensitivity.png", "placebo.png", "holdout_equity.png",
)  # fmt: skip


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    df = df.reset_index() if df.index.name or isinstance(df.index, pd.MultiIndex) else df

    def fmt(v) -> str:
        if isinstance(v, float | np.floating):
            return f"{v:,.{digits}f}" if abs(v) < 1000 else f"{v:,.0f}"
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def _money(stats: dict, keys: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"metric": keys, "value": [stats[k] for k in keys]})


HEADLINE = ["net_pnl", "net_sharpe", "max_drawdown", "worst_day", "trades", "hit_rate",
            "profit_factor", "gross_pnl", "cost_spread_cost", "cost_fees", "cost_slippage",
            "cost_eod_penalty", "cost_forced_penalty"]  # fmt: skip


def report_markdown(reports: Path, models: Path, horizon: int, threshold: float) -> str:
    """Assemble the research report from the files written by the pipeline commands."""
    research = yaml.safe_load((reports / "research_summary.yaml").read_text())
    holdout = yaml.safe_load((models / "holdout.yaml").read_text())
    strategy = yaml.safe_load((models / "strategy.yaml").read_text())
    selection = yaml.safe_load((models / "selection.yaml").read_text())
    comparison = pd.read_csv(reports / "model_comparison.csv")
    comparison = comparison[["horizon", "model", "oos_r2", "ic_spearman", "hit_rate",
                             "folds_r2_pos"]]  # fmt: skip
    freq = pd.read_csv(reports / "selection_frequency.csv", index_col=0)
    kept = pd.DataFrame(
        {"fold": freq.columns,
         "selected features": [", ".join(freq.index[freq[c] >= threshold]) for c in freq]}
    )  # fmt: skip
    coefs = pd.read_csv(reports / f"coefficients_{selection['best_model']}.csv", index_col=0)
    flips = list(coefs.index[coefs["sign_flip"]])
    ic = pd.read_csv(reports / "ic_decay.csv", index_col=0)
    top_ic = ic.reindex(ic[str(horizon)].abs().sort_values(ascending=False).index).head(15)
    wf = pd.read_csv(reports / "strategy_summary.csv", index_col=0)["value"].to_dict()
    hb = holdout["backtest"]
    dsr = research["deflated_sharpe"]
    sections = [
        "# CL microstructure strategy: research report",
        "All data is **simulated** (see the README for why). Net-of-cost numbers come first. "
        "The final holdout was run once, after every choice below was fixed.",
        "## 1. Data calibration", "```\n" + (reports / "calibration.txt").read_text() + "```",
        "## 2. Features: strongest Spearman IC at the primary horizon",
        md_table(top_ic.rename_axis("feature"), 3), "![IC decay](ic_decay.png)",
        "## 3. Models (purged walk-forward, research days only)", md_table(comparison, 4),
        f"Chosen model: **{selection['best_model']}** (best pooled OOS R² at {horizon}s).",
        "![coefficient stability](coef_stability.png)",
        f"Sign flips across folds: {', '.join(flips) if flips else 'none'}.",
        "### Selected features per fold", md_table(kept),
        "## 4. Strategy on the walk-forward test days",
        f"Chosen configuration (from {strategy['configs_tried']} tried): "
        f"`{strategy['overrides']}`, firewalls on.",
        md_table(_money(wf, HEADLINE), 2), "![equity](strategy_equity.png)",
        "### Threshold sweeps", "![ticks](threshold_heatmap.png)", "![pct](percentile_heatmap.png)",
        "### Cost sensitivity",
        f"Break-even round-trip fee: **${research['break_even_fee_rt']:.2f}** per contract "
        "(assumed: $2.50).", "![costs](cost_sensitivity.png)",
        "### Firewalls on vs off",
        md_table(pd.read_csv(reports / "firewalls_on_off.csv", index_col=0).rename_axis("run"), 1),
        "![firewalls](firewalls_on_off.png)",
        "### Liquidation penalties",
        md_table(pd.read_csv(reports / "penalty_sensitivity.csv"), 2),
        "### Placebo",
        f"Real net P&L ${research['placebo']['real_net_pnl']:,.0f} vs shuffled mean "
        f"${research['placebo']['shuffle_mean']:,.0f}; p = {research['placebo']['p_value']:.3f}.",
        "![placebo](placebo.png)",
        "### Independent synthetic markets (frozen pipeline)",
        md_table(pd.read_csv(reports / "seed_robustness.csv"), 4),
        "### Regimes",
        md_table(pd.read_csv(reports / "regime_table.csv").set_index(["split", "group"]), 2),
        "### Selection bias",
        f"Configurations tried: {dsr['configs_tried']}. Annualised Sharpe "
        f"{dsr['sharpe_annual']:.2f}; expected best Sharpe under no skill "
        f"{dsr['expected_max_sharpe_annual_under_null']:.2f}. Probabilistic Sharpe vs 0: "
        f"{dsr['psr_vs_zero']:.3f}. **Deflated Sharpe: {dsr['deflated_sharpe']:.3f}.**",
        "## 5. Final holdout (run once)",
        f"Model refitted on {holdout['train_days'][0]} to {holdout['train_days'][1]}; "
        f"holdout {holdout['holdout_days'][0]} to {holdout['holdout_days'][1]}. "
        f"Forecast OOS R² {holdout['forecast']['oos_r2']:.4f}, Spearman IC "
        f"{holdout['forecast']['ic_spearman']:.3f}.",
        md_table(_money(hb, HEADLINE), 2), "![holdout](holdout_equity.png)",
    ]
    return "\n\n".join(sections) + "\n"


def build_report(reports: Path, models: Path, horizon: int, threshold: float, sample: Path) -> Path:
    """Write ``report.md`` and copy it with its figures to ``sample`` (committed to git)."""
    path = reports / "report.md"
    path.write_text(report_markdown(reports, models, horizon, threshold))
    sample.mkdir(parents=True, exist_ok=True)
    for name in SAMPLE_FILES:
        shutil.copy2(reports / name, sample / name)
    return path
