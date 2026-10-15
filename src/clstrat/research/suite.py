"""The full §8 robustness suite on the walk-forward test days, in one call.

The strategy configuration chosen here (best net Sharpe over the threshold and percentile
grids) is saved to ``data/models/strategy.yaml`` and is what the single holdout run uses.
"""

from __future__ import annotations

import pandas as pd
import yaml

from clstrat.backtest.engine import backtest_inputs, chosen_model
from clstrat.backtest.metrics import summarize
from clstrat.backtest.vectorized import VectorBacktester
from clstrat.config import Config, data_path, reports_path, with_overrides
from clstrat.report.build import (
    plot_cost_sensitivity,
    plot_equity,
    plot_firewall_comparison,
    plot_heatmaps,
    plot_placebo,
)
from clstrat.research.robustness import (
    day_labels,
    deflated_sharpe,
    placebo,
    placebo_p_value,
    regime_table,
    seed_run,
)
from clstrat.research.sweeps import (
    break_even_fee,
    choose_strategy,
    cost_sensitivity,
    daily_stats,
    engine_runner,
    firewall_comparison,
    penalty_sensitivity,
    percentile_sweep,
    threshold_sweep,
)

SHARPE_PANELS = [("net_sharpe", "net Sharpe (annualised)"), ("net_pnl", "net P&L ($)")]


def _select(vb: VectorBacktester, cfg: Config, sweep: Config) -> tuple[dict, list[float]]:
    """Threshold and percentile sweeps, their heatmaps, and the chosen configuration."""
    ticks = threshold_sweep(vb, cfg, sweep["thresholds"])
    pct = percentile_sweep(vb, cfg, sweep["percentiles"])
    ticks.to_csv(reports_path(cfg, "threshold_sweep.csv"), index=False)
    pct.to_csv(reports_path(cfg, "percentile_sweep.csv"), index=False)
    note = "(vectorized, firewalls off, walk-forward days)"
    plot_heatmaps(ticks, "strategy.theta_entry_ticks", "strategy.theta_exit_ticks",
                  SHARPE_PANELS, reports_path(cfg, "threshold_heatmap.png"),
                  f"Tick thresholds {note}")  # fmt: skip
    plot_heatmaps(pct, "strategy.pct_entry", "strategy.pct_exit", SHARPE_PANELS,
                  reports_path(cfg, "percentile_heatmap.png"),
                  f"Percentile thresholds {note}")  # fmt: skip
    choice = choose_strategy({"ticks": ticks, "percentile": pct}, sweep["selection_min_trades"])
    trials = pd.concat([ticks["net_sharpe"], pct["net_sharpe"]]).tolist()
    return choice, trials


def run_suite(cfg: Config, sweep: Config, seeds: bool = True) -> dict:
    """Run every §8 analysis, write tables and plots to the reports directory, and return the
    headline numbers."""
    preds, quantiles, load_bars = backtest_inputs(cfg)
    vb = VectorBacktester(cfg, preds, load_bars, quantiles)
    choice, trials = _select(vb, cfg, sweep)
    strat = with_overrides(cfg, choice["overrides"])
    with open(data_path(cfg, "models", "strategy.yaml"), "w") as fh:
        yaml.safe_dump(choice | {"configs_tried": len(trials)}, fh, sort_keys=False)

    curves = cost_sensitivity(vb, strat, sweep["costs"])
    base_daily = vb.run(strat)
    be_fee = break_even_fee(base_daily, strat["execution"]["fee_per_contract_rt"])
    pd.concat(curves, names=["parameter"]).to_csv(reports_path(cfg, "cost_sensitivity.csv"))
    plot_cost_sensitivity(curves, be_fee, reports_path(cfg, "cost_sensitivity.png"))

    run = engine_runner(preds, load_bars, quantiles)
    fw_table, fw_results = firewall_comparison(run, strat)
    fw_table.to_csv(reports_path(cfg, "firewalls_on_off.csv"))
    plot_firewall_comparison(fw_results, reports_path(cfg, "firewalls_on_off.png"))
    main = fw_results["firewalls on"]
    stats = summarize(main)
    pd.Series(stats).to_csv(reports_path(cfg, "strategy_summary.csv"), header=["value"])
    main.daily.to_csv(reports_path(cfg, "strategy_daily_pnl.csv"))
    main.trades.to_csv(reports_path(cfg, "strategy_trade_log.csv"), index=False)
    main.firewall_log.to_csv(reports_path(cfg, "strategy_firewall_log.csv"), index=False)
    plot_equity(main.daily, main.equity, reports_path(cfg, "strategy_equity.png"),
                "Chosen strategy, walk-forward test days (firewalls on)")  # fmt: skip

    penalties = penalty_sensitivity(run, strat, sweep["penalties"])
    penalties.to_csv(reports_path(cfg, "penalty_sensitivity.csv"), index=False)
    plot_heatmaps(penalties, "eod_penalty_ticks", "forced_liq_penalty_ticks",
                  [("net_pnl", "net P&L ($)"), ("penalty_cost", "penalty cost ($)")],
                  reports_path(cfg, "penalty_sensitivity.png"),
                  "Liquidation penalties (event engine, firewalls on)")  # fmt: skip

    real = daily_stats(base_daily)
    shuffles = placebo(vb, strat, sweep["placebo_shuffles"], cfg["seed"])
    p_value = placebo_p_value(real["net_pnl"], shuffles["net_pnl"])
    shuffles.to_csv(reports_path(cfg, "placebo.csv"), index=False)
    plot_placebo(shuffles["net_pnl"], real["net_pnl"], p_value, reports_path(cfg, "placebo.png"))

    days = [d.date() for d in main.daily.index]
    regimes = regime_table(main.daily, day_labels(cfg, days, load_bars))
    regimes.to_csv(reports_path(cfg, "regime_table.csv"))
    dsr = deflated_sharpe(main.daily["net"], trials)
    pd.Series(dsr).to_csv(reports_path(cfg, "deflated_sharpe.csv"), header=["value"])

    summary = {
        "strategy": choice["overrides"],
        "walk_forward": stats,
        "break_even_fee_rt": be_fee,
        "placebo": {
            "real_net_pnl": real["net_pnl"],
            "p_value": p_value,
            "shuffle_mean": float(shuffles["net_pnl"].mean()),
        },  # fmt: skip
        "deflated_sharpe": dsr,
    }
    if seeds:
        model, root = chosen_model(cfg), data_path(cfg, "seeds")
        table = pd.DataFrame(
            [seed_run(cfg, s, choice["overrides"], model, root) for s in sweep["seeds"]]
        )
        table.to_csv(reports_path(cfg, "seed_robustness.csv"), index=False)
        spread = table.drop(columns="seed").describe().loc[["mean", "min", "max"]]
        summary["seeds"] = spread.to_dict()
    with open(reports_path(cfg, "research_summary.yaml"), "w") as fh:
        yaml.safe_dump(summary, fh, sort_keys=False)
    return summary
