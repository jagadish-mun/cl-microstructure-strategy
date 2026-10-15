"""Parameter sweeps on the walk-forward test days (never the holdout).

Threshold, percentile and cost grids run on the vectorized backtester (firewalls off).
Penalty sensitivity and the firewall comparison need the firewalls, so they use the
event-driven engine.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from datetime import date

import numpy as np
import pandas as pd

from clstrat.backtest.engine import BacktestResult, Engine
from clstrat.backtest.metrics import max_drawdown, sharpe, summarize
from clstrat.backtest.vectorized import VectorBacktester
from clstrat.config import Config, with_overrides

EngineRun = Callable[[Config], BacktestResult]


def daily_stats(daily: pd.DataFrame) -> dict[str, float]:
    """Headline numbers from a daily P&L table (vectorized runs have no intraday marks)."""
    return {
        "net_pnl": float(daily["net"].sum()),
        "gross_pnl": float(daily["gross"].sum()),
        "net_sharpe": sharpe(daily["net"]),
        "max_drawdown_daily": max_drawdown(daily["net"].cumsum()),
        "trades": int(daily["entries"].sum()),
        "costs": float(daily["gross"].sum() - daily["net"].sum()),
    }


def grid_sweep(
    vb: VectorBacktester,
    cfg: Config,
    grid: dict[str, list],
    keep: Callable[[dict], bool] = lambda _params: True,
) -> pd.DataFrame:
    """Run every combination of dotted-key overrides in ``grid`` that ``keep`` accepts."""
    rows = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values, strict=True))
        if keep(params):
            rows.append(params | daily_stats(vb.run(with_overrides(cfg, params))))
    return pd.DataFrame(rows)


def threshold_sweep(vb: VectorBacktester, cfg: Config, grid: Config) -> pd.DataFrame:
    """Tick thresholds: every (θ_entry, θ_exit) with θ_exit < θ_entry."""
    base = with_overrides(cfg, {"strategy.threshold_mode": "ticks"})
    g = {f"strategy.{k}": v for k, v in grid.items()}

    def valid(p: dict) -> bool:
        return p["strategy.theta_exit_ticks"] < p["strategy.theta_entry_ticks"]

    return grid_sweep(vb, base, g, valid)


def percentile_sweep(vb: VectorBacktester, cfg: Config, grid: Config) -> pd.DataFrame:
    """Percentile thresholds on the fold's training predictions, exit below entry."""
    base = with_overrides(cfg, {"strategy.threshold_mode": "percentile"})
    g = {f"strategy.{k}": v for k, v in grid.items()}

    def valid(p: dict) -> bool:
        return p["strategy.pct_exit"] < p["strategy.pct_entry"]

    return grid_sweep(vb, base, g, valid)


def choose_strategy(sweeps: dict[str, pd.DataFrame], min_trades: int) -> dict:
    """Best net Sharpe across all sweeps among configs with at least ``min_trades`` trades.

    Returns the winning overrides (``strategy.*`` keys) and its sweep statistics.
    """
    candidates = []
    for mode, table in sweeps.items():
        eligible = table[table["trades"] >= min_trades]
        if not eligible.empty:
            candidates.append((mode, eligible.loc[eligible["net_sharpe"].idxmax()]))
    if not candidates:
        raise ValueError("no configuration met the minimum trade count")
    mode, row = max(candidates, key=lambda c: c[1]["net_sharpe"])
    params = {k: float(v) for k, v in row.items() if k.startswith("strategy.")}
    stats = {k: float(v) for k, v in row.items() if not k.startswith("strategy.")}
    return {"overrides": {"strategy.threshold_mode": mode} | params, "stats": stats}


def cost_sensitivity(vb: VectorBacktester, cfg: Config, grid: Config) -> dict[str, pd.DataFrame]:
    """Net P&L as each cost parameter rises, everything else held at ``cfg``."""
    return {k: grid_sweep(vb, cfg, {f"execution.{k}": v}) for k, v in grid.items()}


def break_even_fee(daily: pd.DataFrame, fee_rt: float) -> float:
    """Round-trip fee per contract at which this trade list nets exactly zero.

    With ``RT`` round-trip contracts, ``net(fee) = net + (fee_rt - fee) * RT``.
    """
    fees = daily["fees"].sum()
    if fees == 0:
        return math.nan
    round_trips = fees / fee_rt
    return float(fee_rt + daily["net"].sum() / round_trips)


def penalty_sensitivity(run: EngineRun, cfg: Config, grid: Config) -> pd.DataFrame:
    """Event-engine results over the EOD x forced-liquidation penalty grid."""
    rows = []
    for eod, forced in itertools.product(
        grid["eod_penalty_ticks"], grid["forced_liq_penalty_ticks"]
    ):
        s = summarize(
            run(with_overrides(cfg, {
                "penalties.eod_penalty_ticks": eod,
                "penalties.forced_liq_penalty_ticks": forced,
            }))
        )  # fmt: skip
        rows.append(
            {
                "eod_penalty_ticks": eod,
                "forced_liq_penalty_ticks": forced,
                "net_pnl": s["net_pnl"],
                "net_sharpe": s["net_sharpe"],
                "penalty_cost": s["cost_eod_penalty"] + s["cost_forced_penalty"],
            }
        )
    return pd.DataFrame(rows)


def tail_stats(result: BacktestResult) -> dict[str, float]:
    """Net P&L, drawdown and daily tail loss (worst day, 10% expected shortfall)."""
    net = result.daily["net"].sort_values()
    k = max(1, math.ceil(0.1 * len(net)))
    s = summarize(result)
    return {
        "net_pnl": s["net_pnl"],
        "net_sharpe": s["net_sharpe"],
        "max_drawdown": s["max_drawdown"],
        "worst_day": float(net.iloc[0]),
        "es_10pct_day": float(net.iloc[:k].mean()),
        "trades": s["trades"],
        "forced_penalty": s["cost_forced_penalty"],
    }


def firewall_comparison(
    run: EngineRun, cfg: Config
) -> tuple[pd.DataFrame, dict[str, BacktestResult]]:
    """Event-engine runs with firewalls on and off: a tail-risk table and both results."""
    results = {
        label: run(with_overrides(cfg, {"firewalls.enabled": enabled}))
        for label, enabled in (("firewalls on", True), ("firewalls off", False))
    }
    return pd.DataFrame({k: tail_stats(r) for k, r in results.items()}).T, results


def engine_runner(
    preds: pd.DataFrame,
    load_bars: Callable[[date], pd.DataFrame],
    quantiles: dict[int, np.ndarray] | None,
) -> EngineRun:
    """``run(cfg)`` over fixed predictions, with each day's bars loaded once."""
    cache: dict[date, pd.DataFrame] = {}

    def bars(day: date) -> pd.DataFrame:
        if day not in cache:
            cache[day] = load_bars(day)
        return cache[day]

    def run(cfg: Config) -> BacktestResult:
        return Engine(cfg).run(preds, bars, quantiles)

    return run
