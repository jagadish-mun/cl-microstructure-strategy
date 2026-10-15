"""P&L and risk statistics for a backtest. Net-of-cost numbers come first."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from clstrat.backtest.engine import COST_ITEMS, BacktestResult

TRADING_DAYS = 252


def sharpe(daily: pd.Series) -> float:
    """Annualised Sharpe ratio of daily P&L: ``mean / std * sqrt(252)``."""
    sd = daily.std(ddof=1)
    return float(daily.mean() / sd * math.sqrt(TRADING_DAYS)) if sd > 0 else math.nan


def sortino(daily: pd.Series) -> float:
    """Annualised Sortino ratio: mean over downside deviation ``sqrt(mean(min(x, 0)^2))``."""
    downside = math.sqrt(np.mean(np.minimum(daily.to_numpy(), 0.0) ** 2))
    return float(daily.mean() / downside * math.sqrt(TRADING_DAYS)) if downside > 0 else math.nan


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough fall of the cumulative P&L curve, in dollars (>= 0)."""
    if equity.empty:
        return 0.0
    curve = np.concatenate([[0.0], equity.to_numpy()])
    return float(np.max(np.maximum.accumulate(curve) - curve))


def summarize(result: BacktestResult) -> dict[str, float]:
    """Headline statistics (dollars unless noted)."""
    daily, trades = result.daily, result.trades
    net, gross = daily["net"], daily["gross"]
    mdd = max_drawdown(result.equity)
    n = len(trades)
    pnl = trades["pnl_net"] if n else pd.Series(dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    contracts = float(trades["size"].sum()) if n else 0.0
    stats = {
        "net_pnl": float(net.sum()),
        "gross_pnl": float(gross.sum()),
        "net_sharpe": sharpe(net),
        "gross_sharpe": sharpe(gross),
        "net_sortino": sortino(net),
        "max_drawdown": mdd,
        "calmar": float(net.mean() * TRADING_DAYS / mdd) if mdd > 0 else math.nan,
        "days": len(daily),
        "trades": n,
        "hit_rate": float((pnl > 0).mean()) if n else math.nan,
        "avg_win": float(wins.mean()) if len(wins) else math.nan,
        "avg_loss": float(losses.mean()) if len(losses) else math.nan,
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else math.nan,
        "avg_holding_s": float(trades["holding_s"].mean()) if n else math.nan,
        "contracts_per_day": contracts / max(len(daily), 1),
        "pnl_per_contract": float(net.sum() / contracts) if contracts else math.nan,
        "worst_day": float(net.min()) if len(net) else math.nan,
    }
    stats |= {f"cost_{k}": float(daily[k].sum()) for k in COST_ITEMS}
    stats |= {f"fw_{name}": float(n) for name, n in firewall_counts(result).items()}
    return stats


def firewall_counts(result: BacktestResult) -> pd.Series:
    """Number of firewall triggers by firewall name."""
    log = result.firewall_log
    if log.empty:
        return pd.Series(dtype=int, name="triggers")
    return log["firewall"].value_counts().rename("triggers")
