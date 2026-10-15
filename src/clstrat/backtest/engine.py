"""Event-driven backtester: the reference implementation.

For each day the engine walks the decision times in order. At each one it marks the
position to mid, flattens at the end-of-day cut-off, asks the signal rule for a target
position, passes the order through the risk firewalls and executes what is allowed against
the book ``latency`` later. Every fill is booked to a trade episode (flat -> flat), so the
trade log carries entry/exit prices, costs, P&L, exit reason and firewall involvement.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
import yaml

from clstrat.backtest.execution import Book, ExecutionModel, Fill
from clstrat.backtest.firewall import FirewallSet, FirewallState, Routed
from clstrat.backtest.signal import Decision, Position, SignalRule, percentile_of
from clstrat.config import Config, data_path
from clstrat.contract import CL, SessionCalendar

COST_ITEMS = ("fees", "spread_cost", "slippage", "eod_penalty", "forced_penalty")


@dataclass
class Episode:
    """One round trip from flat back to flat."""

    side: int
    entry_ts: pd.Timestamp
    entry_lots: int = 0
    entry_notional: float = 0.0
    exit_lots: int = 0
    exit_notional: float = 0.0
    max_size: int = 0
    cash: float = 0.0
    costs: dict[str, float] = field(default_factory=lambda: dict.fromkeys(COST_ITEMS, 0.0))
    exit_ts: pd.Timestamp | None = None
    exit_reason: str = ""
    firewalls: set[str] = field(default_factory=set)

    def record(self) -> dict:
        """Trade-log row; ``pnl_net`` is after every cost, ``pnl_gross`` before any."""
        fees_pen = self.costs["fees"] + self.costs["eod_penalty"] + self.costs["forced_penalty"]
        net = self.cash - fees_pen
        return {
            "entry_ts": self.entry_ts,
            "exit_ts": self.exit_ts,
            "side": "long" if self.side > 0 else "short",
            "size": self.max_size,
            "entry_px": self.entry_notional / self.entry_lots,
            "exit_px": self.exit_notional / self.exit_lots if self.exit_lots else math.nan,
            **self.costs,
            "pnl_net": net,
            "pnl_gross": net + sum(self.costs.values()),
            "holding_s": (self.exit_ts - self.entry_ts).total_seconds() if self.exit_ts else 0.0,
            "exit_reason": self.exit_reason,
            "firewall": ",".join(sorted(self.firewalls)),
        }


@dataclass
class BacktestResult:
    """Outputs of one run."""

    daily: pd.DataFrame  # per day: net and gross P&L and each cost line item, in dollars
    equity: pd.Series  # cumulative net P&L marked to mid at every decision time
    trades: pd.DataFrame  # one row per episode
    firewall_log: pd.DataFrame  # one row per firewall trigger (empty without firewalls)


class Engine:
    """Reference event-driven backtester. Firewalls run when ``firewalls.enabled`` is set."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.cal = SessionCalendar.from_config(cfg)
        self.rule = SignalRule(cfg)
        self.exec = ExecutionModel(cfg)
        p = cfg["penalties"]
        self.eod_time = pd.Timestamp(p["eod_flatten_time"]).time()
        self.eod_pen = p["eod_penalty_ticks"] * CL.tick_value
        self.forced_pen = p["forced_liq_penalty_ticks"] * CL.tick_value
        self.horizon = cfg["target"]["horizon"]
        step = pd.Timedelta(cfg["sampling"]["decision_interval"]).total_seconds()
        if self.exec.latency_s >= step:
            raise ValueError("latency must be shorter than the decision interval")

    # -- bookkeeping -------------------------------------------------------------------

    def _reset(self) -> None:
        self.pos = Position()
        self.cash = 0.0
        self.episode: Episode | None = None
        self.episodes: list[dict] = []
        self.closed_pnls: list[float] = []
        self.day_costs = dict.fromkeys(COST_ITEMS, 0.0)
        self.firewalls = FirewallSet(self.cfg) if self.cfg["firewalls"]["enabled"] else None

    def _book_fill(
        self, fill: Fill, reason: str, penalty: str | None, firewalls: list[str]
    ) -> None:
        """Apply a fill to cash, position and the open episode(s); split reversals."""
        qty = fill.qty
        if self.pos.side != 0 and np.sign(qty) != self.pos.side and abs(qty) > abs(self.pos.qty):
            closing = -self.pos.qty
            share = abs(closing) / abs(qty)
            self._apply(fill, closing, share, reason, penalty, firewalls)
            self._apply(fill, qty - closing, 1 - share, reason, penalty, firewalls)
        else:
            self._apply(fill, qty, 1.0, reason, penalty, firewalls)

    def _apply(
        self, fill: Fill, qty: int, share: float, reason: str, penalty: str | None, fw: list[str]
    ) -> None:
        lots = abs(qty)
        costs = {
            "fees": fill.fee * share,
            "spread_cost": fill.spread_cost * share,
            "slippage": fill.slippage * share,
            "eod_penalty": self.eod_pen * lots if penalty == "eod" else 0.0,
            "forced_penalty": self.forced_pen * lots if penalty == "forced" else 0.0,
        }
        flow = -qty * fill.price * CL.multiplier
        self.cash += flow - costs["fees"] - costs["eod_penalty"] - costs["forced_penalty"]
        for k, v in costs.items():
            self.day_costs[k] += v
        if self.episode is None:
            self.episode = Episode(side=int(np.sign(qty)), entry_ts=fill.ts)
        ep = self.episode
        ep.cash += flow
        ep.firewalls.update(fw)
        for k, v in costs.items():
            ep.costs[k] += v
        new_qty = self.pos.qty + qty
        if abs(new_qty) > abs(self.pos.qty):  # opening or adding
            ep.entry_lots += lots
            ep.entry_notional += lots * fill.price
            entry_px = ep.entry_notional / ep.entry_lots
            self.pos = Position(new_qty, entry_px, self.pos.entry_ts or fill.ts)
        else:  # reducing or closing
            ep.exit_lots += lots
            ep.exit_notional += lots * fill.price
            self.pos = Position(new_qty, self.pos.entry_px, self.pos.entry_ts)
        ep.max_size = max(ep.max_size, abs(new_qty))
        if new_qty == 0:
            ep.exit_ts, ep.exit_reason = fill.ts, reason
            self.episodes.append(ep.record())
            self.closed_pnls.append(self.episodes[-1]["pnl_net"])
            self.episode = None
            self.pos = Position()

    def trade(
        self, book: Book, bar: int, qty: int, reason: str, penalty: str | None = None,
        firewalls: list[str] | None = None,
    ) -> Fill:  # fmt: skip
        """Execute ``qty`` decided at ``bar`` (filled at ``bar + latency``) and book it."""
        fill = self.exec.market(book, self.exec.fill_index(book, bar), qty)
        self._book_fill(fill, reason, penalty, firewalls or [])
        return fill

    def flatten(self, book: Book, bar: int, reason: str, penalty: str, firewalls=None) -> None:
        """Close any open position with a market order plus the given penalty."""
        if self.pos.qty != 0:
            self.trade(book, bar, -self.pos.qty, reason, penalty, firewalls)

    def mark(self, book: Book, bar: int) -> float:
        """Cumulative net P&L, marking the position to mid."""
        return self.cash + self.pos.qty * book.mid[bar] * CL.multiplier

    # -- main loop -----------------------------------------------------------------------

    def _decisions(self, preds: pd.DataFrame, book: Book, quantiles: np.ndarray | None):
        """Decision inputs for one day, aligned to the book's 1 s bars."""
        bars = book.ts.get_indexer(pd.DatetimeIndex(preds["ts"]))
        dm = pd.Series(book.mid).diff() / CL.tick_size
        rv = (dm.rolling(self.horizon, min_periods=10).std() * math.sqrt(self.horizon)).to_numpy()
        pct = (
            percentile_of(preds["pred"].to_numpy(), quantiles)
            if quantiles is not None
            else np.full(len(preds), np.nan)
        )
        for k, (ts, pred) in enumerate(zip(preds["ts"], preds["pred"], strict=True)):
            i = bars[k]
            yield i, Decision(ts, pred, pct[k], book.mid[i], book.spread_ticks(i), rv[i])

    def run_day(
        self, day: date, preds: pd.DataFrame, bars: pd.DataFrame, quantiles: np.ndarray | None,
        equity_points: list, start_equity: float, peak: float,
    ) -> tuple[float, float]:  # fmt: skip
        """Trade one session; returns (end equity, peak equity)."""
        book = Book.from_bars(bars)
        eod = self.cal.at(day, self.eod_time)
        day_start = self.mark(book, 0)
        equity = start_equity
        for i, d in self._decisions(preds, book, quantiles):
            equity = self.mark(book, i)
            peak = max(peak, equity)
            if d.ts >= eod:
                self.flatten(book, i, "eod", "eod")
                break
            target, reason = self.rule.target(d, self.pos)
            order = Routed(target - self.pos.qty, reason)
            if self.firewalls is not None:
                state = FirewallState(
                    d.ts, i, book, self.pos, equity - day_start, equity, peak, self.closed_pnls
                )
                order = self.firewalls.route(state, order.qty, order.reason)
            if order.qty != 0:
                self.trade(book, i, order.qty, order.reason, order.penalty, order.firewalls)
            equity_points.append((d.ts, self.mark(book, i)))
        self.flatten(book, len(book) - 1 - self.exec.latency_s, "eod", "eod")
        end = self.mark(book, len(book) - 1)
        equity_points.append((book.ts[-1], end))
        return end, max(peak, end)

    def run(
        self,
        predictions: pd.DataFrame,
        load_bars: Callable[[date], pd.DataFrame],
        quantiles: dict[int, np.ndarray] | None = None,
    ) -> BacktestResult:
        """Backtest ``predictions`` (columns ``ts``, ``day``, ``fold``, ``pred``).

        ``quantiles`` maps fold -> training-prediction quantile grid (percentile mode).
        """
        self._reset()
        daily, points = [], []
        equity = peak = 0.0
        for day, preds in predictions.sort_values("ts").groupby("day", sort=True):
            fold = int(preds["fold"].iloc[0])
            q = quantiles.get(fold) if quantiles else None
            self.day_costs = dict.fromkeys(COST_ITEMS, 0.0)
            start = equity
            equity, peak = self.run_day(
                pd.Timestamp(day).date(), preds, load_bars(pd.Timestamp(day).date()), q,
                points, equity, peak,
            )  # fmt: skip
            net = equity - start
            daily.append({"day": pd.Timestamp(day), "net": net} | self.day_costs | {
                "gross": net + sum(self.day_costs.values())})  # fmt: skip
        trades = pd.DataFrame(self.episodes)
        eq = pd.Series(dict(points), name="equity").sort_index()
        return BacktestResult(
            daily=pd.DataFrame(daily).set_index("day"),
            equity=eq,
            trades=trades,
            firewall_log=self.firewalls.log_frame() if self.firewalls else FirewallSet.empty_log(),
        )


def backtest_inputs(
    cfg: Config, model: str | None = None
) -> tuple[pd.DataFrame, dict[int, np.ndarray] | None, Callable[[date], pd.DataFrame]]:
    """Walk-forward predictions of ``model`` (default: the one chosen by ``train``), the
    per-fold training-prediction quantiles (percentile mode only) and a 1 s bar loader."""
    with open(data_path(cfg, "models", "selection.yaml")) as fh:
        model = model or yaml.safe_load(fh)["best_model"]
    preds = pd.read_parquet(data_path(cfg, "models", "predictions.parquet"))
    preds = preds[["ts", "day", "fold", f"pred_{model}"]].rename(columns={f"pred_{model}": "pred"})
    quantiles = None
    if cfg["strategy"]["threshold_mode"] == "percentile":
        q = pd.read_parquet(data_path(cfg, "models", "train_quantiles.parquet"))
        quantiles = {fold: row.to_numpy() for (fold, m), row in q.iterrows() if m == model}

    def load_bars(day: date) -> pd.DataFrame:
        return pd.read_parquet(data_path(cfg, "bars", f"{day.isoformat()}.parquet"))

    return preds, quantiles, load_bars
