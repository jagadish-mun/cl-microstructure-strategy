"""Fast approximate backtester for parameter sweeps.

The path-dependent part (hysteresis, stops, maximum holding time, end-of-day cut-off) is a
small numba state machine over decision times that mirrors ``SignalRule``; fills, costs and
P&L are then computed with array operations. Day arrays are prepared once, so a sweep only
re-runs the kernel.

The one approximation relative to the event-driven ``Engine`` is that firewalls are not
modelled; with firewalls off the two engines agree to the cent (see the parity test).
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import pandas as pd
from numba import njit

from clstrat.backtest.engine import COST_ITEMS
from clstrat.backtest.execution import Book
from clstrat.backtest.signal import percentile_of
from clstrat.config import Config
from clstrat.contract import CL, SessionCalendar

SIZING = {"fixed": 0, "linear": 1, "vol_target": 2}


@dataclass(frozen=True)
class DayArrays:
    """One day's decision inputs and 1 s book, aligned for the kernel."""

    day: date
    fold: int
    bar: np.ndarray  # decision -> bar index
    sec: np.ndarray  # decision time, seconds after the open
    pred: np.ndarray
    pct: np.ndarray
    spread: np.ndarray  # ticks at the decision bar
    rv: np.ndarray  # ticks, over the target horizon
    book: Book


@njit(cache=True)
def _order_price(px, sz, lots, side, coef, tick, levels):
    """Average fill price of one market order (same walk as ``ExecutionModel._walk``)."""
    left, paid, last = lots, 0.0, px[0]
    for k in range(levels):
        take = min(left, sz[k])
        if take > 0:
            paid += take * px[k]
            last = px[k]
            left -= take
        if left == 0:
            return paid / lots
    extra = coef * math.sqrt(left / max(sz[0], 1.0)) * tick
    return (paid + left * (last + side * extra)) / lots


@njit(cache=True)
def _positions(
    bar, sec, pred, pct, spread, rv, mid, bid_px, bid_sz, ask_px, ask_sz, levels, coef,
    eod_sec, lat, n_bars, pct_mode, th_in, th_out, p_in, p_out, require_edge, fee_ticks,
    stop, take, max_hold, sizing, base, max_size, vol_target, tick,
):  # fmt: skip
    """Target position after each decision (mirrors ``SignalRule.target``) and the index of
    the end-of-day decision (-1 if none). The entry price is the order's average fill price
    and the entry time its fill bar, as in the event-driven engine."""
    n = pred.shape[0]
    out = np.zeros(n, np.int64)
    pos, entry_px, entry_sec, eod_k = 0, 0.0, 0.0, -1
    for k in range(n):
        if sec[k] >= eod_sec:
            eod_k = k
            out[k:] = 0
            break
        cost = spread[k] + fee_ticks
        target = pos
        strong_long = (pct[k] >= p_in) if pct_mode else (pred[k] > th_in)
        strong_short = (pct[k] <= 1 - p_in) if pct_mode else (-pred[k] > th_in)
        if require_edge:
            strong_long = strong_long and pred[k] - cost > 0
            strong_short = strong_short and -pred[k] - cost > 0
        if sizing == 0:
            size = base
        elif sizing == 1:
            size = min(max(round(base * abs(pred[k]) / th_in), 1), max_size)
        else:
            raw = vol_target / rv[k] if rv[k] > 0 else base
            size = min(max(round(raw), 1), max_size)
        if pos == 0:
            if strong_long:
                target = size
            elif strong_short:
                target = -size
        else:
            side = 1 if pos > 0 else -1
            pnl = side * (mid[bar[k]] - entry_px) / tick
            weak = (
                ((pct[k] < p_out) if side > 0 else (pct[k] > 1 - p_out))
                if pct_mode
                else side * pred[k] < th_out
            )
            if pnl <= -stop or pnl >= take or sec[k] - entry_sec >= max_hold:
                target = 0
            elif (side > 0 and strong_short) or (side < 0 and strong_long):
                target = -side * size
            elif weak:
                target = 0
        if target != pos:
            f = min(bar[k] + lat, n_bars - 1)
            if target != 0 and (pos == 0 or (target > 0) != (pos > 0)):
                lots = abs(target - pos)
                if target > 0:
                    entry_px = _order_price(ask_px[f], ask_sz[f], lots, 1, coef, tick, levels)
                else:
                    entry_px = _order_price(bid_px[f], bid_sz[f], lots, -1, coef, tick, levels)
                entry_sec = f + 1.0  # bar f ends f + 1 seconds after the open
            pos = target
        out[k] = pos
    return out, eod_k


def walk_levels(
    px: np.ndarray, sz: np.ndarray, lots: np.ndarray, side: np.ndarray, e: Config
) -> tuple[np.ndarray, np.ndarray]:
    """Array version of ``ExecutionModel._walk`` for many orders at once.

    ``px``/``sz`` are ``(orders, levels)`` on the side being taken. Returns each order's total
    price paid (sum of lots x price) and its slippage versus L1 (sum of lots x |price - L1|).
    """
    if e["slippage_model"] != "l2_or_sqrt":
        px, sz = px[:, :1], sz[:, :1]
    before = np.cumsum(sz, axis=1) - sz
    take = np.clip(lots[:, None] - before, 0, sz)
    left = lots - take.sum(axis=1)
    used = take > 0
    last_level = np.where(used.any(axis=1), used.shape[1] - 1 - np.argmax(used[:, ::-1], axis=1), 0)
    last = px[np.arange(len(px)), last_level]
    extra = e["sqrt_impact_coef"] * np.sqrt(left / np.maximum(sz[:, 0], 1.0)) * CL.tick_size
    left_px = last + side * extra
    notional = (take * px).sum(axis=1) + left * left_px
    slip = (take * np.abs(px - px[:, :1])).sum(axis=1) + left * np.abs(left_px - px[:, 0])
    return notional, slip


class VectorBacktester:
    """Prepare day arrays once; ``run(cfg)`` returns daily P&L with cost line items."""

    def __init__(
        self,
        cfg: Config,
        predictions: pd.DataFrame,
        load_bars: Callable[[date], pd.DataFrame],
        quantiles: dict[int, np.ndarray] | None = None,
    ) -> None:
        self.cal = SessionCalendar.from_config(cfg)
        horizon = cfg["target"]["horizon"]
        self.days = []
        for day, preds in predictions.sort_values("ts").groupby("day", sort=True):
            d = pd.Timestamp(day).date()
            book = Book.from_bars(load_bars(d))
            bars = book.ts.get_indexer(pd.DatetimeIndex(preds["ts"]))
            dm = pd.Series(book.mid).diff() / CL.tick_size
            rv = (dm.rolling(horizon, min_periods=10).std() * math.sqrt(horizon)).to_numpy()
            fold = int(preds["fold"].iloc[0])
            p = preds["pred"].to_numpy(float)
            q = quantiles.get(fold) if quantiles else None
            spread = np.round((book.ask_px[bars, 0] - book.bid_px[bars, 0]) / CL.tick_size)
            self.days.append(
                DayArrays(
                    day=d, fold=fold, bar=bars, sec=bars + 1.0, pred=p,
                    pct=percentile_of(p, q) if q is not None else np.full(len(p), np.nan),
                    spread=spread, rv=rv[bars], book=book,
                )
            )  # fmt: skip

    def _day(self, a: DayArrays, cfg: Config) -> dict:
        s, e, pen = cfg["strategy"], cfg["execution"], cfg["penalties"]
        book = a.book
        lat = math.ceil(e["latency_ms"] / 1000)
        depth = e["slippage_model"] == "l2_or_sqrt"
        eod = self.cal.at(a.day, pd.Timestamp(pen["eod_flatten_time"]).time())
        eod_sec = (eod - self.cal.session_bounds(a.day)[0]).total_seconds()
        pos, eod_k = _positions(
            a.bar, a.sec, a.pred, a.pct, a.spread, a.rv, book.mid, book.bid_px, book.bid_sz,
            book.ask_px, book.ask_sz, book.ask_px.shape[1] if depth else 1,
            e["sqrt_impact_coef"], eod_sec, lat, len(book), s["threshold_mode"] == "percentile",
            s["theta_entry_ticks"], s["theta_exit_ticks"], s["pct_entry"], s["pct_exit"],
            s["require_edge"], e["fee_per_contract_rt"] / CL.tick_value, s["stop_loss_ticks"],
            s["take_profit_ticks"], s["max_hold_seconds"], SIZING[s["sizing"]], s["base_size"],
            s["max_size"], s["vol_target_ticks"], CL.tick_size,
        )  # fmt: skip
        qty = np.diff(np.concatenate([[0], pos]))
        fill = np.minimum(a.bar + lat, len(book) - 1)
        eod_flags = np.zeros(len(qty), bool)
        if eod_k >= 0:
            eod_flags[eod_k] = True
        elif pos[-1] != 0:  # no decision reached the cut-off: flatten at the last bar
            qty = np.append(qty, -pos[-1])
            fill = np.append(fill, len(book) - 1)
            eod_flags = np.append(eod_flags, True)
        trade = qty != 0
        qty, fill, eod_flags = qty[trade], fill[trade], eod_flags[trade]
        buy = (qty > 0)[:, None]
        px = np.where(buy, book.ask_px[fill], book.bid_px[fill])
        sz = np.where(buy, book.ask_sz[fill], book.bid_sz[fill])
        lots = np.abs(qty)
        eod_lots = float(np.sum(lots[eod_flags]))
        notional, slip = walk_levels(px, sz, lots, np.sign(qty), e)
        l1 = px[:, 0]
        mid = book.mid[fill]
        flows = -np.sign(qty) * notional * CL.multiplier
        costs = {
            "fees": float(np.sum(lots) * e["fee_per_contract_rt"] / 2),
            "spread_cost": float(np.sum(lots * np.abs(l1 - mid)) * CL.multiplier),
            "slippage": float(np.sum(slip) * CL.multiplier),
            "eod_penalty": eod_lots * pen["eod_penalty_ticks"] * CL.tick_value,
            "forced_penalty": 0.0,
        }
        net = float(np.sum(flows)) - costs["fees"] - costs["eod_penalty"]
        entries = int(np.sum((pos != 0) & (np.concatenate([[0], pos[:-1]]) != pos)))
        return {"day": pd.Timestamp(a.day), "net": net} | costs | {
            "gross": net + sum(costs.values()), "entries": entries}  # fmt: skip

    def shuffled(self, rng: np.random.Generator) -> VectorBacktester:
        """A copy whose predictions (and their percentiles) are permuted within each day."""
        clone = copy.copy(self)
        clone.days = []
        for a in self.days:
            perm = rng.permutation(len(a.pred))
            clone.days.append(replace(a, pred=a.pred[perm], pct=a.pct[perm]))
        return clone

    def run(self, cfg: Config) -> pd.DataFrame:
        """Daily net/gross P&L and costs (same columns as ``Engine`` plus ``entries``)."""
        daily = pd.DataFrame([self._day(a, cfg) for a in self.days]).set_index("day")
        return daily[["net", *COST_ITEMS, "gross", "entries"]]
