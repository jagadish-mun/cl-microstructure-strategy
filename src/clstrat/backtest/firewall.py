"""Risk firewalls. Each is a small class whose ``check(state, qty)`` returns ``None`` (allow)
or a ``Verdict`` to reduce or block the order, or to force-liquidate the position.

Checks run in this fixed order (``FIREWALL_ORDER``); the first liquidation wins and later
checks see the order as already modified:

1. ``max_drawdown``   run equity below peak - limit: liquidate, halt for the rest of the run
2. ``daily_loss``     day P&L below -limit: liquidate, halt for the rest of the day
3. ``loss_streak``    N losing trades in a row: no new risk for T minutes
4. ``news_blackout``  no new risk around the EIA release (optionally flatten at its start)
5. ``vol_breaker``    short-window vol > M x its trailing median: no new risk
6. ``stale_data``     last quote older than X ms: block every order
7. ``spread_guard``   spread wider than k ticks: no new risk
8. ``price_band``     expected fill more than N ticks from the mid: block
9. ``max_order_size`` clip the order size
10. ``max_position``  clip so that |position| stays within the limit
11. ``order_throttle`` at most K orders per rolling minute: no new risk

"No new risk" firewalls let risk-reducing orders through and cut a reversal down to its
closing part. Forced liquidations bypass every other check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from clstrat.backtest.execution import Book, ExecutionModel
from clstrat.backtest.signal import Position
from clstrat.config import Config
from clstrat.contract import CL, SessionCalendar

LIQUIDATE = "liquidate"


@dataclass(frozen=True)
class Verdict:
    """A firewall's decision about one proposed order."""

    action: str  # reduce | block | liquidate
    qty: int
    reason: str
    exit_reason: str | None = None  # overrides the order's exit reason (e.g. news flatten)


@dataclass
class FirewallState:
    """Everything a firewall may look at when an order is proposed."""

    ts: pd.Timestamp
    bar: int
    book: Book
    position: Position
    day_pnl: float
    equity: float
    peak_equity: float
    closed_pnls: list[float]


def reducing_part(pos: int, qty: int) -> int:
    """The part of ``qty`` that only reduces risk: all of it for a pure reduction, the closing
    leg of a reversal, nothing for an opening or increasing order."""
    if pos == 0 or np.sign(qty) == np.sign(pos):
        return 0
    return qty if abs(qty) <= abs(pos) else -pos


def _no_new_risk(s: FirewallState, qty: int, reason: str) -> Verdict | None:
    """Restrict an order to its risk-reducing part (None if nothing changes)."""
    allowed = reducing_part(s.position.qty, qty)
    if allowed == qty:
        return None
    return Verdict("reduce" if allowed else "block", allowed, reason)


class MaxDrawdownHalt:
    name = "max_drawdown"

    def __init__(self, limit: float) -> None:
        self.limit = limit
        self.halted = False

    def check(self, s: FirewallState, qty: int) -> Verdict | None:
        if not self.halted and s.equity < s.peak_equity - self.limit:
            self.halted = True
            return Verdict(LIQUIDATE, -s.position.qty, f"drawdown {s.peak_equity - s.equity:.0f}")
        return Verdict("block", 0, "run halted") if self.halted and qty else None


class DailyLossLimit:
    name = "daily_loss"

    def __init__(self, limit: float) -> None:
        self.limit = limit
        self.halted_day = None

    def check(self, s: FirewallState, qty: int) -> Verdict | None:
        today = s.ts.date()
        if self.halted_day != today and s.day_pnl < -self.limit:
            self.halted_day = today
            return Verdict(LIQUIDATE, -s.position.qty, f"day P&L {s.day_pnl:.0f}")
        return Verdict("block", 0, "day halted") if self.halted_day == today and qty else None


class LossStreakCooldown:
    name = "loss_streak"

    def __init__(self, n: int, cooldown_min: float) -> None:
        self.n = n
        self.cooldown = pd.Timedelta(minutes=cooldown_min)
        self.until: pd.Timestamp | None = None
        self.seen = 0  # closed trades already counted toward the current streak

    def check(self, s: FirewallState, qty: int) -> Verdict | None:
        recent = s.closed_pnls[self.seen :]
        if len(recent) >= self.n and all(p <= 0 for p in recent[-self.n :]):
            self.until = s.ts + self.cooldown
            self.seen = len(s.closed_pnls)
        elif recent and recent[-1] > 0:
            self.seen = len(s.closed_pnls)
        if self.until is not None and s.ts < self.until:
            return _no_new_risk(s, qty, f"cooling down until {self.until:%H:%M:%S}")
        return None


class NewsBlackout:
    name = "news_blackout"

    def __init__(self, cal: SessionCalendar, window: list[str], flatten: bool) -> None:
        self.cal = cal
        self.start, self.end = (pd.Timestamp(t).time() for t in window)
        self.flatten = flatten

    def check(self, s: FirewallState, qty: int) -> Verdict | None:
        day = s.ts.date()
        if not self.cal.is_eia_day(day):
            return None
        if not self.cal.at(day, self.start) <= s.ts < self.cal.at(day, self.end):
            return None
        if self.flatten and s.position.qty:
            return Verdict("reduce", -s.position.qty, "flatten before EIA", "news_flatten")
        return _no_new_risk(s, qty, "EIA blackout")


class VolBreaker:
    name = "vol_breaker"

    def __init__(self, mult: float, short_s: int, median_s: int) -> None:
        self.mult, self.short_s, self.median_s = mult, short_s, median_s
        self._book: Book | None = None
        self._ratio = np.empty(0)

    def ratio(self, book: Book) -> np.ndarray:
        """Causal ratio of short-window realized vol to its trailing median (cached per day)."""
        if book is not self._book:
            dm = pd.Series(book.mid).diff()
            rv = np.sqrt((dm**2).rolling(self.short_s, min_periods=self.short_s).mean())
            med = rv.rolling(self.median_s, min_periods=self.short_s).median()
            self._book, self._ratio = book, (rv / med.where(med > 0)).to_numpy()
        return self._ratio

    def check(self, s: FirewallState, qty: int) -> Verdict | None:
        r = self.ratio(s.book)[s.bar]
        if np.isfinite(r) and r > self.mult:
            return _no_new_risk(s, qty, f"vol {r:.1f}x median")
        return None


class StaleData:
    name = "stale_data"

    def __init__(self, max_ms: float) -> None:
        self.max_s = max_ms / 1000

    def check(self, s: FirewallState, qty: int) -> Verdict | None:
        age = s.book.quote_age[s.bar]
        if qty and age > self.max_s:
            return Verdict("block", 0, f"quote age {age:.1f}s")
        return None


class SpreadGuard:
    name = "spread_guard"

    def __init__(self, max_ticks: float) -> None:
        self.max_ticks = max_ticks

    def check(self, s: FirewallState, qty: int) -> Verdict | None:
        spread = s.book.spread_ticks(s.bar)
        return _no_new_risk(s, qty, f"spread {spread} ticks") if spread > self.max_ticks else None


class PriceBand:
    name = "price_band"

    def __init__(self, band_ticks: float, execution: ExecutionModel) -> None:
        self.band = band_ticks
        self.execution = execution

    def check(self, s: FirewallState, qty: int) -> Verdict | None:
        if not qty:
            return None
        fill = self.execution.market(s.book, s.bar, qty)
        off = abs(fill.price - s.book.mid[s.bar]) / CL.tick_size
        if off > self.band:
            return Verdict("block", 0, f"expected fill {off:.1f} ticks from mid")
        return None


class MaxOrderSize:
    name = "max_order_size"

    def __init__(self, max_lots: int) -> None:
        self.max_lots = max_lots

    def check(self, s: FirewallState, qty: int) -> Verdict | None:
        if abs(qty) > self.max_lots:
            return Verdict("reduce", int(np.sign(qty)) * self.max_lots, f"order {abs(qty)} lots")
        return None


class MaxPosition:
    name = "max_position"

    def __init__(self, max_lots: int) -> None:
        self.max_lots = max_lots

    def check(self, s: FirewallState, qty: int) -> Verdict | None:
        target = s.position.qty + qty
        if abs(target) <= self.max_lots:
            return None
        allowed = int(np.clip(target, -self.max_lots, self.max_lots)) - s.position.qty
        return Verdict("reduce" if allowed else "block", allowed, f"position {target} lots")


class OrderThrottle:
    name = "order_throttle"

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self.sent: list[pd.Timestamp] = []

    def check(self, s: FirewallState, qty: int) -> Verdict | None:
        window_start = s.ts - pd.Timedelta(minutes=1)
        self.sent = [t for t in self.sent if t > window_start]
        if qty and len(self.sent) >= self.per_minute:
            return _no_new_risk(s, qty, f"{len(self.sent)} orders in the last minute")
        return None

    def record(self, ts: pd.Timestamp) -> None:
        self.sent.append(ts)


FIREWALL_ORDER = (
    "max_drawdown", "daily_loss", "loss_streak", "news_blackout", "vol_breaker", "stale_data",
    "spread_guard", "price_band", "max_order_size", "max_position", "order_throttle",
)  # fmt: skip


@dataclass
class Routed:
    """An order after the firewalls: final size, reason, firewalls involved and any penalty."""

    qty: int
    reason: str
    firewalls: list[str] = field(default_factory=list)
    penalty: str | None = None


class FirewallSet:
    """Runs every firewall in ``FIREWALL_ORDER`` and logs each trigger."""

    def __init__(self, cfg: Config) -> None:
        f = cfg["firewalls"]
        execution = ExecutionModel(cfg)
        self.throttle = OrderThrottle(f["max_orders_per_min"])
        walls = [
            MaxDrawdownHalt(f["max_drawdown_usd"]),
            DailyLossLimit(f["daily_loss_limit_usd"]),
            LossStreakCooldown(f["loss_streak_n"], f["loss_streak_cooldown_min"]),
            NewsBlackout(SessionCalendar.from_config(cfg), f["news_blackout"], f["news_flatten"]),
            VolBreaker(f["vol_breaker_mult"], f["vol_breaker_short_s"], f["vol_breaker_median_s"]),
            StaleData(f["stale_quote_ms"]),
            SpreadGuard(f["max_spread_ticks"]),
            PriceBand(f["price_band_ticks"], execution),
            MaxOrderSize(f["max_order_size"]),
            MaxPosition(f["max_position"]),
            self.throttle,
        ]
        if tuple(w.name for w in walls) != FIREWALL_ORDER:
            raise RuntimeError("firewalls are out of the documented order")
        self.walls = walls
        self.log: list[dict] = []

    def route(self, s: FirewallState, qty: int, reason: str) -> Routed:
        """Apply every firewall to a proposed order (``qty`` may be 0: kill switches and the
        news flatten still run)."""
        fired: list[str] = []
        for wall in self.walls:
            verdict = wall.check(s, qty)
            if verdict is None:
                continue
            self.log.append(
                {"ts": s.ts, "firewall": wall.name, "action": verdict.action,
                 "reason": verdict.reason, "qty_before": qty, "qty_after": verdict.qty}
            )  # fmt: skip
            fired.append(wall.name)
            if verdict.action == LIQUIDATE:
                return Routed(verdict.qty, "forced_liquidation", fired, "forced")
            qty = verdict.qty
            reason = verdict.exit_reason or reason
        if qty:
            self.throttle.record(s.ts)
        return Routed(qty, reason, fired)

    LOG_COLUMNS = ("ts", "firewall", "action", "reason", "qty_before", "qty_after")

    def log_frame(self) -> pd.DataFrame:
        """Trigger log as a DataFrame."""
        return pd.DataFrame(self.log, columns=list(self.LOG_COLUMNS))

    @classmethod
    def empty_log(cls) -> pd.DataFrame:
        """An empty trigger log (firewalls disabled)."""
        return pd.DataFrame(columns=list(cls.LOG_COLUMNS))
