"""One test per firewall: trigger it and check the routed order and the log entry."""

import numpy as np
import pandas as pd
import pytest
from _market import OPEN, make_bars

from clstrat.backtest.engine import Engine
from clstrat.backtest.execution import Book
from clstrat.backtest.firewall import FIREWALL_ORDER, FirewallSet, FirewallState, reducing_part
from clstrat.backtest.signal import Position
from clstrat.config import load_config

FLAT = make_bars(np.full(400, 75.00))


def _walls(**overrides) -> FirewallSet:
    return FirewallSet(load_config(overrides={f"firewalls.{k}": v for k, v in overrides.items()}))


def _state(
    pos: int = 0, ts: pd.Timestamp | None = None, bar: int = 100, bars: pd.DataFrame = FLAT,
    day_pnl: float = 0.0, equity: float = 0.0, peak: float = 0.0, closed: list | None = None,
) -> FirewallState:  # fmt: skip
    ts = ts if ts is not None else bars.index[bar]
    position = Position(pos, 75.0, ts - pd.Timedelta(seconds=30)) if pos else Position()
    book = Book.from_bars(bars)
    return FirewallState(ts, bar, book, position, day_pnl, equity, peak, closed or [])


def _only_entry(walls: FirewallSet) -> dict:
    assert len(walls.log) == 1, walls.log
    return walls.log[0]


def test_order_is_documented_and_reducing_part() -> None:
    assert [w.name for w in _walls().walls] == list(FIREWALL_ORDER)
    assert reducing_part(2, -1) == -1 and reducing_part(2, -5) == -2
    assert reducing_part(2, 1) == 0 and reducing_part(0, 1) == 0


def test_max_drawdown_liquidates_and_halts_the_run() -> None:
    walls = _walls(max_drawdown_usd=5000)
    routed = walls.route(_state(pos=2, equity=-1000, peak=4500), 0, "")
    assert (routed.qty, routed.reason, routed.penalty) == (-2, "forced_liquidation", "forced")
    entry = _only_entry(walls)
    assert (entry["firewall"], entry["action"]) == ("max_drawdown", "liquidate")
    later = walls.route(_state(ts=OPEN + pd.Timedelta(days=3), equity=0, peak=4500), 1, "entry")
    assert later.qty == 0 and walls.log[-1]["reason"] == "run halted"


def test_daily_loss_limit_kills_the_day_only() -> None:
    walls = _walls(daily_loss_limit_usd=1500)
    routed = walls.route(_state(pos=-1, day_pnl=-1600), 0, "")
    assert routed.qty == 1 and routed.penalty == "forced"
    assert _only_entry(walls)["firewall"] == "daily_loss"
    assert walls.route(_state(bar=200, day_pnl=-100), 1, "entry").qty == 0
    tomorrow = _state(ts=OPEN + pd.Timedelta(days=1, seconds=100))
    assert walls.route(tomorrow, 1, "entry").qty == 1


def test_loss_streak_cooldown_blocks_new_risk_for_t_minutes() -> None:
    walls = _walls(loss_streak_n=3, loss_streak_cooldown_min=15)
    losses = [-10.0, 5.0, -1.0, -2.0, -3.0]
    assert walls.route(_state(closed=losses), 1, "entry").qty == 0
    entry = _only_entry(walls)
    assert (entry["firewall"], entry["action"], entry["qty_after"]) == ("loss_streak", "block", 0)
    assert walls.route(_state(pos=1, bar=150, closed=losses), -1, "signal_exit").qty == -1
    after = _state(ts=OPEN + pd.Timedelta(minutes=17), closed=losses)
    assert walls.route(after, 1, "entry").qty == 1


def test_news_blackout_blocks_entries_and_can_flatten() -> None:
    at_release = pd.Timestamp("2024-01-10 10:30", tz="America/New_York")
    walls = _walls()
    assert walls.route(_state(ts=at_release), -1, "entry").qty == 0
    assert _only_entry(walls)["firewall"] == "news_blackout"
    flattening = _walls(news_flatten=True)
    routed = flattening.route(_state(pos=2, ts=at_release - pd.Timedelta(minutes=1)), 0, "")
    assert (routed.qty, routed.reason) == (-2, "news_flatten")
    thursday = pd.Timestamp("2024-01-11 10:30", tz="America/New_York")
    assert _walls().route(_state(ts=thursday), -1, "entry").qty == -1


def test_vol_breaker_blocks_entries_in_a_vol_spike() -> None:
    calm = 75.00 + 0.01 * (np.arange(300) % 2)
    wild = 75.00 + 0.10 * (np.arange(100) % 2)
    bars = make_bars(np.concatenate([calm, wild]))
    walls = _walls(vol_breaker_mult=3.0)
    assert walls.route(_state(bars=bars, bar=390), 1, "entry").qty == 0
    entry = _only_entry(walls)
    assert entry["firewall"] == "vol_breaker" and "x median" in entry["reason"]
    assert _walls().route(_state(bars=bars, bar=250), 1, "entry").qty == 1


def test_stale_data_blocks_every_order() -> None:
    bars = FLAT.copy()
    bars["quote_age"] = 5.0
    walls = _walls(stale_quote_ms=2000)
    assert walls.route(_state(pos=1, bars=bars), -1, "signal_exit").qty == 0
    entry = _only_entry(walls)
    assert (entry["firewall"], entry["reason"]) == ("stale_data", "quote age 5.0s")


def test_spread_guard_allows_exits_but_not_entries() -> None:
    wide = make_bars(np.full(400, 75.00), spread_ticks=3)
    walls = _walls(max_spread_ticks=2)
    assert walls.route(_state(bars=wide), 1, "entry").qty == 0
    assert _only_entry(walls)["firewall"] == "spread_guard"
    assert walls.route(_state(pos=1, bars=wide), -1, "signal_exit").qty == -1
    reversal = walls.route(_state(pos=1, bars=wide), -2, "reversal")
    assert reversal.qty == -1 and walls.log[-1]["action"] == "reduce"


def test_price_band_blocks_fills_far_from_mid() -> None:
    walls = _walls(price_band_ticks=0.4)  # even half a tick of spread breaches it
    assert walls.route(_state(), 1, "entry").qty == 0
    entry = _only_entry(walls)
    assert entry["firewall"] == "price_band" and "0.5 ticks" in entry["reason"]


def test_max_order_size_clips() -> None:
    walls = _walls(max_order_size=3, max_position=10)
    routed = walls.route(_state(), -5, "entry")
    assert routed.qty == -3 and routed.firewalls == ["max_order_size"]
    assert _only_entry(walls)["qty_after"] == -3


def test_max_position_clips_to_the_limit() -> None:
    walls = _walls(max_position=5)
    assert walls.route(_state(pos=4), 3, "entry").qty == 1
    assert _only_entry(walls)["firewall"] == "max_position"
    assert walls.route(_state(pos=5), 1, "entry").qty == 0


def test_order_throttle_caps_orders_per_minute() -> None:
    walls = _walls(max_orders_per_min=2)
    for k in (100, 110):
        assert walls.route(_state(bar=k), 1, "entry").qty == 1
    assert walls.route(_state(bar=120), 1, "entry").qty == 0
    assert _only_entry(walls)["firewall"] == "order_throttle"
    assert walls.route(_state(bar=175), 1, "entry").qty == 1  # first order left the window


def test_engine_forced_liquidation_charges_penalty_and_logs() -> None:
    bid = np.full(200, 75.00)
    bid[40:] = 74.00  # a 100-tick collapse after the entry
    bars = make_bars(bid)
    secs = np.arange(10, 201, 10)
    preds = pd.DataFrame(
        {
            "ts": OPEN + pd.to_timedelta(secs, unit="s"),
            "day": OPEN.normalize().tz_localize(None),
            "fold": 0,
            "pred": np.where(secs == 10, 5.0, 2.0),
        }
    )
    cfg = load_config(
        overrides={
            "strategy.require_edge": False,
            "strategy.stop_loss_ticks": 10_000,
            "firewalls.daily_loss_limit_usd": 500,
        }
    )
    result = Engine(cfg).run(preds, lambda _day: bars)
    t = result.trades.iloc[0]
    assert t["exit_reason"] == "forced_liquidation" and "daily_loss" in t["firewall"]
    assert t["forced_penalty"] == pytest.approx(3 * 10.0)
    assert len(result.trades) == 1  # no re-entry for the rest of the day
    counts = result.firewall_log["firewall"].value_counts()
    assert counts["daily_loss"] >= 2  # the liquidation plus blocked re-entries
