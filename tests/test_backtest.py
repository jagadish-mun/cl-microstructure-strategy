"""Hand-computed accounting scenarios for the event-driven engine."""

import numpy as np
import pandas as pd
import pytest

from clstrat.backtest.engine import Engine
from clstrat.backtest.execution import Book, ExecutionModel
from clstrat.backtest.metrics import max_drawdown, sharpe, summarize
from clstrat.config import load_config

OPEN = pd.Timestamp("2024-01-10 09:00", tz="America/New_York")
N = 120


def _cfg(**overrides) -> dict:
    base = {
        "strategy.theta_entry_ticks": 1.0,
        "strategy.theta_exit_ticks": 0.3,
        "strategy.require_edge": False,
        "strategy.stop_loss_ticks": 1000,
        "strategy.take_profit_ticks": 1000,
        "strategy.max_hold_seconds": 10_000,
        "execution.latency_ms": 1000,
        "execution.fee_per_contract_rt": 2.50,
    }
    return load_config(overrides=base | overrides)


def _bars(bid: np.ndarray, size: int = 10, depth: list[tuple[int, int]] | None = None):
    """1 s bars with a 1-tick spread; ``depth`` = [(extra_ticks, size), ...] for L2+."""
    bars = pd.DataFrame(
        {"bid_px": bid, "ask_px": bid + 0.01, "bid_sz": size, "ask_sz": size},
        index=OPEN + pd.to_timedelta(np.arange(1, len(bid) + 1), unit="s"),
    )
    for k, (ticks, sz) in enumerate(depth or [], start=2):
        bars[f"bid_px_{k}"] = bid - ticks * 0.01
        bars[f"ask_px_{k}"] = bid + 0.01 + ticks * 0.01
        bars[f"bid_sz_{k}"] = sz
        bars[f"ask_sz_{k}"] = sz
    bars["mid"] = (bars["bid_px"] + bars["ask_px"]) / 2
    return bars


def _preds(signals: dict[int, float]) -> pd.DataFrame:
    """Decisions every 10 s; ``signals`` maps second -> prediction (others 0)."""
    secs = np.arange(10, N + 1, 10)
    return pd.DataFrame(
        {
            "ts": OPEN + pd.to_timedelta(secs, unit="s"),
            "day": OPEN.normalize().tz_localize(None),
            "fold": 0,
            "pred": [signals.get(int(s), 0.0) for s in secs],
        }
    )


def _run(cfg, bars, preds):
    return Engine(cfg).run(preds, lambda _day: bars)


def _path(levels: dict[int, float]) -> np.ndarray:
    """Bid path: piecewise constant, ``levels`` maps first second -> bid price."""
    bid = np.empty(N)
    for start, price in sorted(levels.items()):
        bid[start - 1 :] = price
    return bid


def test_long_round_trip() -> None:
    # long decided at 10 s, filled at 11 s (ask 75.01); exit decided at 20 s, filled at 21 s
    # (bid 75.05)
    bars = _bars(_path({1: 75.00, 15: 75.05}))
    r = _run(_cfg(), bars, _preds({10: 5.0}))
    t = r.trades.iloc[0]
    assert (t["side"], t["entry_px"], t["exit_px"]) == ("long", 75.01, 75.05)
    assert t["exit_reason"] == "signal_exit"
    assert t["fees"] == pytest.approx(2.50)
    assert t["spread_cost"] == pytest.approx(10.0)  # half a tick on each side
    assert t["pnl_net"] == pytest.approx(40.0 - 2.50)
    assert t["pnl_gross"] == pytest.approx((75.055 - 75.005) * 1000)
    assert r.daily["net"].sum() == pytest.approx(37.50)
    assert r.equity.iloc[-1] == pytest.approx(37.50)


def test_short_round_trip() -> None:
    # short at 11 s bid 75.00; cover decided at 20 s, filled at 21 s ask 74.98 + 0.01
    bars = _bars(_path({1: 75.00, 15: 74.97}))
    r = _run(_cfg(), bars, _preds({10: -5.0}))
    t = r.trades.iloc[0]
    assert (t["side"], t["entry_px"], t["exit_px"]) == ("short", 75.00, 74.98)
    assert t["pnl_net"] == pytest.approx(20.0 - 2.50)
    assert r.daily.loc[:, "fees"].sum() == pytest.approx(2.50)


def test_partial_fill_walks_the_book() -> None:
    # 3 lots against 1 lot at L1, 1 lot one tick worse, 5 lots two ticks worse
    bars = _bars(_path({1: 75.00}), size=1, depth=[(1, 1), (2, 5)])
    fill = ExecutionModel(_cfg()).market(Book.from_bars(bars), 10, 3)
    assert fill.price == pytest.approx((75.01 + 75.02 + 75.03) / 3)
    assert fill.slippage == pytest.approx((0.01 + 0.02) * 1000)
    assert fill.spread_cost == pytest.approx(3 * 5.0)
    assert fill.fee == pytest.approx(3 * 1.25)
    r = _run(_cfg(**{"strategy.base_size": 3}), bars, _preds({10: 5.0}))
    t = r.trades.iloc[0]
    assert t["size"] == 3 and t["slippage"] == pytest.approx(30.0 + 30.0)  # in and out
    # entry avg 75.02, exit avg (75.00 + 74.99 + 74.98) / 3 = 74.99
    assert t["pnl_net"] == pytest.approx(3 * (74.99 - 75.02) * 1000 - 3 * 2.50)


def test_sqrt_impact_without_depth() -> None:
    bars = _bars(_path({1: 75.00}), size=1)
    fill = ExecutionModel(_cfg()).market(Book.from_bars(bars), 5, -3)
    extra = 0.5 * np.sqrt(2 / 1) * 0.01
    assert fill.price == pytest.approx((75.00 + 2 * (75.00 - extra)) / 3)
    assert fill.slippage == pytest.approx(2 * extra * 1000)


def test_end_of_day_flatten_charges_penalty() -> None:
    cfg = _cfg(**{"penalties.eod_flatten_time": "09:00:30", "penalties.eod_penalty_ticks": 1})
    bars = _bars(_path({1: 75.00, 25: 75.02}))
    r = _run(cfg, bars, _preds({10: 5.0, 20: 5.0, 30: 5.0, 40: 5.0}))
    assert len(r.trades) == 1
    t = r.trades.iloc[0]
    assert t["exit_reason"] == "eod" and t["exit_ts"] == OPEN + pd.Timedelta(seconds=31)
    assert t["eod_penalty"] == pytest.approx(10.0)
    assert t["pnl_net"] == pytest.approx((75.02 - 75.01) * 1000 - 2.50 - 10.0)
    assert r.daily["eod_penalty"].sum() == pytest.approx(10.0)


def test_reversal_splits_into_two_episodes() -> None:
    bars = _bars(_path({1: 75.00, 15: 75.03, 25: 75.00}))
    r = _run(_cfg(), bars, _preds({10: 5.0, 20: -5.0, 30: 0.0}))
    first, second = r.trades.iloc[0], r.trades.iloc[1]
    assert first["exit_reason"] == "reversal" and second["side"] == "short"
    assert first["pnl_net"] == pytest.approx((75.03 - 75.01) * 1000 - 2.50)
    assert second["pnl_net"] == pytest.approx((75.03 - 75.01) * 1000 - 2.50)
    assert r.daily["net"].sum() == pytest.approx(first["pnl_net"] + second["pnl_net"])


def test_risk_statistics() -> None:
    assert max_drawdown(pd.Series([10.0, 5.0, 12.0, 2.0, 8.0])) == 10.0
    assert np.isnan(sharpe(pd.Series([1.0, 1.0])))
    bars = _bars(_path({1: 75.00, 15: 75.05}))
    stats = summarize(_run(_cfg(), bars, _preds({10: 5.0})))
    assert stats["trades"] == 1 and stats["hit_rate"] == 1.0
    assert stats["net_pnl"] == pytest.approx(37.5)
    assert stats["gross_pnl"] - stats["net_pnl"] == pytest.approx(12.5)
