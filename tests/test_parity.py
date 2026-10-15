"""The vectorized backtester matches the event-driven engine with firewalls off."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from clstrat.backtest.engine import Engine
from clstrat.backtest.vectorized import VectorBacktester
from clstrat.config import load_config
from clstrat.contract import SessionCalendar
from clstrat.data.bars import time_bars
from clstrat.data.synthetic import simulate_day
from clstrat.features.pipeline import targets

DAYS = [date(2024, 1, 9), date(2024, 1, 10)]


@pytest.fixture(scope="module")
def market():
    """Two simulated sessions and noisy 'predictions' built from the realised target."""
    cfg = load_config()
    cal = SessionCalendar.from_config(cfg)
    rng = np.random.default_rng(0)
    bars, frames = {}, []
    for k, day in enumerate(DAYS):
        d = simulate_day(cfg, cal, day, 0, 75.0, seed=100 + k)
        b = time_bars(d.quotes, d.trades, cal.session_bounds(day)[0], cal.session_seconds())
        bars[day] = b
        y = targets(b, [60], 1)["y_60"].fillna(0).to_numpy()
        rows = np.arange(9, len(b), 10)
        frames.append(pd.DataFrame({
            "ts": b.index[rows], "day": pd.Timestamp(day), "fold": 0,
            "pred": 0.3 * y[rows] + rng.normal(0, 1.5, len(rows)),
        }))  # fmt: skip
    return pd.concat(frames, ignore_index=True), bars


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"strategy.theta_exit_ticks": 0.0, "strategy.max_hold_seconds": 60},
        {"strategy.threshold_mode": "percentile", "strategy.require_edge": False},
        {"strategy.sizing": "vol_target", "strategy.vol_target_ticks": 1.0},
        {"strategy.sizing": "linear", "strategy.max_size": 4, "strategy.stop_loss_ticks": 3},
    ],
)
def test_vectorized_matches_event_engine(market, overrides) -> None:
    preds, bars = market
    cfg = load_config(overrides={"firewalls.enabled": False} | overrides)
    quantiles = {0: np.quantile(preds["pred"], np.linspace(0, 1, 201))}
    event = Engine(cfg).run(preds, bars.__getitem__, quantiles)
    fast = VectorBacktester(cfg, preds, bars.__getitem__, quantiles).run(cfg)
    assert len(event.trades) > 20
    assert fast["entries"].sum() == len(event.trades)
    for col in ("net", "gross", "fees", "spread_cost", "slippage", "eod_penalty"):
        np.testing.assert_allclose(fast[col], event.daily[col], atol=0.01, err_msg=col)
