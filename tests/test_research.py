"""Sweep mechanics, strategy choice, break-even cost, placebo and deflated Sharpe."""

import numpy as np
import pandas as pd
import pytest

from clstrat.config import load_config
from clstrat.research.robustness import (
    deflated_sharpe,
    expected_max_sharpe,
    placebo_p_value,
    probabilistic_sharpe,
    regime_table,
)
from clstrat.research.sweeps import break_even_fee, choose_strategy, threshold_sweep

DAYS = pd.date_range("2024-01-02", periods=20, name="day")


class StubBacktester:
    """Daily P&L that rewards high entry thresholds, to exercise the sweep plumbing."""

    def __init__(self) -> None:
        self.calls: list[tuple[float, float]] = []

    def run(self, cfg: dict) -> pd.DataFrame:
        s = cfg["strategy"]
        self.calls.append((s["theta_entry_ticks"], s["theta_exit_ticks"]))
        rng = np.random.default_rng(int(10 * s["theta_entry_ticks"]))
        net = rng.normal(10 * s["theta_entry_ticks"], 50, len(DAYS))
        return pd.DataFrame({"net": net, "gross": net + 30, "fees": 10.0, "entries": 5}, index=DAYS)


def test_threshold_sweep_skips_exit_above_entry() -> None:
    stub = StubBacktester()
    grid = {"theta_entry_ticks": [1.0, 2.0], "theta_exit_ticks": [0.5, 1.5]}
    table = threshold_sweep(stub, load_config(), grid)
    assert sorted(stub.calls) == [(1.0, 0.5), (2.0, 0.5), (2.0, 1.5)]
    assert set(table.columns) >= {"net_pnl", "net_sharpe", "trades", "costs"}
    assert (table["trades"] == 100).all()
    np.testing.assert_allclose(table["costs"], 600.0)


def test_choose_strategy_respects_minimum_trades() -> None:
    ticks = pd.DataFrame(
        {"strategy.theta_entry_ticks": [1.0, 3.0], "net_sharpe": [1.0, 9.0], "trades": [500, 5]}
    )
    pct = pd.DataFrame({"strategy.pct_entry": [0.95], "net_sharpe": [2.0], "trades": [300]})
    choice = choose_strategy({"ticks": ticks, "percentile": pct}, min_trades=100)
    expected = {"strategy.threshold_mode": "percentile", "strategy.pct_entry": 0.95}
    assert choice["overrides"] == expected
    with pytest.raises(ValueError):
        choose_strategy({"ticks": ticks}, min_trades=10_000)


def test_break_even_fee() -> None:
    # 10 round trips at $2.50 = $25 of fees; net -$5 -> break-even at $2.00 per round trip
    daily = pd.DataFrame({"fees": [12.5, 12.5], "net": [-2.0, -3.0]})
    assert break_even_fee(daily, 2.5) == pytest.approx(2.0)


def test_placebo_p_value() -> None:
    shuffles = pd.Series(np.arange(99, dtype=float))
    assert placebo_p_value(1000.0, shuffles) == pytest.approx(1 / 100)
    assert placebo_p_value(-1.0, shuffles) == pytest.approx(1.0)


def test_deflated_sharpe_penalises_more_trials() -> None:
    rng = np.random.default_rng(0)
    daily = pd.Series(rng.normal(50, 100, 60))
    few, many = rng.normal(0, 1, 5), rng.normal(0, 1, 500)
    assert expected_max_sharpe(many / 16) > expected_max_sharpe(few / 16)
    d_few, d_many = deflated_sharpe(daily, few), deflated_sharpe(daily, many)
    assert d_many["deflated_sharpe"] < d_few["deflated_sharpe"] <= d_few["psr_vs_zero"]
    assert probabilistic_sharpe(daily, 0.0) == pytest.approx(d_few["psr_vs_zero"])
    assert 0.9 < probabilistic_sharpe(daily, 0.0) <= 1.0


def test_regime_table_groups_every_label() -> None:
    daily = pd.DataFrame({"net": [10.0, -5.0, 20.0, 0.0], "gross": 30.0}, index=DAYS[:4])
    labels = pd.DataFrame(
        {"realized_vol": ["low", "high", "low", "high"], "eia": ["EIA", "no", "no", "no"]},
        index=DAYS[:4],
    )
    table = regime_table(daily, labels)
    assert table.loc[("realized_vol", "low"), "net_pnl"] == 30.0
    assert table.loc[("eia", "no"), "days"] == 3
    assert table.loc[("realized_vol", "high"), "positive_days"] == 0.0
