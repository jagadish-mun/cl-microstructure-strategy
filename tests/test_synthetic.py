"""Synthetic generator: reproducibility, schema and calibration within tolerance."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from clstrat.config import load_config
from clstrat.contract import SessionCalendar
from clstrat.data.synthetic import generate_dataset, simulate_day
from clstrat.data.validate import calibration_stats

EIA_DAY = date(2024, 1, 10)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def cal(cfg):
    return SessionCalendar.from_config(cfg)


def test_same_seed_gives_identical_output(cfg, cal) -> None:
    a = simulate_day(cfg, cal, EIA_DAY, 0, 75.0, seed=11)
    b = simulate_day(cfg, cal, EIA_DAY, 0, 75.0, seed=11)
    c = simulate_day(cfg, cal, EIA_DAY, 0, 75.0, seed=12)
    pd.testing.assert_frame_equal(a.quotes, b.quotes)
    pd.testing.assert_frame_equal(a.trades, b.trades)
    assert not a.trades["price"].equals(c.trades["price"])


def test_schema_and_eia_event(cfg, cal) -> None:
    d = simulate_day(cfg, cal, EIA_DAY, 0, 75.0, seed=3)
    assert {"ts", "bid_px", "bid_sz", "ask_px", "ask_sz", "bid_px_5", "ask_sz_5"} <= set(d.quotes)
    assert list(d.trades.columns) == ["ts", "price", "size", "side"]
    assert str(d.quotes["ts"].dt.tz) == "America/New_York"
    assert set(d.trades["side"].unique()) == {-1, 1}
    assert (d.quotes["ask_px"] > d.quotes["bid_px"]).all()
    assert d.trades["ts"].is_monotonic_increasing
    assert d.events["type"].tolist() == ["EIA"]
    assert d.events["ts"].iloc[0] == cal.at(EIA_DAY, cal.eia_time)


def test_dataset_written_with_metadata(cfg, tmp_path) -> None:
    small = {**cfg, "data": {**cfg["data"], "n_days": 2}}
    days = generate_dataset(small, tmp_path)
    assert len(days) == 2
    assert (tmp_path / "generator_config.yaml").exists()
    assert len(list((tmp_path / "trades").glob("*.parquet"))) == 2


def test_calibration_within_tolerance(cfg, cal) -> None:
    """Looser bands than the full 60-day report because only 20 days are simulated here."""
    days = cal.trading_days(cfg["data"]["start_date"], 20)
    seeds = np.random.SeedSequence(cfg["seed"]).generate_state(len(days))

    def sessions():
        price = cfg["data"]["start_price"]
        for i, (day, seed) in enumerate(zip(days, seeds, strict=True)):
            d = simulate_day(cfg, cal, day, i % 2, price, int(seed))
            price = d.close_price
            yield day, d.quotes, d.trades

    s = calibration_stats(sessions(), cfg)
    vol = cfg["data"]["annual_vol"]
    assert 0.75 * vol < s["annual_vol"] < 1.25 * vol
    assert 1.0 <= s["mean_spread_ticks"] <= 1.3
    assert 3 <= s["trades_per_sec"] <= 10
    assert s["mid_ret_1s_acf1"] < 0
    assert s["sign_acf_1"] > s["sign_acf_100"] > 0.01  # positive and slowly decaying
    assert 0.0 < s["probe_oos_r2"] < 0.04
    assert 0.02 < s["probe_rank_ic"] < 0.15
