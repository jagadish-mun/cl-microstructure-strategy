"""Data-quality cleaning and 1-second bar construction."""

from datetime import date

import pandas as pd
import pytest

from clstrat.config import load_config
from clstrat.data.bars import time_bars
from clstrat.data.validate import clean_day

OPEN = pd.Timestamp("2024-01-10 09:00", tz="America/New_York")


def _ts(seconds: list[float]) -> pd.Series:
    return pd.Series(OPEN + pd.to_timedelta(seconds, unit="s"))


def _quotes(seconds, bid, ask, bid_sz=None, ask_sz=None) -> pd.DataFrame:
    n = len(seconds)
    return pd.DataFrame(
        {
            "ts": _ts(seconds),
            "bid_px": bid,
            "bid_sz": bid_sz or [5] * n,
            "ask_px": ask,
            "ask_sz": ask_sz or [5] * n,
        }
    )


def test_cleaning_flags_and_removes_bad_rows() -> None:
    quotes = _quotes(
        [0.0, 2.0, 1.0, 3.0, 3.0, 4.0, 5.0, 6.0],
        [75.00, 75.00, 75.00, 75.02, 75.01, 75.01, 75.01, 77.00],
        [75.01, 75.01, 75.00, 75.01, 75.02, 75.02, 75.02, 77.01],
        bid_sz=[5, 5, 5, 5, 5, 0, 5, 5],
    )
    trades = pd.DataFrame(
        {"ts": _ts([0.5, 0.5, 1.5]), "price": [75.01] * 3, "size": [1, 1, 0], "side": [1, 1, -1]}
    )
    q, t, s = clean_day(date(2024, 1, 10), quotes, trades, load_config())
    assert s.out_of_order == 1
    assert s.locked == 1  # the 1.0 s row
    assert s.duplicate_ts == 2  # quote at 3.0 s (first copy) + duplicated trade row
    assert s.bad_size == 2  # zero bid size + zero trade size
    assert s.price_spikes == 1  # the 77.00 quote is 199 ticks from the trailing median
    assert s.crossed == 1  # the first 3.0 s row is both duplicated and crossed
    assert q["ts"].is_monotonic_increasing and len(q) == 4 and len(t) == 1


def test_bars_only_use_data_before_bar_end() -> None:
    quotes = _quotes(
        [0.2, 0.9, 1.0, 2.5], [75.00, 75.01, 75.03, 75.02], [75.01, 75.02, 75.04, 75.03]
    )
    trades = pd.DataFrame(
        {
            "ts": _ts([0.5, 0.99, 1.0]),
            "price": [75.01, 75.01, 75.04],
            "size": [2, 3, 7],
            "side": [1, -1, 1],
        }
    )
    bars = time_bars(quotes, trades, OPEN, 3)
    first, second = bars.iloc[0], bars.iloc[1]
    assert bars.index[0] == OPEN + pd.Timedelta(seconds=1)
    assert first["bid_px"] == 75.01  # the 1.0 s quote belongs to the next bar
    assert first["signed_vol"] == -1 and first["n_trades"] == 2
    assert second["signed_vol"] == 7 and second["bid_px"] == 75.03
    assert first["quote_age"] == pytest.approx(0.1)
    assert bars.iloc[2]["mid"] == 75.025  # 2.5 s quote is visible at the 3 s bar end
