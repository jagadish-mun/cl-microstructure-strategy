"""Real-data adapters: CSV fixture, Lee-Ready classification, volume roll, vendor mapping."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from clstrat.config import load_config
from clstrat.data.adapters import (
    CsvAdapter,
    ParquetStore,
    VendorFileAdapter,
    front_month,
    lee_ready,
    make_source,
    tick_rule,
)

FIXTURE = Path(__file__).parent / "fixtures" / "csv"
TZ = "America/New_York"
D1, D2, D3, D4 = date(2024, 3, 13), date(2024, 3, 14), date(2024, 3, 15), date(2024, 3, 18)


def _ts(seconds: list[float]) -> pd.Series:
    return pd.Series(pd.Timestamp("2024-03-13 09:00", tz=TZ) + pd.to_timedelta(seconds, unit="s"))


def test_tick_rule_handles_zero_ticks() -> None:
    price = pd.Series([10.0, 10.0, 10.01, 10.01, 10.0, 10.0])
    assert tick_rule(price).tolist() == [0, 0, 1, 1, -1, -1]


def test_lee_ready_quote_rule_then_tick_rule() -> None:
    quotes = pd.DataFrame(
        {"ts": _ts([0.0, 2.0]), "bid_px": [9.99, 10.00], "ask_px": [10.01, 10.02]}
    )
    trades = pd.DataFrame(
        {
            "ts": _ts([-1.0, 0.5, 1.0, 1.5, 2.0, 3.0]),
            "price": [10.00, 10.01, 9.99, 10.00, 10.00, 10.01],
        }
    )
    # -1 s: no quote and no prior trade -> 0; 0.5 s above mid -> +1; 1 s below -> -1;
    # 1.5 s at mid after an uptick -> +1; 2 s uses the 0 s quote (strictly-before rule), at mid
    # after a zero-tick that followed an uptick -> +1; 3 s at the new mid after an uptick -> +1
    assert lee_ready(trades, quotes).tolist() == [0, 1, -1, 1, 1, 1]


def test_front_month_rolls_on_previous_day_volume_and_never_back() -> None:
    vol = pd.DataFrame(
        {"2024-04": [9, 2, 50, 30], "2024-05": [1, 20, 10, 5]}, index=[D1, D2, D3, D4]
    )
    assert front_month(vol).tolist() == ["2024-04", "2024-04", "2024-05", "2024-05"]


def test_csv_adapter_on_fixture() -> None:
    src = CsvAdapter(FIXTURE, TZ)
    assert src.days() == [D1, D2, D3, D4]
    q = src.load_quotes(D1)
    t = src.load_trades(D1)
    assert str(q["ts"].dt.tz) == TZ and "expiry" not in q
    assert len(q) == 2 and (q["bid_px"] == 79.50).all()  # April contract only
    assert t.columns.tolist() == ["ts", "price", "size", "side"]
    assert t["side"].tolist() == [1, -1, 1]  # ask, bid, at-mid after an uptick
    assert src.load_trades(D2)["size"].tolist() == [2]  # still April
    assert src.load_trades(D3)["size"].tolist() == [10]  # rolled to May
    assert src.load_trades(D4)["price"].tolist() == [79.80]  # never rolls back
    assert src.load_trades(D4)["side"].tolist() == [-1]


def test_vendor_adapter_maps_columns_and_sides(tmp_path) -> None:
    for table in ("quotes", "trades"):
        (tmp_path / table).mkdir()
    t0 = pd.Timestamp("2024-03-13 13:00", tz="UTC").value  # 09:00 EDT
    pd.DataFrame(
        {
            "ts_event": [t0],
            "bid_px_00": [80.0],
            "ask_px_00": [80.02],
            "bid_sz_00": [4],
            "ask_sz_00": [6],
            "symbol": ["CLK4"],
        }
    ).to_parquet(tmp_path / "quotes" / "2024-03-13.parquet")
    pd.DataFrame(
        {
            "ts_event": t0 + np.array([1, 2, 3]) * 10**9,
            "price": [80.02, 80.0, 80.02],
            "size": [1, 2, 3],
            "side": ["A", "B", "N"],
            "symbol": ["CLK4"] * 3,
        }
    ).to_parquet(tmp_path / "trades" / "2024-03-13.parquet")
    src = VendorFileAdapter(tmp_path, TZ)
    t = src.load_trades(D1)
    assert VendorFileAdapter.symbol_to_expiry("CLK4") == "2024-05"
    assert t["ts"].iloc[0] == pd.Timestamp("2024-03-13 09:00:01", tz=TZ)
    assert t["side"].tolist() == [1, -1, 1]  # "N" re-classified by Lee-Ready (above mid)
    assert src.load_quotes(D1)[["bid_px", "ask_sz"]].iloc[0].tolist() == [80.0, 6]


def test_make_source_dispatch() -> None:
    cfg = load_config()
    assert isinstance(make_source(cfg), ParquetStore)
    cfg["data"]["source"] = "csv"
    cfg["data"]["csv"]["root"] = str(FIXTURE)
    assert make_source(cfg).days()[0] == D1
