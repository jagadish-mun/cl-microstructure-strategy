"""Tick data to regular-hours 1-second bars.

Bar ``k`` (label = its right edge, ``open + k s``) summarises everything with timestamp in
``[open + (k-1) s, open + k s)``. Book columns are the state *as of* the bar end, so every
value labelled ``t`` is known at ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from clstrat.contract import CL

BOOK_PREFIXES = ("bid_px", "bid_sz", "ask_px", "ask_sz")


def book_columns(quotes: pd.DataFrame) -> list[str]:
    """Quote columns describing the book (L1 plus any ``_2.._5`` depth levels)."""
    return [c for c in quotes.columns if c.startswith(BOOK_PREFIXES)]


def _bin(ts: pd.Series, open_ts: pd.Timestamp, n_sec: int) -> np.ndarray:
    """Zero-based 1 s bin of each timestamp; -1 for rows outside the session."""
    b = ((ts - open_ts) // pd.Timedelta(seconds=1)).to_numpy()
    return np.where((b >= 0) & (b < n_sec), b, -1)


def time_bars(
    quotes: pd.DataFrame, trades: pd.DataFrame, open_ts: pd.Timestamp, n_sec: int
) -> pd.DataFrame:
    """Build 1 s bars for one session.

    Columns: L1/L2+ book as of bar end, ``mid``, ``mid_high``/``mid_low`` over the bar,
    ``buy_vol``, ``sell_vol``, ``signed_vol``, ``n_trades``, ``vwap`` (NaN without trades),
    ``quote_age`` (seconds since the last quote update at bar end).
    """
    idx = pd.RangeIndex(n_sec)
    q = quotes.assign(_b=_bin(quotes["ts"], open_ts, n_sec)).query("_b >= 0")
    q = q.assign(mid=(q["bid_px"] + q["ask_px"]) / 2)
    grouped = q.groupby("_b")
    book = grouped[book_columns(quotes) + ["mid"]].last().reindex(idx).ffill()
    last_quote = grouped["ts"].last().reindex(idx).ffill()
    prev_mid = book["mid"].shift(1)
    high = np.fmax(grouped["mid"].max().reindex(idx), prev_mid).fillna(book["mid"])
    low = np.fmin(grouped["mid"].min().reindex(idx), prev_mid).fillna(book["mid"])

    t = trades.assign(_b=_bin(trades["ts"], open_ts, n_sec)).query("_b >= 0")
    t = t.assign(
        buy=np.where(t["side"] > 0, t["size"], 0),
        sell=np.where(t["side"] < 0, t["size"], 0),
        notional=t["price"] * t["size"],
    )
    tg = t.groupby("_b")
    flow = pd.DataFrame(
        {
            "buy_vol": tg["buy"].sum(),
            "sell_vol": tg["sell"].sum(),
            "n_trades": tg["size"].size(),
            "notional": tg["notional"].sum(),
        }
    ).reindex(idx, fill_value=0)
    volume = flow["buy_vol"] + flow["sell_vol"]

    ends = open_ts + pd.to_timedelta(idx + 1, unit="s")
    bars = book.assign(
        mid_high=high,
        mid_low=low,
        buy_vol=flow["buy_vol"].astype(float),
        sell_vol=flow["sell_vol"].astype(float),
        signed_vol=(flow["buy_vol"] - flow["sell_vol"]).astype(float),
        n_trades=flow["n_trades"].astype(float),
        vwap=(flow["notional"] / volume.where(volume > 0)),
        quote_age=(ends - pd.DatetimeIndex(last_quote)).total_seconds(),
    )
    bars.index = pd.DatetimeIndex(ends, name="ts")
    return bars


def spread_ticks(bars: pd.DataFrame) -> pd.Series:
    """Quoted spread in ticks."""
    return ((bars["ask_px"] - bars["bid_px"]) / CL.tick_size).round()
