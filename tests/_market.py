"""Hand-built 1 s books shared by the backtest and firewall tests."""

import numpy as np
import pandas as pd

OPEN = pd.Timestamp("2024-01-10 09:00", tz="America/New_York")  # an EIA Wednesday


def make_bars(
    bid: np.ndarray,
    size: int = 10,
    depth: list[tuple[int, int]] | None = None,
    spread_ticks: int = 1,
    open_ts: pd.Timestamp = OPEN,
) -> pd.DataFrame:
    """1 s bars from a bid path; ``depth`` = [(extra_ticks, size), ...] adds L2+ levels."""
    ask = bid + spread_ticks * 0.01
    bars = pd.DataFrame(
        {"bid_px": bid, "ask_px": ask, "bid_sz": size, "ask_sz": size},
        index=open_ts + pd.to_timedelta(np.arange(1, len(bid) + 1), unit="s"),
    )
    for k, (ticks, sz) in enumerate(depth or [], start=2):
        bars[f"bid_px_{k}"] = bid - ticks * 0.01
        bars[f"ask_px_{k}"] = ask + ticks * 0.01
        bars[f"bid_sz_{k}"] = sz
        bars[f"ask_sz_{k}"] = sz
    bars["mid"] = (bars["bid_px"] + bars["ask_px"]) / 2
    bars["quote_age"] = 0.0
    return bars
