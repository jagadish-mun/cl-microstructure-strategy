"""Liquidity and activity features on 1 s bars (trailing windows only)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from clstrat.contract import CL
from clstrat.data.bars import spread_ticks


def spread(bars: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """``spread = (ask - bid) / tick`` and its trailing mean ``spread_w``."""
    s = spread_ticks(bars)
    out = {"spread": s} | {f"spread_{w}": s.rolling(w, min_periods=w).mean() for w in windows}
    return pd.DataFrame(out)


def trade_intensity(
    bars: pd.DataFrame, windows: list[int], tod_rate: np.ndarray | None
) -> pd.DataFrame:
    """``intensity_w = (Σ_w trades / w) / tod_rate(t)``: trade rate relative to the average
    rate at the same time of day on previous days (NaN when there is no history yet)."""
    norm = pd.Series(np.nan if tod_rate is None else tod_rate, index=bars.index)
    out = {}
    for w in windows:
        rate = bars["n_trades"].rolling(w, min_periods=w).sum() / w
        out[f"intensity_{w}"] = rate / norm.where(norm > 0)
    return pd.DataFrame(out)


def kyle_lambda(bars: pd.DataFrame, windows: list[int], min_window: int) -> pd.DataFrame:
    """``kyle_w = cov_w(Δm, SV) / var_w(SV)``: ticks of mid move per lot of signed volume,
    from a rolling regression of 1 s mid changes on 1 s signed volume (0 if no flow)."""
    dm = bars["mid"].diff() / CL.tick_size
    sv = bars["signed_vol"]
    out = {}
    for w in (w for w in windows if w >= min_window):
        var = sv.rolling(w, min_periods=w).var()
        out[f"kyle_{w}"] = (dm.rolling(w, min_periods=w).cov(sv) / var).mask(var == 0, 0.0)
    return pd.DataFrame(out)


def amihud(bars: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """``amihud_w = Σ_w |r| / Σ_w V`` (bps of absolute 1 s log return per lot traded)."""
    abs_ret = np.log(bars["mid"]).diff().abs() * 1e4
    out = {}
    for w in windows:
        vol = bars["volume"].rolling(w, min_periods=w).sum()
        out[f"amihud_{w}"] = (abs_ret.rolling(w, min_periods=w).sum() / vol).mask(vol == 0, 0.0)
    return pd.DataFrame(out)


def liquidity_features(bars: pd.DataFrame, f: dict, tod_rate: np.ndarray | None) -> pd.DataFrame:
    """All liquidity features for one session."""
    w = f["windows"]
    return pd.concat(
        [
            spread(bars, w),
            trade_intensity(bars, w, tod_rate),
            kyle_lambda(bars, w, f["min_stat_window"]),
            amihud(bars, w),
        ],
        axis=1,
    )
