"""Price-based features on 1 s bars: momentum, reversion, skew and volatility.

``m_t`` is the mid, ``r_t = ln(m_t / m_t-1)`` the 1 s log return, ``H_s``/``L_s`` the
high/low mid within second ``s``. All windows are trailing, so values at ``t`` are causal.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from clstrat.config import Config
from clstrat.contract import CL

BPS = 1e4


def _log_ret(bars: pd.DataFrame) -> pd.Series:
    return np.log(bars["mid"]).diff()


def momentum(bars: pd.DataFrame, windows: list[int], ema_pairs: list[list[int]]) -> pd.DataFrame:
    """``ret_w = ln(m_t / m_t-w)`` in bps, and ``ema_f_s = (EMA_f(m) - EMA_s(m)) / tick``."""
    logm = np.log(bars["mid"])
    out = {f"ret_{w}": (logm - logm.shift(w)) * BPS for w in windows}
    for fast, slow in ema_pairs:
        ema_f = bars["mid"].ewm(span=fast, adjust=False).mean()
        ema_s = bars["mid"].ewm(span=slow, adjust=False).mean()
        out[f"ema_{fast}_{slow}"] = (ema_f - ema_s) / CL.tick_size
    return pd.DataFrame(out)


def reversion(bars: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Distance of the mid from trailing anchors (0 when the anchor range is flat).

    - ``z_vwap_w = (m_t - VWAP_w) / std_w(m)`` with ``VWAP_w = Σ_w notional / Σ_w volume``
      (VWAP falls back to the mid when the window has no trades)
    - ``z_mean_w = (m_t - mean_w(m)) / std_w(m)``
    - ``range_pos_w = (m_t - min_w L) / (max_w H - min_w L) - 0.5``, in [-0.5, 0.5]
    """
    m = bars["mid"]
    out = {}
    for w in windows:
        sd = m.rolling(w, min_periods=w).std()
        vol = bars["volume"].rolling(w, min_periods=w).sum()
        vwap = (bars["notional"].rolling(w, min_periods=w).sum() / vol.where(vol > 0)).fillna(m)
        hi = bars["mid_high"].rolling(w, min_periods=w).max()
        lo = bars["mid_low"].rolling(w, min_periods=w).min()
        rng = hi - lo
        out[f"z_vwap_{w}"] = ((m - vwap) / sd).mask(sd == 0, 0.0).where(vol.notna())
        out[f"z_mean_{w}"] = ((m - m.rolling(w, min_periods=w).mean()) / sd).mask(sd == 0, 0.0)
        out[f"range_pos_{w}"] = ((m - lo) / rng - 0.5).mask(rng == 0, 0.0)
    return pd.DataFrame(out)


def skew(bars: pd.DataFrame, windows: list[int], min_window: int) -> pd.DataFrame:
    """Two different "skews", named apart.

    - ``ret_skew_w``: rolling sample skewness of 1 s log returns (return-distribution skew),
      0 when the window's returns are all equal
    - ``book_skew = Σ_{l>=2} (bsz_l - asz_l) / Σ_{l>=2} (bsz_l + asz_l)``: asymmetry of the
      deeper book (levels 2-5), only when depth levels exist
    """
    r = _log_ret(bars)
    out = {}
    for w in (w for w in windows if w >= min_window):
        complete = r.rolling(w, min_periods=1).count() == w
        out[f"ret_skew_{w}"] = r.rolling(w, min_periods=w).skew().fillna(0.0).where(complete)
    deep = [k for k in range(2, 6) if f"bid_sz_{k}" in bars]
    if deep:
        num = sum(bars[f"bid_sz_{k}"] - bars[f"ask_sz_{k}"] for k in deep)
        den = sum(bars[f"bid_sz_{k}"] + bars[f"ask_sz_{k}"] for k in deep)
        out["book_skew"] = num / den
    return pd.DataFrame(out)


def volatility(bars: pd.DataFrame, windows: list[int], ratio: list[int]) -> pd.DataFrame:
    """Per-second volatility estimates in bps.

    - ``rv_w = sqrt(mean_w(r^2))``
    - ``park_w = sqrt(mean_w(ln(H/L)^2) / (4 ln 2))`` (Parkinson high-low)
    - ``vol_ratio_a_b = rv_a / rv_b`` (vol-of-vol proxy: short vs long realized vol)
    """
    r2 = _log_ret(bars) ** 2
    hl2 = np.log(bars["mid_high"] / bars["mid_low"]) ** 2
    out = {}
    for w in windows:
        out[f"rv_{w}"] = np.sqrt(r2.rolling(w, min_periods=w).mean()) * BPS
        out[f"park_{w}"] = np.sqrt(hl2.rolling(w, min_periods=w).mean() / (4 * math.log(2))) * BPS
    short, long_ = ratio
    rv_s = np.sqrt(r2.rolling(short, min_periods=short).mean())
    rv_l = np.sqrt(r2.rolling(long_, min_periods=long_).mean())
    out[f"vol_ratio_{short}_{long_}"] = rv_s / rv_l.replace(0, np.nan)
    return pd.DataFrame(out)


def price_features(bars: pd.DataFrame, f: Config) -> pd.DataFrame:
    """All price-based features for one session."""
    w = f["windows"]
    return pd.concat(
        [
            momentum(bars, w, f["ema_pairs"]),
            reversion(bars, w),
            skew(bars, w, f["min_stat_window"]),
            volatility(bars, w, f["vol_ratio_windows"]),
        ],
        axis=1,
    )
