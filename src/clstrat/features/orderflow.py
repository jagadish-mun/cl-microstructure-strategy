"""Order-flow features on 1 s bars. Every value at bar ``t`` uses bars at or before ``t``.

Notation: ``B_s``/``S_s`` buy/sell aggressor volume in second ``s``, ``V_s`` total volume,
``bsz_l``/``asz_l`` bid/ask size at level ``l``, ``Σ_w`` a trailing sum over ``w`` seconds.
"""

from __future__ import annotations

import pandas as pd

from clstrat.config import Config
from clstrat.contract import CL


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """``num / den``; 0 where the denominator is 0 (no activity, no imbalance), NaN stays NaN."""
    return (num / den.where(den != 0)).mask(den == 0, 0.0)


def trade_imbalance(bars: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """``trade_imb_w = Σ_w (B - S) / Σ_w (B + S)``, in [-1, 1]."""
    out = {}
    for w in windows:
        signed = bars["signed_vol"].rolling(w, min_periods=w).sum()
        gross = (bars["buy_vol"] + bars["sell_vol"]).rolling(w, min_periods=w).sum()
        out[f"trade_imb_{w}"] = _ratio(signed, gross)
    return pd.DataFrame(out)


def signed_volume(bars: pd.DataFrame, windows: list[int], norm_window: int) -> pd.DataFrame:
    """Signed aggressor volume.

    - ``signed_vol_w = Σ_w (B - S)`` in lots
    - ``signed_vol_norm_w = signed_vol_w / (w * mean_{norm_window}(V))``: signed flow relative
      to the recent volume rate
    """
    rate = bars["volume"].rolling(norm_window, min_periods=1).mean()
    out = {}
    for w in windows:
        raw = bars["signed_vol"].rolling(w, min_periods=w).sum()
        out[f"signed_vol_{w}"] = raw
        out[f"signed_vol_norm_{w}"] = _ratio(raw, w * rate)
    return pd.DataFrame(out)


def order_flow_imbalance(bars: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """``ofi_w = Σ_w e_n / mean_w((bsz_1 + asz_1) / 2)``: Cont-Kukanov-Stoikov OFI summed over
    the window and normalised by average L1 depth (``e_n`` per quote update, see ``bars``)."""
    depth = (bars["bid_sz"] + bars["ask_sz"]) / 2
    out = {}
    for w in windows:
        total = bars["ofi"].rolling(w, min_periods=w).sum()
        out[f"ofi_{w}"] = total / depth.rolling(w, min_periods=w).mean()
    return pd.DataFrame(out)


def _depth_suffixes(bars: pd.DataFrame) -> list[str]:
    """Column suffixes of available book levels: ``""`` for L1, then ``"_2"``.. if present."""
    return [""] + [f"_{k}" for k in range(2, 6) if f"bid_sz_{k}" in bars]


def book_imbalance(bars: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Queue imbalance.

    - ``book_imb = (bsz_1 - asz_1) / (bsz_1 + asz_1)`` and its trailing mean ``book_imb_w``
    - ``book_imb_depth = Σ_l (bsz_l - asz_l) / l / Σ_l (bsz_l + asz_l) / l`` over L1..L5
      (weights 1/l), only when depth levels exist
    """
    imb = (bars["bid_sz"] - bars["ask_sz"]) / (bars["bid_sz"] + bars["ask_sz"])
    out = {"book_imb": imb}
    out |= {f"book_imb_{w}": imb.rolling(w, min_periods=w).mean() for w in windows}
    suffixes = _depth_suffixes(bars)
    if len(suffixes) > 1:
        bid = sum(bars[f"bid_sz{s}"] / level for level, s in enumerate(suffixes, start=1))
        ask = sum(bars[f"ask_sz{s}"] / level for level, s in enumerate(suffixes, start=1))
        out["book_imb_depth"] = (bid - ask) / (bid + ask)
    return pd.DataFrame(out)


def microprice_deviation(bars: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """``micro_dev = (microprice - mid) / tick`` with
    ``microprice = (ask * bsz_1 + bid * asz_1) / (bsz_1 + asz_1)``, plus trailing means."""
    depth = bars["bid_sz"] + bars["ask_sz"]
    micro = (bars["ask_px"] * bars["bid_sz"] + bars["bid_px"] * bars["ask_sz"]) / depth
    dev = (micro - bars["mid"]) / CL.tick_size
    out = {"micro_dev": dev}
    out |= {f"micro_dev_{w}": dev.rolling(w, min_periods=w).mean() for w in windows}
    return pd.DataFrame(out)


def orderflow_features(bars: pd.DataFrame, f: Config) -> pd.DataFrame:
    """All order-flow features for one session."""
    w = f["windows"]
    return pd.concat(
        [
            trade_imbalance(bars, w),
            signed_volume(bars, w, f["volume_norm_window"]),
            order_flow_imbalance(bars, w),
            book_imbalance(bars, w),
            microprice_deviation(bars, w),
        ],
        axis=1,
    )
