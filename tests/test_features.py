"""No-lookahead guarantees for every feature, plus target construction."""

from collections.abc import Callable
from datetime import date

import numpy as np
import pandas as pd
import pytest

from clstrat.config import load_config
from clstrat.contract import SessionCalendar
from clstrat.data.bars import time_bars
from clstrat.data.synthetic import simulate_day
from clstrat.features import liquidity, orderflow, price
from clstrat.features.pipeline import (
    History,
    context_features,
    day_features,
    feature_columns,
    sample_decisions,
    targets,
)

DAY = date(2024, 1, 10)  # an EIA Wednesday
CUT = 7000  # seconds after the open; everything after this is perturbed

CFG = load_config()
F = CFG["features"]
W = F["windows"]
TOD = np.full(19800, 3.0)

FEATURE_FUNCTIONS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "trade_imbalance": lambda b: orderflow.trade_imbalance(b, W),
    "signed_volume": lambda b: orderflow.signed_volume(b, W, F["volume_norm_window"]),
    "order_flow_imbalance": lambda b: orderflow.order_flow_imbalance(b, W),
    "book_imbalance": lambda b: orderflow.book_imbalance(b, W),
    "microprice_deviation": lambda b: orderflow.microprice_deviation(b, W),
    "momentum": lambda b: price.momentum(b, W, F["ema_pairs"]),
    "reversion": lambda b: price.reversion(b, W),
    "skew": lambda b: price.skew(b, W, F["min_stat_window"]),
    "volatility": lambda b: price.volatility(b, W, F["vol_ratio_windows"]),
    "spread": lambda b: liquidity.spread(b, W),
    "trade_intensity": lambda b: liquidity.trade_intensity(b, W, TOD),
    "kyle_lambda": lambda b: liquidity.kyle_lambda(b, W, F["min_stat_window"]),
    "amihud": lambda b: liquidity.amihud(b, W),
    "context": lambda b: context_features(b, F, b.index[3600], 1e-4),
}


@pytest.fixture(scope="module")
def session():
    cal = SessionCalendar.from_config(CFG)
    d = simulate_day(CFG, cal, DAY, 1, 75.0, seed=5)
    open_ts = cal.session_bounds(DAY)[0]
    return d, open_ts, time_bars(d.quotes, d.trades, open_ts, cal.session_seconds())


def _perturb_after(bars: pd.DataFrame, cut: int, seed: int = 0) -> pd.DataFrame:
    """Scramble every numeric bar value strictly after row ``cut`` (keeping it plausible)."""
    rng = np.random.default_rng(seed)
    out = bars.astype(float)
    tail = out.iloc[cut + 1 :]
    for col in out.columns:
        noise = rng.uniform(0.5, 1.5, len(tail))
        if col.endswith("_px") or col.startswith("mid") or col == "vwap":
            noise = 1 + (noise - 1) * 0.02  # keep prices near the true level
        out.iloc[cut + 1 :, out.columns.get_loc(col)] = tail[col].to_numpy() * noise
    return out


def _assert_prefix_equal(a: pd.DataFrame, b: pd.DataFrame, cut: int, name: str) -> None:
    pd.testing.assert_frame_equal(a.iloc[: cut + 1], b.iloc[: cut + 1], obj=name, check_dtype=False)


@pytest.mark.parametrize("name", sorted(FEATURE_FUNCTIONS))
def test_feature_has_no_lookahead(session, name: str) -> None:
    _, _, bars = session
    fn = FEATURE_FUNCTIONS[name]
    clean = fn(bars)
    assert clean.iloc[CUT].notna().any(), f"{name} is all-NaN at the cut"
    _assert_prefix_equal(clean, fn(_perturb_after(bars, CUT)), CUT, name)


def test_full_feature_matrix_has_no_lookahead(session) -> None:
    _, _, bars = session
    history = History(F["history_days"])
    history.update(np.full(66, 3.0), 1e-4)
    full = day_features(bars, CFG, bars.index[3600], history)
    cols = feature_columns(full)
    assert len(cols) > 60
    perturbed = day_features(_perturb_after(bars, CUT), CFG, bars.index[3600], history)
    _assert_prefix_equal(full[cols], perturbed[cols], CUT, "day_features")


def test_bars_ignore_ticks_at_or_after_bar_end(session) -> None:
    d, open_ts, bars = session
    cut_ts = bars.index[CUT]  # right edge of bar CUT
    q, t = d.quotes.copy(), d.trades.copy()
    late_q, late_t = q["ts"] >= cut_ts, t["ts"] >= cut_ts
    q.loc[late_q, ["bid_px", "ask_px"]] += 1.0
    q.loc[late_q, "bid_sz"] *= 3
    t.loc[late_t, "side"] *= -1
    t.loc[late_t, "size"] += 7
    rebuilt = time_bars(q, t, open_ts, len(bars))
    _assert_prefix_equal(bars, rebuilt, CUT, "bars")


def test_targets_start_after_latency(session) -> None:
    _, _, bars = session
    y = targets(bars, [10, 60], latency_s=1)
    mid = bars["mid"].to_numpy()
    k = 5000
    assert y["y_60"].iloc[k] == pytest.approx((mid[k + 61] - mid[k + 1]) / 0.01)
    assert y["y_10"].iloc[-11:].isna().all()  # no target past the close


def test_decision_sampling_grid(session) -> None:
    _, _, bars = session
    sampled = sample_decisions(bars[["mid"]], DAY, 10)
    assert (sampled["sec"] % 10 == 0).all() and len(sampled) == len(bars) // 10
    assert (
        (sampled["ts"] - bars.index[0] + pd.Timedelta(seconds=1))
        .dt.total_seconds()
        .equals(sampled["sec"].astype(float))
    )
