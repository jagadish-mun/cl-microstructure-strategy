"""Feature matrix and targets: clean ticks -> 1 s bars -> features -> decision-time samples.

Each day is processed on its own, so rolling windows, forward fills and targets never cross a
session (or contract-roll) boundary. Two context features use earlier days only: the
time-of-day trade-rate norm and the volatility-regime indicator.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from clstrat.config import Config
from clstrat.contract import CL, SessionCalendar
from clstrat.data.adapters import DataSource
from clstrat.data.bars import time_bars
from clstrat.data.validate import QualitySummary, clean_day
from clstrat.features.liquidity import liquidity_features
from clstrat.features.orderflow import orderflow_features
from clstrat.features.price import price_features

META_COLUMNS = ["ts", "day", "sec", "eia_day"]


@dataclass
class History:
    """Per-day summaries of previous sessions, used for causal cross-day norms."""

    max_days: int
    trade_rates: deque = field(default_factory=deque)  # trades/sec per time-of-day bucket
    day_vols: deque = field(default_factory=deque)  # per-second realized vol of the day

    def update(self, rate_profile: np.ndarray, day_vol: float) -> None:
        """Append one finished day, dropping the oldest beyond ``max_days``."""
        self.trade_rates.append(rate_profile)
        self.day_vols.append(day_vol)
        if len(self.trade_rates) > self.max_days:
            self.trade_rates.popleft()
            self.day_vols.popleft()

    def tod_rate(self) -> np.ndarray | None:
        """Average trade rate per bucket over the stored days, or None without history."""
        return np.mean(self.trade_rates, axis=0) if self.trade_rates else None

    def mean_vol(self) -> float:
        """Average per-second realized vol over the stored days (NaN without history)."""
        return float(np.mean(self.day_vols)) if self.day_vols else math.nan


def _bucket_of_second(n_sec: int, bucket_min: int) -> np.ndarray:
    return np.arange(n_sec) // (60 * bucket_min)


def day_summary(bars: pd.DataFrame, bucket_min: int) -> tuple[np.ndarray, float]:
    """Trade rate per time-of-day bucket and per-second realized vol for a finished day."""
    buckets = _bucket_of_second(len(bars), bucket_min)
    rates = pd.Series(bars["n_trades"].to_numpy()).groupby(buckets).mean().to_numpy()
    r = np.log(bars["mid"]).diff().dropna()
    return rates, float(np.sqrt(np.mean(r**2)))


def context_features(
    bars: pd.DataFrame, f: Config, eia_ts: pd.Timestamp | None, regime_norm: float
) -> pd.DataFrame:
    """Calendar and context.

    - ``tod_sin``/``tod_cos``: time of day as a phase over the session
    - ``min_to_eia``: minutes until the EIA release (negative after), clipped to
      ``±eia_clip_min``; ``+eia_clip_min`` on non-EIA days
    - ``eia_day``: 1 on EIA release days
    - ``vol_regime = rv_{regime_window} / mean daily rv of previous days`` (causal regime)
    """
    n = len(bars)
    phase = 2 * np.pi * np.arange(1, n + 1) / n
    clip = f["eia_clip_min"]
    if eia_ts is None:
        to_eia = np.full(n, float(clip))
    else:
        to_eia = np.clip((eia_ts - bars.index).total_seconds() / 60, -clip, clip)
    r2 = np.log(bars["mid"]).diff() ** 2
    w = f["regime_window"]
    rv = np.sqrt(r2.rolling(w, min_periods=min(w, 60)).mean())
    return pd.DataFrame(
        {
            "tod_sin": np.sin(phase),
            "tod_cos": np.cos(phase),
            "min_to_eia": to_eia,
            "eia_day": float(eia_ts is not None),
            "vol_regime": rv / regime_norm,
        },
        index=bars.index,
    )


def targets(bars: pd.DataFrame, horizons: list[int], latency_s: int) -> pd.DataFrame:
    """``y_h = (m_{t+latency+h} - m_{t+latency}) / tick``: forward mid change in ticks,
    starting when an order sent at ``t`` could first reach the market. NaN past the close."""
    m = bars["mid"]
    start = m.shift(-latency_s)
    return pd.DataFrame(
        {f"y_{h}": (m.shift(-(latency_s + h)) - start) / CL.tick_size for h in horizons}
    )


def eia_release(cfg: Config, cal: SessionCalendar, day: date) -> pd.Timestamp | None:
    """Scheduled EIA release time on ``day``, or None (no release or events disabled)."""
    if cfg["data"]["eia_events"] and cal.is_eia_day(day):
        return cal.at(day, cal.eia_time)
    return None


def latency_seconds(cfg: Config) -> int:
    """Execution latency rounded up to whole 1 s bars."""
    return math.ceil(cfg["execution"]["latency_ms"] / 1000)


def day_features(
    bars: pd.DataFrame, cfg: Config, eia_ts: pd.Timestamp | None, history: History
) -> pd.DataFrame:
    """All features and targets for one day on the 1 s grid."""
    f = cfg["features"]
    tod = history.tod_rate()
    tod_per_sec = None
    if tod is not None:
        buckets = np.minimum(_bucket_of_second(len(bars), f["intensity_bucket_min"]), len(tod) - 1)
        tod_per_sec = tod[buckets]
    return pd.concat(
        [
            orderflow_features(bars, f),
            price_features(bars, f),
            liquidity_features(bars, f, tod_per_sec),
            context_features(bars, f, eia_ts, history.mean_vol()),
            targets(bars, cfg["target"]["horizons_report"], latency_seconds(cfg)),
        ],
        axis=1,
    )


def sample_decisions(frame: pd.DataFrame, day: date, step_s: int) -> pd.DataFrame:
    """Keep rows on the decision grid (every ``step_s`` seconds after the open)."""
    sec = np.arange(1, len(frame) + 1)
    out = frame[sec % step_s == 0].copy()
    out.insert(0, "sec", sec[sec % step_s == 0])
    out.insert(0, "day", pd.Timestamp(day))
    out.insert(0, "ts", out.index)
    return out.reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model inputs: everything that is not metadata or a target."""
    return [c for c in df.columns if c not in META_COLUMNS and not c.startswith("y_")]


def build_dataset(
    cfg: Config, source: DataSource, bars_dir: Path | None = None
) -> tuple[pd.DataFrame, list[QualitySummary]]:
    """Run the feature pipeline over every day of ``source``.

    Writes each day's 1 s bars to ``bars_dir`` (if given) for the backtester and returns the
    decision-time dataset plus per-day data-quality summaries.
    """
    cal = SessionCalendar.from_config(cfg)
    f = cfg["features"]
    step = int(pd.Timedelta(cfg["sampling"]["decision_interval"]).total_seconds())
    history = History(f["history_days"])
    frames, summaries = [], []
    if bars_dir is not None:
        bars_dir.mkdir(parents=True, exist_ok=True)
    for day in source.days():
        raw_quotes, raw_trades = source.load_quotes(day), source.load_trades(day)
        quotes, trades, summary = clean_day(day, raw_quotes, raw_trades, cfg)
        summaries.append(summary)
        bars = time_bars(quotes, trades, cal.session_bounds(day)[0], cal.session_seconds())
        if bars_dir is not None:
            bars.to_parquet(bars_dir / f"{day.isoformat()}.parquet")
        feats = day_features(bars, cfg, eia_release(cfg, cal, day), history)
        frames.append(sample_decisions(feats, day, step))
        history.update(*day_summary(bars, f["intensity_bucket_min"]))
    return pd.concat(frames, ignore_index=True), summaries


def research_days(df: pd.DataFrame, holdout_days: int) -> pd.DataFrame:
    """Rows from every day except the final holdout (which stays untouched until the end)."""
    days = np.sort(df["day"].unique())
    return df[df["day"].isin(days[: len(days) - holdout_days])]


def ic_decay(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Spearman correlation of each feature with the forward return at each horizon."""
    rows = {}
    for col in feature_columns(df):
        rows[col] = {}
        for h in horizons:
            pair = df[[col, f"y_{h}"]].dropna()
            rows[col][h] = spearmanr(pair[col], pair[f"y_{h}"])[0] if len(pair) > 2 else np.nan
    return pd.DataFrame(rows).T.rename_axis("feature")
