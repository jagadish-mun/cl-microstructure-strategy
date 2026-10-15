"""Data-quality checks and the synthetic-market calibration report."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from clstrat.config import Config
from clstrat.contract import CL, SessionCalendar
from clstrat.data.bars import spread_ticks, time_bars
from clstrat.features.orderflow import book_imbalance, microprice_deviation, trade_imbalance
from clstrat.features.price import momentum, skew

# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------


@dataclass
class QualitySummary:
    """Per-day counts of problems found (and removed, except gaps, which are only reported)."""

    day: str
    quotes: int
    trades: int
    out_of_order: int = 0
    duplicate_ts: int = 0
    crossed: int = 0
    locked: int = 0
    bad_size: int = 0
    price_spikes: int = 0
    gaps: int = 0
    max_gap_s: float = 0.0


def _sort(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Stable sort by ``ts``; returns the number of rows that were out of order."""
    out_of_order = int((df["ts"].diff() < pd.Timedelta(0)).sum())
    return df.sort_values("ts", kind="stable").reset_index(drop=True), out_of_order


def _spikes(price: pd.Series, window: int, max_ticks: float) -> pd.Series:
    """Rows more than ``max_ticks`` from the trailing (causal) rolling median."""
    med = price.rolling(window, min_periods=1).median().shift(1).fillna(price)
    return (price - med).abs() / CL.tick_size > max_ticks


def clean_quotes(quotes: pd.DataFrame, cfg: Config, summary: QualitySummary) -> pd.DataFrame:
    """Sort, de-duplicate (keep last), drop crossed/locked books, bad sizes and price spikes."""
    v = cfg["validation"]
    q, summary.out_of_order = _sort(quotes)
    dup = q["ts"].duplicated(keep="last")
    crossed = q["bid_px"] > q["ask_px"]
    locked = q["bid_px"] == q["ask_px"]
    bad_size = (q["bid_sz"] <= 0) | (q["ask_sz"] <= 0)
    spikes = _spikes((q["bid_px"] + q["ask_px"]) / 2, v["spike_window"], v["spike_ticks"])
    summary.duplicate_ts = int(dup.sum())
    summary.crossed = int(crossed.sum())
    summary.locked = int(locked.sum())
    summary.bad_size = int(bad_size.sum())
    summary.price_spikes = int(spikes.sum())
    q = q[~(dup | crossed | locked | bad_size | spikes)].reset_index(drop=True)
    gaps = q["ts"].diff().dt.total_seconds()
    summary.gaps = int((gaps > v["max_gap_s"]).sum())
    summary.max_gap_s = float(gaps.max()) if len(gaps) > 1 else 0.0
    return q


def clean_trades(trades: pd.DataFrame, cfg: Config, summary: QualitySummary) -> pd.DataFrame:
    """Sort, drop exact duplicate rows, non-positive sizes and price spikes."""
    v = cfg["validation"]
    t, ooo = _sort(trades)
    dup = t.duplicated(keep="first")
    bad = t["size"] <= 0
    spikes = _spikes(t["price"], v["spike_window"], v["spike_ticks"])
    summary.out_of_order += ooo
    summary.duplicate_ts += int(dup.sum())
    summary.bad_size += int(bad.sum())
    summary.price_spikes += int(spikes.sum())
    return t[~(dup | bad | spikes)].reset_index(drop=True)


def clean_day(
    day: date, quotes: pd.DataFrame, trades: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame, QualitySummary]:
    """Clean one day's quotes and trades and return the quality summary."""
    summary = QualitySummary(day.isoformat(), len(quotes), len(trades))
    return clean_quotes(quotes, cfg, summary), clean_trades(trades, cfg, summary), summary


def quality_table(summaries: Iterable[QualitySummary]) -> pd.DataFrame:
    """Summaries as a DataFrame, one row per day."""
    return pd.DataFrame([asdict(s) for s in summaries]).set_index("day")


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _acf(x: np.ndarray, lag: int) -> float:
    """Sample autocorrelation at ``lag``."""
    x = x - x.mean()
    return float(np.dot(x[:-lag], x[lag:]) / np.dot(x, x)) if len(x) > lag else np.nan


def _probe_design(bars: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """A compact, causal linear probe used to check how predictable the generator is.

    It reuses the production feature functions: trade imbalance, signed volume and past
    returns over each window, L1 and depth-weighted book imbalance, deep-book skew and
    microprice deviation. Target: forward mid change in ticks over the primary horizon,
    starting ``latency`` after the decision time.
    """
    windows = cfg["features"]["windows"]
    step = int(pd.Timedelta(cfg["sampling"]["decision_interval"]).total_seconds())
    lat = int(np.ceil(cfg["execution"]["latency_ms"] / 1000))
    h = cfg["target"]["horizon"]
    X = pd.concat(
        [
            trade_imbalance(bars, windows),
            momentum(bars, windows, []),
            book_imbalance(bars, []),
            skew(bars, [], 0),
            microprice_deviation(bars, []),
        ],
        axis=1,
    )
    mid = bars["mid"]
    y = (mid.shift(-(lat + h)) - mid.shift(-lat)) / CL.tick_size
    rows = np.arange(len(bars)) % step == step - 1
    keep = rows & X.notna().all(axis=1).to_numpy() & y.notna().to_numpy()
    return X.to_numpy()[keep], y.to_numpy()[keep]


def _probe_fit(designs: list[tuple[np.ndarray, np.ndarray]], train_frac: float) -> dict[str, float]:
    """OLS on the first ``train_frac`` of days, scored on the rest (R^2 vs the zero forecast)."""
    n_train = max(1, int(len(designs) * train_frac))
    Xtr = np.vstack([d[0] for d in designs[:n_train]])
    ytr = np.concatenate([d[1] for d in designs[:n_train]])
    Xte = np.vstack([d[0] for d in designs[n_train:]])
    yte = np.concatenate([d[1] for d in designs[n_train:]])
    mu, sd = Xtr.mean(0), Xtr.std(0)
    Atr = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd])
    beta, *_ = np.linalg.lstsq(Atr, ytr, rcond=None)
    pred = np.column_stack([np.ones(len(Xte)), (Xte - mu) / sd]) @ beta
    return {
        "probe_oos_r2": float(1 - np.sum((yte - pred) ** 2) / np.sum(yte**2)),
        "probe_rank_ic": float(spearmanr(pred, yte)[0]),
        "probe_n_test": int(len(yte)),
    }


def calibration_stats(
    sessions: Iterable[tuple[date, pd.DataFrame, pd.DataFrame]], cfg: Config
) -> dict[str, float]:
    """Calibration statistics (§2.6) over a sequence of ``(day, quotes, trades)``.

    The predictability probe is fitted and scored on research days only; the final
    ``cv.holdout_days`` days are excluded.
    """
    cal = SessionCalendar.from_config(cfg)
    c = cfg["calibration"]
    n_sec = cal.session_seconds()
    rv_sum, spreads, rate, ret1_acf = [], [], [], []
    sign_acf: dict[int, list[float]] = {lag: [] for lag in c["acf_sign_lags"]}
    designs = []
    for day, quotes, trades in sessions:
        bars = time_bars(quotes, trades, cal.session_bounds(day)[0], n_sec)
        logmid = np.log(bars["mid"].to_numpy())
        r = np.diff(logmid[:: c["rv_bar_s"]])
        rv_sum.append(np.sum(r**2))
        spreads.append(spread_ticks(bars).mean())
        rate.append(len(trades) / n_sec)
        ret1_acf.append(_acf(np.diff(bars["mid"].to_numpy()), 1))
        side = trades["side"].to_numpy(float)
        for lag in sign_acf:
            sign_acf[lag].append(_acf(side, lag))
        designs.append(_probe_design(bars, cfg))
    stats = {
        "annual_vol": float(np.sqrt(np.mean(rv_sum) * 252)),
        "mean_spread_ticks": float(np.mean(spreads)),
        "trades_per_sec": float(np.mean(rate)),
        "mid_ret_1s_acf1": float(np.nanmean(ret1_acf)),
    }
    stats |= {f"sign_acf_{lag}": float(np.nanmean(v)) for lag, v in sign_acf.items()}
    research = designs[: max(2, len(designs) - cfg["cv"]["holdout_days"])]  # never the holdout
    return stats | _probe_fit(research, c["probe_train_frac"])


CALIBRATION_TARGETS: dict[str, tuple[float, float]] = {
    "mean_spread_ticks": (1.0, 1.3),
    "mid_ret_1s_acf1": (-0.5, 0.0),
    "probe_oos_r2": (0.005, 0.03),
    "probe_rank_ic": (0.03, 0.10),
}


def target_ranges(cfg: Config) -> dict[str, tuple[float, float]]:
    """Acceptance ranges for the calibration statistics (§2.6)."""
    vol = cfg["data"]["annual_vol"]
    return {"annual_vol": (0.85 * vol, 1.15 * vol), "sign_acf_1": (0.05, 1.0)} | CALIBRATION_TARGETS


def format_calibration(stats: dict[str, float], cfg: Config) -> str:
    """Plain-text calibration report with pass/fail per target."""
    ranges = target_ranges(cfg)
    lines = ["Calibration report", "-" * 60]
    for key, value in stats.items():
        status = ""
        if key in ranges:
            lo, hi = ranges[key]
            status = f"[{lo:g}, {hi:g}] " + ("OK" if lo <= value <= hi else "OUT OF RANGE")
        lines.append(f"{key:<22}{value:>12.4f}   {status}")
    return "\n".join(lines)
