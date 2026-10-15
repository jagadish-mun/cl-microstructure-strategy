"""Robustness checks: placebo, regimes, deflated Sharpe and independent synthetic markets."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kurtosis, norm, skew

from clstrat.backtest.engine import Engine
from clstrat.backtest.metrics import summarize
from clstrat.backtest.vectorized import VectorBacktester
from clstrat.config import Config, data_path, with_overrides
from clstrat.contract import SessionCalendar
from clstrat.data.adapters import make_source
from clstrat.data.synthetic import generate_dataset
from clstrat.features.pipeline import build_dataset
from clstrat.models.evaluate import model_comparison, run_walk_forward, train_quantiles
from clstrat.research.sweeps import daily_stats

EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# Placebo
# ---------------------------------------------------------------------------


def placebo(vb: VectorBacktester, cfg: Config, n: int, seed: int) -> pd.DataFrame:
    """Net results after shuffling the predictions within each day, ``n`` times.

    Shuffling keeps each day's prediction distribution (so trade counts and costs are
    similar) but destroys the timing information; a real signal should beat it clearly.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame([daily_stats(vb.shuffled(rng).run(cfg)) for _ in range(n)])


def placebo_p_value(real: float, placebo_values: pd.Series) -> float:
    """Share of shuffles at least as good as the real result (with the +1 correction)."""
    return float((1 + (placebo_values >= real).sum()) / (1 + len(placebo_values)))


# ---------------------------------------------------------------------------
# Regimes
# ---------------------------------------------------------------------------


def day_labels(
    cfg: Config, days: list[date], load_bars: Callable[[date], pd.DataFrame]
) -> pd.DataFrame:
    """Per-day labels for the regime split.

    - ``realized_vol``: above or below the median realized vol of the days analysed
      (works on any data)
    - ``generator_regime``: the simulator's own calm/volatile state (synthetic data only)
    - ``eia``: EIA release day or not
    """
    cal = SessionCalendar.from_config(cfg)
    index = pd.DatetimeIndex([pd.Timestamp(d) for d in days], name="day")
    vols = np.array([np.log(load_bars(d)["mid"]).diff().std() for d in days])
    labels = pd.DataFrame(
        {
            "realized_vol": np.where(vols > np.median(vols), "high-vol half", "low-vol half"),
            "eia": ["EIA day" if cal.is_eia_day(d) else "non-EIA day" for d in days],
        },
        index=index,
    )
    meta = data_path(cfg, "synthetic", "generator_config.yaml")
    if cfg["data"]["source"] == "synthetic" and meta.exists():
        with open(meta) as fh:
            states = yaml.safe_load(fh)["regimes"]
        labels["generator_regime"] = ["volatile" if states[d.isoformat()] else "calm" for d in days]
    return labels


def regime_table(daily: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Net P&L per group for every label column."""
    joined = daily.join(labels)
    rows = []
    for column in labels.columns:
        for name, g in joined.groupby(column):
            rows.append(
                {
                    "split": column,
                    "group": name,
                    "days": len(g),
                    "net_pnl": g["net"].sum(),
                    "net_per_day": g["net"].mean(),
                    "gross_per_day": g["gross"].mean(),
                    "positive_days": (g["net"] > 0).mean(),
                }
            )
    return pd.DataFrame(rows).set_index(["split", "group"])


# ---------------------------------------------------------------------------
# Multiple testing
# ---------------------------------------------------------------------------


def expected_max_sharpe(trial_sharpes: np.ndarray) -> float:
    """Expected maximum of ``N`` per-period Sharpe ratios under the null of no skill
    (Bailey & López de Prado 2014), using the variance across the trials."""
    n = len(trial_sharpes)
    sd = np.std(trial_sharpes, ddof=1)
    return float(
        sd
        * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n) + EULER_GAMMA * norm.ppf(1 - 1 / (n * math.e)))
    )


def probabilistic_sharpe(daily: pd.Series, benchmark: float) -> float:
    """P(true per-period Sharpe > benchmark), adjusting for sample length, skew and kurtosis."""
    sr = daily.mean() / daily.std(ddof=1)
    t = len(daily)
    g3, g4 = skew(daily), kurtosis(daily, fisher=False)
    denom = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr**2))
    return float(norm.cdf((sr - benchmark) * math.sqrt(t - 1) / denom))


def deflated_sharpe(daily: pd.Series, trial_sharpes_annual: np.ndarray) -> dict[str, float]:
    """Deflated Sharpe ratio of the selected strategy given every configuration tried."""
    trials = np.asarray(trial_sharpes_annual, float)
    trials = trials[np.isfinite(trials)] / math.sqrt(252)
    sr0 = expected_max_sharpe(trials)
    return {
        "configs_tried": len(trials),
        "sharpe_annual": float(daily.mean() / daily.std(ddof=1) * math.sqrt(252)),
        "expected_max_sharpe_annual_under_null": sr0 * math.sqrt(252),
        "psr_vs_zero": probabilistic_sharpe(daily, 0.0),
        "deflated_sharpe": probabilistic_sharpe(daily, sr0),
        "days": len(daily),
    }


# ---------------------------------------------------------------------------
# Independent synthetic markets
# ---------------------------------------------------------------------------


def seed_run(cfg: Config, seed: int, strategy: dict, model: str, root: Path) -> dict:
    """Regenerate the market with ``seed`` and rerun the frozen pipeline on it.

    Generator, features, model type and strategy settings are all fixed; only the random
    market changes. Research days only, as in the main run.
    """
    h = cfg["target"]["horizon"]
    c = with_overrides(
        cfg,
        {"seed": seed, "paths.data_dir": str(root / f"seed_{seed}"), "target.horizons_report": [h]},
    )
    generate_dataset(c, data_path(c, "synthetic"))
    df, _ = build_dataset(c, make_source(c), data_path(c, "bars"))
    wf = run_walk_forward(df, c)
    fit = model_comparison(wf, c["models"]).loc[(h, model)]
    preds = wf.predictions[h].rename(columns={f"pred_{model}": "pred"})
    q = train_quantiles(wf, h)
    quantiles = {fold: row.to_numpy() for (fold, m), row in q.iterrows() if m == model}

    def load_bars(day: date) -> pd.DataFrame:
        return pd.read_parquet(data_path(c, "bars", f"{day.isoformat()}.parquet"))

    result = Engine(with_overrides(c, strategy)).run(preds, load_bars, quantiles)
    s = summarize(result)
    return {
        "seed": seed,
        "oos_r2": fit["oos_r2"],
        "ic_spearman": fit["ic_spearman"],
        "net_pnl": s["net_pnl"],
        "gross_pnl": s["gross_pnl"],
        "net_sharpe": s["net_sharpe"],
        "max_drawdown": s["max_drawdown"],
        "trades": s["trades"],
    }
