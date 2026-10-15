"""The final holdout: the last ``cv.holdout_days`` days, used exactly once.

The model type chosen on the walk-forward folds is refitted on the ``cv.train_days`` research
days just before the holdout (same purged, train-only preprocessing), then the strategy
chosen by the robustness suite trades the holdout days with firewalls on. The outcome is
stored in ``data/models/holdout.yaml``; once that file exists the run is never repeated.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import yaml

from clstrat.backtest.engine import Engine, chosen_model
from clstrat.backtest.metrics import summarize
from clstrat.config import Config, data_path, reports_path, with_overrides
from clstrat.features.pipeline import feature_columns
from clstrat.models.cv import Fold
from clstrat.models.evaluate import forecast_metrics, run_fold
from clstrat.models.preprocess import prepare_rows
from clstrat.report.build import plot_equity


def holdout_fold(days: list[pd.Timestamp], cfg: Config) -> Fold:
    """Train on the last ``train_days`` research days, test on the holdout days."""
    c = cfg["cv"]
    holdout = tuple(days[-c["holdout_days"] :])
    train = tuple(days[-c["holdout_days"] - c["train_days"] : -c["holdout_days"]])
    return Fold(index=-1, train_days=train, test_days=holdout)


def run_holdout(cfg: Config) -> dict:
    """Run the holdout once (or return the stored result if it has already been run)."""
    marker = data_path(cfg, "models", "holdout.yaml")
    if marker.exists():
        with open(marker) as fh:
            return yaml.safe_load(fh) | {"note": "holdout already run; stored result shown"}
    with open(data_path(cfg, "models", "strategy.yaml")) as fh:
        strategy = yaml.safe_load(fh)["overrides"]
    model = chosen_model(cfg)
    h = cfg["target"]["horizon"]
    df = pd.read_parquet(data_path(cfg, "features", "dataset.parquet"))
    rows = prepare_rows(df, feature_columns(df), cfg["preprocess"]["warmup_seconds"])
    fold = holdout_fold(sorted(rows["day"].unique()), cfg)
    preds, fits, _ = run_fold(rows, fold, h, cfg)
    fit = next(f for f in fits if f.model == model)
    forecast = forecast_metrics(preds[f"y_{h}"].to_numpy(), preds[f"pred_{model}"].to_numpy())

    def load_bars(day: date) -> pd.DataFrame:
        return pd.read_parquet(data_path(cfg, "bars", f"{day.isoformat()}.parquet"))

    preds = preds.rename(columns={f"pred_{model}": "pred"})
    result = Engine(with_overrides(cfg, strategy)).run(
        preds, load_bars, {fold.index: np.asarray(fit.train_quantiles)}
    )
    stats = summarize(result)
    result.daily.to_csv(reports_path(cfg, "holdout_daily_pnl.csv"))
    result.trades.to_csv(reports_path(cfg, "holdout_trade_log.csv"), index=False)
    plot_equity(result.daily, result.equity, reports_path(cfg, "holdout_equity.png"),
                "Final holdout (run once): cumulative P&L")  # fmt: skip
    outcome = {
        "model": model,
        "train_days": [str(d.date()) for d in (fold.train_days[0], fold.train_days[-1])],
        "holdout_days": [str(d.date()) for d in (fold.test_days[0], fold.test_days[-1])],
        "selected_features": fit.features,
        "forecast": {k: float(v) for k, v in forecast.items()},
        "strategy": strategy,
        "backtest": {k: float(v) for k, v in stats.items()},
    }
    with open(marker, "w") as fh:
        yaml.safe_dump(outcome, fh, sort_keys=False)
    return outcome
