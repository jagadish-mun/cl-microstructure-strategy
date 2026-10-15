"""Walk-forward model evaluation: OOS R^2 vs zero, IC, hit rate, coefficient stability."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr

from clstrat.config import Config
from clstrat.features.pipeline import feature_columns, latency_seconds, research_days
from clstrat.models.cv import Fold, fold_indices, inner_splits, walk_forward_folds
from clstrat.models.preprocess import Preprocessor, prepare_rows
from clstrat.models.regressors import make_model

BASELINES = ("zero", "ar1")

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def oos_r2(y: np.ndarray, pred: np.ndarray) -> float:
    """``1 - Σ(y - ŷ)² / Σ y²``: R^2 against the zero forecast, not the sample mean."""
    return float(1 - np.sum((y - pred) ** 2) / np.sum(y**2))


def hit_rate(y: np.ndarray, pred: np.ndarray) -> float:
    """Share of rows where the predicted sign matches the realised sign (ties excluded)."""
    mask = (y != 0) & (pred != 0)
    return float(np.mean(np.sign(y[mask]) == np.sign(pred[mask]))) if mask.any() else math.nan


def forecast_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """OOS R^2, Pearson and Spearman IC, hit rate and row count."""
    varies = np.std(pred) > 0
    return {
        "oos_r2": oos_r2(y, pred),
        "ic_pearson": float(pearsonr(pred, y)[0]) if varies else math.nan,
        "ic_spearman": float(spearmanr(pred, y)[0]) if varies else math.nan,
        "hit_rate": hit_rate(y, pred),
        "n": len(y),
    }


# ---------------------------------------------------------------------------
# Walk-forward runner
# ---------------------------------------------------------------------------


@dataclass
class FoldFit:
    """One model fitted on one fold at one horizon."""

    fold: int
    horizon: int
    model: str
    features: list[str]
    coefs: np.ndarray | None
    tvalues: np.ndarray | None
    alpha: float | None
    n_train: int
    n_test: int
    train_quantiles: np.ndarray


@dataclass
class WalkForwardResult:
    """Out-of-sample predictions per horizon, fitted-model records and selection frequencies."""

    predictions: dict[int, pd.DataFrame]
    fits: list[FoldFit] = field(default_factory=list)
    selection: dict[int, pd.DataFrame] = field(default_factory=dict)


def hac_lags(cfg: Config, horizon: int) -> int:
    """Newey-West lags: the number of decision steps a target overlaps with its neighbours."""
    step = pd.Timedelta(cfg["sampling"]["decision_interval"]).total_seconds()
    return math.ceil((horizon + latency_seconds(cfg)) / step)


def model_rows(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Research-day rows (holdout removed), warmup dropped and features filled within days."""
    research = research_days(df, cfg["cv"]["holdout_days"])
    return prepare_rows(research, feature_columns(df), cfg["preprocess"]["warmup_seconds"])


class FoldTrainer:
    """Fits the preprocessing and every model on one training set at one horizon."""

    def __init__(self, cfg: Config, horizon: int, seed: int) -> None:
        self.cfg = cfg
        self.horizon = horizon
        self.seed = seed
        self.lat = latency_seconds(cfg)

    def _splits(self, train: pd.DataFrame) -> list:
        return inner_splits(
            train["day"].to_numpy(), train["ts"], self.cfg["model_params"]["inner_splits"],
            self.horizon, self.lat, self.cfg["cv"]["embargo_seconds"],
        )  # fmt: skip

    def _preprocessor(self, p: Config) -> Preprocessor:
        return Preprocessor(p, self.cfg["model_params"]["lasso_alphas"], self.seed)

    def fit(self, train: pd.DataFrame, features: list[str]) -> dict[str, tuple]:
        """Return ``{model: (preprocessor, fitted_model)}`` for every configured model."""
        y = train[f"y_{self.horizon}"].to_numpy()
        days = train["day"].to_numpy()
        splits = self._splits(train)
        main = self._preprocessor(self.cfg["preprocess"]).fit(train[features], y, days, splits)
        ar_cfg = {**self.cfg["preprocess"], "selection": "none"}
        ar_col = [f"ret_{self.horizon}"]
        ar = self._preprocessor(ar_cfg).fit(train[ar_col], y, days, splits)
        fitted = {}
        for name in self.cfg["models"]:
            pre = ar if name == "ar1" else main
            model = make_model(name, self.cfg, hac_lags(self.cfg, self.horizon))
            model.fit(pre.transform(train), pre.clip_target(y), splits)
            fitted[name] = (pre, model)
        return fitted


def _record(fold: int, horizon: int, name: str, pre, model, train, n_test, grid) -> FoldFit:
    train_pred = model.predict(pre.transform(train))
    linear = name != "zero"
    return FoldFit(
        fold=fold,
        horizon=horizon,
        model=name,
        features=pre.selected_names if linear else [],
        coefs=getattr(model, "coef_", None),
        tvalues=getattr(model, "tvalues_", None),
        alpha=getattr(model, "alpha_", None),
        n_train=len(train),
        n_test=n_test,
        train_quantiles=np.quantile(train_pred, np.linspace(0, 1, grid)),
    )


def run_fold(
    rows: pd.DataFrame, fold: Fold, horizon: int, cfg: Config
) -> tuple[pd.DataFrame, list[FoldFit], pd.Series]:
    """Fit on the purged training days of ``fold`` and predict its test days."""
    target = f"y_{horizon}"
    tr, te = fold_indices(rows, fold, horizon, latency_seconds(cfg), cfg["cv"]["embargo_seconds"])
    train = rows.iloc[tr].dropna(subset=[target]).reset_index(drop=True)
    test = rows.iloc[te].dropna(subset=[target]).reset_index(drop=True)
    features = feature_columns(rows)
    fitted = FoldTrainer(cfg, horizon, cfg["seed"] + fold.index).fit(train, features)
    out = test[["ts", "day", target]].assign(fold=fold.index)
    grid = cfg["model_params"]["quantile_grid"]
    fits = []
    for name, (pre, model) in fitted.items():
        out[f"pred_{name}"] = model.predict(pre.transform(test))
        fits.append(_record(fold.index, horizon, name, pre, model, train, len(test), grid))
    main_pre = next(pre for name, (pre, _) in fitted.items() if name not in BASELINES)
    freq = pd.Series(main_pre.frequencies, index=features, name=fold.index)
    return out, fits, freq


def run_walk_forward(df: pd.DataFrame, cfg: Config) -> WalkForwardResult:
    """Purged walk-forward over the research days for every reporting horizon."""
    rows = model_rows(df, cfg)
    c = cfg["cv"]
    folds = walk_forward_folds(sorted(rows["day"].unique()), c["train_days"], c["test_days"])
    result = WalkForwardResult(predictions={})
    for h in cfg["target"]["horizons_report"]:
        preds, freqs = [], []
        for fold in folds:
            out, fits, freq = run_fold(rows, fold, h, cfg)
            preds.append(out)
            freqs.append(freq)
            result.fits.extend(fits)
        result.predictions[h] = pd.concat(preds, ignore_index=True)
        result.selection[h] = pd.concat(freqs, axis=1)
    return result


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def model_comparison(result: WalkForwardResult, models: list[str]) -> pd.DataFrame:
    """Pooled OOS metrics per (horizon, model), with per-fold R^2 mean, std and share > 0."""
    rows = []
    for h, pred in result.predictions.items():
        y = pred[f"y_{h}"].to_numpy()
        for m in models:
            p = pred[f"pred_{m}"].to_numpy()
            fold_r2 = pred.groupby("fold").apply(
                lambda g, m=m, h=h: oos_r2(g[f"y_{h}"].to_numpy(), g[f"pred_{m}"].to_numpy())
            )
            rows.append(
                {"horizon": h, "model": m} | forecast_metrics(y, p) | {
                    "fold_r2_mean": fold_r2.mean(),
                    "fold_r2_std": fold_r2.std(),
                    "folds_r2_pos": float((fold_r2 > 0).mean()),
                }
            )  # fmt: skip
    return pd.DataFrame(rows).set_index(["horizon", "model"])


def fold_metrics(result: WalkForwardResult, horizon: int, models: list[str]) -> pd.DataFrame:
    """Per-fold OOS metrics at one horizon."""
    pred = result.predictions[horizon]
    rows = []
    for fold, g in pred.groupby("fold"):
        y = g[f"y_{horizon}"].to_numpy()
        for m in models:
            rows.append({"fold": fold, "model": m} | forecast_metrics(y, g[f"pred_{m}"].to_numpy()))
    return pd.DataFrame(rows).set_index(["fold", "model"])


def coefficient_table(result: WalkForwardResult, horizon: int, model: str) -> pd.DataFrame:
    """Standardised coefficients, features x folds (NaN where a feature was not selected)."""
    cols = {
        f.fold: pd.Series(f.coefs, index=f.features)
        for f in result.fits
        if f.horizon == horizon and f.model == model and f.coefs is not None
    }
    return pd.DataFrame(cols).sort_index()


def sign_flips(coefs: pd.DataFrame) -> pd.Series:
    """True for features whose non-zero coefficient changes sign across folds."""
    pos = (coefs > 0).any(axis=1)
    neg = (coefs < 0).any(axis=1)
    return (pos & neg).rename("sign_flip")


def best_model(comparison: pd.DataFrame, horizon: int) -> str:
    """Non-baseline model with the highest pooled OOS R^2 at ``horizon`` (walk-forward only)."""
    table = comparison.loc[horizon]
    return str(table.drop(index=[m for m in BASELINES if m in table.index])["oos_r2"].idxmax())


def train_quantiles(result: WalkForwardResult, horizon: int) -> pd.DataFrame:
    """Quantile grid of each model's in-sample training predictions, per fold. Used by the
    percentile-threshold strategy variant to place test predictions in the training law."""
    rows = {(f.fold, f.model): f.train_quantiles for f in result.fits if f.horizon == horizon}
    frame = pd.DataFrame(rows).T
    frame.index.names = ["fold", "model"]
    frame.columns = [f"q{i:03d}" for i in range(frame.shape[1])]
    return frame


def save_results(result: WalkForwardResult, cfg: Config, models_dir: Path) -> dict:
    """Write primary-horizon predictions, training-prediction quantiles and the model choice."""
    h = cfg["target"]["horizon"]
    models_dir.mkdir(parents=True, exist_ok=True)
    result.predictions[h].to_parquet(models_dir / "predictions.parquet", index=False)
    train_quantiles(result, h).to_parquet(models_dir / "train_quantiles.parquet")
    comparison = model_comparison(result, cfg["models"])
    choice = {
        "horizon": h,
        "best_model": best_model(comparison, h),
        "metrics": {k: float(v) for k, v in comparison.loc[(h, best_model(comparison, h))].items()},
    }
    with open(models_dir / "selection.yaml", "w") as fh:
        yaml.safe_dump(choice, fh, sort_keys=False)
    return choice
