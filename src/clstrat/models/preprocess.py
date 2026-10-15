"""Train-only preprocessing: NaN fill, winsorisation, scaling and feature selection.

``prepare_rows`` is stateless (drops each session's warmup and forward-fills within a day).
``Preprocessor.fit`` learns every parameter from training rows only; ``transform`` applies
them unchanged to any other rows.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LassoCV

from clstrat.config import Config


def prepare_rows(df: pd.DataFrame, features: list[str], warmup_s: int) -> pd.DataFrame:
    """Drop rows inside each session's warmup, then forward-fill features within each day."""
    out = df[df["sec"] > warmup_s].copy()
    out[features] = out.groupby("day")[features].ffill()
    return out.reset_index(drop=True)


def correlation_filter(X: np.ndarray, y: np.ndarray, threshold: float) -> list[int]:
    """Greedy filter: visit features by |corr with y| (descending) and keep one unless its
    |correlation| with an already kept feature exceeds ``threshold``."""
    corr = np.corrcoef(X, rowvar=False)
    target = np.abs([np.corrcoef(X[:, j], y)[0, 1] for j in range(X.shape[1])])
    kept: list[int] = []
    for j in np.argsort(-np.nan_to_num(target)):
        if all(abs(corr[j, k]) <= threshold for k in kept):
            kept.append(int(j))
    return sorted(kept)


def _lasso_cv(X: np.ndarray, y: np.ndarray, alphas: list[float], splits: list) -> LassoCV:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return LassoCV(alphas=alphas, cv=splits, max_iter=5000).fit(X, y)


def stability_frequencies(
    X: np.ndarray,
    y: np.ndarray,
    days: np.ndarray,
    alpha: float,
    n_boot: int,
    frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Share of day-block subsamples in which Lasso(alpha) gives each feature a non-zero
    coefficient (Meinshausen-Bühlmann stability selection, subsampling whole days)."""
    unique = np.unique(days)
    n_pick = max(2, int(round(frac * len(unique))))
    hits = np.zeros(X.shape[1])
    for _ in range(n_boot):
        rows = np.isin(days, rng.choice(unique, n_pick, replace=False))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            coef = Lasso(alpha=alpha, max_iter=5000).fit(X[rows], y[rows]).coef_
        hits += coef != 0
    return hits / n_boot


class Preprocessor:
    """sklearn-style fit/transform; every learned parameter comes from the training rows."""

    def __init__(self, p: Config, lasso_alphas: list[float], seed: int) -> None:
        self.p = p
        self.lasso_alphas = lasso_alphas
        self.rng = np.random.default_rng(seed)

    def fit(self, X: pd.DataFrame, y: np.ndarray, days: np.ndarray, splits: list) -> Preprocessor:
        """Learn fill values, winsor bounds, scaler and the selected feature subset."""
        self.columns = list(X.columns)
        values = X.to_numpy(float)
        self.medians = np.nanmedian(values, axis=0)
        self.medians = np.where(np.isnan(self.medians), 0.0, self.medians)
        values = np.where(np.isnan(values), self.medians, values)
        lo_q, hi_q = self.p["winsor_quantiles"]
        self.lo, self.hi = np.quantile(values, [lo_q, hi_q], axis=0)
        self.y_lo, self.y_hi = np.quantile(y, [lo_q, hi_q])
        values = np.clip(values, self.lo, self.hi)
        self.mean = values.mean(axis=0)
        self.std = values.std(axis=0)
        self.std[self.std == 0] = 1.0
        Z = (values - self.mean) / self.std
        self.selected, self.frequencies = self._select(Z, self.clip_target(y), days, splits)
        return self

    def _select(
        self, Z: np.ndarray, y: np.ndarray, days: np.ndarray, splits: list
    ) -> tuple[list[int], np.ndarray]:
        mode = self.p["selection"]
        n = Z.shape[1]
        if mode == "none":
            return list(range(n)), np.ones(n)
        live = [j for j in correlation_filter(Z, y, self.p["corr_filter"]) if Z[:, j].std() > 0]
        lasso = _lasso_cv(Z[:, live], y, self.lasso_alphas, splits)
        freq = np.zeros(n)
        if mode == "lasso":
            freq[live] = lasso.coef_ != 0
        elif mode == "lasso_stability":
            freq[live] = stability_frequencies(
                Z[:, live], y, days, lasso.alpha_,
                self.p["stability_bootstraps"], self.p["stability_sample_frac"], self.rng,
            )  # fmt: skip
        else:
            raise ValueError(f"unknown selection mode: {mode}")
        keep = [j for j in range(n) if freq[j] >= self.p["stability_threshold"]]
        if not keep:  # always keep the single most stable feature
            keep = [int(np.argmax(freq))]
        return keep, freq

    @property
    def selected_names(self) -> list[str]:
        """Names of the features that survived selection."""
        return [self.columns[j] for j in self.selected]

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fill, clip and standardise with the training parameters; keep selected columns."""
        values = X[self.columns].to_numpy(float)
        values = np.where(np.isnan(values), self.medians, values)
        values = (np.clip(values, self.lo, self.hi) - self.mean) / self.std
        return values[:, self.selected]

    def clip_target(self, y: np.ndarray) -> np.ndarray:
        """Training-only target winsorisation (identity when disabled)."""
        return np.clip(y, self.y_lo, self.y_hi) if self.p["winsor_target"] else y
