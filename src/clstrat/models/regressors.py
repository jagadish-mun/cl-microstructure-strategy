"""Forecast models: zero and AR(1) baselines, OLS with Newey-West errors, Ridge and Lasso.

Every model takes a preprocessed design matrix ``X`` and a training target ``y`` in ticks.
Ridge and Lasso choose alpha on ``splits``: purged, day-blocked splits of the training fold.
"""

from __future__ import annotations

import warnings

import numpy as np
import statsmodels.api as sm
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV, RidgeCV

from clstrat.config import Config


class ZeroModel:
    """Predicts no move. The benchmark for out-of-sample R^2 on returns."""

    coef_: np.ndarray | None = None
    alpha_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, splits: list) -> ZeroModel:
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))


class LinearModel:
    """OLS fitted by statsmodels; HAC (Newey-West) standard errors with ``hac_lags`` lags,
    because overlapping targets make plain OLS errors far too small. Also used for AR(1),
    where ``X`` is the single past return over the target horizon."""

    alpha_: float | None = None

    def __init__(self, hac_lags: int) -> None:
        self.hac_lags = hac_lags

    def fit(self, X: np.ndarray, y: np.ndarray, splits: list) -> LinearModel:
        res = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit(
            cov_type="HAC", cov_kwds={"maxlags": self.hac_lags}
        )
        self.intercept_, self.coef_ = res.params[0], res.params[1:]
        self.tvalues_ = res.tvalues[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.intercept_ + X @ self.coef_


class PenalisedModel:
    """Ridge or Lasso with alpha picked by purged inner time-series CV."""

    def __init__(self, kind: str, alphas: list[float]) -> None:
        self.kind = kind
        self.alphas = alphas

    def fit(self, X: np.ndarray, y: np.ndarray, splits: list) -> PenalisedModel:
        if self.kind == "ridge":
            est = RidgeCV(alphas=self.alphas, cv=splits, scoring="neg_mean_squared_error")
        else:
            est = LassoCV(alphas=self.alphas, cv=splits, max_iter=5000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.est = est.fit(X, y)
        self.coef_, self.alpha_ = self.est.coef_, float(self.est.alpha_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.est.predict(X)


Model = ZeroModel | LinearModel | PenalisedModel


def make_model(name: str, cfg: Config, hac_lags: int) -> Model:
    """Instantiate a model by its config name."""
    mp = cfg["model_params"]
    if name == "zero":
        return ZeroModel()
    if name in ("ar1", "ols"):
        return LinearModel(hac_lags)
    if name == "ridge":
        return PenalisedModel("ridge", mp["ridge_alphas"])
    if name == "lasso":
        return PenalisedModel("lasso", mp["lasso_alphas"])
    raise ValueError(f"unknown model: {name}")
