"""Preprocessing leakage, purged walk-forward CV, metrics and regressors."""

import numpy as np
import pandas as pd
import pytest

from clstrat.config import load_config
from clstrat.models.cv import (
    Fold,
    embargo_gap,
    fold_indices,
    inner_splits,
    purge_mask,
    walk_forward_folds,
)
from clstrat.models.evaluate import hit_rate, oos_r2, run_fold
from clstrat.models.preprocess import Preprocessor, correlation_filter, prepare_rows
from clstrat.models.regressors import LinearModel, PenalisedModel

CFG = load_config()
H, LAT, EMB = 60, 1, 120


def _toy_dataset(n_days: int = 8, per_day: int = 120, seed: int = 0) -> pd.DataFrame:
    """Decision rows every 10 s; y_60 depends on two of five features."""
    rng = np.random.default_rng(seed)
    frames = []
    for d in range(n_days):
        day = pd.Timestamp("2024-01-02") + pd.Timedelta(days=d)
        sec = np.arange(1, per_day + 1) * 10
        X = rng.normal(size=(per_day, 5))
        y = 0.8 * X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=2.0, size=per_day)
        frame = pd.DataFrame(X, columns=["ret_60", "f1", "f2", "f3", "f4"])
        frame.insert(0, "sec", sec)
        frame.insert(0, "day", day)
        frame.insert(0, "ts", (day + pd.Timedelta(hours=9) + pd.to_timedelta(sec, "s")).tz_localize(
            "America/New_York"))  # fmt: skip
        frames.append(frame.assign(eia_day=0.0, y_60=y))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def test_walk_forward_folds_are_disjoint_and_ordered() -> None:
    days = list(pd.date_range("2024-01-01", periods=50))
    folds = walk_forward_folds(days, 20, 5)
    assert len(folds) == 6
    tested = [d for f in folds for d in f.test_days]
    assert len(tested) == len(set(tested)) == 30
    for f in folds:
        assert max(f.train_days) < min(f.test_days)
        assert not set(f.train_days) & set(f.test_days)


def test_no_training_row_inside_purge_or_embargo_zone() -> None:
    ts = pd.Series(pd.date_range("2024-01-02 09:00", periods=2000, freq="10s", tz="UTC"))
    test = ts.iloc[800:1000]
    train_idx = np.r_[0:800, 1000:2000]
    keep = purge_mask(ts.iloc[train_idx], test, H, LAT, EMB)
    kept = ts.iloc[train_idx[keep]]
    start, end = test.min(), test.max()
    gap = embargo_gap(H, LAT, EMB)
    reach = pd.Timedelta(seconds=H + LAT)
    assert gap >= reach
    assert not ((kept >= start - gap) & (kept <= end + gap)).any()
    assert not ((kept + reach >= start) & (kept <= end + reach)).any()
    assert keep.sum() == 2000 - 200 - 2 * int(gap.total_seconds() // 10)


def test_fold_indices_purge_when_days_touch() -> None:
    df = _toy_dataset(n_days=4)
    # squeeze the calendar so day k+1 starts 5 s after day k ends: the purge must bind
    df["ts"] = pd.Series(pd.date_range("2024-01-02", periods=len(df), freq="10s", tz="UTC"))
    days = sorted(df["day"].unique())
    fold = Fold(0, tuple(days[:3]), (days[3],))
    tr, te = fold_indices(df, fold, H, LAT, EMB)
    test_start = df["ts"].iloc[te].min()
    assert df["ts"].iloc[tr].max() < test_start - embargo_gap(H, LAT, EMB)
    assert len(tr) < 3 * 120


def test_inner_splits_expand_and_never_validate_on_training_days() -> None:
    df = _toy_dataset(n_days=10)
    splits = inner_splits(df["day"].to_numpy(), df["ts"], 4, H, LAT, EMB)
    assert len(splits) == 4
    for tr, va in splits:
        assert df["day"].iloc[tr].max() < df["day"].iloc[va].min()
    assert len(splits[0][0]) < len(splits[-1][0])


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def test_preprocessor_parameters_come_from_training_rows_only() -> None:
    df = _toy_dataset()
    feats = ["ret_60", "f1", "f2", "f3", "f4"]
    train = df[df["day"] < df["day"].unique()[5]]
    p = {**CFG["preprocess"], "selection": "none"}
    pre = Preprocessor(p, CFG["model_params"]["lasso_alphas"], 0)
    pre.fit(train[feats], train["y_60"].to_numpy(), train["day"].to_numpy(), [])
    lo, hi = np.quantile(train[feats].to_numpy(), p["winsor_quantiles"], axis=0)
    np.testing.assert_allclose(pre.lo, lo)
    np.testing.assert_allclose(pre.hi, hi)
    clipped = np.clip(train[feats].to_numpy(), lo, hi)
    np.testing.assert_allclose(pre.mean, clipped.mean(axis=0))
    extreme = df.iloc[[0]].copy()
    extreme[feats] = 1e6
    np.testing.assert_allclose(pre.transform(extreme), ((hi - pre.mean) / pre.std)[None, :])


def test_fold_fit_is_unaffected_by_test_data() -> None:
    """Scramble test-day features and targets: every train-fitted quantity must be identical."""
    cfg = {**CFG, "models": ["zero", "ar1", "ols", "ridge", "lasso"]}
    cfg["preprocess"] = {**CFG["preprocess"], "stability_bootstraps": 5}
    rows = _toy_dataset(n_days=8)
    days = sorted(rows["day"].unique())
    fold = Fold(0, tuple(days[:6]), tuple(days[6:]))
    _, fits_a, freq_a = run_fold(rows, fold, H, cfg)
    scrambled = rows.copy()
    test = scrambled["day"].isin(fold.test_days)
    cols = ["ret_60", "f1", "f2", "f3", "f4", "y_60"]
    scrambled.loc[test, cols] = np.random.default_rng(9).normal(50, 30, (test.sum(), len(cols)))
    out_b, fits_b, freq_b = run_fold(scrambled, fold, H, cfg)
    pd.testing.assert_series_equal(freq_a, freq_b)
    for a, b in zip(fits_a, fits_b, strict=True):
        assert a.features == b.features and a.alpha == b.alpha
        np.testing.assert_array_equal(a.train_quantiles, b.train_quantiles)
        if a.coefs is not None:
            np.testing.assert_array_equal(a.coefs, b.coefs)
    assert out_b["y_60"].abs().mean() > 10  # the test rows really were scrambled


def test_prepare_rows_fills_within_day_only() -> None:
    df = _toy_dataset(n_days=2, per_day=40)
    df.loc[df.index[39], "f1"] = 7.0  # last row of day 0
    df.loc[df.index[40:45], "f1"] = np.nan  # first rows of day 1
    df.loc[df.index[46], "f1"] = np.nan
    out = prepare_rows(df, ["f1"], warmup_s=0)
    assert out["f1"].iloc[40:45].isna().all()  # not filled from the previous day
    assert out["f1"].iloc[46] == df["f1"].iloc[45]
    assert len(prepare_rows(df, ["f1"], warmup_s=300)) == 2 * 10


def test_correlation_filter_keeps_the_more_predictive_twin() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=500)
    y = a + rng.normal(size=500)
    X = np.column_stack([a + 0.4 * rng.normal(size=500), a, rng.normal(size=500)])  # ρ≈0.93
    assert correlation_filter(X, y, 0.9) == [1, 2]


# ---------------------------------------------------------------------------
# Metrics and models
# ---------------------------------------------------------------------------


def test_oos_r2_is_against_zero_and_hit_rate() -> None:
    y = np.array([1.0, -2.0, 3.0, 0.0])
    assert oos_r2(y, np.zeros(4)) == 0.0
    assert oos_r2(y, y) == 1.0
    assert oos_r2(y, np.full(4, y.mean())) < 1 - np.var(y) * 4 / np.sum(y**2) + 1e-12
    assert hit_rate(y, np.array([0.5, -1.0, -1.0, 2.0])) == pytest.approx(2 / 3)


def test_ols_hac_and_penalised_models_recover_signal() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(3000, 3))
    y = X @ np.array([1.0, 0.0, -0.5]) + rng.normal(size=3000)
    ols = LinearModel(hac_lags=6).fit(X, y, [])
    np.testing.assert_allclose(ols.coef_, [1.0, 0.0, -0.5], atol=0.06)
    assert abs(ols.tvalues_[0]) > 20 and abs(ols.tvalues_[1]) < 3
    splits = [(np.arange(0, 1500), np.arange(1500, 3000))]
    lasso = PenalisedModel("lasso", [0.01, 0.1, 1.0]).fit(X, y, splits)
    assert lasso.alpha_ in (0.01, 0.1) and lasso.coef_[0] > 0.8
    ridge = PenalisedModel("ridge", [0.1, 1e6]).fit(X, y, splits)
    assert ridge.alpha_ == 0.1
