"""Purged walk-forward cross-validation with an embargo (López de Prado, ch. 7).

Targets overlap in time (a 60 s target is sampled every 10 s), so a training row whose
target window reaches into the test period leaks test information. Such rows are purged,
and an embargo removes training rows within a gap of at least ``horizon + latency`` on either
side of the test period.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    """One walk-forward split, by trading day."""

    index: int
    train_days: tuple[pd.Timestamp, ...]
    test_days: tuple[pd.Timestamp, ...]


def walk_forward_folds(days: list[pd.Timestamp], train_days: int, test_days: int) -> list[Fold]:
    """Rolling windows: ``train_days`` days of training, then the next ``test_days`` days of
    test, stepping forward by ``test_days``. Every test day appears in at most one fold."""
    days = sorted(days)
    folds = []
    start = 0
    while start + train_days + test_days <= len(days):
        train = tuple(days[start : start + train_days])
        test = tuple(days[start + train_days : start + train_days + test_days])
        folds.append(Fold(len(folds), train, test))
        start += test_days
    return folds


def embargo_gap(horizon_s: int, latency_s: int, embargo_s: int) -> pd.Timedelta:
    """Embargo length: the configured value, but never shorter than ``horizon + latency``."""
    return pd.Timedelta(seconds=max(embargo_s, horizon_s + latency_s))


def purge_mask(
    train_ts: pd.Series,
    test_ts: pd.Series,
    horizon_s: int,
    latency_s: int,
    embargo_s: int,
) -> np.ndarray:
    """Boolean mask of training rows to keep.

    A row at ``t`` is dropped if its target window ``[t + latency, t + latency + horizon]``
    overlaps the test period ``[test_start, test_end + latency + horizon]`` (purge), or if
    ``t`` lies within the embargo gap before the test start or after the test end.
    """
    start, end = test_ts.min(), test_ts.max()
    reach = pd.Timedelta(seconds=horizon_s + latency_s)
    gap = embargo_gap(horizon_s, latency_s, embargo_s)
    t = train_ts
    overlaps = (t + reach >= start) & (t <= end + reach)
    before = (t >= start - gap) & (t < start)
    after = (t > end) & (t <= end + gap)
    return ~(overlaps | before | after).to_numpy()


def fold_indices(
    df: pd.DataFrame, fold: Fold, horizon_s: int, latency_s: int, embargo_s: int
) -> tuple[np.ndarray, np.ndarray]:
    """Row positions of ``df`` (needs ``day`` and ``ts``) for the fold's purged train and test."""
    train = np.flatnonzero(df["day"].isin(fold.train_days).to_numpy())
    test = np.flatnonzero(df["day"].isin(fold.test_days).to_numpy())
    keep = purge_mask(df["ts"].iloc[train], df["ts"].iloc[test], horizon_s, latency_s, embargo_s)
    return train[keep], test


def inner_splits(
    days: np.ndarray, ts: pd.Series, n_splits: int, horizon_s: int, latency_s: int, embargo_s: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding, day-blocked, purged splits inside a training fold (for choosing alphas).

    Days are cut into ``n_splits + 1`` consecutive blocks; split ``k`` trains on blocks
    ``0..k-1`` and validates on block ``k``. Positions are relative to the inputs.
    """
    blocks = np.array_split(np.unique(days), n_splits + 1)
    ts = ts.reset_index(drop=True)
    splits = []
    for k in range(1, n_splits + 1):
        tr = np.flatnonzero(np.isin(days, np.concatenate(blocks[:k])))
        va = np.flatnonzero(np.isin(days, blocks[k]))
        keep = purge_mask(ts.iloc[tr], ts.iloc[va], horizon_s, latency_s, embargo_s)
        splits.append((tr[keep], va))
    return splits
