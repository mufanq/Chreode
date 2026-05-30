"""Shared metrics for ZESTA control-only time extrapolation."""
from __future__ import annotations

from typing import Iterable

import numpy as np


DEFAULT_MMD_GAMMAS = (2.0, 1.0, 0.5, 0.1, 0.01, 0.005)


def _as_2d_float(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    return arr


def r2_between_means(true, pred) -> float:
    """CellFlow-style R2 between true and predicted distribution means."""
    x = _as_2d_float(true)
    y = _as_2d_float(pred)
    mx = x.mean(axis=0)
    my = y.mean(axis=0)
    denom = np.sum((mx - mx.mean()) ** 2)
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((mx - my) ** 2) / denom)


def squared_energy_distance(true, pred) -> float:
    """Energy distance used by CellFlow's metric implementation.

    CellFlow computes this with pairwise squared Euclidean distances. Algebra
    reduces that estimator to 2 * ||mean(true) - mean(pred)||^2, which avoids
    materializing an O(n^2) distance matrix for large ZESTA subsets.
    """
    x = _as_2d_float(true)
    y = _as_2d_float(pred)
    diff = x.mean(axis=0) - y.mean(axis=0)
    return float(2.0 * np.dot(diff, diff))


def _rbf_mean(x: np.ndarray, y: np.ndarray, gamma: float, block_size: int) -> float:
    total = 0.0
    count = 0
    y_norm = np.sum(y * y, axis=1)
    for start in range(0, x.shape[0], block_size):
        xb = x[start:start + block_size]
        d2 = np.sum(xb * xb, axis=1)[:, None] + y_norm[None, :] - 2.0 * xb @ y.T
        total += np.exp(-gamma * np.maximum(d2, 0.0)).sum()
        count += d2.size
    return float(total / max(count, 1))


def scalar_mmd(true, pred, gammas: Iterable[float] = DEFAULT_MMD_GAMMAS,
               max_samples: int = 1000, seed: int = 0, block_size: int = 256) -> float:
    """CellFlow-style scalar RBF MMD with deterministic subsampling."""
    x = _as_2d_float(true)
    y = _as_2d_float(pred)
    rng = np.random.default_rng(seed)
    if x.shape[0] > max_samples:
        x = x[np.sort(rng.choice(x.shape[0], size=max_samples, replace=False))]
    if y.shape[0] > max_samples:
        y = y[np.sort(rng.choice(y.shape[0], size=max_samples, replace=False))]

    vals = []
    for gamma in gammas:
        xx = _rbf_mean(x, x, gamma, block_size)
        xy = _rbf_mean(x, y, gamma, block_size)
        yy = _rbf_mean(y, y, gamma, block_size)
        vals.append(xx + yy - 2.0 * xy)
    return float(np.nanmean(vals))


def distribution_metrics(true, pred, *, seed: int = 0, mmd_max_samples: int = 1000) -> dict[str, float]:
    """Metrics used for direct comparison with CellFlow reports."""
    return {
        "r_squared_mean": r2_between_means(true, pred),
        "squared_energy_distance": squared_energy_distance(true, pred),
        "scalar_mmd": scalar_mmd(true, pred, seed=seed, max_samples=mmd_max_samples),
    }
