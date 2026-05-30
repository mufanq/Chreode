"""CellStream-style trajectory and velocity metrics."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors


def distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Distance correlation without depending on `dcor`."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if x.shape[0] < 2 or y.shape[0] < 2:
        return float("nan")
    a = squareform(pdist(x, metric="euclidean"))
    b = squareform(pdist(y, metric="euclidean"))
    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()
    dcov2 = np.mean(A * B)
    dvar_x = np.mean(A * A)
    dvar_y = np.mean(B * B)
    denom = np.sqrt(max(dvar_x * dvar_y, 0.0))
    if denom <= 1e-15:
        return float("nan")
    return float(np.sqrt(max(dcov2, 0.0) / denom))


def temporal_consistency_radius(z: np.ndarray, labels: np.ndarray, *, radius: float = 0.05) -> float:
    z = np.asarray(z, dtype=np.float32)
    labels = np.asarray(labels)
    dists = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
    neighbors = dists <= float(radius)
    same = labels[:, None] == labels[None, :]
    denom = max(float(neighbors.sum()), 1.0)
    return float((neighbors & same).sum() / denom)


def velocity_consistency_radius(z: np.ndarray, labels: np.ndarray, velocity: np.ndarray,
                                *, radius: float = 0.05) -> float:
    z = np.asarray(z, dtype=np.float32)
    labels = np.asarray(labels)
    velocity = np.asarray(velocity, dtype=np.float32)
    dists = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
    mask = (dists <= float(radius)) & (labels[:, None] != labels[None, :])
    return _masked_cosine_mean(velocity, mask)


def temporal_consistency_knn(z: np.ndarray, labels: np.ndarray, *, k: int = 20) -> float:
    z = np.asarray(z, dtype=np.float32)
    labels = np.asarray(labels)
    n_neighbors = min(int(k) + 1, z.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(z)
    inds = nn.kneighbors(z, return_distance=False)[:, 1:]
    return float((labels[inds] == labels[:, None]).mean())


def velocity_consistency_knn(z: np.ndarray, labels: np.ndarray, velocity: np.ndarray, *, k: int = 20) -> float:
    z = np.asarray(z, dtype=np.float32)
    labels = np.asarray(labels)
    velocity = np.asarray(velocity, dtype=np.float32)
    n_neighbors = min(int(k) + 1, z.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(z)
    inds = nn.kneighbors(z, return_distance=False)[:, 1:]
    rows = np.repeat(np.arange(z.shape[0])[:, None], inds.shape[1], axis=1)
    mask = labels[inds] != labels[:, None]
    cos = _pairwise_cosine(velocity[rows], velocity[inds])
    vals = cos[mask]
    return float(np.nanmean(vals)) if vals.size else float("nan")


def _pairwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = (a * b).sum(axis=-1)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    out = dot / np.maximum(denom, 1e-12)
    return np.clip(out, -1.0, 1.0)


def _masked_cosine_mean(velocity: np.ndarray, mask: np.ndarray) -> float:
    vel_i = velocity[:, None, :]
    vel_j = velocity[None, :, :]
    cos = _pairwise_cosine(vel_i, vel_j)
    vals = cos[mask]
    return float(np.nanmean(vals)) if vals.size else float("nan")
