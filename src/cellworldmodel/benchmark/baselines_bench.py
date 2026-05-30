"""Trivial + OT-oracle baselines for BranchSBM-style benchmarks.

These operate at the population level (no explicit source-target pairing),
matching the evaluation protocol of the BranchSBM paper.

Provides:
  - Identity: pred = source (no transformation)
  - MeanShift: pred = source + (mean_target_train - mean_source_train)
  - OT Barycentric Oracle: pred_i = sum_j T_ij * y_j with full target access
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def identity_baseline(source: torch.Tensor, K: int = 1) -> torch.Tensor:
    """ẑ = z (copy source, repeated K times per source cell)."""
    if K == 1:
        return source.clone()
    return source.repeat_interleave(K, dim=0)


def mean_shift_baseline(
    source_test: torch.Tensor,
    source_train: torch.Tensor,
    target_train: torch.Tensor,
    K: int = 1,
) -> torch.Tensor:
    """ẑ = z + (mean_target_train - mean_source_train).

    Uses train-side means to avoid test leakage, applies to source_test.
    """
    shift = target_train.mean(0) - source_train.mean(0)
    pred = source_test + shift.to(source_test.device)
    if K == 1:
        return pred
    return pred.repeat_interleave(K, dim=0)


def ot_barycentric_baseline(
    source: torch.Tensor,
    target: torch.Tensor,
    reg: float = 0.0,
    max_samples: int = 2000,
    seed: int = 42,
) -> torch.Tensor:
    """OT barycentric oracle: predict each source's barycenter under OT plan.

    For each source_i, compute:
        pred_i = sum_j T_ij * target_j / sum_j T_ij

    where T is the OT plan between source and target (exact if reg=0, sinkhorn otherwise).

    THIS IS AN ORACLE — uses access to the test target distribution directly.
    Always mark as oracle in results tables; do not present as a predictive model.

    Args:
        source: (N_s, D)
        target: (N_t, D)
        reg: entropic regularization. 0 = exact EMD (POT.emd). >0 = sinkhorn.
        max_samples: subsample cap for both sides (tractability on large pop).
        seed: random seed for subsampling.

    Returns:
        pred: (N_s, D) predicted target barycenter per source cell.
              If subsampled, returns predictions only for the subsampled sources.
    """
    import ot

    rng = np.random.default_rng(seed)
    src = source.detach().cpu().numpy().astype(np.float64)
    tgt = target.detach().cpu().numpy().astype(np.float64)

    # Subsample if needed (exact OT is O(n³))
    if src.shape[0] > max_samples:
        idx_s = rng.choice(src.shape[0], size=max_samples, replace=False)
        src = src[idx_s]
    if tgt.shape[0] > max_samples:
        idx_t = rng.choice(tgt.shape[0], size=max_samples, replace=False)
        tgt = tgt[idx_t]

    n_s, n_t = src.shape[0], tgt.shape[0]
    a = np.ones(n_s, dtype=np.float64) / n_s
    b = np.ones(n_t, dtype=np.float64) / n_t
    M = ot.dist(src, tgt, metric="sqeuclidean")

    if reg == 0.0:
        T = ot.emd(a, b, M)
    else:
        T = ot.sinkhorn(a, b, M, reg=reg)

    # Barycentric: pred_i = (T @ tgt) / T.sum(axis=1, keepdims=True)
    row_sum = T.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum < 1e-12, 1.0, row_sum)  # avoid div0
    pred = (T @ tgt) / row_sum

    return torch.from_numpy(pred.astype(np.float32))
