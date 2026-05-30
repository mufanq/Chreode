"""Tests for benchmark baselines (Identity, Mean shift, OT barycentric)."""
from __future__ import annotations

import pytest
import torch

from cellworldmodel.benchmark.baselines_bench import (
    identity_baseline,
    mean_shift_baseline,
    ot_barycentric_baseline,
)


def test_identity_baseline_shape():
    X = torch.randn(50, 3)
    pred = identity_baseline(X, K=1)
    assert pred.shape == (50, 3)
    assert torch.allclose(pred, X)


def test_identity_baseline_K():
    X = torch.randn(10, 5)
    pred = identity_baseline(X, K=4)
    assert pred.shape == (40, 5)


def test_mean_shift_basic():
    """If train mean shift is d, pred should equal source + d."""
    src_train = torch.zeros(100, 3)
    tgt_train = torch.ones(100, 3) * 5.0
    src_test = torch.zeros(20, 3)
    pred = mean_shift_baseline(src_test, src_train, tgt_train, K=1)
    assert pred.shape == (20, 3)
    # Shift = 5.0 in all dims
    assert torch.allclose(pred, torch.ones(20, 3) * 5.0, atol=1e-5)


def test_ot_barycentric_gaussian_shift():
    """OT barycentric between two Gaussians should map ~linearly with shift."""
    torch.manual_seed(0)
    # Source: N(0, I). Target: N(3, I). Barycentric should shift cells roughly by +3.
    X = torch.randn(100, 2)
    Y = torch.randn(100, 2) + 3.0
    pred = ot_barycentric_baseline(X, Y, reg=0.0)
    assert pred.shape == (100, 2)
    # Mean of pred should be close to mean of Y (≈ 3.0)
    assert abs(pred.mean(0)[0].item() - 3.0) < 0.3


def test_ot_barycentric_subsample():
    """Large inputs get subsampled; output shape == subsampled source size."""
    torch.manual_seed(0)
    X = torch.randn(5000, 10)
    Y = torch.randn(5000, 10) + 1.0
    pred = ot_barycentric_baseline(X, Y, reg=0.0, max_samples=500)
    assert pred.shape == (500, 10)


def test_ot_barycentric_differs_from_identity():
    """OT pred should produce a population closer to Y than identity."""
    torch.manual_seed(0)
    X = torch.randn(200, 3)
    Y = torch.randn(200, 3) + torch.tensor([2.0, -1.0, 0.5])
    pred = ot_barycentric_baseline(X, Y, reg=0.0)
    # pred mean closer to Y mean than X mean
    d_pred = (pred.mean(0) - Y.mean(0)).norm().item()
    d_x = (X.mean(0) - Y.mean(0)).norm().item()
    assert d_pred < d_x * 0.3, f"OT did not move toward target: {d_pred} vs {d_x}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
