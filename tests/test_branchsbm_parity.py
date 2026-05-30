"""Numeric parity test: our BranchSBM-style metrics vs BranchSBM repo originals.

We import BranchSBM's own `mix_rbf_mmd2` and `wasserstein` directly from
3rdparty/BranchSBM/src/utils.py and compare outputs to ours on shared random
tensors. Any nontrivial mismatch is a bug in our implementation.

Run with:
    PYTHONPATH=src:3rdparty/BranchSBM pytest src/cellworldmodel/tests/test_branchsbm_parity.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Make BranchSBM's src package importable as `src.utils`.
BRANCHSBM_ROOT = Path(__file__).parent.parent.parent.parent / "3rdparty" / "BranchSBM"
sys.path.insert(0, str(BRANCHSBM_ROOT))

try:
    from src.utils import mix_rbf_mmd2 as branchsbm_mix_rbf_mmd2
    from src.utils import wasserstein as branchsbm_wasserstein
    BRANCHSBM_AVAILABLE = True
except ImportError:
    BRANCHSBM_AVAILABLE = False

from cellworldmodel.benchmark.common_metrics import (
    branchsbm_mmd2_biased_fixed_sigma,
    compute_branchsbm_style_metrics,
    exact_w_pq,
)


pytestmark = pytest.mark.skipif(
    not BRANCHSBM_AVAILABLE,
    reason="BranchSBM repo at 3rdparty/BranchSBM not importable",
)


DEFAULT_SIGMA = [0.01, 0.1, 1.0, 10.0, 100.0]


def test_biased_mmd_matches_branchsbm_small():
    """Biased MMD with fixed σ list should match BranchSBM's mix_rbf_mmd2(biased=True)."""
    torch.manual_seed(0)
    X = torch.randn(50, 10)
    Y = torch.randn(50, 10) + 0.5

    ours = branchsbm_mmd2_biased_fixed_sigma(X, Y, sigma_list=DEFAULT_SIGMA).item()
    theirs = branchsbm_mix_rbf_mmd2(X, Y, sigma_list=DEFAULT_SIGMA, biased=True).item()

    assert abs(ours - theirs) < 1e-5, f"MMD mismatch: ours={ours:.6g} vs theirs={theirs:.6g}"


def test_biased_mmd_matches_branchsbm_larger():
    """Repeat at larger n to rule out accumulation differences."""
    torch.manual_seed(1)
    X = torch.randn(500, 50)
    Y = torch.randn(500, 50) + 0.3

    ours = branchsbm_mmd2_biased_fixed_sigma(X, Y, sigma_list=DEFAULT_SIGMA).item()
    theirs = branchsbm_mix_rbf_mmd2(X, Y, sigma_list=DEFAULT_SIGMA, biased=True).item()

    # Allow 1e-4 relative (float32 accumulation over 500*500 terms)
    rel_err = abs(ours - theirs) / (abs(theirs) + 1e-12)
    assert rel_err < 1e-4, f"MMD rel_err={rel_err:.3e} — ours={ours:.6g} theirs={theirs:.6g}"


def test_biased_mmd_different_distributions():
    """Ensure both implementations produce the same nonzero MMD on distinct dists."""
    torch.manual_seed(2)
    X = torch.randn(200, 5)
    Y = torch.randn(200, 5) * 3.0 + 5.0  # scaled + shifted

    ours = branchsbm_mmd2_biased_fixed_sigma(X, Y, sigma_list=DEFAULT_SIGMA).item()
    theirs = branchsbm_mix_rbf_mmd2(X, Y, sigma_list=DEFAULT_SIGMA, biased=True).item()

    assert ours > 1e-3
    assert abs(ours - theirs) < 1e-5


def test_w1_matches_branchsbm():
    """Our exact_w_pq(p=1) vs BranchSBM's wasserstein(power=1)."""
    torch.manual_seed(3)
    X = torch.randn(100, 8)
    Y = torch.randn(100, 8) + 1.0

    ours_w1 = exact_w_pq(X, Y, p=1)
    theirs_w1 = branchsbm_wasserstein(X, Y, power=1)

    rel_err = abs(ours_w1 - theirs_w1) / (abs(theirs_w1) + 1e-12)
    assert rel_err < 1e-4, f"W1 mismatch: ours={ours_w1:.6g} theirs={theirs_w1:.6g}"


def test_w2_matches_branchsbm():
    """Our exact_w_pq(p=2) vs BranchSBM's wasserstein(power=2)."""
    torch.manual_seed(4)
    X = torch.randn(100, 8)
    Y = torch.randn(100, 8) + 1.0

    ours_w2 = exact_w_pq(X, Y, p=2)
    theirs_w2 = branchsbm_wasserstein(X, Y, power=2)

    rel_err = abs(ours_w2 - theirs_w2) / (abs(theirs_w2) + 1e-12)
    assert rel_err < 1e-4, f"W2 mismatch: ours={ours_w2:.6g} theirs={theirs_w2:.6g}"


def test_compute_distribution_distances_parity():
    """End-to-end: our compute_branchsbm_style_metrics top-2 should match
    BranchSBM's compute_distribution_distances (pred[:, :2], true[:, :2])."""
    from src.branch_flow_net_test import compute_distribution_distances

    torch.manual_seed(5)
    D = 50
    pred = torch.randn(300, D)
    true = torch.randn(300, D) + 0.5

    # BranchSBM calls `compute_distribution_distances(pred[:, :2], true[:, :2], pred_full, true_full)`
    theirs = compute_distribution_distances(
        pred[:, :2], true[:, :2], pred_full=pred, true_full=true
    )

    # Our 5-trial subsample avg should be close (not exact — different random subsampling)
    ours = compute_branchsbm_style_metrics(
        pred, true, n_trials=1, n_top_pcs=2, seed=42,
    )

    # W1 top-2 scale should match
    rel_err_w1 = abs(ours["branchsbm_w1_top2_mean"] - theirs["W1"]) / (theirs["W1"] + 1e-12)
    rel_err_w2 = abs(ours["branchsbm_w2_top2_mean"] - theirs["W2"]) / (theirs["W2"] + 1e-12)
    rel_err_mmd = abs(ours["branchsbm_mmd_full_mean"] - theirs["MMD"]) / (theirs["MMD"] + 1e-12)

    # Allow 10% rel_err because single subsample vs their full-sample eval
    # (they have n_pred=n_true=300 so no subsample; ours subsamples to n_min=300 = same)
    # This should actually match tightly
    assert rel_err_w1 < 0.05, f"W1 top-2 rel_err={rel_err_w1:.3e}"
    assert rel_err_w2 < 0.05, f"W2 top-2 rel_err={rel_err_w2:.3e}"
    assert rel_err_mmd < 0.05, f"MMD rel_err={rel_err_mmd:.3e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
