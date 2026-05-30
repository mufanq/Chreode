"""Unit tests for BranchSBM adapter. Uses actual data files."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cellworldmodel.benchmark.branchsbm_adapter import (
    ClonidineAdapter,
    MouseHematopoiesisAdapter,
    TrametinibAdapter,
    VeresAdapter,
    get_adapter,
)


@pytest.fixture
def mouse_adapter():
    return MouseHematopoiesisAdapter()


def test_mouse_load(mouse_adapter):
    assert mouse_adapter.dim == 2
    # From earlier investigation: 1429 source cells at t=0, 5788 at t=2
    assert mouse_adapter.coords_by_t[0.0].shape == (1429, 2)
    assert mouse_adapter.coords_by_t[2.0].shape == (5788, 2)


def test_mouse_train_test_split(mouse_adapter):
    # 80/20 split of 1429 ≈ 1143/286
    assert len(mouse_adapter.train_src_idx) == int(1429 * 0.8)
    assert len(mouse_adapter.test_src_idx) == 1429 - int(1429 * 0.8)
    # No overlap
    overlap = set(mouse_adapter.train_src_idx) & set(mouse_adapter.test_src_idx)
    assert len(overlap) == 0


def test_mouse_get_transition_train(mouse_adapter):
    batch = mouse_adapter.get_transition(split="train")
    assert batch.source.dtype == torch.float32
    assert batch.source.shape[1] == 2
    assert batch.target.shape[1] == 2
    assert batch.delta == 2.0  # t=0 → t=2
    assert len(batch.source) == int(1429 * 0.8)


def test_mouse_sample_batches(mouse_adapter):
    src = mouse_adapter.sample_source_batch(batch_size=64, split="train")
    tgt = mouse_adapter.sample_target_batch(batch_size=128)
    assert src.shape == (64, 2)
    assert tgt.shape == (128, 2)


def test_mouse_test_split_disjoint():
    """Train and test source samples should be disjoint."""
    adapter = MouseHematopoiesisAdapter(seed=0)
    train_batch = adapter.get_transition(split="train")
    test_batch = adapter.get_transition(split="test")
    # Check no coordinates are shared (within numerical precision)
    train_set = set(map(tuple, train_batch.source.numpy().round(5)))
    test_set = set(map(tuple, test_batch.source.numpy().round(5)))
    # Some coords might coincide by chance with low precision, but most shouldn't
    overlap = train_set & test_set
    assert len(overlap) < 5, f"train/test source overlap too large: {len(overlap)}"


@pytest.mark.slow
def test_clonidine_50d_load():
    """Clonidine loading (larger file, runs slower)."""
    adapter = ClonidineAdapter(pcs=50)
    assert adapter.dim == 50
    batch = adapter.get_transition(split="train")
    assert batch.source.shape[1] == 50
    assert batch.target.shape[1] == 50
    assert batch.delta == 1.0
    # Check cluster labels available
    labels = adapter.get_target_cluster_labels()
    assert labels.shape[0] == batch.target.shape[0]
    assert labels.min() >= 0


@pytest.mark.slow
def test_trametinib_load():
    adapter = TrametinibAdapter(pcs=50)
    assert adapter.dim == 50
    batch = adapter.get_transition(split="train")
    assert batch.source.shape[1] == 50
    # Trametinib has 3 perturbed clusters
    labels = adapter.get_target_cluster_labels()
    n_clusters = len(set(labels.tolist()))
    # From earlier investigation: 8 clusters but 3 major + some rare
    assert n_clusters >= 3


def test_factory():
    a = get_adapter("mouse")
    assert isinstance(a, MouseHematopoiesisAdapter)
    v = get_adapter("veres")
    assert isinstance(v, VeresAdapter)


@pytest.mark.slow
def test_veres_load():
    """Veres 30D pancreatic β-cell differentiation loading."""
    adapter = VeresAdapter()
    assert adapter.dim == 30
    # 8 timepoints (0..7)
    assert adapter.timepoints == [0, 1, 2, 3, 4, 5, 6, 7]
    batch = adapter.get_transition(split="train")
    assert batch.source.shape[1] == 30
    assert batch.target.shape[1] == 30
    assert batch.delta == 7.0  # t=0 → t=7


@pytest.mark.slow
def test_veres_train_test_split():
    adapter = VeresAdapter(seed=0)
    batch_train = adapter.get_transition(split="train")
    batch_test = adapter.get_transition(split="test")
    # Train and test source should be disjoint subsets of t=0
    n_t0 = adapter.coords_by_t[0].shape[0]
    assert len(batch_train.source) + len(batch_test.source) == n_t0


@pytest.mark.slow
def test_veres_intermediate():
    adapter = VeresAdapter()
    # Intermediate t=4 should exist
    t4 = adapter.get_intermediate(4)
    assert t4.shape[1] == 30
    assert t4.shape[0] > 0


@pytest.mark.slow
def test_veres_cluster_labels():
    adapter = VeresAdapter()
    labels = adapter.get_target_cluster_labels(n_clusters=11)
    tgt_n = adapter.coords_by_t[7].shape[0]
    assert labels.shape == (tgt_n,)
    assert len(set(labels.tolist())) == 11


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
