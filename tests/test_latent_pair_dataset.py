"""Unit tests for LatentPairDataset."""

import numpy as np
import pytest
import torch

from cellworldmodel.data.latent_pair_dataset import LatentPairDataset


class TestLatentPairDataset:
    def _make_dataset(self, n_pairs=100, n_cells=50, dim=8):
        rng = np.random.default_rng(42)
        z_state = rng.standard_normal((n_cells, dim)).astype(np.float32)
        src_ids = rng.integers(0, n_cells, size=n_pairs).astype(np.int64)
        tgt_ids = rng.integers(0, n_cells, size=n_pairs).astype(np.int64)
        src_times = rng.uniform(0, 10, size=n_pairs).astype(np.float32)
        tgt_times = src_times + rng.uniform(0.5, 2, size=n_pairs).astype(np.float32)
        probs = rng.uniform(0.01, 1.0, size=n_pairs).astype(np.float32)
        return LatentPairDataset(src_ids, tgt_ids, src_times, tgt_times, probs, z_state)

    def test_length(self):
        ds = self._make_dataset(n_pairs=50)
        assert len(ds) == 50

    def test_getitem_shapes(self):
        ds = self._make_dataset(n_pairs=10, dim=16)
        item = ds[0]
        assert item["z_t"].shape == (16,)
        assert item["z_t1"].shape == (16,)
        assert item["time_t"].shape == ()
        assert item["time_t1"].shape == ()
        assert item["probability"].shape == ()

    def test_getitem_correct_values(self):
        z_state = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        ds = LatentPairDataset(
            source_int_ids=np.array([0, 2]),
            target_int_ids=np.array([1, 0]),
            source_times=np.array([1.0, 2.0]),
            target_times=np.array([2.0, 3.0]),
            probabilities=np.array([0.5, 0.8]),
            z_state=z_state,
        )
        item0 = ds[0]
        np.testing.assert_array_equal(item0["z_t"].numpy(), [1, 2, 3])
        np.testing.assert_array_equal(item0["z_t1"].numpy(), [4, 5, 6])
        assert float(item0["time_t"]) == 1.0
        assert float(item0["probability"]) == pytest.approx(0.5)

        item1 = ds[1]
        np.testing.assert_array_equal(item1["z_t"].numpy(), [7, 8, 9])
        np.testing.assert_array_equal(item1["z_t1"].numpy(), [1, 2, 3])

    def test_rejects_unmapped_ids(self):
        z = np.zeros((5, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="unmapped"):
            LatentPairDataset(
                source_int_ids=np.array([0, -1]),
                target_int_ids=np.array([1, 2]),
                source_times=np.array([0.0, 1.0]),
                target_times=np.array([1.0, 2.0]),
                probabilities=np.array([0.5, 0.5]),
                z_state=z,
            )

    def test_rejects_out_of_bounds_ids(self):
        z = np.zeros((5, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="too small"):
            LatentPairDataset(
                source_int_ids=np.array([0, 10]),
                target_int_ids=np.array([1, 2]),
                source_times=np.array([0.0, 1.0]),
                target_times=np.array([1.0, 2.0]),
                probabilities=np.array([0.5, 0.5]),
                z_state=z,
            )

    def test_sampling_weights(self):
        ds = self._make_dataset(n_pairs=20)
        weights = ds.get_sampling_weights()
        assert weights.shape == (20,)
        assert torch.all(weights >= 0)
        assert torch.all(torch.isfinite(weights))

    def test_works_with_mmap(self, tmp_path):
        """Verify dataset works with memmapped z_state."""
        z = np.random.default_rng(42).standard_normal((10, 4)).astype(np.float32)
        path = tmp_path / "z_state.npy"
        np.save(path, z)
        z_mmap = np.load(path, mmap_mode="r")

        ds = LatentPairDataset(
            source_int_ids=np.array([0, 5]),
            target_int_ids=np.array([3, 8]),
            source_times=np.array([0.0, 1.0]),
            target_times=np.array([1.0, 2.0]),
            probabilities=np.array([0.5, 0.5]),
            z_state=z_mmap,
        )
        item = ds[0]
        np.testing.assert_array_almost_equal(item["z_t"].numpy(), z[0])
        np.testing.assert_array_almost_equal(item["z_t1"].numpy(), z[3])


    def test_empty_dataset(self):
        """Empty splits should be allowed."""
        z = np.zeros((5, 3), dtype=np.float32)
        ds = LatentPairDataset(
            source_int_ids=np.array([], dtype=np.int64),
            target_int_ids=np.array([], dtype=np.int64),
            source_times=np.array([], dtype=np.float32),
            target_times=np.array([], dtype=np.float32),
            probabilities=np.array([], dtype=np.float32),
            z_state=z,
        )
        assert len(ds) == 0

    def test_length_mismatch_raises(self):
        z = np.zeros((5, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="length"):
            LatentPairDataset(
                source_int_ids=np.array([0, 1]),
                target_int_ids=np.array([1]),  # wrong length
                source_times=np.array([0.0, 1.0]),
                target_times=np.array([1.0, 2.0]),
                probabilities=np.array([0.5, 0.5]),
                z_state=z,
            )

    def test_copy_on_read_no_alias(self):
        """Mutating returned tensor must not modify z_state."""
        z = np.array([[1, 2], [3, 4]], dtype=np.float32)
        ds = LatentPairDataset(
            source_int_ids=np.array([0]),
            target_int_ids=np.array([1]),
            source_times=np.array([0.0]),
            target_times=np.array([1.0]),
            probabilities=np.array([0.5]),
            z_state=z,
        )
        item = ds[0]
        item["z_t"][0] = 999.0
        assert z[0, 0] == 1.0  # original unchanged

    def test_exact_sampling_weights(self):
        """Verify per-target normalization with known values."""
        z = np.zeros((5, 2), dtype=np.float32)
        # Two pairs to same target (id=1), different probs
        ds = LatentPairDataset(
            source_int_ids=np.array([0, 2]),
            target_int_ids=np.array([1, 1]),
            source_times=np.array([0.0, 0.0]),
            target_times=np.array([1.0, 1.0]),
            probabilities=np.array([0.3, 0.7]),
            z_state=z,
        )
        w = ds.get_sampling_weights()
        # Both target id=1, total=1.0, so weights = prob/total = prob
        np.testing.assert_allclose(w.numpy(), [0.3, 0.7], atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
