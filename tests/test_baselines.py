"""Unit tests for Stage 1 additional baselines."""

import numpy as np
import pytest

from cellworldmodel.evaluation.baselines import (
    DeltaMeanBaseline,
    DeltaRidgeBaseline,
    ot_barycentric_predict,
)


class TestDeltaMeanBaseline:
    def test_identity_when_no_change(self):
        """If targets == sources, delta-mean should predict z_t (no change)."""
        rng = np.random.default_rng(42)
        N, D = 100, 8
        z = rng.standard_normal((N, D)).astype(np.float32)
        times_src = np.full(N, 1.0, dtype=np.float32)
        times_tgt = np.full(N, 2.0, dtype=np.float32)
        dk = np.array(["ds1"] * N)

        dm = DeltaMeanBaseline()
        dm.fit(z, z, times_src, times_tgt, dk)  # target == source → delta = 0
        pred = dm.predict(z, times_src, times_tgt, dk)
        np.testing.assert_allclose(pred, z, atol=1e-6)

    def test_constant_shift(self):
        """If all pairs have same delta, prediction should be z_t + delta."""
        rng = np.random.default_rng(42)
        N, D = 50, 4
        z_src = rng.standard_normal((N, D)).astype(np.float32)
        shift = np.array([1.0, -0.5, 0.2, 0.0], dtype=np.float32)
        z_tgt = z_src + shift
        times_src = np.full(N, 0.0, dtype=np.float32)
        times_tgt = np.full(N, 1.0, dtype=np.float32)
        dk = np.array(["ds1"] * N)

        dm = DeltaMeanBaseline()
        dm.fit(z_src, z_tgt, times_src, times_tgt, dk)

        # Test on new source cells
        z_test = rng.standard_normal((10, D)).astype(np.float32)
        pred = dm.predict(z_test, np.full(10, 0.0), np.full(10, 1.0), np.array(["ds1"] * 10))
        np.testing.assert_allclose(pred, z_test + shift, atol=1e-5)

    def test_per_transition_grouping(self):
        """Different transitions should have different delta means."""
        D = 4
        z_src = np.zeros((2, D), dtype=np.float32)
        z_tgt = np.array([[1, 0, 0, 0], [0, 2, 0, 0]], dtype=np.float32)
        times_src = np.array([0.0, 1.0], dtype=np.float32)
        times_tgt = np.array([1.0, 2.0], dtype=np.float32)
        dk = np.array(["ds1", "ds1"])

        dm = DeltaMeanBaseline()
        dm.fit(z_src, z_tgt, times_src, times_tgt, dk)

        # Query transition 0→1
        pred = dm.predict(np.zeros((1, D)), np.array([0.0]), np.array([1.0]), np.array(["ds1"]))
        np.testing.assert_allclose(pred, [[1, 0, 0, 0]], atol=1e-6)

        # Query transition 1→2
        pred = dm.predict(np.zeros((1, D)), np.array([1.0]), np.array([2.0]), np.array(["ds1"]))
        np.testing.assert_allclose(pred, [[0, 2, 0, 0]], atol=1e-6)

    def test_raises_before_fit(self):
        dm = DeltaMeanBaseline()
        with pytest.raises(RuntimeError):
            dm.predict(np.zeros((1, 4)), np.array([0.0]), np.array([1.0]), np.array(["ds1"]))

    def test_raises_on_empty_fit(self):
        dm = DeltaMeanBaseline()
        with pytest.raises(ValueError):
            dm.fit(np.zeros((0, 4)), np.zeros((0, 4)), np.array([]), np.array([]), np.array([]))

    def test_fallback_to_global(self):
        """Unseen transition should use global delta."""
        D = 4
        z_src = np.zeros((2, D), dtype=np.float32)
        z_tgt = np.array([[2, 0, 0, 0], [0, 2, 0, 0]], dtype=np.float32)
        times_src = np.array([0.0, 1.0], dtype=np.float32)
        times_tgt = np.array([1.0, 2.0], dtype=np.float32)
        dk = np.array(["ds1", "ds1"])

        dm = DeltaMeanBaseline()
        dm.fit(z_src, z_tgt, times_src, times_tgt, dk)

        # Query unseen transition 5→6
        pred = dm.predict(np.zeros((1, D)), np.array([5.0]), np.array([6.0]), np.array(["ds1"]))
        expected_global = np.array([[1, 1, 0, 0]], dtype=np.float32)  # mean of [2,0,0,0] and [0,2,0,0]
        np.testing.assert_allclose(pred, expected_global, atol=1e-6)


class TestDeltaRidgeBaseline:
    def test_linear_shift(self):
        """Ridge should learn a constant shift perfectly."""
        rng = np.random.default_rng(42)
        N, D = 200, 8
        z_src = rng.standard_normal((N, D)).astype(np.float32)
        shift = rng.standard_normal(D).astype(np.float32)
        z_tgt = z_src + shift
        dt = np.ones(N, dtype=np.float32)

        dr = DeltaRidgeBaseline(alpha=0.01)
        dr.fit(z_src, z_tgt, dt)

        z_test = rng.standard_normal((20, D)).astype(np.float32)
        pred = dr.predict(z_test, np.ones(20, dtype=np.float32))
        np.testing.assert_allclose(pred, z_test + shift, atol=0.1)

    def test_residual_form(self):
        """Output should always be z_t + something (residual form)."""
        rng = np.random.default_rng(42)
        N, D = 100, 4
        z_src = rng.standard_normal((N, D)).astype(np.float32)
        z_tgt = z_src + 0.5
        dt = np.ones(N, dtype=np.float32)

        dr = DeltaRidgeBaseline(alpha=1.0)
        dr.fit(z_src, z_tgt, dt)

        z_test = np.zeros((1, D), dtype=np.float32)
        pred = dr.predict(z_test, np.array([1.0]))
        # Prediction should be close to z_test + 0.5 = 0.5
        assert pred.shape == (1, D)
        assert np.all(np.isfinite(pred))

    def test_raises_before_fit(self):
        dr = DeltaRidgeBaseline()
        with pytest.raises(RuntimeError):
            dr.predict(np.zeros((1, 4)), np.array([1.0]))


class TestOTBarycentricOracle:
    def test_single_target(self):
        """With one OT target per source, prediction == that target."""
        D = 4
        z_all = np.array([
            [0, 0, 0, 0],  # cell 0 (source)
            [1, 2, 3, 4],  # cell 1 (target)
        ], dtype=np.float32)

        source_int_ids = np.array([0], dtype=np.int64)
        target_int_ids = np.array([1], dtype=np.int64)
        ot_probs = np.array([1.0], dtype=np.float32)
        pair_indices = np.array([0], dtype=np.int64)

        pred = ot_barycentric_predict(source_int_ids, target_int_ids, ot_probs,
                                       z_all, pair_indices)
        np.testing.assert_allclose(pred, [[1, 2, 3, 4]], atol=1e-6)

    def test_weighted_average(self):
        """With multiple targets, should be weighted average."""
        D = 2
        z_all = np.array([
            [0, 0],  # cell 0 (source)
            [2, 0],  # cell 1 (target, prob 0.75)
            [0, 4],  # cell 2 (target, prob 0.25)
        ], dtype=np.float32)

        # Two pairs from same source
        source_int_ids = np.array([0, 0], dtype=np.int64)
        target_int_ids = np.array([1, 2], dtype=np.int64)
        ot_probs = np.array([0.75, 0.25], dtype=np.float32)
        pair_indices = np.array([0, 1], dtype=np.int64)

        pred = ot_barycentric_predict(source_int_ids, target_int_ids, ot_probs,
                                       z_all, pair_indices)
        # Both pairs from same source → both get same barycenter
        expected = np.array([0.75 * 2 + 0.25 * 0, 0.75 * 0 + 0.25 * 4])
        np.testing.assert_allclose(pred[0], expected, atol=1e-6)
        np.testing.assert_allclose(pred[1], expected, atol=1e-6)

    def test_fallback_mean(self):
        """Source without OT neighbors should use fallback."""
        D = 2
        z_all = np.array([
            [0, 0],  # cell 0 (source, no OT neighbors in test pairs)
            [1, 1],  # cell 1 (not connected)
        ], dtype=np.float32)

        # No valid pairs for source 0 (target is -1 = unmapped)
        source_int_ids = np.array([0], dtype=np.int64)
        target_int_ids = np.array([-1], dtype=np.int64)
        ot_probs = np.array([0.5], dtype=np.float32)
        pair_indices = np.array([0], dtype=np.int64)

        fallback = np.array([5, 5], dtype=np.float32)
        pred = ot_barycentric_predict(source_int_ids, target_int_ids, ot_probs,
                                       z_all, pair_indices, fallback_mean=fallback)
        np.testing.assert_allclose(pred, [[5, 5]], atol=1e-6)

    def test_raises_without_fallback(self):
        """Should raise if no OT neighbors and no fallback."""
        D = 2
        z_all = np.array([[0, 0], [1, 1]], dtype=np.float32)
        source_int_ids = np.array([0], dtype=np.int64)
        target_int_ids = np.array([-1], dtype=np.int64)
        ot_probs = np.array([0.5], dtype=np.float32)
        pair_indices = np.array([0], dtype=np.int64)

        with pytest.raises(ValueError):
            ot_barycentric_predict(source_int_ids, target_int_ids, ot_probs,
                                    z_all, pair_indices, fallback_mean=None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
