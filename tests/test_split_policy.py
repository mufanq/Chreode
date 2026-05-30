import numpy as np

from cellworldmodel.training.split_policy import build_timepoint_splits, split_indices


def test_split_indices_are_deterministic_and_disjoint():
    a = split_indices(100, seed=7)
    b = split_indices(100, seed=7)
    assert np.array_equal(a.train, b.train)
    assert np.array_equal(a.val, b.val)
    assert np.array_equal(a.test, b.test)
    all_idx = np.concatenate([a.train, a.val, a.test])
    assert sorted(all_idx.tolist()) == list(range(100))
    assert len(set(a.train) & set(a.val)) == 0
    assert len(set(a.train) & set(a.test)) == 0
    assert len(set(a.val) & set(a.test)) == 0


def test_build_timepoint_splits_covers_each_population():
    coords = {0.0: np.zeros((10, 2), dtype=np.float32), 1.0: np.zeros((20, 2), dtype=np.float32)}
    splits = build_timepoint_splits(coords, seed=11)
    assert set(splits) == {0.0, 1.0}
    assert sum(len(x) for x in (splits[0.0].train, splits[0.0].val, splits[0.0].test)) == 10
    assert sum(len(x) for x in (splits[1.0].train, splits[1.0].val, splits[1.0].test)) == 20
