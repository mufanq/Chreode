import numpy as np
import pandas as pd

from cellworldmodel.foundation.transition_index import (
    FoundationTransitionIdSampler,
    build_transition_index,
    ordered_pairs,
    write_transition_index,
)


def make_catalog(tmp_path):
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    rows = []
    gid = 0
    for leaf in ["A", "B"]:
        for t in [1.0, 2.0, 3.0]:
            for _ in range(3):
                rows.append({
                    "global_cell_id": gid,
                    "foundation_split": "train",
                    "leaf_dataset": leaf,
                    "timepoint": t,
                })
                gid += 1
    pd.DataFrame(rows).to_parquet(catalog / "cell_index.parquet", index=False)
    return catalog


def test_ordered_pairs():
    assert ordered_pairs([1, 2, 3], "adjacent") == [(1.0, 2.0), (2.0, 3.0)]
    assert ordered_pairs([1, 2, 3], "all_ordered") == [(1.0, 2.0), (1.0, 3.0), (2.0, 3.0)]


def test_build_transition_index_and_sampler(tmp_path):
    catalog = make_catalog(tmp_path)
    index = build_transition_index(catalog)
    assert len(index.transitions) == 6
    assert set(index.transitions["leaf_dataset"]) == {"A", "B"}
    out = tmp_path / "transition"
    manifest = write_transition_index(index, out)
    assert manifest["n_transitions"] == 6
    sampler = FoundationTransitionIdSampler(catalog, index.transitions)
    batch = sampler.sample(5, np.random.default_rng(0))
    assert batch.source_ids.shape == (5,)
    assert batch.target_ids.shape == (5,)
    assert batch.delta > 0
