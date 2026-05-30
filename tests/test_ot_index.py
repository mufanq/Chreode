"""Tests for PackedOT CSR index builder."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cellworldmodel.data.ot_index import (
    PackedOT,
    build_packed_ot_for_transition,
    load_packed_ot,
    save_packed_ot,
    strip_moscot_suffix,
    transition_slug,
)


# ---------------------------------------------------------------------------
# strip_moscot_suffix
# ---------------------------------------------------------------------------


class TestStripMoscotSuffix:
    def test_simple(self):
        assert strip_moscot_suffix("AAACAGCCAACAGCCT-1_0") == "AAACAGCCAACAGCCT-1"

    def test_double_digit(self):
        assert strip_moscot_suffix("CELL-1_12") == "CELL-1"

    def test_no_suffix(self):
        assert strip_moscot_suffix("AAACAGCCAACAGCCT-1") == "AAACAGCCAACAGCCT-1"

    def test_gse140802_style(self):
        """GSE140802 has pipe-separated IDs with underscores in the prefix."""
        assert strip_moscot_suffix("d2_3|AGCACGTA-GGCGGTTT_0") == "d2_3|AGCACGTA-GGCGGTTT"

    def test_preserves_internal_underscore(self):
        assert strip_moscot_suffix("LSK_d2_1|AAACGTGA-AAAGCCTA_0") == "LSK_d2_1|AAACGTGA-AAAGCCTA"

    def test_no_trailing_digits(self):
        """Suffix must end with digits only."""
        assert strip_moscot_suffix("CELL_abc") == "CELL_abc"


# ---------------------------------------------------------------------------
# transition_slug
# ---------------------------------------------------------------------------


class TestTransitionSlug:
    def test_simple(self):
        assert transition_slug("GSE275562", 14.5, 15.5) == "GSE275562__t14p50__to__t15p50"

    def test_nested(self):
        slug = transition_slug("GSE115943::C1", 0.0, 0.5)
        assert slug == "GSE115943__C1__t0p00__to__t0p50"


# ---------------------------------------------------------------------------
# Build PackedOT from synthetic CSV
# ---------------------------------------------------------------------------


def _make_test_csv(tmp_path: Path, pairs: list, direction: str = "source_to_target") -> str:
    """Create a synthetic OT CSV."""
    rows = []
    for src, tgt, prob in pairs:
        rows.append({
            "source_time": 1.0,
            "target_time": 2.0,
            "source_cell": f"{src}_0",  # add MOSCOT suffix
            "target_cell": f"{tgt}_1",
            "probability": prob,
            "direction": direction,
        })
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "ot.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def _make_cell_id_map(cells_t1: list, cells_t2: list, dataset_key: str = "TEST"):
    """Create cell_id → int_id mapping."""
    mapping = {}
    int_id = 0
    for c in cells_t1:
        mapping[(dataset_key, 1.0, c)] = int_id
        int_id += 1
    for c in cells_t2:
        mapping[(dataset_key, 2.0, c)] = int_id
        int_id += 1
    return mapping


class TestBuildPackedOT:
    def test_basic(self, tmp_path):
        """Basic build with 2 source cells, 3 target cells."""
        csv = _make_test_csv(tmp_path, [
            ("A", "X", 0.6), ("A", "Y", 0.3), ("A", "Z", 0.1),
            ("B", "X", 0.2), ("B", "Y", 0.8),
        ])
        cell_map = _make_cell_id_map(["A", "B"], ["X", "Y", "Z"])
        all_ids = set(cell_map.values())

        packed = build_packed_ot_for_transition(
            csv_path=csv, dataset_key="TEST",
            source_time=1.0, target_time=2.0,
            cell_id_to_int=cell_map, train_int_ids=all_ids,
        )

        assert packed is not None
        assert packed.n_source == 2
        assert packed.nnz == 5  # 3 + 2
        # Probabilities should be normalized per source
        for i in range(packed.n_source):
            lo, hi = packed.row_ptr[i], packed.row_ptr[i + 1]
            probs = packed.prob[lo:hi]
            np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-6)

    def test_reverse_only_returns_none(self, tmp_path):
        """target_to_source only → should return None."""
        csv = _make_test_csv(tmp_path, [
            ("A", "X", 0.5), ("A", "Y", 0.5),
        ], direction="target_to_source")
        cell_map = _make_cell_id_map(["A"], ["X", "Y"])

        packed = build_packed_ot_for_transition(
            csv_path=csv, dataset_key="TEST",
            source_time=1.0, target_time=2.0,
            cell_id_to_int=cell_map, train_int_ids=set(cell_map.values()),
        )
        assert packed is None

    def test_train_filter(self, tmp_path):
        """Only pairs where both source AND target are in train should survive."""
        csv = _make_test_csv(tmp_path, [
            ("A", "X", 0.5), ("A", "Y", 0.5),
            ("B", "X", 0.3), ("B", "Y", 0.7),
        ])
        cell_map = _make_cell_id_map(["A", "B"], ["X", "Y"])
        # Only A and X are in train (not B, not Y)
        train_ids = {cell_map[("TEST", 1.0, "A")], cell_map[("TEST", 2.0, "X")]}

        packed = build_packed_ot_for_transition(
            csv_path=csv, dataset_key="TEST",
            source_time=1.0, target_time=2.0,
            cell_id_to_int=cell_map, train_int_ids=train_ids,
        )

        assert packed is not None
        assert packed.n_source == 1  # only A
        assert packed.nnz == 1  # only A→X
        np.testing.assert_allclose(packed.prob[0], 1.0, atol=1e-6)

    def test_truncation(self, tmp_path):
        """With k_cap=2, each source should have at most 2 targets."""
        csv = _make_test_csv(tmp_path, [
            ("A", "X", 0.5), ("A", "Y", 0.3), ("A", "Z", 0.15), ("A", "W", 0.05),
        ])
        cell_map = _make_cell_id_map(["A"], ["X", "Y", "Z", "W"])
        all_ids = set(cell_map.values())

        packed = build_packed_ot_for_transition(
            csv_path=csv, dataset_key="TEST",
            source_time=1.0, target_time=2.0,
            cell_id_to_int=cell_map, train_int_ids=all_ids,
            k_cap=2,
        )

        assert packed is not None
        assert packed.nnz == 2  # truncated to top 2
        assert packed.truncated_degree[0] == 2
        assert packed.original_degree[0] == 4
        np.testing.assert_allclose(packed.prob.sum(), 1.0, atol=1e-6)

    def test_empty_after_train_filter(self, tmp_path):
        """If no pairs survive train filter → return None."""
        csv = _make_test_csv(tmp_path, [("A", "X", 1.0)])
        cell_map = _make_cell_id_map(["A"], ["X"])
        # Train is empty
        packed = build_packed_ot_for_transition(
            csv_path=csv, dataset_key="TEST",
            source_time=1.0, target_time=2.0,
            cell_id_to_int=cell_map, train_int_ids=set(),
        )
        assert packed is None


# ---------------------------------------------------------------------------
# Save / Load roundtrip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        """Save then load should produce identical PackedOT."""
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        csv = _make_test_csv(csv_dir, [
            ("A", "X", 0.6), ("A", "Y", 0.4),
            ("B", "X", 0.3), ("B", "Y", 0.7),
        ])
        cell_map = _make_cell_id_map(["A", "B"], ["X", "Y"])
        all_ids = set(cell_map.values())

        original = build_packed_ot_for_transition(
            csv_path=csv, dataset_key="TEST",
            source_time=1.0, target_time=2.0,
            cell_id_to_int=cell_map, train_int_ids=all_ids,
        )
        assert original is not None

        npz_path = tmp_path / "test.npz"
        save_packed_ot(original, npz_path)
        loaded = load_packed_ot(npz_path)

        np.testing.assert_array_equal(loaded.source_int_ids, original.source_int_ids)
        np.testing.assert_array_equal(loaded.target_int_ids, original.target_int_ids)
        np.testing.assert_array_equal(loaded.row_ptr, original.row_ptr)
        np.testing.assert_array_equal(loaded.col_idx, original.col_idx)
        np.testing.assert_allclose(loaded.prob, original.prob, atol=1e-7)
        assert loaded.metadata["dataset_key"] == "TEST"
        assert loaded.metadata["mass_threshold"] == 0.99
