"""Stage 1 additional baselines: Delta-mean, Delta-Ridge, OT Barycentric Oracle.

These baselines operate in PCA latent space (128-dim) and produce predictions
that can be decoded back to gene space for evaluation.

All three use the same train/test pair structure as SimplePairPredictor:
- Train pairs: source ∈ train, target ∈ train
- Test pairs:  source ∈ train, target ∈ test

Usage:
    # Delta-mean
    dm = DeltaMeanBaseline()
    dm.fit(z_sources_train, z_targets_train, src_times_train, tgt_times_train, dataset_keys_train)
    pred_z = dm.predict(z_sources_test, src_times_test, tgt_times_test, dataset_keys_test)

    # Delta-Ridge
    dr = DeltaRidgeBaseline(alpha=1.0)
    dr.fit(z_sources_train, z_targets_train, delta_times_train)
    pred_z = dr.predict(z_sources_test, delta_times_test)

    # OT Barycentric Oracle (uses test OT coupling — oracle, not predictive)
    pred_z = ot_barycentric_predict(source_int_ids, target_int_ids, ot_probs, z_targets, fallback_mean)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _transition_key(dataset_key: str, src_time: float, tgt_time: float) -> str:
    """Canonical key for a (dataset, source_time, target_time) transition."""
    return f"{dataset_key}|{src_time:.6f}|{tgt_time:.6f}"


# ═══════════════════════════════════════════════════════════════════════
# Delta-mean baseline
# ═══════════════════════════════════════════════════════════════════════


class DeltaMeanBaseline:
    """Predict z_{t+Δ} = z_t + mean_delta(dataset, t_src, t_tgt).

    For each (dataset, source_time, target_time) transition group in train,
    computes the average displacement vector. At test time, adds the
    matching mean delta to each source cell.

    Falls back to global mean delta if a transition is unseen.
    """

    def __init__(self):
        self.delta_means: Dict[str, np.ndarray] = {}
        self.global_delta: Optional[np.ndarray] = None
        self._fitted = False

    def fit(
        self,
        z_sources: np.ndarray,
        z_targets: np.ndarray,
        src_times: np.ndarray,
        tgt_times: np.ndarray,
        dataset_keys: np.ndarray,
    ) -> "DeltaMeanBaseline":
        """Fit on train pairs.

        Args:
            z_sources: (N, D) source cell PCA latents
            z_targets: (N, D) target cell PCA latents
            src_times: (N,) source timepoints
            tgt_times: (N,) target timepoints
            dataset_keys: (N,) dataset key strings
        """
        if len(z_sources) == 0:
            raise ValueError("DeltaMeanBaseline.fit() called with empty training data")
        deltas = z_targets - z_sources  # (N, D)

        # Group by transition
        groups: Dict[str, list] = defaultdict(list)
        for i in range(len(deltas)):
            key = _transition_key(str(dataset_keys[i]), float(src_times[i]), float(tgt_times[i]))
            groups[key].append(i)

        self.delta_means = {}
        for key, indices in groups.items():
            self.delta_means[key] = deltas[indices].mean(axis=0)

        self.global_delta = deltas.mean(axis=0)
        self._fitted = True
        logger.info("DeltaMeanBaseline: fitted %d transition groups, %d train pairs",
                     len(self.delta_means), len(deltas))
        return self

    def predict(
        self,
        z_sources: np.ndarray,
        src_times: np.ndarray,
        tgt_times: np.ndarray,
        dataset_keys: np.ndarray,
    ) -> np.ndarray:
        """Predict target latents.

        Args:
            z_sources: (M, D) source cell PCA latents
            src_times: (M,) source timepoints
            tgt_times: (M,) target timepoints
            dataset_keys: (M,) dataset key strings

        Returns:
            pred_z: (M, D) predicted target latents
        """
        if not self._fitted:
            raise RuntimeError("DeltaMeanBaseline.fit() must be called before predict()")
        M, D = z_sources.shape
        pred = np.empty_like(z_sources)
        n_fallback = 0

        for i in range(M):
            key = _transition_key(str(dataset_keys[i]), float(src_times[i]), float(tgt_times[i]))
            delta = self.delta_means.get(key)
            if delta is None:
                delta = self.global_delta
                n_fallback += 1
            pred[i] = z_sources[i] + delta

        if n_fallback > 0:
            logger.warning("DeltaMeanBaseline: %d/%d test pairs used global fallback", n_fallback, M)
        return pred


# ═══════════════════════════════════════════════════════════════════════
# Delta-Ridge baseline
# ═══════════════════════════════════════════════════════════════════════


class DeltaRidgeBaseline:
    """Predict z_{t+Δ} = z_t + Ridge([z_t, Δ]).

    Fits a Ridge regression from [z_source, delta_time] → (z_target - z_source).
    This is a linear source-conditioned predictor in residual form.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.ridge = None

    def fit(
        self,
        z_sources: np.ndarray,
        z_targets: np.ndarray,
        delta_times: np.ndarray,
    ) -> "DeltaRidgeBaseline":
        """Fit on train pairs.

        Args:
            z_sources: (N, D) source cell PCA latents
            z_targets: (N, D) target cell PCA latents
            delta_times: (N,) time differences (t_target - t_source)
        """
        from sklearn.linear_model import Ridge

        X = np.column_stack([z_sources, delta_times.reshape(-1, 1)])  # (N, D+1)
        Y = z_targets - z_sources  # (N, D)

        self.ridge = Ridge(alpha=self.alpha)
        self.ridge.fit(X, Y)
        logger.info("DeltaRidgeBaseline: fitted on %d train pairs (alpha=%.2f)", len(X), self.alpha)
        return self

    def predict(
        self,
        z_sources: np.ndarray,
        delta_times: np.ndarray,
    ) -> np.ndarray:
        """Predict target latents.

        Args:
            z_sources: (M, D) source cell PCA latents
            delta_times: (M,) time differences

        Returns:
            pred_z: (M, D) predicted target latents
        """
        if self.ridge is None:
            raise RuntimeError("DeltaRidgeBaseline.fit() must be called before predict()")
        X = np.column_stack([z_sources, delta_times.reshape(-1, 1)])  # (M, D+1)
        delta_pred = self.ridge.predict(X)  # (M, D)
        return z_sources + delta_pred


# ═══════════════════════════════════════════════════════════════════════
# OT Barycentric Oracle
# ═══════════════════════════════════════════════════════════════════════


def ot_barycentric_predict(
    source_int_ids: np.ndarray,
    target_int_ids: np.ndarray,
    ot_probs: np.ndarray,
    z_all: np.ndarray,
    pair_indices: np.ndarray,
    fallback_mean: Optional[np.ndarray] = None,
) -> np.ndarray:
    """OT barycentric oracle: weighted average of OT targets per source.

    This is an ORACLE baseline — it uses test-time OT coupling (sees future).
    Must be clearly marked as oracle/reference in results tables.

    For each test pair (source_i, target_j), we find all OT targets of source_i
    among the test pairs, compute the weighted mean of their latents.

    Args:
        source_int_ids: (N_total,) Phase 0 int_id for source cell of each pair
        target_int_ids: (N_total,) Phase 0 int_id for target cell of each pair
        ot_probs: (N_total,) OT probabilities for each pair
        z_all: (N_cells, D) PCA latents for ALL cells (indexed by int_id)
        pair_indices: (M,) indices into source/target/prob arrays for test pairs
        fallback_mean: (D,) fallback if source has no OT neighbors

    Returns:
        pred_z: (M, D) predicted target latents (one per test pair)
    """
    D = z_all.shape[1]
    M = len(pair_indices)
    pred = np.zeros((M, D), dtype=np.float32)

    # Build per-source OT row from test pairs
    # Group: source_int_id → [(target_int_id, prob), ...]
    source_to_targets: Dict[int, list] = defaultdict(list)
    for idx in pair_indices:
        src = int(source_int_ids[idx])
        tgt = int(target_int_ids[idx])
        prob = float(ot_probs[idx])
        if src >= 0 and tgt >= 0 and prob > 0:
            source_to_targets[src].append((tgt, prob))

    # Pre-compute barycenter for each unique source
    source_barycenters: Dict[int, np.ndarray] = {}
    for src, tgt_probs in source_to_targets.items():
        tgt_ids = np.array([t for t, _ in tgt_probs], dtype=np.int64)
        probs = np.array([p for _, p in tgt_probs], dtype=np.float64)
        probs /= probs.sum()  # re-normalize
        z_tgts = z_all[tgt_ids]  # (k, D)
        source_barycenters[src] = (probs[:, None] * z_tgts).sum(axis=0).astype(np.float32)

    # Fill predictions
    n_fallback = 0
    for i, idx in enumerate(pair_indices):
        src = int(source_int_ids[idx])
        bary = source_barycenters.get(src)
        if bary is not None:
            pred[i] = bary
        elif fallback_mean is not None:
            pred[i] = fallback_mean
            n_fallback += 1
        else:
            raise ValueError(
                f"OT Barycentric Oracle: source {src} has no OT neighbors in test pairs "
                f"and no fallback_mean provided. This should not happen for valid test pairs."
            )

    n_sources = len(set(int(source_int_ids[idx]) for idx in pair_indices))
    n_with_ot = len(source_barycenters)
    logger.info("OT Barycentric Oracle: %d test pairs, %d unique sources, %d with OT neighbors, "
                "%d fallback", M, n_sources, n_with_ot, n_fallback)
    return pred


# ═══════════════════════════════════════════════════════════════════════
# Helper: extract latents + metadata for baseline fitting
# ═══════════════════════════════════════════════════════════════════════


def extract_pair_latents(
    pairs_df,
    pair_indices: np.ndarray,
    source_int_ids: np.ndarray,
    target_int_ids: np.ndarray,
    latent_z_state: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract latent vectors and metadata for a set of pairs.

    Args:
        pairs_df: DataFrame with source_time, target_time, dataset_id columns
        pair_indices: (M,) indices into pairs_df
        source_int_ids: (N_total,) int_ids for source cells
        target_int_ids: (N_total,) int_ids for target cells
        latent_z_state: (N_cells, D) cached PCA latents

    Returns:
        z_sources: (M, D) source latents
        z_targets: (M, D) target latents
        src_times: (M,) source timepoints
        tgt_times: (M,) target timepoints
        dataset_ids: (M,) legacy dataset_id strings (not canonical Phase 0 dataset_key)
    """
    src_ids = source_int_ids[pair_indices]
    tgt_ids = target_int_ids[pair_indices]

    # Fail hard on unmapped IDs — numpy treats -1 as last row, which is silently wrong.
    # split_pairs() should have already removed unmapped pairs.
    valid = (src_ids >= 0) & (tgt_ids >= 0)
    if not valid.all():
        n_bad = int((~valid).sum())
        raise ValueError(
            f"extract_pair_latents: {n_bad}/{len(pair_indices)} pairs have unmapped "
            f"endpoints (int_id == -1). This should not happen after split_pairs() "
            f"filtering. Check Phase 0 cell_index coverage."
        )

    z_sources = latent_z_state[src_ids].astype(np.float32)
    z_targets = latent_z_state[tgt_ids].astype(np.float32)

    sub_df = pairs_df.iloc[pair_indices]
    src_times = sub_df["source_time"].to_numpy(dtype=np.float32)
    tgt_times = sub_df["target_time"].to_numpy(dtype=np.float32)

    # Note: this is the legacy dataset_id from the DataLoader, not the canonical
    # Phase 0 dataset_key. For Delta-mean grouping this is fine as long as it's
    # used consistently across train/test from the same pairs_df.
    if "dataset_id" in sub_df.columns:
        dataset_ids = sub_df["dataset_id"].to_numpy(dtype=str)
    else:
        dataset_ids = np.array(["unknown"] * len(pair_indices))

    return z_sources, z_targets, src_times, tgt_times, dataset_ids
