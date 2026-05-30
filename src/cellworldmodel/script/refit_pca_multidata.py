#!/usr/bin/env python3
"""Refit PCA on large multi-dataset train_core with balanced sampling.

Addresses the 19.5% EVR bottleneck: the original PCA was fit on only 9K cells
from GSE275562 Split A. This script fits on balanced cells from multiple
datasets, with per-dataset cap enforced AFTER union across folds.

Also computes PCA reconstruction floor (encode→decode MSE) as a diagnostic.
The floor represents the irreducible error for any model constrained to
predict through this PCA's latent space and linear decode.

Usage:
  python -m cellworldmodel.script.refit_pca_multidata \
    --phase0-root output/phase0 \
    --data-root data/processed/genhui_all/unified_h5ad_moscot_growth_rate \
    --datasets E-MTAB-6967 GSE106340 GSE115943 \
    --fold-ids "B::E-MTAB-6967::test_t6p75" "B::GSE106340::test_t8p00" \
               "B::GSE115943::test_t0p50" \
    --max-cells-per-dataset 50000 \
    --latent-dims 128 256 \
    --output-dir output/phase0/representations \
    --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import IncrementalPCA, PCA

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cellworldmodel.data.preprocessing import (
    normalize_log1p,
    save_pca_state,
)
from cellworldmodel.data.split import (
    get_fold,
    get_partition_ids,
    load_cell_index,
    load_split_manifest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("refit_pca")


def parse_args():
    p = argparse.ArgumentParser(description="Refit PCA on multi-dataset train_core")
    p.add_argument("--phase0-root", type=str, required=True)
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--datasets", nargs="+", required=True,
                   help="Dataset families to include (e.g., E-MTAB-6967 GSE106340)")
    p.add_argument("--fold-ids", nargs="+", required=True,
                   help="Fold IDs for train cell selection (one per dataset)")
    p.add_argument("--max-cells-per-dataset", type=int, default=50000,
                   help="Max cells sampled per dataset family (balanced, enforced after union)")
    p.add_argument("--latent-dims", nargs="+", type=int, default=[128, 256],
                   help="PCA dimensions to fit (e.g., 128 256)")
    p.add_argument("--output-dir", type=str, default="output/phase0/representations")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _run_config_slug(args) -> str:
    """Generate a short unique slug from ALL run config for artifact naming."""
    key_obj = {
        "datasets": sorted(args.datasets),
        "fold_ids": sorted(args.fold_ids),
        "max_cells_per_dataset": int(args.max_cells_per_dataset),
        "latent_dims": sorted(int(x) for x in args.latent_dims),
        "seed": int(args.seed),
    }
    key = json.dumps(key_obj, sort_keys=True)
    short_hash = hashlib.md5(key.encode()).hexdigest()[:8]
    return f"multidata_{len(args.datasets)}ds_{short_hash}"


def collect_train_cells_balanced(
    cell_index: pd.DataFrame,
    manifest: dict,
    phase0_root: Path,
    fold_ids: list[str],
    datasets: list[str],
    max_per_dataset: int,
    seed: int,
) -> np.ndarray:
    """Collect train cell int_ids from multiple folds, balanced across datasets.

    Strategy: first union all eligible train cells per dataset across ALL folds,
    then sample up to max_per_dataset from each dataset's union. This guarantees
    the per-dataset cap regardless of how many folds are provided.
    """
    rng = np.random.default_rng(seed)
    manifest_dir = Path(phase0_root)

    # Step 1: gather union of eligible train cells per dataset
    eligible_by_dataset: dict[str, set] = {ds: set() for ds in datasets}

    for fold_id in fold_ids:
        protocol = fold_id.split("::")[0]
        fold = get_fold(manifest, protocol, fold_id)
        if fold is None:
            logger.warning("Fold %s not found, skipping", fold_id)
            continue

        train_ids = set(get_partition_ids(fold, "train", manifest_dir=manifest_dir).tolist())
        train_cells = cell_index[cell_index["int_id"].isin(train_ids)]

        for ds in datasets:
            # Match dataset family (e.g., GSE115943 matches GSE115943::C1, GSE115943::C2)
            mask = train_cells["dataset_key"].str.startswith(ds)
            ds_ids = train_cells.loc[mask, "int_id"].values
            eligible_by_dataset[ds].update(ds_ids.tolist())

    # Step 2: sample up to max_per_dataset from each dataset's union
    sampled = []
    for ds in datasets:
        ids = np.array(sorted(eligible_by_dataset[ds]), dtype=np.int64)
        if len(ids) == 0:
            logger.warning("Dataset %s: 0 eligible train cells across all folds", ds)
            continue
        if len(ids) > max_per_dataset:
            ids = rng.choice(ids, size=max_per_dataset, replace=False)
        sampled.append(ids)
        logger.info("  %s: %d eligible → %d sampled", ds,
                     len(eligible_by_dataset[ds]), len(ids))

    if not sampled:
        raise ValueError("No train cells collected from any dataset")

    combined = np.concatenate(sampled)
    logger.info("Total: %d cells (balanced, cap=%d per dataset)", len(combined), max_per_dataset)
    return combined


def load_expression_for_cells(
    cell_index: pd.DataFrame,
    int_ids: np.ndarray,
    data_root: Path,
    target_sum: float = 1e4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and normalize expression for specified cells.

    Returns:
        matrix: (N, n_genes) float32 normalized expression
        ids_array: (N,) int64 corresponding int_ids (same order)
    """
    data_root = Path(data_root)
    id_set = set(int_ids.tolist())
    cells = cell_index[cell_index["int_id"].isin(id_set)]
    file_groups = cells.groupby("h5ad_relpath")

    collected = []
    collected_ids = []
    n_requested = len(id_set)
    t0 = time.time()

    for h5ad_relpath, group in file_groups:
        h5ad_path = data_root / h5ad_relpath
        logger.info("  Loading %s (%d cells)...", h5ad_relpath, len(group))
        adata = sc.read_h5ad(h5ad_path)
        obs_name_to_pos = {name: i for i, name in enumerate(adata.obs_names)}

        positions = []
        ids = []
        n_missing = 0
        for _, row in group.iterrows():
            pos = obs_name_to_pos.get(row["obs_name"])
            if pos is not None:
                positions.append(pos)
                ids.append(row["int_id"])
            else:
                n_missing += 1

        if n_missing > 0:
            logger.warning("  %s: %d/%d cells missing from h5ad obs_names",
                           h5ad_relpath, n_missing, len(group))

        if not positions:
            del adata
            continue

        X = adata.X[positions]
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = normalize_log1p(X, target_sum)
        collected.append(X)
        collected_ids.extend(ids)

        del adata  # free memory

    if not collected:
        raise ValueError("No expression data loaded")

    matrix = np.vstack(collected).astype(np.float32)
    ids_array = np.array(collected_ids, dtype=np.int64)

    n_loaded = len(ids_array)
    if n_loaded < n_requested:
        logger.warning("Loaded %d/%d requested cells (%d missing)",
                       n_loaded, n_requested, n_requested - n_loaded)
    logger.info("Loaded expression: %s (%.1fs)", matrix.shape, time.time() - t0)
    return matrix, ids_array


def fit_and_save_pca(
    matrix: np.ndarray,
    latent_dim: int,
    output_dir: Path,
    version_tag: str,
    seed: int = 42,
) -> dict:
    """Fit PCA and save state. Uses IncrementalPCA for large matrices (>100K cells)."""
    n_cells, n_genes = matrix.shape
    effective_dim = min(latent_dim, n_cells - 1, n_genes)
    logger.info("Fitting PCA with %d components on (%d, %d) matrix...",
                effective_dim, n_cells, n_genes)
    t0 = time.time()

    if n_cells > 500_000:
        # Use IncrementalPCA only for very large matrices (>500K cells ≈ 66GB)
        # Benchmark: IncrementalPCA is ~16x slower than randomized on 150K cells
        # but uses less peak memory (streaming). Only worth it when matrix won't fit in RAM.
        logger.info("Using IncrementalPCA (n_cells=%d > 500K)", n_cells)
        pca = IncrementalPCA(n_components=effective_dim, batch_size=10_000)
        pca.fit(matrix)
    else:
        # Randomized SVD: fast (minutes not hours) and sufficient for <=500K cells
        # 150K×33075 ≈ 20GB, fits in RAM easily on 8×A6000 node
        pca = PCA(n_components=effective_dim, svd_solver="randomized", random_state=seed)
        pca.fit(matrix)

    pca_state = {
        "components": pca.components_.astype(np.float32),
        "mean": pca.mean_.astype(np.float32),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
        "latent_dim": effective_dim,
        "target_sum": 1e4,
        "n_fit_cells": n_cells,
    }

    evr = pca_state["explained_variance_ratio"]
    logger.info("PCA%d: cumulative EVR = %.4f (%.1f%%), fit in %.1fs",
                effective_dim, evr.sum(), 100 * evr.sum(), time.time() - t0)

    # Save
    save_dir = Path(output_dir) / f"pca{effective_dim}_{version_tag}"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_pca_state(pca_state, save_dir / "pca_state.pkl")

    # Save scree plot data
    np.savez(save_dir / "scree_data.npz",
             explained_variance_ratio=evr,
             cumulative_evr=np.cumsum(evr))
    logger.info("Saved to %s", save_dir)

    return pca_state


def compute_reconstruction_floor(
    matrix: np.ndarray,
    pca_state: dict,
    chunk_size: int = 8192,
) -> dict:
    """Compute PCA reconstruction error (encode→decode MSE) in chunks.

    This is the irreducible error floor for models working in this PCA space.
    Note: this floor only applies to models whose outputs are decoded by this
    same PCA inverse transform (i.e., Stage 1 and Stage 2).

    Chunked to avoid OOM for large matrices.
    """
    components = pca_state["components"]  # (latent_dim, n_genes)
    mean = pca_state["mean"]              # (n_genes,)
    n_cells = matrix.shape[0]

    sum_sq = 0.0
    sum_abs = 0.0
    sum_corr = 0.0

    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        chunk = matrix[start:end]

        # Encode + decode
        z = (chunk - mean) @ components.T
        recon = z @ components + mean

        # Accumulate MSE/MAE
        diff = chunk - recon
        sum_sq += float((diff ** 2).sum())
        sum_abs += float(np.abs(diff).sum())

        # Per-cell Pearson (chunked)
        # cor(recon_i, truth_i) for each cell i
        recon_centered = recon - recon.mean(axis=1, keepdims=True)
        truth_centered = chunk - chunk.mean(axis=1, keepdims=True)
        num = (recon_centered * truth_centered).sum(axis=1)
        den = np.sqrt((recon_centered ** 2).sum(axis=1) * (truth_centered ** 2).sum(axis=1))
        den = np.where(den > 0, den, 1.0)
        sum_corr += float((num / den).sum())

    n_values = n_cells * matrix.shape[1]
    return {
        "reconstruction_mse": sum_sq / n_values,
        "reconstruction_mae": sum_abs / n_values,
        "reconstruction_pearson": sum_corr / n_cells,
        "latent_dim": pca_state["latent_dim"],
        "n_cells": n_cells,
        "n_genes": matrix.shape[1],
    }


def main():
    args = parse_args()
    phase0_root = Path(args.phase0_root)
    version_tag = _run_config_slug(args)

    # Load Phase 0 artifacts
    cell_index = load_cell_index(phase0_root / "cell_index.parquet")
    manifest = load_split_manifest(phase0_root / "split_manifest.json")

    # Collect balanced train cells
    logger.info("=== Collecting train cells from %d datasets ===", len(args.datasets))
    train_ids = collect_train_cells_balanced(
        cell_index, manifest, phase0_root,
        args.fold_ids, args.datasets,
        args.max_cells_per_dataset, args.seed,
    )

    # Load expression
    logger.info("=== Loading expression data ===")
    matrix, loaded_ids = load_expression_for_cells(
        cell_index, train_ids, Path(args.data_root),
    )

    # Fit PCA for each latent_dim
    results = {}
    for dim in args.latent_dims:
        logger.info("=== Fitting PCA%d ===", dim)
        pca_state = fit_and_save_pca(matrix, dim, Path(args.output_dir), version_tag, args.seed)

        # Compute reconstruction floor (chunked)
        logger.info("=== Computing PCA%d reconstruction floor ===", dim)
        floor = compute_reconstruction_floor(matrix, pca_state)
        results[f"pca{dim}_new"] = {
            "evr_fit_time": float(pca_state["explained_variance_ratio"].sum()),
            "n_fit_cells": pca_state["n_fit_cells"],
            "version_tag": version_tag,
            **floor,
        }
        logger.info("PCA%d reconstruction floor: MSE=%.6f | MAE=%.6f | Pearson=%.4f",
                     dim, floor["reconstruction_mse"], floor["reconstruction_mae"],
                     floor["reconstruction_pearson"])

    # Also compute floor for old PCA for comparison
    old_pca_path = phase0_root / "representations" / "pca128_traincore_v1" / "pca_state.pkl"
    if old_pca_path.exists():
        from cellworldmodel.data.preprocessing import load_pca_state
        old_pca = load_pca_state(old_pca_path)
        old_floor = compute_reconstruction_floor(matrix, old_pca)
        results["pca128_old_traincore_v1"] = {
            "evr_fit_time": float(old_pca["explained_variance_ratio"].sum()),
            "evr_note": "fit-time EVR on old 9K-cell sample, NOT on this evaluation set",
            "n_fit_cells": old_pca.get("n_fit_cells", "unknown"),
            **old_floor,
        }
        logger.info("Old PCA128 reconstruction floor: MSE=%.6f | Pearson=%.4f",
                     old_floor["reconstruction_mse"], old_floor["reconstruction_pearson"])

    # Save summary
    summary_path = Path(args.output_dir) / f"pca_refit_summary_{version_tag}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved summary to %s", summary_path)

    # Print comparison
    logger.info("=== PCA Comparison ===")
    for name, res in sorted(results.items()):
        logger.info("  %-35s EVR(fit)=%.4f | Recon MSE=%.6f | Recon Pearson=%.4f | cells=%s",
                     name, res["evr_fit_time"], res["reconstruction_mse"],
                     res["reconstruction_pearson"], res["n_fit_cells"])


if __name__ == "__main__":
    main()
