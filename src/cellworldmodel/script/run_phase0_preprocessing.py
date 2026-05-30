#!/usr/bin/env python3
"""Phase 0: End-to-end preprocessing pipeline.

Produces all artifacts needed for Stage 1/2 training:
  1. cell_index.parquet       — global cell registry
  2. split_manifest.json      — train/val/test definitions
  3. pca_state.pkl            — PCA fit on train_core
  4. latent_z_state.npy       — all cells in PCA(128) space
  5. whitening_stats.npz      — mean/std for L2Norm(Whiten(z_state))
  6. harmony_state.pkl        — Harmony fit on train_core (optional)
  7. latent_z_int.npy          — all cells in Harmony space (optional)
  8. packed_ot/train_core/*.npz — OT CSR for training

Usage:
  python -m cellworldmodel.script.run_phase0_preprocessing \\
    --data-root data/processed/genhui_all/unified_h5ad_moscot_growth_rate \\
    --out-root output/phase0 \\
    --protocol A \\
    --fold-id A__dev_gse275562 \\
    --skip-harmony
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for imports
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cellworldmodel.data.split import (
    build_cell_index,
    build_split_manifest,
    discover_h5ad_catalog,
    get_fold,
    get_partition_ids,
    list_folds,
    load_cell_index,
    load_split_manifest,
    save_cell_index,
    save_split_manifest,
)
from cellworldmodel.data.preprocessing import (
    compute_whitening_stats,
    fit_harmony_on_train,
    fit_pca_on_train_cells,
    load_pca_state,
    save_harmony_state,
    save_pca_state,
    save_whitening_stats,
    transform_all_cells_to_z_int,
    transform_all_cells_to_z_state,
)
from cellworldmodel.data.ot_index import build_packed_ot_for_fold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase0")


def parse_args():
    p = argparse.ArgumentParser(description="Phase 0: preprocessing pipeline")
    p.add_argument("--data-root", type=str, required=True,
                   help="Root of h5ad/OT data directory")
    p.add_argument("--out-root", type=str, default="output/phase0",
                   help="Output directory for artifacts")

    # What to build
    p.add_argument("--build-manifest", action="store_true",
                   help="Build cell_index.parquet + split_manifest.json")
    p.add_argument("--materialize", action="store_true",
                   help="Materialize latent/OT artifacts for a specific fold")

    # Fold selection (for --materialize)
    p.add_argument("--protocol", type=str, default="A",
                   help="Split protocol (A/B/C/D)")
    p.add_argument("--fold-id", type=str, default=None,
                   help="Specific fold ID to materialize")

    # Options
    p.add_argument("--skip-harmony", action="store_true",
                   help="Skip Harmony fitting (use identity for z_int)")
    p.add_argument("--pca-dim", type=int, default=128)
    p.add_argument("--pca-fit-cells", type=int, default=50000,
                   help="Max cells to sample for PCA fit")
    p.add_argument("--ot-mass-threshold", type=float, default=0.99)
    p.add_argument("--ot-k-cap", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 1: Build global artifacts
# ---------------------------------------------------------------------------


def step_build_manifest(data_root: Path, out_root: Path, seed: int):
    """Build cell_index.parquet + split_manifest.json."""
    t0 = time.time()

    # Discover h5ad catalog
    logger.info("Scanning data directory: %s", data_root)
    catalog = discover_h5ad_catalog(data_root)
    logger.info("Found %d h5ad files across %d dataset_keys",
                len(catalog), catalog["dataset_key"].nunique())

    # Build cell index
    logger.info("Building cell_index (reading h5ad metadata)...")
    cell_index = build_cell_index(data_root, catalog=catalog)
    logger.info("Cell index: %d cells, %d dataset_keys, %d timepoints",
                len(cell_index), cell_index["dataset_key"].nunique(),
                cell_index["timepoint"].nunique())

    # Save cell index
    ci_path = out_root / "cell_index.parquet"
    save_cell_index(cell_index, ci_path)
    logger.info("Saved cell_index to %s", ci_path)

    # Build split manifest
    logger.info("Building split manifest (all protocols)...")
    # Build OT-aware s2t graph for intelligent fold selection
    from cellworldmodel.data.split import build_s2t_graph_from_data
    logger.info("Building s2t transition graph for OT-aware fold selection...")
    s2t_graph = build_s2t_graph_from_data(data_root)

    manifest = build_split_manifest(
        cell_index, data_root=str(data_root),
        s2t_graph_per_series=s2t_graph,
    )

    # Log summary
    for protocol in ["A", "B", "C", "D"]:
        folds = list_folds(manifest, protocol)
        logger.info("Protocol %s: %d folds", protocol, len(folds))

    # Save manifest
    sm_path = out_root / "split_manifest.json"
    save_split_manifest(manifest, sm_path)
    logger.info("Saved split_manifest to %s", sm_path)

    elapsed = time.time() - t0
    logger.info("Build manifest completed in %.1fs", elapsed)

    return cell_index, manifest


# ---------------------------------------------------------------------------
# Step 2: Materialize artifacts for a fold
# ---------------------------------------------------------------------------


def step_materialize(
    data_root: Path,
    out_root: Path,
    cell_index: pd.DataFrame,
    manifest: dict,
    protocol: str,
    fold_id: str,
    pca_dim: int,
    pca_fit_cells: int,
    skip_harmony: bool,
    ot_mass_threshold: float,
    ot_k_cap: int,
    seed: int,
):
    """Materialize PCA/Harmony/OT artifacts for one fold."""
    t0 = time.time()

    fold = get_fold(manifest, protocol, fold_id)
    if fold is None:
        available = list_folds(manifest, protocol)
        logger.error("Fold %s not found in protocol %s. Available: %s",
                     fold_id, protocol, available[:10])
        sys.exit(1)

    manifest_dir = out_root  # sidecar npz files are relative to manifest's parent dir
    train_ids = get_partition_ids(fold, "train", manifest_dir=manifest_dir)
    val_ids = get_partition_ids(fold, "val", manifest_dir=manifest_dir)
    test_ids = get_partition_ids(fold, "test", manifest_dir=manifest_dir)
    train_ids_set = set(train_ids.tolist())  # set for membership tests (OT, PCA)
    train_ids_arr = np.sort(train_ids)       # sorted array for numpy indexing (whitening, harmony)
    logger.info("Fold %s: train=%d, val=%d, test=%d cells",
                fold_id, len(train_ids), len(val_ids), len(test_ids))

    # Artifact directory layout:
    # representations/pca128_traincore_v1/ — canonical, shared across folds
    # folds/{fold_slug}/packed_ot/         — per-fold OT
    # folds/{fold_slug}/reports/           — per-fold QC
    fold_slug = fold_id.replace("::", "__").replace("/", "_").replace("|", "_")
    repr_dir = out_root / "representations" / f"pca{pca_dim}_traincore_v1"
    repr_dir.mkdir(parents=True, exist_ok=True)
    fold_dir = out_root / "folds" / fold_slug
    fold_dir.mkdir(parents=True, exist_ok=True)

    # --- PCA ---
    pca_path = repr_dir / "pca_state.pkl"
    import hashlib
    train_hash = hashlib.sha256(np.sort(train_ids).tobytes()).hexdigest()[:16]

    # List of all derived artifacts that depend on PCA
    _derived_artifacts = [
        repr_dir / "latent_z_state.npy",
        repr_dir / "whitening_stats.npz",
        repr_dir / "harmony_state.pkl",
        repr_dir / "latent_z_int.npy",
    ]

    if pca_path.exists():
        pca_state = load_pca_state(pca_path)
        saved_hash = pca_state.get("provenance", {}).get("train_ids_hash")
        if not saved_hash or saved_hash != train_hash:
            reason = "missing provenance" if not saved_hash else f"hash mismatch (saved={saved_hash} current={train_hash})"
            logger.warning(
                "Cached PCA invalid: %s. Invalidating ALL derived artifacts and refitting.", reason
            )
            # Invalidate PCA + ALL downstream artifacts
            for artifact in [pca_path] + _derived_artifacts:
                if artifact.exists():
                    artifact.unlink()
                    logger.info("  Deleted stale: %s", artifact.name)
        else:
            logger.info("PCA state loaded and provenance verified: %s", pca_path)

    if not pca_path.exists():
        logger.info("Fitting PCA on train cells (dim=%d, max_cells=%d)...",
                     pca_dim, pca_fit_cells)
        pca_state = fit_pca_on_train_cells(
            data_root=data_root,
            cell_index=cell_index,
            train_int_ids=train_ids_set,
            latent_dim=pca_dim,
            max_cells_for_fit=pca_fit_cells,
            seed=seed,
        )
        # Add provenance metadata (canonical PCA is shared across folds)
        pca_state["provenance"] = {
            "fit_source": "canonical_traincore",
            "first_materialized_by_fold": fold_id,
            "pca_dim": pca_dim,
            "seed": seed,
            "n_train": len(train_ids),
            "train_ids_hash": train_hash,
            "data_root": str(data_root),
        }
        save_pca_state(pca_state, pca_path)
        logger.info("PCA explained variance (top 10): %s",
                     np.array2string(pca_state["explained_variance_ratio"][:10], precision=3))
        logger.info("PCA total explained variance: %.3f",
                     pca_state["explained_variance_ratio"].sum())

    # --- Transform all cells to z_state ---
    z_state_path = repr_dir / "latent_z_state.npy"
    if z_state_path.exists():
        logger.info("z_state already exists, loading: %s", z_state_path)
        z_state = np.load(z_state_path)
    else:
        logger.info("Transforming all %d cells to z_state...", len(cell_index))
        z_state = transform_all_cells_to_z_state(
            data_root=data_root,
            cell_index=cell_index,
            pca_state=pca_state,
        )
        np.save(z_state_path, z_state)
        logger.info("Saved z_state: shape=%s, dtype=%s", z_state.shape, z_state.dtype)

    # NaN check
    n_nan = np.isnan(z_state).any(axis=1).sum()
    if n_nan > 0:
        logger.warning("z_state has %d rows with NaN!", n_nan)

    # --- Whitening stats ---
    ws_path = repr_dir / "whitening_stats.npz"
    if ws_path.exists():
        logger.info("Whitening stats already exist: %s", ws_path)
    else:
        logger.info("Computing whitening stats on train cells...")
        w_stats = compute_whitening_stats(z_state, train_int_ids=train_ids_arr)
        save_whitening_stats(w_stats, ws_path)
        logger.info("Whitening mean norm: %.3f, std range: [%.3f, %.3f]",
                     np.linalg.norm(w_stats["mean"]),
                     w_stats["std"].min(), w_stats["std"].max())

    # --- Harmony ---
    z_int_path = repr_dir / "latent_z_int.npy"
    if skip_harmony:
        logger.info("Skipping Harmony (--skip-harmony). Using z_state as z_int.")
        if not z_int_path.exists():
            np.save(z_int_path, z_state)
    else:
        harmony_path = repr_dir / "harmony_state.pkl"
        if z_int_path.exists():
            logger.info("z_int already exists: %s", z_int_path)
        else:
            logger.info("Fitting Harmony on train cells...")
            harmony_state = fit_harmony_on_train(
                z_state=z_state,
                cell_index=cell_index,
                train_int_ids=train_ids_arr,
            )
            save_harmony_state(harmony_state, harmony_path)

            logger.info("Transforming all cells to z_int...")
            z_int = transform_all_cells_to_z_int(
                z_state=z_state,
                cell_index=cell_index,
                harmony_state=harmony_state,
                train_int_ids=train_ids_arr,
            )
            np.save(z_int_path, z_int)
            logger.info("Saved z_int: shape=%s", z_int.shape)

    # --- PackedOT (per-fold: different folds have different train cells) ---
    ot_dir = fold_dir / "packed_ot"
    ot_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building PackedOT CSR indices (train cells only)...")
    qc_df = build_packed_ot_for_fold(
        data_root=data_root,
        cell_index=cell_index,
        train_int_ids=train_ids_set,
        output_dir=ot_dir,
        mass_threshold=ot_mass_threshold,
        k_cap=ot_k_cap,
    )

    # Save QC report (per-fold)
    reports_dir = fold_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    qc_path = reports_dir / "transition_qc.parquet"
    qc_df.to_parquet(qc_path, index=False)
    logger.info("Saved transition QC to %s", qc_path)

    # QC summary
    n_built = (qc_df["status"] == "built").sum()
    n_missing = (qc_df["status"] == "missing_forward_direction").sum()
    n_empty = (qc_df["status"] == "empty_after_filter").sum()
    total_sources = qc_df.loc[qc_df["status"] == "built", "n_source_rows"].sum()
    total_nnz = qc_df.loc[qc_df["status"] == "built", "nnz"].sum()

    # Save summary JSON
    summary = {
        "fold_id": fold_id,
        "protocol": protocol,
        "n_cells_total": len(cell_index),
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "pca_dim": pca_dim,
        "pca_explained_variance": float(pca_state["explained_variance_ratio"].sum()),
        "z_state_shape": list(z_state.shape),
        "z_state_nan_rows": int(n_nan),
        "skip_harmony": skip_harmony,
        "ot_transitions_built": int(n_built),
        "ot_transitions_missing_forward": int(n_missing),
        "ot_transitions_empty": int(n_empty),
        "ot_total_source_rows": int(total_sources),
        "ot_total_nnz": int(total_nnz),
        "ot_mass_threshold": ot_mass_threshold,
        "ot_k_cap": ot_k_cap,
    }
    summary_path = reports_dir / "phase0_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary to %s", summary_path)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Phase 0 materialization complete in %.1fs", elapsed)
    logger.info("  Fold: %s", fold_id)
    logger.info("  Cells: %d total, %d train", len(cell_index), len(train_ids))
    logger.info("  PCA: %d dim, %.1f%% variance explained",
                pca_dim, 100 * pca_state["explained_variance_ratio"].sum())
    logger.info("  OT: %d transitions built, %d sources, %d nnz",
                n_built, total_sources, total_nnz)
    logger.info("  Missing forward direction: %d transitions (%.0f%%)",
                n_missing, 100 * n_missing / max(len(qc_df), 1))
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        logger.error("Data root does not exist: %s", data_root)
        sys.exit(1)

    # If neither flag specified, do both
    do_manifest = args.build_manifest or (not args.build_manifest and not args.materialize)
    do_materialize = args.materialize or (not args.build_manifest and not args.materialize)

    cell_index = None
    manifest = None

    # Step 1: Build manifest
    if do_manifest:
        ci_path = out_root / "cell_index.parquet"
        sm_path = out_root / "split_manifest.json"

        if ci_path.exists() and sm_path.exists() and not args.build_manifest:
            logger.info("Global artifacts already exist, loading...")
            cell_index = load_cell_index(ci_path)
            manifest = load_split_manifest(sm_path)
        else:
            cell_index, manifest = step_build_manifest(data_root, out_root, args.seed)

    # Step 2: Materialize
    if do_materialize:
        if cell_index is None:
            ci_path = out_root / "cell_index.parquet"
            sm_path = out_root / "split_manifest.json"
            if not ci_path.exists() or not sm_path.exists():
                logger.error("Global artifacts not found. Run with --build-manifest first.")
                sys.exit(1)
            cell_index = load_cell_index(ci_path)
            manifest = load_split_manifest(sm_path)

        # Auto-select fold if not specified
        fold_id = args.fold_id
        if fold_id is None:
            folds = list_folds(manifest, args.protocol)
            if not folds:
                logger.error("No folds for protocol %s", args.protocol)
                sys.exit(1)
            fold_id = folds[0]
            logger.info("Auto-selected fold: %s", fold_id)

        step_materialize(
            data_root=data_root,
            out_root=out_root,
            cell_index=cell_index,
            manifest=manifest,
            protocol=args.protocol,
            fold_id=fold_id,
            pca_dim=args.pca_dim,
            pca_fit_cells=args.pca_fit_cells,
            skip_harmony=args.skip_harmony,
            ot_mass_threshold=args.ot_mass_threshold,
            ot_k_cap=args.ot_k_cap,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
