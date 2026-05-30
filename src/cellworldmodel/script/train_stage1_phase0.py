#!/usr/bin/env python3
"""Stage 1 training using Phase 0 artifacts.

Like train_simple_pair_predictor.py but uses:
- Phase 0 split manifest (cell-level, split-safe)
- Phase 0 cached PCA (single canonical)
- Phase 0 cell_index for consistent ID mapping

Usage:
  python -m cellworldmodel.script.train_stage1_phase0 \\
    --phase0-root output/phase0 \\
    --data-root data/processed/genhui_all/unified_h5ad_moscot_growth_rate \\
    --datasets GSE275562 \\
    --protocol A \\
    --output-dir output/stage1 \\
    --experiment smoke_gse275562
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cellworldmodel.data.phase1_pairs import Phase1PairAdapter
from cellworldmodel.evaluation.baselines import (
    DeltaMeanBaseline,
    DeltaRidgeBaseline,
    extract_pair_latents,
    ot_barycentric_predict,
)
from cellworldmodel.pipeline.simple_pair_predictor import (
    DEFAULT_EXPERIMENT_CONFIG,
    SimplePairPredictor,
    build_output_paths,
    decode_latent,
    deep_merge,
    encode_expression,
    evaluate_model,
    finalize_metric_sums,
    init_metric_sums,
    make_time_features,
    normalize_log1p_expression,
    set_seed,
    summarize_time_statistics,
    compute_target_time_means,
    compute_target_time_means_from_latent,
    train_one_round,
    update_metric_sums,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage1_phase0")


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1 training with Phase 0 artifacts")
    p.add_argument("--phase0-root", type=str, required=True)
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Datasets to train on (default: all)")
    p.add_argument("--protocol", type=str, default="A")
    p.add_argument("--fold-id", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="output/stage1")
    p.add_argument("--experiment", type=str, default="stage1_default")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pca-dim", type=int, default=128)
    p.add_argument("--pca-state-path", type=str, default=None,
                   help="Override PCA state path (e.g., for refit PCA). If None, uses default.")
    p.add_argument("--ablation", type=str, default=None,
                   choices=["time_only", "z_only"],
                   help="Input ablation: 'time_only' zeros z_t input, 'z_only' zeros time features")
    return p.parse_args()


class AblationWrapper(torch.nn.Module):
    """Wraps SimplePairPredictor to zero out inputs for ablation study.

    - time_only: zeros z_t (keeps residual connection, so model becomes z_t + f(0, time))
    - z_only: zeros time features (model becomes z_t + f(z_t, 0))
    """

    def __init__(self, model, ablation: str):
        super().__init__()
        self.model = model
        self.ablation = ablation

    def forward(self, z_t, time_features):
        if self.ablation == "time_only":
            z_input = torch.zeros_like(z_t)
        elif self.ablation == "z_only":
            time_features = torch.zeros_like(time_features)
            z_input = z_t
        else:
            z_input = z_t

        # Reconstruct the full forward: inputs = cat([z_input, time_features])
        # residual = network(inputs), return z_t + residual
        # Note: we still add to the ORIGINAL z_t (not zeroed), so residual connection is preserved
        inputs = torch.cat([z_input, time_features], dim=1)
        residual = self.model.network(inputs)
        return z_t + residual

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, *a, **kw):
        return self.model.load_state_dict(*a, **kw)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --- Phase 1 adapter ---
    adapter = Phase1PairAdapter(
        phase0_root=args.phase0_root,
        data_root=args.data_root,
        protocol=args.protocol,
        fold_id=args.fold_id,
        pca_dim=args.pca_dim,
    )

    # --- Load legacy dataset ---
    sys.path.insert(0, str(Path(args.data_root)))
    from moscot_dataloader import MultiDatasetMoscotLoader, collate_with_gene_names as legacy_collate

    # Resolve canonical dataset names → legacy loader IDs via lightweight directory scan
    selected = args.datasets
    if selected is not None:
        # Use legacy loader's _discover_all_datasets() without full data loading
        # This only scans directory structure, does NOT read h5ad/OT/expression data
        _temp = MultiDatasetMoscotLoader.__new__(MultiDatasetMoscotLoader)
        _temp.root_dir = Path(args.data_root)
        _temp.verbose = False
        all_info = _temp._discover_all_datasets()

        # Build canonical↔legacy mapping from discovered h5ad_dir paths
        canonical_to_legacy: Dict[str, str] = {}
        for legacy_id, info in all_info.items():
            h5ad_dir = Path(info["h5ad_dir"])
            try:
                rel = h5ad_dir.relative_to(Path(args.data_root))
            except ValueError:
                continue
            parts = rel.parts
            family = parts[0]
            if len(parts) >= 3 and parts[1] == "h5ad_out":
                dk = f"{family}::{'/'.join(parts[2:])}"
            else:
                dk = family
            canonical_to_legacy[dk] = legacy_id
            canonical_to_legacy[legacy_id] = legacy_id  # also accept legacy directly

        # Also build family→[legacy_ids] mapping for family-level expansion
        family_to_legacy: Dict[str, list] = {}
        for legacy_id, info in all_info.items():
            h5ad_dir = Path(info["h5ad_dir"])
            try:
                rel = h5ad_dir.relative_to(Path(args.data_root))
            except ValueError:
                continue
            family = rel.parts[0]
            family_to_legacy.setdefault(family, []).append(legacy_id)

        # Resolve user names: try exact match first, then family expansion
        legacy_selected = []
        for name in selected:
            resolved = canonical_to_legacy.get(name)
            if resolved:
                legacy_selected.append(resolved)
            elif name in family_to_legacy:
                # Family-level name: expand to all subsets
                expanded = family_to_legacy[name]
                logger.info("Expanding family '%s' → %s", name, expanded)
                legacy_selected.extend(expanded)
            else:
                logger.warning("Dataset '%s' not found. Available families: %s, keys: %s",
                               name, sorted(family_to_legacy.keys()),
                               sorted(set(canonical_to_legacy.values())))
        if not legacy_selected:
            logger.error("No valid datasets.")
            sys.exit(1)
        selected = legacy_selected

    logger.info("Loading MultiDatasetMoscotLoader with datasets=%s...", selected)
    dataset = MultiDatasetMoscotLoader(
        root_dir=args.data_root,
        selected_datasets=selected,
        min_probability=0.01,
        direction_filter="source_to_target",
        return_dataset_id=True,
        return_probability=True,
        return_direction=True,
        return_growth_rates=False,
        verbose=True,
    )

    # --- Split pairs using Phase 0 cell splits (pass dataset for robust ID mapping) ---
    logger.info("Splitting pairs using Phase 0 manifest...")
    splits = adapter.split_pairs(dataset.pairs_df, dataset_loader=dataset)

    # Preflight check: abort if train is empty
    if len(splits["train"]) == 0:
        logger.error("No train pairs! This fold may not have source_to_target pairs "
                      "where source AND target are both in train. "
                      "Try a different fold or dataset with more s2t transitions.")
        sys.exit(1)
    if len(splits["val"]) == 0:
        logger.warning("No val pairs — will skip validation and use train loss for checkpointing.")
    if len(splits["test"]) == 0:
        logger.warning("No test pairs — test evaluation will be skipped.")

    # --- Load Phase 0 PCA ---
    if args.pca_state_path:
        from cellworldmodel.data.preprocessing import load_pca_state
        logger.info("Loading custom PCA state from %s...", args.pca_state_path)
        pca_state_np = load_pca_state(args.pca_state_path)
    else:
        logger.info("Loading default Phase 0 PCA state...")
        pca_state_np = adapter.get_pca_state()
    # Convert numpy PCA state to torch format for SimplePairPredictor compatibility
    pca_state = {
        "components": torch.tensor(pca_state_np["components"], dtype=torch.float32),
        "mean": torch.tensor(pca_state_np["mean"], dtype=torch.float32),
        "explained_variance_ratio": torch.tensor(
            pca_state_np["explained_variance_ratio"], dtype=torch.float32
        ),
    }
    logger.info("PCA: %d dims, %.1f%% variance explained",
                pca_state_np["latent_dim"],
                100 * pca_state_np["explained_variance_ratio"].sum())

    # --- Compute time stats and target time means (on train split) ---
    time_stats = summarize_time_statistics(dataset.pairs_df, splits["train"])
    target_sum = pca_state_np.get("target_sum", 10000.0)

    repr_dir = Path(args.phase0_root) / "representations" / f"pca{args.pca_dim}_traincore_v1"
    z_state_path = repr_dir / "latent_z_state.npy"
    if z_state_path.exists():
        logger.info("Loading cached z_state from %s (mmap)...", z_state_path)
        latent_z_state = np.load(z_state_path, mmap_mode="r")
        _, target_int_ids = adapter.map_pair_endpoints(dataset.pairs_df, dataset_loader=dataset)
        target_time_means = compute_target_time_means_from_latent(
            pairs_df=dataset.pairs_df,
            train_indices=splits["train"],
            target_int_ids=target_int_ids,
            latent_z_state=latent_z_state,
            pca_state=pca_state,
        )
        logger.info("Computed target-time means from cached z_state (%d train pairs)",
                    len(splits["train"]))
    else:
        logger.warning("Phase 0 z_state cache not found at %s; falling back to slow gene-space loop",
                       z_state_path)
        latent_z_state = None  # additional baselines will be skipped
        target_time_means = compute_target_time_means(
            dataset=dataset,
            train_indices=splits["train"],
            pca_state=pca_state,
            normalize_target_sum=target_sum,
        )

    # --- Build model ---
    model_cfg = DEFAULT_EXPERIMENT_CONFIG["model"]
    model = SimplePairPredictor(
        latent_dim=pca_state_np["latent_dim"],
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)
    logger.info("Model: %s", model)
    logger.info("Parameters: %d", sum(p.numel() for p in model.parameters()))

    # Apply ablation wrapper if requested
    if args.ablation:
        logger.info("Ablation mode: %s", args.ablation)
        model = AblationWrapper(model, args.ablation).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-5
    )

    # --- Build dataloaders ---
    from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

    train_subset = Subset(dataset, splits["train"])

    # Weighted sampling for train
    train_weights = dataset.get_sampling_weights()[splits["train"]]
    train_sampler = WeightedRandomSampler(
        weights=train_weights, num_samples=len(train_subset), replacement=True
    )

    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=4,
        collate_fn=legacy_collate, pin_memory=True,
    )

    # Val loader (may be empty)
    val_loader = None
    if len(splits["val"]) > 0:
        val_subset = Subset(dataset, splits["val"])
        val_loader = DataLoader(
            val_subset, batch_size=args.batch_size,
            shuffle=False, num_workers=4,
            collate_fn=legacy_collate, pin_memory=True,
        )

    # --- Output paths ---
    paths = build_output_paths(args.output_dir, args.experiment)

    # Save config
    run_config = {
        "phase0_root": args.phase0_root,
        "data_root": args.data_root,
        "datasets": args.datasets,
        "protocol": args.protocol,
        "fold_id": adapter.fold_id,
        "pca_dim": args.pca_dim,
        "batch_size": args.batch_size,
        "rounds": args.rounds,
        "lr": args.lr,
        "seed": args.seed,
        "n_train_pairs": len(splits["train"]),
        "n_val_pairs": len(splits["val"]),
        "n_test_pairs": len(splits["test"]),
    }
    with open(paths["resolved_config"], "w") as f:
        json.dump(run_config, f, indent=2, default=str)

    # --- Training loop ---
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 3
    global_step = 0

    for round_idx in range(args.rounds):
        logger.info("=== Round %d/%d ===", round_idx + 1, args.rounds)

        # Train
        train_metrics = train_one_round(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            pca_state=pca_state,
            time_stats=time_stats,
            normalize_target_sum=target_sum,
            device=device,
            gene_loss_weight=0.1,
            gradient_clip_norm=1.0,
            round_idx=round_idx,
            global_step_start=global_step,
        )
        global_step = train_metrics["global_step_end"]
        logger.info("Train loss: %.4f (latent: %.4f, gene: %.4f)",
                     train_metrics["train_loss"],
                     train_metrics["train_latent_loss"],
                     train_metrics["train_gene_loss"])

        # Validate (if val pairs exist)
        if len(splits["val"]) > 0:
            val_result = evaluate_model(
                model=model,
                loader=val_loader,
                pca_state=pca_state,
                time_stats=time_stats,
                normalize_target_sum=target_sum,
                target_time_means=target_time_means,
                device=device,
                max_batches=32,
            )
            # evaluate_model returns nested dict: {"model": {...}, "identity_baseline": {...}, ...}
            val_model = val_result["model"]
            val_score = val_model["gene_mse"]
            logger.info("Val gene_mse=%.6f | pearson=%.4f | identity=%.6f | time_mean=%.6f",
                         val_model["gene_mse"],
                         val_model["pearson_mean"],
                         val_result["identity_baseline"]["gene_mse"],
                         val_result["target_time_mean_baseline"]["gene_mse"])
        else:
            # No val pairs: use train loss as proxy
            val_score = train_metrics["train_loss"]
            logger.info("No val pairs — using train loss %.4f as proxy", val_score)

        # Early stopping / checkpointing
        if val_score < best_val_loss:
            best_val_loss = val_score
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "round": round_idx,
                "val_score": best_val_loss,
                "pca_state": pca_state,
                "time_stats": time_stats,
                "run_config": run_config,
            }, paths["checkpoint"])
            logger.info("Saved best checkpoint (val_score=%.6f)", best_val_loss)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at round %d", round_idx + 1)
                break

    # --- Test evaluation ---
    if len(splits["test"]) == 0:
        logger.warning("Skipping test evaluation — no test pairs.")
        test_model = {"gene_mse": float("nan"), "pearson_mean": float("nan")}
        test_result = {"model": test_model, "identity_baseline": test_model, "target_time_mean_baseline": test_model}
    else:
        logger.info("=== Final Test Evaluation ===")
        test_subset = Subset(dataset, splits["test"])
        test_loader = DataLoader(
            test_subset, batch_size=args.batch_size,
            shuffle=False, num_workers=4,
            collate_fn=legacy_collate, pin_memory=True,
        )

        # Load best checkpoint
        ckpt = torch.load(paths["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        test_result = evaluate_model(
            model=model,
            loader=test_loader,
            pca_state=pca_state,
            time_stats=time_stats,
            normalize_target_sum=target_sum,
            target_time_means=target_time_means,
            device=device,
            max_batches=None,
        )

    test_model = test_result["model"]
    logger.info("Test results:")
    logger.info("  Gene MSE: %.6f", test_model["gene_mse"])
    logger.info("  Pearson:  %.4f", test_model["pearson_mean"])
    logger.info("  vs Identity:  %.6f", test_result["identity_baseline"]["gene_mse"])
    logger.info("  vs Time-mean: %.6f", test_result["target_time_mean_baseline"]["gene_mse"])

    # --- Additional baselines (Delta-mean, Delta-Ridge, OT Barycentric Oracle) ---
    # These baselines predict in PCA latent space. To ensure fair comparison with
    # the model, we evaluate them against ACTUAL gene expressions (not PCA-reconstructed),
    # by iterating through the test DataLoader (same pipeline as evaluate_model).
    baseline_results = {}
    nan_baseline = {"gene_mse": float("nan"), "gene_mae": float("nan"), "pearson_mean": float("nan")}

    if len(splits["test"]) > 0 and latent_z_state is not None:
        logger.info("=== Computing additional baselines ===")
        source_int_ids, target_int_ids = adapter.map_pair_endpoints(
            dataset.pairs_df, dataset_loader=dataset
        )

        # Extract train pair latents for fitting
        z_src_train, z_tgt_train, t_src_train, t_tgt_train, dk_train = extract_pair_latents(
            dataset.pairs_df, splits["train"], source_int_ids, target_int_ids, latent_z_state
        )
        # Extract test pair latents for baseline prediction
        z_src_test, z_tgt_test, t_src_test, t_tgt_test, dk_test = extract_pair_latents(
            dataset.pairs_df, splits["test"], source_int_ids, target_int_ids, latent_z_state
        )
        dt_train = t_tgt_train - t_src_train
        dt_test = t_tgt_test - t_src_test

        # Pre-compute all baseline predictions in latent space
        baseline_pred_z = {}

        # 1) Delta-mean
        try:
            dm = DeltaMeanBaseline()
            dm.fit(z_src_train, z_tgt_train, t_src_train, t_tgt_train, dk_train)
            baseline_pred_z["delta_mean"] = dm.predict(z_src_test, t_src_test, t_tgt_test, dk_test)
        except Exception as e:
            logger.warning("Delta-mean baseline fit failed: %s", e)

        # 2) Delta-Ridge
        try:
            dr = DeltaRidgeBaseline(alpha=1.0)
            dr.fit(z_src_train, z_tgt_train, dt_train)
            baseline_pred_z["delta_ridge"] = dr.predict(z_src_test, dt_test)
        except Exception as e:
            logger.warning("Delta-Ridge baseline fit failed: %s", e)

        # 3) OT Barycentric Oracle
        try:
            ot_probs = dataset.pairs_df["probability"].to_numpy(dtype=np.float32)
            tgt_ids_test = target_int_ids[splits["test"]]
            valid_tgt = tgt_ids_test[tgt_ids_test >= 0]
            fallback = latent_z_state[valid_tgt].mean(axis=0).astype(np.float32) if len(valid_tgt) > 0 else None
            baseline_pred_z["ot_barycentric_oracle"] = ot_barycentric_predict(
                source_int_ids=source_int_ids,
                target_int_ids=target_int_ids,
                ot_probs=ot_probs,
                z_all=latent_z_state,
                pair_indices=splits["test"],
                fallback_mean=fallback,
            )
        except Exception as e:
            logger.warning("OT Barycentric Oracle baseline fit failed: %s", e)

        # Evaluate all baselines against ACTUAL gene expressions via DataLoader
        # (same target as model evaluation — not PCA-reconstructed)
        baseline_states = {name: init_metric_sums() for name in baseline_pred_z}
        pair_offset = 0

        test_subset = Subset(dataset, splits["test"])
        test_loader_baselines = DataLoader(
            test_subset, batch_size=args.batch_size,
            shuffle=False, num_workers=4,
            collate_fn=legacy_collate, pin_memory=True,
        )

        with torch.no_grad():
            for batch in test_loader_baselines:
                expr_t1 = batch["expr_t1"]
                target_gene = normalize_log1p_expression(expr_t1, target_sum=target_sum)
                bs = target_gene.shape[0]

                for name, pred_z_all in baseline_pred_z.items():
                    chunk = pred_z_all[pair_offset:pair_offset + bs]
                    pred_gene = torch.from_numpy(
                        (chunk @ pca_state_np["components"] + pca_state_np["mean"]).astype(np.float32)
                    )
                    update_metric_sums(baseline_states[name], pred_gene, target_gene)

                pair_offset += bs

        latent_dim = int(pca_state["components"].shape[0])
        for name in baseline_pred_z:
            baseline_results[name] = finalize_metric_sums(baseline_states[name], latent_dim)
            baseline_results[name]["n_pairs"] = pair_offset
            logger.info("  %-20s Gene MSE=%.6f | Pearson=%.4f",
                         name + ":", baseline_results[name]["gene_mse"],
                         baseline_results[name]["pearson_mean"])

    for name in ["delta_mean", "delta_ridge", "ot_barycentric_oracle"]:
        if name not in baseline_results:
            baseline_results[name] = nan_baseline

    # Save test report
    def _serialize(d):
        return {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in d.items()}

    report = {
        "fold_id": adapter.fold_id,
        "datasets": args.datasets,
        "train_pairs": len(splits["train"]),
        "val_pairs": len(splits["val"]),
        "test_pairs": len(splits["test"]),
        "best_val_score": float(best_val_loss),
        "test_model": _serialize(test_model),
        "test_identity_baseline": _serialize(test_result["identity_baseline"]),
        "test_time_mean_baseline": _serialize(test_result["target_time_mean_baseline"]),
        "test_delta_mean_baseline": _serialize(baseline_results["delta_mean"]),
        "test_delta_ridge_baseline": _serialize(baseline_results["delta_ridge"]),
        "test_ot_barycentric_oracle": _serialize(baseline_results["ot_barycentric_oracle"]),
    }
    report_path = paths["base_dir"] / "reports" / "test_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved test report to %s", report_path)


if __name__ == "__main__":
    main()
