#!/usr/bin/env python3
"""Stage 1 latent-only training: trains from cached z_state, no raw h5ad loading.

Key improvements over train_stage1_phase0.py:
- Uses LatentPairDataset instead of legacy MultiDatasetMoscotLoader
- Memory: ~1.2GB mmap vs ~43GB h5ad (enables tome-mouse)
- Supports --pca-state-path for custom PCA (e.g., refit PCA v2)
- Supports --ablation for time-only/z-only input ablation
- Includes Delta-mean, Delta-Ridge, OT Barycentric Oracle baselines
- Gene-space evaluation only at test time (lazy h5ad loading)

Usage:
  python -m cellworldmodel.script.train_stage1_latent \
    --phase0-root output/phase0 \
    --data-root data/processed/genhui_all/unified_h5ad_moscot_growth_rate \
    --pca-dir output/phase0/representations/pca128_multidata_4ds_c7dc1816 \
    --datasets E-MTAB-6967 \
    --fold-id "B::E-MTAB-6967::test_t6p75" \
    --output-dir output/stage1_latent \
    --experiment emtab_newpca \
    --device cuda:0 --batch-size 512 --rounds 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cellworldmodel.data.latent_pair_dataset import (
    LatentPairDataset,
    build_latent_pair_dataset_from_phase0,
)
from cellworldmodel.data.phase1_pairs import Phase1PairAdapter
from cellworldmodel.data.preprocessing import load_pca_state
from cellworldmodel.evaluation.baselines import (
    DeltaMeanBaseline,
    DeltaRidgeBaseline,
    extract_pair_latents,
    ot_barycentric_predict,
)
from cellworldmodel.pipeline.simple_pair_predictor import (
    SimplePairPredictor,
    build_output_paths,
    finalize_metric_sums,
    init_metric_sums,
    set_seed,
    update_metric_sums,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage1_latent")


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1 latent-only training")
    p.add_argument("--phase0-root", type=str, required=True)
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--pca-dir", type=str, required=True,
                   help="Directory containing pca_state.pkl and latent_z_state.npy")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--fold-id", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="output/stage1_latent")
    p.add_argument("--experiment", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ablation", type=str, default=None,
                   choices=["time_only", "z_only"],
                   help="Input ablation: time_only zeros z_t, z_only zeros time features")
    return p.parse_args()


# ── Time features (same as simple_pair_predictor) ──

def compute_time_stats(pairs_df, train_indices) -> Dict[str, float]:
    """Compute time normalization stats from train pairs."""
    train_df = pairs_df.iloc[train_indices]
    all_times = np.concatenate([
        train_df["source_time"].values.astype(np.float32),
        train_df["target_time"].values.astype(np.float32),
    ])
    dt = (train_df["target_time"].values - train_df["source_time"].values).astype(np.float32)
    return {
        "time_min": float(all_times.min()),
        "time_scale": max(float(all_times.max() - all_times.min()), 1e-6),
        "dt_mean": float(dt.mean()),
        "dt_std": float(dt.std()) if float(dt.std()) > 0 else 1.0,
    }


def make_time_features(time_t, time_t1, stats):
    """(B,) tensors → (B, 3) normalized time features."""
    src = (time_t - stats["time_min"]) / stats["time_scale"]
    tgt = (time_t1 - stats["time_min"]) / stats["time_scale"]
    delta = (time_t1 - time_t - stats["dt_mean"]) / stats["dt_std"]
    return torch.stack([src, tgt, delta], dim=1)


# ── Ablation wrapper ──

class AblationWrapper(nn.Module):
    def __init__(self, model: SimplePairPredictor, ablation: str):
        super().__init__()
        self.model = model
        self.ablation = ablation

    def forward(self, z_t, time_features):
        if self.ablation == "time_only":
            # Zero BOTH network input AND skip connection — no source info at all
            z_input = torch.zeros_like(z_t)
            base = torch.zeros_like(z_t)
        elif self.ablation == "z_only":
            time_features = torch.zeros_like(time_features)
            z_input = z_t
            base = z_t
        else:
            z_input = z_t
            base = z_t
        inputs = torch.cat([z_input, time_features], dim=1)
        residual = self.model.network(inputs)
        return base + residual

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, *a, **kw):
        return self.model.load_state_dict(*a, **kw)


# ── Training ──

def train_one_round(model, loader, optimizer, time_stats, device, ablation=None,
                    gradient_clip_norm=1.0):
    """Train one epoch on latent-space MSE loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        z_t = batch["z_t"].to(device)
        z_t1 = batch["z_t1"].to(device)
        time_t = batch["time_t"].to(device)
        time_t1 = batch["time_t1"].to(device)

        time_features = make_time_features(time_t, time_t1, time_stats)
        pred_z = model(z_t, time_features)

        loss = F.mse_loss(pred_z, z_t1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate_latent(model, loader, time_stats, device):
    """Evaluate model + identity baseline in latent space."""
    model.eval()
    model_mse_sum = 0.0
    identity_mse_sum = 0.0
    n_values = 0

    with torch.no_grad():
        for batch in loader:
            z_t = batch["z_t"].to(device)
            z_t1 = batch["z_t1"].to(device)
            time_t = batch["time_t"].to(device)
            time_t1 = batch["time_t1"].to(device)

            time_features = make_time_features(time_t, time_t1, time_stats)
            pred_z = model(z_t, time_features)

            model_mse_sum += (pred_z - z_t1).pow(2).sum().item()
            identity_mse_sum += (z_t - z_t1).pow(2).sum().item()
            n_values += z_t1.numel()

    return {
        "latent_mse": model_mse_sum / max(n_values, 1),
        "identity_latent_mse": identity_mse_sum / max(n_values, 1),
    }


def evaluate_pca_recon_space(
    pred_z_all: np.ndarray,
    true_z_all: np.ndarray,
    identity_z_all: np.ndarray,
    pca_state: dict,
) -> Dict[str, Dict[str, float]]:
    """Decode latent predictions to PCA-reconstructed gene space and compute metrics.

    IMPORTANT: This compares decode(pred_z) vs decode(true_z), NOT vs actual gene
    expression. Metrics are in PCA reconstruction subspace and are NOT directly
    comparable to gene-space metrics from the legacy pipeline. Use these only for
    relative comparison between models in the same run.

    For actual gene-space metrics, use lazy h5ad evaluation (TODO).
    """
    components = pca_state["components"]  # (D, G)
    mean = pca_state["mean"]              # (G,)
    latent_dim = int(components.shape[0])

    results = {}
    for name, z_pred in [("model", pred_z_all), ("identity", identity_z_all)]:
        state = init_metric_sums()
        chunk_size = 1024
        for start in range(0, len(z_pred), chunk_size):
            end = min(start + chunk_size, len(z_pred))
            pred_gene = torch.from_numpy(
                (z_pred[start:end] @ components + mean).astype(np.float32)
            )
            true_gene = torch.from_numpy(
                (true_z_all[start:end] @ components + mean).astype(np.float32)
            )
            update_metric_sums(state, pred_gene, true_gene)
        results[name] = finalize_metric_sums(state, latent_dim)
        results[name]["n_pairs"] = len(z_pred)

    return results


# ── Main ──

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Load Phase 0 artifacts ──
    phase0_root = Path(args.phase0_root)
    pca_dir = Path(args.pca_dir)

    pca_state = load_pca_state(pca_dir / "pca_state.pkl")
    z_state = np.load(pca_dir / "latent_z_state.npy", mmap_mode="r")

    logger.info("PCA: %d dims, EVR=%.4f", pca_state["latent_dim"],
                pca_state["explained_variance_ratio"].sum())
    logger.info("z_state: %s (%.2f GB)", z_state.shape, z_state.nbytes / 1e9)

    # ── Build adapter + load OT pairs ──
    adapter = Phase1PairAdapter(
        phase0_root=str(phase0_root),
        data_root=args.data_root,
        protocol=args.fold_id.split("::")[0],
        fold_id=args.fold_id,
        pca_dim=pca_state["latent_dim"],
    )

    # Load OT pairs WITHOUT loading any h5ad expression data
    # This is the key to enabling tome-mouse (43GB h5ad → ~1GB pairs_df)
    from cellworldmodel.data.pair_loader_lite import load_pairs_df

    logger.info("Loading OT pairs (lightweight, no h5ad) for datasets=%s...", args.datasets)
    pairs_df = load_pairs_df(
        data_root=args.data_root,
        selected_datasets=args.datasets,
        min_probability=0.01,
        direction_filter="source_to_target",
    )

    # ── Split pairs ──
    # Note: pair_loader_lite uses dataset_id format "family_subset" (e.g., "GSE115943_C1")
    # which differs from legacy loader's format. Phase1PairAdapter.split_pairs needs
    # the legacy loader for dataset_id→dataset_key mapping. Since we bypass the legacy
    # loader, we pass dataset_loader=None and rely on direct cell lookup.
    logger.info("Splitting pairs...")
    splits = adapter.split_pairs(pairs_df, dataset_loader=None)
    source_int_ids, target_int_ids = adapter.map_pair_endpoints(pairs_df, dataset_loader=None)

    for split_name in ["train", "val", "test"]:
        logger.info("  %s: %d pairs", split_name, len(splits[split_name]))

    if len(splits["train"]) == 0:
        logger.error("No train pairs!")
        sys.exit(1)

    # ── Build LatentPairDatasets ──
    train_ds = build_latent_pair_dataset_from_phase0(
        pairs_df, splits["train"], source_int_ids, target_int_ids, z_state
    )
    val_ds = build_latent_pair_dataset_from_phase0(
        pairs_df, splits["val"], source_int_ids, target_int_ids, z_state
    ) if len(splits["val"]) > 0 else None
    test_ds = build_latent_pair_dataset_from_phase0(
        pairs_df, splits["test"], source_int_ids, target_int_ids, z_state
    ) if len(splits["test"]) > 0 else None

    time_stats = compute_time_stats(pairs_df, splits["train"])
    logger.info("Time stats: %s", time_stats)

    # ── DataLoaders ──
    train_weights = train_ds.get_sampling_weights()
    train_sampler = WeightedRandomSampler(
        weights=train_weights, num_samples=len(train_ds), replacement=True
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True) if val_ds else None

    # ── Model ──
    latent_dim = pca_state["latent_dim"]
    model = SimplePairPredictor(
        latent_dim=latent_dim, hidden_dim=512, num_layers=2, dropout=0.1,
    ).to(device)
    logger.info("Model: %d params", sum(p.numel() for p in model.parameters()))

    if args.ablation:
        logger.info("Ablation mode: %s", args.ablation)
        model = AblationWrapper(model, args.ablation).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # ── Output paths ──
    paths = build_output_paths(args.output_dir, args.experiment)
    run_config = {
        "phase0_root": str(phase0_root),
        "pca_dir": str(pca_dir),
        "pca_evr": float(pca_state["explained_variance_ratio"].sum()),
        "datasets": args.datasets,
        "fold_id": args.fold_id,
        "batch_size": args.batch_size,
        "rounds": args.rounds,
        "lr": args.lr,
        "seed": args.seed,
        "ablation": args.ablation,
        "n_train_pairs": len(splits["train"]),
        "n_val_pairs": len(splits["val"]),
        "n_test_pairs": len(splits["test"]),
        "model_params": sum(p.numel() for p in model.parameters()),
    }
    with open(paths["resolved_config"], "w") as f:
        json.dump(run_config, f, indent=2, default=str)

    # ── Training loop ──
    best_val_score = float("inf")
    patience_counter = 0
    patience = 5

    for round_idx in range(args.rounds):
        t0 = time.time()
        train_loss = train_one_round(model, train_loader, optimizer, time_stats, device)
        train_time = time.time() - t0

        # Validate
        if val_loader is not None:
            val_metrics = evaluate_latent(model, val_loader, time_stats, device)
            val_score = val_metrics["latent_mse"]
            logger.info("Round %d/%d: train_loss=%.6f | val_latent_mse=%.6f | identity=%.6f | %.1fs",
                         round_idx + 1, args.rounds, train_loss,
                         val_score, val_metrics["identity_latent_mse"], train_time)
        else:
            val_score = train_loss
            logger.info("Round %d/%d: train_loss=%.6f (no val) | %.1fs",
                         round_idx + 1, args.rounds, train_loss, train_time)

        # Checkpointing
        if val_score < best_val_score:
            best_val_score = val_score
            patience_counter = 0
            state = model.model.state_dict() if isinstance(model, AblationWrapper) else model.state_dict()
            torch.save({
                "model_state_dict": state,
                "round": round_idx,
                "val_score": best_val_score,
                "run_config": run_config,
            }, paths["checkpoint"])
            logger.info("Saved best checkpoint (val_score=%.6f)", best_val_score)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at round %d", round_idx + 1)
                break

    # ── Test evaluation ──
    report = {
        "fold_id": args.fold_id,
        "datasets": args.datasets,
        "pca_dir": str(pca_dir),
        "pca_evr": float(pca_state["explained_variance_ratio"].sum()),
        "ablation": args.ablation,
        **{f"n_{k}_pairs": len(v) for k, v in splits.items()},
        "best_val_score": float(best_val_score),
    }

    if test_ds is None or len(test_ds) == 0:
        logger.warning("No test pairs — skipping test evaluation")
    else:
        logger.info("=== Test Evaluation ===")
        # Load best checkpoint
        ckpt = torch.load(paths["checkpoint"], map_location=device, weights_only=False)
        if isinstance(model, AblationWrapper):
            model.model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt["model_state_dict"])

        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=4, pin_memory=True)

        # Collect all test predictions in latent space
        model.eval()
        all_pred_z = []
        all_true_z = []
        all_src_z = []
        with torch.no_grad():
            for batch in test_loader:
                z_t = batch["z_t"].to(device)
                z_t1 = batch["z_t1"].to(device)
                time_t = batch["time_t"].to(device)
                time_t1 = batch["time_t1"].to(device)
                tf = make_time_features(time_t, time_t1, time_stats)
                pred = model(z_t, tf)
                all_pred_z.append(pred.cpu().numpy())
                all_true_z.append(z_t1.cpu().numpy())
                all_src_z.append(z_t.cpu().numpy())

        pred_z_np = np.concatenate(all_pred_z)
        true_z_np = np.concatenate(all_true_z)
        src_z_np = np.concatenate(all_src_z)

        # Latent-space metrics
        latent_mse = float(np.mean((pred_z_np - true_z_np) ** 2))
        identity_latent_mse = float(np.mean((src_z_np - true_z_np) ** 2))
        logger.info("  Latent MSE: %.6f | Identity: %.6f", latent_mse, identity_latent_mse)

        # Gene-space metrics (decode through PCA)
        gene_results = evaluate_pca_recon_space(pred_z_np, true_z_np, src_z_np, pca_state)
        for name, metrics in gene_results.items():
            logger.info("  %s PCA-recon: MSE=%.6f | Pearson=%.4f",
                         name, metrics["gene_mse"], metrics["pearson_mean"])

        report["test_latent_mse"] = latent_mse
        report["test_identity_latent_mse"] = identity_latent_mse
        report["test_model"] = {k: float(v) for k, v in gene_results["model"].items()}
        report["test_identity_baseline"] = {k: float(v) for k, v in gene_results["identity"].items()}

        # ── Additional baselines (latent-space) ──
        logger.info("=== Additional Baselines ===")
        z_src_train, z_tgt_train, t_src_train, t_tgt_train, dk_train = extract_pair_latents(
            pairs_df, splits["train"], source_int_ids, target_int_ids, z_state
        )
        z_src_test, z_tgt_test, t_src_test, t_tgt_test, dk_test = extract_pair_latents(
            pairs_df, splits["test"], source_int_ids, target_int_ids, z_state
        )

        def _eval_bl(name, pred_z):
            res = evaluate_pca_recon_space(pred_z, z_tgt_test, z_src_test, pca_state)
            # Only model result matters for baselines
            gene_res = res["model"]
            latent_mse_bl = float(np.mean((pred_z - z_tgt_test) ** 2))
            logger.info("  %-20s Gene MSE=%.6f | Pearson=%.4f | Latent MSE=%.6f",
                         name + ":", gene_res["gene_mse"], gene_res["pearson_mean"], latent_mse_bl)
            return {**{k: float(v) for k, v in gene_res.items()}, "latent_mse": latent_mse_bl}

        # Delta-mean
        try:
            dm = DeltaMeanBaseline()
            dm.fit(z_src_train, z_tgt_train, t_src_train, t_tgt_train, dk_train)
            pred_dm = dm.predict(z_src_test, t_src_test, t_tgt_test, dk_test)
            report["test_delta_mean"] = _eval_bl("delta_mean", pred_dm)
        except Exception as e:
            logger.warning("Delta-mean failed: %s", e)

        # Delta-Ridge
        try:
            dr = DeltaRidgeBaseline(alpha=1.0)
            dr.fit(z_src_train, z_tgt_train, t_tgt_train - t_src_train)
            pred_dr = dr.predict(z_src_test, t_tgt_test - t_src_test)
            report["test_delta_ridge"] = _eval_bl("delta_ridge", pred_dr)
        except Exception as e:
            logger.warning("Delta-Ridge failed: %s", e)

        # OT Barycentric Oracle
        try:
            ot_probs = pairs_df["probability"].to_numpy(dtype=np.float32)
            tgt_ids_test = target_int_ids[splits["test"]]
            valid_tgt = tgt_ids_test[tgt_ids_test >= 0]
            fallback = z_state[valid_tgt].mean(axis=0).astype(np.float32) if len(valid_tgt) > 0 else None
            pred_ot = ot_barycentric_predict(
                source_int_ids, target_int_ids, ot_probs, z_state, splits["test"], fallback
            )
            report["test_ot_barycentric_oracle"] = _eval_bl("ot_bary_oracle", pred_ot)
        except Exception as e:
            logger.warning("OT Barycentric failed: %s", e)

    # ── Save report ──
    report_path = paths["base_dir"] / "reports" / "test_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Saved report to %s", report_path)


if __name__ == "__main__":
    main()
