"""Endpoint benchmark evaluation for CellWorldModel methods."""
from __future__ import annotations

import numpy as np
import torch

from cellworldmodel.benchmark.baselines_bench import (
    identity_baseline,
    mean_shift_baseline,
    ot_barycentric_baseline,
)
from cellworldmodel.benchmark.common_metrics import (
    compute_dual_protocol_metrics,
    compute_per_branch_dual_protocol_metrics,
)
from cellworldmodel.benchmark.registry import get_target_labels


@torch.no_grad()
def predict_model(model, src: torch.Tensor, delta: float, K: int, dim: int,
                  device: torch.device, eval_batch: int = 512) -> torch.Tensor:
    preds = []
    for i in range(0, src.shape[0], eval_batch):
        src_b = src[i:i + eval_batch]
        eps = torch.randn(src_b.shape[0], K, dim, device=device)
        delta_t = torch.full((src_b.shape[0],), delta, device=device, dtype=src_b.dtype)
        pred_b = model(src_b, delta_t, eps).reshape(-1, dim)
        preds.append(pred_b)
    return torch.cat(preds, dim=0)


def evaluate_all(
    adapter, model, device, cfg: dict, dataset: str, method: str, seed: int,
    n_source_subsample: int = 2000, max_preds: int = 4000,
) -> dict:
    """Evaluate {model, Identity, Mean shift, OT oracle} using dual protocol + per-branch."""
    torch.manual_seed(seed + 1)
    model.eval()

    test_batch = adapter.get_transition(split="test")
    src_full = test_batch.source
    tgt_pool_full = test_batch.target
    # Cap target pool to keep MMD / W2 pairwise-dist matrices tractable.
    # Weinreb d6 = 54K cells → pairwise would OOM. Use same max_preds cap.
    if tgt_pool_full.shape[0] > max_preds:
        tgt_idx = torch.randperm(tgt_pool_full.shape[0])[:max_preds]
        tgt_pool = tgt_pool_full[tgt_idx].to(device)
    else:
        tgt_pool = tgt_pool_full.to(device)
    dim = adapter.dim
    delta = test_batch.delta

    n_src = src_full.shape[0]
    if n_src > n_source_subsample:
        idx = torch.randperm(n_src)[:n_source_subsample]
        src = src_full[idx].to(device)
    else:
        src = src_full.to(device)

    # Model predictions (method-specific forward, but same interface)
    # PCCellDriftMLP needs .eval() to skip create_graph=True (breaking no_grad context)
    preds_model = predict_model(model, src, delta, cfg["K"], dim, device)

    # Baselines (shared)
    train_batch = adapter.get_transition(split="train")
    src_train = train_batch.source.to(device)
    tgt_train = train_batch.target.to(device)
    preds_identity = identity_baseline(src, K=cfg["K"]).to(device)
    preds_mean_shift = mean_shift_baseline(src, src_train, tgt_train, K=cfg["K"]).to(device)
    preds_ot = ot_barycentric_baseline(src.cpu(), tgt_train.cpu(), reg=0.0, max_samples=2000, seed=seed).to(device)
    preds_ot = preds_ot.repeat_interleave(cfg["K"], dim=0)

    def _cap(t: torch.Tensor) -> torch.Tensor:
        if t.shape[0] > max_preds:
            idx_p = torch.randperm(t.shape[0], device=t.device)[:max_preds]
            return t[idx_p]
        return t

    preds_model = _cap(preds_model)
    preds_identity = _cap(preds_identity)
    preds_mean_shift = _cap(preds_mean_shift)
    preds_ot = _cap(preds_ot)

    target_labels_full = get_target_labels(dataset, adapter)
    # If we subsampled tgt_pool, subsample labels to match
    if target_labels_full is not None and tgt_pool_full.shape[0] > max_preds:
        target_labels = target_labels_full[tgt_idx.cpu().numpy()]
    else:
        target_labels = target_labels_full

    model_name = {"m1": "M1 (BR-CellDrift)", "m2": "M2 (PC-CellDrift)",
                  "m7": "M7 (BR + Drift V)", "m8": "M8 (PC + Drift V)",
                  "m9": "M9 (DriftDiT-1D)",
                  "m10": "M10 (DiT + Drift V + Land)"}[method]
    results: dict[str, dict] = {}
    for name, preds in [
        (model_name, preds_model),
        ("Identity", preds_identity),
        ("Mean shift", preds_mean_shift),
        ("OT barycentric (oracle)", preds_ot),
    ]:
        if target_labels is not None:
            results[name] = compute_per_branch_dual_protocol_metrics(
                preds, tgt_pool, target_labels, seed=seed,
            )
        else:
            results[name] = {"combined": compute_dual_protocol_metrics(preds, tgt_pool, seed=seed)}
    return results


def print_summary_table(results: dict, dataset: str, method: str):
    print(f"\n{'='*80}\n=== [{method}] COMBINED metrics — {dataset} ===\n{'='*80}")
    header = f"\n{'Method':<30} {'W1_top2':>12} {'W2_top2':>12} {'MMD_full':>12} {'W2_full':>12}"
    print(header)
    print("-" * len(header))
    for name, branches in results.items():
        c = branches["combined"]
        print(f"{name:<30} {c.get('branchsbm_w1_top2_mean', float('nan')):>12.4f} "
              f"{c.get('branchsbm_w2_top2_mean', float('nan')):>12.4f} "
              f"{c.get('branchsbm_mmd_full_mean', float('nan')):>12.4f} "
              f"{c.get('ours_w2_full', float('nan')):>12.4f}")

