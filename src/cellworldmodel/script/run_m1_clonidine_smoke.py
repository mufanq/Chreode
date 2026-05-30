"""M1 smoke test on Clonidine 50D perturbation dataset.

Trains BR-CellDrift-MLP with MMD + W2 on control → Clonidine-perturbed cells.
Uses 50D PCA (Tahoe-100M A549 cell line).

Expected: M1 should clearly beat Identity and Mean shift, and be competitive
with BranchSBM branched (ep100 on a6000).

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m \\
        cellworldmodel.script.run_m1_clonidine_smoke --epochs 500 --seed 0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from cellworldmodel.benchmark.branchsbm_adapter import ClonidineAdapter
from cellworldmodel.benchmark.common_metrics import (
    compute_all_distributional_metrics,
    compute_dual_protocol_metrics,
    mmd2_unbiased_multi_sigma,
    sinkhorn_w2,
)
from cellworldmodel.model.br_celldrift_bench import BRCellDriftMLP


def train_m1(
    adapter: ClonidineAdapter,
    model: BRCellDriftMLP,
    device: torch.device,
    epochs: int,
    batch_size: int,
    K: int,
    lr: float,
    lambda_mmd: float,
    lambda_w2: float,
    log_every: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = optim.Adam(model.parameters(), lr=lr)
    history = []

    model.train()
    train_batch = adapter.get_transition(split="train")
    dim = adapter.dim
    delta = train_batch.delta

    for ep in range(epochs):
        src = adapter.sample_source_batch(batch_size, split="train", rng=rng).to(device)
        tgt = adapter.sample_target_batch(batch_size, rng=rng).to(device)

        eps = torch.randn(src.shape[0], K, dim, device=device)
        delta_t = torch.full((src.shape[0],), delta, device=device, dtype=src.dtype)

        z_hat = model(src, delta_t, eps)
        z_hat_flat = z_hat.reshape(-1, dim)

        loss_mmd = mmd2_unbiased_multi_sigma(z_hat_flat, tgt)
        loss_w2 = sinkhorn_w2(z_hat_flat, tgt, epsilon=0.1, num_iters=50)
        loss = lambda_mmd * loss_mmd + lambda_w2 * loss_w2

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        if ep % log_every == 0 or ep == epochs - 1:
            history.append({
                "epoch": ep,
                "loss": float(loss.item()),
                "mmd2": float(loss_mmd.item()),
                "w2_approx": float(loss_w2.item()),
            })
            print(f"[ep {ep:4d}] loss={loss.item():.4f}  mmd²={loss_mmd.item():.4f}  w2≈{loss_w2.item():.4f}")

    return history


@torch.no_grad()
def evaluate_predictions(
    adapter: ClonidineAdapter,
    model: BRCellDriftMLP,
    device: torch.device,
    K: int,
    seed: int,
    n_source_subsample: int = 2000,  # avoid OOM on large test source
) -> dict:
    torch.manual_seed(seed + 1)
    model.eval()

    batch = adapter.get_transition(split="test")
    src_full = batch.source
    tgt_pool = batch.target.to(device)
    dim = adapter.dim
    delta = batch.delta

    # Subsample source to avoid OOM
    n_src = src_full.shape[0]
    if n_src > n_source_subsample:
        idx = torch.randperm(n_src)[:n_source_subsample]
        src = src_full[idx].to(device)
    else:
        src = src_full.to(device)

    # M1 predictions in batches to avoid peak memory
    eval_batch = 512
    preds_list = []
    for i in range(0, src.shape[0], eval_batch):
        src_b = src[i:i+eval_batch]
        eps = torch.randn(src_b.shape[0], K, dim, device=device)
        delta_t = torch.full((src_b.shape[0],), delta, device=device, dtype=src_b.dtype)
        pred_b = model(src_b, delta_t, eps).reshape(-1, dim)
        preds_list.append(pred_b)
    preds = torch.cat(preds_list, dim=0)

    identity_pred = src.repeat_interleave(K, dim=0)

    train_batch = adapter.get_transition(split="train")
    mean_shift = train_batch.target.mean(0).to(device) - train_batch.source.mean(0).to(device)
    mean_shift_pred = (src + mean_shift).repeat_interleave(K, dim=0)

    # compute_all_distributional_metrics already subsamples for exact W1/W2
    # MMD + Sinkhorn are on full preds vs target (still could OOM if preds too big)
    # Cap preds to 4K if needed
    max_preds = 4000
    if preds.shape[0] > max_preds:
        idx_p = torch.randperm(preds.shape[0], device=device)[:max_preds]
        preds = preds[idx_p]
        identity_pred = identity_pred[idx_p]
        mean_shift_pred = mean_shift_pred[idx_p]

    results = {
        "M1 (BR-CellDrift)": compute_dual_protocol_metrics(preds, tgt_pool, seed=seed),
        "Identity": compute_dual_protocol_metrics(identity_pred, tgt_pool, seed=seed),
        "Mean shift": compute_dual_protocol_metrics(mean_shift_pred, tgt_pool, seed=seed),
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcs", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--K", type=int, default=4, dest="K")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--noise-dim", type=int, default=32)
    parser.add_argument("--lambda-mmd", type=float, default=1.0)
    parser.add_argument("--lambda-w2", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str,
                        default="output/benchmark/m1_clonidine_smoke")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    adapter = ClonidineAdapter(pcs=args.pcs, seed=args.seed)
    train_batch = adapter.get_transition(split="train")
    print(f"Dataset: Clonidine {args.pcs}D")
    print(f"  Train source (DMSO): {train_batch.source.shape}")
    print(f"  Target (perturbed): {train_batch.target.shape}")
    print(f"  Delta: {train_batch.delta}")

    # Clonidine delta=1, so tau_init = 1/ln2 ≈ 1.44
    tau_init = 1.0 / np.log(2)
    model = BRCellDriftMLP(
        dim=adapter.dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        noise_dim=args.noise_dim,
        time_emb_dim=64,
        tau_init=float(tau_init),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: BR-CellDrift-MLP, {n_params:,} params")

    print(f"\nTraining M1 for {args.epochs} epochs ...")
    t0 = time.time()
    history = train_m1(
        adapter, model, device,
        epochs=args.epochs, batch_size=args.batch_size, K=args.K, lr=args.lr,
        lambda_mmd=args.lambda_mmd, lambda_w2=args.lambda_w2,
        log_every=args.log_every, seed=args.seed,
    )
    train_time = time.time() - t0
    print(f"Training time: {train_time:.1f}s")

    print(f"\nEvaluating on test split ...")
    results = evaluate_predictions(adapter, model, device, K=args.K, seed=args.seed)

    # BranchSBM-style table (for direct paper comparison)
    print(f"\n=== BranchSBM-style protocol (paper Table 3 compatible) ===")
    print(f"  - W1/W2 on top-2 PCs, biased MMD on full dims, 5 trials subsample")
    print(f"\n{'Method':<22} {'W1_top2':>10} {'W2_top2':>10} {'MMD_full':>10} {'W1_full':>10} {'W2_full':>10}")
    print("-" * 76)
    for name, m in results.items():
        print(
            f"{name:<22} "
            f"{m['branchsbm_w1_top2_mean']:>8.3f}±{m['branchsbm_w1_top2_std']:.3f} "
            f"{m['branchsbm_w2_top2_mean']:>8.3f}±{m['branchsbm_w2_top2_std']:.3f} "
            f"{m['branchsbm_mmd_full_mean']:>8.4f}±{m['branchsbm_mmd_full_std']:.4f} "
            f"{m['branchsbm_w1_full_mean']:>8.3f}±{m['branchsbm_w1_full_std']:.3f} "
            f"{m['branchsbm_w2_full_mean']:>8.3f}±{m['branchsbm_w2_full_std']:.3f}"
        )

    # Our stricter protocol
    print(f"\n=== Our stricter protocol (full-dim, unbiased MMD, median-heuristic σ) ===")
    print(f"\n{'Method':<22} {'MMD² (med)':>12} {'Sinkhorn W²':>12} {'W1_full':>10} {'W2_full':>10}")
    print("-" * 70)
    for name, m in results.items():
        print(
            f"{name:<22} "
            f"{m['ours_mmd2_unbiased_median']:>12.4f} "
            f"{m['ours_sinkhorn_w2_full']:>12.4f} "
            f"{m['ours_w1_full']:>10.4f} {m['ours_w2_full']:>10.4f}"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump({
            "args": vars(args),
            "n_params": n_params,
            "train_time_s": train_time,
            "history": history,
            "eval": results,
        }, f, indent=2)
    print(f"\nSaved results to {out_dir}/results.json")

    # Summary: compare on both protocols
    print(f"\n=== Summary ===")
    for proto, w2_key in [
        ("BranchSBM top-2 PC", "branchsbm_w2_top2_mean"),
        ("Our full-dim", "ours_w2_full"),
    ]:
        m1_w2 = results["M1 (BR-CellDrift)"][w2_key]
        id_w2 = results["Identity"][w2_key]
        ms_w2 = results["Mean shift"][w2_key]
        s1 = "✓" if m1_w2 < id_w2 else "✗"
        s2 = "✓" if m1_w2 < ms_w2 else "✗"
        print(f"[{proto}] M1={m1_w2:.3f}  Identity={id_w2:.3f} ({s1})  Mean shift={ms_w2:.3f} ({s2})")


if __name__ == "__main__":
    main()
