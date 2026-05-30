"""M1 smoke test on Mouse Hematopoiesis 2D dataset.

Trains BR-CellDrift-MLP with MMD + W2 loss on the simplest BranchSBM dataset
(2D, 2 branches, ~1400 source cells) to verify:
  1. Training loop runs without error
  2. Loss decreases monotonically
  3. Final predicted population is closer to true target than trivial baselines

Baselines:
  - Identity: ẑ = z (no change)
  - Mean shift: ẑ = z + (mean_target - mean_source)
  - Source: just use source cells as "prediction" (shouldn't beat this!)

Expected runtime on unites1 with 1 A6000: ~2-5 minutes.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m \\
        cellworldmodel.script.run_m1_mouse_smoke --epochs 200 --seed 0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from cellworldmodel.benchmark.branchsbm_adapter import MouseHematopoiesisAdapter
from cellworldmodel.benchmark.common_metrics import (
    compute_all_distributional_metrics,
    compute_dual_protocol_metrics,
    mmd2_unbiased_multi_sigma,
    sinkhorn_w2,
)
from cellworldmodel.model.br_celldrift_bench import BRCellDriftMLP


def train_m1(
    adapter: MouseHematopoiesisAdapter,
    model: BRCellDriftMLP,
    device: torch.device,
    epochs: int = 200,
    batch_size: int = 256,
    K: int = 8,
    lr: float = 1e-3,
    lambda_mmd: float = 1.0,
    lambda_w2: float = 1.0,
    log_every: int = 20,
    seed: int = 0,
) -> list[dict]:
    """Train M1: BR-CellDrift-MLP with MMD + W2 loss."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    opt = optim.Adam(model.parameters(), lr=lr)
    history = []

    model.train()
    train_batch = adapter.get_transition(split="train")
    target_all = train_batch.target.to(device)  # fixed target population
    delta = train_batch.delta
    dim = adapter.dim

    for ep in range(epochs):
        # Sample source + target batches
        src = adapter.sample_source_batch(batch_size, split="train", rng=rng).to(device)
        tgt = adapter.sample_target_batch(batch_size, rng=rng).to(device)

        # K noise samples per source
        eps = torch.randn(src.shape[0], K, dim, device=device)
        delta_t = torch.full((src.shape[0],), delta, device=device, dtype=src.dtype)

        # Forward
        z_hat = model(src, delta_t, eps)  # (B, K, dim)
        z_hat_flat = z_hat.reshape(-1, dim)  # (B*K, dim)

        # Population losses
        loss_mmd = mmd2_unbiased_multi_sigma(z_hat_flat, tgt)
        loss_w2 = sinkhorn_w2(z_hat_flat, tgt, epsilon=0.05, num_iters=50)

        loss = lambda_mmd * loss_mmd + lambda_w2 * loss_w2

        opt.zero_grad()
        loss.backward()
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
    adapter: MouseHematopoiesisAdapter,
    model: BRCellDriftMLP,
    device: torch.device,
    K: int = 8,
    seed: int = 0,
) -> dict:
    """Evaluate M1 on test source cells. Compares against trivial baselines."""
    torch.manual_seed(seed + 1)
    model.eval()

    batch = adapter.get_transition(split="test")
    src = batch.source.to(device)
    tgt_pool = batch.target.to(device)
    delta = batch.delta
    dim = adapter.dim

    # M1 predictions (K samples per source, pooled)
    eps = torch.randn(src.shape[0], K, dim, device=device)
    delta_t = torch.full((src.shape[0],), delta, device=device, dtype=src.dtype)
    preds = model(src, delta_t, eps).reshape(-1, dim)  # (N_test * K, dim)

    # Trivial baselines
    # 1. Identity: predict source itself (each source counted K times for fair comparison)
    identity_pred = src.repeat_interleave(K, dim=0)  # (N_test * K, dim)
    # 2. Mean shift: z + mean_shift where shift from train
    train_batch = adapter.get_transition(split="train")
    mean_shift = train_batch.target.mean(0).to(device) - train_batch.source.mean(0).to(device)
    mean_shift_pred = (src + mean_shift).repeat_interleave(K, dim=0)

    results = {
        "M1 (BR-CellDrift)": compute_dual_protocol_metrics(preds, tgt_pool, seed=seed),
        "Identity": compute_dual_protocol_metrics(identity_pred, tgt_pool, seed=seed),
        "Mean shift": compute_dual_protocol_metrics(mean_shift_pred, tgt_pool, seed=seed),
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--K", type=int, default=8, dest="K")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--noise-dim", type=int, default=8)
    parser.add_argument("--lambda-mmd", type=float, default=1.0)
    parser.add_argument("--lambda-w2", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str,
                        default="output/benchmark/m1_mouse_smoke")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Data
    adapter = MouseHematopoiesisAdapter(seed=args.seed)
    train_batch = adapter.get_transition(split="train")
    print(f"Dataset: Mouse Hematopoiesis 2D")
    print(f"  Train source: {train_batch.source.shape}, target: {train_batch.target.shape}")
    print(f"  Delta: {train_batch.delta}")

    # Model
    # Set tau_init so alpha(delta) ≈ 0.5 at delta=2
    tau_init = train_batch.delta / np.log(2)
    model = BRCellDriftMLP(
        dim=adapter.dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        noise_dim=args.noise_dim,
        time_emb_dim=32,
        tau_init=float(tau_init),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: BR-CellDrift-MLP, {n_params:,} params, tau_init={tau_init:.3f}")

    # Train
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

    # Evaluate
    print(f"\nEvaluating on test split ...")
    results = evaluate_predictions(adapter, model, device, K=args.K, seed=args.seed)

    # BranchSBM-style (top-2 PC == full-dim for Mouse 2D data)
    print(f"\n=== BranchSBM-style protocol (paper Table 2 compatible) ===")
    print(f"  Note: Mouse is 2D so top-2 PC == full-dim W1/W2")
    print(f"\n{'Method':<22} {'W1_top2':>10} {'W2_top2':>10} {'MMD_full':>10}")
    print("-" * 60)
    for name, m in results.items():
        print(
            f"{name:<22} "
            f"{m['branchsbm_w1_top2_mean']:>8.3f}±{m['branchsbm_w1_top2_std']:.3f} "
            f"{m['branchsbm_w2_top2_mean']:>8.3f}±{m['branchsbm_w2_top2_std']:.3f} "
            f"{m['branchsbm_mmd_full_mean']:>8.4f}±{m['branchsbm_mmd_full_std']:.4f}"
        )

    # Our stricter protocol
    print(f"\n=== Our stricter protocol (unbiased MMD + median-heuristic σ) ===")
    print(f"\n{'Method':<22} {'MMD² (med)':>12} {'Sinkhorn W²':>12} {'W1_full':>10} {'W2_full':>10}")
    print("-" * 70)
    for name, m in results.items():
        print(
            f"{name:<22} "
            f"{m['ours_mmd2_unbiased_median']:>12.4f} "
            f"{m['ours_sinkhorn_w2_full']:>12.4f} "
            f"{m['ours_w1_full']:>10.4f} "
            f"{m['ours_w2_full']:>10.4f}"
        )

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump({
            "args": vars(args),
            "tau_init": float(tau_init),
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
