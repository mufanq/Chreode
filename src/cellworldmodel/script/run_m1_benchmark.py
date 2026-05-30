"""Unified M1 (BR-CellDrift-MLP) benchmark runner.

Trains M1 on one of the BranchSBM datasets (mouse/clonidine/trametinib/veres)
and evaluates against Identity + Mean shift + OT barycentric oracle baselines.

Reports dual protocol metrics (branchsbm_* + ours_*) plus per-branch
(post-hoc nearest-target-cluster assignment, where target labels exist).

Usage:
    PYTHONPATH=src python -m cellworldmodel.script.run_m1_benchmark \\
        --dataset mouse --epochs 300 --seed 0
    PYTHONPATH=src python -m cellworldmodel.script.run_m1_benchmark \\
        --dataset clonidine --pcs 50 --epochs 500 --seed 0
    PYTHONPATH=src python -m cellworldmodel.script.run_m1_benchmark \\
        --dataset trametinib --pcs 50 --epochs 500 --seed 0
    PYTHONPATH=src python -m cellworldmodel.script.run_m1_benchmark \\
        --dataset veres --pcs 30 --epochs 500 --seed 0

Outputs: {output_dir}/results.json with per-method dual-protocol metrics.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim

from cellworldmodel.benchmark.baselines_bench import (
    identity_baseline,
    mean_shift_baseline,
    ot_barycentric_baseline,
)
from cellworldmodel.benchmark.branchsbm_adapter import (
    ClonidineAdapter,
    MouseHematopoiesisAdapter,
    NormanAdapter,
    TrametinibAdapter,
    VeresAdapter,
)
from cellworldmodel.benchmark.common_metrics import (
    compute_dual_protocol_metrics,
    compute_per_branch_dual_protocol_metrics,
    mmd2_unbiased_multi_sigma,
    sinkhorn_w2,
)
from cellworldmodel.model.br_celldrift_bench import BRCellDriftMLP


# Per-dataset default training config (overridable via CLI).
DATASET_CONFIGS = {
    "mouse": {
        "hidden_dim": 128, "n_layers": 3, "noise_dim": 8, "time_emb_dim": 32,
        "batch_size": 256, "K": 8, "lr": 1e-3,
        "lambda_mmd": 1.0, "lambda_w2": 1.0, "sinkhorn_eps": 0.05,
        "grad_clip": None, "default_epochs": 300,
    },
    "clonidine": {
        "hidden_dim": 512, "n_layers": 3, "noise_dim": 32, "time_emb_dim": 64,
        "batch_size": 256, "K": 4, "lr": 3e-4,
        "lambda_mmd": 1.0, "lambda_w2": 0.5, "sinkhorn_eps": 0.1,
        "grad_clip": 1.0, "default_epochs": 500,
    },
    "trametinib": {
        "hidden_dim": 512, "n_layers": 3, "noise_dim": 32, "time_emb_dim": 64,
        "batch_size": 256, "K": 4, "lr": 3e-4,
        "lambda_mmd": 1.0, "lambda_w2": 0.5, "sinkhorn_eps": 0.1,
        "grad_clip": 1.0, "default_epochs": 500,
    },
    "veres": {
        "hidden_dim": 512, "n_layers": 3, "noise_dim": 16, "time_emb_dim": 32,
        "batch_size": 256, "K": 8, "lr": 3e-4,
        "lambda_mmd": 1.0, "lambda_w2": 1.0, "sinkhorn_eps": 0.05,
        "grad_clip": 1.0, "default_epochs": 500,
    },
    "norman": {
        "hidden_dim": 512, "n_layers": 3, "noise_dim": 32, "time_emb_dim": 64,
        "batch_size": 256, "K": 4, "lr": 3e-4,
        "lambda_mmd": 1.0, "lambda_w2": 0.5, "sinkhorn_eps": 0.1,
        "grad_clip": 1.0, "default_epochs": 500,
    },
}


def build_adapter(dataset: str, pcs: int, seed: int):
    if dataset == "mouse":
        return MouseHematopoiesisAdapter(seed=seed)
    if dataset == "clonidine":
        return ClonidineAdapter(pcs=pcs, seed=seed)
    if dataset == "trametinib":
        return TrametinibAdapter(pcs=pcs, seed=seed)
    if dataset == "veres":
        return VeresAdapter(seed=seed, dim=pcs)
    if dataset == "norman":
        # Norman: prefer scDFM preprocessed version (5000 HVG + splits) if available,
        # fallback to Weinberger Figshare version (2000 HVG + guide_identity column).
        scdfm_path = (Path(__file__).parent.parent.parent.parent
                      / "3rdparty" / "scDFM" / "data" / "norman" / "norman.h5ad")
        data_path = str(scdfm_path) if scdfm_path.exists() else None
        return NormanAdapter(
            data_path=data_path,
            split_seed=seed, precomputed_pca_dim=pcs, split_method="additive",
            n_top_genes=5000,
        )
    raise ValueError(f"Unknown dataset: {dataset}")


def get_target_labels(dataset: str, adapter) -> Optional[np.ndarray]:
    """Get target cluster labels for per-branch eval, if available."""
    if dataset == "clonidine":
        return adapter.get_target_cluster_labels()
    if dataset == "trametinib":
        return adapter.get_target_cluster_labels()
    if dataset == "veres":
        return adapter.get_target_cluster_labels(n_clusters=11)
    if dataset == "mouse":
        # Mouse has no explicit labels; use KMeans(2) on target
        from sklearn.cluster import KMeans
        tgt = adapter.coords_by_t[adapter.timepoints[-1]]
        km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(tgt)
        return km.labels_.astype(np.int64)
    if dataset == "norman":
        # Norman per-branch eval not applicable (targets vary per test condition)
        return None
    return None


def train_m1(
    adapter, model, device, cfg: dict, epochs: int, seed: int, log_every: int = 50,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    opt = optim.Adam(model.parameters(), lr=cfg["lr"])
    history = []
    model.train()

    train_batch = adapter.get_transition(split="train")
    delta = train_batch.delta
    dim = adapter.dim

    for ep in range(epochs):
        src = adapter.sample_source_batch(cfg["batch_size"], split="train", rng=rng).to(device)
        tgt = adapter.sample_target_batch(cfg["batch_size"], rng=rng).to(device)

        eps = torch.randn(src.shape[0], cfg["K"], dim, device=device)
        delta_t = torch.full((src.shape[0],), delta, device=device, dtype=src.dtype)

        z_hat = model(src, delta_t, eps)
        z_hat_flat = z_hat.reshape(-1, dim)

        loss_mmd = mmd2_unbiased_multi_sigma(z_hat_flat, tgt)
        loss_w2 = sinkhorn_w2(z_hat_flat, tgt, epsilon=cfg["sinkhorn_eps"], num_iters=50)
        loss = cfg["lambda_mmd"] * loss_mmd + cfg["lambda_w2"] * loss_w2

        opt.zero_grad()
        loss.backward()
        if cfg["grad_clip"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["grad_clip"])
        opt.step()

        if ep % log_every == 0 or ep == epochs - 1:
            history.append({
                "epoch": ep, "loss": float(loss.item()),
                "mmd2": float(loss_mmd.item()), "w2_approx": float(loss_w2.item()),
            })
            print(f"[ep {ep:4d}] loss={loss.item():.4f}  mmd²={loss_mmd.item():.4f}  w2≈{loss_w2.item():.4f}")

    return history


@torch.no_grad()
def predict_m1(model, src: torch.Tensor, delta: float, K: int, dim: int,
               device: torch.device, eval_batch: int = 512) -> torch.Tensor:
    preds = []
    for i in range(0, src.shape[0], eval_batch):
        src_b = src[i:i + eval_batch]
        eps = torch.randn(src_b.shape[0], K, dim, device=device)
        delta_t = torch.full((src_b.shape[0],), delta, device=device, dtype=src_b.dtype)
        pred_b = model(src_b, delta_t, eps).reshape(-1, dim)
        preds.append(pred_b)
    return torch.cat(preds, dim=0)


@torch.no_grad()
def evaluate_all(
    adapter, model, device, cfg: dict, dataset: str, seed: int,
    n_source_subsample: int = 2000, max_preds: int = 4000,
) -> dict:
    """Run M1 + 3 baselines + per-branch eval."""
    torch.manual_seed(seed + 1)
    model.eval()

    test_batch = adapter.get_transition(split="test")
    src_full = test_batch.source
    tgt_pool = test_batch.target.to(device)
    dim = adapter.dim
    delta = test_batch.delta

    # Subsample test source for tractable OT / MMD on large datasets
    n_src = src_full.shape[0]
    if n_src > n_source_subsample:
        idx = torch.randperm(n_src)[:n_source_subsample]
        src = src_full[idx].to(device)
    else:
        src = src_full.to(device)

    # M1 preds (stochastic, K samples per source pooled)
    preds_m1 = predict_m1(model, src, delta, cfg["K"], dim, device)

    # Baselines
    train_batch = adapter.get_transition(split="train")
    src_train = train_batch.source.to(device)
    tgt_train = train_batch.target.to(device)

    preds_identity = identity_baseline(src, K=cfg["K"]).to(device)
    preds_mean_shift = mean_shift_baseline(src, src_train, tgt_train, K=cfg["K"]).to(device)
    preds_ot = ot_barycentric_baseline(
        src.cpu(), tgt_train.cpu(), reg=0.0, max_samples=2000, seed=seed
    ).to(device)
    # OT bary returns (n_subsample, D); repeat K times for fair comparison vs M1's K samples
    preds_ot = preds_ot.repeat_interleave(cfg["K"], dim=0)

    # Cap preds to max_preds for MMD/Sinkhorn tractability
    def _cap(t: torch.Tensor) -> torch.Tensor:
        if t.shape[0] > max_preds:
            idx_p = torch.randperm(t.shape[0], device=t.device)[:max_preds]
            return t[idx_p]
        return t

    preds_m1 = _cap(preds_m1)
    preds_identity = _cap(preds_identity)
    preds_mean_shift = _cap(preds_mean_shift)
    preds_ot = _cap(preds_ot)

    # Target labels (for per-branch eval)
    target_labels = get_target_labels(dataset, adapter)

    results: dict[str, dict] = {}
    for name, preds in [
        ("M1 (BR-CellDrift)", preds_m1),
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


def print_summary_table(results: dict, dataset: str):
    """Print a paper-style comparison table."""
    print(f"\n{'='*80}")
    print(f"=== COMBINED (pooled) metrics — {dataset} ===")
    print(f"{'='*80}")
    header = f"\n{'Method':<30} {'W1_top2':>12} {'W2_top2':>12} {'MMD_full':>12} {'W2_full':>12}"
    print(header)
    print("-" * len(header))
    for name, branches in results.items():
        c = branches["combined"]
        w1_top2 = c.get("branchsbm_w1_top2_mean", float("nan"))
        w2_top2 = c.get("branchsbm_w2_top2_mean", float("nan"))
        mmd = c.get("branchsbm_mmd_full_mean", float("nan"))
        w2_full = c.get("ours_w2_full", float("nan"))
        print(f"{name:<30} {w1_top2:>12.4f} {w2_top2:>12.4f} {mmd:>12.4f} {w2_full:>12.4f}")

    # Per-branch (if any)
    first_method = next(iter(results.values()))
    branch_keys = [k for k in first_method if k.startswith("branch_")]
    if branch_keys:
        print(f"\n{'='*80}")
        print(f"=== PER-BRANCH metrics (post-hoc nearest-target assignment) — {dataset} ===")
        print(f"{'='*80}")
        for bk in sorted(branch_keys):
            print(f"\n--- {bk} ---")
            print(f"{'Method':<30} {'n_pred':>8} {'n_true':>8} {'W1_top2':>12} {'W2_top2':>12} {'MMD':>12}")
            for name, branches in results.items():
                b = branches.get(bk, {})
                if b.get("skipped"):
                    print(f"{name:<30} {b.get('n_pred', 0):>8} {b.get('n_true', 0):>8}  (skipped — too few cells)")
                    continue
                n_p = b.get("n_pred", 0)
                n_t = b.get("n_true", 0)
                w1 = b.get("branchsbm_w1_top2_mean", float("nan"))
                w2 = b.get("branchsbm_w2_top2_mean", float("nan"))
                mmd = b.get("branchsbm_mmd_full_mean", float("nan"))
                print(f"{name:<30} {n_p:>8} {n_t:>8} {w1:>12.4f} {w2:>12.4f} {mmd:>12.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["mouse", "clonidine", "trametinib", "veres", "norman"])
    parser.add_argument("--pcs", type=int, default=None,
                        help="PCA dim (clonidine: 50/100/150; trametinib: 50; veres: 30). "
                             "For mouse, ignored (always 2D).")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Training epochs. If None, use dataset default.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)
    # Optional overrides
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--noise-dim", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--K", type=int, default=None, dest="K")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    # Resolve config
    cfg = dict(DATASET_CONFIGS[args.dataset])
    for key in ("hidden_dim", "n_layers", "noise_dim", "batch_size", "K", "lr"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val
    epochs = args.epochs if args.epochs is not None else cfg["default_epochs"]

    # Default pcs per dataset
    if args.pcs is None:
        pcs = {"mouse": 2, "clonidine": 50, "trametinib": 50, "veres": 30, "norman": 128}[args.dataset]
    else:
        pcs = args.pcs

    output_dir = args.output_dir or f"output/benchmark/m1_{args.dataset}_pcs{pcs}_seed{args.seed}"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    adapter = build_adapter(args.dataset, pcs=pcs, seed=args.seed)
    train_batch = adapter.get_transition(split="train")
    print(f"Dataset: {args.dataset} (dim={adapter.dim})")
    print(f"  Train src: {train_batch.source.shape}, target: {train_batch.target.shape}")
    print(f"  Delta: {train_batch.delta}")

    tau_init = train_batch.delta / np.log(2)  # α(δ) ≈ 0.5 at delta=δ
    model = BRCellDriftMLP(
        dim=adapter.dim, hidden_dim=cfg["hidden_dim"], n_layers=cfg["n_layers"],
        noise_dim=cfg["noise_dim"], time_emb_dim=cfg["time_emb_dim"],
        tau_init=float(tau_init),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: BR-CellDrift-MLP, {n_params:,} params, tau_init={tau_init:.3f}")

    print(f"\nTraining for {epochs} epochs ...")
    t0 = time.time()
    history = train_m1(adapter, model, device, cfg, epochs=epochs, seed=args.seed,
                       log_every=args.log_every)
    train_time = time.time() - t0
    print(f"Train time: {train_time:.1f}s")

    print(f"\nEvaluating ...")
    results = evaluate_all(adapter, model, device, cfg, args.dataset, seed=args.seed)
    print_summary_table(results, args.dataset)

    # Save
    out_file = out_path / "results.json"
    with open(out_file, "w") as f:
        json.dump({
            "args": vars(args), "cfg": cfg, "pcs": pcs, "epochs": epochs,
            "tau_init": float(tau_init), "n_params": int(n_params),
            "train_time_s": float(train_time),
            "history": history, "eval": results,
        }, f, indent=2, default=str)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
