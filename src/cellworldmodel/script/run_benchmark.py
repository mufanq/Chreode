"""Unified M1/M2/M7/M8 benchmark runner.

Trains one of {M1 BR + MMD/W2, M2 PC + MMD/W2, M7 BR + Drift V, M8 PC + Drift V + L_down}
on a BranchSBM dataset and evaluates against Identity + Mean shift + OT barycentric oracle.

Method dispatch (decisions in agent/human-review/m2-m7-m8-decisions.typ):
  M1: BRCellDriftMLP,  loss = λ_MMD·MMD + λ_W2·W2                           (baseline, saturated on Veres)
  M2: PCCellDriftMLP,  loss = λ_MMD·MMD + λ_W2·W2                           (PC architecture ablation)
  M7: BRCellDriftMLP,  loss = λ_MMD·MMD + λ_W2·W2 + λ_drift·L_drift         (core Drifting Model method)
  M8: PCCellDriftMLP,  loss = λ_MMD·MMD + λ_W2·W2 + λ_drift·L_drift + λ_down·L_down   (full PC+Drift)

Usage:
    PYTHONPATH=src python -m cellworldmodel.script.run_benchmark \\
        --method m7 --dataset veres --epochs 500 --seed 0

Output: {output_dir}/results.json with per-method dual-protocol metrics + 4 baselines.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from pathlib import Path


from cellworldmodel.benchmark.config_overrides import apply_common_overrides
from cellworldmodel.benchmark.configs import DATASET_CONFIGS, DEFAULT_PCS
from cellworldmodel.benchmark.experiment_registry import add_experiment_arg, apply_experiment_args
from cellworldmodel.model.drift_dit_1d import DRIFT_DIT_1D_MODELS
from cellworldmodel.script.wandb_utils import (
    add_wandb_args,
    flatten_numeric,
    maybe_init_wandb,
    wandb_run_info,
    wandb_summary_update,
)


METHODS = {"m1", "m2", "m7", "m8", "m9", "m10"}


# Backward-compatible imports for scripts that used to import builders from this module.
from cellworldmodel.benchmark.registry import build_adapter, build_model
from cellworldmodel.training.benchmark_loop import train_method
from cellworldmodel.training.checkpointing import (
    apply_model_config_from_checkpoint,
    load_shape_matched_checkpoint,
    save_model_checkpoint,
)


from cellworldmodel.evaluation.benchmark import evaluate_all, print_summary_table


def set_model_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    add_experiment_arg(parser)
    parser.add_argument("--method", required=False, choices=sorted(METHODS))
    parser.add_argument("--dataset", required=True,
                        choices=["mouse", "clonidine", "trametinib", "veres", "norman",
                                 "weinreb_hvg", "weinreb_scvi", "veres_scvi",
                                 "paper_weinreb_scvi128", "paper_veres_scvi128"])
    parser.add_argument("--pcs", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0,
                        help="Model seed (noise sampling, param init)")
    parser.add_argument("--split-seed", type=int, default=None,
                        help="Data split seed for train/test partitioning. "
                        "If None, uses --seed (legacy behavior, NOT apples-to-apples). "
                        "For fair 5-seed runs, pass a fixed value like 42.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-config-checkpoint", type=str, default=None,
                        help="Use model-shape config from this checkpoint before applying CLI overrides.")
    parser.add_argument("--init-checkpoint", type=str, default=None,
                        help="Shape-matched checkpoint initialization before training.")
    parser.add_argument("--init-min-match-ratio", type=float, default=0.8,
                        help="Minimum fraction of model tensors that must be initialized.")
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--noise-dim", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--K", type=int, default=None, dest="K")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--lr-schedule", choices=["none", "warmup_cosine"], default=None)
    parser.add_argument("--warmup-frac", type=float, default=None)
    parser.add_argument("--ema-decay", type=float, default=None,
                        help="Maintain EMA during training and load EMA weights before evaluation.")
    parser.add_argument("--down-n-mc", type=int, default=None,
                        help="Number of MC noise samples for DiT predict_mean in L_down.")
    parser.add_argument("--down-antithetic", action="store_true",
                        help="Use antithetic epsilon pairs for DiT predict_mean in L_down.")
    parser.add_argument("--lambda-drift", type=float, default=None)
    parser.add_argument("--lambda-down", type=float, default=None)
    parser.add_argument("--loss-balancer",
                        choices=["fixed", "uncertainty", "relobralo", "dwa", "gradnorm_lite", "rlw"],
                        default=None, help="Loss balancing strategy for MMD/W2/drift/down components.")
    parser.add_argument("--loss-balancer-temperature", type=float, default=None)
    parser.add_argument("--loss-balancer-lookback-prob", type=float, default=None)
    parser.add_argument("--loss-balancer-alpha", type=float, default=None)
    parser.add_argument("--loss-balancer-max-multiplier", type=float, default=None)
    parser.add_argument("--dit-size", choices=sorted(DRIFT_DIT_1D_MODELS), default=None,
                        help="DiT backbone size for m9/m10. Default: cfg['dit_size'] or tiny.")
    parser.add_argument("--waddington-dit", action="store_true",
                        help="Use explicit Waddington-DiT residual for m9/m10.")
    parser.add_argument("--curl-rank", type=int, default=None,
                        help="Low-rank curl rank for Waddington residual.")
    parser.add_argument("--wdit-curl-update",
                        choices=["additive", "cayley_direct", "cayley_residual",
                                 "hybrid_delta", "hard_delta_cayley_residual"],
                        default=None,
                        help="Curl update mode for Waddington-DiT.")
    parser.add_argument("--wdit-curl-time-mode", choices=["full", "state_only"], default=None,
                        help="Whether W-DiT curl receives Delta or is state-only.")
    parser.add_argument("--wdit-hybrid-delta0", type=float, default=None,
                        help="Delta midpoint for hybrid additive/Cayley curl gate.")
    parser.add_argument("--wdit-hybrid-slope", type=float, default=None,
                        help="Slope for hybrid additive/Cayley curl gate.")
    parser.add_argument("--wdit-hard-delta0", type=float, default=None,
                        help="Delta threshold for hard additive-to-Cayley residual switching.")
    parser.add_argument("--wdit-time-embedding",
                        choices=["legacy_fourier", "bounded_lowfreq_fourier", "time2vec"],
                        default=None, help="Delta embedding mode for Waddington-DiT.")
    parser.add_argument("--wdit-time-delta-transform", choices=["normalized", "log1p"], default=None,
                        help="Delta scaling transform for non-legacy Waddington-DiT time embeddings.")
    parser.add_argument("--wdit-time-delta-scale", type=float, default=None,
                        help="Manual Delta scale for non-legacy Waddington-DiT time embeddings.")
    parser.add_argument("--wdit-curl-time-embedding",
                        choices=["same", "legacy_fourier", "bounded_lowfreq_fourier", "time2vec"],
                        default=None, help="Separate curl-branch Delta embedding mode for Waddington-DiT.")
    parser.add_argument("--wdit-curl-time-delta-transform", choices=["normalized", "log1p"], default=None,
                        help="Delta scaling transform for separate curl-branch time embedding.")
    parser.add_argument("--wdit-curl-time-delta-scale", type=float, default=None,
                        help="Manual Delta scale for separate curl-branch time embedding.")
    parser.add_argument("--lambda-wdit-a-fro", type=float, default=None,
                        help="Frobenius regularization weight for W-DiT A factors.")
    parser.add_argument("--lambda-wdit-curl", type=float, default=None,
                        help="Curl vector norm regularization weight for W-DiT.")
    parser.add_argument("--drift-pos-ratio", type=float, default=None,
                        help="If set, sample this many positives per generated negative for L_drift. "
                        "Example: 1.0 => N_pos=B*K, 2.0 => N_pos=2*B*K. "
                        "Unset preserves historical N_pos=B behavior.")
    parser.add_argument("--drift-balance-sample-counts", action="store_true",
                        help="Subtract log(N_pos)/log(N_neg) from pos/neg logits in L_drift "
                        "so unequal empirical supports represent equal-mass distributions.")
    parser.add_argument("--state-chunk-dim", type=int, default=None,
                        help="DiT v2: split dim into multiple state tokens (m9/m10 only). "
                        "None=legacy 1-token. Recommended: 8 for 64-dim latent.")
    parser.add_argument("--learned-state-tokens", type=int, default=None,
                        help="Use dense learned tokenizer Linear(dim -> S*H) with S state tokens (m9/m10 only).")
    parser.add_argument("--disable-rope", action="store_true",
                        help="Disable RoPE in DriftDiT1D attention (m9/m10 only).")
    parser.add_argument("--save-checkpoint", action="store_true",
                        help="Save final model state_dict to output-dir/checkpoint_final.pt.")
    parser.add_argument("--multi-delta", action="store_true",
                        help="Multi-Δ joint training: each step samples uniformly from "
                        "all (source_t, target_t) pairs with source_t < target_t. "
                        "Requires adapter.timepoints with >=3 timepoints (e.g. Weinreb d2/d4/d6 → "
                        "3 transitions: d2→d4, d2→d6, d4→d6).")
    parser.add_argument("--md-endpoint-prob", type=float, default=None,
                        help="For --multi-delta, set probability of endpoint pair "
                        "(first_timepoint, last_timepoint); remaining pairs share the rest.")
    parser.add_argument("--log-every", type=int, default=50)
    add_wandb_args(parser)
    args = parser.parse_args()

    cfg = dict(DATASET_CONFIGS[args.dataset])
    try:
        recipe_method, recipe_epochs, recipe_save_checkpoint = apply_experiment_args(args, cfg)
    except ValueError as exc:
        parser.error(str(exc))
    args.method = recipe_method
    model_config_updates = {}
    if args.model_config_checkpoint:
        model_config_updates = apply_model_config_from_checkpoint(cfg, args.model_config_checkpoint)
        print(f"Applied model config from {args.model_config_checkpoint}: {sorted(model_config_updates)}")
    apply_common_overrides(args, cfg)
    epochs = args.epochs if args.epochs is not None else (
        recipe_epochs if recipe_epochs is not None else cfg["default_epochs"]
    )
    save_checkpoint = bool(args.save_checkpoint or recipe_save_checkpoint)

    pcs = args.pcs if args.pcs is not None else DEFAULT_PCS[args.dataset]

    output_dir = args.output_dir or f"output/benchmark/{args.method}_{args.dataset}_pcs{pcs}_seed{args.seed}"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    split_seed = args.split_seed if args.split_seed is not None else args.seed
    adapter = build_adapter(args.dataset, pcs=pcs, seed=split_seed)
    train_batch = adapter.get_transition(split="train")
    print(f"Dataset: {args.dataset} (dim={adapter.dim}) method={args.method}")
    print(f"  Train src: {train_batch.source.shape}, target: {train_batch.target.shape}")
    print(f"  Delta: {train_batch.delta}  |  split_seed={split_seed}, model_seed={args.seed}")

    tau_init = train_batch.delta / np.log(2)
    set_model_seed(args.seed)
    model = build_model(args.method, adapter.dim, cfg, float(tau_init)).to(device)
    init_info = None
    if args.init_checkpoint:
        init_info = load_shape_matched_checkpoint(
            model,
            args.init_checkpoint,
            min_match_ratio=float(args.init_min_match_ratio),
        )
        print(f"Loaded init checkpoint: {init_info}")
    n_params = sum(p.numel() for p in model.parameters())
    model_names = {"m1": "BR-CellDrift-MLP", "m2": "PC-CellDrift-MLP",
                   "m7": "BR+Drift", "m8": "PC+Drift+Down",
                   "m9": "DriftDiT-1D", "m10": "DiT+Drift+Down"}
    print(f"Model: {model_names[args.method]}, {n_params:,} params, tau_init={tau_init:.3f}")
    if args.method in ("m9", "m10"):
        print("  DiT detail: "
              f"dit_size={cfg.get('dit_size', 'tiny')}, "
              f"waddington_dit={bool(cfg.get('waddington_dit', False))}, "
              f"state_chunk_dim={cfg.get('state_chunk_dim')}, "
              f"learned_state_tokens={cfg.get('learned_state_tokens')}, "
              f"use_rope={not bool(cfg.get('disable_rope', False))}, "
              f"num_state_tokens={getattr(model, 'num_state_tokens', 'na')}, "
              f"tokenizer_mode={getattr(model, 'tokenizer_mode', 'na')}, "
              f"hidden_dim={getattr(model, 'hidden_dim', 'na')}")

    # Shift-magnitude diagnostic (GPT review): compute alpha(Δ) and initial shift ratio
    with torch.no_grad():
        delta_t = torch.full((1,), train_batch.delta)
        if hasattr(model, "alpha_gate"):
            alpha_at_delta = float(model.alpha_gate(delta_t.to(device)).item())
        else:
            alpha_at_delta = float("nan")
    try:
        src_np = train_batch.source.numpy()
        tgt_np = train_batch.target.numpy()
        shift_norm_data = float(np.linalg.norm(tgt_np.mean(0) - src_np.mean(0)))
    except Exception:
        shift_norm_data = float("nan")
    print(f"  alpha(Δ={train_batch.delta}) = {alpha_at_delta:.3f}  |  "
          f"src→tgt center shift = {shift_norm_data:.3f} units")

    wandb_config = {
        "script": "run_benchmark",
        "args": vars(args),
        "cfg": cfg,
        "dataset": args.dataset,
        "method": args.method,
        "split_seed": split_seed,
        "model_seed": args.seed,
        "pcs": pcs,
        "epochs": epochs,
        "n_params": int(n_params),
        "tau_init": float(tau_init),
    }
    wandb_run = maybe_init_wandb(
        args,
        config=wandb_config,
        output_dir=out_path,
        default_name=f"{args.method}-{args.dataset}-seed{args.seed}",
        default_group=f"{args.method}-{args.dataset}",
    )

    def log_train_step(info: dict) -> None:
        step = int(info.get("epoch", 0))
        metrics = {f"train/{k}": v for k, v in info.items()}
        if wandb_run is not None:
            wandb_run.log(flatten_numeric(metrics), step=step)

    print(f"\nTraining for {epochs} epochs ...")
    t0 = time.time()
    history = train_method(args.method, adapter, model, device, cfg,
                           epochs=epochs, seed=args.seed, log_every=args.log_every,
                           log_callback=log_train_step)
    train_time = time.time() - t0
    print(f"Train time: {train_time:.1f}s")

    # Post-train shift diagnostic
    model.eval()
    with torch.no_grad():
        test_batch = adapter.get_transition(split="test")
        src_test = test_batch.source.to(device)
        delta_t = torch.full((src_test.shape[0],), test_batch.delta, device=device, dtype=src_test.dtype)
        K_diag = min(cfg["K"], 4)
        eps = torch.randn(src_test.shape[0], K_diag, adapter.dim, device=device)
        preds = model(src_test, delta_t, eps).reshape(-1, adapter.dim).cpu().numpy()
        src_test_np = src_test.cpu().numpy()
        tgt_np = test_batch.target.numpy()
        pred_center = preds.mean(0)
        src_center = src_test_np.mean(0)
        tgt_center = tgt_np.mean(0)
        actual_shift = float(np.linalg.norm(pred_center - src_center))
        ideal_shift = float(np.linalg.norm(tgt_center - src_center))
        shift_ratio = actual_shift / max(ideal_shift, 1e-8)
        # Direction cosine
        actual_vec = pred_center - src_center
        ideal_vec = tgt_center - src_center
        cos = float(np.dot(actual_vec, ideal_vec) / (
            np.linalg.norm(actual_vec) * np.linalg.norm(ideal_vec) + 1e-12))
    shift_diag = {
        "alpha_at_delta": alpha_at_delta,
        "src_to_tgt_shift_data": shift_norm_data,
        "actual_shift_mag": actual_shift,
        "ideal_shift_mag": ideal_shift,
        "shift_ratio": shift_ratio,
        "direction_cosine": cos,
    }
    print(f"\n[Shift diagnostic] alpha={alpha_at_delta:.3f}, "
          f"actual_shift={actual_shift:.3f}, ideal_shift={ideal_shift:.3f}, "
          f"ratio={shift_ratio:.3f}, cos={cos:.3f}")

    print(f"\nEvaluating ...")
    results = evaluate_all(adapter, model, device, cfg, args.dataset, args.method, seed=args.seed)
    print_summary_table(results, args.dataset, args.method)
    if wandb_run is not None:
        final_metrics = {
            "train/train_time_s": float(train_time),
            "shift": shift_diag,
            "eval": results,
        }
        wandb_run.log(flatten_numeric(final_metrics), step=epochs)
        wandb_summary_update(wandb_run, final_metrics)

    ckpt_file = None
    if save_checkpoint:
        ckpt_file = out_path / "checkpoint_final.pt"
        save_model_checkpoint(
            ckpt_file,
            model,
            method=args.method,
            dataset=args.dataset,
            cfg=cfg,
            pcs=pcs,
            epochs=epochs,
            split_seed=split_seed,
            model_seed=args.seed,
            n_params=int(n_params),
            tau_init=float(tau_init),
            init_checkpoint=args.init_checkpoint,
            init_info=init_info,
            model_config_checkpoint=args.model_config_checkpoint,
            model_config_updates=model_config_updates,
        )
        print(f"Saved checkpoint to {ckpt_file}")

    out_file = out_path / "results.json"
    with open(out_file, "w") as f:
        json.dump({
            "args": vars(args), "method": args.method, "cfg": cfg, "pcs": pcs, "epochs": epochs,
            "tau_init": float(tau_init), "n_params": int(n_params),
            "split_seed": split_seed, "model_seed": args.seed,
            "shift_diagnostic": shift_diag,
            "checkpoint": str(ckpt_file) if ckpt_file is not None else None,
            "init_checkpoint": args.init_checkpoint,
            "init_info": init_info,
            "model_config_checkpoint": args.model_config_checkpoint,
            "model_config_updates": model_config_updates,
            "wandb": wandb_run_info(wandb_run),
            "train_time_s": float(train_time), "history": history, "eval": results,
        }, f, indent=2, default=str)
    print(f"\nSaved to {out_file}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
