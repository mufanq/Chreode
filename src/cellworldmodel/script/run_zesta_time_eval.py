"""Train/evaluate CellWorldModel methods on ZESTA control-only time extrapolation.

Task:
  source: control cells at t=18
  train targets: control t=24/36/48
  OOD target: control t=72

This is intentionally separate from `run_intermediate_eval.py` because the
training target set is not the final endpoint only: each step samples one
training target time, while evaluation reports all train targets and OOD t=72.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cellworldmodel.benchmark.experiment_registry import add_experiment_arg, apply_experiment_args
from cellworldmodel.benchmark.config_overrides import apply_common_overrides
from cellworldmodel.benchmark.zesta_time_adapter import ZestaTimeAdapter
from cellworldmodel.benchmark.registry import build_model
from cellworldmodel.script.wandb_utils import (
    add_wandb_args,
    flatten_numeric,
    maybe_init_wandb,
    wandb_run_info,
    wandb_summary_update,
)
from cellworldmodel.training.benchmark_loop import train_method
from cellworldmodel.training.checkpointing import save_model_checkpoint
from cellworldmodel.training.transition_sampler import TimepointTransitionSampler, all_ordered_pairs
from cellworldmodel.evaluation.zesta import evaluate_zesta


METHODS = {"m1", "m2", "m7", "m8", "m9", "m10"}


def set_model_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_cfg() -> dict:
    return {
        "hidden_dim": 512,
        "n_layers": 3,
        "noise_dim": 32,
        "time_emb_dim": 64,
        "batch_size": 256,
        "K": 8,
        "lr": 3e-4,
        "lambda_mmd": 1.0,
        "lambda_w2": 1.0,
        "lambda_drift": 1.0,
        "lambda_down": 0.1,
        "sinkhorn_eps": 0.05,
        "grad_clip": 1.0,
        "dit_size": "tiny",
        "multi_delta": False,
        "md_endpoint_prob": None,
    }



def main() -> None:
    parser = argparse.ArgumentParser()
    add_experiment_arg(parser)
    parser.add_argument("--method", required=False, choices=sorted(METHODS))
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--rep-key", default="X_aligned")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--K", type=int, default=None, dest="K")
    parser.add_argument("--dit-size", choices=["tiny", "small", "base"], default=None)
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
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--lr-schedule", choices=["none", "warmup_cosine"], default=None)
    parser.add_argument("--warmup-frac", type=float, default=None)
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument("--down-n-mc", type=int, default=None)
    parser.add_argument("--down-antithetic", action="store_true")
    parser.add_argument("--loss-balancer",
                        choices=["fixed", "uncertainty", "relobralo", "dwa", "gradnorm_lite", "rlw"],
                        default=None, help="Loss balancing strategy for MMD/W2/drift/down components.")
    parser.add_argument("--loss-balancer-temperature", type=float, default=None)
    parser.add_argument("--loss-balancer-lookback-prob", type=float, default=None)
    parser.add_argument("--loss-balancer-alpha", type=float, default=None)
    parser.add_argument("--loss-balancer-max-multiplier", type=float, default=None)
    parser.add_argument("--state-chunk-dim", type=int, default=None)
    parser.add_argument("--learned-state-tokens", type=int, default=None)
    parser.add_argument("--disable-rope", action="store_true")
    parser.add_argument("--drift-pos-ratio", type=float, default=None)
    parser.add_argument("--drift-balance-sample-counts", action="store_true")
    parser.add_argument(
        "--multi-delta",
        action="store_true",
        help="Train on all ordered pairs among source_t and train_target_times instead of source_t-only pairs.",
    )
    parser.add_argument("--md-endpoint-prob", type=float, default=None,
                        help="For --multi-delta, probability of source_t→last train target pair.")
    parser.add_argument("--split-policy", choices=["legacy", "per_timepoint"], default=None,
                        help="Training split policy. Named experiments may set this.")
    parser.add_argument("--max-cells-per-timepoint", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--skip-ot-oracle", action="store_true",
                        help="Skip exact OT oracle baseline during evaluation (much faster for ZESTA).")
    parser.add_argument("--cellflow-metrics-only", action="store_true",
                        help="Only compute CellFlow-style R2/energy/scalar-MMD metrics.")
    add_wandb_args(parser)
    args = parser.parse_args()

    cfg = default_cfg()
    try:
        recipe_method, recipe_epochs, recipe_save_checkpoint = apply_experiment_args(args, cfg)
    except ValueError as exc:
        parser.error(str(exc))
    args.method = recipe_method
    epochs = args.epochs if args.epochs is not None else (
        recipe_epochs if recipe_epochs is not None else 500
    )
    apply_common_overrides(args, cfg)
    cfg.setdefault("split_policy", "per_timepoint")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = ZestaTimeAdapter(
        data_path=args.data_path,
        rep_key=args.rep_key,
        seed=args.split_seed,
        max_cells_per_timepoint=args.max_cells_per_timepoint,
    )
    print(f"ZESTA rep={args.rep_key} dim={adapter.dim}")
    for t, arr in adapter.coords_by_t.items():
        print(f"  control t={t:g}: {arr.shape}")

    median_delta = float(np.median([t - adapter.source_t for t in adapter.train_target_times]))
    tau_init = median_delta / np.log(2)
    set_model_seed(args.seed)
    model = build_model(args.method, adapter.dim, cfg, tau_init).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model {args.method}: {n_params:,} params, tau_init={tau_init:.3f}")

    if cfg.get("split_policy") != "per_timepoint":
        raise ValueError("run_zesta_time_eval now uses the shared sampler loop; use split_policy=per_timepoint")
    train_timepoints = [adapter.source_t] + list(adapter.train_target_times)
    sampler = TimepointTransitionSampler(
        adapter,
        split_seed=args.split_seed,
        pairs=all_ordered_pairs(train_timepoints) if cfg.get("multi_delta", False)
        else [(adapter.source_t, float(t)) for t in adapter.train_target_times],
        endpoint_prob=cfg.get("md_endpoint_prob"),
        endpoint_pair=(adapter.source_t, float(adapter.train_target_times[-1])),
        split_ratios=tuple(cfg.get("split_ratios", (0.7, 0.1, 0.2))),
        reference_target_times=adapter.train_target_times,
    )
    print(f"Split policy: per_timepoint ratios={cfg.get('split_ratios', (0.7, 0.1, 0.2))}")

    wandb_config = {
        "script": "run_zesta_time_eval",
        "args": vars(args),
        "cfg": cfg,
        "dataset": "zesta",
        "method": args.method,
        "split_seed": args.split_seed,
        "model_seed": args.seed,
        "epochs": epochs,
        "n_params": int(n_params),
        "tau_init": float(tau_init),
        "source_t": float(adapter.source_t),
        "train_target_times": [float(t) for t in adapter.train_target_times],
        "target_times": [float(t) for t in adapter.target_times],
    }
    wandb_run = maybe_init_wandb(
        args,
        config=wandb_config,
        output_dir=out,
        default_name=f"zesta-{args.method}-seed{args.seed}",
        default_group=f"zesta-{args.method}",
    )

    def log_train_step(info: dict) -> None:
        if wandb_run is None:
            return
        step = int(info.get("epoch", 0))
        wandb_run.log(flatten_numeric({f"train/{k}": v for k, v in info.items()}), step=step)

    t0 = time.time()
    history = train_method(
        args.method, adapter, model, device, cfg, epochs, args.seed,
        args.log_every, log_callback=log_train_step, sampler=sampler,
    )
    train_time = time.time() - t0
    print(f"Train time: {train_time:.1f}s")
    results = evaluate_zesta(adapter, model, device, cfg, seed=args.seed,
                             include_ot_oracle=not args.skip_ot_oracle,
                             cellflow_metrics_only=args.cellflow_metrics_only,
                             sampler=sampler)
    if wandb_run is not None:
        final_metrics = {
            "train": {"train_time_s": float(train_time)},
            "eval": {"zesta": results},
        }
        wandb_run.log(flatten_numeric(final_metrics), step=epochs)
        wandb_summary_update(wandb_run, final_metrics)

    ckpt = None
    save_checkpoint = bool(args.save_checkpoint or recipe_save_checkpoint)
    if save_checkpoint:
        ckpt = out / "checkpoint_final.pt"
        save_model_checkpoint(
            ckpt,
            model,
            method=args.method,
            dataset="zesta",
            cfg=cfg,
            rep_key=args.rep_key,
            epochs=epochs,
            seed=args.seed,
            split_seed=args.split_seed,
            n_params=int(n_params),
            tau_init=float(tau_init),
        )

    with open(out / "results.json", "w") as f:
        json.dump({
            "method": args.method,
            "dataset": "zesta",
            "seed": args.seed,
            "split_seed": args.split_seed,
            "args": vars(args),
            "cfg": cfg,
            "epochs": epochs,
            "n_params": int(n_params),
            "tau_init": float(tau_init),
            "checkpoint": str(ckpt) if ckpt else None,
            "wandb": wandb_run_info(wandb_run),
            "train_time_s": float(train_time),
            "history": history,
            "eval": results,
        }, f, indent=2, default=str)
    print(f"Saved to {out / 'results.json'}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
