"""Train + evaluate at intermediate timepoints (Mouse t=1, Veres t=1..6).

Paper protocol:
  - Mouse: Table 3 reports BranchSBM at t=1 (intermediate) AND t=2 (endpoint).
    Our run_benchmark.py tests only t=2 (endpoint). This script adds t=1.
  - Veres: Table 2 reports BranchSBM at t=1..t=7 (all 7 timepoints, full 30D W1).
    Our run_benchmark.py tests only t=7 (endpoint). This script adds t=1..t=6.

One-step generator extrapolates to intermediate Δ via the α(Δ) gate:
    ẑ = z + α(Δ) · R_θ(z, Δ, ε)
The model was trained only on the endpoint Δ (2 for Mouse, 7 for Veres).
Intermediate Δ ∈ (0, endpoint) are out-of-training-range inputs.

Usage:
    PYTHONPATH=src python -m cellworldmodel.script.run_intermediate_eval \\
        --method m1 --dataset mouse --seed 0

Output JSON (output/intermediate_eval/{method}_{dataset}_seed{N}/):
  {
    "endpoint": {...dual-protocol metrics at t=final...},
    "intermediate": {
      "t=1": {...metrics...},
      "t=3": {...},
      ...
    }
  }
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cellworldmodel.benchmark.configs import DATASET_CONFIGS
from cellworldmodel.benchmark.config_overrides import apply_common_overrides
from cellworldmodel.benchmark.experiment_registry import add_experiment_arg, apply_experiment_args
from cellworldmodel.benchmark.registry import build_adapter, build_model
from cellworldmodel.training.benchmark_loop import train_method
from cellworldmodel.training.checkpointing import (
    apply_model_config_from_checkpoint,
    load_shape_matched_checkpoint,
    save_model_checkpoint,
)
from cellworldmodel.training.transition_sampler import TimepointTransitionSampler
from cellworldmodel.benchmark.common_metrics import mmd2_unbiased_multi_sigma, sinkhorn_w2
from cellworldmodel.evaluation.intermediate import evaluate_intermediate
from cellworldmodel.evaluation.prediction import predict_at_delta
from cellworldmodel.script.wandb_utils import (
    add_wandb_args,
    flatten_numeric,
    maybe_init_wandb,
    wandb_run_info,
    wandb_summary_update,
)


def set_model_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _subsample_fixed(x: torch.Tensor, max_n: int, *, seed: int) -> torch.Tensor:
    if x.shape[0] <= int(max_n):
        return x
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    idx = torch.randperm(x.shape[0], generator=generator)[:int(max_n)]
    return x[idx]


def _build_validation_callback(model, adapter, sampler: TimepointTransitionSampler | None,
                               device, cfg: dict, args, *, split: str = "val"):
    if sampler is None:
        return None
    source_t = float(adapter.timepoints[0])
    source = _subsample_fixed(
        sampler.get_population(source_t, split=split),
        int(args.validation_source_subsample),
        seed=int(args.seed) + 9101,
    ).to(device)
    targets = {}
    for i, t in enumerate(adapter.timepoints):
        t = float(t)
        if t == source_t:
            continue
        targets[t] = _subsample_fixed(
            sampler.get_population(t, split=split),
            int(args.validation_max_target),
            seed=int(args.seed) + 9201 + i,
        ).to(device)

    def validate(epoch: int) -> dict:
        rows = []
        for t, target in targets.items():
            delta = float(t - source_t)
            preds = predict_at_delta(model, source, delta, cfg["K"], adapter.dim, device)
            if preds.shape[0] > int(args.validation_max_preds):
                generator = torch.Generator(device=preds.device)
                generator.manual_seed(int(args.seed) + int(epoch) + int(t * 1000))
                idx = torch.randperm(preds.shape[0], device=preds.device, generator=generator)[
                    :int(args.validation_max_preds)
                ]
                preds = preds[idx]
            w2_value = sinkhorn_w2(
                preds,
                target,
                epsilon=float(cfg["sinkhorn_eps"]),
                num_iters=50,
            )
            mmd_value = mmd2_unbiased_multi_sigma(preds, target)
            rows.append({
                "target_t": float(t),
                "w2": float(w2_value.item()),
                "mmd": float(mmd_value.item()),
            })
        w2_values = [row["w2"] for row in rows]
        mmd_values = [row["mmd"] for row in rows]
        return {
            "val_w2_mean": float(np.mean(w2_values)),
            "val_mmd_mean": float(np.mean(mmd_values)),
            "val_objective": float(np.mean(w2_values) + np.mean(mmd_values)),
            "val_targets": rows,
            "val_split": split,
        }

    return validate


def _best_validation_record(history: list[dict]) -> dict | None:
    for item in reversed(history):
        if item.get("event") == "best_validation":
            return item
    return None


def _load_growth_scores_by_t(path: str | None, adapter) -> dict[float, np.ndarray] | None:
    if not path:
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    weights = payload.get("w")
    if not isinstance(weights, (list, tuple)):
        raise ValueError(f"Growth file {path} does not contain list-like 'w'")
    if len(weights) != len(adapter.timepoints):
        raise ValueError(f"Growth file has {len(weights)} timepoints, adapter has {len(adapter.timepoints)}")
    out = {}
    for t, arr in zip(adapter.timepoints, weights):
        arr = np.asarray(arr, dtype=np.float32)
        expected = adapter.coords_by_t[float(t)].shape[0]
        if arr.shape[0] != expected:
            raise ValueError(
                f"Growth weights for t={t} have length {arr.shape[0]}, expected {expected}. "
                "Use a full-length growth file such as Weinreb2020_growth-all_kegg.pt."
            )
        out[float(t)] = arr
    return out


def main():
    parser = argparse.ArgumentParser()
    add_experiment_arg(parser)
    parser.add_argument("--method", required=False, choices=["m1", "m2", "m7", "m8", "m9", "m10"])
    parser.add_argument("--dataset", required=True,
                        choices=[
                            "mouse", "veres", "weinreb_scvi", "weinreb_hvg", "veres_scvi",
                            "paper_weinreb_scvi128", "paper_veres_scvi128",
                        ])
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--pcs", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-config-checkpoint", type=str, default=None,
                        help="Use model-shape config from this checkpoint before applying CLI overrides.")
    parser.add_argument("--init-checkpoint", type=str, default=None,
                        help="Shape-matched checkpoint initialization before training.")
    parser.add_argument("--init-min-match-ratio", type=float, default=0.8,
                        help="Minimum fraction of model tensors that must be initialized.")
    parser.add_argument("--log-every", type=int, default=100)
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
    parser.add_argument("--loss-balancer",
                        choices=["fixed", "uncertainty", "relobralo", "dwa", "gradnorm_lite", "rlw"],
                        default=None, help="Loss balancing strategy for MMD/W2/drift/down components.")
    parser.add_argument("--loss-balancer-temperature", type=float, default=None)
    parser.add_argument("--loss-balancer-lookback-prob", type=float, default=None)
    parser.add_argument("--loss-balancer-alpha", type=float, default=None)
    parser.add_argument("--loss-balancer-max-multiplier", type=float, default=None)
    parser.add_argument("--state-chunk-dim", type=int, default=None,
                        help="DiT v2 tokenization: dim-split into state tokens (m9/m10 only)")
    parser.add_argument("--learned-state-tokens", type=int, default=None,
                        help="Use dense learned tokenizer Linear(dim -> S*H) with S state tokens (m9/m10 only).")
    parser.add_argument("--disable-rope", action="store_true",
                        help="Disable RoPE in DriftDiT1D attention (m9/m10 only).")
    parser.add_argument("--dit-size", choices=["tiny", "small", "base"], default=None,
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
                        "Unset preserves historical N_pos=B behavior.")
    parser.add_argument("--drift-balance-sample-counts", action="store_true",
                        help="Subtract log(N_pos)/log(N_neg) from pos/neg logits in L_drift.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--K", type=int, default=None, dest="K")
    parser.add_argument("--save-checkpoint", action="store_true",
                        help="Save final model state_dict to output-dir/checkpoint_final.pt.")
    parser.add_argument("--val-every", type=int, default=0,
                        help="Run validation every N epochs. 0 disables validation/early stopping.")
    parser.add_argument("--validation-metric", default="val_w2_mean",
                        choices=["val_w2_mean", "val_mmd_mean", "val_objective"])
    parser.add_argument("--validation-source-subsample", type=int, default=512)
    parser.add_argument("--validation-max-target", type=int, default=1024)
    parser.add_argument("--validation-max-preds", type=int, default=1024)
    parser.add_argument("--early-stopping-patience", type=int, default=None,
                        help="Stop after this many epochs without validation improvement.")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--no-restore-best-validation", action="store_true",
                        help="Keep final weights instead of restoring the validation-best state.")
    parser.add_argument("--skip-intermediate-eval", action="store_true",
                        help="Only train/save checkpoint; useful when a downstream evaluator is run separately.")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="Save training-curve checkpoints every N epochs under output-dir/checkpoints.")
    parser.add_argument("--growth-weight-path", type=str, default=None,
                        help="Optional PRESCIENT-style full-length growth score file.")
    parser.add_argument("--growth-weight-scale", type=float, default=1.0,
                        help="Scale applied in exp(scale * Delta * growth_score).")
    parser.add_argument("--multi-delta", action="store_true",
                        help="Multi-Δ joint training (see run_benchmark.py --multi-delta).")
    parser.add_argument("--md-endpoint-prob", type=float, default=None,
                        help="For --multi-delta, set probability of endpoint pair.")
    parser.add_argument("--split-policy", choices=["legacy", "per_timepoint"], default=None,
                        help="Training/eval split policy. Named experiments may set this.")
    add_wandb_args(parser)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    epochs = args.epochs if args.epochs is not None else (
        recipe_epochs if recipe_epochs is not None else cfg["default_epochs"]
    )
    apply_common_overrides(args, cfg)
    from cellworldmodel.benchmark.configs import DEFAULT_PCS
    pcs = args.pcs if args.pcs is not None else DEFAULT_PCS.get(args.dataset, 50)

    output_dir = args.output_dir or f"output/intermediate_eval/{args.method}_{args.dataset}_seed{args.seed}"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    split_seed = args.split_seed if args.split_seed is not None else args.seed
    adapter = build_adapter(args.dataset, pcs=pcs, seed=split_seed)
    train_batch = adapter.get_transition(split="train")
    print(f"Dataset: {args.dataset} (dim={adapter.dim}, timepoints={adapter.timepoints})")
    print(f"Trained Δ (endpoint) = {train_batch.delta}")

    tau_init = float(train_batch.delta / np.log(2))
    set_model_seed(args.seed)
    model = build_model(args.method, adapter.dim, cfg, tau_init).to(device)
    init_info = None
    if args.init_checkpoint:
        init_info = load_shape_matched_checkpoint(
            model,
            args.init_checkpoint,
            min_match_ratio=float(args.init_min_match_ratio),
        )
        print(f"Loaded init checkpoint: {init_info}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.method}, {n_params:,} params")
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

    sampler = None
    if cfg.get("split_policy") == "per_timepoint":
        growth_scores_by_t = _load_growth_scores_by_t(args.growth_weight_path, adapter)
        sampler = TimepointTransitionSampler(
            adapter,
            split_seed=split_seed,
            endpoint_prob=cfg.get("md_endpoint_prob"),
            split_ratios=tuple(cfg.get("split_ratios", (0.7, 0.1, 0.2))),
            growth_scores_by_t=growth_scores_by_t,
            growth_weight_scale=float(args.growth_weight_scale),
        )
        print(f"Split policy: per_timepoint ratios={cfg.get('split_ratios', (0.7, 0.1, 0.2))}")
        if growth_scores_by_t is not None:
            print(f"Growth weights: {args.growth_weight_path} scale={args.growth_weight_scale}")
    else:
        print("Split policy: legacy")

    wandb_config = {
        "script": "run_intermediate_eval",
        "args": vars(args),
        "cfg": cfg,
        "dataset": args.dataset,
        "method": args.method,
        "split_seed": split_seed,
        "model_seed": args.seed,
        "pcs": pcs,
        "epochs": epochs,
        "n_params": int(n_params),
        "tau_init": tau_init,
        "timepoints": [float(t) for t in adapter.timepoints],
        "growth_weight_path": args.growth_weight_path,
        "growth_weight_scale": args.growth_weight_scale,
    }
    wandb_run = maybe_init_wandb(
        args,
        config=wandb_config,
        output_dir=out_path,
        default_name=f"inter-{args.method}-{args.dataset}-seed{args.seed}",
        default_group=f"inter-{args.method}-{args.dataset}",
    )

    def log_train_step(info: dict) -> None:
        if wandb_run is None:
            return
        step = int(info.get("epoch", 0))
        wandb_run.log(flatten_numeric({f"train/{k}": v for k, v in info.items()}), step=step)

    print(f"\nTraining {args.method} for {epochs} epochs (seed {args.seed}) ...")
    validation_callback = _build_validation_callback(model, adapter, sampler, device, cfg, args)

    def checkpoint_metadata(extra: dict | None = None) -> dict:
        payload = dict(
            method=args.method,
            dataset=args.dataset,
            cfg=cfg,
            epochs=epochs,
            pcs=pcs,
            n_params=int(n_params),
            split_seed=split_seed,
            model_seed=args.seed,
            tau_init=tau_init,
            init_checkpoint=args.init_checkpoint,
            init_info=init_info,
            model_config_checkpoint=args.model_config_checkpoint,
            model_config_updates=model_config_updates,
        )
        if extra:
            payload.update(extra)
        return payload

    def save_curve_checkpoint(ep: int, model_obj, info: dict) -> None:
        ckpt_dir = out_path / "checkpoints"
        ckpt_file = ckpt_dir / f"checkpoint_epoch_{int(ep) + 1:05d}.pt"
        save_model_checkpoint(
            ckpt_file,
            model_obj,
            **checkpoint_metadata({
                "curve_epoch": int(ep) + 1,
                "curve_checkpoint_info": info,
            }),
        )
        print(f"Saved curve checkpoint to {ckpt_file}")

    t0 = time.time()
    history = train_method(args.method, adapter, model, device, cfg, epochs=epochs,
                           seed=args.seed, log_every=args.log_every,
                           log_callback=log_train_step,
                           sampler=sampler,
                           validation_callback=validation_callback,
                           validation_every=args.val_every,
                           validation_metric=args.validation_metric,
                           early_stopping_patience=args.early_stopping_patience,
                           early_stopping_min_delta=args.early_stopping_min_delta,
                           restore_best_validation=not args.no_restore_best_validation,
                           checkpoint_callback=save_curve_checkpoint if args.checkpoint_every > 0 else None,
                           checkpoint_every=args.checkpoint_every)
    train_time = time.time() - t0
    best_validation = _best_validation_record(history)
    print(f"Train time: {train_time:.1f}s")

    if args.skip_intermediate_eval:
        print("\nSkipping intermediate/test evaluation.")
        results = {}
    else:
        print(f"\nEvaluating at each timepoint (intermediate + endpoint) ...")
        results = evaluate_intermediate(model, adapter, device, cfg, seed=args.seed, sampler=sampler)
    if wandb_run is not None:
        final_metrics = {
            "train": {"train_time_s": float(train_time)},
            "eval": {"intermediate": results},
        }
        wandb_run.log(flatten_numeric(final_metrics), step=epochs)
        wandb_summary_update(wandb_run, final_metrics)

    ckpt_file = None
    save_checkpoint = bool(args.save_checkpoint or recipe_save_checkpoint)
    if save_checkpoint:
        ckpt_file = out_path / "checkpoint_final.pt"
        checkpoint_metadata_final = checkpoint_metadata({"best_validation": best_validation})
        save_model_checkpoint(ckpt_file, model, **checkpoint_metadata_final)
        print(f"Saved checkpoint to {ckpt_file}")
        if best_validation is not None:
            best_ckpt_file = out_path / "checkpoint_best_val.pt"
            save_model_checkpoint(best_ckpt_file, model, **checkpoint_metadata_final)
            print(f"Saved validation-best checkpoint to {best_ckpt_file}")

    out_file = out_path / "results.json"
    with open(out_file, "w") as f:
        json.dump({
            "method": args.method, "dataset": args.dataset, "seed": args.seed,
            "cfg": cfg, "epochs": epochs, "pcs": pcs, "n_params": int(n_params),
            "checkpoint": str(ckpt_file) if ckpt_file is not None else None,
            "init_checkpoint": args.init_checkpoint,
            "init_info": init_info,
            "best_validation": best_validation,
            "train_history": history,
            "model_config_checkpoint": args.model_config_checkpoint,
            "model_config_updates": model_config_updates,
            "wandb": wandb_run_info(wandb_run),
            "train_time_s": float(train_time), "tau_init": tau_init,
            "intermediate_eval": results,
        }, f, indent=2, default=str)
    print(f"\nSaved to {out_file}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
