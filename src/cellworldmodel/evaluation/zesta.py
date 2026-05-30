"""ZESTA control-time evaluation."""
from __future__ import annotations

import torch

from cellworldmodel.benchmark.baselines_bench import (
    identity_baseline,
    mean_shift_baseline,
    ot_barycentric_baseline,
)
from cellworldmodel.benchmark.common_metrics import compute_dual_protocol_metrics
from cellworldmodel.benchmark.zesta_metrics import distribution_metrics
from cellworldmodel.evaluation.prediction import predict_at_delta
from cellworldmodel.training.transition_sampler import TimepointTransitionSampler


@torch.no_grad()
def evaluate_zesta(adapter, model, device, cfg: dict, seed: int,
                   max_source: int = 2000, max_target: int = 4000, max_preds: int = 4000,
                   include_ot_oracle: bool = True, cellflow_metrics_only: bool = False,
                   sampler: TimepointTransitionSampler | None = None) -> dict:
    torch.manual_seed(seed + 1)
    model.eval()
    if sampler is not None:
        src_full = sampler.get_population(adapter.source_t, split="test")
    else:
        src_full = adapter.get_transition(split="test", target_t=adapter.train_target_times[-1]).source
    if src_full.shape[0] > max_source:
        src = src_full[torch.randperm(src_full.shape[0])[:max_source]].to(device)
    else:
        src = src_full.to(device)

    if sampler is not None:
        src_train = sampler.get_population(adapter.source_t, split="train").to(device)
    else:
        src_train = adapter.get_transition(split="train", target_t=adapter.train_target_times[-1]).source.to(device)

    out: dict[str, dict] = {}
    for target_t in adapter.target_times:
        delta = float(target_t - adapter.source_t)
        if sampler is not None:
            tgt_full = sampler.get_population(target_t, split="test")
        else:
            tgt_full = torch.from_numpy(adapter.coords_by_t[target_t])
        if tgt_full.shape[0] > max_target:
            tgt = tgt_full[torch.randperm(tgt_full.shape[0])[:max_target]].to(device)
        else:
            tgt = tgt_full.to(device)

        preds_model = predict_at_delta(model, src, delta, cfg["K"], adapter.dim, device)
        preds_identity = identity_baseline(src, K=cfg["K"]).to(device)
        if sampler is not None:
            tgt_train = sampler.get_population(target_t, split="train").to(device)
        else:
            tgt_train = torch.from_numpy(adapter.coords_by_t[target_t]).to(device)
        preds_mean = mean_shift_baseline(src, src_train, tgt_train, K=cfg["K"]).to(device)
        preds_ot = None
        if include_ot_oracle:
            preds_ot = ot_barycentric_baseline(src.cpu(), tgt_train.cpu(), reg=0.0, max_samples=2000, seed=seed).to(device)
            preds_ot = preds_ot.repeat_interleave(cfg["K"], dim=0)

        def cap(x: torch.Tensor) -> torch.Tensor:
            if x.shape[0] > max_preds:
                return x[torch.randperm(x.shape[0], device=x.device)[:max_preds]]
            return x

        pred_model = cap(preds_model)
        pred_identity = cap(preds_identity)
        pred_mean = cap(preds_mean)
        tgt_np = tgt.detach().cpu().numpy()
        entry = {
            "delta": delta,
            "cellflow_metrics": {
                "model": distribution_metrics(tgt_np, pred_model.detach().cpu().numpy(), seed=seed),
                "identity": distribution_metrics(tgt_np, pred_identity.detach().cpu().numpy(), seed=seed),
                "mean_shift": distribution_metrics(tgt_np, pred_mean.detach().cpu().numpy(), seed=seed),
            },
        }
        if cellflow_metrics_only:
            entry["model"] = {}
            entry["identity"] = {}
            entry["mean_shift"] = {}
        else:
            entry["model"] = compute_dual_protocol_metrics(pred_model, tgt, seed=seed)
            entry["identity"] = compute_dual_protocol_metrics(pred_identity, tgt, seed=seed)
            entry["mean_shift"] = compute_dual_protocol_metrics(pred_mean, tgt, seed=seed)
        if preds_ot is not None:
            pred_ot = cap(preds_ot)
            entry["ot_oracle"] = (
                {} if cellflow_metrics_only
                else compute_dual_protocol_metrics(pred_ot, tgt, seed=seed)
            )
            entry["cellflow_metrics"]["ot_oracle"] = distribution_metrics(
                tgt_np, pred_ot.detach().cpu().numpy(), seed=seed
            )
        out[f"t={target_t:g}"] = entry
        if cellflow_metrics_only:
            cm = entry["cellflow_metrics"]["model"]
            print(
                f"t={target_t:g} Δ={delta:g}: "
                f"R2={cm['r_squared_mean']:.4f}, "
                f"E={cm['squared_energy_distance']:.4f}, "
                f"MMD={cm['scalar_mmd']:.4f}"
            )
        else:
            print(
                f"t={target_t:g} Δ={delta:g}: "
                f"model W2={entry['model']['branchsbm_w2_top2_mean']:.4f}, "
                f"MMD={entry['model']['branchsbm_mmd_full_mean']:.4f}"
            )
    return out
