"""Shared benchmark training loop for CellWorldModel methods."""
from __future__ import annotations

import copy
from typing import Callable, Optional

import numpy as np
import torch
import torch.optim as optim

from cellworldmodel.benchmark.common_metrics import mmd2_unbiased_multi_sigma, sinkhorn_w2
from cellworldmodel.model.pc_celldrift_bench import downhill_loss
from cellworldmodel.training.drift_loss import (
    drift_stopgrad_loss_from_raw,
    median_heuristic_temperatures,
    normalize_features,
)
from cellworldmodel.training.loss_balancer import (
    LossComponent,
    build_loss_balancer,
    select_gradnorm_params,
)
from cellworldmodel.training.transition_sampler import TimepointTransitionSampler


def build_optimizer(model, cfg: dict, extra_params=None):
    optimizer_name = cfg.get("optimizer", "adam")
    params = list(model.parameters())
    if extra_params is not None:
        params.extend(list(extra_params))
    if optimizer_name == "adam":
        return optim.Adam(params, lr=cfg["lr"])
    if optimizer_name == "adamw":
        return optim.AdamW(
            params,
            lr=cfg["lr"],
            betas=(0.9, 0.95),
            weight_decay=float(cfg.get("weight_decay", 0.01)),
        )
    raise ValueError(f"Unknown optimizer={optimizer_name!r}")


def build_scheduler(opt, cfg: dict, epochs: int):
    if cfg.get("lr_schedule") != "warmup_cosine":
        return None
    warmup_steps = max(1, int(round(float(cfg.get("warmup_frac", 0.05)) * epochs)))
    total_steps = max(1, epochs)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)


def build_transition_pairs(timepoints: list[float], endpoint_prob=None):
    transition_pairs = [(s, t) for i, s in enumerate(timepoints) for t in timepoints[i + 1:]]
    if len(transition_pairs) < 2:
        raise ValueError(
            f"multi_delta requires >=2 transition pairs, got {len(transition_pairs)} "
            f"from timepoints={timepoints}"
        )
    transition_probs = None
    if endpoint_prob is not None:
        endpoint_prob = float(endpoint_prob)
        if not (0.0 < endpoint_prob < 1.0):
            raise ValueError(f"md_endpoint_prob must be in (0,1), got {endpoint_prob}")
        endpoint_pair = (timepoints[0], timepoints[-1])
        if endpoint_pair not in transition_pairs:
            raise ValueError(f"Endpoint pair {endpoint_pair} not in transition_pairs={transition_pairs}")
        transition_probs = np.full(len(transition_pairs), (1.0 - endpoint_prob) / (len(transition_pairs) - 1))
        transition_probs[transition_pairs.index(endpoint_pair)] = endpoint_prob
    return transition_pairs, transition_probs


def train_method(
    method: str,
    adapter,
    model,
    device,
    cfg: dict,
    epochs: int,
    seed: int,
    log_every: int = 50,
    log_callback: Optional[Callable[[dict], None]] = None,
    sampler: TimepointTransitionSampler | None = None,
    validation_callback: Optional[Callable[[int], dict]] = None,
    validation_every: int | None = None,
    validation_metric: str = "val_w2_mean",
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    restore_best_validation: bool = True,
    checkpoint_callback: Optional[Callable[[int, object, dict], None]] = None,
    checkpoint_every: int | None = None,
) -> list[dict]:
    """Unified training loop supporting M1/M2/M7/M8/M9/M10."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    history = []
    model.train()

    ema_model = None
    ema_decay = cfg.get("ema_decay")
    if ema_decay is not None:
        ema_decay = float(ema_decay)
        ema_model = copy.deepcopy(model).to(device)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)

    train_batch = None if sampler is not None else adapter.get_transition(split="train")
    delta = None if train_batch is None else train_batch.delta
    dim = sampler.dim if sampler is not None else adapter.dim
    use_drift = method in ("m7", "m8", "m10")
    use_down = method in ("m8", "m10")
    component_names = ["mmd", "w2"]
    if use_drift:
        component_names.append("drift")
    if use_down:
        component_names.append("down")
    loss_balancer = build_loss_balancer(cfg, component_names, seed=seed).to(device)
    opt = build_optimizer(model, cfg, extra_params=loss_balancer.parameters())
    scheduler = build_scheduler(opt, cfg, epochs)
    gradnorm_params = (
        select_gradnorm_params(model) if loss_balancer.requires_model_params else None
    )
    print(f"[{method}] loss_balancer={cfg.get('loss_balancer', 'fixed')} components={component_names}")
    use_validation = validation_callback is not None and validation_every is not None and int(validation_every) > 0
    validation_every = int(validation_every or 0)
    best_validation_metric = float("inf")
    best_validation_state = None
    best_validation_info: dict | None = None
    latest_validation_info: dict | None = None
    epochs_since_validation_improvement = 0
    checkpoint_every = int(checkpoint_every or 0)
    if use_validation:
        print(
            f"[{method}] validation every {validation_every} epochs; "
            f"metric={validation_metric}; patience={early_stopping_patience}"
        )

    sampler_mode = sampler is not None
    multi_delta = bool(cfg.get("multi_delta", False))
    if sampler_mode:
        print(f"[{method}] sampler training: {len(sampler.pairs)} transitions = {sampler.pairs}")
        if sampler.pair_probs is not None:
            print(f"[{method}] sampler transition probabilities = {[round(float(p), 4) for p in sampler.pair_probs]}")
        transition_pairs = []
        transition_probs = None
    elif multi_delta:
        transition_pairs, transition_probs = build_transition_pairs(
            list(adapter.timepoints), cfg.get("md_endpoint_prob")
        )
        print(f"[{method}] multi-Δ training: {len(transition_pairs)} transitions = {transition_pairs}")
        if transition_probs is not None:
            print(f"[{method}] multi-Δ transition probabilities = {[round(float(p), 4) for p in transition_probs]}")
    else:
        transition_pairs = []
        transition_probs = None

    feature_stats = None
    if use_drift:
        target_ref = sampler.reference_target(split="train").to(device) if sampler_mode else train_batch.target.to(device)
        _, scale, mean, std = normalize_features(target_ref)
        feature_stats = {"mean": mean, "std": std, "scale": scale}
        with torch.no_grad():
            phi_ref = (target_ref - mean) / std * scale
            taus = median_heuristic_temperatures(phi_ref[:1024], multipliers=(0.2, 0.5, 1.5))
        print(f"[{method}] Drift τ (median-heuristic) = {[f'{t:.3f}' for t in taus]}")
    else:
        taus = (0.02, 0.05, 0.2)

    for ep in range(epochs):
        if sampler_mode:
            batch = sampler.sample_train_batch(cfg["batch_size"], rng)
            src_t = batch.source_t
            tgt_t = batch.target_t
            step_delta = batch.delta
            src = batch.source.to(device)
            tgt = batch.target.to(device)
        elif multi_delta:
            if transition_probs is None:
                pair_idx = int(rng.integers(len(transition_pairs)))
            else:
                pair_idx = int(rng.choice(len(transition_pairs), p=transition_probs))
            src_t, tgt_t = transition_pairs[pair_idx]
            step_delta = float(tgt_t) - float(src_t)
            src = adapter.sample_source_batch(
                cfg["batch_size"], split="train", source_t=src_t, rng=rng
            ).to(device)
            tgt = adapter.sample_target_batch(
                cfg["batch_size"], target_t=tgt_t, rng=rng
            ).to(device)
        else:
            src_t = None
            tgt_t = None
            src = adapter.sample_source_batch(cfg["batch_size"], split="train", rng=rng).to(device)
            tgt = adapter.sample_target_batch(cfg["batch_size"], rng=rng).to(device)
            step_delta = delta
        eps = torch.randn(src.shape[0], cfg["K"], dim, device=device)
        delta_t = torch.full((src.shape[0],), step_delta, device=device, dtype=src.dtype)

        z_hat = model(src, delta_t, eps)
        z_hat_flat = z_hat.reshape(-1, dim)

        loss_mmd = mmd2_unbiased_multi_sigma(z_hat_flat, tgt)
        info = {"mmd2": float(loss_mmd.item())}
        pred_weight = None
        if sampler_mode and batch.source_weight is not None:
            pred_weight = batch.source_weight.to(device=device, dtype=z_hat_flat.dtype).repeat_interleave(cfg["K"])
            info["growth_weight_min"] = float(batch.source_weight.min().item())
            info["growth_weight_max"] = float(batch.source_weight.max().item())
            info["growth_weight_mean"] = float(batch.source_weight.float().mean().item())
        loss_w2 = sinkhorn_w2(
            z_hat_flat,
            tgt,
            epsilon=cfg["sinkhorn_eps"],
            num_iters=50,
            weight_x=pred_weight,
        )
        info["w2_approx"] = float(loss_w2.item())
        components = [
            LossComponent("mmd", loss_mmd, float(cfg["lambda_mmd"])),
            LossComponent("w2", loss_w2, float(cfg["lambda_w2"])),
        ]

        if use_drift:
            drift_pos_ratio = cfg.get("drift_pos_ratio")
            if drift_pos_ratio is None:
                tgt_drift = tgt
            else:
                n_drift_pos = max(1, int(round(float(drift_pos_ratio) * z_hat_flat.shape[0])))
                tgt_drift = adapter.sample_target_batch(
                    n_drift_pos, target_t=tgt_t if multi_delta else None, rng=rng
                ).to(device)
            loss_drift, drift_info = drift_stopgrad_loss_from_raw(
                z_gen=z_hat_flat,
                z_pos=tgt_drift,
                z_neg=None,
                temperatures=taus,
                normalize_features_first=True,
                feature_stats=feature_stats,
                balance_sample_counts=bool(cfg.get("drift_balance_sample_counts", False)),
            )
            components.append(LossComponent("drift", loss_drift, float(cfg["lambda_drift"])))
            info["drift_loss"] = drift_info["loss_value"]
            info["drift_norm"] = drift_info["drift_norm"]
            info["drift_n_pos"] = int(tgt_drift.shape[0])
            info["drift_n_neg"] = int(z_hat_flat.shape[0])

        if use_down:
            if method == "m10":
                z_det = model.predict_mean(
                    src,
                    delta_t,
                    n_mc=int(cfg.get("down_n_mc", 32)),
                    antithetic=bool(cfg.get("down_antithetic", False)),
                )
            else:
                z_det = model.predict_mean(src, delta_t)
            loss_down = downhill_loss(model, src, z_det, delta_t)
            components.append(LossComponent("down", loss_down, float(cfg["lambda_down"])))
            info["down_loss"] = float(loss_down.item())

        loss, balance_info = loss_balancer.combine(
            components,
            step=ep,
            model_params=gradnorm_params,
        )
        info.update(balance_info)

        if hasattr(model, "waddington_regularization"):
            lambda_a = float(cfg.get("lambda_wdit_a_fro", 0.0) or 0.0)
            lambda_curl = float(cfg.get("lambda_wdit_curl", 0.0) or 0.0)
            if lambda_a > 0.0 or lambda_curl > 0.0:
                reg_info = model.waddington_regularization(src, delta_t)
                if lambda_a > 0.0:
                    loss = loss + lambda_a * reg_info["wdit_a_fro"]
                    info["wdit_a_fro"] = float(reg_info["wdit_a_fro"].item())
                if lambda_curl > 0.0:
                    loss = loss + lambda_curl * reg_info["wdit_curl_sq"]
                    info["wdit_curl_sq"] = float(reg_info["wdit_curl_sq"].item())

        opt.zero_grad()
        loss.backward()
        if cfg["grad_clip"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["grad_clip"])
        opt.step()
        if scheduler is not None:
            scheduler.step()
        if ema_model is not None:
            with torch.no_grad():
                for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(ema_decay).add_(model_p.data, alpha=1.0 - ema_decay)
                for ema_b, model_b in zip(ema_model.buffers(), model.buffers()):
                    ema_b.data.copy_(model_b.data)

        if ep % log_every == 0 or ep == epochs - 1:
            info["epoch"] = ep
            info["loss"] = float(loss.item())
            info["lr"] = float(opt.param_groups[0]["lr"])
            if multi_delta or sampler_mode:
                info["src_t"] = float(src_t)
                info["tgt_t"] = float(tgt_t)
                info["delta"] = float(step_delta)
            history.append(info)
            parts = [f"loss={info['loss']:.4f}", f"mmd²={info['mmd2']:.4f}", f"w2≈{info['w2_approx']:.4f}"]
            if multi_delta or sampler_mode:
                parts.insert(0, f"({src_t}→{tgt_t} Δ={step_delta})")
            if "drift_loss" in info:
                parts.append(f"drift={info['drift_loss']:.4f}")
            if "down_loss" in info:
                parts.append(f"down={info['down_loss']:.4f}")
            print(f"[{method} ep {ep:4d}] " + "  ".join(parts))
            if log_callback is not None:
                log_callback(dict(info))

        if use_validation and ((ep + 1) % validation_every == 0 or ep == epochs - 1):
            was_training = model.training
            model.eval()
            with torch.no_grad():
                val_info = dict(validation_callback(ep))
            if was_training:
                model.train()
            val_metric = float(val_info[validation_metric])
            improved = val_metric < (best_validation_metric - float(early_stopping_min_delta))
            event = {
                "event": "validation",
                "epoch": ep,
                **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in val_info.items()},
                "validation_metric": validation_metric,
                "improved": bool(improved),
            }
            latest_validation_info = dict(event)
            history.append(event)
            print(
                f"[{method} ep {ep:4d}] val {validation_metric}={val_metric:.4f} "
                f"best={best_validation_metric:.4f} improved={improved}"
            )
            if log_callback is not None:
                log_callback(dict(event))
            if improved:
                best_validation_metric = val_metric
                epochs_since_validation_improvement = 0
                best_validation_state = copy.deepcopy(model.state_dict())
                best_validation_info = dict(event)
            else:
                epochs_since_validation_improvement += validation_every
                if (
                    early_stopping_patience is not None
                    and epochs_since_validation_improvement >= int(early_stopping_patience)
                ):
                    stop_info = {
                        "event": "early_stopping",
                        "epoch": ep,
                        "best_epoch": None if best_validation_info is None else int(best_validation_info["epoch"]),
                        "best_validation_metric": float(best_validation_metric),
                        "epochs_since_improvement": int(epochs_since_validation_improvement),
                    }
                    history.append(stop_info)
                    print(
                        f"[{method}] Early stopping at epoch {ep}; "
                        f"best epoch={stop_info['best_epoch']} "
                        f"{validation_metric}={best_validation_metric:.4f}"
                    )
                    break

        if checkpoint_callback is not None and checkpoint_every > 0 and (
            (ep + 1) % checkpoint_every == 0 or ep == epochs - 1
        ):
            checkpoint_info = {
                "event": "checkpoint",
                "epoch": ep,
                "epoch_1based": ep + 1,
                "latest_validation": latest_validation_info,
                "best_validation": best_validation_info,
            }
            checkpoint_callback(ep, model, checkpoint_info)

    if ema_model is not None and cfg.get("eval_with_ema", True):
        model.load_state_dict(ema_model.state_dict())
        print(f"[{method}] Loaded EMA weights for evaluation (decay={ema_decay})")

    if use_validation and best_validation_state is not None:
        best_record = {
            "event": "best_validation",
            "epoch": int(best_validation_info["epoch"]) if best_validation_info is not None else None,
            "validation_metric": validation_metric,
            "best_validation_metric": float(best_validation_metric),
            "restored": bool(restore_best_validation),
        }
        history.append(best_record)
        if restore_best_validation:
            model.load_state_dict(best_validation_state)
            print(
                f"[{method}] Restored best validation weights from epoch "
                f"{best_record['epoch']} ({validation_metric}={best_validation_metric:.4f})"
            )

    return history
