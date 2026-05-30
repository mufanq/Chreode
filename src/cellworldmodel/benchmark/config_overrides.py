"""Shared CLI-to-config override helpers for benchmark runners."""
from __future__ import annotations


COMMON_OVERRIDE_KEYS = (
    "hidden_dim",
    "n_layers",
    "noise_dim",
    "batch_size",
    "K",
    "lr",
    "lambda_drift",
    "lambda_down",
    "loss_balancer",
    "loss_balancer_temperature",
    "loss_balancer_lookback_prob",
    "loss_balancer_alpha",
    "loss_balancer_max_multiplier",
)


def apply_common_overrides(args, cfg: dict) -> None:
    """Apply common benchmark CLI overrides in one place."""
    for key in COMMON_OVERRIDE_KEYS:
        if hasattr(args, key):
            val = getattr(args, key)
            if val is not None:
                cfg[key] = val
    if getattr(args, "state_chunk_dim", None) is not None:
        cfg["state_chunk_dim"] = args.state_chunk_dim
    if getattr(args, "learned_state_tokens", None) is not None:
        cfg["learned_state_tokens"] = args.learned_state_tokens
    if getattr(args, "disable_rope", False):
        cfg["disable_rope"] = True
    if getattr(args, "dit_size", None) is not None:
        cfg["dit_size"] = args.dit_size
    if getattr(args, "waddington_dit", False):
        cfg["waddington_dit"] = True
    if getattr(args, "curl_rank", None) is not None:
        cfg["curl_rank"] = args.curl_rank
    if getattr(args, "wdit_curl_update", None) is not None:
        cfg["wdit_curl_update"] = args.wdit_curl_update
    if getattr(args, "wdit_curl_time_mode", None) is not None:
        cfg["wdit_curl_time_mode"] = args.wdit_curl_time_mode
    if getattr(args, "wdit_hybrid_delta0", None) is not None:
        cfg["wdit_hybrid_delta0"] = args.wdit_hybrid_delta0
    if getattr(args, "wdit_hybrid_slope", None) is not None:
        cfg["wdit_hybrid_slope"] = args.wdit_hybrid_slope
    if getattr(args, "wdit_hard_delta0", None) is not None:
        cfg["wdit_hard_delta0"] = args.wdit_hard_delta0
    if getattr(args, "wdit_time_embedding", None) is not None:
        cfg["wdit_time_embedding"] = args.wdit_time_embedding
    if getattr(args, "wdit_time_delta_transform", None) is not None:
        cfg["wdit_time_delta_transform"] = args.wdit_time_delta_transform
    if getattr(args, "wdit_time_delta_scale", None) is not None:
        cfg["wdit_time_delta_scale"] = args.wdit_time_delta_scale
    if getattr(args, "wdit_curl_time_embedding", None) is not None:
        cfg["wdit_curl_time_embedding"] = args.wdit_curl_time_embedding
    if getattr(args, "wdit_curl_time_delta_transform", None) is not None:
        cfg["wdit_curl_time_delta_transform"] = args.wdit_curl_time_delta_transform
    if getattr(args, "wdit_curl_time_delta_scale", None) is not None:
        cfg["wdit_curl_time_delta_scale"] = args.wdit_curl_time_delta_scale
    if getattr(args, "lambda_wdit_a_fro", None) is not None:
        cfg["lambda_wdit_a_fro"] = args.lambda_wdit_a_fro
    if getattr(args, "lambda_wdit_curl", None) is not None:
        cfg["lambda_wdit_curl"] = args.lambda_wdit_curl
    if getattr(args, "drift_pos_ratio", None) is not None:
        cfg["drift_pos_ratio"] = args.drift_pos_ratio
    if getattr(args, "drift_balance_sample_counts", False):
        cfg["drift_balance_sample_counts"] = True
    if getattr(args, "optimizer", None) is not None:
        cfg["optimizer"] = args.optimizer
    if getattr(args, "weight_decay", None) is not None:
        cfg["weight_decay"] = args.weight_decay
    if getattr(args, "lr_schedule", None) is not None and args.lr_schedule != "none":
        cfg["lr_schedule"] = args.lr_schedule
    if getattr(args, "warmup_frac", None) is not None:
        cfg["warmup_frac"] = args.warmup_frac
    if getattr(args, "ema_decay", None) is not None:
        cfg["ema_decay"] = args.ema_decay
    if getattr(args, "down_n_mc", None) is not None:
        cfg["down_n_mc"] = args.down_n_mc
    if getattr(args, "down_antithetic", False):
        cfg["down_antithetic"] = True
    if getattr(args, "multi_delta", False):
        cfg["multi_delta"] = True
    if getattr(args, "md_endpoint_prob", None) is not None:
        cfg["md_endpoint_prob"] = args.md_endpoint_prob
    if getattr(args, "split_policy", None) is not None:
        cfg["split_policy"] = args.split_policy
