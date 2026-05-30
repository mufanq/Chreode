from __future__ import annotations

import json
import logging
import math
import random
import socket
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"


DEFAULT_EXPERIMENT_CONFIG: Dict[str, object] = {
    "data_root": "../data/processed/genhui_all/unified_h5ad_moscot_growth_rate",
    "dataloader_module_root": "../data/processed/genhui_all/unified_h5ad_moscot_growth_rate",
    "selected_datasets": ["GSE275562"],
    "min_probability": 0.01,
    "max_probability": 1.0,
    "max_pairs_per_dataset": None,
    "cache_expressions": False,
    "return_dataset_id": False,
    "return_probability": False,
    "return_direction": False,
    "return_growth_rates": False,
    "verbose_loader": False,
    "split": {
        "train_fraction": 0.8,
        "val_fraction": 0.1,
        "test_fraction": 0.1,
        "group_by": "target_cell",
    },
    "preprocess": {
        "normalize_target_sum": 10000.0,
        "pca_fit_cells": 1024,
    },
    "model": {
        "latent_dim": 128,
        "hidden_dim": 512,
        "num_layers": 2,
        "dropout": 0.1,
    },
    "training": {
        "batch_size": 128,
        "eval_batch_size": 128,
        "num_workers": 4,
        "train_samples_per_round": None,
        "rounds": 5,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "gene_loss_weight": 0.1,
        "gradient_clip_norm": 1.0,
        "early_stop_patience": 2,
        "val_max_batches": 32,
        "eval_max_batches": None,
    },
    "wandb": {
        "enabled": True,
        "project": "CellWorldModel",
        "entity": None,
        "mode": "online",
        "tags": ["simple-pipeline"],
    },
    "runtime": {
        "seed": 42,
        "device": "cuda:7",
        "run_host": "local",
    },
}


def deep_merge(base: MutableMapping, override: Mapping) -> MutableMapping:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _resolve_from_code_root(config_path: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    code_root = config_path.resolve().parent.parent
    return str((code_root / path).resolve())


def load_experiment_config(config_path: str, experiment_name: str) -> Dict[str, object]:
    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    root_cfg = loaded.get("simple_pair_predictor", {})
    experiments = root_cfg.get("experiments", {})
    if experiment_name not in experiments:
        available = ", ".join(sorted(experiments))
        raise KeyError(f"Experiment '{experiment_name}' not found. Available: {available}")

    resolved = deep_merge(DEFAULT_EXPERIMENT_CONFIG, root_cfg.get("defaults", {}))
    resolved = deep_merge(resolved, experiments[experiment_name])
    if "output_dir" in root_cfg:
        resolved["output_dir"] = root_cfg["output_dir"]

    resolved["experiment_name"] = experiment_name
    resolved["config_path"] = str(config_file.resolve())
    resolved["data_root"] = _resolve_from_code_root(config_file, str(resolved["data_root"]))
    resolved["dataloader_module_root"] = _resolve_from_code_root(
        config_file, str(resolved.get("dataloader_module_root", resolved["data_root"]))
    )
    resolved["output_dir"] = _resolve_from_code_root(
        config_file, str(resolved.get("output_dir", "../output/simple_pair_predictor"))
    )
    return resolved


def save_yaml(data: Mapping, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=False, allow_unicode=False)


def save_json(data: Mapping, path: Path) -> None:
    def default(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=default)


def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("simple_pair_predictor")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_name)
    return torch.device("cpu")


def get_hostname() -> str:
    return socket.gethostname()


def import_genhui_dataloader(module_root: str):
    module_path = Path(module_root).resolve()
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

    from moscot_dataloader import MultiDatasetMoscotLoader, collate_with_gene_names

    return MultiDatasetMoscotLoader, collate_with_gene_names


def build_dataset(config: Mapping[str, object]):
    loader_cls, collate_fn = import_genhui_dataloader(str(config["dataloader_module_root"]))
    dataset = loader_cls(
        root_dir=str(config["data_root"]),
        selected_datasets=config.get("selected_datasets"),
        min_probability=float(config["min_probability"]),
        max_probability=float(config["max_probability"]),
        max_pairs_per_dataset=config.get("max_pairs_per_dataset"),
        cache_expressions=bool(config.get("cache_expressions", False)),
        return_dataset_id=bool(config.get("return_dataset_id", False)),
        return_probability=bool(config.get("return_probability", False)),
        return_direction=bool(config.get("return_direction", False)),
        return_growth_rates=bool(config.get("return_growth_rates", False)),
        verbose=bool(config.get("verbose_loader", False)),
    )
    return dataset, collate_fn


def group_key_columns(group_by: str) -> List[str]:
    if group_by == "target_cell":
        return ["dataset_id", "source_time", "target_time", "target_cell"]
    if group_by == "source_cell":
        return ["dataset_id", "source_time", "target_time", "source_cell"]
    raise ValueError(f"Unsupported group_by value: {group_by}")


def split_pair_indices(
    pairs_df: pd.DataFrame,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    group_by: str,
    seed: int,
) -> Dict[str, np.ndarray]:
    total_fraction = train_fraction + val_fraction + test_fraction
    if not math.isclose(total_fraction, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("train/val/test fractions must sum to 1.0")

    rng = np.random.default_rng(seed)
    group_cols = group_key_columns(group_by)
    splits = {"train": [], "val": [], "test": []}

    transition_cols = ["dataset_id", "source_time", "target_time"]
    for _, transition_df in pairs_df.groupby(transition_cols, sort=False):
        group_df = transition_df[group_cols].drop_duplicates().copy()
        group_records = [tuple(row) for row in group_df.itertuples(index=False, name=None)]
        rng.shuffle(group_records)

        n_groups = len(group_records)
        if n_groups == 1:
            train_groups = set(group_records)
            val_groups = set()
            test_groups = set()
        else:
            n_val = int(round(n_groups * val_fraction))
            n_test = int(round(n_groups * test_fraction))
            if n_groups >= 3:
                n_val = max(1, n_val)
                n_test = max(1, n_test)
            if n_val + n_test >= n_groups:
                overflow = n_val + n_test - (n_groups - 1)
                if n_test >= overflow:
                    n_test -= overflow
                else:
                    n_val -= overflow - n_test
                    n_test = 0

            val_groups = set(group_records[:n_val])
            test_groups = set(group_records[n_val:n_val + n_test])
            train_groups = set(group_records[n_val + n_test:])
            if not train_groups:
                train_groups = {group_records[-1]}
                test_groups.discard(group_records[-1])
                val_groups.discard(group_records[-1])

        keys = transition_df[group_cols].apply(lambda row: tuple(row), axis=1)
        transition_indices = transition_df.index.to_numpy(dtype=np.int64)
        for key, idx in zip(keys, transition_indices, strict=True):
            if key in train_groups:
                splits["train"].append(int(idx))
            elif key in val_groups:
                splits["val"].append(int(idx))
            else:
                splits["test"].append(int(idx))

    return {
        name: np.array(sorted(values), dtype=np.int64)
        for name, values in splits.items()
    }


def normalize_log1p_expression(expr: torch.Tensor, target_sum: float) -> torch.Tensor:
    expr = expr.to(torch.float32)
    if expr.ndim == 1:
        expr = expr.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    counts = expr.sum(dim=1, keepdim=True)
    scale = torch.where(counts > 0, target_sum / counts, torch.zeros_like(counts))
    transformed = torch.log1p(expr * scale)
    return transformed.squeeze(0) if squeeze else transformed


def sample_cells_for_pca(
    dataset,
    train_indices: np.ndarray,
    pca_fit_cells: int,
    normalize_target_sum: float,
    seed: int,
) -> np.ndarray:
    if len(train_indices) == 0:
        raise ValueError("Training split is empty; cannot fit PCA")

    rng = np.random.default_rng(seed)
    n_pairs = max(1, math.ceil(pca_fit_cells / 2))
    chosen = rng.choice(train_indices, size=min(len(train_indices), n_pairs), replace=False)
    sampled: List[np.ndarray] = []

    for idx in chosen:
        sample = dataset[int(idx)]
        for key in ("expr_t", "expr_t1"):
            transformed = normalize_log1p_expression(sample[key], target_sum=normalize_target_sum)
            sampled.append(transformed.cpu().numpy())
            if len(sampled) >= pca_fit_cells:
                break
        if len(sampled) >= pca_fit_cells:
            break

    matrix = np.stack(sampled, axis=0).astype(np.float32)
    return matrix


def fit_pca_from_training_pairs(
    dataset,
    train_indices: np.ndarray,
    latent_dim: int,
    pca_fit_cells: int,
    normalize_target_sum: float,
    seed: int,
) -> Dict[str, torch.Tensor]:
    sample_matrix = sample_cells_for_pca(
        dataset=dataset,
        train_indices=train_indices,
        pca_fit_cells=pca_fit_cells,
        normalize_target_sum=normalize_target_sum,
        seed=seed,
    )
    effective_latent_dim = min(latent_dim, sample_matrix.shape[0] - 1, sample_matrix.shape[1])
    if effective_latent_dim < 2:
        raise ValueError("Need at least 2 effective PCA dimensions")

    pca = PCA(n_components=effective_latent_dim, svd_solver="randomized", random_state=seed)
    pca.fit(sample_matrix)

    return {
        "components": torch.tensor(pca.components_, dtype=torch.float32),
        "mean": torch.tensor(pca.mean_, dtype=torch.float32),
        "explained_variance_ratio": torch.tensor(
            pca.explained_variance_ratio_, dtype=torch.float32
        ),
    }


def encode_expression(expr: torch.Tensor, pca_state: Mapping[str, torch.Tensor], target_sum: float) -> torch.Tensor:
    transformed = normalize_log1p_expression(expr, target_sum=target_sum)
    mean = pca_state["mean"].to(transformed.device)
    components = pca_state["components"].to(transformed.device)
    return (transformed - mean) @ components.T


def decode_latent(latent: torch.Tensor, pca_state: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mean = pca_state["mean"].to(latent.device)
    components = pca_state["components"].to(latent.device)
    return latent @ components + mean


def summarize_time_statistics(pairs_df: pd.DataFrame, train_indices: np.ndarray) -> Dict[str, float]:
    train_pairs = pairs_df.iloc[train_indices]
    time_values = np.concatenate(
        [
            train_pairs["source_time"].to_numpy(dtype=np.float32),
            train_pairs["target_time"].to_numpy(dtype=np.float32),
        ]
    )
    dt_values = (
        train_pairs["target_time"].to_numpy(dtype=np.float32)
        - train_pairs["source_time"].to_numpy(dtype=np.float32)
    )
    time_min = float(time_values.min())
    time_max = float(time_values.max())
    dt_mean = float(dt_values.mean())
    dt_std = float(dt_values.std()) if float(dt_values.std()) > 0 else 1.0
    return {
        "time_min": time_min,
        "time_max": time_max,
        "time_scale": max(time_max - time_min, 1e-6),
        "dt_mean": dt_mean,
        "dt_std": dt_std,
    }


def time_key(time_value: float) -> str:
    return f"{float(time_value):.6f}"


def compute_target_time_means(
    dataset,
    train_indices: np.ndarray,
    pca_state: Mapping[str, torch.Tensor],
    normalize_target_sum: float,
) -> Dict[str, torch.Tensor]:
    sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}

    for idx in train_indices.tolist():
        sample = dataset[int(idx)]
        target_expr = normalize_log1p_expression(sample["expr_t1"], target_sum=normalize_target_sum).cpu()
        key = time_key(float(sample["time_t1"].item()))
        if key not in sums:
            sums[key] = target_expr.clone()
            counts[key] = 0
        sums[key] += target_expr
        counts[key] += 1

    means = {key: sums[key] / counts[key] for key in sums}
    if not means:
        raise ValueError("Could not compute target-time means from empty training data")
    global_mean = torch.stack(list(means.values()), dim=0).mean(dim=0)
    means["_global"] = global_mean
    return means


def _as_numpy_float32(array_like) -> np.ndarray:
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(array_like, dtype=np.float32)


def compute_target_time_means_from_latent(
    pairs_df: pd.DataFrame,
    train_indices: np.ndarray,
    target_int_ids: np.ndarray,
    latent_z_state: np.ndarray,
    pca_state: Mapping[str, object],
) -> Dict[str, torch.Tensor]:
    """Fast target-time mean baseline using cached Phase 0 PCA latents.

    This reproduces the *pair-weighted* semantics of ``compute_target_time_means``
    but avoids calling ``dataset.__getitem__`` for every training pair.

    Steps:
      1. Gather target cell int_ids for train pairs.
      2. Index Phase 0 ``latent_z_state`` once to get all target latents.
      3. Average in 128-d latent space by target timepoint.
      4. Decode each timepoint mean back to gene space via PCA inverse transform.

    Notes:
      - The decoded means live in the PCA reconstruction subspace, which is the
        correct space for Stage 1 model outputs and evaluation baselines.
      - Means are pair-weighted (one contribution per train pair), matching the
        legacy slow implementation exactly apart from PCA reconstruction.
      - ``_global`` preserves legacy behavior: unweighted average across
        timepoint-specific means (not pair-weighted across all rows).
    """
    train_indices = np.asarray(train_indices, dtype=np.int64)
    target_int_ids = np.asarray(target_int_ids, dtype=np.int64)

    if train_indices.size == 0:
        raise ValueError("Could not compute target-time means from empty training data")
    if len(target_int_ids) != len(pairs_df):
        raise ValueError(
            f"target_int_ids length ({len(target_int_ids)}) does not match pairs_df length ({len(pairs_df)})"
        )

    train_target_ids = target_int_ids[train_indices]
    missing_mask = train_target_ids < 0
    if np.any(missing_mask):
        raise ValueError(
            "target_int_ids contains unmapped train targets; Phase 0 cell mapping is incomplete "
            f"for {int(missing_mask.sum())} training pairs"
        )

    train_target_times = pairs_df.iloc[train_indices]["target_time"].to_numpy(dtype=np.float32)
    train_target_latent = np.asarray(latent_z_state[train_target_ids], dtype=np.float32)
    if train_target_latent.ndim != 2:
        raise ValueError(
            f"Expected latent_z_state[target_ids] to be 2D, got shape {train_target_latent.shape}"
        )

    unique_times, inverse = np.unique(train_target_times, return_inverse=True)
    if unique_times.size == 0:
        raise ValueError("No target times found in training split")

    sums_latent = np.zeros((unique_times.size, train_target_latent.shape[1]), dtype=np.float64)
    np.add.at(sums_latent, inverse, train_target_latent)
    counts = np.bincount(inverse, minlength=unique_times.size).astype(np.int64)
    means_latent = (sums_latent / counts[:, None]).astype(np.float32)

    components = _as_numpy_float32(pca_state["components"])
    mean = _as_numpy_float32(pca_state["mean"])
    means_gene = (means_latent @ components + mean[None, :]).astype(np.float32)

    means = {
        time_key(float(time_value)): torch.from_numpy(means_gene[row_idx].copy())
        for row_idx, time_value in enumerate(unique_times.tolist())
    }
    global_mean = torch.from_numpy(means_gene.mean(axis=0).astype(np.float32, copy=False).copy())
    means["_global"] = global_mean
    return means


def compute_subset_sampling_weights(dataset, subset_indices: np.ndarray) -> torch.Tensor:
    full_weights = dataset.get_sampling_weights().numpy()
    subset_weights = full_weights[subset_indices]
    subset_weights = np.clip(subset_weights.astype(np.float64), a_min=1e-8, a_max=None)
    return torch.tensor(subset_weights, dtype=torch.double)


def build_dataloader(
    dataset,
    collate_fn,
    subset_indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    train: bool,
    device: torch.device,
    train_samples_per_round: Optional[int] = None,
) -> DataLoader:
    subset = Subset(dataset, subset_indices.tolist())
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_fn,
        "persistent_workers": num_workers > 0,
    }

    if train:
        weights = compute_subset_sampling_weights(dataset, subset_indices)
        full_round = (
            train_samples_per_round is None or
            int(train_samples_per_round) >= len(subset_indices)
        )
        num_samples = len(subset_indices) if full_round else int(train_samples_per_round)
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=num_samples,
            replacement=not full_round,
        )
        return DataLoader(subset, sampler=sampler, shuffle=False, **loader_kwargs)

    return DataLoader(subset, shuffle=False, **loader_kwargs)


def resolve_train_round_settings(
    train_subset_size: int,
    train_samples_per_round: Optional[int],
    batch_size: int,
) -> Dict[str, object]:
    full_train_round = (
        train_samples_per_round is None or
        int(train_samples_per_round) >= train_subset_size
    )
    samples_per_round = train_subset_size if full_train_round else int(train_samples_per_round)
    steps_per_round = math.ceil(samples_per_round / batch_size)
    unit_name = "round" if full_train_round else "sampled_cycle"
    return {
        "full_train_round": full_train_round,
        "samples_per_round": samples_per_round,
        "steps_per_round": steps_per_round,
        "unit_name": unit_name,
    }


class SimplePairPredictor(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        in_dim = latent_dim + 3
        layers: List[nn.Module] = []
        current_dim = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, latent_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, z_t: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([z_t, time_features], dim=1)
        return z_t + self.network(inputs)


def make_time_features(
    time_t: torch.Tensor,
    time_t1: torch.Tensor,
    time_stats: Mapping[str, float],
) -> torch.Tensor:
    time_scale = float(time_stats["time_scale"])
    dt_mean = float(time_stats["dt_mean"])
    dt_std = float(time_stats["dt_std"])

    source = (time_t - float(time_stats["time_min"])) / time_scale
    target = (time_t1 - float(time_stats["time_min"])) / time_scale
    delta = ((time_t1 - time_t) - dt_mean) / dt_std
    return torch.stack([source, target, delta], dim=1)


def rowwise_pearson(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_centered = pred - pred.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)
    denom = pred_centered.norm(dim=1) * target_centered.norm(dim=1)
    corr = (pred_centered * target_centered).sum(dim=1) / denom.clamp_min(1e-8)
    return torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def init_metric_sums() -> Dict[str, float]:
    return {
        "sum_sq": 0.0,
        "sum_abs": 0.0,
        "sum_corr": 0.0,
        "sum_latent_sq": 0.0,
        "n_values": 0.0,
        "n_cells": 0.0,
    }


def update_metric_sums(
    state: MutableMapping[str, float],
    pred: torch.Tensor,
    target: torch.Tensor,
    latent_pred: Optional[torch.Tensor] = None,
    latent_target: Optional[torch.Tensor] = None,
) -> None:
    diff = pred - target
    state["sum_sq"] += float(diff.pow(2).sum().item())
    state["sum_abs"] += float(diff.abs().sum().item())
    state["sum_corr"] += float(rowwise_pearson(pred, target).sum().item())
    state["n_values"] += float(pred.numel())
    state["n_cells"] += float(pred.shape[0])
    if latent_pred is not None and latent_target is not None:
        state["sum_latent_sq"] += float((latent_pred - latent_target).pow(2).sum().item())


def finalize_metric_sums(state: Mapping[str, float], latent_dim: int) -> Dict[str, float]:
    n_values = max(state["n_values"], 1.0)
    n_cells = max(state["n_cells"], 1.0)
    metrics = {
        "gene_mse": state["sum_sq"] / n_values,
        "gene_mae": state["sum_abs"] / n_values,
        "pearson_mean": state["sum_corr"] / n_cells,
    }
    if state["sum_latent_sq"] > 0:
        metrics["latent_mse"] = state["sum_latent_sq"] / (n_cells * latent_dim)
    return metrics


def build_time_mean_predictions(
    time_t1: torch.Tensor,
    target_time_means: Mapping[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    global_mean = target_time_means["_global"]
    rows = [
        target_time_means.get(time_key(float(value)), global_mean)
        for value in time_t1.detach().cpu().tolist()
    ]
    return torch.stack(rows, dim=0).to(device)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    pca_state: Mapping[str, torch.Tensor],
    time_stats: Mapping[str, float],
    normalize_target_sum: float,
    target_time_means: Mapping[str, torch.Tensor],
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    model_state = init_metric_sums()
    identity_state = init_metric_sums()
    time_mean_state = init_metric_sums()
    latent_dim = int(pca_state["components"].shape[0])
    n_pairs = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            expr_t = batch["expr_t"].to(device)
            expr_t1 = batch["expr_t1"].to(device)
            time_t = batch["time_t"].to(device)
            time_t1 = batch["time_t1"].to(device)

            target_gene = normalize_log1p_expression(expr_t1, target_sum=normalize_target_sum)
            source_gene = normalize_log1p_expression(expr_t, target_sum=normalize_target_sum)
            z_t = encode_expression(expr_t, pca_state, normalize_target_sum)
            target_z = encode_expression(expr_t1, pca_state, normalize_target_sum)
            time_features = make_time_features(time_t, time_t1, time_stats).to(device)

            pred_z = model(z_t, time_features)
            pred_gene = decode_latent(pred_z, pca_state)
            identity_gene = source_gene
            time_mean_gene = build_time_mean_predictions(time_t1, target_time_means, device)
            n_pairs += int(expr_t.shape[0])

            update_metric_sums(model_state, pred_gene, target_gene, pred_z, target_z)
            update_metric_sums(identity_state, identity_gene, target_gene)
            update_metric_sums(time_mean_state, time_mean_gene, target_gene)

    return {
        "model": {
            **finalize_metric_sums(model_state, latent_dim),
            "n_pairs": n_pairs,
        },
        "identity_baseline": {
            **finalize_metric_sums(identity_state, latent_dim),
            "n_pairs": n_pairs,
        },
        "target_time_mean_baseline": {
            **finalize_metric_sums(time_mean_state, latent_dim),
            "n_pairs": n_pairs,
        },
    }


def train_one_round(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    pca_state: Mapping[str, torch.Tensor],
    time_stats: Mapping[str, float],
    normalize_target_sum: float,
    device: torch.device,
    gene_loss_weight: float,
    gradient_clip_norm: float,
    round_idx: int,
    global_step_start: int = 0,
    step_logger=None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_latent = 0.0
    total_gene = 0.0
    n_batches = 0
    n_pairs = 0
    global_step = global_step_start

    for batch in loader:
        expr_t = batch["expr_t"].to(device)
        expr_t1 = batch["expr_t1"].to(device)
        time_t = batch["time_t"].to(device)
        time_t1 = batch["time_t1"].to(device)

        target_gene = normalize_log1p_expression(expr_t1, target_sum=normalize_target_sum)
        z_t = encode_expression(expr_t, pca_state, normalize_target_sum)
        target_z = encode_expression(expr_t1, pca_state, normalize_target_sum)
        time_features = make_time_features(time_t, time_t1, time_stats).to(device)

        pred_z = model(z_t, time_features)
        pred_gene = decode_latent(pred_z, pca_state)

        latent_loss = F.mse_loss(pred_z, target_z)
        gene_loss = F.smooth_l1_loss(pred_gene, target_gene)
        loss = latent_loss + gene_loss_weight * gene_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        total_loss += float(loss.item())
        total_latent += float(latent_loss.item())
        total_gene += float(gene_loss.item())
        n_batches += 1
        n_pairs += int(expr_t.shape[0])
        global_step += 1

        if step_logger is not None:
            step_logger(
                global_step,
                {
                    "round": round_idx,
                    "train_step_loss": float(loss.item()),
                    "train_step_latent_loss": float(latent_loss.item()),
                    "train_step_gene_loss": float(gene_loss.item()),
                    "train_pairs_seen_so_far": n_pairs,
                    "optimizer_step_in_round": n_batches,
                    "batch_pairs": int(expr_t.shape[0]),
                },
            )

    denom = max(n_batches, 1)
    return {
        "train_loss": total_loss / denom,
        "train_latent_loss": total_latent / denom,
        "train_gene_loss": total_gene / denom,
        "train_pairs_seen": n_pairs,
        "optimizer_steps": n_batches,
        "global_step_end": global_step,
    }


def build_output_paths(output_dir: str, experiment_name: str) -> Dict[str, Path]:
    base_dir = Path(output_dir) / experiment_name
    paths = {
        "base_dir": base_dir,
        "artifacts_dir": base_dir / "artifacts",
        "metrics_dir": base_dir / "metrics",
        "logs_dir": base_dir / "logs",
        "resolved_config": base_dir / "resolved_config.yaml",
        "checkpoint": base_dir / "artifacts" / "model.pt",
        "splits": base_dir / "artifacts" / "splits.npz",
        "train_summary": base_dir / "metrics" / "train_summary.json",
        "eval_report": base_dir / "reports" / "eval.json",
        "train_done": base_dir / "artifacts" / "train.done",
    }
    for key in ("artifacts_dir", "metrics_dir", "logs_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    (base_dir / "reports").mkdir(parents=True, exist_ok=True)
    return paths


def prepare_experiment(config_path: str, experiment_name: str, logger: logging.Logger):
    config = load_experiment_config(config_path, experiment_name)
    set_seed(int(config["runtime"]["seed"]))
    device = resolve_device(str(config["runtime"]["device"]))
    dataset, collate_fn = build_dataset(config)

    split_cfg = config["split"]
    splits = split_pair_indices(
        pairs_df=dataset.pairs_df,
        train_fraction=float(split_cfg["train_fraction"]),
        val_fraction=float(split_cfg["val_fraction"]),
        test_fraction=float(split_cfg["test_fraction"]),
        group_by=str(split_cfg["group_by"]),
        seed=int(config["runtime"]["seed"]),
    )

    preprocess_cfg = config["preprocess"]
    model_cfg = config["model"]
    pca_state = fit_pca_from_training_pairs(
        dataset=dataset,
        train_indices=splits["train"],
        latent_dim=int(model_cfg["latent_dim"]),
        pca_fit_cells=int(preprocess_cfg["pca_fit_cells"]),
        normalize_target_sum=float(preprocess_cfg["normalize_target_sum"]),
        seed=int(config["runtime"]["seed"]),
    )
    time_stats = summarize_time_statistics(dataset.pairs_df, splits["train"])
    target_time_means = compute_target_time_means(
        dataset=dataset,
        train_indices=splits["train"],
        pca_state=pca_state,
        normalize_target_sum=float(preprocess_cfg["normalize_target_sum"]),
    )

    logger.info(
        "Prepared experiment %s on %s with %s train / %s val / %s test pairs",
        experiment_name,
        device,
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )
    return config, dataset, collate_fn, splits, pca_state, time_stats, target_time_means, device


def serialize_target_time_means(target_time_means: Mapping[str, torch.Tensor]) -> Dict[str, List[float]]:
    return {key: value.cpu().tolist() for key, value in target_time_means.items()}


def deserialize_target_time_means(serialized: Mapping[str, Sequence[float]]) -> Dict[str, torch.Tensor]:
    return {key: torch.tensor(value, dtype=torch.float32) for key, value in serialized.items()}


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Dict[str, object]:
    return torch.load(checkpoint_path, map_location=device, weights_only=False)
