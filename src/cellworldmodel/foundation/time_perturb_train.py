"""Training loop for perturbation-as-temporal-transition benchmarks."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from cellworldmodel.foundation.action import GeneSetPerturbationEncoder
from cellworldmodel.foundation.dynamics_loader import load_foundation_transition
from cellworldmodel.foundation.io_utils import write_json
from cellworldmodel.foundation.perturbation_population_losses import (
    PopulationLossWeights,
    PopulationPerturbationLoss,
)
from cellworldmodel.foundation.perturbation_population_models import (
    PopulationPredictorConfig,
    ResponseDecoderConfig,
    build_population_predictor,
    build_response_decoder,
)
from cellworldmodel.foundation.time_perturb_dataset import (
    TimeResolvedPerturbationDataOptions,
    TimeResolvedPerturbationDataset,
)
from cellworldmodel.foundation.vae_eval import load_vae_checkpoint


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _metrics(pred: np.ndarray, target: np.ndarray, control: np.ndarray, top_idx: np.ndarray) -> dict[str, float]:
    delta_true = target - control
    delta_pred = pred - control
    return {
        "mse": float(np.mean((pred - target) ** 2)),
        "pearson": _pearson(pred, target),
        "delta_pearson": _pearson(delta_pred, delta_true),
        "top_mse": float(np.mean((pred[top_idx] - target[top_idx]) ** 2)),
        "top_pearson": _pearson(pred[top_idx], target[top_idx]),
        "top_delta_pearson": _pearson(delta_pred[top_idx], delta_true[top_idx]),
        "opposite_direction_fraction": float(np.mean(np.sign(delta_pred[top_idx]) != np.sign(delta_true[top_idx]))),
    }


class TemporalActionProjector(nn.Module):
    """Mix perturbation identity with elapsed developmental time.

    This keeps downstream perturbation prediction in the same mathematical form
    as stage-2 dynamics: a latent transition conditioned on a time interval.
    """

    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(action_dim) + 6),
            nn.Linear(int(action_dim) + 6, max(128, int(action_dim) * 2)),
            nn.SiLU(),
            nn.Linear(max(128, int(action_dim) * 2), int(action_dim)),
        )

    @staticmethod
    def time_features(delta: torch.Tensor) -> torch.Tensor:
        delta = delta.to(dtype=torch.float32)
        log_delta = torch.log1p(delta)
        norm = delta / 30.0
        return torch.stack([
            norm,
            log_delta / torch.log(torch.tensor(31.0, device=delta.device)),
            torch.sin(2.0 * torch.pi * delta / 7.0),
            torch.cos(2.0 * torch.pi * delta / 7.0),
            torch.sin(2.0 * torch.pi * delta / 30.0),
            torch.cos(2.0 * torch.pi * delta / 30.0),
        ], dim=-1)

    def forward(self, action: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        features = self.time_features(delta).to(device=action.device, dtype=action.dtype)
        return self.net(torch.cat([action, features], dim=-1))


@dataclass(frozen=True)
class TimePerturbationTrainOptions:
    adata: str | Path
    gene_vocab: str | Path
    vae_checkpoint: str | Path
    output_dir: str | Path
    route: str
    init_checkpoint: str | Path | None = None
    init_name: str = "random"
    condition_col: str = "condition_label"
    time_col: str = "time"
    control_label: str = "control"
    perturbation_label: str = "NOTCH1_KO"
    source_time: float = 0.0
    source_policy: str = "fixed"
    train_times: tuple[float, ...] = (2.0, 5.0, 10.0)
    eval_times: tuple[float, ...] = (14.0, 30.0)
    action_genes: tuple[str, ...] = ("NOTCH1",)
    action_sign: float = -1.0
    action_modality_id: int = 1
    input_log1p: bool = False
    experiment: str = "g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw"
    dit_size: str = "small"
    max_steps: int = 1200
    set_size: int = 96
    eval_batch_size: int = 256
    max_eval_source_cells: int = 1024
    k_samples: int = 2
    action_dim: int = 64
    n_programs: int = 8
    lr: float = 3e-4
    seed: int = 0
    device: str | None = None
    latent_mmd_weight: float = 1.0
    latent_w2_weight: float = 0.0
    expr_bulk_weight: float = 1.0
    delta_bulk_weight: float = 0.0
    de_bulk_weight: float = 2.0
    delta_cosine_weight: float = 0.2
    sinkhorn_eps: float = 0.05
    sinkhorn_iters: int = 50
    top_k: int = 100
    disable_kick: bool = False
    disable_field: bool = False
    flat_action: bool = False
    adapter_components: str = "full"
    calibrate_potential: bool = False
    rollout_steps: int = 4
    disable_rollout: bool = False
    disable_action_time: bool = False
    virtual_time_min: float = 0.25
    virtual_time_max: float = 1.75
    locked_time_transform: str = "log_bounded"
    locked_time_scale: float = 30.0
    prediction_mode: str = "state"
    response_decoder: str = "none"
    response_programs: int = 32
    sparse_programs: bool = False
    nonnegative_basis: bool = False
    set_context_decoder: bool = False


class TimePerturbationTrainer:
    def __init__(self, options: TimePerturbationTrainOptions) -> None:
        self.options = options
        self.output_dir = Path(options.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(int(options.seed))
        torch.manual_seed(int(options.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(options.seed))
        self.device = torch.device(options.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dataset = TimeResolvedPerturbationDataset(TimeResolvedPerturbationDataOptions(
            adata=options.adata,
            gene_vocab=options.gene_vocab,
            condition_col=options.condition_col,
            time_col=options.time_col,
            control_label=options.control_label,
            perturbation_label=options.perturbation_label,
            source_time=float(options.source_time),
            source_policy=str(options.source_policy),
            train_times=tuple(float(t) for t in options.train_times),
            eval_times=tuple(float(t) for t in options.eval_times),
            action_genes=tuple(options.action_genes),
            action_sign=float(options.action_sign),
            action_modality_id=int(options.action_modality_id),
            input_log1p=bool(options.input_log1p),
            top_k=int(options.top_k),
        ))
        self.vae = load_vae_checkpoint(options.vae_checkpoint, self.device)
        for param in self.vae.model.parameters():
            param.requires_grad_(False)
        latent_dim = int(self.vae.config["latent_dim"])
        self.base_transition, self.load_info = load_foundation_transition(
            checkpoint=options.init_checkpoint,
            latent_dim=latent_dim,
            device=self.device,
            experiment=str(options.experiment),
            dit_size=str(options.dit_size),
            batch_size=int(options.set_size),
            k_samples=int(options.k_samples),
            lr=float(options.lr),
            action_dim=0,
        )
        self.action_encoder = GeneSetPerturbationEncoder(self.dataset.n_genes, int(options.action_dim)).to(self.device)
        self.temporal_action = TemporalActionProjector(int(options.action_dim)).to(self.device)
        self.predictor = build_population_predictor(
            config=PopulationPredictorConfig(
                route=str(options.route),
                latent_dim=latent_dim,
                action_dim=int(options.action_dim),
                n_programs=int(options.n_programs),
                disable_kick=bool(options.disable_kick),
                disable_field=bool(options.disable_field),
                flat_action=bool(options.flat_action),
                adapter_components=str(options.adapter_components),
                k_samples=int(options.k_samples),
                calibrate_potential=bool(options.calibrate_potential),
                rollout_steps=int(options.rollout_steps),
                disable_rollout=bool(options.disable_rollout),
                disable_action_time=bool(options.disable_action_time),
                virtual_time_min=float(options.virtual_time_min),
                virtual_time_max=float(options.virtual_time_max),
                locked_time_transform=str(options.locked_time_transform),
                locked_time_scale=float(options.locked_time_scale),
            ),
            base_transition=self.base_transition,
        ).to(self.device)
        self.loss_computer = PopulationPerturbationLoss(PopulationLossWeights(
            latent_mmd=float(options.latent_mmd_weight),
            latent_w2=float(options.latent_w2_weight),
            expr_bulk=float(options.expr_bulk_weight),
            delta_bulk=float(options.delta_bulk_weight),
            de_bulk=float(options.de_bulk_weight),
            delta_cosine=float(options.delta_cosine_weight),
            sinkhorn_eps=float(options.sinkhorn_eps),
            sinkhorn_iters=int(options.sinkhorn_iters),
        ))
        if options.prediction_mode not in {"state", "delta_residual"}:
            raise ValueError("prediction_mode must be one of {'state', 'delta_residual'}")
        self.response_decoder = build_response_decoder(
            ResponseDecoderConfig(
                response_decoder=str(options.response_decoder),
                n_genes=self.dataset.n_genes,
                latent_dim=latent_dim,
                action_dim=int(options.action_dim),
                response_programs=int(options.response_programs),
                use_sparse_programs=bool(options.sparse_programs),
                nonnegative_basis=bool(options.nonnegative_basis),
                use_set_context=bool(options.set_context_decoder),
            )
        )
        if options.prediction_mode == "delta_residual" and self.response_decoder is None:
            raise ValueError("prediction_mode='delta_residual' requires --response-decoder")
        if self.response_decoder is not None:
            self.response_decoder = self.response_decoder.to(self.device)
        params = list(self.predictor.parameters()) + list(self.action_encoder.parameters()) + list(self.temporal_action.parameters())
        if self.response_decoder is not None:
            params += list(self.response_decoder.parameters())
        self.trainable_params = [p for p in params if p.requires_grad]
        self.opt = torch.optim.AdamW(self.trainable_params, lr=float(options.lr), betas=(0.9, 0.95), weight_decay=0.01)
        self.history: list[dict[str, Any]] = []

    def _encode(self, x: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            z, _ = self.vae.model.encode(torch.from_numpy(x).to(self.device), None)
        return z

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.model.decode(z, None)

    def _action(self, n: int, delta: float) -> torch.Tensor:
        gene_ids, signs, modality_ids, strengths, mask = self.dataset.action_gene_arrays(n)
        base_action = self.action_encoder(
            gene_ids=torch.from_numpy(gene_ids).to(self.device),
            signs=torch.from_numpy(signs).to(self.device),
            modality_ids=torch.from_numpy(modality_ids).to(self.device),
            strengths=torch.from_numpy(strengths).to(self.device),
            mask=torch.from_numpy(mask).to(self.device),
        )
        delta_t = torch.full((int(n),), float(delta), device=self.device, dtype=base_action.dtype)
        return self.temporal_action(base_action, delta_t)

    def _predictor_forward(self, z: torch.Tensor, action: torch.Tensor, delta: float):
        if getattr(self.predictor, "requires_delta", False):
            delta_t = torch.full((z.shape[0],), float(delta), device=z.device, dtype=z.dtype)
            return self.predictor(z, action, delta_t)
        return self.predictor(z, action)

    def _predict_x(self, source_x: torch.Tensor, pred_z: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        if self.options.prediction_mode == "state":
            return self._decode(pred_z), {}
        if self.response_decoder is None:
            raise RuntimeError("delta_residual mode requires a response decoder")
        return self.response_decoder(source_x, pred_z, action)

    def train_step(self, step: int) -> dict[str, Any]:
        time_value = float(self.rng.choice(np.asarray(self.dataset.train_times, dtype=np.float32)))
        control_x, target_x = self.dataset.sample_set_pair(time_value, int(self.options.set_size), self.rng)
        src_z = self._encode(control_x)
        target_z = self._encode(target_x)
        delta = time_value - float(self.options.source_time)
        action = self._action(src_z.shape[0], delta=delta)
        out = self._predictor_forward(src_z, action, delta=delta)
        source_tensor = torch.from_numpy(control_x).to(self.device)
        pred_x, decoder_info = self._predict_x(source_tensor, out.z, action)
        loss, loss_info = self.loss_computer(
            pred_z=out.z,
            target_z=target_z,
            pred_x=pred_x,
            target_x=torch.from_numpy(target_x).to(self.device),
            control_mean=torch.from_numpy(self.dataset.source_mean_for_time(time_value)).to(self.device),
            de_idx=torch.from_numpy(self.dataset.de_idx_by_time[time_value]).to(self.device),
        )
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.trainable_params, 1.0)
        self.opt.step()
        return {
            "step": int(step + 1),
            "time": float(time_value),
            "loss": float(loss.item()),
            **loss_info,
            **out.aux,
            **decoder_info,
        }

    def _predict_mean_for_time(self, time_value: float) -> np.ndarray:
        control_ids = np.asarray(self.dataset.source_indices_for_time(float(time_value)), dtype=np.int64)
        if int(self.options.max_eval_source_cells) > 0 and len(control_ids) > int(self.options.max_eval_source_cells):
            control_ids = self.rng.choice(control_ids, size=int(self.options.max_eval_source_cells), replace=False)
        chunks = []
        with torch.no_grad():
            for start in range(0, len(control_ids), int(self.options.eval_batch_size)):
                rows = control_ids[start:start + int(self.options.eval_batch_size)]
                x = self.dataset.load_rows(rows)
                z = self._encode(x)
                delta = float(time_value) - float(self.options.source_time)
                action = self._action(z.shape[0], delta=delta)
                pred_z = self._predictor_forward(z, action, delta=delta).z
                source_tensor = torch.from_numpy(x).to(self.device)
                pred_x, _ = self._predict_x(source_tensor, pred_z, action)
                chunks.append(pred_x.detach().cpu().numpy())
        return np.concatenate(chunks, axis=0).mean(axis=0).astype(np.float32)

    def evaluate(self, split: str) -> pd.DataFrame:
        times = self.dataset.train_times if split == "train" else self.dataset.eval_times
        rows = []
        self.predictor.eval()
        self.action_encoder.eval()
        self.temporal_action.eval()
        if self.response_decoder is not None:
            self.response_decoder.eval()
        for time_value in times:
            pred = self._predict_mean_for_time(float(time_value))
            target = self.dataset.target_mean_by_time[float(time_value)]
            top_idx = self.dataset.de_idx_by_time[float(time_value)]
            rows.append({
                "split": split,
                "time": float(time_value),
                "n_target": int(len(self.dataset.target_idx_by_time[float(time_value)])),
                **_metrics(pred, target, self.dataset.source_mean_for_time(float(time_value)), top_idx),
            })
        self.predictor.train()
        self.action_encoder.train()
        self.temporal_action.train()
        if self.response_decoder is not None:
            self.response_decoder.train()
        return pd.DataFrame(rows)

    def fit(self) -> dict[str, Any]:
        t0 = time.time()
        for step in range(int(self.options.max_steps)):
            row = self.train_step(step)
            self.history.append(row)
            if step == 0 or (step + 1) % 50 == 0:
                print(json.dumps(row, sort_keys=True), flush=True)
        pd.DataFrame(self.history).to_csv(self.output_dir / "history.tsv", sep="\t", index=False)
        eval_df = pd.concat([self.evaluate("train"), self.evaluate("eval")], ignore_index=True)
        eval_df.to_csv(self.output_dir / "metrics.tsv", sep="\t", index=False)
        self.save_checkpoint(self.output_dir / "model.pt")
        summary = {
            "route": self.options.route,
            "init_name": self.options.init_name,
            "init_checkpoint": str(self.options.init_checkpoint) if self.options.init_checkpoint else None,
            "load_info": self.load_info,
            "max_steps": int(self.options.max_steps),
            "set_size": int(self.options.set_size),
            "elapsed_s": float(time.time() - t0),
            "prediction_mode": str(self.options.prediction_mode),
            "response_decoder": str(self.options.response_decoder),
            "loss_weights": {
                "latent_mmd": float(self.options.latent_mmd_weight),
                "latent_w2": float(self.options.latent_w2_weight),
                "expr_bulk": float(self.options.expr_bulk_weight),
                "delta_bulk": float(self.options.delta_bulk_weight),
                "de_bulk": float(self.options.de_bulk_weight),
                "delta_cosine": float(self.options.delta_cosine_weight),
            },
            "dataset": {
                "adata": str(self.options.adata),
                "n_cells": int(self.dataset.adata.n_obs),
                "n_genes": int(self.dataset.adata.n_vars),
                "n_vocab_genes": int(self.dataset.n_genes),
                "n_mapped_genes": int(len(self.dataset.mapped_vocab_idx)),
                "control_cells": int(len(self.dataset.control_idx)),
                "source_policy": str(self.options.source_policy),
                "train_times": [float(t) for t in self.dataset.train_times],
                "eval_times": [float(t) for t in self.dataset.eval_times],
            },
            "metrics": eval_df.to_dict(orient="records"),
        }
        write_json(self.output_dir / "summary.json", summary)
        return summary

    def save_checkpoint(self, path: str | Path) -> None:
        torch.save({
            "predictor_state_dict": self.predictor.state_dict(),
            "action_encoder_state_dict": self.action_encoder.state_dict(),
            "temporal_action_state_dict": self.temporal_action.state_dict(),
            "response_decoder_state_dict": self.response_decoder.state_dict() if self.response_decoder is not None else None,
            "config": {
                "route": self.options.route,
                "init_name": self.options.init_name,
                "action_dim": int(self.options.action_dim),
                "n_programs": int(self.options.n_programs),
                "max_steps": int(self.options.max_steps),
                "prediction_mode": str(self.options.prediction_mode),
                "response_decoder": str(self.options.response_decoder),
                "response_programs": int(self.options.response_programs),
                "source_policy": str(self.options.source_policy),
                "train_times": [float(t) for t in self.options.train_times],
                "eval_times": [float(t) for t in self.options.eval_times],
                "locked_time_transform": str(self.options.locked_time_transform),
                "locked_time_scale": float(self.options.locked_time_scale),
            },
        }, path)


def train_time_perturbation(options: TimePerturbationTrainOptions) -> dict[str, Any]:
    return TimePerturbationTrainer(options).fit()
