#!/usr/bin/env python
"""Run GEARS with foundation cell-state embeddings injected into its hidden state."""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if np.isnan(value) or np.isinf(value) else value
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def _mean_metric(metric_by_pert: dict[str, dict[str, float]], perts: list[str], metric: str) -> float | None:
    values = [metric_by_pert[p][metric] for p in perts if p in metric_by_pert and metric in metric_by_pert[p]]
    return float(np.mean(values)) if values else None


def _subgroup_summary(metric_by_pert: dict[str, dict[str, float]], subgroup: dict, metrics: list[str]) -> dict:
    out = {}
    for name, perts in subgroup.get("test_subgroup", {}).items():
        out[name] = {metric: _mean_metric(metric_by_pert, list(perts), metric) for metric in metrics}
        out[name]["n_conditions"] = int(len(perts))
    return out


def _condition_rows_from_loader(pert_data) -> list:
    seen: set[int] = set()
    rows = []
    for loader in pert_data.dataloader.values():
        for graph in loader.dataset:
            obj_id = id(graph)
            if obj_id in seen:
                continue
            seen.add(obj_id)
            rows.append(graph)
    return rows


class FoundationEmbeddingExtractor:
    def __init__(self, args, pert_data) -> None:
        from cellworldmodel.benchmark.registry import build_model
        from cellworldmodel.foundation.dynamics_train import build_foundation_dynamics_cfg
        from cellworldmodel.foundation.gene_space import build_source_to_vocab, foundation_gene_names_from_vocab
        from cellworldmodel.foundation.vae_eval import load_vae_checkpoint

        self.args = args
        self.device = torch.device(args.device)
        self.vae = load_vae_checkpoint(args.vae_checkpoint, self.device)
        for param in self.vae.model.parameters():
            param.requires_grad_(False)
        self.vae.model.eval()
        self.ours_genes = foundation_gene_names_from_vocab(args.gene_vocab)
        self.source_to_vocab = build_source_to_vocab(pert_data.gene_names.values.tolist(), self.ours_genes)
        self.n_ours_genes = len(self.ours_genes)
        self.transition = None
        if args.foundation_source in {"static_dit", "dynamics_dit"}:
            if args.dynamics_checkpoint is None:
                raise ValueError(f"--dynamics-checkpoint is required for foundation_source={args.foundation_source}")
            ckpt = torch.load(args.dynamics_checkpoint, map_location="cpu", weights_only=False)
            ckpt_cfg = dict(ckpt.get("config", {}))
            if "train_cfg" in ckpt_cfg:
                method = str(ckpt_cfg.get("method", "m10"))
                train_cfg = dict(ckpt_cfg["train_cfg"])
                tau_init = 1.0
            else:
                method, train_cfg, tau_init = build_foundation_dynamics_cfg(
                    experiment=args.experiment,
                    dit_size=args.dit_size,
                    batch_size=args.batch_size,
                    k_samples=args.foundation_k_samples,
                    lr=args.lr,
                )
            train_cfg["action_dim"] = 0
            train_cfg["loss_balancer"] = "fixed"
            self.transition = build_model(
                method,
                int(self.vae.config["latent_dim"]),
                train_cfg,
                tau_init=tau_init,
            ).to(self.device)
            self.transition.load_state_dict(ckpt["model_state_dict"])
            self.transition.eval()
            for param in self.transition.parameters():
                param.requires_grad_(False)

    def _to_foundation_expression(self, x_gears: torch.Tensor) -> torch.Tensor:
        x_gears = x_gears.to(self.device, dtype=torch.float32)
        out = torch.zeros((x_gears.shape[0], self.n_ours_genes), device=self.device, dtype=torch.float32)
        keep = self.source_to_vocab >= 0
        if np.any(keep):
            source_idx = torch.as_tensor(np.flatnonzero(keep), device=self.device, dtype=torch.long)
            target_idx = torch.as_tensor(self.source_to_vocab[keep], device=self.device, dtype=torch.long)
            out[:, target_idx] = x_gears[:, source_idx]
        return out

    @torch.no_grad()
    def encode(self, x_gears: torch.Tensor) -> torch.Tensor:
        x_full = self._to_foundation_expression(x_gears)
        z, _ = self.vae.model.encode(x_full, None)
        if self.transition is None:
            return z
        delta = torch.ones(z.shape[0], device=self.device, dtype=z.dtype)
        return self.transition.predict_mean(z, delta, action=None, n_mc=int(self.args.foundation_k_samples))


def attach_foundation_embeddings(pert_data, extractor: FoundationEmbeddingExtractor, batch_size: int) -> dict[str, Any]:
    graphs = _condition_rows_from_loader(pert_data)
    t0 = time.time()
    for start in range(0, len(graphs), batch_size):
        chunk = graphs[start:start + batch_size]
        x = torch.stack([graph.x.reshape(-1) for graph in chunk], dim=0)
        z = extractor.encode(x).detach().cpu().float()
        for graph, row in zip(chunk, z):
            graph.foundation_z = row
    z_all = torch.stack([graph.foundation_z for graph in graphs], dim=0)
    return {
        "n_graphs": int(len(graphs)),
        "latent_dim": int(z_all.shape[1]),
        "elapsed_s": float(time.time() - t0),
        "mean": float(z_all.mean().item()),
        "std": float(z_all.std().item()),
    }


def make_foundation_gears_model(base_cls, *, foundation_dim: int, foundation_mode: str):
    class FoundationGEARSModel(base_cls):
        def __init__(self, args):
            super().__init__(args)
            self.foundation_mode = str(foundation_mode)
            self.foundation_dim = int(foundation_dim)
            hidden_size = int(args["hidden_size"])
            self.foundation_proj = nn.Sequential(
                nn.LayerNorm(self.foundation_dim),
                nn.Linear(self.foundation_dim, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )

        def forward(self, data):
            x, pert_idx = data.x, data.pert_idx
            if self.no_perturb:
                out = x.reshape(-1, 1)
                out = torch.split(torch.flatten(out), self.num_genes)
                return torch.stack(out)
            num_graphs = len(data.batch.unique())
            if not hasattr(data, "foundation_z"):
                raise ValueError("GEARS batch is missing foundation_z; attach embeddings before training.")
            foundation_z = data.foundation_z.to(self.args["device"]).reshape(num_graphs, self.foundation_dim)
            foundation_state = self.foundation_proj(foundation_z)

            gene_ids = torch.arange(self.num_genes, device=self.args["device"]).repeat(num_graphs)
            emb = self.gene_emb(gene_ids)
            emb = self.bn_emb(emb)
            base_emb = self.emb_trans(emb)

            if self.foundation_mode == "replace":
                base_emb = foundation_state[:, None, :].expand(num_graphs, self.num_genes, -1).reshape(num_graphs * self.num_genes, -1)
            elif self.foundation_mode == "add":
                base_emb = base_emb + foundation_state.repeat_interleave(self.num_genes, dim=0)
            else:
                raise ValueError(f"Unknown foundation_mode={self.foundation_mode!r}")

            pos_emb = self.emb_pos(gene_ids)
            for idx, layer in enumerate(self.layers_emb_pos):
                pos_emb = layer(pos_emb, self.G_coexpress, self.G_coexpress_weight)
                if idx < len(self.layers_emb_pos) - 1:
                    pos_emb = pos_emb.relu()

            base_emb = base_emb + 0.2 * pos_emb
            base_emb = self.emb_trans_v2(base_emb)

            pert_index = []
            for idx, i in enumerate(pert_idx):
                for j in i:
                    if j != -1:
                        pert_index.append([idx, j])
            pert_index = torch.tensor(pert_index).T

            pert_global_emb = self.pert_emb(torch.arange(self.num_perts, device=self.args["device"]))
            for idx, layer in enumerate(self.sim_layers):
                pert_global_emb = layer(pert_global_emb, self.G_sim, self.G_sim_weight)
                if idx < self.num_layers - 1:
                    pert_global_emb = pert_global_emb.relu()

            base_emb = base_emb.reshape(num_graphs, self.num_genes, -1)
            if pert_index.shape[0] != 0:
                pert_track = {}
                for i, j in enumerate(pert_index[0]):
                    if j.item() in pert_track:
                        pert_track[j.item()] = pert_track[j.item()] + pert_global_emb[pert_index[1][i]]
                    else:
                        pert_track[j.item()] = pert_global_emb[pert_index[1][i]]
                if len(list(pert_track.values())) > 0:
                    if len(list(pert_track.values())) == 1:
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values()) * 2))
                    else:
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values())))
                    for idx, j in enumerate(pert_track.keys()):
                        base_emb[j] = base_emb[j] + emb_total[idx]

            base_emb = base_emb.reshape(num_graphs * self.num_genes, -1)
            base_emb = self.bn_pert_base(base_emb)
            base_emb = self.transform(base_emb)
            out = self.recovery_w(base_emb)
            out = out.reshape(num_graphs, self.num_genes, -1)
            out = out.unsqueeze(-1) * self.indv_w1
            w = torch.sum(out, axis=2)
            out = w + self.indv_b1

            cross_gene_embed = self.cross_gene_state(out.reshape(num_graphs, self.num_genes, -1).squeeze(2))
            cross_gene_embed = cross_gene_embed.repeat(1, self.num_genes)
            cross_gene_embed = cross_gene_embed.reshape([num_graphs, self.num_genes, -1])
            cross_gene_out = torch.cat([out, cross_gene_embed], 2)
            cross_gene_out = cross_gene_out * self.indv_w2
            cross_gene_out = torch.sum(cross_gene_out, axis=2)
            out = cross_gene_out + self.indv_b2
            out = out.reshape(num_graphs * self.num_genes, -1) + x.reshape(-1, 1)
            out = torch.split(torch.flatten(out), self.num_genes)

            if self.uncertainty:
                out_logvar = self.uncertainty_w(base_emb)
                out_logvar = torch.split(torch.flatten(out_logvar), self.num_genes)
                return torch.stack(out), torch.stack(out_logvar)
            return torch.stack(out)

    return FoundationGEARSModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gears-repo", type=Path, default=Path("3rdparty/GEARS"))
    parser.add_argument("--data-dir", type=Path, default=Path("output/gears/data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-name", default="norman")
    parser.add_argument("--split", default="simulation")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gene-vocab", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--dynamics-checkpoint", type=Path, default=None)
    parser.add_argument("--foundation-source", choices=["vae", "static_dit", "dynamics_dit"], required=True)
    parser.add_argument("--foundation-mode", choices=["add", "replace"], default="replace")
    parser.add_argument("--foundation-k-samples", type=int, default=2)
    parser.add_argument("--foundation-embed-batch-size", type=int, default=256)
    parser.add_argument("--experiment", default="g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw")
    parser.add_argument("--dit-size", default="small", choices=["tiny", "small", "base"])
    parser.add_argument("--save-test-res", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(args.gears_repo.resolve()))
    from gears import GEARS, PertData
    from gears.inference import compute_metrics, deeper_analysis, evaluate, non_dropout_analysis
    import gears.gears as gears_module
    from gears.model import GEARS_Model

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    pert_data = PertData(str(args.data_dir))
    pert_data.load(data_name=args.data_name)
    pert_data.prepare_split(split=args.split, seed=args.seed)
    pert_data.get_dataloader(batch_size=args.batch_size, test_batch_size=args.test_batch_size)

    extractor = FoundationEmbeddingExtractor(args, pert_data)
    embed_summary = attach_foundation_embeddings(pert_data, extractor, args.foundation_embed_batch_size)

    gears_module.GEARS_Model = make_foundation_gears_model(
        GEARS_Model,
        foundation_dim=int(embed_summary["latent_dim"]),
        foundation_mode=args.foundation_mode,
    )
    model = GEARS(pert_data, device=args.device, weight_bias_track=False)
    model.model_initialize(hidden_size=args.hidden_size)
    model.train(epochs=args.epochs, lr=args.lr)

    summary: dict[str, Any] = {
        "data_name": args.data_name,
        "split": args.split,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "test_batch_size": args.test_batch_size,
        "hidden_size": args.hidden_size,
        "lr": args.lr,
        "device": args.device,
        "foundation_source": args.foundation_source,
        "foundation_mode": args.foundation_mode,
        "vae_checkpoint": str(args.vae_checkpoint),
        "dynamics_checkpoint": str(args.dynamics_checkpoint) if args.dynamics_checkpoint else None,
        "embedding": embed_summary,
        "set2conditions": pert_data.set2conditions,
    }

    if "test_loader" in pert_data.dataloader:
        test_res = evaluate(
            pert_data.dataloader["test_loader"],
            model.best_model,
            model.config["uncertainty"],
            args.device,
        )
        test_metrics, test_pert_metrics = compute_metrics(test_res)
        deeper = deeper_analysis(model.adata, test_res)
        non_dropout = non_dropout_analysis(model.adata, test_res)
        summary["test_metrics"] = test_metrics
        summary["subgroup_metrics"] = _subgroup_summary(
            test_pert_metrics,
            model.subgroup or {},
            ["mse", "pearson", "mse_de", "pearson_de"],
        )
        summary["subgroup_deeper_metrics"] = _subgroup_summary(
            deeper,
            model.subgroup or {},
            ["pearson_delta", "pearson_delta_de", "mse_top20_de", "pearson_delta_top20_de"],
        )
        summary["subgroup_non_dropout_metrics"] = _subgroup_summary(
            non_dropout,
            model.subgroup or {},
            [
                "frac_opposite_direction_top20_non_dropout",
                "frac_sigma_below_1_non_dropout",
                "mse_top20_de_non_dropout",
                "pearson_delta_top20_de_non_dropout",
            ],
        )
        with (args.output_dir / "test_pert_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(test_pert_metrics), handle, indent=2, sort_keys=True)
        with (args.output_dir / "deeper_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(deeper), handle, indent=2, sort_keys=True)
        with (args.output_dir / "non_dropout_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(non_dropout), handle, indent=2, sort_keys=True)
        if args.save_test_res:
            with (args.output_dir / "test_res.pkl").open("wb") as handle:
                pickle.dump(test_res, handle)

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(summary), handle, indent=2, sort_keys=True)
    model.save_model(str(args.output_dir / "model"))
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
