#!/usr/bin/env python
"""Run CellStream-style downstream tasks for CellStream and CWM models."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from cellworldmodel.benchmark.cellstream_adapter import (
    CELLSTREAM_ROOT,
    CellStreamDataset,
    CellStreamTimepointAdapter,
    load_cellstream_dataset,
)
from cellworldmodel.benchmark.common_metrics import mmd2_unbiased_multi_sigma, sinkhorn_w2
from cellworldmodel.benchmark.configs import DATASET_CONFIGS
from cellworldmodel.benchmark.experiment_registry import EXPERIMENTS
from cellworldmodel.benchmark.registry import build_model
from cellworldmodel.evaluation.cellstream_metrics import (
    distance_correlation,
    temporal_consistency_knn,
    temporal_consistency_radius,
    velocity_consistency_knn,
    velocity_consistency_radius,
)
from cellworldmodel.evaluation.prediction import predict_at_delta
from cellworldmodel.training.benchmark_loop import train_method
from cellworldmodel.training.checkpointing import checkpoint_model_config, load_shape_matched_checkpoint
from cellworldmodel.training.transition_sampler import TimepointTransitionSampler


ROOT = Path(__import__("os").environ.get("CHREODE_ROOT", "."))
INIT_CHECKPOINTS = {
    "scratch": None,
    "a1_static": ROOT / "output/foundation/genhui_v1/dynamics/vae2_staticdit2/model.pt",
    "a2_dynamics": ROOT / "output/foundation/genhui_v1/dynamics/vae2_dynamicsdit2/model.pt",
}


def _cellstream_modules():
    path = CELLSTREAM_ROOT / "CellStream"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    from Autoencoder import Autoencoder  # noqa: WPS433
    from Net import Net_UOT  # noqa: WPS433
    return Autoencoder, Net_UOT


def _cellstream_param(dataset: str) -> tuple[str, str]:
    key = dataset.lower()
    if key in {"sim", "simdata", "simdata2d_example"}:
        return "SimData2D_example", "example"
    if key == "emt":
        return "EMT", "example"
    if key == "ipsc":
        return "ipsc", "ipsc_final"
    if key == "mosta":
        return "mosta", "final"
    raise ValueError(dataset)


class LoadedCellStream:
    def __init__(self, dataset: CellStreamDataset, device: torch.device) -> None:
        Autoencoder, NetUOT = _cellstream_modules()
        data_name, param_name = _cellstream_param(dataset.name)
        self.dataset = dataset
        self.device = device
        ae_state = torch.load(
            CELLSTREAM_ROOT / "params" / data_name / f"AE_{param_name}.pth",
            map_location=device,
            weights_only=False,
        )
        net_state = torch.load(
            CELLSTREAM_ROOT / "params" / data_name / f"net_{param_name}.pth",
            map_location=device,
            weights_only=False,
        )
        ae_hidden = int(ae_state["encoder.0.weight"].shape[0])
        net_hidden = int(net_state["v.net.0.0.weight"].shape[0])
        self.net = NetUOT(in_out_dim=2, hidden_dim=net_hidden, n_hiddens=4).to(device)
        self.autoencoder = Autoencoder(input_dim=dataset.dim + 1, hidden_dim=ae_hidden, latent_dim=2).to(device)
        self.autoencoder.load_state_dict(ae_state)
        self.net.load_state_dict(net_state)
        self.autoencoder.eval()
        self.net.eval()

    def info_tensor(self) -> torch.Tensor:
        info = np.concatenate([self.dataset.labels[:, None], self.dataset.values], axis=1).astype(np.float32)
        return torch.from_numpy(info).to(self.device)

    @torch.no_grad()
    def encode(self) -> np.ndarray:
        return self.autoencoder.encoded(self.info_tensor()).detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def velocity(self, z: np.ndarray | None = None) -> np.ndarray:
        z_t = torch.from_numpy(self.encode() if z is None else z).to(self.device)
        labels = torch.from_numpy(self.dataset.labels.astype(np.float32)).to(self.device)
        out = []
        for start in range(0, z_t.shape[0], 1024):
            zz = z_t[start:start + 1024]
            tt = labels[start:start + 1024]
            vals = [self.net.v(tt[i], zz[i:i + 1]).squeeze(0) for i in range(zz.shape[0])]
            out.append(torch.stack(vals, dim=0).detach().cpu())
        return torch.cat(out, dim=0).numpy().astype(np.float32)

    @torch.no_grad()
    def rollout(self, z0: np.ndarray, delta: float, *, steps: int = 50) -> np.ndarray:
        z = torch.from_numpy(z0.astype(np.float32)).to(self.device)
        t = torch.zeros((), dtype=torch.float32, device=self.device)
        dt = float(delta) / float(max(1, steps))
        for _ in range(max(1, steps)):
            z = z + self.net.v(t, z) * dt
            t = t + dt
        return z.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def decode_values(self, z: np.ndarray) -> np.ndarray:
        z_t = torch.from_numpy(z.astype(np.float32)).to(self.device)
        decoded = self.autoencoder.decoded(z_t).detach().cpu().numpy().astype(np.float32)
        return decoded[:, 1:]


def _latent_true_velocity(cs: LoadedCellStream) -> np.ndarray | None:
    ds = cs.dataset
    if ds.real_v is None:
        return None
    info = cs.info_tensor().detach()
    real = torch.from_numpy(np.concatenate([np.ones((ds.values.shape[0], 1), dtype=np.float32), ds.real_v], axis=1)).to(cs.device)
    out = []
    for i in range(info.shape[0]):
        x = info[i].detach().clone().requires_grad_(True)

        def fn(v):
            return cs.autoencoder.encoded(v.unsqueeze(0)).squeeze(0)

        jac = torch.autograd.functional.jacobian(fn, x)
        out.append((jac @ real[i]).detach().cpu())
    return torch.stack(out, dim=0).numpy().astype(np.float32)


def _dataset_from_space(base: CellStreamDataset, coords: np.ndarray) -> CellStreamDataset:
    coords_by_t = {
        float(t): coords[np.isclose(base.labels, float(t))].astype(np.float32).copy()
        for t in sorted(np.unique(base.labels.astype(float)))
    }
    return replace(base, values=coords.astype(np.float32), coords_by_t=coords_by_t, real_v=None, real_g=None)


def _build_cwm_cfg(experiment: str, model_config_checkpoint: Path | None = None) -> dict:
    cfg = dict(DATASET_CONFIGS["paper_weinreb_scvi128"])
    EXPERIMENTS[experiment].apply_to_cfg(cfg)
    if model_config_checkpoint is not None:
        cfg.update(checkpoint_model_config(model_config_checkpoint))
    return cfg


def build_cwm(space_ds: CellStreamDataset, *, experiment: str, seed: int,
              device: torch.device, init: str,
              model_config_checkpoint: Path | None = None) -> tuple[torch.nn.Module, dict]:
    if model_config_checkpoint is None and init != "scratch":
        model_config_checkpoint = INIT_CHECKPOINTS[init]
    cfg = _build_cwm_cfg(experiment, model_config_checkpoint)
    adapter = CellStreamTimepointAdapter(space_ds)
    tau_init = float((adapter.timepoints[-1] - adapter.timepoints[0]) / np.log(2))
    torch.manual_seed(seed)
    model = build_model("m10", adapter.dim, cfg, tau_init=tau_init).to(device)
    init_info = {"init": init, "checkpoint": None, "loaded": 0, "target": len(model.state_dict())}
    ckpt = INIT_CHECKPOINTS[init]
    if ckpt is not None:
        info = load_shape_matched_checkpoint(model, ckpt, min_match_ratio=0.9)
        init_info.update(info)
        init_info["checkpoint"] = str(ckpt)
    return model, {"cfg": cfg, "tau_init": tau_init, "init_info": init_info}


def train_cwm(space_ds: CellStreamDataset, *, experiment: str, epochs: int,
              seed: int, device: torch.device, init: str,
              model_config_checkpoint: Path | None = None) -> tuple[torch.nn.Module, dict]:
    model, info = build_cwm(
        space_ds,
        experiment=experiment,
        seed=seed,
        device=device,
        init=init,
        model_config_checkpoint=model_config_checkpoint,
    )
    cfg = info["cfg"]
    adapter = CellStreamTimepointAdapter(space_ds)
    sampler = TimepointTransitionSampler(adapter, split_seed=seed, split_ratios=(0.7, 0.1, 0.2))
    history = train_method(
        "m10",
        adapter,
        model,
        device,
        cfg,
        epochs=epochs,
        seed=seed,
        log_every=max(1, epochs // 5),
        sampler=sampler,
    )
    info["history"] = history
    return model, info


@torch.no_grad()
def cwm_velocity(model, z: np.ndarray, *, delta: float, cfg: dict, device: torch.device) -> np.ndarray:
    src = torch.from_numpy(z.astype(np.float32)).to(device)
    outs = []
    for start in range(0, src.shape[0], 512):
        zz = src[start:start + 512]
        d = torch.full((zz.shape[0],), float(delta), device=device, dtype=zz.dtype)
        if hasattr(model, "predict_mean"):
            try:
                pred = model.predict_mean(zz, d, n_mc=32)
            except TypeError:
                pred = model.predict_mean(zz, d)
        else:
            pred = predict_at_delta(model, zz, delta, cfg["K"], z.shape[1], device).reshape(zz.shape[0], cfg["K"], -1).mean(dim=1)
        outs.append(((pred - zz) / float(delta)).detach().cpu())
    return torch.cat(outs, dim=0).numpy().astype(np.float32)


def _compute_consistency(z: np.ndarray, labels: np.ndarray, velocity: np.ndarray, *, prefix: str,
                         max_cells: int = 4000) -> dict:
    if z.shape[0] > int(max_cells):
        idx = np.linspace(0, z.shape[0] - 1, int(max_cells)).astype(int)
        z = z[idx]
        labels = labels[idx]
        velocity = velocity[idx]
    return {
        f"{prefix}_tc_radius005": temporal_consistency_radius(z, labels, radius=0.05),
        f"{prefix}_vc_radius005": velocity_consistency_radius(z, labels, velocity, radius=0.05),
        f"{prefix}_tc_knn20": temporal_consistency_knn(z, labels, k=20),
        f"{prefix}_vc_knn20": velocity_consistency_knn(z, labels, velocity, k=20),
    }


def _endpoint_metrics(space_ds: CellStreamDataset, predict_fn, *, device: torch.device, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    source_t = space_ds.timepoints[0]
    source_split = space_ds.splits_by_t[source_t].test
    src_all = space_ds.coords_by_t[source_t][source_split]
    if src_all.shape[0] > 256:
        src_all = src_all[np.sort(rng.choice(src_all.shape[0], size=256, replace=False))]
    rows = []
    for target_t in space_ds.timepoints[1:]:
        target_split = space_ds.splits_by_t[target_t].test
        target = space_ds.coords_by_t[target_t][target_split]
        if target.shape[0] > 512:
            target = target[np.sort(rng.choice(target.shape[0], size=512, replace=False))]
        pred = predict_fn(src_all, float(target_t - source_t), target_t)
        if pred.shape[0] > 512:
            pred = pred[np.sort(rng.choice(pred.shape[0], size=512, replace=False))]
        pred_t = torch.from_numpy(pred.astype(np.float32)).to(device)
        tgt_t = torch.from_numpy(target.astype(np.float32)).to(device)
        rows.append({
            "target_time": float(target_t),
            "n_pred": int(pred.shape[0]),
            "n_target": int(target.shape[0]),
            "sinkhorn_w2": float(sinkhorn_w2(pred_t, tgt_t, epsilon=0.05, num_iters=50).item()),
            "mmd2": float(mmd2_unbiased_multi_sigma(pred_t, tgt_t).item()),
        })
    return rows


def _linear_predictor(space_ds: CellStreamDataset):
    source_t = space_ds.timepoints[0]
    means = {t: space_ds.coords_by_t[t][space_ds.splits_by_t[t].train].mean(axis=0) for t in space_ds.timepoints}

    def predict(src: np.ndarray, _delta: float, target_t: float) -> np.ndarray:
        return (src + (means[float(target_t)] - means[source_t])[None, :]).astype(np.float32)

    return predict


def _plot_space(space_ds: CellStreamDataset, velocities: dict[str, np.ndarray],
                predictors: dict[str, callable], out_dir: Path, *, title: str, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    z = space_ds.values
    labels = space_ds.labels
    if z.shape[1] > 2:
        pca = PCA(n_components=2, random_state=seed).fit(z)
        z2 = pca.transform(z)
        vel2 = {k: pca.components_ @ v.T for k, v in velocities.items()}
        vel2 = {k: v.T for k, v in vel2.items()}
    else:
        pca = None
        z2 = z
        vel2 = velocities
    for method, vel in vel2.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(z2[:, 0], z2[:, 1], c=labels, s=4, cmap="viridis", alpha=0.5)
        idx = np.linspace(0, z2.shape[0] - 1, min(180, z2.shape[0])).astype(int)
        ax.quiver(z2[idx, 0], z2[idx, 1], vel[idx, 0], vel[idx, 1], color="black", alpha=0.55, width=0.0025)
        ax.set_title(f"{title}: {method} velocity")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(sc, ax=ax, label="time")
        fig.tight_layout()
        fig.savefig(out_dir / f"{method}_velocity.png", dpi=180)
        plt.close(fig)


def _evaluate_cwm_model(base: CellStreamDataset, space_name: str, space_ds: CellStreamDataset,
                        model, cfg: dict, *, method_name: str, device: torch.device,
                        seed: int, train_epochs: int | None, tau_init: float,
                        init_info: dict,
                        real_velocity: np.ndarray | None = None) -> tuple[dict, list[dict], np.ndarray, callable]:
    delta_v = _velocity_delta(space_ds)
    vel = cwm_velocity(model, space_ds.values, delta=delta_v, cfg=cfg, device=device)
    row = {
        "dataset": base.name,
        "space": space_name,
        "method": method_name,
        **_compute_consistency(space_ds.values, space_ds.labels, vel, prefix="metric"),
        "velocity_accuracy": (
            distance_correlation(real_velocity, vel) if real_velocity is not None
            else float("nan")
        ),
        "train_epochs": train_epochs,
        "tau_init": float(tau_init),
        "init_loaded": int(init_info.get("loaded", 0)),
        "init_target": int(init_info.get("target", 0)),
        "init_checkpoint": init_info.get("checkpoint"),
    }

    def cwm_predict(src: np.ndarray, delta: float, _target_t: float) -> np.ndarray:
        src_t = torch.from_numpy(src.astype(np.float32)).to(device)
        pred = predict_at_delta(model, src_t, delta, cfg["K"], space_ds.dim, device)
        return pred.detach().cpu().numpy().astype(np.float32)

    endpoint_rows = []
    for erow in _endpoint_metrics(space_ds, cwm_predict, device=device, seed=seed):
        erow.update({"dataset": base.name, "space": space_name, "method": method_name})
        endpoint_rows.append(erow)
    return row, endpoint_rows, vel, cwm_predict

    source_t = space_ds.timepoints[0]
    source = space_ds.coords_by_t[source_t][space_ds.splits_by_t[source_t].test]
    rng = np.random.default_rng(seed)
    if source.shape[0] > 35:
        source = source[np.sort(rng.choice(source.shape[0], size=35, replace=False))]
    for method, fn in predictors.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(z2[:, 0], z2[:, 1], c=labels, s=4, cmap="viridis", alpha=0.25)
        seqs = []
        for target_t in space_ds.timepoints:
            if target_t == source_t:
                pred = source
            else:
                pred = fn(source, float(target_t - source_t), target_t)
            seqs.append(pred)
        for i in range(source.shape[0]):
            pts = np.stack([seq[i] for seq in seqs], axis=0)
            pts2 = pca.transform(pts) if pca is not None else pts
            ax.plot(pts2[:, 0], pts2[:, 1], color="black", alpha=0.45, linewidth=0.8)
        ax.set_title(f"{title}: {method} rollout")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(out_dir / f"{method}_trajectory.png", dpi=180)
        plt.close(fig)


def run_dataset(args: argparse.Namespace, dataset_name: str, device: torch.device) -> tuple[list[dict], list[dict]]:
    base = load_cellstream_dataset(dataset_name, seed=args.seed)
    cs = LoadedCellStream(base, device)
    cs_z = cs.encode()
    cs_v = cs.velocity(cs_z)
    latent_true_v = _latent_true_velocity(cs) if base.real_v is not None else None
    spaces = {
        "native": base,
        "cellstream_latent": _dataset_from_space(base, cs_z),
    }
    requested_spaces = set(args.spaces)
    spaces = {name: ds for name, ds in spaces.items() if name in requested_spaces}
    rows = []
    endpoint_rows = []

    if args.run_cellstream_baseline:
        rows.append({
            "dataset": base.name,
            "space": "cellstream_latent",
            "method": "CellStream_pretrained",
            **_compute_consistency(cs_z, base.labels, cs_v, prefix="metric"),
            "velocity_accuracy": (
                distance_correlation(latent_true_v, cs_v) if latent_true_v is not None else float("nan")
            ),
        })
        cs_endpoint = _endpoint_metrics(
            spaces["cellstream_latent"],
            lambda src, delta, _t: cs.rollout(src, delta),
            device=device,
            seed=args.seed,
        )
        for row in cs_endpoint:
            row.update({"dataset": base.name, "space": "cellstream_latent", "method": "CellStream_pretrained"})
            endpoint_rows.append(row)
        native_endpoint = _endpoint_metrics(
            spaces["native"],
            lambda src, delta, _t: cs.decode_values(cs.rollout(cs.autoencoder.encoded(
                torch.from_numpy(np.concatenate([
                    np.full((src.shape[0], 1), spaces["native"].timepoints[0], dtype=np.float32),
                    src.astype(np.float32),
                ], axis=1)).to(device)
            ).detach().cpu().numpy().astype(np.float32), delta)),
            device=device,
            seed=args.seed,
        )
        for row in native_endpoint:
            row.update({"dataset": base.name, "space": "native", "method": "CellStream_decoded"})
            endpoint_rows.append(row)

    for space_name, space_ds in spaces.items():
        linear_vel = _linear_velocity(space_ds)
        rows.append({
            "dataset": base.name,
            "space": space_name,
            "method": "linear_time_delta",
            **_compute_consistency(space_ds.values, space_ds.labels, linear_vel, prefix="metric"),
            "velocity_accuracy": (
                distance_correlation(base.real_v, linear_vel) if space_name == "native" and base.real_v is not None
                else distance_correlation(latent_true_v, linear_vel) if space_name == "cellstream_latent" and latent_true_v is not None
                else float("nan")
            ),
        })
        for row in _endpoint_metrics(space_ds, _linear_predictor(space_ds), device=device, seed=args.seed):
            row.update({"dataset": base.name, "space": space_name, "method": "linear_time_delta"})
            endpoint_rows.append(row)

        plot_velocities = {"linear_time_delta": linear_vel}
        plot_predictors = {"linear_time_delta": _linear_predictor(space_ds)}
        for init in args.cwm_inits:
            real_velocity = base.real_v if space_name == "native" else latent_true_v
            model, info = build_cwm(
                space_ds,
                experiment=args.experiment,
                seed=args.seed,
                device=device,
                init=init,
                model_config_checkpoint=args.model_config_checkpoint,
            )
            if args.include_zero_shot:
                method_name = f"CWM_{init}_zero_shot_partial"
                zrow, z_endpoint, zvel, zpredict = _evaluate_cwm_model(
                    base, space_name, space_ds, model, info["cfg"],
                    method_name=method_name, device=device, seed=args.seed,
                    train_epochs=0, tau_init=float(info["tau_init"]), init_info=info["init_info"],
                    real_velocity=real_velocity,
                )
                rows.append(zrow)
                endpoint_rows.extend(z_endpoint)
                plot_velocities[method_name] = zvel
                plot_predictors[method_name] = zpredict
            if args.epochs > 0:
                model, info = train_cwm(
                    space_ds,
                    experiment=args.experiment,
                    epochs=args.epochs,
                    seed=args.seed,
                    device=device,
                    init=init,
                    model_config_checkpoint=args.model_config_checkpoint,
                )
                method_name = f"CWM_{init}_finetune_{args.experiment}"
                frow, f_endpoint, fvel, fpredict = _evaluate_cwm_model(
                    base, space_name, space_ds, model, info["cfg"],
                    method_name=method_name, device=device, seed=args.seed,
                    train_epochs=int(args.epochs), tau_init=float(info["tau_init"]), init_info=info["init_info"],
                    real_velocity=real_velocity,
                )
                rows.append(frow)
                endpoint_rows.extend(f_endpoint)
                plot_velocities[method_name] = fvel
                plot_predictors[method_name] = fpredict
        if args.make_plots:
            _plot_space(
                space_ds,
                velocities=plot_velocities,
                predictors=plot_predictors,
                out_dir=args.output_dir / base.name / space_name,
                title=f"{base.name} {space_name}",
                seed=args.seed,
            )
    return rows, endpoint_rows


def _velocity_delta(space_ds: CellStreamDataset) -> float:
    diffs = np.diff(np.asarray(space_ds.timepoints, dtype=np.float32))
    return float(max(np.min(diffs), 1e-3))


def _linear_velocity(space_ds: CellStreamDataset) -> np.ndarray:
    means = {t: space_ds.coords_by_t[t][space_ds.splits_by_t[t].train].mean(axis=0) for t in space_ds.timepoints}
    out = np.zeros_like(space_ds.values, dtype=np.float32)
    tps = space_ds.timepoints
    for i, t in enumerate(tps):
        if i < len(tps) - 1:
            nxt = tps[i + 1]
            vel = (means[nxt] - means[t]) / max(float(nxt - t), 1e-6)
        else:
            prv = tps[i - 1]
            vel = (means[t] - means[prv]) / max(float(t - prv), 1e-6)
        out[np.isclose(space_ds.labels, t)] = vel[None, :]
    return out.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["sim", "emt"])
    parser.add_argument("--spaces", nargs="+", choices=["native", "cellstream_latent"], default=["native", "cellstream_latent"])
    parser.add_argument("--output-dir", type=Path, default=Path("output/paper_bench/cellstream_style"))
    parser.add_argument("--experiment", default="g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw")
    parser.add_argument("--model-config-checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--cwm-inits", nargs="+", choices=sorted(INIT_CHECKPOINTS), default=["scratch"])
    parser.add_argument("--include-zero-shot", action="store_true")
    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-cellstream-baseline", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    all_rows = []
    all_endpoint_rows = []
    for dataset in args.datasets:
        rows, endpoint_rows = run_dataset(args, dataset, device)
        all_rows.extend(rows)
        all_endpoint_rows.extend(endpoint_rows)
        pd.DataFrame(all_rows).to_csv(args.output_dir / "metrics.partial.tsv", sep="\t", index=False)
        pd.DataFrame(all_endpoint_rows).to_csv(args.output_dir / "endpoint_metrics.partial.tsv", sep="\t", index=False)
    metrics = pd.DataFrame(all_rows)
    endpoints = pd.DataFrame(all_endpoint_rows)
    metrics.to_csv(args.output_dir / "metrics.tsv", sep="\t", index=False)
    endpoints.to_csv(args.output_dir / "endpoint_metrics.tsv", sep="\t", index=False)
    manifest = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "metrics": all_rows,
        "endpoint_metrics": all_endpoint_rows,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(metrics.to_string(index=False))
    print(endpoints.to_string(index=False))


if __name__ == "__main__":
    main()
