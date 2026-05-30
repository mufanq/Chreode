#!/usr/bin/env python
"""Evaluate full foundation A0/A1/A2 on Weinreb clonal fate Pearson."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cellworldmodel.evaluation.fate_clonal import (
    cellwise_dataframe,
    load_clonal_fate_inputs,
    score_fate_predictions,
    summarize_fate_scores,
)
from cellworldmodel.foundation.dynamics_loader import load_foundation_transition


ROOT = Path(__import__("os").environ.get("CHREODE_ROOT", "."))


def _linear_delta_prediction(
    z: np.ndarray,
    meta: pd.DataFrame,
    source: np.ndarray,
    *,
    source_time: float,
    target_time: float,
) -> np.ndarray:
    means = {}
    for t in sorted(meta["time"].unique()):
        train = (np.isclose(meta["time"].astype(float).to_numpy(), float(t))) & (meta["split"].astype(str).to_numpy() == "train")
        if train.any():
            means[float(t)] = z[train].mean(axis=0)
    if source_time not in means or target_time not in means:
        return source.copy()
    delta = means[target_time] - means[source_time]
    return source + delta[None, :]


def _predict_model_samples(
    model,
    source: np.ndarray,
    *,
    delta: float,
    n_sim: int,
    batch_size: int,
    chunk_k: int,
    device: str,
    seed: int,
    use_predict_mean: bool = False,
    predict_mean_n_mc: int = 32,
) -> np.ndarray:
    dev = torch.device(device)
    rng = torch.Generator(device=dev)
    rng.manual_seed(seed)
    n_out = 1 if use_predict_mean else int(n_sim)
    out = np.empty((source.shape[0], n_out, source.shape[1]), dtype=np.float32)
    for start in range(0, source.shape[0], batch_size):
        end = min(start + batch_size, source.shape[0])
        z = torch.from_numpy(source[start:end].astype(np.float32)).to(dev)
        d = torch.full((z.shape[0],), float(delta), device=dev, dtype=z.dtype)
        if use_predict_mean:
            with torch.no_grad():
                pred = model.predict_mean(z, d, n_mc=int(predict_mean_n_mc)).detach().cpu().numpy()
            out[start:end, :1] = pred[:, None, :].astype(np.float32)
            continue
        chunks = []
        remaining = n_sim
        while remaining > 0:
            k = min(chunk_k, remaining)
            eps = torch.randn(z.shape[0], k, z.shape[1], device=dev, dtype=z.dtype, generator=rng)
            with torch.no_grad():
                pred = model(z, d, eps).detach().cpu().numpy()
            chunks.append(pred.astype(np.float32))
            remaining -= k
        out[start:end] = np.concatenate(chunks, axis=1)
    return out


def run(args: argparse.Namespace) -> dict:
    data = load_clonal_fate_inputs(
        representations=args.representations,
        metadata=args.metadata,
        clonal=args.clonal,
        space=args.space,
        atlas_split=args.atlas_split,
        max_eval_cells=args.max_eval_cells,
        seed=args.seed,
    )
    z = data.z
    meta = data.meta
    source = data.source

    method_preds: dict[str, np.ndarray] = {
        "A0_identity": np.repeat(source[:, None, :], args.n_sim, axis=1),
        "linear_time_delta": np.repeat(
            _linear_delta_prediction(z, meta, source, source_time=2.0, target_time=6.0)[:, None, :],
            args.n_sim,
            axis=1,
        ),
    }
    infos: dict[str, dict] = {
        "A0_identity": {"base": "source replay"},
        "linear_time_delta": {"base": "train mean d2_to_d6 delta"},
    }
    if args.space == "scvi128":
        for name, ckpt in {
            "A1_static_dit": args.static_checkpoint,
            "A2_dynamics_dit": args.dynamic_checkpoint,
        }.items():
            model, info = load_foundation_transition(
                checkpoint=ckpt,
                latent_dim=z.shape[1],
                device=args.device,
                experiment=args.experiment,
                dit_size=args.dit_size,
                batch_size=args.batch_size,
                k_samples=args.chunk_k,
                lr=3e-4,
            )
            method_preds[name] = _predict_model_samples(
                model,
                source,
                delta=4.0,
                n_sim=args.n_sim,
                batch_size=args.batch_size,
                chunk_k=args.chunk_k,
                device=args.device,
                seed=args.seed + len(name),
                use_predict_mean=bool(args.use_predict_mean),
                predict_mean_n_mc=int(args.predict_mean_n_mc),
            )
            infos[name] = info

    rows = []
    cellwise_rows = []
    for method, pred in method_preds.items():
        scored = score_fate_predictions(pred, data, source=source, tie_policy=args.tie_policy)
        metrics = summarize_fate_scores(data, scored)
        rows.append(
            {
                "benchmark": "fate",
                "space": args.space,
                "method": method,
                "n_sim": int(pred.shape[1]),
                "atlas_split": args.atlas_split,
                **metrics,
                "info": infos.get(method, {}),
            }
        )
        if args.dump_cellwise:
            cellwise_rows.append(cellwise_dataframe(data, scored, method=method))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: v for k, v in row.items() if k != "info"} for row in rows]).to_csv(
        args.output_dir / "metrics.tsv", sep="\t", index=False
    )
    if args.dump_cellwise and cellwise_rows:
        args.dump_cellwise.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(cellwise_rows, ignore_index=True).to_csv(args.dump_cellwise, sep="\t", index=False)
    payload = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representations", type=Path, default=Path("output/paper_bench/representations/weinreb/representations.npz"))
    parser.add_argument("--metadata", type=Path, default=Path("output/paper_bench/representations/weinreb/metadata.tsv"))
    parser.add_argument("--clonal", type=Path, default=Path("output/phase0/weinreb_clonal.npz"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--space", choices=["scvi128", "pca"], default="scvi128")
    parser.add_argument("--atlas-split", choices=["all", "train", "val", "test"], default="all")
    parser.add_argument("--tie-policy", choices=["majority", "other_on_tie"], default="majority")
    parser.add_argument("--dump-cellwise", type=Path, default=None)
    parser.add_argument("--static-checkpoint", type=Path, default=Path("output/foundation/genhui_v1/dynamics/vae2_staticdit2/model.pt"))
    parser.add_argument("--dynamic-checkpoint", type=Path, default=Path("output/foundation/genhui_v1/dynamics/vae2_dynamicsdit2/model.pt"))
    parser.add_argument("--experiment", default="g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw")
    parser.add_argument("--dit-size", default="small")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-k", type=int, default=128)
    parser.add_argument("--n-sim", type=int, default=512)
    parser.add_argument("--use-predict-mean", action="store_true")
    parser.add_argument("--predict-mean-n-mc", type=int, default=32)
    parser.add_argument("--max-eval-cells", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
