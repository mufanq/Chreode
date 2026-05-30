#!/usr/bin/env python
"""Evaluate Chreode A0/A1/A2 and simple baselines on exported representations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from cellworldmodel.benchmark.common_metrics import sinkhorn_w2
from cellworldmodel.benchmark.zesta_metrics import scalar_mmd
from cellworldmodel.foundation.dynamics_loader import load_foundation_transition


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 2:
        return float("nan")
    a = a[keep]
    b = b[keep]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _metric_trials(
    pred: np.ndarray,
    true: np.ndarray,
    *,
    n: int,
    trials: int,
    seed: int,
    sinkhorn_eps: float,
    sinkhorn_iters: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    w2_vals = []
    mmd_vals = []
    mean_mse_vals = []
    mean_corr_vals = []
    for _ in range(int(trials)):
        p_idx = np.arange(pred.shape[0])
        t_idx = np.arange(true.shape[0])
        if pred.shape[0] > n:
            p_idx = rng.choice(p_idx, size=n, replace=False)
        if true.shape[0] > n:
            t_idx = rng.choice(t_idx, size=n, replace=False)
        p = pred[np.sort(p_idx)].astype(np.float32)
        y = true[np.sort(t_idx)].astype(np.float32)
        with torch.no_grad():
            w2 = sinkhorn_w2(
                torch.from_numpy(p),
                torch.from_numpy(y),
                epsilon=float(sinkhorn_eps),
                num_iters=int(sinkhorn_iters),
            ).item()
        w2_vals.append(float(w2))
        mmd_vals.append(float(scalar_mmd(y, p, max_samples=min(int(n), 1000), seed=int(rng.integers(0, 2**31 - 1)))))
        pm = p.mean(axis=0)
        ym = y.mean(axis=0)
        mean_mse_vals.append(float(np.mean((pm - ym) ** 2)))
        mean_corr_vals.append(_pearson(pm, ym))
    return {
        "sinkhorn_w2_mean": float(np.mean(w2_vals)),
        "sinkhorn_w2_std": float(np.std(w2_vals, ddof=1)) if len(w2_vals) > 1 else 0.0,
        "mmd_mean": float(np.mean(mmd_vals)),
        "mmd_std": float(np.std(mmd_vals, ddof=1)) if len(mmd_vals) > 1 else 0.0,
        "mean_mse": float(np.mean(mean_mse_vals)),
        "mean_pearson": float(np.nanmean(mean_corr_vals)),
    }


def _standardize_from_train(z: np.ndarray, train_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = z[train_ids].mean(axis=0, keepdims=True)
    sd = np.maximum(z[train_ids].std(axis=0, keepdims=True), 1e-6)
    return (z - mu) / sd, mu.astype(np.float32), sd.astype(np.float32)


def _linear_predict(
    source: np.ndarray,
    *,
    train_means: dict[float, np.ndarray],
    source_time: float,
    target_time: float,
) -> np.ndarray:
    times = np.asarray(sorted(t for t in train_means if t != source_time), dtype=np.float32)
    if len(times) == 0:
        return source.copy()
    y = np.stack([train_means[float(t)] - train_means[source_time] for t in times], axis=0)
    x = np.log1p(times - float(source_time))[:, None]
    model = Ridge(alpha=1.0)
    model.fit(x, y)
    delta = model.predict(np.asarray([[np.log1p(float(target_time) - float(source_time))]], dtype=np.float32))[0]
    return (source + delta[None, :]).astype(np.float32)


def _predict_transition(
    model,
    source_raw: np.ndarray,
    *,
    delta: float,
    mu: np.ndarray,
    sd: np.ndarray,
    device: str,
    batch_size: int,
    n_mc: int,
) -> np.ndarray:
    dev = torch.device(device)
    chunks = []
    for start in range(0, source_raw.shape[0], batch_size):
        z = torch.from_numpy(source_raw[start:start + batch_size].astype(np.float32)).to(dev)
        d = torch.full((z.shape[0],), float(delta), device=dev, dtype=z.dtype)
        with torch.no_grad():
            pred = model.predict_mean(z, d, action=None, n_mc=int(n_mc)).detach().cpu().numpy()
        chunks.append(((pred - mu) / sd).astype(np.float32))
    return np.concatenate(chunks, axis=0)


def _time_ids(meta: pd.DataFrame, time: float, split: str) -> np.ndarray:
    mask = np.isclose(meta["time"].astype(float).to_numpy(), float(time)) & (meta["split"].astype(str).to_numpy() == split)
    return np.where(mask)[0].astype(np.int64)


def _train_means(z: np.ndarray, meta: pd.DataFrame, train_times: Iterable[float]) -> dict[float, np.ndarray]:
    out = {}
    for t in train_times:
        ids = _time_ids(meta, float(t), "train")
        if len(ids):
            out[float(t)] = z[ids].mean(axis=0)
    return out


def evaluate(args: argparse.Namespace) -> dict:
    rep = np.load(args.representations)
    meta = pd.read_csv(args.metadata, sep="\t")
    raw = rep[args.space].astype(np.float32)
    train_ids = np.where(meta["split"].astype(str).to_numpy() == "train")[0].astype(np.int64)
    z, mu, sd = _standardize_from_train(raw, train_ids)
    source_ids = _time_ids(meta, args.source_time, "test")
    if len(source_ids) == 0:
        raise ValueError(f"No test source cells for time={args.source_time}")
    rng = np.random.default_rng(args.seed)
    if len(source_ids) > args.max_source_cells:
        source_ids = np.sort(rng.choice(source_ids, size=args.max_source_cells, replace=False))
    source_z = z[source_ids]
    source_raw = raw[source_ids]
    train_means = _train_means(z, meta, [args.source_time, *args.train_times])

    transitions = {}
    if args.space == "scvi128":
        for name, ckpt in {
            "A1_static_dit": args.static_checkpoint,
            "A2_dynamics_dit": args.dynamic_checkpoint,
        }.items():
            model, info = load_foundation_transition(
                checkpoint=ckpt,
                latent_dim=raw.shape[1],
                device=args.device,
                experiment=args.experiment,
                dit_size=args.dit_size,
                batch_size=args.batch_size,
                k_samples=args.n_mc,
                lr=3e-4,
            )
            transitions[name] = (model, info)

    rows = []
    for t in args.eval_times:
        target_ids = _time_ids(meta, float(t), "test")
        if len(target_ids) == 0:
            continue
        true = z[target_ids]
        methods = {
            "identity_source_replay": source_z,
            "linear_time_delta": _linear_predict(
                source_z,
                train_means=train_means,
                source_time=float(args.source_time),
                target_time=float(t),
            ),
        }
        if args.space == "scvi128":
            methods["A0_vae_only_identity"] = source_z
            for name, (model, info) in transitions.items():
                methods[name] = _predict_transition(
                    model,
                    source_raw,
                    delta=float(t) - float(args.source_time),
                    mu=mu,
                    sd=sd,
                    device=args.device,
                    batch_size=args.batch_size,
                    n_mc=args.n_mc,
                )
        for method, pred in methods.items():
            metrics = _metric_trials(
                pred,
                true,
                n=args.metric_cells,
                trials=args.metric_trials,
                seed=args.seed + int(float(t) * 1000) + len(method),
                sinkhorn_eps=args.sinkhorn_eps,
                sinkhorn_iters=args.sinkhorn_iters,
            )
            rows.append({
                "benchmark": args.benchmark,
                "space": args.space,
                "method": method,
                "source_time": float(args.source_time),
                "target_time": float(t),
                "n_source": int(pred.shape[0]),
                "n_target": int(true.shape[0]),
                **metrics,
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "metrics.tsv", sep="\t", index=False)
    summary = {
        "benchmark": args.benchmark,
        "space": args.space,
        "representations": str(args.representations),
        "metadata": str(args.metadata),
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=["weinreb", "veres"])
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--space", choices=["scvi128", "pca"], required=True)
    parser.add_argument("--source-time", type=float, required=True)
    parser.add_argument("--train-times", type=float, nargs="+", required=True)
    parser.add_argument("--eval-times", type=float, nargs="+", required=True)
    parser.add_argument("--static-checkpoint", type=Path, default=Path("output/foundation/genhui_v1/dynamics/vae2_staticdit2/model.pt"))
    parser.add_argument("--dynamic-checkpoint", type=Path, default=Path("output/foundation/genhui_v1/dynamics/vae2_dynamicsdit2/model.pt"))
    parser.add_argument("--experiment", default="g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw")
    parser.add_argument("--dit-size", default="small")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-mc", type=int, default=8)
    parser.add_argument("--max-source-cells", type=int, default=2048)
    parser.add_argument("--metric-cells", type=int, default=2000)
    parser.add_argument("--metric-trials", type=int, default=5)
    parser.add_argument("--sinkhorn-eps", type=float, default=0.05)
    parser.add_argument("--sinkhorn-iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
