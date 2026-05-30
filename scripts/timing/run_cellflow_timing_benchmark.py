#!/usr/bin/env python
"""Timing benchmark for a saved CellFlow model on the ZESTA control task."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import jax
import cellflow
from cellflow.data._dataloader import PredictionSampler

from scripts.cellflow.run_zesta_control_time_cellflow import load_zesta_splits


def _block_until_ready(value):
    if isinstance(value, dict):
        for item in value.values():
            jax.block_until_ready(item)
    else:
        jax.block_until_ready(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--target-time", type=float, default=72.0)
    parser.add_argument("--data-path", type=Path, default=Path("data/external/cellflow/zesta.h5ad"))
    args = parser.parse_args()

    cf = cellflow.model.CellFlow.load(str(args.model_path))
    _, adata_source, _, covariate_data, meta = load_zesta_splits(
        args.data_path,
        rep_key="X_aligned",
        source_t=18.0,
        train_targets=[24.0, 36.0, 48.0],
        eval_targets=[float(args.target_time)],
        split_ratio=0.8,
        split_policy="per_timepoint",
        seed=42,
        max_cells_per_timepoint=10000,
        max_source_predict=4000,
        max_eval_cells=4000,
    )
    cov_one = covariate_data[
        covariate_data["condition"] == f"control_control_{int(args.target_time)}"
    ].reset_index(drop=True)
    if cov_one.empty:
        raise ValueError(f"No covariate row for target_time={args.target_time}")
    pred_data = cf._dm.get_prediction_data(
        adata_source[:1].copy(),
        sample_rep="X_aligned",
        covariate_data=cov_one,
        condition_id_key="condition",
    )
    batch = PredictionSampler(pred_data).sample()
    key = next(iter(batch["source"]))
    source = batch["source"][key]
    condition = batch["condition"][key]

    def predict_one():
        return cf.solver.predict(source, condition)

    for _ in range(int(args.warmup)):
        _block_until_ready(predict_one())
    times = []
    for _ in range(int(args.n)):
        t0 = time.perf_counter()
        _block_until_ready(predict_one())
        times.append((time.perf_counter() - t0) * 1000.0)
    times_sorted = sorted(times)
    payload = {
        "date": "2026-05-07",
        "hardware": str(jax.devices()[0]),
        "model_path": str(args.model_path),
        "task": "CellFlow saved compact ZESTA control_control_18_to_72, single source cell, fp32/JAX",
        "nfe_label": "50--100 ODE",
        "ms_per_query": float(statistics.median(times)),
        "gflops_per_query": None,
        "timing": {
            "median_ms": float(statistics.median(times)),
            "mean_ms": float(statistics.mean(times)),
            "p10_ms": float(times_sorted[int(0.1 * len(times_sorted))]),
            "p90_ms": float(times_sorted[max(0, int(0.9 * len(times_sorted)) - 1)]),
            "n": int(args.n),
        },
        "source_shape": tuple(int(x) for x in source.shape),
        "condition_shapes": {k: tuple(int(x) for x in v.shape) for k, v in condition.items()},
        "split_meta": meta,
        "notes": [
            "Timing uses direct cf.solver.predict(source, condition) after JIT warmup, not full cf.predict data-manager overhead.",
            "GFLOPs are not reported because this is a JAX/diffrax solve and torch.profiler FLOP accounting does not apply.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
