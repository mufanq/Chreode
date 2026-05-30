#!/usr/bin/env python
"""Run and persist the official GEARS Norman simulation baseline."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


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
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def _mean_metric(metric_by_pert: dict[str, dict[str, float]], perts: list[str], metric: str) -> float | None:
    values = [metric_by_pert[p][metric] for p in perts if p in metric_by_pert and metric in metric_by_pert[p]]
    if not values:
        return None
    return float(np.mean(values))


def _subgroup_summary(metric_by_pert: dict[str, dict[str, float]], subgroup: dict, metrics: list[str]) -> dict:
    out = {}
    for name, perts in subgroup.get("test_subgroup", {}).items():
        out[name] = {metric: _mean_metric(metric_by_pert, list(perts), metric) for metric in metrics}
        out[name]["n_conditions"] = int(len(perts))
    return out


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
    parser.add_argument("--save-test-res", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(args.gears_repo.resolve()))
    from gears import GEARS, PertData
    from gears.inference import compute_metrics, deeper_analysis, evaluate, non_dropout_analysis

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
