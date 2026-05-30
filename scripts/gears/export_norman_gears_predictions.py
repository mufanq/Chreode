#!/usr/bin/env python
"""Export predictions from a saved official GEARS Norman model.

Run this inside the GEARS uv environment, e.g.

  .venv/gears/bin/python scripts/gears/export_norman_gears_predictions.py ...
"""

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
        return None if np.isnan(value) or np.isinf(value) else value
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gears-repo", type=Path, default=Path("3rdparty/GEARS"))
    parser.add_argument("--data-dir", type=Path, default=Path("output/gears/data"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-name", default="norman")
    parser.add_argument("--split", default="simulation")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    sys.path.insert(0, str(args.gears_repo.resolve()))
    from gears import GEARS, PertData
    from gears.inference import compute_metrics, evaluate

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pert_data = PertData(str(args.data_dir))
    pert_data.load(data_name=args.data_name)
    pert_data.prepare_split(split=args.split, seed=args.seed)
    pert_data.get_dataloader(batch_size=args.batch_size, test_batch_size=args.test_batch_size)

    model = GEARS(pert_data, device=args.device, weight_bias_track=False)
    model.load_pretrained(str(args.model_dir))
    test_res = evaluate(
        pert_data.dataloader["test_loader"],
        model.best_model,
        model.config["uncertainty"],
        args.device,
    )
    test_metrics, _ = compute_metrics(test_res)
    gene_names = np.asarray(pert_data.adata.var["gene_name"].to_numpy(), dtype=str)
    out_npz = args.output_dir / "predictions.npz"
    np.savez_compressed(
        out_npz,
        pred=np.asarray(test_res["pred"], dtype=np.float32),
        truth=np.asarray(test_res["truth"], dtype=np.float32),
        conditions=np.asarray(test_res["pert_cat"], dtype=str),
        gene_names=gene_names,
    )
    with (args.output_dir / "test_res.pkl").open("wb") as handle:
        pickle.dump(test_res, handle)
    summary = {
        "model_dir": str(args.model_dir),
        "prediction": str(out_npz),
        "test_res": str(args.output_dir / "test_res.pkl"),
        "data_name": args.data_name,
        "split": args.split,
        "seed": int(args.seed),
        "test_metrics": test_metrics,
    }
    with (args.output_dir / "export_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(summary), handle, indent=2, sort_keys=True)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
