#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from pipeline.simple_pair_predictor import (
    SimplePairPredictor,
    build_dataloader,
    build_output_paths,
    deserialize_target_time_means,
    evaluate_model,
    load_checkpoint,
    load_experiment_config,
    prepare_experiment,
    save_json,
    setup_logging,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the simple pair predictor")
    parser.add_argument("--config", required=True, help="Path to simple_pair_predictor config YAML")
    parser.add_argument("--experiment", required=True, help="Experiment name in config")
    parser.add_argument("--output", required=True, help="Path to output JSON report")
    parser.add_argument("--log-file", default=None, help="Optional log file")
    args = parser.parse_args()

    logger = setup_logging(args.log_file)
    config = load_experiment_config(args.config, args.experiment)
    paths = build_output_paths(str(config["output_dir"]), config["experiment_name"])

    (
        config,
        dataset,
        collate_fn,
        _splits,
        _pca_state,
        _time_stats,
        _target_time_means,
        device,
    ) = prepare_experiment(args.config, args.experiment, logger)

    splits = np.load(paths["splits"])
    checkpoint = load_checkpoint(paths["checkpoint"], device=device)

    model = SimplePairPredictor(
        latent_dim=int(checkpoint["pca_state"]["components"].shape[0]),
        hidden_dim=int(config["model"]["hidden_dim"]),
        num_layers=int(config["model"]["num_layers"]),
        dropout=float(config["model"]["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    training_cfg = config["training"]
    preprocess_cfg = config["preprocess"]
    eval_loaders = {
        split_name: build_dataloader(
            dataset=dataset,
            collate_fn=collate_fn,
            subset_indices=splits[split_name],
            batch_size=int(training_cfg["eval_batch_size"]),
            num_workers=int(training_cfg["num_workers"]),
            train=False,
            device=device,
        )
        for split_name in ("train", "val", "test")
    }
    target_time_means = deserialize_target_time_means(checkpoint["target_time_means"])
    metrics = {
        split_name: evaluate_model(
            model=model,
            loader=loader,
            pca_state=checkpoint["pca_state"],
            time_stats=checkpoint["time_stats"],
            normalize_target_sum=float(preprocess_cfg["normalize_target_sum"]),
            target_time_means=target_time_means,
            device=device,
            max_batches=training_cfg.get("eval_max_batches"),
        )
        for split_name, loader in eval_loaders.items()
    }

    report = {
        "experiment_name": config["experiment_name"],
        "checkpoint": str(paths["checkpoint"]),
        "device": str(device),
        "metrics": metrics,
    }
    save_json(report, Path(args.output))
    logger.info("Evaluation report written to %s", args.output)


if __name__ == "__main__":
    main()
