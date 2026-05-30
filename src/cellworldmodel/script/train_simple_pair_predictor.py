#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch
try:
    import wandb
except ImportError:  # pragma: no cover - handled at runtime after env setup
    wandb = None

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from pipeline.simple_pair_predictor import (
    SimplePairPredictor,
    build_dataloader,
    build_output_paths,
    evaluate_model,
    get_hostname,
    load_experiment_config,
    prepare_experiment,
    resolve_train_round_settings,
    save_json,
    save_yaml,
    serialize_target_time_means,
    setup_logging,
    train_one_round,
)


def maybe_init_wandb(config, output_dir: Path, logger):
    if wandb is None:
        logger.warning("wandb is not installed; skipping online experiment tracking")
        return None

    wandb_cfg = config["wandb"]
    if not wandb_cfg.get("enabled", True):
        return None

    mode = wandb_cfg.get("mode", "online")
    netrc_path = Path.home() / ".netrc"
    if mode == "online" and "WANDB_API_KEY" not in os.environ and not netrc_path.exists():
        logger.warning("WANDB_API_KEY not set and no obvious prior login found; using offline mode")
        mode = "offline"

    try:
        run = wandb.init(
            project=wandb_cfg.get("project", "CellWorldModel"),
            entity=wandb_cfg.get("entity"),
            name=config["experiment_name"],
            config=config,
            tags=wandb_cfg.get("tags"),
            notes=config.get("notes"),
            dir=str(output_dir),
            mode=mode,
        )
    except Exception as exc:
        logger.warning("wandb.init failed (%s); retrying in offline mode", exc)
        run = wandb.init(
            project=wandb_cfg.get("project", "CellWorldModel"),
            entity=wandb_cfg.get("entity"),
            name=config["experiment_name"],
            config=config,
            tags=wandb_cfg.get("tags"),
            notes=config.get("notes"),
            dir=str(output_dir),
            mode="offline",
        )
    return run


def build_wandb_run_info(wandb_run):
    if wandb_run is None:
        return {
            "enabled": False,
            "run_id": None,
            "run_name": None,
            "run_path": None,
            "run_url": None,
            "local_run_dir": None,
            "local_files_dir": None,
            "local_output_log": None,
        }

    local_files_dir = Path(wandb_run.dir).resolve()
    local_run_dir = local_files_dir.parent
    run_path = None
    raw_path = getattr(wandb_run, "path", None)
    if raw_path:
        if isinstance(raw_path, (list, tuple)):
            run_path = "/".join(str(part) for part in raw_path)
        else:
            run_path = str(raw_path)

    return {
        "enabled": True,
        "run_id": getattr(wandb_run, "id", None),
        "run_name": getattr(wandb_run, "name", None),
        "run_path": run_path,
        "run_url": getattr(wandb_run, "url", None),
        "local_run_dir": str(local_run_dir),
        "local_files_dir": str(local_files_dir),
        "local_output_log": str(local_files_dir / "output.log"),
    }


def main():
    parser = argparse.ArgumentParser(description="Train the simple pair predictor")
    parser.add_argument("--config", required=True, help="Path to simple_pair_predictor config YAML")
    parser.add_argument("--experiment", required=True, help="Experiment name in config")
    parser.add_argument("--log-file", default=None, help="Optional log file")
    args = parser.parse_args()

    logger = setup_logging(args.log_file)
    config = load_experiment_config(args.config, args.experiment)
    paths = build_output_paths(str(config["output_dir"]), config["experiment_name"])
    save_yaml(config, paths["resolved_config"])

    (
        config,
        dataset,
        collate_fn,
        splits,
        pca_state,
        time_stats,
        target_time_means,
        device,
    ) = prepare_experiment(args.config, args.experiment, logger)

    logger.info("Using host %s", get_hostname())
    if device.type == "cuda":
        logger.info("Using GPU %s", torch.cuda.get_device_name(device))
    else:
        logger.info("Using CPU")

    training_cfg = config["training"]
    preprocess_cfg = config["preprocess"]
    model_cfg = config["model"]

    train_loader = build_dataloader(
        dataset=dataset,
        collate_fn=collate_fn,
        subset_indices=splits["train"],
        batch_size=int(training_cfg["batch_size"]),
        num_workers=int(training_cfg["num_workers"]),
        train=True,
        device=device,
        train_samples_per_round=training_cfg.get("train_samples_per_round"),
    )
    val_loader = build_dataloader(
        dataset=dataset,
        collate_fn=collate_fn,
        subset_indices=splits["val"],
        batch_size=int(training_cfg["eval_batch_size"]),
        num_workers=int(training_cfg["num_workers"]),
        train=False,
        device=device,
    )
    test_loader = build_dataloader(
        dataset=dataset,
        collate_fn=collate_fn,
        subset_indices=splits["test"],
        batch_size=int(training_cfg["eval_batch_size"]),
        num_workers=int(training_cfg["num_workers"]),
        train=False,
        device=device,
    )

    model = SimplePairPredictor(
        latent_dim=int(pca_state["components"].shape[0]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )

    wandb_run = maybe_init_wandb(config, paths["base_dir"], logger)
    wandb_run_info = build_wandb_run_info(wandb_run)

    train_round_info = resolve_train_round_settings(
        train_subset_size=len(splits["train"]),
        train_samples_per_round=training_cfg.get("train_samples_per_round"),
        batch_size=int(training_cfg["batch_size"]),
    )
    logger.info(
        "Sample counts: total_pairs=%s, train_pairs=%s, val_pairs=%s, test_pairs=%s, "
        "train_pairs_per_%s=%s, optimizer_steps_per_%s=%s",
        len(dataset),
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
        train_round_info["unit_name"],
        train_round_info["samples_per_round"],
        train_round_info["unit_name"],
        train_round_info["steps_per_round"],
    )

    best_metric = None
    best_round = -1
    best_payload = None
    history = []
    patience = 0
    global_step = 0
    train_log_path = str(Path(args.log_file).resolve()) if args.log_file else None
    eval_log_path = str((paths["logs_dir"] / "eval.log").resolve())

    if wandb_run is not None:
        wandb_run.summary["total_pairs"] = int(len(dataset))
        wandb_run.summary["train_pairs_per_round"] = int(train_round_info["samples_per_round"])
        wandb_run.summary["optimizer_steps_per_round"] = int(train_round_info["steps_per_round"])
        if train_log_path is not None:
            wandb_run.summary["train_log_path"] = train_log_path
        wandb_run.summary["eval_log_path"] = eval_log_path

    def log_train_step(step: int, metrics: dict):
        if wandb_run is None:
            return
        wandb.log(metrics, step=step)

    for round_idx in range(1, int(training_cfg["rounds"]) + 1):
        train_metrics = train_one_round(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            pca_state=pca_state,
            time_stats=time_stats,
            normalize_target_sum=float(preprocess_cfg["normalize_target_sum"]),
            device=device,
            gene_loss_weight=float(training_cfg["gene_loss_weight"]),
            gradient_clip_norm=float(training_cfg["gradient_clip_norm"]),
            round_idx=round_idx,
            global_step_start=global_step,
            step_logger=log_train_step,
        )
        global_step = int(train_metrics["global_step_end"])
        val_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            pca_state=pca_state,
            time_stats=time_stats,
            normalize_target_sum=float(preprocess_cfg["normalize_target_sum"]),
            target_time_means=target_time_means,
            device=device,
            max_batches=training_cfg.get("val_max_batches"),
        )

        round_metrics = {
            "round": round_idx,
            **train_metrics,
            "val_model_gene_mse": val_metrics["model"]["gene_mse"],
            "val_model_gene_mae": val_metrics["model"]["gene_mae"],
            "val_model_pearson_mean": val_metrics["model"]["pearson_mean"],
            "val_identity_gene_mse": val_metrics["identity_baseline"]["gene_mse"],
            "val_target_time_mean_gene_mse": val_metrics["target_time_mean_baseline"]["gene_mse"],
            "val_pairs_evaluated": val_metrics["model"]["n_pairs"],
            "global_step_end": global_step,
        }
        history.append(round_metrics)
        logger.info("Round %s metrics: %s", round_idx, round_metrics)
        if wandb_run is not None:
            wandb.log(round_metrics, step=global_step)

        score = val_metrics["model"]["gene_mse"]
        if best_metric is None or score < best_metric:
            best_metric = score
            best_round = round_idx
            patience = 0
            best_payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "splits": {key: value.tolist() for key, value in splits.items()},
                "pca_state": {key: value.cpu() for key, value in pca_state.items()},
                "time_stats": dict(time_stats),
                "target_time_means": serialize_target_time_means(target_time_means),
                "best_round": best_round,
                "best_val_metrics": val_metrics,
                "history": history,
                "training_schedule": train_round_info,
                "hostname": get_hostname(),
                "wandb": wandb_run_info,
                "log_paths": {
                    "train_log": train_log_path,
                    "eval_log": eval_log_path,
                },
            }
            torch.save(best_payload, paths["checkpoint"])
        else:
            patience += 1

        if patience >= int(training_cfg["early_stop_patience"]):
            logger.info("Early stopping triggered at round %s", round_idx)
            break

    if best_payload is None:
        raise RuntimeError("Training did not produce a checkpoint")

    checkpoint = torch.load(paths["checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_eval_loader = build_dataloader(
        dataset=dataset,
        collate_fn=collate_fn,
        subset_indices=splits["train"],
        batch_size=int(training_cfg["eval_batch_size"]),
        num_workers=int(training_cfg["num_workers"]),
        train=False,
        device=device,
    )
    final_metrics = {
        "train": evaluate_model(
            model=model,
            loader=train_eval_loader,
            pca_state=checkpoint["pca_state"],
            time_stats=checkpoint["time_stats"],
            normalize_target_sum=float(preprocess_cfg["normalize_target_sum"]),
            target_time_means={k: torch.tensor(v, dtype=torch.float32) for k, v in checkpoint["target_time_means"].items()},
            device=device,
            max_batches=training_cfg.get("eval_max_batches"),
        ),
        "val": evaluate_model(
            model=model,
            loader=val_loader,
            pca_state=checkpoint["pca_state"],
            time_stats=checkpoint["time_stats"],
            normalize_target_sum=float(preprocess_cfg["normalize_target_sum"]),
            target_time_means={k: torch.tensor(v, dtype=torch.float32) for k, v in checkpoint["target_time_means"].items()},
            device=device,
            max_batches=training_cfg.get("eval_max_batches"),
        ),
        "test": evaluate_model(
            model=model,
            loader=test_loader,
            pca_state=checkpoint["pca_state"],
            time_stats=checkpoint["time_stats"],
            normalize_target_sum=float(preprocess_cfg["normalize_target_sum"]),
            target_time_means={k: torch.tensor(v, dtype=torch.float32) for k, v in checkpoint["target_time_means"].items()},
            device=device,
            max_batches=training_cfg.get("eval_max_batches"),
        ),
    }

    split_summary = {
        key: int(len(value))
        for key, value in splits.items()
    }
    train_summary = {
        "experiment_name": config["experiment_name"],
        "hostname": get_hostname(),
        "device": str(device),
        "selected_datasets": config["selected_datasets"],
        "total_pairs": int(len(dataset)),
        "split_sizes": split_summary,
        "train_pairs_per_round": int(train_round_info["samples_per_round"]),
        "optimizer_steps_per_round": int(train_round_info["steps_per_round"]),
        "full_train_round": bool(train_round_info["full_train_round"]),
        "completed_rounds": int(len(history)),
        "best_round": best_round,
        "global_steps_completed": int(global_step),
        "wandb": wandb_run_info,
        "log_paths": {
            "train_log": train_log_path,
            "eval_log": eval_log_path,
        },
        "history": history,
        "final_metrics": final_metrics,
        "dataset_info": dataset.get_dataset_info(),
    }
    save_json(train_summary, paths["train_summary"])
    if wandb_run is not None:
        wandb.summary["total_pairs"] = int(len(dataset))
        wandb.summary["train_pairs_per_round"] = int(train_round_info["samples_per_round"])
        wandb.summary["optimizer_steps_per_round"] = int(train_round_info["steps_per_round"])
        wandb.summary["completed_rounds"] = int(len(history))
        wandb.summary["best_round"] = best_round
        wandb.summary["global_steps_completed"] = int(global_step)
        wandb.summary["split_sizes"] = split_summary
        wandb.summary["test_model_gene_mse"] = final_metrics["test"]["model"]["gene_mse"]
        wandb.summary["test_identity_gene_mse"] = final_metrics["test"]["identity_baseline"]["gene_mse"]
        wandb.summary["test_target_time_mean_gene_mse"] = final_metrics["test"]["target_time_mean_baseline"]["gene_mse"]
        wandb.finish()

    import numpy as np

    np.savez_compressed(paths["splits"], **splits)
    paths["train_done"].write_text("done\n", encoding="utf-8")
    logger.info("Training artifacts written to %s", paths["base_dir"])


if __name__ == "__main__":
    main()
