"""Validate CellWorldModel Snakemake experiment config files."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from cellworldmodel.benchmark.experiment_registry import experiment_names


VALID_SPLIT_POLICIES = {"legacy", "per_timepoint"}
VALID_WANDB_MODES = {"online", "offline", "disabled"}


def configured_experiments(cfg: dict) -> list[str]:
    if "experiments" in cfg:
        exps = cfg["experiments"]
        if isinstance(exps, dict):
            return list(exps.keys())
        if isinstance(exps, list):
            return [str(x) for x in exps]
        raise ValueError("experiments must be a list or mapping")
    if "experiment" not in cfg:
        raise ValueError("config must contain either experiment or experiments")
    return [str(cfg["experiment"])]


def require(cfg: dict, key: str) -> None:
    if key not in cfg:
        raise ValueError(f"Missing required key: {key}")


def validate_config(path: Path) -> dict:
    cfg = yaml.safe_load(open(path))
    if not isinstance(cfg, dict):
        raise ValueError("Top-level config must be a mapping")
    require(cfg, "output_root")
    exps = configured_experiments(cfg)
    valid_exps = set(experiment_names())
    unknown = [x for x in exps if x not in valid_exps]
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}; expected one of {sorted(valid_exps)}")
    if "overrides" in cfg:
        overrides = cfg["overrides"] or {}
        split_policy = overrides.get("split_policy")
        if split_policy is not None and split_policy not in VALID_SPLIT_POLICIES:
            raise ValueError(f"Invalid overrides.split_policy={split_policy!r}")
    wandb = cfg.get("wandb", {}) or {}
    mode = wandb.get("mode")
    if mode is not None and mode not in VALID_WANDB_MODES:
        raise ValueError(f"Invalid wandb.mode={mode!r}")
    resources = cfg.get("resources", {}) or {}
    for key in ("cpus", "mem_mb", "runtime_min"):
        if key in resources and int(resources[key]) <= 0:
            raise ValueError(f"resources.{key} must be positive")
    return cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("configfiles", nargs="+", type=Path)
    args = p.parse_args()
    for path in args.configfiles:
        cfg = validate_config(path)
        print(f"OK {path}: experiments={configured_experiments(cfg)} output_root={cfg['output_root']}")


if __name__ == "__main__":
    main()
