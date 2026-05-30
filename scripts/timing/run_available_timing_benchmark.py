#!/usr/bin/env python
"""Timing microbenchmark for paper methods with loadable checkpoints.

This intentionally reports only methods whose checkpoints can be loaded from
the current workspace. Missing methods are recorded in the JSON instead of
being approximated by unrelated surrogate models.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import torch


ROOT = Path(__file__).resolve().parents[2]


def _time_callable(fn: Callable[[], torch.Tensor], *, warmup: int, n: int, device: torch.device) -> dict:
    for _ in range(int(warmup)):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(int(n)):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times_sorted = sorted(times)
    return {
        "median_ms": float(statistics.median(times)),
        "mean_ms": float(statistics.mean(times)),
        "p10_ms": float(times_sorted[int(0.1 * len(times_sorted))]),
        "p90_ms": float(times_sorted[max(0, int(0.9 * len(times_sorted)) - 1)]),
        "n": int(n),
    }


def _profile_flops(fn: Callable[[], torch.Tensor], *, device: torch.device) -> float | None:
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, with_flops=True) as prof:
            fn()
            if device.type == "cuda":
                torch.cuda.synchronize()
        return float(sum((event.flops or 0) for event in prof.key_averages()) / 1e9)
    except Exception as exc:  # pragma: no cover - profiler support is env-dependent.
        print(f"[warn] FLOP profiling failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


def _load_chreode(device: torch.device, checkpoint: Path):
    sys.path.insert(0, str(ROOT / "src"))
    from cellworldmodel.benchmark.registry import build_model

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_cfg = ckpt["config"]
    model = build_model(
        model_cfg["method"],
        int(model_cfg["latent_dim"]),
        model_cfg["train_cfg"],
        tau_init=1.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    latent_dim = int(model_cfg["latent_dim"])
    z = torch.randn(1, latent_dim, device=device)
    delta = torch.tensor([4.0], device=device)
    epsilon = torch.zeros(1, 1, latent_dim, device=device)

    def fn() -> torch.Tensor:
        return model(z, delta, epsilon)

    return fn


def _load_prescient(device: torch.device, run_dir: Path):
    sys.path.insert(0, str(ROOT / "3rdparty" / "prescient-analysis" / "src"))
    import train as prescient_train

    cfg = SimpleNamespace(**torch.load(run_dir / "config.pt", map_location="cpu", weights_only=False))
    model = prescient_train.AutoGenerator(cfg).to(device)
    state = torch.load(run_dir / "train.best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    x0 = torch.randn(1, int(cfg.x_dim), device=device)
    noise = torch.zeros_like(x0)
    dt = 0.1

    def one_step() -> torch.Tensor:
        return model._step(x0, dt, noise)

    def nfe_88() -> torch.Tensor:
        x = x0
        for _ in range(88):
            x = model._step(x, dt, noise)
        return x

    return one_step, nfe_88


def _load_branchsbm(device: torch.device, checkpoint: Path):
    sys.path.insert(0, str(ROOT / "3rdparty" / "BranchSBM"))
    from src.networks.flow_mlp import VelocityNet

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    branch_indices = sorted({int(key.split(".")[1]) for key in state if key.startswith("flow_nets.")})
    if not branch_indices:
        raise ValueError(f"No flow_nets.* tensors found in {checkpoint}")
    idx = branch_indices[0]
    sub_state = {
        ".".join(key.split(".")[2:]): value
        for key, value in state.items()
        if key.startswith(f"flow_nets.{idx}.")
    }
    # VelocityNet is an MLP over concat([t, x]). Infer dimensions from the
    # checkpoint so this works for both Veres and Weinreb scVI-128 runs.
    out_dim = int(sub_state["model.6.bias"].shape[0])
    hidden_dims = [
        int(sub_state["model.0.bias"].shape[0]),
        int(sub_state["model.2.bias"].shape[0]),
        int(sub_state["model.4.bias"].shape[0]),
    ]
    model = VelocityNet(dim=out_dim, hidden_dims=hidden_dims, activation="selu", batch_norm=False).to(device)
    model.load_state_dict(sub_state)
    model.eval()
    x0 = torch.randn(1, out_dim, device=device)

    def one_flow() -> torch.Tensor:
        return model(torch.tensor([0.5], device=device), x0)

    def nfe_40() -> torch.Tensor:
        x = x0
        for step in range(40):
            t = torch.tensor([step / 39.0], device=device)
            x = x + model(t, x) / 40.0
        return x

    return one_flow, nfe_40


def _measure(name: str, fn: Callable[[], torch.Tensor], *, nfe: int, warmup: int, n: int, device: torch.device,
             checkpoint: str) -> dict:
    timing = _time_callable(fn, warmup=warmup, n=n, device=device)
    return {
        "nfe": int(nfe),
        "ms_per_query": timing["median_ms"],
        "gflops_per_query": _profile_flops(fn, device=device),
        "timing": timing,
        "checkpoint": checkpoint,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "output/timing/timing_benchmark_20260507_available.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--chreode-checkpoint",
        type=Path,
        default=ROOT / "output/foundation/genhui_v1/dynamics/vae2_dynamicsdit2/model.pt",
    )
    parser.add_argument(
        "--prescient-run-dir",
        type=Path,
        default=ROOT / "output/paper_bench/prescient/weinreb_scvi128_interpolate/"
        "d26-softplus_2_400-0.1_0.1_1e-06-0.1_0.1_0.005/seed_1",
    )
    parser.add_argument(
        "--branchsbm-checkpoint",
        type=Path,
        default=ROOT / "3rdparty/BranchSBM/checkpoints/scrna/"
        "05_03_1641_paper_veres_scvi128_branched_ep100/flow_model/epoch=99-step=1300.ckpt",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    hardware = torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor()
    results = {}
    missing = {}

    if args.chreode_checkpoint.exists():
        chreode_fn = _load_chreode(device, args.chreode_checkpoint)
        results["chreode"] = _measure(
            "chreode",
            chreode_fn,
            nfe=1,
            warmup=args.warmup,
            n=args.n,
            device=device,
            checkpoint=str(args.chreode_checkpoint.relative_to(ROOT)),
        )
    else:
        missing["chreode"] = f"missing checkpoint: {args.chreode_checkpoint}"

    if (args.prescient_run_dir / "config.pt").exists() and (args.prescient_run_dir / "train.best.pt").exists():
        pres_one, pres_88 = _load_prescient(device, args.prescient_run_dir)
        results["prescient_one_euler"] = _measure(
            "prescient_one_euler",
            pres_one,
            nfe=1,
            warmup=args.warmup,
            n=args.n,
            device=device,
            checkpoint=str((args.prescient_run_dir / "train.best.pt").relative_to(ROOT)),
        )
        results["prescient_88_euler"] = _measure(
            "prescient_88_euler",
            pres_88,
            nfe=88,
            warmup=max(3, args.warmup // 3),
            n=max(20, args.n // 3),
            device=device,
            checkpoint=str((args.prescient_run_dir / "train.best.pt").relative_to(ROOT)),
        )
    else:
        missing["prescient"] = f"missing run artifacts under: {args.prescient_run_dir}"

    if args.branchsbm_checkpoint.exists():
        branch_one, branch_40 = _load_branchsbm(device, args.branchsbm_checkpoint)
        results["branchsbm_one_flow"] = _measure(
            "branchsbm_one_flow",
            branch_one,
            nfe=1,
            warmup=args.warmup,
            n=args.n,
            device=device,
            checkpoint=str(args.branchsbm_checkpoint.relative_to(ROOT)),
        )
        results["branchsbm_40_euler"] = _measure(
            "branchsbm_40_euler",
            branch_40,
            nfe=40,
            warmup=max(3, args.warmup // 3),
            n=max(20, args.n // 3),
            device=device,
            checkpoint=str(args.branchsbm_checkpoint.relative_to(ROOT)),
        )
    else:
        missing["branchsbm"] = f"missing checkpoint: {args.branchsbm_checkpoint}"

    for method in ["cellflow", "scgen", "cpa"]:
        missing[method] = "no loadable inference checkpoint / runner found in current workspace"

    payload = {
        "date": "2026-05-07",
        "hardware": hardware,
        "device_type": str(device),
        "task": "operator microbenchmark, single source latent query, fp32, batch size 1",
        "notes": [
            "Only methods with loadable checkpoints in this workspace are reported.",
            "Missing methods are not approximated by surrogate architectures.",
            "FLOPs are from torch.profiler with_flops=True and may undercount autograd-gradient operations.",
        ],
        "results": results,
        "missing": missing,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
