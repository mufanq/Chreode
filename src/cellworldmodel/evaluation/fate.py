"""Weinreb fate Pearson evaluation."""
from __future__ import annotations

import os
import random
import time
from collections import Counter
from pathlib import Path

import annoy
import numpy as np
import pandas as pd
import scipy.stats
import torch

from cellworldmodel.benchmark.configs import DATASET_CONFIGS
from cellworldmodel.benchmark.registry import build_model
from cellworldmodel.benchmark.weinreb_scvi_adapter import WeinrebScVIAdapter
from cellworldmodel.training.benchmark_loop import train_method
from cellworldmodel.training.checkpointing import load_model_checkpoint


# Project root. Override via the CHREODE_ROOT environment variable; defaults to
# the current working directory so commands like
#   cd Chreode && python -m cellworldmodel.evaluation.fate ...
# work without any setup.
ROOT = Path(os.environ.get("CHREODE_ROOT", "."))


def run_fate_pearson(method: str, seed: int, split_seed: int = 42,
                     epochs: int | None = None, n_sim: int = 2000,
                     checkpoint: str | None = None,
                     verbose: bool = True,
                     train_log_callback=None) -> dict:
    t0 = time.time()
    eval_seed = int(seed) + 20260427
    random.seed(eval_seed)
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(eval_seed)

    clonal = np.load(ROOT / "output" / "phase0" / "weinreb_clonal.npz", allow_pickle=True)
    timepoints = clonal["timepoints"]
    cell_type_idx = clonal["cell_type_idx"]
    early_cells = clonal["early_cells"]
    neu_mo_mask = clonal["neu_mo_mask"]
    heldout_mask = clonal["heldout_mask"]
    smoothed_gt = clonal["smoothed_groundtruth"]
    if verbose:
        print(
            f"Clonal artifacts loaded. early={early_cells.sum()}, "
            f"heldout={heldout_mask.sum()}, has_smoothed_gt={(~np.isnan(smoothed_gt)).sum()}"
        )

    latent = np.load(ROOT / "output" / "scvi" / "v1_weinreb" / "latent_z.npy")
    meta = pd.read_parquet(ROOT / "output" / "scvi" / "v1_weinreb" / "latent_metadata.parquet")
    assert len(latent) == len(meta), f"latent/meta length mismatch {len(latent)} vs {len(meta)}"

    d6_mask = (timepoints == 6)
    d6_latent = latent[d6_mask]
    d6_celltype_idx = cell_type_idx[d6_mask]
    if verbose:
        print(f"d6 cells: {d6_mask.sum()}, latent shape {d6_latent.shape}")
    ann_index = annoy.AnnoyIndex(latent.shape[1], "euclidean")
    for i, vec in enumerate(d6_latent):
        ann_index.add_item(i, vec.astype(np.float32))
    ann_index.build(10)
    if verbose:
        print("Annoy d6 index built.")

    adapter = WeinrebScVIAdapter(seed=split_seed)
    cfg = dict(DATASET_CONFIGS["weinreb_scvi"])
    epochs = epochs or cfg["default_epochs"]
    train_batch = adapter.get_transition(split="train")
    tau_init = float(train_batch.delta / np.log(2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint is not None:
        ckpt = load_model_checkpoint(checkpoint, map_location=device)
        if "cfg" in ckpt:
            cfg.update(ckpt["cfg"])
        model = build_model(method, adapter.dim, cfg, tau_init).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        if verbose:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"Loaded {method} checkpoint {checkpoint} ({n_params:,} params)")
    else:
        model = build_model(method, adapter.dim, cfg, tau_init).to(device)
        if verbose:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"Training {method} for {epochs} epochs (seed={seed}), {n_params:,} params")
        train_method(
            method, adapter, model, device, cfg,
            epochs=epochs, seed=seed, log_every=100, log_callback=train_log_callback,
        )
    model.eval()

    early_heldout = early_cells & heldout_mask & neu_mo_mask & (~np.isnan(smoothed_gt))
    early_heldout_idx = np.where(early_heldout)[0]
    if verbose:
        print(f"Evaluating on {early_heldout.sum()} early heldout Neu/Mono-lineage cells with clonal GT")

    if len(early_heldout_idx) > 2000:
        rng = np.random.default_rng(seed)
        early_heldout_idx = rng.choice(early_heldout_idx, size=2000, replace=False)

    delta_to_d6 = 4.0
    scores = np.zeros(len(early_heldout_idx))
    has_any = np.zeros(len(early_heldout_idx), dtype=bool)
    dim = latent.shape[1]
    K = min(n_sim, 256)

    with torch.no_grad():
        for j, cell_idx in enumerate(early_heldout_idx):
            src = torch.tensor(latent[cell_idx:cell_idx + 1], device=device, dtype=torch.float32)
            neu = 0
            mono = 0
            remaining = n_sim
            while remaining > 0:
                k = min(remaining, K)
                eps = torch.randn(1, k, dim, device=device)
                delta_t = torch.full((1,), delta_to_d6, device=device, dtype=torch.float32)
                preds = model(src, delta_t, eps).reshape(-1, dim).cpu().numpy()
                for pred in preds:
                    nn = ann_index.get_nns_by_vector(pred.astype(np.float32), 20)
                    nn_labels = d6_celltype_idx[nn]
                    label = Counter(nn_labels).most_common(1)[0][0]
                    if label == 5:
                        mono += 1
                    elif label == 6:
                        neu += 1
                remaining -= k
            scores[j] = (neu + 1) / (neu + mono + 2)
            has_any[j] = (neu + mono) > 0
            if verbose and (j + 1) % 100 == 0:
                print(f"  {j+1}/{len(early_heldout_idx)}: latest score={scores[j]:.3f} (neu={neu}, mono={mono})")

    gt = smoothed_gt[early_heldout_idx]
    r_all, p_all = scipy.stats.pearsonr(gt, scores)
    r_mask, p_mask = scipy.stats.pearsonr(gt[has_any], scores[has_any])
    elapsed = time.time() - t0
    if verbose:
        print(f"\n=== {method} seed {seed} ===")
        print(f"n_evaluated: {len(early_heldout_idx)}, n_with_pred: {has_any.sum()}")
        print(f"Pearson r (all):    {r_all:.4f} (p={p_all:.2e})")
        print(f"Pearson r (masked): {r_mask:.4f} (p={p_mask:.2e})  ← paper main")
        print(f"Elapsed: {elapsed:.1f}s")

    return {
        "method": method,
        "seed": seed,
        "split_seed": split_seed,
        "eval_seed": eval_seed,
        "checkpoint": checkpoint,
        "n_evaluated": len(early_heldout_idx),
        "n_with_pred": int(has_any.sum()),
        "n_sim_per_cell": n_sim,
        "pearson_r_all": float(r_all),
        "pearson_p_all": float(p_all),
        "pearson_r_masked": float(r_mask),
        "pearson_p_masked": float(p_mask),
        "train_time_s": elapsed,
    }
