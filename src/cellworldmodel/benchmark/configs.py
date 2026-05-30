"""Per-dataset hyperparameter configs for M1/M2/M7/M8 benchmark runs.

Extracted from run_benchmark.py so both run_benchmark.py and
run_intermediate_eval.py can share without cross-script imports.
"""
from __future__ import annotations


DATASET_CONFIGS: dict[str, dict] = {
    "mouse": {
        "hidden_dim": 128, "n_layers": 3, "noise_dim": 8, "time_emb_dim": 32,
        "batch_size": 256, "K": 8, "lr": 1e-3,
        "lambda_mmd": 1.0, "lambda_w2": 1.0, "lambda_drift": 1.0, "lambda_down": 0.1,
        "sinkhorn_eps": 0.05, "grad_clip": None, "default_epochs": 300,
    },
    "clonidine": {
        "hidden_dim": 512, "n_layers": 3, "noise_dim": 32, "time_emb_dim": 64,
        "batch_size": 256, "K": 4, "lr": 3e-4,
        "lambda_mmd": 1.0, "lambda_w2": 0.5, "lambda_drift": 1.0, "lambda_down": 0.1,
        "sinkhorn_eps": 0.1, "grad_clip": 1.0, "default_epochs": 500,
    },
    "trametinib": {
        "hidden_dim": 512, "n_layers": 3, "noise_dim": 32, "time_emb_dim": 64,
        "batch_size": 256, "K": 4, "lr": 3e-4,
        "lambda_mmd": 1.0, "lambda_w2": 0.5, "lambda_drift": 1.0, "lambda_down": 0.1,
        "sinkhorn_eps": 0.1, "grad_clip": 1.0, "default_epochs": 500,
    },
    "veres": {
        "hidden_dim": 512, "n_layers": 3, "noise_dim": 16, "time_emb_dim": 32,
        "batch_size": 256, "K": 8, "lr": 3e-4,
        "lambda_mmd": 1.0, "lambda_w2": 1.0, "lambda_drift": 1.0, "lambda_down": 0.1,
        "sinkhorn_eps": 0.05, "grad_clip": 1.0, "default_epochs": 500,
    },
    "norman": {
        "hidden_dim": 512, "n_layers": 3, "noise_dim": 32, "time_emb_dim": 64,
        "batch_size": 256, "K": 4, "lr": 3e-4,
        "lambda_mmd": 1.0, "lambda_w2": 0.5, "lambda_drift": 1.0, "lambda_down": 0.1,
        "sinkhorn_eps": 0.1, "grad_clip": 1.0, "default_epochs": 500,
    },
    "weinreb_hvg": {
        # PCA-50 space (same as PRESCIENT input), 3 timepoints (d2/d4/d6), delta=4.
        # Similar scale to Veres 30D. Used for apples-to-apples vs PRESCIENT.
        "hidden_dim": 512, "n_layers": 3, "noise_dim": 32, "time_emb_dim": 64,
        "batch_size": 256, "K": 8, "lr": 3e-4,
        "lambda_mmd": 1.0, "lambda_w2": 1.0, "lambda_drift": 1.0, "lambda_down": 0.1,
        "sinkhorn_eps": 0.05, "grad_clip": 1.0, "default_epochs": 500,
    },
    "weinreb_scvi": {
        # scVI 64-dim latent (ortholog-filtered, from output/scvi/v1_weinreb/).
        # Gaussian prior → d2→d6 shift expected 1-3 units (vs PCA-50 12.68).
        "hidden_dim": 512, "n_layers": 3, "noise_dim": 32, "time_emb_dim": 64,
        "batch_size": 256, "K": 8, "lr": 3e-4,
        "lambda_mmd": 1.0, "lambda_w2": 1.0, "lambda_drift": 1.0, "lambda_down": 0.1,
        "sinkhorn_eps": 0.05, "grad_clip": 1.0, "default_epochs": 500,
    },
    "veres_scvi": {
        # scVI 64-dim latent on Veres Stage 5 (ortholog-renamed, 51K × 13,660 → 64D).
        # 8 timepoints (CellWeek 0..7), shift ~2.8 units (vs BranchSBM PCA-30 ~8).
        "hidden_dim": 512, "n_layers": 3, "noise_dim": 32, "time_emb_dim": 64,
        "batch_size": 256, "K": 8, "lr": 3e-4,
        "lambda_mmd": 1.0, "lambda_w2": 1.0, "lambda_drift": 1.0, "lambda_down": 0.1,
        "sinkhorn_eps": 0.05, "grad_clip": 1.0, "default_epochs": 500,
    },
    "paper_weinreb_scvi128": {
        # Foundation VAE 128D latent exported in output/paper_bench/representations.
        "hidden_dim": 512, "n_layers": 3, "noise_dim": 32, "time_emb_dim": 64,
        "batch_size": 256, "K": 8, "lr": 3e-4,
        "lambda_mmd": 1.0, "lambda_w2": 1.0, "lambda_drift": 1.0, "lambda_down": 0.1,
        "sinkhorn_eps": 0.05, "grad_clip": 1.0, "default_epochs": 500,
    },
    "paper_veres_scvi128": {
        "hidden_dim": 512, "n_layers": 3, "noise_dim": 32, "time_emb_dim": 64,
        "batch_size": 256, "K": 8, "lr": 3e-4,
        "lambda_mmd": 1.0, "lambda_w2": 1.0, "lambda_drift": 1.0, "lambda_down": 0.1,
        "sinkhorn_eps": 0.05, "grad_clip": 1.0, "default_epochs": 500,
    },
}

DEFAULT_PCS = {
    "mouse": 2, "clonidine": 50, "trametinib": 50, "veres": 30,
    "norman": 128, "weinreb_hvg": 50, "weinreb_scvi": 64, "veres_scvi": 64,
    "paper_weinreb_scvi128": 128, "paper_veres_scvi128": 128,
}
