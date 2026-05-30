"""Thin wrapper around scvi-tools for Cell World Model encoder.

Provides unified save/load/encode/decode interfaces so downstream code
(WeinrebScVIAdapter, MouseEmbryoScVIAdapter, Stage 2 runner) does not depend
on scvi-tools internals.

Data scaling note (Weinreb, mouse embryo unified):
  - Input AnnData is L1-normalized float (target_sum ~2500 for Weinreb,
    per-dataset target_sum for mouse embryo). True raw integer counts NOT
    publicly available (Ruichen README).
  - Standard preprocessing: `sc.pp.normalize_total(target_sum=1e4) + sc.pp.log1p`
    so all datasets share same scale, then scVI with `gene_likelihood='normal'`
    on log1p-transformed data. This matches common practice for
    "pre-normalized" single-cell data (e.g., scGen, scVI tutorials on
    pre-processed data).

Usage:
    from cellworldmodel.model.scvi_encoder import ScVIEncoder
    enc = ScVIEncoder(n_latent=64, n_hidden=256, n_layers=2,
                      batch_key="Time point", gene_likelihood="normal")
    enc.fit(adata, max_epochs=200, batch_size=1024)
    z = enc.encode(adata)  # (n_cells, 64) numpy
    enc.save("output/scvi/v1_weinreb/")
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import scanpy as sc
import torch

try:
    import scvi
    from scvi.model import SCVI
except ImportError as e:
    raise ImportError("scvi-tools required. pip install scvi-tools") from e


class ScVIEncoder:
    """scVI wrapper: preprocess (normalize + log1p), fit, encode, save/load."""

    def __init__(
        self,
        n_latent: int = 64,
        n_hidden: int = 256,
        n_layers: int = 2,
        batch_key: Optional[str] = None,
        categorical_covariate_keys: Optional[list[str]] = None,
        gene_likelihood: str = "normal",  # "nb" | "zinb" | "normal"
        dispersion: str = "gene",
        dropout_rate: float = 0.1,
        target_sum: float = 1e4,
    ):
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.batch_key = batch_key
        self.categorical_covariate_keys = categorical_covariate_keys
        self.gene_likelihood = gene_likelihood
        self.dispersion = dispersion
        self.dropout_rate = dropout_rate
        self.target_sum = target_sum
        self.model: Optional[SCVI] = None
        self.adata_used = None

    def _preprocess(self, adata, inplace: bool = False):
        """Normalize to target_sum + log1p if not already done. Returns the adata object."""
        if not inplace:
            adata = adata.copy()
        # Re-normalize (in case input was L1-normalized to non-1e4)
        sc.pp.normalize_total(adata, target_sum=self.target_sum)
        sc.pp.log1p(adata)
        return adata

    def setup(self, adata):
        """Run scvi-tools setup_anndata on adata (in-place modifies adata.uns)."""
        adata = self._preprocess(adata)
        SCVI.setup_anndata(
            adata,
            batch_key=self.batch_key,
            categorical_covariate_keys=self.categorical_covariate_keys,
        )
        return adata

    def fit(
        self,
        adata,
        max_epochs: int = 200,
        batch_size: int = 1024,
        early_stopping: bool = True,
        accelerator: str = "auto",
        **train_kwargs,
    ):
        """Train scVI model on adata.

        adata can be un-preprocessed (we call normalize_total + log1p internally).
        """
        adata_setup = self.setup(adata)
        self.adata_used = adata_setup

        self.model = SCVI(
            adata_setup,
            n_latent=self.n_latent,
            n_hidden=self.n_hidden,
            n_layers=self.n_layers,
            dispersion=self.dispersion,
            gene_likelihood=self.gene_likelihood,
            dropout_rate=self.dropout_rate,
        )
        print(f"[scVI] Model: n_latent={self.n_latent}, n_hidden={self.n_hidden}, "
              f"n_layers={self.n_layers}, likelihood={self.gene_likelihood}")
        n_params = sum(p.numel() for p in self.model.module.parameters())
        print(f"[scVI] Params: {n_params:,}")

        self.model.train(
            max_epochs=max_epochs,
            batch_size=batch_size,
            early_stopping=early_stopping,
            accelerator=accelerator,
            **train_kwargs,
        )
        return self

    def encode(self, adata=None) -> np.ndarray:
        """Return latent z for adata. If adata is None, use training adata."""
        assert self.model is not None, "Call fit() first"
        if adata is None:
            adata = self.adata_used
        else:
            # Assume adata is already preprocessed matching setup; if not, run setup
            if "_scvi_batch" not in adata.obs.columns:
                adata = self._preprocess(adata)
                SCVI.setup_anndata(
                    adata, batch_key=self.batch_key,
                    categorical_covariate_keys=self.categorical_covariate_keys,
                )
        z = self.model.get_latent_representation(adata)
        return np.asarray(z, dtype=np.float32)

    def save(self, save_dir: str | Path, overwrite: bool = True) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        assert self.model is not None, "Call fit() first"
        self.model.save(str(save_dir), overwrite=overwrite, save_anndata=True)
        # Also save config
        import yaml
        with open(save_dir / "encoder_config.yaml", "w") as f:
            yaml.safe_dump({
                "n_latent": self.n_latent,
                "n_hidden": self.n_hidden,
                "n_layers": self.n_layers,
                "batch_key": self.batch_key,
                "categorical_covariate_keys": self.categorical_covariate_keys,
                "gene_likelihood": self.gene_likelihood,
                "dispersion": self.dispersion,
                "dropout_rate": self.dropout_rate,
                "target_sum": self.target_sum,
            }, f)
        print(f"[scVI] Saved to {save_dir}")

    @classmethod
    def load(cls, save_dir: str | Path, adata=None) -> "ScVIEncoder":
        save_dir = Path(save_dir)
        import yaml
        with open(save_dir / "encoder_config.yaml") as f:
            cfg = yaml.safe_load(f)
        enc = cls(**cfg)
        enc.model = SCVI.load(str(save_dir), adata=adata)
        enc.adata_used = enc.model.adata
        return enc
