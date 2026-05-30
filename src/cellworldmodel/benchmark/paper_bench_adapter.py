"""Adapters over paper-benchmark exported foundation scVI representations."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cellworldmodel.benchmark.branchsbm_adapter import TimePointAdapter
from cellworldmodel.training.split_policy import SplitIndices


ROOT = Path(__file__).parent.parent.parent.parent


class PaperBenchScVI128Adapter(TimePointAdapter):
    """Timepoint adapter for `output/paper_bench/representations/*`.

    This keeps fine-tuning in the same 128D foundation latent space used by the
    A0/A1/A2 zero-shot evaluators and by the A2 Genhui checkpoint.
    """

    def __init__(self, dataset: str, split_ratio: float = 0.8, seed: int = 42):
        self.dataset = dataset
        root = ROOT / "output" / "paper_bench" / "representations" / dataset
        reps_path = root / "representations.npz"
        meta_path = root / "metadata.tsv"
        if not reps_path.exists():
            raise FileNotFoundError(f"paper-bench representations not found: {reps_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"paper-bench metadata not found: {meta_path}")
        reps = np.load(reps_path)
        meta = pd.read_csv(meta_path, sep="\t")
        z = reps["scvi128"].astype(np.float32)
        if len(z) != len(meta):
            raise ValueError(f"representation/meta length mismatch: {len(z)} vs {len(meta)}")

        self.meta = meta
        self.coords_by_t = {}
        self.splits_by_t: dict[float, SplitIndices] = {}
        for t in sorted(meta["time"].astype(float).unique()):
            mask = np.isclose(meta["time"].astype(float).to_numpy(), float(t))
            self.coords_by_t[float(t)] = z[mask]
            local_split = meta.loc[mask, "split"].astype(str).to_numpy()
            self.splits_by_t[float(t)] = SplitIndices(
                train=np.where(local_split == "train")[0].astype(np.int64),
                val=np.where(local_split == "val")[0].astype(np.int64),
                test=np.where(local_split == "test")[0].astype(np.int64),
            )
        self.dim = int(z.shape[1])
        self.timepoints = [float(t) for t in sorted(self.coords_by_t)]
        self._meta_name = f"paper_{dataset}_scvi128"

        self._final_celltypes = None
        if "cell_type" in meta.columns:
            final_t = self.timepoints[-1]
            mask_final = np.isclose(meta["time"].astype(float).to_numpy(), final_t)
            self._final_celltypes = meta.loc[mask_final, "cell_type"].astype(str).to_numpy()

        # Preserve the exported paper-bench split instead of creating a new one.
        source_split = self.splits_by_t[self.timepoints[0]]
        self.train_src_idx = source_split.train
        self.test_src_idx = source_split.test
        self.seed = seed

    def get_intermediate(self, t: float) -> torch.Tensor:
        t = float(t)
        if t not in self.coords_by_t:
            raise ValueError(f"timepoint {t} not in {self.timepoints}")
        return torch.from_numpy(self.coords_by_t[t])

    def get_target_cluster_labels(self, n_clusters: int = 11, seed: int = 42) -> np.ndarray:
        if self._final_celltypes is None:
            from sklearn.cluster import KMeans
            target = self.coords_by_t[self.timepoints[-1]]
            return KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(target).labels_.astype(np.int64)
        unique = sorted(set(self._final_celltypes))
        lut = {name: i for i, name in enumerate(unique)}
        return np.asarray([lut[name] for name in self._final_celltypes], dtype=np.int64)
