"""ZESTA control-only time extrapolation adapter.

This adapter uses CellFlow's downloaded ZESTA dataset:

    data/external/cellflow/zesta.h5ad

Task definition for the first comparison:
  - source: control cells at t=18
  - train targets: control cells at t=24/36/48
  - OOD target: control cells at t=72
  - representation: obsm["X_aligned"] by default

The h5ad is large (2.7M cells), so we read only control-cell latent
representations from h5py into memory. The control subset is about 610K cells
and X_aligned has 100 dimensions, which is tractable as float32.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence

import h5py
import numpy as np
import torch


DEFAULT_ZESTA_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "data" / "external" / "cellflow" / "zesta.h5ad"
)


@dataclass
class ZestaTimeBatch:
    source: torch.Tensor
    target: torch.Tensor
    delta: float
    source_time: float
    target_time: float
    meta: dict


class ZestaTimeAdapter:
    """Control-only ZESTA time extrapolation adapter."""

    _meta_name = "zesta_control_time"

    def __init__(
        self,
        data_path: Optional[Path | str] = None,
        rep_key: str = "X_aligned",
        source_t: float = 18.0,
        target_times: Sequence[float] = (24.0, 36.0, 48.0, 72.0),
        train_target_times: Sequence[float] = (24.0, 36.0, 48.0),
        split_ratio: float = 0.8,
        seed: int = 42,
        max_cells_per_timepoint: Optional[int] = None,
    ) -> None:
        self.path = Path(data_path) if data_path else DEFAULT_ZESTA_PATH
        if not self.path.exists():
            raise FileNotFoundError(f"ZESTA h5ad not found: {self.path}")
        self.rep_key = rep_key
        self.source_t = float(source_t)
        self.target_times = [float(t) for t in target_times]
        self.train_target_times = [float(t) for t in train_target_times]
        self.timepoints = [self.source_t] + self.target_times
        self.seed = seed

        self.coords_by_t: dict[float, np.ndarray] = {}
        with h5py.File(self.path, "r") as f:
            rep = f[f"obsm/{rep_key}"]
            timepoints = f["obs/timepoint"][:].astype(float)
            is_control = f["obs/is_control"][:].astype(bool)
            self.dim = int(rep.shape[1])
            rng = np.random.default_rng(seed)
            for t in self.timepoints:
                idx = np.where(is_control & np.isclose(timepoints, t))[0]
                if max_cells_per_timepoint is not None and idx.shape[0] > max_cells_per_timepoint:
                    idx = np.sort(rng.choice(idx, size=max_cells_per_timepoint, replace=False))
                self.coords_by_t[float(t)] = rep[idx].astype(np.float32)

        self._init_split(split_ratio, seed)

    def _init_split(self, split_ratio: float, seed: int) -> None:
        n0 = self.coords_by_t[self.source_t].shape[0]
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n0)
        n_train = int(n0 * split_ratio)
        self.train_src_idx = perm[:n_train]
        self.test_src_idx = perm[n_train:]

    def get_transition(
        self,
        split: Literal["train", "test"] = "train",
        source_t: Optional[float] = None,
        target_t: Optional[float] = None,
    ) -> ZestaTimeBatch:
        if source_t is None:
            source_t = self.source_t
        if target_t is None:
            target_t = self.train_target_times[-1]
        source_t = float(source_t)
        target_t = float(target_t)
        src_all = self.coords_by_t[source_t]
        if source_t == self.source_t:
            idx = self.train_src_idx if split == "train" else self.test_src_idx
            src = src_all[idx]
        else:
            src = src_all
        tgt = self.coords_by_t[target_t]
        return ZestaTimeBatch(
            source=torch.from_numpy(src),
            target=torch.from_numpy(tgt),
            delta=float(target_t - source_t),
            source_time=source_t,
            target_time=target_t,
            meta={"dataset": self._meta_name, "rep_key": self.rep_key},
        )

    def sample_source_batch(
        self,
        batch_size: int,
        split: Literal["train", "test"] = "train",
        source_t: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> torch.Tensor:
        if rng is None:
            rng = np.random.default_rng()
        if source_t is None:
            source_t = self.source_t
        source_t = float(source_t)
        src_all = self.coords_by_t[source_t]
        if source_t == self.source_t:
            avail = self.train_src_idx if split == "train" else self.test_src_idx
        else:
            avail = np.arange(src_all.shape[0])
        idx = rng.choice(avail, size=batch_size, replace=batch_size > len(avail))
        return torch.from_numpy(src_all[idx])

    def sample_target_batch(
        self,
        batch_size: int,
        target_t: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> torch.Tensor:
        if rng is None:
            rng = np.random.default_rng()
        if target_t is None:
            target_t = self.train_target_times[-1]
        tgt_all = self.coords_by_t[float(target_t)]
        idx = rng.choice(tgt_all.shape[0], size=batch_size, replace=batch_size > tgt_all.shape[0])
        return torch.from_numpy(tgt_all[idx])


def _main() -> None:
    adapter = ZestaTimeAdapter()
    print(f"dim={adapter.dim}, timepoints={adapter.timepoints}")
    for t, arr in adapter.coords_by_t.items():
        print(f"t={t}: {arr.shape}")
    b = adapter.get_transition(split="train", target_t=48.0)
    print(f"train source={b.source.shape}, target={b.target.shape}, delta={b.delta}")


if __name__ == "__main__":
    _main()
