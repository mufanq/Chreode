"""Transition index and sampler scaffolding for foundation dynamics."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FoundationTransitionIndex:
    transitions: pd.DataFrame
    cell_index_path: str
    manifest: dict


@dataclass(frozen=True)
class FoundationTransitionBatchIds:
    source_ids: np.ndarray
    target_ids: np.ndarray
    delta: float
    source_t: float
    target_t: float
    leaf_dataset: str
    transition_id: int


def ordered_pairs(times: Sequence[float], policy: str = "all_ordered") -> list[tuple[float, float]]:
    times = sorted(float(t) for t in times)
    if policy == "adjacent":
        return list(zip(times[:-1], times[1:]))
    if policy == "endpoint":
        return [(times[0], times[-1])] if len(times) >= 2 else []
    if policy == "all_ordered":
        return [(s, t) for i, s in enumerate(times) for t in times[i + 1:]]
    raise ValueError(f"Unknown transition pair policy: {policy}")


def build_transition_index(
    catalog_dir: str | Path,
    *,
    split: str = "train",
    pair_policy: str = "all_ordered",
) -> FoundationTransitionIndex:
    catalog_dir = Path(catalog_dir)
    cell_index_path = catalog_dir / "cell_index.parquet"
    cells = pd.read_parquet(cell_index_path, columns=[
        "global_cell_id", "foundation_split", "leaf_dataset", "timepoint"
    ])
    cells = cells[cells["foundation_split"] == split].copy()
    rows = []
    transition_id = 0
    for leaf, leaf_df in cells.groupby("leaf_dataset", sort=True):
        times = sorted(float(t) for t in leaf_df["timepoint"].dropna().unique())
        for source_t, target_t in ordered_pairs(times, pair_policy):
            n_source = int(((leaf_df["timepoint"] == source_t)).sum())
            n_target = int(((leaf_df["timepoint"] == target_t)).sum())
            if n_source == 0 or n_target == 0:
                continue
            max_delta = max(1e-8, float(max(times) - min(times)))
            rows.append({
                "transition_id": transition_id,
                "leaf_dataset": str(leaf),
                "source_t": float(source_t),
                "target_t": float(target_t),
                "delta": float(target_t - source_t),
                "delta_norm": float((target_t - source_t) / max_delta),
                "split": split,
                "n_source": n_source,
                "n_target": n_target,
            })
            transition_id += 1
    transitions = pd.DataFrame(rows)
    manifest = {
        "catalog_dir": str(catalog_dir),
        "cell_index_path": str(cell_index_path),
        "split": split,
        "pair_policy": pair_policy,
        "n_transitions": int(len(transitions)),
        "n_leaf_datasets": int(transitions["leaf_dataset"].nunique()) if len(transitions) else 0,
    }
    return FoundationTransitionIndex(
        transitions=transitions,
        cell_index_path=str(cell_index_path),
        manifest=manifest,
    )


def write_transition_index(index: FoundationTransitionIndex, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index.transitions.to_parquet(output_dir / "transition_index.parquet", index=False)
    with (output_dir / "transition_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(index.manifest, handle, indent=2, sort_keys=True)
    return index.manifest


class FoundationTransitionIdSampler:
    """Sample source/target global cell ids for latent dynamics training."""

    def __init__(
        self,
        catalog_dir: str | Path,
        transition_index: pd.DataFrame,
        *,
        split: str = "train",
        leaf_sampling_alpha: float = 0.5,
    ) -> None:
        catalog_dir = Path(catalog_dir)
        self.cells = pd.read_parquet(catalog_dir / "cell_index.parquet", columns=[
            "global_cell_id", "foundation_split", "leaf_dataset", "timepoint"
        ])
        self.cells = self.cells[self.cells["foundation_split"] == split].copy()
        self.transitions = transition_index.reset_index(drop=True)
        if self.transitions.empty:
            raise ValueError("transition_index is empty")
        leaf_counts = self.transitions["leaf_dataset"].value_counts().sort_index()
        weights = leaf_counts.to_numpy(dtype=float) ** float(leaf_sampling_alpha)
        self.leaves = leaf_counts.index.to_numpy(dtype=object)
        self.leaf_probs = weights / weights.sum()
        self._pool_cache: dict[tuple[str, float], np.ndarray] = {}

    def _pool(self, leaf: str, timepoint: float) -> np.ndarray:
        key = (str(leaf), float(timepoint))
        cached = self._pool_cache.get(key)
        if cached is not None:
            return cached
        mask = (self.cells["leaf_dataset"] == leaf) & (self.cells["timepoint"] == timepoint)
        pool = self.cells.loc[mask, "global_cell_id"].to_numpy(dtype=np.int64)
        if len(pool) == 0:
            raise ValueError(f"No cells for leaf={leaf!r} timepoint={timepoint}")
        self._pool_cache[key] = pool
        return pool

    def sample(self, batch_size: int, rng: np.random.Generator) -> FoundationTransitionBatchIds:
        leaf = str(rng.choice(self.leaves, p=self.leaf_probs))
        options = self.transitions[self.transitions["leaf_dataset"] == leaf]
        row = options.iloc[int(rng.integers(len(options)))]
        source_pool = self._pool(leaf, float(row.source_t))
        target_pool = self._pool(leaf, float(row.target_t))
        source_ids = rng.choice(source_pool, size=batch_size, replace=batch_size > len(source_pool))
        target_ids = rng.choice(target_pool, size=batch_size, replace=batch_size > len(target_pool))
        return FoundationTransitionBatchIds(
            source_ids=source_ids.astype(np.int64),
            target_ids=target_ids.astype(np.int64),
            delta=float(row.delta),
            source_t=float(row.source_t),
            target_t=float(row.target_t),
            leaf_dataset=leaf,
            transition_id=int(row.transition_id),
        )
