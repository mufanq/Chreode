"""Shared Weinreb clonal FATE scoring utilities."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.neighbors import NearestNeighbors


MONOCYTE_LABEL = 5
NEUTROPHIL_LABEL = 6


@dataclass
class ClonalFateInputs:
    z: np.ndarray
    meta: pd.DataFrame
    timepoints: np.ndarray
    labels: np.ndarray
    celltype_order: np.ndarray
    eval_idx: np.ndarray
    source: np.ndarray
    truth: np.ndarray
    metadata_split: np.ndarray
    d6_z: np.ndarray
    d6_labels: np.ndarray
    atlas_split: str


@dataclass
class FateScoreResult:
    scores: np.ndarray
    has_any: np.ndarray
    neu_count: np.ndarray
    mono_count: np.ndarray
    majority_label: np.ndarray
    majority_label_name: np.ndarray
    other_count: np.ndarray
    disp_norm_mean: np.ndarray | None


def pearson_or_nan(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size < 2 or y.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    r, p = scipy.stats.pearsonr(x, y)
    return float(r), float(p)


def load_clonal_fate_inputs(
    *,
    representations: str | Path,
    metadata: str | Path,
    clonal: str | Path,
    space: str,
    atlas_split: str,
    max_eval_cells: int,
    seed: int,
) -> ClonalFateInputs:
    reps = np.load(representations)
    meta = pd.read_csv(metadata, sep="\t")
    z = reps[space].astype(np.float32)
    clonal_data = np.load(clonal, allow_pickle=True)
    timepoints = clonal_data["timepoints"]
    labels = clonal_data["cell_type_idx"].astype(np.int64)
    celltype_order = clonal_data["celltype_order"]
    early_cells = clonal_data["early_cells"]
    heldout = clonal_data["heldout_mask"]
    neu_mo = clonal_data["neu_mo_mask"]
    truth_all = clonal_data["smoothed_groundtruth"]
    if len(z) != len(timepoints):
        raise ValueError(f"representation/clonal length mismatch: {len(z)} vs {len(timepoints)}")
    if len(meta) != len(timepoints):
        raise ValueError(f"metadata/clonal length mismatch: {len(meta)} vs {len(timepoints)}")
    if not np.allclose(meta["time"].astype(float).to_numpy(), timepoints.astype(float)):
        raise ValueError("metadata time column is not aligned with clonal timepoints")

    eval_mask = early_cells & heldout & neu_mo & np.isfinite(truth_all)
    eval_idx = np.where(eval_mask)[0]
    if eval_idx.size > int(max_eval_cells):
        rng = np.random.default_rng(seed)
        eval_idx = np.sort(rng.choice(eval_idx, size=int(max_eval_cells), replace=False))

    d6_mask = np.isclose(timepoints.astype(float), 6.0)
    atlas_split = str(atlas_split)
    if atlas_split != "all":
        if atlas_split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported atlas_split={atlas_split!r}")
        d6_mask = d6_mask & (meta["split"].astype(str).to_numpy() == atlas_split)
    if not d6_mask.any():
        raise ValueError(f"No d6 cells available for atlas_split={atlas_split!r}")

    return ClonalFateInputs(
        z=z,
        meta=meta,
        timepoints=timepoints,
        labels=labels,
        celltype_order=celltype_order,
        eval_idx=eval_idx,
        source=z[eval_idx].astype(np.float32),
        truth=truth_all[eval_idx].astype(np.float64),
        metadata_split=meta["split"].astype(str).to_numpy()[eval_idx],
        d6_z=z[d6_mask].astype(np.float32),
        d6_labels=labels[d6_mask],
        atlas_split=atlas_split,
    )


def score_fate_predictions(
    pred: np.ndarray,
    data: ClonalFateInputs,
    *,
    source: np.ndarray | None = None,
    n_neighbors: int = 20,
    tie_policy: str = "majority",
) -> FateScoreResult:
    """Score stochastic endpoint predictions by d6 Neu/Mono 20-NN fate bias."""
    if pred.ndim != 3:
        raise ValueError(f"Expected pred shape (B, K, D), got {pred.shape}")
    if pred.shape[0] != data.eval_idx.shape[0]:
        raise ValueError(f"Prediction B={pred.shape[0]} does not match eval cells={data.eval_idx.shape[0]}")

    nn_model = NearestNeighbors(n_neighbors=int(n_neighbors), metric="euclidean")
    nn_model.fit(data.d6_z.astype(np.float32))
    flat = pred.reshape(-1, pred.shape[-1]).astype(np.float32)
    scores = np.zeros(pred.shape[0], dtype=np.float64)
    has_any = np.zeros(pred.shape[0], dtype=bool)
    neu_count = np.zeros(pred.shape[0], dtype=np.int64)
    mono_count = np.zeros(pred.shape[0], dtype=np.int64)
    other_count = np.zeros(pred.shape[0], dtype=np.int64)
    majority_label = np.full(pred.shape[0], -1, dtype=np.int64)
    majority_label_name = np.full(pred.shape[0], "NA", dtype=object)
    if tie_policy not in {"majority", "other_on_tie"}:
        raise ValueError(f"Unsupported tie_policy={tie_policy!r}")

    for i in range(pred.shape[0]):
        nns = nn_model.kneighbors(flat[i * pred.shape[1]:(i + 1) * pred.shape[1]], return_distance=False)
        label_counts: Counter[int] = Counter()
        for nn in nns:
            common = Counter(data.d6_labels[nn]).most_common(2)
            label = int(common[0][0])
            if tie_policy == "other_on_tie" and len(common) > 1 and common[0][1] == common[1][1]:
                label = -1
            label_counts[label] += 1
            if label == MONOCYTE_LABEL:
                mono_count[i] += 1
            elif label == NEUTROPHIL_LABEL:
                neu_count[i] += 1
            elif label == -1:
                other_count[i] += 1
        if label_counts:
            majority_label[i] = int(label_counts.most_common(1)[0][0])
            if 0 <= majority_label[i] < len(data.celltype_order):
                majority_label_name[i] = str(data.celltype_order[majority_label[i]])
            elif majority_label[i] == -1:
                majority_label_name[i] = "Other"
        scores[i] = (neu_count[i] + 1) / (neu_count[i] + mono_count[i] + 2)
        has_any[i] = (neu_count[i] + mono_count[i]) > 0

    disp_norm_mean = None
    if source is not None:
        disp_norm_mean = np.linalg.norm(pred - source[:, None, :], axis=-1).mean(axis=1)

    return FateScoreResult(
        scores=scores,
        has_any=has_any,
        neu_count=neu_count,
        mono_count=mono_count,
        majority_label=majority_label,
        majority_label_name=majority_label_name,
        other_count=other_count,
        disp_norm_mean=disp_norm_mean,
    )


def summarize_fate_scores(data: ClonalFateInputs, result: FateScoreResult) -> dict[str, float | int]:
    r_all, p_all = pearson_or_nan(data.truth, result.scores)
    r_mask, p_mask = pearson_or_nan(data.truth[result.has_any], result.scores[result.has_any])
    return {
        "n_evaluated": int(data.eval_idx.size),
        "n_with_pred": int(result.has_any.sum()),
        "pearson_r_all": r_all,
        "pearson_p_all": p_all,
        "pearson_r_masked": r_mask,
        "pearson_p_masked": p_mask,
    }


def cellwise_dataframe(data: ClonalFateInputs, result: FateScoreResult, *, method: str) -> pd.DataFrame:
    out = pd.DataFrame({
        "method": method,
        "cell_idx": data.eval_idx.astype(np.int64),
        "truth": data.truth,
        "score": result.scores,
        "has_any": result.has_any,
        "neu_count": result.neu_count,
        "mono_count": result.mono_count,
        "other_count": result.other_count,
        "majority_label": result.majority_label,
        "majority_label_name": result.majority_label_name,
        "metadata_split": data.metadata_split,
        "atlas_split": data.atlas_split,
    })
    if result.disp_norm_mean is not None:
        out["disp_norm_mean"] = result.disp_norm_mean
    return out
