"""Dataset adapter for time-resolved perturbation prediction benchmarks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellworldmodel.foundation.gene_space import foundation_gene_names_from_vocab, gene_key


def _row_normalize_log1p(x: sp.spmatrix | np.ndarray, target_sum: float = 1e4) -> sp.csr_matrix:
    x_csr = x.tocsr() if sp.issparse(x) else sp.csr_matrix(np.asarray(x))
    sums = np.asarray(x_csr.sum(axis=1)).ravel()
    scale = np.divide(float(target_sum), sums, out=np.zeros_like(sums, dtype=np.float64), where=sums > 0)
    x_csr = x_csr.multiply(scale[:, None]).tocsr()
    x_csr.data = np.log1p(x_csr.data)
    return x_csr.astype(np.float32)


def _map_to_vocab(
    adata: ad.AnnData,
    vocab_genes: list[str],
    *,
    input_log1p: bool,
) -> tuple[sp.csr_matrix, np.ndarray]:
    source_genes = [str(x) for x in adata.var_names]
    source_map = {gene_key(gene): i for i, gene in enumerate(source_genes)}
    src_idx: list[int] = []
    dst_idx: list[int] = []
    for dst, gene in enumerate(vocab_genes):
        src = source_map.get(gene_key(gene))
        if src is not None:
            src_idx.append(src)
            dst_idx.append(dst)
    if input_log1p:
        x = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(np.asarray(adata.X, dtype=np.float32))
    else:
        x = _row_normalize_log1p(adata.X)
    if not src_idx:
        return sp.csr_matrix((adata.n_obs, len(vocab_genes)), dtype=np.float32), np.asarray([], dtype=np.int64)
    src_arr = np.asarray(src_idx, dtype=np.int64)
    dst_arr = np.asarray(dst_idx, dtype=np.int64)
    x_sel = x[:, src_arr].tocoo()
    out = sp.csr_matrix(
        (x_sel.data.astype(np.float32, copy=False), (x_sel.row, dst_arr[x_sel.col])),
        shape=(adata.n_obs, len(vocab_genes)),
        dtype=np.float32,
    )
    return out, dst_arr


@dataclass(frozen=True)
class TimeResolvedPerturbationDataOptions:
    adata: str | Path
    gene_vocab: str | Path
    condition_col: str = "condition_label"
    time_col: str = "time"
    control_label: str = "control"
    perturbation_label: str = "NOTCH1_KO"
    source_time: float = 0.0
    source_policy: str = "fixed"
    train_times: tuple[float, ...] = (2.0, 5.0, 10.0)
    eval_times: tuple[float, ...] = (14.0, 30.0)
    action_genes: tuple[str, ...] = ("NOTCH1",)
    action_sign: float = -1.0
    action_modality_id: int = 1
    max_action_genes: int = 4
    input_log1p: bool = False
    top_k: int = 100


class TimeResolvedPerturbationDataset:
    """Control-at-source-time to perturbed-future-time population dataset."""

    def __init__(self, options: TimeResolvedPerturbationDataOptions) -> None:
        self.options = options
        self.adata = ad.read_h5ad(str(options.adata))
        self.vocab_genes = foundation_gene_names_from_vocab(options.gene_vocab)
        self.x_vocab, self.mapped_vocab_idx = _map_to_vocab(
            self.adata,
            self.vocab_genes,
            input_log1p=bool(options.input_log1p),
        )
        self.obs = self.adata.obs.copy()
        self.condition = self.obs[options.condition_col].astype(str).to_numpy()
        self.time = self.obs[options.time_col].astype(float).to_numpy()
        self.control_idx = self._select(options.control_label, float(options.source_time))
        if self.control_idx.size == 0:
            raise ValueError("No source control cells found for requested source_time")
        self.train_times = tuple(float(t) for t in options.train_times)
        self.eval_times = tuple(float(t) for t in options.eval_times)
        if options.source_policy not in {"fixed", "matched_time"}:
            raise ValueError("source_policy must be one of {'fixed', 'matched_time'}")
        self.target_idx_by_time = {
            float(t): self._select(options.perturbation_label, float(t))
            for t in self.train_times + self.eval_times
        }
        missing = [t for t, idx in self.target_idx_by_time.items() if idx.size == 0]
        if missing:
            raise ValueError(f"No target cells found for times={missing}")
        self.control_idx_by_time = {
            float(t): self._select(options.control_label, float(t))
            for t in self.train_times + self.eval_times
        }
        if options.source_policy == "matched_time":
            missing_source = [t for t, idx in self.control_idx_by_time.items() if idx.size == 0]
            if missing_source:
                raise ValueError(f"No matched-time source control cells found for times={missing_source}")
        self.control_mean = self.mean_for_indices(self.control_idx)
        self.control_mean_by_time = {
            time: self.mean_for_indices(idx)
            for time, idx in self.control_idx_by_time.items()
            if idx.size > 0
        }
        self.target_mean_by_time = {
            time: self.mean_for_indices(idx)
            for time, idx in self.target_idx_by_time.items()
        }
        self.de_idx_by_time = {
            time: self._top_de_idx(self.target_mean_by_time[time], self.source_mean_for_time(time), int(options.top_k))
            for time in self.target_idx_by_time
        }
        self.gene_to_id = self._build_gene_to_id()

    @property
    def n_genes(self) -> int:
        return len(self.vocab_genes)

    def _select(self, label: str, time: float) -> np.ndarray:
        mask = (self.condition == str(label)) & np.isclose(self.time.astype(float), float(time))
        return np.flatnonzero(mask)

    def _build_gene_to_id(self) -> dict[str, int]:
        gene_vocab = pd.read_parquet(self.options.gene_vocab)
        out: dict[str, int] = {}
        for idx, row in gene_vocab.reset_index(drop=True).iterrows():
            for key in ("human_symbol", "canonical_gene", "mouse_symbol"):
                value = str(row.get(key, "")).upper()
                if value and value != "NAN":
                    out.setdefault(value, int(idx))
        return out

    def _top_de_idx(self, target_mean: np.ndarray, source_mean: np.ndarray, top_k: int) -> np.ndarray:
        delta = np.abs(target_mean - source_mean)
        top_k = min(int(top_k), int(delta.size))
        return np.argsort(delta)[-top_k:].astype(np.int64)

    def source_indices_for_time(self, time: float) -> np.ndarray:
        if self.options.source_policy == "matched_time":
            return self.control_idx_by_time[float(time)]
        return self.control_idx

    def source_mean_for_time(self, time: float) -> np.ndarray:
        if self.options.source_policy == "matched_time":
            return self.control_mean_by_time[float(time)]
        return self.control_mean

    def mean_for_indices(self, idx: np.ndarray) -> np.ndarray:
        return np.asarray(self.x_vocab[idx].mean(axis=0)).ravel().astype(np.float32)

    def load_rows(self, rows: np.ndarray) -> np.ndarray:
        return self.x_vocab[np.asarray(rows, dtype=np.int64)].toarray().astype(np.float32, copy=False)

    def sample_set_pair(
        self,
        time: float,
        set_size: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        target_idx = self.target_idx_by_time[float(time)]
        source_idx = self.source_indices_for_time(float(time))
        control_rows = rng.choice(source_idx, size=int(set_size), replace=int(set_size) > len(source_idx))
        target_rows = rng.choice(target_idx, size=int(set_size), replace=int(set_size) > len(target_idx))
        return self.load_rows(control_rows), self.load_rows(target_rows)

    def action_gene_arrays(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        max_genes = int(self.options.max_action_genes)
        gene_ids = np.full(max_genes, self.n_genes, dtype=np.int64)
        signs = np.zeros(max_genes, dtype=np.float32)
        modality_ids = np.zeros(max_genes, dtype=np.int64)
        strengths = np.ones(max_genes, dtype=np.float32)
        mask = np.zeros(max_genes, dtype=bool)
        for i, gene in enumerate(self.options.action_genes[:max_genes]):
            gene_ids[i] = self.gene_to_id.get(str(gene).upper(), self.n_genes)
            signs[i] = float(self.options.action_sign)
            modality_ids[i] = int(self.options.action_modality_id)
            strengths[i] = 1.0
            mask[i] = True
        return (
            np.repeat(gene_ids[None, :], int(n), axis=0),
            np.repeat(signs[None, :], int(n), axis=0),
            np.repeat(modality_ids[None, :], int(n), axis=0),
            np.repeat(strengths[None, :], int(n), axis=0),
            np.repeat(mask[None, :], int(n), axis=0),
        )
