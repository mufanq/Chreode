# Reproducing the Chreode paper

This directory contains one markdown per experiment reported in the paper.
Read them in this order:

1. **[00_setup.md](00_setup.md)** — environment, HF downloads, sanity check.
2. **[01_pretrain.md](01_pretrain.md)** — Stage 1 VAE + Stage 2 W-DiT pretrain
   on the 2.47M-cell mouse embryonic atlas (§4 / App. A).
   *Optional* — for evaluation, just download the released backbone via
   `scripts/download_weights.py`.
3. **[02_weinreb.md](02_weinreb.md)** — Weinreb hematopoiesis fine-tune
   (Table 1, §5.1).
4. **[03_veres.md](03_veres.md)** — Veres islet differentiation fine-tune
   (Table 2, §5.2).
5. **[04_fate.md](04_fate.md)** — Weinreb clonal fate **zero-shot**
   (Table 3, §5.3).
6. **[05_norman.md](05_norman.md)** — Norman Perturb-seq via GEARS embedding
   replacement (Table 4, §5.4). **The paper numbers are 1-seed**, see
   [known_issues.md](known_issues.md).
7. **[06_velocity_consistency.md](06_velocity_consistency.md)** —
   CellStream-style velocity consistency on EMT and MOSTA
   (Table 8, App. H).
8. **[07_timing.md](07_timing.md)** — Inference latency benchmark
   (Table 7, App. G).
9. **[known_issues.md](known_issues.md)** — three protocol facts that are
   not in the paper but affect reproduction.

## Required artifacts before running anything

Every doc assumes these three downloads completed:

```bash
# Place releases under data/ and checkpoints/ in the repo root.
python scripts/download_weights.py            # ≈ 4.2 GB
python scripts/download_downstream_weights.py # ≈ 230 MB
python scripts/download_phase0.py             # ≈ 5.6 GB
```

These pull from:

- [`mufanq/chreode-pretrained`](https://huggingface.co/mufanq/chreode-pretrained)
- [`mufanq/chreode-downstream`](https://huggingface.co/mufanq/chreode-downstream)
- [`mufanq/chreode-phase0`](https://huggingface.co/datasets/mufanq/chreode-phase0)

After download the relevant paths line up with what every doc references:

```
Chreode/
├── checkpoints/
│   ├── pretrained/{vae.pt, dynamics_dit.pt, static_dit.pt}
│   └── downstream/{weinreb_seed{0,1,2}.pt, veres_seed{0,1,2}.pt}
└── data/
    └── phase0/
        ├── cell_index.parquet
        ├── split_manifest.json
        ├── orthologs/mouse_human_1to1.parquet
        ├── representations/{pca_state.pkl, z_state.zarr}
        └── downstream/
            ├── weinreb_ortholog.h5ad
            ├── veres_ortholog.h5ad
            └── weinreb_clonal.npz
```

## Expected runtimes (single A100, fp32, batch 512 train / 64 eval)

| Experiment | Released ckpt only | Full retrain |
|---|---|---|
| 01 Pretrain Stage 1 (VAE) | — | ≈ 12 h |
| 01 Pretrain Stage 2 (W-DiT) | — | ≈ 18 h |
| 02 Weinreb fine-tune (3 seeds × 5000 epochs) | ≈ 4 h | same |
| 03 Veres fine-tune (3 seeds × 5000 epochs) | ≈ 6 h | same |
| 04 Weinreb fate (zero-shot inference + 20-NN) | ≈ 10 min | — |
| 05 Norman GEARS embedding replace (1 seed, 20 epochs) | ≈ 90 min | same |
| 06 EMT + MOSTA velocity consistency (3 seeds) | ≈ 20 min | — |
| 07 Timing benchmark | ≈ 2 min | — |

## Hardware

The paper reports single-A100 runs. The released checkpoints were also trained
on A100. Other GPUs work but exact numbers may drift due to fp32 reductions
and Sinkhorn iteration order. The Snakemake workflow accepts a `partition`
override; see [00_setup.md](00_setup.md) §Cluster.
