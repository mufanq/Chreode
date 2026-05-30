# 06 · Velocity consistency (Table 8, Appendix H)

**Task.** Measure CellStream-style velocity consistency on EMT (epithelial –
mesenchymal transition) and MOSTA datasets, using Chreode as a one-step
estimator of $\Delta z$.

**Metrics.**

- VC: cosine consistency of inferred velocities across neighbors;
- $W_2$ at endpoint: Sinkhorn $W_2$ distance between predicted and observed
  endpoint populations.

## Expected numbers (3 seeds)

| Dataset | Method | VC ↑ | endpoint $W_2$ ↓ |
|---|---|---|---|
| EMT | CellStream | 0.9743 | — |
| EMT | **Chreode** | **0.9983** | — |
| MOSTA | CellStream | 0.9638 | 0.0358 |
| MOSTA | **Chreode** | **0.9992** | **0.0355** |

## Prerequisites

```bash
python scripts/download_weights.py
python scripts/download_phase0.py   # EMT + MOSTA preprocessed h5ads
```

The Phase 0 release bundles the cellstream-style versions of EMT (`emt_cellstream.h5ad`)
and MOSTA (`mosta_cellstream.h5ad`).

## Run (≈ 20 min, 1 GPU)

```bash
export CHREODE_ROOT=$PWD

for seed in 0 1 2; do
  PYTHONPATH=src python scripts/paper_bench/run_cellstream_style_eval.py \
    --datasets emt mosta \
    --spaces native cellstream_latent \
    --experiment g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw \
    --model-config-checkpoint checkpoints/pretrained/dynamics_dit.pt \
    --cwm-inits dynamics_dit \
    --seed ${seed} \
    --output-dir output/reproduce/velocity_seed${seed}/
done

PYTHONPATH=src python scripts/paper_bench/summarize_cellstream_style_eval.py \
  --inputs "output/reproduce/velocity_seed*/" \
  --out output/reproduce/velocity_summary.json
```

## What the script does

1. Loads frozen Chreode backbone.
2. Encodes EMT (or MOSTA) cells to latent space.
3. For each cell, calls `WaddingtonDiT1D(z, delta=1)` to estimate
   $\hat z_{t+1}$; the inferred velocity is $\hat v = \hat z_{t+1} - z$.
4. Computes neighborhood cosine consistency over k=15 neighbors.
5. (MOSTA only) Computes Sinkhorn $W_2$ between predicted-endpoint
   distribution and observed-endpoint distribution.

CellStream baseline numbers in the table above come from the CellStream
paper / its released checkpoints; see `paper/chreode.pdf` Appendix H for the
license and re-implementation note.
