# 02 · Weinreb hematopoiesis (Table 1, §5.1)

**Task.** Predict population at d4 and d6 from d2 source cells, fine-tuning the
pretrained Chreode backbone on each seed for 5000 epochs.

**Metric.** Sinkhorn $W_2$ in the shared scVI-128 latent (ε = 0.1, 100
iterations) against the observed d4 / d6 populations.

## Expected numbers

| Method | d4 $W_2$ ↓ | d6 $W_2$ ↓ |
|---|---|---|
| Identity replay | 2.6949 | 4.4553 |
| Linear time-delta | 2.5326 | 3.7934 |
| PRESCIENT | 1.9096 ± 0.0143 | 1.8846 ± 0.0249 |
| PI-SDE | 1.7452 ± 0.0410 | 1.8401 ± 0.0182 |
| Scratch W-DiT | 1.6387 ± 0.0524 | 2.0478 ± 0.0961 |
| **Chreode (fine-tune from Stage 2)** | **1.5133 ± 0.0757** | **1.6884 ± 0.0362** |

Three seeds (0, 1, 2).

## Prerequisites

```bash
python scripts/download_weights.py             # Stage 1 + Stage 2 backbone
python scripts/download_downstream_weights.py  # Released fine-tuned heads (optional)
python scripts/download_phase0.py              # weinreb_ortholog.h5ad + cell_index
```

## Fine-tune + evaluate (≈ 1.5 h per seed on 1×A100)

```bash
for seed in 0 1 2; do
  PYTHONPATH=src python -m cellworldmodel.script.run_intermediate_eval \
    --method m10 \
    --dataset weinreb_scvi \
    --experiment g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw \
    --init-checkpoint checkpoints/pretrained/dynamics_dit.pt \
    --epochs 5000 \
    --seed ${seed} \
    --output-dir output/reproduce/weinreb_seed${seed}/
done
```

## Aggregate the three seeds

The script writes `output/reproduce/weinreb_seed${seed}/results.json` per
seed. Aggregate with this one-liner:

```bash
python - <<'PY'
import json, glob, statistics as s
runs = [json.load(open(p)) for p in sorted(glob.glob(
    "output/reproduce/weinreb_seed*/results.json"))]
for day in ("t=2", "t=4"):  # d4 = t=2 from source d2; d6 = t=4 from source d2
    w2 = [r["intermediate"][day]["sinkhorn_w2_full_latent"] for r in runs]
    print(f"{day:6s}  W2 = {s.mean(w2):.4f} +/- {s.stdev(w2):.4f}  (n={len(w2)})")
PY
```

Compare to the table above; differences within ±0.02 are within
reduction-order noise.

## Evaluate the released fine-tuned heads only (skip retraining)

If you downloaded `checkpoints/downstream/weinreb_seed{0,1,2}.pt` and just
want to score them:

```bash
for seed in 0 1 2; do
  PYTHONPATH=src python -m cellworldmodel.script.run_intermediate_eval \
    --method m10 \
    --dataset weinreb_scvi \
    --experiment g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw \
    --model-config-checkpoint checkpoints/downstream/weinreb_seed${seed}.pt \
    --init-checkpoint         checkpoints/downstream/weinreb_seed${seed}.pt \
    --epochs 0 \
    --seed ${seed} \
    --output-dir output/reproduce/weinreb_eval_seed${seed}/
done
```

`--epochs 0` skips fine-tuning and runs eval-only on the loaded weights.

The script reuses the same fine-tune loop the paper used; configuration is
read from `config/paper_bench/branchsbm_weinreb_scvi128_compact.yaml` (only
shared-vocabulary metric settings; not BranchSBM model).

## Configuration touched

- `config/paper_bench/branchsbm_weinreb_scvi128_compact.yaml` — sets the
  Sinkhorn-evaluation protocol (ε, iterations, scVI-128 latent).
- `config/foundation_genhui_v1.yaml` `dynamics.experiment` — must match the
  pretrain experiment ID (`g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw`)
  or the released backbone won't load.

## What the script does internally

1. Loads frozen scVI-128 encoder + Chreode backbone.
2. Standardizes the latent by train-only $\mu, \sigma$ (no test leak).
3. For each (d2 → d4) and (d2 → d6) transition, calls
   `WaddingtonDiT1D(z, delta=d, action=None)` to generate K=32 samples per
   source cell.
4. Computes Sinkhorn $W_2$ against the held-out target population in the
   same latent.

Detailed Weinreb metrics (top-2 PCs, MMD, full-latent $W_1$) reported in
Table 11 of the paper appear in `output/reproduce/weinreb_eval_seed*/details.json`.
