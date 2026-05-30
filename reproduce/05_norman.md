# 05 · Norman Perturb-seq via GEARS embedding replacement (Table 4, §5.4)

**Task.** Replace the gene-state hidden embedding inside the official GEARS
perturbation predictor with the Chreode foundation embedding of the same
Norman cells, then train GEARS as normal for 20 epochs.

**Metric.** Shared-vocabulary DE20 MSE / Pearson r / Δr on the Norman
Perturb-seq test split.

> **Important:** The paper numbers are **1-seed (seed = 1)**. A 3-seed
> rerun does not preserve the ranking. See [known_issues.md](known_issues.md)
> §1. The released pipeline keeps the 1-seed protocol so this doc reproduces
> the paper number; for confidence intervals you should run additional seeds
> yourself.

## Expected numbers (1 seed, 20 epochs)

| Arm | shared DE20 MSE ↓ | shared DE20 r ↑ | shared DE20 Δr ↑ |
|---|---|---|---|
| GEARS official | 0.21208 | 0.79911 | 0.75931 |
| + VAE replace | 0.21262 | 0.79231 | 0.75556 |
| + Static-DiT replace | 0.19358 | 0.80789 | 0.77007 |
| **+ Dynamics-DiT replace** | **0.18580** | **0.81288** | **0.77135** |

(−12.4% MSE relative to GEARS official.)

## Prerequisites

```bash
python scripts/download_weights.py     # vae.pt, static_dit.pt, dynamics_dit.pt
python scripts/download_phase0.py      # ortholog vocab
```

You also need the **GEARS** library and the official Norman dataset. GEARS is
sensitive to its PyTorch / numpy / PyG stack, so we keep it in a separate
venv and call it through the helper script.

## GEARS environment (separate venv)

On most GPUs (Ampere, Hopper) the standard GEARS install works:

```bash
uv venv .venv-gears && source .venv-gears/bin/activate
uv pip install \
    torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
uv pip install torch-scatter torch-sparse torch-cluster torch-geometric \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
uv pip install cell-gears scanpy anndata pandas numpy==1.26.4
```

**On Blackwell (sm_120) you instead need PyTorch 2.12-dev + numpy 1.26.4 +
`USE_FLAX=0`.** This is one of the operational facts not in the paper; see
[known_issues.md](known_issues.md) §3.

GEARS will auto-download the Norman dataset on first run to
`data/gears_norman/`.

## Run (1 seed, ≈ 90 min on 1×A100)

The wrapper precomputes the foundation embedding for every Norman cell,
monkey-patches the GEARS model class to use it, and then runs the GEARS
training/eval loop unchanged. From the **`.venv-gears`** environment:

```bash
PYTHONPATH=src python scripts/gears/run_norman_gears_foundation_emb.py \
  --foundation-source vae       --foundation-mode replace --epochs 20 \
  --seed 1 --output-dir output/reproduce/norman/vae_replace/

PYTHONPATH=src python scripts/gears/run_norman_gears_foundation_emb.py \
  --foundation-source static    --foundation-mode replace --epochs 20 \
  --seed 1 --output-dir output/reproduce/norman/static_replace/

PYTHONPATH=src python scripts/gears/run_norman_gears_foundation_emb.py \
  --foundation-source dynamics  --foundation-mode replace --epochs 20 \
  --seed 1 --output-dir output/reproduce/norman/dynamics_replace/
```

The GEARS-official baseline (no replacement) is:

```bash
PYTHONPATH=src python scripts/gears/run_norman_gears_foundation_emb.py \
  --foundation-source none --epochs 20 --seed 1 \
  --output-dir output/reproduce/norman/gears_official/
```

## Aggregate

Each arm writes `output/reproduce/norman/<arm>/shared_vocab_summary.json`.
Read the four files and compare to the table above:

```bash
python - <<'PY'
import json, glob
for path in sorted(glob.glob("output/reproduce/norman/*/shared_vocab_summary.json")):
    arm = path.split("/")[-2]
    d = json.load(open(path))["overall"]
    print(f"{arm:20s}  DE20 MSE = {d['shared_de_mse']:.5f}  "
          f"r = {d['shared_de_pearson']:.5f}  "
          f"Δr = {d['shared_de_delta_pearson']:.5f}")
PY
```

## What the foundation embedding actually is

For an arm, the per-cell hidden embedding fed into GEARS in place of its
default gene-state vector is:

| Arm | Formula |
|---|---|
| `vae` | $z_c = \mathcal{E}_\text{VAE}(x_c)$ |
| `static` | $F_\text{static-DiT}(z_c, \Delta = 1)$ |
| `dynamics` | $F_\text{dynamics-DiT}(z_c, \Delta = 1)$ |

Same Stage-1 encoder for every arm; Static-DiT is the reconstruction-only
control Stage 2, and Dynamics-DiT is the paper's main backbone.

## Why a separate GEARS venv

The GEARS upstream pin is `numpy<1.27`, but our main Chreode env tracks the
latest scanpy which transitively wants `numpy>=1.26`. We keep GEARS in its
own venv so the official 0.0.6 release loads without modification.
