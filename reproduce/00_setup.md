# 00 · Setup

Before running any reproduce script, complete these three steps.

## 1. Python environment

```bash
git clone https://github.com/mufanq/Chreode.git
cd Chreode

# Recommended: uv (or python -m venv)
uv venv && source .venv/bin/activate

# PyTorch first, matching your GPU. Example for CUDA 12.1:
uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Chreode + workflow + scVI:
uv pip install -e ".[scvi,workflow]"
```

Smoke test:

```bash
python -c "import cellworldmodel, torch; print(cellworldmodel.__version__, torch.cuda.is_available())"
# 0.1.0 True
```

If the second value is `False` your CUDA install is wrong — fix this before
continuing.

## 2. Download artifacts

```bash
# Set HF token if private mirror (not needed for public release).
# export HF_TOKEN=hf_...

python scripts/download_weights.py             # ≈ 4.2 GB  → checkpoints/pretrained/
python scripts/download_downstream_weights.py  # ≈ 230 MB → checkpoints/downstream/
python scripts/download_phase0.py              # ≈ 5.6 GB → data/phase0/
```

These call `huggingface_hub.snapshot_download` and resume on interruption.

## 3. Sanity check (1 minute, no GPU)

```bash
python -c "
import torch, cellworldmodel
from cellworldmodel.benchmark.registry import build_model
from cellworldmodel.training.checkpointing import load_shape_matched_checkpoint
print('cellworldmodel', cellworldmodel.__version__)
print('torch         ', torch.__version__, 'cuda', torch.cuda.is_available())

# Try loading the backbone with a tiny dummy config:
ckpt = torch.load('checkpoints/pretrained/dynamics_dit.pt', map_location='cpu',
                  weights_only=False)
print('backbone keys: ', list(ckpt.keys())[:5], '...')
print('OK')
"
```

If this prints `OK` without an exception, the install and the backbone
download are correct and you can move on to any of the per-experiment docs.

## Cluster (Slurm) override

The Snakemake workflow under `workflow/foundation/` reads `partition` from the
config. To target your local partition, override via the CLI:

```bash
snakemake -s workflow/foundation/Snakefile \
  --config partition=gpu \
  --use-conda --cores 32 --jobs 8 \
  -- weinreb_seedfix_all
```

Or hard-edit `config/foundation_genhui_v1.yaml` under `resources:`. The
released checkpoints used `partition: blackwell,a100`. Single-GPU is fine for
every downstream task; only pretrain Stage 2 benefits from longer wall-clock.

## Environment variables Chreode honors

| Variable | Meaning | Default |
|---|---|---|
| `CHREODE_ROOT` | Repo root path. Used by `evaluation/fate.py`. | `.` (cwd) |
| `HF_TOKEN` | HuggingFace token (only needed for non-public mirrors). | — |
| `WANDB_MODE` | Set to `offline` to disable W&B sync. | `online` |
| `WANDB_PROJECT` | W&B project name. | `chreode-reproduce` |

Set them in `.env` (gitignored) and `source .env` before running.
