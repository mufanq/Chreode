# 01 · Pretrain Stage 1 + Stage 2

**Paper:** §4, Appendix A.1 (Stage 1 scVI), Appendix A.2 (Stage 2 W-DiT),
Appendix B (corpus). **This step is optional** — for evaluation, download
the released backbone via [`scripts/download_weights.py`](../scripts/download_weights.py).

## What this trains

| Stage | Module | Loss | Steps | Wall-clock (1×A100) |
|---|---|---|---|---|
| 1 | scVI encoder + decoder, latent 128 | ELBO (Normal likelihood) + KL | ~1,678 (≈ 2 epochs over 2.47M cells, batch 4096) | ≈ 12 h |
| 2 | Waddington-DiT (Small: 384/12/6, 4 register tokens) | MMD + Sinkhorn $W_2$ + drift + downhill | 3,356 (batch 512) | ≈ 18 h |

## Config

`config/foundation_genhui_v1.yaml` is the single config that drives both
stages. Key entries you may want to inspect or override:

```yaml
vae:
  policy: vae_b_log1p_normal
  batch_strategy: b2_encoder_nobatch_decoder_residual
  latent_dims: [128]                  # paper uses 128
  hidden_dim: 512
  n_layers: 3
  batch_size: 4096
  max_epochs: 100                     # released ckpt stops at 2 epochs
  ...

dynamics:
  experiment: g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw
  temporal_objective: temporal_dynamics
  temporal_name: vae2_dynamicsdit2
  ...
```

The Stage 2 experiment name `g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw`
selects the W-DiT variant with **Time2Vec potential time embedding**, **bounded
low-frequency Fourier antisymmetric (curl) time embedding**, learnable
elementwise σ, and AdamW with cosine warmup — the configuration ablated as
"selected" in Table 10.

## Data

Pretrain data lives at `data/processed/genhui_all/unified_h5ad_moscot_growth_rate/`
after Phase 0 preprocessing. Pull the preprocessed catalog from HuggingFace:

```bash
python scripts/download_phase0.py   # ≈ 5.6 GB
```

This places:

- `data/phase0/cell_index.parquet` — 2.47M cell registry
- `data/phase0/split_manifest.json` — train/val/test + held-out families
- `data/phase0/orthologs/mouse_human_1to1.parquet` — 16,520 ortholog vocab
- `data/processed/genhui_all/unified_h5ad_moscot_growth_rate/*.h5ad` — 7 mouse embryo datasets

If you prefer to regenerate Phase 0 from raw datasets, see
[`scripts/run_phase0_preprocessing.py`](../src/cellworldmodel/script/run_phase0_preprocessing.py).
That entry point fetches the 7 mouse embryo datasets from their public sources
and rebuilds the catalog. **It takes ≈ 6 h and ≈ 200 GB of intermediate disk**.

## Run

```bash
# Stage 1 only
snakemake -s workflow/foundation/Snakefile \
  --config config_path=config/foundation_genhui_v1.yaml \
  --cores 16 --use-conda \
  -- vae_full

# Stage 2 only (depends on Stage 1 output)
snakemake -s workflow/foundation/Snakefile \
  --config config_path=config/foundation_genhui_v1.yaml \
  --cores 16 --use-conda \
  -- dynamics_dynamicsdit2

# Or run the whole chain:
snakemake -s workflow/foundation/Snakefile \
  --config config_path=config/foundation_genhui_v1.yaml \
  --cores 16 --use-conda \
  -- vae_full dynamics_dynamicsdit2 dynamics_staticdit2
```

The `dynamics_staticdit2` target produces the Static-DiT control arm used in
the Norman §5.4 ablation.

## Expected output

```
output/foundation/genhui_v1/
├── vae/
│   └── full_scvi1024_l128_vae2/
│       ├── epoch_2.pt           # ← Stage 1 paper checkpoint
│       └── qc.json
└── dynamics/
    ├── vae2_dynamicsdit2/
    │   ├── model.pt             # ← Stage 2 paper checkpoint (Dynamics-DiT)
    │   └── history.tsv
    └── vae2_staticdit2/
        ├── model.pt             # ← Stage 2 control (Static-DiT)
        └── history.tsv
```

## Expected metrics

From the paper (Appendix B):

| Quantity | Value |
|---|---|
| Stage 1 held-out reconstruction Pearson | **0.586** |
| Stage 1 val −ELBO | 0.774 |
| Stage 1 latent z mean / std | −0.017 / 0.84 |
| Stage 2 last-10 median MMD | 0.037 |
| Stage 2 last-10 median Sinkhorn $W_2$ | 31.5 |
| Stage 2 last-10 median drift loss | 8.68 |
| Stage 2 last-10 median downhill loss | ≈ 0 |

If you re-pretrain on a different GPU type, expect ±2% on these numbers
because of fp32 reduction order.
