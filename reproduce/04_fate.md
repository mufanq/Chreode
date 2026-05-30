# 04 · Weinreb clonal fate (Table 3, §5.3) — zero-shot

**Task.** Given a d2 source population, predict each cell's clonal
Neu / (Neu + Mono) fate ratio using the **frozen, zero-shot** Chreode
backbone. No fine-tuning.

**Metric.** Pearson r between predicted and observed fate ratios, masked
to clones whose d6 atlas neighborhood is dominated by Neu or Mono cells.
Reported as $r_\text{masked}$ at $n$ clones (the count after masking).

## Expected numbers

| Method | $r_\text{masked}$ | $n$ (clones) |
|---|---|---|
| Static-DiT (control) | 0.3825 | 335 |
| WOT | 0.4593 | 265 |
| moscot | 0.4624 | 265 |
| scDiffEq | 0.4627 | 296 |
| **Chreode (zero-shot)** | **0.4680** | **183** |

(kNN and PRESCIENT entries from earlier drafts were removed from the final
paper — see [known_issues.md](known_issues.md).)

## Prerequisites

```bash
python scripts/download_weights.py      # dynamics_dit.pt + vae.pt
python scripts/download_phase0.py       # weinreb_ortholog.h5ad + weinreb_clonal.npz
```

## Run (≈ 10 min, 1 GPU)

```bash
export CHREODE_ROOT=$PWD
PYTHONPATH=src python scripts/paper_bench/eval_foundation_fate.py \
  --static-checkpoint  checkpoints/pretrained/static_dit.pt \
  --dynamic-checkpoint checkpoints/pretrained/dynamics_dit.pt \
  --experiment g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw \
  --space scvi128 \
  --output-dir output/reproduce/fate/
```

The script writes:

- `output/reproduce/fate/static.json` — Static-DiT control row
- `output/reproduce/fate/dynamic.json` — Chreode (zero-shot) row

Read `r_masked` and `n_masked` from each. Expected values match the table
above to four decimals.

## What the script does

1. Loads frozen backbone (no fine-tune).
2. For each d2 cell, simulates `n_sim` rollouts of `WaddingtonDiT1D(delta=4)`
   to produce d6 latents.
3. Embeds those latents in the held-out d6 atlas, picks 20 nearest
   neighbors per simulated point, and assigns a fate ratio.
4. Aggregates per-clone, masks clones with ambiguous neighborhoods, computes
   masked Pearson against ground-truth Neu / (Neu + Mono).

## Why zero-shot only

The paper deliberately does NOT report fine-tuned fate results. Population
$W_2$ fine-tuning **destroys** the clone-conditioned fate ranking: predicted
cells saturate near $q \approx 1$. The fine-tuned route is shipped as the
optional flag `--allow-finetune` but is documented as inferior — see App. C
"Fate fine-tune ablation" in the paper.

## Static-DiT control row

`eval_foundation_fate.py` produces the Static-DiT control line in the same
invocation above — see `output/reproduce/fate/static.json`. Compare its
`r_masked` to the Static-DiT 0.3825 entry in the table above.
