# Known issues

Three protocol facts are not in the paper but matter for reproducing it.
This file documents them so that downstream users do not silently get
different numbers and conclude the released code is broken.

## 1. Norman §5.4 numbers are 1-seed

**What the paper reports.** Table 4 in §5.4:

| Arm | shared DE20 MSE | Δ vs official |
|---|---|---|
| GEARS official | 0.21208 | — |
| + VAE replace | 0.21262 | +0.3% |
| + Static-DiT replace | 0.19358 | −8.7% |
| + Dynamics-DiT replace | **0.18580** | **−12.4%** |

These numbers come from **a single seed** (seed = 1). The training is
deterministic given that seed.

**What 3-seed reruns show.** A rerun across seeds {0, 1, 2} gives:

| Arm | shared DE20 MSE (3-seed mean) |
|---|---|
| GEARS official | 0.2272 |
| + Static-DiT replace | 0.2099 |
| + Dynamics-DiT replace | 0.2199 |

i.e. Static-DiT becomes the best arm under 3 seeds, reversing the paper's
ranking. The reason appears to be high seed-to-seed variance in the GEARS
20-epoch optimization for this dataset, not a model issue.

**What the released pipeline does.** `scripts/reproduce_norman.sh` runs
seed = 1 only, reproducing the paper number. To run multiple seeds:

```bash
for seed in 0 1 2 3 4; do
  PYTHONPATH=src python src/cellworldmodel/script/run_norman_gears_foundation_emb.py \
    --foundation-source dynamics --foundation-mode replace --epochs 20 \
    --seed ${seed} --output-dir output/reproduce/norman/dynamics_replace_seed${seed}/
done
```

We recommend any new claim about this number bring confidence intervals.

## 2. Stage-1 VAE uses a batch-covariate fallback for unseen cells

**What the config does.** In `config/foundation_genhui_v1.yaml`:

```yaml
perturbation:
  allow_unknown_batch: true
```

This is the **B1 batch fallback**: at perturbation-eval time, Norman cells
(which were not in the pretrain atlas and therefore have no learned
batch-covariate code) are encoded with a **zero/null batch code** rather
than an exception being raised.

**Why this matters.** Strict zero-shot ought to forbid any covariate
exchange between pretrain and downstream populations. With the B1 fallback,
the decoder's batch-residual term gets a default value for Norman cells; the
encoder's batch term is ignored. Empirically the encoder is robust to the
fallback for Norman, but it does mean the paper's "strict-zero-shot" label
is more accurately "few-shot-friendly zero-shot".

**What the released pipeline does.** It honors the same flag the paper used.
To toggle it off and see the difference:

```bash
PYTHONPATH=src python src/cellworldmodel/script/run_norman_gears_foundation_emb.py \
  --foundation-source dynamics --foundation-mode replace --epochs 20 \
  --seed 1 --strict-batch  \
  --output-dir output/reproduce/norman/strict_batch/
```

`--strict-batch` raises on any unseen `batch_key` value, so any Norman cell
that lacked a leaf-dataset assignment at preprocessing time will fail loud.

## 3. GEARS on Blackwell (sm_120) needs a non-standard stack

GEARS's released wheels and most PyTorch Geometric wheels do not target
sm_120 yet (as of 2026-05). On a Blackwell GPU you will see CUDA "no kernel
image" errors during the GEARS forward.

**The combination we verified works:**

| Package | Version |
|---|---|
| PyTorch | `2.12.0.dev` (nightly with sm_120 kernels) |
| numpy | `1.26.4` (GEARS's upper bound) |
| torch-scatter / sparse / cluster / geometric | nightly matching torch 2.12 |
| `USE_FLAX` env var | `0` (disables the Flax shim some GEARS imports try to load) |

On Ampere / Hopper the regular GEARS install in [05_norman.md](05_norman.md)
works without any of this.

## 4. Drop-in datasets that did not make the paper

These have code paths in the repo but are NOT covered by the paper's claims
or the released checkpoints:

- **ZESTA control extrapolation** — early experiments showed Chreode is
  competitive but not best on this task; section deleted from the paper
  during revision. The benchmark adapter is still in
  `src/cellworldmodel/benchmark/zesta_*.py` but is not part of the
  reproduce/ flow.
- **NOTCH1 / RENGE / Aissa time-resolved perturbation** — own-head
  perturbation routes that did not pass the §5.4 selection bar.
- **Norman own-head (KTVU + NUSB-SS)** — alternative Norman protocol; not
  in the paper.
- **Hamiltonian-bio branch** — alternative residual prior; rejected by
  GPT-assisted review and replaced with the antisymmetric S formulation
  shipped here.

We left the code in to support follow-up work but did not write
`reproduce/` scripts for them.

## 5. fp32 reduction order

Chreode evaluation uses fp32 across the board (Sinkhorn iterations, MMD,
fate ranking). Different GPUs reduce in different orders, which produces
~±0.01–0.02 drift on Sinkhorn $W_2$. None of the paper's qualitative
conclusions hinge on differences smaller than that.
