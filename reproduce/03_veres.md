# 03 · Veres islet differentiation (Table 2, §5.2)

**Task.** Predict population at each of t1 … t7 from t0 source cells,
fine-tuning the pretrained Chreode backbone for 5000 epochs per seed.

**Metric.** Sinkhorn $W_2$ in the shared scVI-128 latent (ε = 0.1, 100
iterations).

## Expected numbers

Per-timepoint $W_2$ (mean over 3 seeds; bold = best per row):

| t | **Chreode** | Scratch | PI-SDE | PRESCIENT |
|---|---|---|---|---|
| t1 | **2.4009 ± 0.0658** | 2.8648 ± 0.1058 | 3.1161 ± 0.0253 | 3.2033 ± 0.0107 |
| t2 | **2.7711 ± 0.1854** | 2.8371 ± 0.1149 | 2.9183 ± 0.0132 | 3.2088 ± 0.0187 |
| t3 | **2.6403 ± 0.1141** | 2.8759 ± 0.1815 | 2.7508 ± 0.0752 | 3.3224 ± 0.1644 |
| t4 | **2.4048 ± 0.1020** | 2.4760 ± 0.0534 | 2.7555 ± 0.0422 | 3.1567 ± 0.0780 |
| t5 | **2.4269 ± 0.0205** | 2.5674 ± 0.0660 | 2.4697 ± 0.0145 | 3.0537 ± 0.0468 |
| t6 | **2.7621 ± 0.1143** | 2.9128 ± 0.0604 | 2.8481 ± 0.1363 | 3.4527 ± 0.2476 |
| t7 | **2.9132 ± 0.1704** | 3.0892 ± 0.1144 | 2.9490 ± 0.0980 | 3.8556 ± 0.4083 |
| **avg** | **2.6171** | 2.8033 | 2.8296 | 3.3219 |

Three seeds (0, 1, 2).

## Prerequisites

```bash
python scripts/download_weights.py
python scripts/download_downstream_weights.py
python scripts/download_phase0.py     # veres_ortholog.h5ad
```

## Fine-tune + evaluate (≈ 2 h per seed on 1×A100)

```bash
for seed in 0 1 2; do
  PYTHONPATH=src python -m cellworldmodel.script.run_intermediate_eval \
    --method m10 \
    --dataset veres_scvi \
    --experiment g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw \
    --init-checkpoint checkpoints/pretrained/dynamics_dit.pt \
    --epochs 5000 \
    --seed ${seed} \
    --output-dir output/reproduce/veres_seed${seed}/
done
```

## Aggregate the three seeds

```bash
python - <<'PY'
import json, glob, statistics as s
runs = [json.load(open(p)) for p in sorted(glob.glob(
    "output/reproduce/veres_seed*/results.json"))]
print(f"{'t':>4}  {'W2 mean':>9}  {'std':>6}")
ws_per_t = {}
for t in ("t=1", "t=2", "t=3", "t=4", "t=5", "t=6", "t=7"):
    ws = [r["intermediate"][t]["sinkhorn_w2_full_latent"] for r in runs]
    ws_per_t[t] = ws
    print(f"{t:>4}  {s.mean(ws):>9.4f}  {s.stdev(ws):>6.4f}")
avg = [sum(ws_per_t[t])/len(ws_per_t[t]) for t in ws_per_t]
print(f"\navg over t1..t7 mean = {sum(avg)/len(avg):.4f}")
PY
```

## Evaluate released fine-tuned heads only

```bash
for seed in 0 1 2; do
  PYTHONPATH=src python -m cellworldmodel.script.run_intermediate_eval \
    --method m10 --dataset veres_scvi \
    --experiment g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw \
    --model-config-checkpoint checkpoints/downstream/veres_seed${seed}.pt \
    --init-checkpoint         checkpoints/downstream/veres_seed${seed}.pt \
    --epochs 0 \
    --seed ${seed} \
    --output-dir output/reproduce/veres_eval_seed${seed}/
done
```

## Configuration touched

- `config/paper_bench/branchsbm_veres_scvi128.yaml`
- `config/foundation_genhui_v1.yaml`

## Source–target setup

Source population: t0 (control). Targets: independent populations at t1, …, t7.
The fine-tune uses the same multi-Δ sampling as pretrain: uniform over all
ordered $(t_i, t_j)$ pairs with $i < j$, so the same checkpoint produces
predictions at every elapsed time.
