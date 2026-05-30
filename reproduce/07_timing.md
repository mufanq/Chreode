# 07 · Inference timing (Table 7, Appendix G)

**Task.** Measure single-cell forward-pass latency on A100, batch 1, fp32,
and convert to NFE (number of function evaluations per query).

## Expected numbers

| Method | Median latency / query | NFE per query | GFLOPs |
|---|---|---|---|
| PRESCIENT | 194.33 ms | many (SDE solver) | 0.07 |
| CellFlow | 410.36 ms | many (ODE solver) | — |
| **Chreode** | **64.79 ms** | **1** (incl. $-\nabla U$ autograd) | **0.90** |

## Prerequisites

```bash
python scripts/download_weights.py
```

CPU-only also works; the relative comparison is what matters.

## Run (≈ 2 min)

```bash
PYTHONPATH=src python scripts/timing/run_available_timing_benchmark.py \
  --device cuda:0 \
  --n 100 --warmup 10 \
  --output output/reproduce/timing.json
```

This warms up for 10 forward passes, then times 100 consecutive forward
passes (batch 1, fp32) including the autograd backward for $-\nabla_z U_\theta$.
The "1 NFE" claim follows because $\hat z_{t+\Delta}$ is produced by a single
`R_θ` call, not by a multi-step ODE/SDE integrator. Read the median from
`output/reproduce/timing.json`.

## PRESCIENT / CellFlow rows

The PRESCIENT and CellFlow rows in Table 7 come from running each baseline's
public reproduction script on the same A100 with batch 1; we cite each
baseline's repo in `paper/chreode.pdf` Appendix G. We do not ship those
baseline timing scripts in this repo to avoid bundling other projects'
code — please consult the respective upstreams if you want to re-time them.
