"""Aggregate all 4-method × 5-seed intermediate eval results into a summary table.

Reads output/intermediate_eval/{m1,m2,m7,m8}_{mouse,veres}_seed{0..4}/results.json
and produces:
  - Per-dataset, per-method, per-timepoint mean ± std across 5 seeds
  - Side-by-side vs paper BranchSBM (Mouse Table 3 / Veres Table 2)
  - Typst tables for pasting into dashboard

Usage: PYTHONPATH=src python scripts/aggregate_intermediate_eval.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
BENCH_DIR = ROOT / "output" / "intermediate_eval"

METHODS = ["m1", "m2", "m7", "m8"]
METHOD_LABELS = {
    "m1": "Stochastic baseline",
    "m2": "Waddington baseline",
    "m7": "Stochastic + Drifting",
    "m8": "Full model",
}

# Paper reference
# Mouse Table 3 BranchSBM (5 seeds)
PAPER_MOUSE = {
    1.0: {"w1": (0.366, 0.034), "w2": (0.479, 0.044)},
    2.0: {"w1": (0.210, 0.042), "w2": (0.265, 0.046)},
}
# Veres Table 2 BranchSBM (5 seeds, full 30D W1)
PAPER_VERES_W1_FULL = {1: 11.9774, 2: 7.4643, 3: 11.5204, 4: 11.2593,
                      5: 10.2888, 6: 8.7301, 7: 6.8702}


def load_runs(method: str, dataset: str, seeds: list[int] = None) -> dict:
    """Load all seed results for a method+dataset. Return per-timepoint lists of metrics."""
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]
    per_t = defaultdict(lambda: defaultdict(list))
    for s in seeds:
        path = BENCH_DIR / f"{method}_{dataset}_seed{s}" / "results.json"
        if not path.exists():
            continue
        with open(path) as f:
            d = json.load(f)
        for key, r in d["intermediate_eval"].items():
            t = float(key.split("=")[1])
            per_t[t]["w1_top2"].append(r["branchsbm_w1_top2_mean"])
            per_t[t]["w2_top2"].append(r["branchsbm_w2_top2_mean"])
            per_t[t]["w1_full"].append(r["branchsbm_w1_full_mean"])
            per_t[t]["w2_full"].append(r["branchsbm_w2_full_mean"])
            per_t[t]["mmd"].append(r["branchsbm_mmd_full_mean"])
    return per_t


def fmt(arr):
    if not arr:
        return "—"
    a = np.asarray(arr)
    if len(a) == 1:
        return f"{a[0]:.3f}"
    return f"{a.mean():.3f}±{a.std(ddof=1):.3f}"


def emit_typst_mouse():
    """Mouse table: 4 methods × {t=1, t=2} × {W1, W2}."""
    rows = []
    for m in METHODS:
        per_t = load_runs(m, "mouse")
        t1_w1 = fmt(per_t[1.0]["w1_top2"])
        t1_w2 = fmt(per_t[1.0]["w2_top2"])
        t2_w1 = fmt(per_t[2.0]["w1_top2"])
        t2_w2 = fmt(per_t[2.0]["w2_top2"])
        n = len(per_t[2.0]["w2_top2"])
        rows.append((METHOD_LABELS[m], t1_w1, t1_w2, t2_w1, t2_w2, n))

    out = []
    out.append("#table(")
    out.append("  columns: (1.3fr, 0.8fr, 0.8fr, 0.8fr, 0.8fr),")
    out.append("  inset: 4pt, align: center, stroke: 0.4pt + gray,")
    out.append("  fill: (_, row) => if row == 0 { rgb(\"#edf2f7\") },")
    out.append("  table.header([*Method*], [*t=1 W1*], [*t=1 W2*], [*t=2 W1*], [*t=2 W2*]),")
    for label, t1w1, t1w2, t2w1, t2w2, n in rows:
        tag = f" ({n}s)" if n > 0 else ""
        out.append(f"  [{label}{tag}], [{t1w1}], [{t1w2}], [{t2w1}], [{t2w2}],")
    p = PAPER_MOUSE
    out.append(f"  [*BranchSBM paper* (5s)], "
               f"[{p[1.0]['w1'][0]:.3f}±{p[1.0]['w1'][1]:.3f}], "
               f"[{p[1.0]['w2'][0]:.3f}±{p[1.0]['w2'][1]:.3f}], "
               f"[{p[2.0]['w1'][0]:.3f}±{p[2.0]['w1'][1]:.3f}], "
               f"[{p[2.0]['w2'][0]:.3f}±{p[2.0]['w2'][1]:.3f}],")
    out.append(")")
    return "\n".join(out)


def emit_typst_veres_w1_full():
    """Veres table: 4 methods × {t=1..7} × W1_full (paper protocol)."""
    rows = []
    for m in METHODS:
        per_t = load_runs(m, "veres")
        row = [METHOD_LABELS[m]]
        for t in range(1, 8):
            row.append(fmt(per_t[float(t)]["w1_full"]))
        n = len(per_t[7.0]["w1_full"])
        rows.append((row, n))

    out = []
    out.append("#table(")
    out.append("  columns: (1.3fr, " + ", ".join(["0.65fr"] * 7) + "),")
    out.append("  inset: 3pt, align: center, stroke: 0.4pt + gray,")
    out.append("  fill: (_, row) => if row == 0 { rgb(\"#edf2f7\") },")
    out.append("  table.header([*Method*], " +
               ", ".join([f"[*t={t}*]" for t in range(1, 8)]) + "),")
    for row, n in rows:
        tag = f" ({n}s)" if n > 0 else ""
        out.append(f"  [{row[0]}{tag}], " + ", ".join(f"[{v}]" for v in row[1:]) + ",")
    paper_cells = ", ".join(f"[{PAPER_VERES_W1_FULL[t]:.2f}]" for t in range(1, 8))
    out.append(f"  [*BranchSBM paper* (5s)], {paper_cells},")
    out.append(")")
    return "\n".join(out)


if __name__ == "__main__":
    print("=== Mouse: 4 methods × 5 seeds × {t=1, t=2} (top-2 PC protocol) ===\n")
    print(emit_typst_mouse())
    print("\n\n=== Veres: 4 methods × 5 seeds × {t=1..7} (W1 full 30D, paper protocol) ===\n")
    print(emit_typst_veres_w1_full())
