#!/usr/bin/env python3
"""Aggregate paper-benchmark outputs into compact TSV snapshots.

This script is deliberately lightweight: it preserves each baseline family's
native metrics while putting status and file provenance in one place. The
shared Sinkhorn/MMD evaluator is still used for Chreode/simple/temporal-OT
results; BranchSBM and PRESCIENT are reported with their own reproduced
metrics until prediction export is available.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__import__("os").environ.get("CHREODE_ROOT", "."))
OUT = ROOT / "output/paper_bench/summary"


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _f(x: object, ndigits: int = 4) -> str:
    try:
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return str(x)


def aggregate_chreode_simple() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base = ROOT / "output/paper_bench/results/chreode_simple"
    for path in sorted(base.glob("*/metrics.tsv")):
        for row in _read_tsv(path):
            rows.append(
                {
                    "source": "chreode_simple",
                    "benchmark": row["benchmark"],
                    "space": row["space"],
                    "method": row["method"],
                    "target_time": row["target_time"],
                    "sinkhorn_w2_mean_down": _f(row["sinkhorn_w2_mean"]),
                    "sinkhorn_w2_std": _f(row["sinkhorn_w2_std"]),
                    "mmd_mean_down": _f(row["mmd_mean"]),
                    "mmd_std": _f(row["mmd_std"]),
                    "mean_mse_down": _f(row["mean_mse"]),
                    "mean_pearson_up": _f(row["mean_pearson"]),
                    "path": str(path.relative_to(ROOT)),
                }
            )
    return rows


def aggregate_temporal_ot() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base = ROOT / "output/paper_bench/results/temporal_ot"
    for path in sorted(base.glob("*/metrics.tsv")):
        for row in _read_tsv(path):
            rows.append(
                {
                    "source": "temporal_ot",
                    "benchmark": row["benchmark"],
                    "space": row["space"],
                    "method": row["method"],
                    "target_time": row["target_time"],
                    "sinkhorn_w2_mean_down": _f(row["sinkhorn_w2_mean"]),
                    "sinkhorn_w2_std": _f(row["sinkhorn_w2_std"]),
                    "mmd_mean_down": _f(row["mmd_mean"]),
                    "mmd_std": _f(row["mmd_std"]),
                    "mean_mse_down": _f(row["mean_mse"]),
                    "path": str(path.relative_to(ROOT)),
                }
            )
    return rows


def aggregate_cellflow() -> list[dict[str, object]]:
    path = ROOT / "output/paper_bench/results/cellflow_zesta_per_timepoint_compact_5000/results.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    rows: list[dict[str, object]] = []
    eval_payload = payload.get("eval_by_time") or payload.get("eval") or {}
    for key, value in eval_payload.items():
        model = value.get("model", value)
        rows.append(
            {
                "source": "cellflow",
                "benchmark": "zesta",
                "space": "X_aligned",
                "method": "CellFlow_per_timepoint_compact5000",
                "target_time": key.replace("t=", "").replace("t", ""),
                "r2_mean_up": _f(model.get("r2_mean") or model.get("r_squared_mean")),
                "energy_down": _f(model.get("energy") or model.get("squared_energy_distance")),
                "mmd_down": _f(model.get("mmd") or model.get("scalar_mmd")),
                "path": str(path.relative_to(ROOT)),
            }
        )
    return rows


def aggregate_zesta_temporal_ot() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((ROOT / "output/paper_bench/results").glob("zesta_temporal_ot*/results.json")):
        if "smoke" in path.parts[-2]:
            continue
        payload = json.loads(path.read_text())
        for key, value in payload.get("results", {}).items():
            model = value.get("model", value)
            rows.append(
                {
                    "source": "temporal_ot",
                    "benchmark": "zesta",
                    "space": "X_aligned",
                    "method": f"zesta_wot_{value.get('mode', 'unknown')}",
                    "target_time": key.replace("t=", "").replace("t", ""),
                    "r2_mean_up": _f(model.get("r_squared_mean")),
                    "energy_down": _f(model.get("squared_energy_distance")),
                    "mmd_down": _f(model.get("scalar_mmd")),
                    "path": str(path.relative_to(ROOT)),
                }
            )
    return rows


def aggregate_zesta_linear() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((ROOT / "output/paper_bench/results").glob("zesta_linear*/results.json")):
        payload = json.loads(path.read_text())
        for key, value in payload.get("results", {}).items():
            model = value.get("model", value)
            identity = value.get("identity")
            rows.append(
                {
                    "source": "linear_sanity",
                    "benchmark": "zesta",
                    "space": "X_aligned",
                    "method": "linear_time_delta",
                    "target_time": key.replace("t=", "").replace("t", ""),
                    "r2_mean_up": _f(model.get("r_squared_mean")),
                    "energy_down": _f(model.get("squared_energy_distance")),
                    "mmd_down": _f(model.get("scalar_mmd")),
                    "path": str(path.relative_to(ROOT)),
                }
            )
            if identity:
                rows.append(
                    {
                        "source": "linear_sanity",
                        "benchmark": "zesta",
                        "space": "X_aligned",
                        "method": "identity_source_replay",
                        "target_time": key.replace("t=", "").replace("t", ""),
                        "r2_mean_up": _f(identity.get("r_squared_mean")),
                        "energy_down": _f(identity.get("squared_energy_distance")),
                        "mmd_down": _f(identity.get("scalar_mmd")),
                        "path": str(path.relative_to(ROOT)),
                    }
                )
    return rows


def aggregate_zesta_moscot() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((ROOT / "output/paper_bench/results").glob("zesta_moscot*/results.json")):
        if "smoke" in path.parts[-2]:
            continue
        payload = json.loads(path.read_text())
        for key, value in payload.get("results", {}).items():
            model = value.get("model", value)
            rows.append(
                {
                    "source": "moscot",
                    "benchmark": "zesta",
                    "space": "X_aligned",
                    "method": value.get("mode", "moscot_temporal_ot"),
                    "target_time": key.replace("t=", "").replace("t", ""),
                    "r2_mean_up": _f(model.get("r_squared_mean")),
                    "energy_down": _f(model.get("squared_energy_distance")),
                    "mmd_down": _f(model.get("scalar_mmd")),
                    "path": str(path.relative_to(ROOT)),
                }
            )
    return rows


def aggregate_zesta_mioflow() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((ROOT / "output/paper_bench/results").glob("zesta_mioflow*/results.json")):
        if "smoke" in path.parts[-2]:
            continue
        payload = json.loads(path.read_text())
        for key, value in payload.get("results", {}).items():
            model = value.get("model", value)
            rows.append(
                {
                    "source": "mioflow",
                    "benchmark": "zesta",
                    "space": "X_aligned",
                    "method": "MIOFlow_family_neural_ode",
                    "target_time": key.replace("t=", "").replace("t", ""),
                    "r2_mean_up": _f(model.get("r_squared_mean")),
                    "energy_down": _f(model.get("squared_energy_distance")),
                    "mmd_down": _f(model.get("scalar_mmd")),
                    "path": str(path.relative_to(ROOT)),
                }
            )
    return rows


def aggregate_branchsbm() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    paths = list((ROOT / "3rdparty/BranchSBM/results").glob("05_03_*paper_veres_*_ep100/metrics.json"))
    # Early PCA30 runs were submitted before run_name was wired through the
    # wrapper, so BranchSBM wrote them as timestamped "..._None" directories.
    # Keep an explicit map rather than guessing from metrics scale.
    paths += [
        ROOT / "3rdparty/BranchSBM/results/05_03_1748_None/metrics.json",
        ROOT / "3rdparty/BranchSBM/results/05_03_1835_None/metrics.json",
    ]
    paths += list((ROOT / "3rdparty/BranchSBM/results").glob("05_04_*branchsbm_weinreb_*_compact/metrics.json"))
    for path in sorted(p for p in paths if p.exists()):
        name = path.parent.name
        if name.startswith("05_03_1748_None"):
            benchmark, space, method = "veres", "pca30", "BranchSBM"
        elif name.startswith("05_03_1835_None"):
            benchmark, space, method = "veres", "pca30", "SingleSBM"
        elif "branchsbm_weinreb" in name:
            benchmark = "weinreb"
            space = "scvi128" if "scvi128" in name else "pca50" if "pca50" in name else "unknown"
            method = "BranchSBM_compact"
        else:
            benchmark = "veres"
            space = "scvi128" if "scvi128" in name else "pca30" if "pca30" in name else "native_or_unknown"
            method = "SingleSBM" if "single" in name else "BranchSBM"
        payload = json.loads(path.read_text())
        for key, value in sorted(payload.items()):
            if not key.startswith("t") or not key.endswith("_combined"):
                continue
            rows.append(
                {
                    "source": "branchsbm_native_eval",
                    "benchmark": benchmark,
                    "space": space,
                    "method": method,
                    "target_time": key.split("_", 1)[0].replace("t", ""),
                    "w1_top_metric_down": _f(value.get("W1_mean")),
                    "w2_top_metric_down": _f(value.get("W2_mean")),
                    "mmd_top_metric_down": _f(value.get("MMD_mean")),
                    "w1_full_down": _f(value.get("W1_full_mean")),
                    "w2_full_down": _f(value.get("W2_full_mean")),
                    "mmd_full_down": _f(value.get("MMD_full_mean")),
                    "path": str(path.relative_to(ROOT)),
                }
            )
    return rows


def aggregate_branchsbm_weinreb_endpoint() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base = ROOT / "output/paper_bench/results/branchsbm_weinreb_endpoint"
    for path in sorted(base.glob("*/metrics.tsv")):
        if "smoke" in path.parts[-2]:
            continue
        for row in _read_tsv(path):
            rows.append(
                {
                    "source": "branchsbm_endpoint_eval",
                    "benchmark": row.get("benchmark", "weinreb"),
                    "space": row.get("space", "unknown"),
                    "method": row.get("method", "BranchSBM_endpoint"),
                    "target_time": row.get("target_time"),
                    "sinkhorn_w2_mean_down": _f(row.get("sinkhorn_w2_mean")),
                    "sinkhorn_w2_std": _f(row.get("sinkhorn_w2_std")),
                    "mmd_mean_down": _f(row.get("mmd_mean")),
                    "mmd_std": _f(row.get("mmd_std")),
                    "mean_mse_down": _f(row.get("mean_mse")),
                    "mean_pearson_up": _f(row.get("mean_pearson")),
                    "eval_source": row.get("eval_source"),
                    "n_source": row.get("n_source"),
                    "n_pred": row.get("n_pred"),
                    "n_target": row.get("n_target"),
                    "n_branches": row.get("n_branches"),
                    "path": str(path.relative_to(ROOT)),
                }
            )
    return rows


def aggregate_prescient() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((ROOT / "output/paper_bench/prescient").glob("**/interpolate.log")):
        parts = path.parts
        run = parts[parts.index("prescient") + 1]
        seed = path.parent.name.replace("seed_", "")
        benchmark = "weinreb" if run.startswith("weinreb") else "veres"
        space = "scvi128" if "scvi128" in run else "pca50" if "pca50" in run else "pca30" if "pca30" in run else "unknown"
        for row in _read_tsv(path):
            rows.append(
                {
                    "source": "prescient_native_eval",
                    "benchmark": benchmark,
                    "space": space,
                    "method": "PRESCIENT",
                    "seed": seed,
                    "epoch": row.get("epoch"),
                    "eval": row.get("eval"),
                    "target_time": row.get("t"),
                    "sinkhorn_loss_down": _f(row.get("loss")),
                    "path": str(path.relative_to(ROOT)),
                }
            )
    return rows


def aggregate_existing_fate() -> list[dict[str, object]]:
    path = ROOT / "output/fate_pearson_weinreb/5seed_summary.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    rows: list[dict[str, object]] = []
    prescient = payload.get("prescient", {})
    if prescient:
        rows.append(
            {
                "source": "existing_fate_repro",
                "benchmark": "fate",
                "space": "PRESCIENT_PCA50",
                "method": "PRESCIENT",
                "r_masked_up": _f(prescient.get("r_masked_5seed_mean")),
                "r_masked_std": _f(prescient.get("r_masked_5seed_std")),
                "r_all_up": _f(prescient.get("r_all_5seed_mean")),
                "r_all_std": _f(prescient.get("r_all_5seed_std")),
                "path": str(path.relative_to(ROOT)),
            }
        )
    for method, value in sorted(payload.get("rows", {}).items()):
        rows.append(
            {
                "source": "existing_fate_repro",
                "benchmark": "fate",
                "space": "legacy_scvi",
                "method": method,
                "r_masked_up": _f(value.get("r_masked_mean")),
                "r_masked_std": _f(value.get("r_masked_std")),
                "r_all_up": _f(value.get("r_all_mean")),
                "r_all_std": _f(value.get("r_all_std")),
                "path": str(path.relative_to(ROOT)),
            }
        )
    return rows


def aggregate_foundation_fate() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((ROOT / "output/paper_bench/results").glob("fate_foundation_*/metrics.tsv")):
        if "smoke" in path.parts[-2]:
            continue
        for row in _read_tsv(path):
            rows.append(
                {
                    "source": "foundation_fate",
                    "benchmark": "fate",
                    "space": row.get("space"),
                    "method": row.get("method"),
                    "r_masked_up": _f(row.get("pearson_r_masked")),
                    "r_all_up": _f(row.get("pearson_r_all")),
                    "n_evaluated": row.get("n_evaluated"),
                    "n_with_pred": row.get("n_with_pred"),
                    "n_sim": row.get("n_sim"),
                    "path": str(path.relative_to(ROOT)),
                }
            )
    return rows


def aggregate_fate_priors_ot() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((ROOT / "output/paper_bench/results").glob("fate_priors_ot_*/metrics.tsv")):
        for row in _read_tsv(path):
            rows.append(
                {
                    "source": "fate_priors_ot",
                    "benchmark": "fate",
                    "space": row.get("space"),
                    "method": row.get("method"),
                    "r_masked_up": _f(row.get("pearson_r_masked")),
                    "r_all_up": _f(row.get("pearson_r_all")),
                    "mse_down": _f(row.get("mse")),
                    "n_evaluated": row.get("n_evaluated"),
                    "n_with_pred": row.get("n_with_pred"),
                    "path": str(path.relative_to(ROOT)),
                }
            )
    return rows


def aggregate_fate_moscot() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((ROOT / "output/paper_bench/results").glob("fate_moscot_*/metrics.tsv")):
        if "smoke" in path.parts[-2]:
            continue
        for row in _read_tsv(path):
            rows.append(
                {
                    "source": "moscot",
                    "benchmark": "fate",
                    "space": row.get("space"),
                    "method": row.get("method"),
                    "r_masked_up": _f(row.get("pearson_r_masked")),
                    "r_all_up": _f(row.get("pearson_r_all")),
                    "mse_down": _f(row.get("mse")),
                    "n_evaluated": row.get("n_evaluated"),
                    "n_with_pred": row.get("n_with_pred"),
                    "path": str(path.relative_to(ROOT)),
                }
            )
    return rows


def aggregate_fate_scdiffeq() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((ROOT / "output/paper_bench/results").glob("fate_scdiffeq_*/metrics.tsv")):
        if "smoke" in path.parts[-2]:
            continue
        for row in _read_tsv(path):
            rows.append(
                {
                    "source": "scdiffeq",
                    "benchmark": "fate",
                    "space": row.get("space"),
                    "method": row.get("method"),
                    "r_masked_up": _f(row.get("pearson_r_masked")),
                    "r_all_up": _f(row.get("pearson_r_all")),
                    "n_evaluated": row.get("n_evaluated"),
                    "n_with_pred": row.get("n_with_pred"),
                    "n_sim": row.get("n_sim"),
                    "path": str(path.relative_to(ROOT)),
                }
            )
    return rows


def aggregate_pisde() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((ROOT / "output/paper_bench/results").glob("pisde_*/*/*/*/interpolate.log")):
        if "smoke" in path.parts[-5]:
            continue
        run_name = path.parts[-5]
        pieces = run_name.split("_")
        if len(pieces) < 3:
            continue
        benchmark = pieces[1]
        space = pieces[2]
        run_tag = "_".join(pieces[3:]) if len(pieces) > 3 else "first_pass"
        for row in _read_tsv(path):
            rows.append(
                {
                    "source": "pisde",
                    "benchmark": benchmark,
                    "space": space,
                    "method": "PI-SDE",
                    "run_tag": run_tag,
                    "epoch": row.get("epoch"),
                    "eval": row.get("eval"),
                    "target_time": row.get("t"),
                    "sinkhorn_loss_down": _f(row.get("loss")),
                    "path": str(path.relative_to(ROOT)),
                }
            )
    return rows


def make_status(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    counts: dict[tuple[str, str, str, str], list[float]] = {}
    for row in rows:
        if row.get("source") == "prescient_native_eval" and str(row.get("epoch")) != "best":
            continue
        key = (
            str(row.get("source", "")),
            str(row.get("benchmark", "")),
            str(row.get("space", "")),
            str(row.get("method", "")),
        )
        value = None
        for metric in (
            "sinkhorn_w2_mean_down",
            "sinkhorn_loss_down",
            "w2_full_down",
            "r2_mean_up",
            "r_masked_up",
        ):
            if metric in row and row[metric] not in ("", "None"):
                try:
                    value = float(row[metric])
                    break
                except Exception:
                    pass
        counts.setdefault(key, [])
        if value is not None:
            counts[key].append(value)
    return [
        {
            "source": key[0],
            "benchmark": key[1],
            "space": key[2],
            "method": key[3],
            "n_rows": len(values),
            "primary_metric_mean_snapshot": _f(mean(values)) if values else "",
        }
        for key, values in sorted(counts.items())
    ]


def main() -> None:
    rows = (
        aggregate_chreode_simple()
        + aggregate_temporal_ot()
        + aggregate_cellflow()
        + aggregate_zesta_temporal_ot()
        + aggregate_zesta_linear()
        + aggregate_zesta_moscot()
        + aggregate_zesta_mioflow()
        + aggregate_branchsbm()
        + aggregate_branchsbm_weinreb_endpoint()
        + aggregate_prescient()
        + aggregate_existing_fate()
        + aggregate_foundation_fate()
        + aggregate_fate_priors_ot()
        + aggregate_fate_moscot()
        + aggregate_fate_scdiffeq()
        + aggregate_pisde()
    )
    _write_tsv(OUT / "current_metrics.tsv", rows)
    _write_tsv(OUT / "current_status.tsv", make_status(rows))
    print(f"Wrote {OUT / 'current_metrics.tsv'} ({len(rows)} rows)")
    print(f"Wrote {OUT / 'current_status.tsv'}")


if __name__ == "__main__":
    main()
