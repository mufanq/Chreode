#!/usr/bin/env python
"""Aggregate CellStream-style downstream runs across seeds."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEYS = ["dataset", "space", "method"]
METRICS = [
    "metric_tc_radius005",
    "metric_vc_radius005",
    "metric_tc_knn20",
    "metric_vc_knn20",
    "velocity_accuracy",
]


def _mean_pm(s: pd.Series, digits: int = 4) -> str:
    vals = pd.to_numeric(s, errors="coerce").dropna()
    if vals.empty:
        return "NA"
    if vals.shape[0] == 1:
        return f"{vals.iloc[0]:.{digits}f}"
    return f"{vals.mean():.{digits}f}±{vals.std(ddof=1):.{digits}f}"


def _collect(root: Path, name: str) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("seed*/*/" + name)):
        df = pd.read_csv(path, sep="\t")
        seed_txt = path.parent.parent.name.replace("seed", "")
        df["seed"] = int(seed_txt)
        df["path"] = str(path)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    for key, group in df.groupby(KEYS, sort=True):
        row = dict(zip(KEYS, key))
        row["n"] = int(group["seed"].nunique())
        for metric in METRICS:
            if metric in group:
                row[metric] = _mean_pm(group[metric])
                row[f"{metric}_mean"] = pd.to_numeric(group[metric], errors="coerce").mean()
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_endpoint(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    per_run = (
        df.groupby(KEYS + ["seed"], sort=True)
        .agg(avg_sinkhorn_w2=("sinkhorn_w2", "mean"), avg_mmd2=("mmd2", "mean"), n_targets=("target_time", "count"))
        .reset_index()
    )
    rows = []
    for key, group in per_run.groupby(KEYS, sort=True):
        row = dict(zip(KEYS, key))
        row["n"] = int(group["seed"].nunique())
        row["avg_sinkhorn_w2"] = _mean_pm(group["avg_sinkhorn_w2"])
        row["avg_sinkhorn_w2_mean"] = float(group["avg_sinkhorn_w2"].mean())
        row["avg_mmd2"] = _mean_pm(group["avg_mmd2"])
        row["avg_mmd2_mean"] = float(group["avg_mmd2"].mean())
        row["n_targets"] = int(group["n_targets"].max())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("output/paper_bench/cellstream_style_formal_20260506"))
    args = parser.parse_args()
    metrics = _collect(args.root, "metrics.tsv")
    endpoints = _collect(args.root, "endpoint_metrics.tsv")
    metric_summary = summarize_metrics(metrics)
    endpoint_summary = summarize_endpoint(endpoints)
    args.root.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.root / "metrics_all.tsv", sep="\t", index=False)
    endpoints.to_csv(args.root / "endpoint_metrics_all.tsv", sep="\t", index=False)
    metric_summary.to_csv(args.root / "metrics_summary.tsv", sep="\t", index=False)
    endpoint_summary.to_csv(args.root / "endpoint_summary.tsv", sep="\t", index=False)
    print("Metrics summary")
    print(metric_summary.to_string(index=False) if not metric_summary.empty else "EMPTY")
    print("\nEndpoint summary")
    print(endpoint_summary.to_string(index=False) if not endpoint_summary.empty else "EMPTY")


if __name__ == "__main__":
    main()
