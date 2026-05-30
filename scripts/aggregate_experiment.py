#!/usr/bin/env python
"""Generic experiment aggregation entrypoint.

Currently delegates to the workflow result aggregator. This wrapper gives
Snakemake and users a stable command name while we keep metric-schema details in
the highbudget_md aggregator.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.highbudget_md.aggregate_workflow_results import collect_fate, collect_training, write_outputs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--summary-json", type=Path, required=True)
    p.add_argument("--summary-tsv", type=Path, required=True)
    args = p.parse_args()

    training = collect_training(args.root)
    fate = collect_fate(args.root)
    write_outputs(args.root, training, fate, args.summary_json, args.summary_tsv)
    print(f"Wrote {args.summary_json}")
    print(f"Wrote {args.summary_tsv}")


if __name__ == "__main__":
    main()
