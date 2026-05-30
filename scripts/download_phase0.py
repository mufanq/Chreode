#!/usr/bin/env python3
"""Download Phase-0 preprocessing artifacts from HuggingFace Datasets.

Pulls the mouse-human ortholog vocabulary, the unified cell index, the
train/val/test split manifest, the cached PCA / z_state representations,
and the downstream-task h5ad slices used by reproduce/02-06.

Usage:
    python scripts/download_phase0.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "mufanq/chreode-phase0"
DEST = Path("data/phase0")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--revision", default="main")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dest", type=Path, default=DEST)
    args = p.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset {REPO_ID} @ {args.revision} -> {args.dest}")

    snapshot_download(
        repo_id=REPO_ID,
        revision=args.revision,
        repo_type="dataset",
        local_dir=args.dest,
        force_download=args.force,
        token=os.environ.get("HF_TOKEN"),
    )

    total = 0
    for f in sorted(args.dest.rglob("*")):
        if f.is_file():
            total += f.stat().st_size
    print(f"Done. Total size: {total / (1024**3):.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
