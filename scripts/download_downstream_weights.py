#!/usr/bin/env python3
"""Download fine-tuned downstream Chreode checkpoints from HuggingFace Hub.

Pulls the 3-seed Weinreb and Veres fine-tuned heads into
``checkpoints/downstream/``.

Usage:
    python scripts/download_downstream_weights.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "WhenceFade/chreode-downstream"
DEST = Path("checkpoints/downstream")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--revision", default="main")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dest", type=Path, default=DEST)
    args = p.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID} @ {args.revision} -> {args.dest}")

    snapshot_download(
        repo_id=REPO_ID,
        revision=args.revision,
        local_dir=args.dest,
        force_download=args.force,
        token=os.environ.get("HF_TOKEN"),
    )

    for f in sorted(args.dest.rglob("*.pt")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.relative_to(args.dest)}  ({size_mb:.1f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
