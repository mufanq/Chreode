#!/usr/bin/env python3
"""Download pretrained Chreode weights from HuggingFace Hub.

Pulls the Stage 1 scVI encoder, the Stage 2 Waddington-DiT (Dynamics) backbone,
and the Stage 2 Static-DiT control arm into ``checkpoints/pretrained/``.

Usage:
    python scripts/download_weights.py
    python scripts/download_weights.py --revision main --force

Environment:
    HF_TOKEN    Optional HuggingFace access token. Not needed for the public
                release; only required if you mirror to a private repo.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "WhenceFade/chreode-pretrained"
DEST = Path("checkpoints/pretrained")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--revision", default="main",
                   help="HF Hub revision / branch / tag to pull (default: main).")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if files already exist.")
    p.add_argument("--dest", type=Path, default=DEST,
                   help=f"Destination directory (default: {DEST}).")
    args = p.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID} @ {args.revision} -> {args.dest}")

    path = snapshot_download(
        repo_id=REPO_ID,
        revision=args.revision,
        local_dir=args.dest,
        force_download=args.force,
        token=os.environ.get("HF_TOKEN"),
    )

    print(f"Done. Files now under: {path}")
    for f in sorted(args.dest.rglob("*.pt")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.relative_to(args.dest)}  ({size_mb:.1f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
