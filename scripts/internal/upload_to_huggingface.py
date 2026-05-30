#!/usr/bin/env python3
"""Upload Chreode artifacts (pretrained / downstream / phase0) to HuggingFace.

Intended for the project owner; not part of the user-facing reproduce flow.
The local source paths are derived from the active CellWorldModel checkout,
not the released Chreode repo, because the released repo intentionally does
not ship checkpoints.

Run from a workstation that has these paths and a valid HF token in
``~/.cache/huggingface/token`` or ``HF_TOKEN``:

    python scripts/internal/upload_to_huggingface.py --what pretrained
    python scripts/internal/upload_to_huggingface.py --what downstream
    python scripts/internal/upload_to_huggingface.py --what phase0
    python scripts/internal/upload_to_huggingface.py --what all

Re-running is idempotent — HuggingFace ``upload_folder`` overwrites identical
files without re-uploading bytes when the hashes match.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from shutil import copy2
import tempfile

from huggingface_hub import HfApi, create_repo

REPO_PRETRAINED = "WhenceFade/chreode-pretrained"
REPO_DOWNSTREAM = "WhenceFade/chreode-downstream"
REPO_PHASE0     = "WhenceFade/chreode-phase0"

CWM_ROOT = Path(
    os.environ.get(
        "CWM_SOURCE_ROOT",
        "/playpen-shared/mufan/github/CellWorldModel",
    )
)


def _ensure_repo(api: HfApi, repo_id: str, repo_type: str) -> None:
    try:
        create_repo(repo_id, repo_type=repo_type, exist_ok=True, private=False)
        print(f"  repo ready: {repo_id} (type={repo_type})")
    except Exception as exc:
        print(f"  repo create failed: {exc}")
        raise


def stage_pretrained(staging: Path) -> None:
    """Copy Stage 1 VAE + Stage 2 Dynamics-DiT + Stage 2 Static-DiT into staging.

    Paper paths (see reproduce/known_issues.md):
      Stage 1 VAE:           output/foundation/genhui_v1/vae/full_scvi1024_l128_vae2/epoch_2.pt
      Stage 2 Dynamics-DiT:  output/foundation/genhui_v1/dynamics/vae2_dynamicsdit2/model.pt
      Stage 2 Static-DiT:    output/foundation/genhui_v1/dynamics/vae2_staticdit2/model.pt
    """
    src = [
        ("vae.pt",          CWM_ROOT / "output/foundation/genhui_v1/vae/full_scvi1024_l128_vae2/epoch_2.pt"),
        ("dynamics_dit.pt", CWM_ROOT / "output/foundation/genhui_v1/dynamics/vae2_dynamicsdit2/model.pt"),
        ("static_dit.pt",   CWM_ROOT / "output/foundation/genhui_v1/dynamics/vae2_staticdit2/model.pt"),
    ]
    for dst_name, src_path in src:
        if not src_path.exists():
            raise FileNotFoundError(f"Source weight missing: {src_path}")
        dst = staging / dst_name
        print(f"  copy {src_path}  ->  {dst.name}")
        copy2(src_path, dst)

    (staging / "README.md").write_text(_pretrained_readme())


def stage_downstream(staging: Path) -> None:
    src_root = CWM_ROOT / "output/workflow_wdit_loss_balancer_seedfix_validation_20260429"
    pairs = []
    for task, dataset_tag in (("weinreb", "weinreb_scvi"),
                              ("veres",   "veres_scvi")):
        for seed in (0, 1, 2):
            sub = (src_root / "intermediate" / dataset_tag
                   / f"g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw_{dataset_tag}_seed{seed}")
            # paper run wrote `checkpoint_final.pt`
            for name in ("checkpoint_final.pt", "model.pt"):
                if (sub / name).exists():
                    pairs.append((f"{task}_seed{seed}.pt", sub / name))
                    break
            else:
                cands = list(sub.glob("*.pt"))
                if not cands:
                    raise FileNotFoundError(f"No checkpoint under {sub}")
                pairs.append((f"{task}_seed{seed}.pt", cands[0]))
    for dst_name, src_path in pairs:
        dst = staging / dst_name
        print(f"  copy {src_path.name}  ->  {dst.name}")
        copy2(src_path, dst)

    (staging / "README.md").write_text(_downstream_readme())


def stage_phase0(staging: Path) -> None:
    """Copy the bits of output/phase0/ that the reproduce flow actually reads."""
    src_phase0 = CWM_ROOT / "output/phase0"
    if not src_phase0.exists():
        raise FileNotFoundError(f"Phase 0 root missing: {src_phase0}")
    items = [
        "cell_index.parquet",
        "split_manifest.json",
        "orthologs/mouse_human_1to1.parquet",
        "weinreb_ortholog.h5ad",
        "veres_ortholog.h5ad",
        "weinreb_clonal.npz",
        "representations/pca_state.pkl",
    ]
    for rel in items:
        src = src_phase0 / rel
        if not src.exists():
            print(f"  WARNING: phase0 item missing, skipping: {rel}")
            continue
        dst = staging / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"  copy {rel}")
        copy2(src, dst)

    (staging / "README.md").write_text(_phase0_readme())


def _pretrained_readme() -> str:
    return (
        "---\n"
        "license: mit\n"
        "tags:\n"
        "- single-cell\n"
        "- foundation-model\n"
        "- waddington\n"
        "library_name: pytorch\n"
        "---\n\n"
        "# Chreode pretrained backbone\n\n"
        "Files:\n\n"
        "- `vae.pt` — Stage 1 scVI encoder (latent 128, 2 epochs, batch 4096).\n"
        "- `dynamics_dit.pt` — Stage 2 Waddington-DiT (main paper backbone, "
        "experiment `g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw`).\n"
        "- `static_dit.pt` — Stage 2 reconstruction-only control (used for §5.3 "
        "and §5.4 control arms).\n\n"
        "See https://github.com/mufanq/Chreode for the loading code and the paper.\n"
    )


def _downstream_readme() -> str:
    return (
        "---\n"
        "license: mit\n"
        "tags:\n"
        "- single-cell\n"
        "- foundation-model\n"
        "library_name: pytorch\n"
        "---\n\n"
        "# Chreode downstream fine-tuned heads\n\n"
        "Three seeds each for Weinreb and Veres, fine-tuned 5000 epochs on top "
        "of the released Chreode backbone (`WhenceFade/chreode-pretrained`).\n\n"
        "See https://github.com/mufanq/Chreode reproduce/02_weinreb.md and "
        "reproduce/03_veres.md for usage.\n"
    )


def _phase0_readme() -> str:
    return (
        "---\n"
        "license: mit\n"
        "tags:\n"
        "- single-cell\n"
        "task_categories:\n"
        "- other\n"
        "---\n\n"
        "# Chreode Phase-0 preprocessing\n\n"
        "Cached preprocessing artifacts used by the Chreode reproduce flow:\n\n"
        "- `orthologs/mouse_human_1to1.parquet` — 16,520-gene mouse-human ortholog "
        "vocabulary (Ensembl BioMart, confidence=1).\n"
        "- `cell_index.parquet` — 2.47M-row registry over the 7-dataset pretrain corpus.\n"
        "- `split_manifest.json` — train/val/test + held-out family + external dataset.\n"
        "- `representations/pca_state.pkl` — normalized PCA basis for downstream baselines.\n"
        "- `weinreb_ortholog.h5ad`, `veres_ortholog.h5ad`, `weinreb_clonal.npz` — downstream slices.\n\n"
        "See https://github.com/mufanq/Chreode reproduce/00_setup.md.\n"
    )


def upload(staging: Path, repo_id: str, repo_type: str) -> None:
    api = HfApi()
    _ensure_repo(api, repo_id, repo_type)
    print(f"  uploading {staging} -> {repo_id}")
    api.upload_folder(
        folder_path=str(staging),
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message="Initial release",
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--what", choices=["pretrained", "downstream", "phase0", "all"],
                   required=True)
    p.add_argument("--dry-run", action="store_true",
                   help="Stage files and print what would happen, but do not upload.")
    args = p.parse_args()

    work = [
        ("pretrained", REPO_PRETRAINED, "model",   stage_pretrained),
        ("downstream", REPO_DOWNSTREAM, "model",   stage_downstream),
        ("phase0",     REPO_PHASE0,     "dataset", stage_phase0),
    ]
    selected = [w for w in work if args.what in (w[0], "all")]

    for name, repo_id, repo_type, stage_fn in selected:
        print(f"\n=== {name} ===")
        with tempfile.TemporaryDirectory(prefix=f"chreode-stage-{name}-") as tmp:
            staging = Path(tmp)
            stage_fn(staging)
            if args.dry_run:
                print("  --dry-run set, skipping upload")
            else:
                upload(staging, repo_id, repo_type)

    return 0


if __name__ == "__main__":
    sys.exit(main())
