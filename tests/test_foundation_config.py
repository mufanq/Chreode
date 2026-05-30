from pathlib import Path

import pytest
import yaml

from cellworldmodel.foundation.config import load_foundation_config


def write_config(tmp_path: Path, **updates) -> Path:
    cfg = {
        "output_root": "output/foundation/test",
        "data_root": "/data/genhui",
        "gene_vocab": {
            "source": "output/phase0/orthologs/mouse_human_1to1.parquet",
            "canonical_order": "mouse_unified_order_filtered",
        },
        "splits": {
            "split_seed": 42,
            "strict_zero_shot": {
                "heldout_families": ["GSE275562"],
                "external": ["Norman"],
            },
        },
        "vae": {
            "policy": "vae_b_log1p_normal",
            "batch_strategy": "b1_leaf_dataset",
            "smoke_batch_strategies": ["b1_leaf_dataset", "b3_none"],
            "latent_dims": [128, 256],
        },
        "dynamics": {
            "experiment": "g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw",
            "controls": [
                "g2a_m10_wdit_time2vecu_lowfreqcurl_adamw",
                "g2a_m10_md_adamw",
            ],
            "transition_pairs": "all_ordered",
        },
    }
    cfg.update(updates)
    path = tmp_path / "foundation.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_load_foundation_config_defaults(tmp_path):
    cfg = load_foundation_config(write_config(tmp_path))
    assert cfg.output_root == "output/foundation/test"
    assert cfg.gene_vocab.canonical_order == "mouse_unified_order_filtered"
    assert cfg.splits.heldout_families == ("GSE275562",)
    assert cfg.splits.external == ("Norman",)
    assert cfg.splits.split_ratios == (0.7, 0.1, 0.2)
    assert cfg.vae.batch_strategy == "b1_leaf_dataset"
    assert cfg.vae.smoke_batch_strategies == ("b1_leaf_dataset", "b3_none")
    assert cfg.vae.latent_dims == (128, 256)
    assert cfg.vae.smoke_batch_size == 256
    assert cfg.dynamics.experiment == "g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw"
    assert cfg.dynamics.controls == (
        "g2a_m10_wdit_time2vecu_lowfreqcurl_adamw",
        "g2a_m10_md_adamw",
    )


def test_rejects_invalid_batch_strategy(tmp_path):
    path = write_config(tmp_path, vae={"batch_strategy": "bad", "latent_dims": [128]})
    with pytest.raises(ValueError, match="vae.batch_strategy"):
        load_foundation_config(path)


def test_rejects_empty_latent_dims(tmp_path):
    path = write_config(tmp_path, vae={"latent_dims": []})
    with pytest.raises(ValueError, match="latent_dims"):
        load_foundation_config(path)
