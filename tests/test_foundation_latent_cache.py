import json

import numpy as np
import pandas as pd
import torch

from cellworldmodel.benchmark.registry import build_model
from cellworldmodel.foundation.dynamics_train import build_foundation_dynamics_cfg
from cellworldmodel.foundation.foundation_experiment_registry import foundation_experiment_names
from cellworldmodel.foundation.latent_cache import LatentCacheDataset
from cellworldmodel.foundation.action import GeneSetPerturbationEncoder
from cellworldmodel.foundation.perturbation_dataset import normalize_norman_condition
from cellworldmodel.foundation.pretrain_protocols import pretrain_protocol_names


def test_latent_cache_dataset_loads_requested_id_order(tmp_path):
    cache = tmp_path / "latents"
    (cache / "train").mkdir(parents=True)
    np.save(cache / "train" / "shard_000000.npy", np.asarray([[1, 2], [3, 4]], dtype=np.float32))
    np.save(cache / "train" / "shard_000001.npy", np.asarray([[5, 6]], dtype=np.float32))
    pd.DataFrame([
        {"global_cell_id": 10, "foundation_split": "train", "shard_path": "train/shard_000000.npy", "row_in_shard": 0},
        {"global_cell_id": 20, "foundation_split": "train", "shard_path": "train/shard_000000.npy", "row_in_shard": 1},
        {"global_cell_id": 30, "foundation_split": "train", "shard_path": "train/shard_000001.npy", "row_in_shard": 0},
    ]).to_parquet(cache / "index.parquet", index=False)
    (cache / "manifest.json").write_text(json.dumps({"latent_dim": 2}), encoding="utf-8")

    dataset = LatentCacheDataset(cache)
    z = dataset.load_ids([30, 10, 20])

    np.testing.assert_allclose(z, np.asarray([[5, 6], [1, 2], [3, 4]], dtype=np.float32))


def test_foundation_pretrain_recipe_names_are_registered():
    assert {"vae_warmup", "static_dit_reconstruction", "temporal_dynamics"}.issubset(pretrain_protocol_names())
    assert {"vae2_only", "vae2_staticdit2", "vae2_dynamicsdit2"}.issubset(foundation_experiment_names())


def test_foundation_dynamics_cfg_sets_foundation_delta_scale():
    transitions = pd.DataFrame({"delta": [1.0, 2.0, 4.0]})
    method, cfg, tau_init = build_foundation_dynamics_cfg(
        experiment="g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw",
        dit_size="small",
        batch_size=32,
        k_samples=4,
        lr=1e-4,
        transition_index=transitions,
    )

    assert method == "m10"
    assert cfg["dit_size"] == "small"
    assert cfg["batch_size"] == 32
    assert cfg["K"] == 4
    assert cfg["wdit_time_delta_scale"] == 4.0
    assert cfg["wdit_curl_time_delta_scale"] == 4.0
    assert tau_init > 0.0


def test_static_dit_constant_delta_has_model_gradients():
    method, cfg, tau_init = build_foundation_dynamics_cfg(
        experiment="g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw",
        dit_size="tiny",
        batch_size=2,
        k_samples=1,
        lr=1e-4,
    )
    model = build_model(method, dim=4, cfg=cfg, tau_init=tau_init)
    z = torch.randn(2, 4)
    delta = torch.ones(2)
    eps = torch.zeros(2, 1, 4)
    out = model(z, delta, eps).squeeze(1)
    loss = out.pow(2).mean()
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_waddington_dit_accepts_action_conditioning():
    method, cfg, tau_init = build_foundation_dynamics_cfg(
        experiment="g2a_m10_wdit_time2vecu_lowfreqcurl_uncertainty_adamw",
        dit_size="tiny",
        batch_size=2,
        k_samples=3,
        lr=1e-4,
    )
    cfg["action_dim"] = 5
    model = build_model(method, dim=4, cfg=cfg, tau_init=tau_init)
    z = torch.randn(2, 4)
    delta = torch.ones(2)
    eps = torch.randn(2, 3, 4)
    action = torch.randn(2, 5)
    out = model(z, delta, eps, action=action)
    assert out.shape == (2, 3, 4)
    mean = model.predict_mean(z, delta, action=action)
    assert mean.shape == (2, 4)


def test_normalize_norman_condition():
    assert normalize_norman_condition("ctrl") == "ctrl"
    assert normalize_norman_condition("A_B__A_B") == "A+B"
    assert normalize_norman_condition("B+A") == "A+B"


def test_geneset_action_encoder_permutation_invariant_and_sign_sensitive():
    encoder = GeneSetPerturbationEncoder(n_genes=10, action_dim=6, hidden_dim=12)
    gene_ids = torch.tensor([[1, 2], [2, 1]])
    signs = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    modality = torch.zeros(2, 2, dtype=torch.long)
    strengths = torch.ones(2, 2)
    mask = torch.ones(2, 2, dtype=torch.bool)
    out = encoder(gene_ids, signs, modality, strengths, mask)
    assert out.shape == (2, 6)
    torch.testing.assert_close(out[0], out[1])

    flipped = encoder(gene_ids[:1], -signs[:1], modality[:1], strengths[:1], mask[:1])
    assert not torch.allclose(out[:1], flipped)
