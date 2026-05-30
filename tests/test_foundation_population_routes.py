from __future__ import annotations

import torch

from cellworldmodel.foundation.perturbation_population_models import (
    GeneGraphPriorConfig,
    GeneProgramResponseDecoder,
    InternalWaddingtonFieldSurgery,
    PopulationPredictorConfig,
    ResponseDecoderConfig,
    build_population_predictor,
    build_response_decoder,
)
from cellworldmodel.model.waddington_dit_1d import WaddingtonDiT1D


def _tiny_wdit(dim: int = 6) -> WaddingtonDiT1D:
    return WaddingtonDiT1D(
        dim=dim,
        hidden_dim=24,
        depth=1,
        num_heads=4,
        num_register_tokens=1,
        time_emb_dim=8,
        curl_rank=2,
        time_embedding_mode="bounded_lowfreq_fourier",
        curl_time_embedding_mode="bounded_lowfreq_fourier",
        curl_time_mode="separate",
    )


def test_internal_waddington_adapters_have_gradients() -> None:
    torch.manual_seed(0)
    model = build_population_predictor(
        config=PopulationPredictorConfig(
            route="route1_internal",
            latent_dim=6,
            action_dim=4,
            n_programs=3,
            adapter_components="full",
        ),
        base_transition=_tiny_wdit(6),
    )
    assert isinstance(model, InternalWaddingtonFieldSurgery)
    z = torch.randn(5, 6)
    action = torch.randn(5, 4)
    out = model(z, action)
    loss = out.z.pow(2).mean()
    loss.backward()
    assert model.u_basis.grad is not None and model.u_basis.grad.abs().sum().item() > 0
    assert model.p_basis.grad is not None and model.p_basis.grad.abs().sum().item() > 0
    assert model.q_basis.grad is not None and model.q_basis.grad.abs().sum().item() > 0
    assert model.kick_net[-1].weight.grad is not None


def test_hybrid_rollout_population_route_builds() -> None:
    torch.manual_seed(0)
    model = build_population_predictor(
        config=PopulationPredictorConfig(
            route="hybrid_rollout",
            latent_dim=6,
            action_dim=4,
            n_programs=3,
            k_samples=2,
        ),
        base_transition=_tiny_wdit(6),
    )
    z = torch.randn(5, 6)
    action = torch.randn(5, 4)
    out = model(z, action)
    loss = out.z.pow(2).mean()
    loss.backward()
    assert out.z.shape == z.shape
    assert "gate_mean" in out.aux
    assert model.kick_net[-1].weight.grad is not None


def test_program_response_decoder_has_basis_gradient() -> None:
    torch.manual_seed(0)
    decoder = build_response_decoder(ResponseDecoderConfig(
        response_decoder="program",
        n_genes=11,
        latent_dim=6,
        action_dim=4,
        response_programs=5,
    ))
    assert isinstance(decoder, GeneProgramResponseDecoder)
    coarse = torch.randn(7, 11)
    z = torch.randn(7, 6)
    action = torch.randn(7, 4)
    pred, info = decoder(coarse, z, action)
    loss = pred[:, :3].pow(2).mean()
    loss.backward()
    assert decoder.program_to_gene.grad is not None
    assert decoder.program_to_gene.grad.abs().sum().item() > 0
    assert "program_decoder_delta_norm" in info


def test_ktvu_rollout_predictor_has_delta_u_gradients() -> None:
    torch.manual_seed(0)
    model = build_population_predictor(
        config=PopulationPredictorConfig(
            route="ktvu_rollout",
            latent_dim=6,
            action_dim=4,
            n_programs=3,
            rollout_steps=2,
        ),
        base_transition=_tiny_wdit(6),
    )
    z = torch.randn(5, 6)
    action = torch.randn(5, 4)
    out = model(z, action)
    loss = out.z.pow(2).mean()
    loss.backward()
    assert out.z.shape == z.shape
    assert out.tensors is not None and "program_coeff" in out.tensors
    assert model.delta_u_basis[-1].weight.grad is not None
    assert model.delta_u_basis[-1].weight.grad.abs().sum().item() > 0
    assert "virtual_time_mean" in out.aux


def test_ktvu_ablation_flags_disable_expected_parts() -> None:
    torch.manual_seed(0)
    model = build_population_predictor(
        config=PopulationPredictorConfig(
            route="ktvu_rollout",
            latent_dim=6,
            action_dim=4,
            n_programs=3,
            rollout_steps=2,
            disable_kick=True,
            disable_field=True,
            disable_rollout=True,
            disable_action_time=True,
        ),
        base_transition=_tiny_wdit(6),
    )
    z = torch.randn(5, 6)
    action = torch.randn(5, 4)
    out = model(z, action)
    torch.testing.assert_close(out.z, z)
    assert out.aux["rollout_steps"] == 0.0
    assert out.aux["kick_norm"] == 0.0
    assert abs(out.aux["virtual_time_mean"] - 1.0) < 1e-6


def test_native_u_bridge_predictor_has_feature_adapter_gradients() -> None:
    torch.manual_seed(0)
    model = build_population_predictor(
        config=PopulationPredictorConfig(
            route="native_u_bridge",
            latent_dim=6,
            action_dim=4,
            n_programs=3,
            rollout_steps=1,
        ),
        base_transition=_tiny_wdit(6),
    )
    z = torch.randn(5, 6)
    action = torch.randn(5, 4)
    out = model(z, action)
    out.z.pow(2).mean().backward()
    assert out.tensors is not None and "program_coeff" in out.tensors
    assert model.gamma_basis.grad is not None
    assert model.gamma_basis.grad.abs().sum().item() > 0
    assert "u_gamma_norm" in out.aux


def test_shared_coefficient_response_decoder_uses_predictor_coefficients() -> None:
    decoder = build_response_decoder(ResponseDecoderConfig(
        response_decoder="shared_program",
        n_genes=9,
        latent_dim=6,
        action_dim=4,
        response_programs=3,
        nonnegative_basis=True,
    ))
    coarse = torch.randn(5, 9)
    z = torch.randn(5, 6)
    action = torch.randn(5, 4)
    coeff = torch.softmax(torch.randn(5, 3), dim=-1).requires_grad_(True)
    pred, info = decoder(coarse, z, action, {"program_coeff": coeff})
    pred.pow(2).mean().backward()
    assert coeff.grad is not None
    assert decoder.program_to_gene.grad is not None
    assert "shared_decoder_delta_norm" in info


def test_shared_signed_response_decoder_uses_predictor_coefficients() -> None:
    decoder = build_response_decoder(ResponseDecoderConfig(
        response_decoder="shared_signed",
        n_genes=9,
        latent_dim=6,
        action_dim=4,
        response_programs=3,
    ))
    coarse = torch.randn(5, 9)
    z = torch.randn(5, 6)
    action = torch.randn(5, 4)
    coeff = torch.softmax(torch.randn(5, 3), dim=-1).requires_grad_(True)
    pred, info = decoder(coarse, z, action, {"program_coeff": coeff})
    pred.pow(2).mean().backward()
    assert coeff.grad is not None
    assert decoder.up_basis.grad is not None
    assert decoder.down_basis.grad is not None
    assert "shared_signed_delta_norm" in info


def test_structured_program_decoder_options_have_gradients() -> None:
    decoder = build_response_decoder(ResponseDecoderConfig(
        response_decoder="program",
        n_genes=9,
        latent_dim=6,
        action_dim=4,
        response_programs=5,
        use_sparse_programs=True,
        nonnegative_basis=True,
        use_set_context=True,
    ))
    coarse = torch.randn(6, 9)
    z = torch.randn(6, 6)
    action = torch.randn(6, 4)
    pred, _ = decoder(coarse, z, action)
    loss = pred.pow(2).mean()
    loss.backward()
    assert decoder.program_to_gene.grad is not None
    assert decoder.program_to_gene.grad.abs().sum().item() > 0


def test_gene_token_response_decoder_builds() -> None:
    decoder = build_response_decoder(ResponseDecoderConfig(
        response_decoder="gene_token",
        n_genes=9,
        latent_dim=6,
        action_dim=4,
        response_programs=7,
    ))
    coarse = torch.randn(6, 9)
    z = torch.randn(6, 6)
    action = torch.randn(6, 4)
    torch.nn.init.normal_(decoder.query[-1].weight, std=0.02)
    pred, info = decoder(coarse, z, action)
    assert pred.shape == coarse.shape
    pred.pow(2).mean().backward()
    assert "gene_token_delta_norm" in info


def test_gene_token_structured_options_have_gradients() -> None:
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_weight = torch.ones(3)
    decoder = build_response_decoder(ResponseDecoderConfig(
        response_decoder="gene_token",
        n_genes=9,
        latent_dim=6,
        action_dim=4,
        response_programs=7,
        use_sparse_programs=True,
        nonnegative_basis=True,
        use_set_context=True,
        graph_prior=GeneGraphPriorConfig(
            mode="both",
            edge_index=edge_index,
            edge_weight=edge_weight,
            basis_weight=0.1,
            output_weight=0.1,
        ),
    ))
    coarse = torch.randn(6, 9)
    z = torch.randn(6, 6)
    action = torch.randn(6, 4)
    torch.nn.init.normal_(decoder.query[-1].weight, std=0.02)
    pred, info = decoder(coarse, z, action)
    pred.pow(2).mean().backward()
    assert decoder.gene_emb.grad is not None
    assert decoder.gene_emb.grad.abs().sum().item() > 0
    assert "gene_token_gate_entropy" in info


def test_gene_token_zero_weight_graph_matches_base() -> None:
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_weight = torch.ones(3)
    torch.manual_seed(1)
    base = build_response_decoder(ResponseDecoderConfig(
        response_decoder="gene_token",
        n_genes=9,
        latent_dim=6,
        action_dim=4,
        response_programs=7,
    ))
    graph = build_response_decoder(ResponseDecoderConfig(
        response_decoder="gene_token",
        n_genes=9,
        latent_dim=6,
        action_dim=4,
        response_programs=7,
        graph_prior=GeneGraphPriorConfig(
            mode="both",
            edge_index=edge_index,
            edge_weight=edge_weight,
            basis_weight=0.0,
            output_weight=0.0,
        ),
    ))
    graph.load_state_dict(base.state_dict(), strict=False)
    coarse = torch.randn(6, 9)
    z = torch.randn(6, 6)
    action = torch.randn(6, 4)
    torch.nn.init.normal_(base.query[-1].weight, std=0.02)
    graph.query[-1].weight.data.copy_(base.query[-1].weight.data)
    graph.query[-1].bias.data.copy_(base.query[-1].bias.data)
    base_pred, _ = base(coarse, z, action)
    graph_pred, _ = graph(coarse, z, action)
    torch.testing.assert_close(graph_pred, base_pred)


def test_gene_graph_mpnn_response_decoder_has_gradients() -> None:
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_weight = torch.ones(4)
    decoder = build_response_decoder(ResponseDecoderConfig(
        response_decoder="gene_mpnn",
        n_genes=9,
        latent_dim=6,
        action_dim=4,
        response_programs=7,
        use_sparse_programs=True,
        nonnegative_basis=True,
        graph_prior=GeneGraphPriorConfig(
            mode="basis",
            edge_index=edge_index,
            edge_weight=edge_weight,
            basis_weight=0.05,
        ),
        graph_layers=2,
    ))
    coarse = torch.randn(6, 9)
    z = torch.randn(6, 6)
    action = torch.randn(6, 4)
    torch.nn.init.normal_(decoder.query[-1].weight, std=0.02)
    pred, info = decoder(coarse, z, action)
    pred.pow(2).mean().backward()
    assert decoder.gene_emb.grad is not None
    assert decoder.graph_projs[0].weight.grad is not None
    assert "gene_graph_mpnn_layers" in info


def test_gene_mpnn_residual_response_decoder_has_gradients() -> None:
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_weight = torch.ones(4)
    decoder = build_response_decoder(ResponseDecoderConfig(
        response_decoder="gene_mpnn_residual",
        n_genes=9,
        latent_dim=6,
        action_dim=4,
        response_programs=7,
        use_sparse_programs=True,
        nonnegative_basis=True,
        graph_prior=GeneGraphPriorConfig(
            mode="both",
            edge_index=edge_index,
            edge_weight=edge_weight,
            basis_weight=0.05,
            output_weight=0.05,
        ),
        graph_layers=2,
    ))
    coarse = torch.randn(6, 9)
    z = torch.randn(6, 6)
    action = torch.randn(6, 4)
    torch.nn.init.normal_(decoder.query[-1].weight, std=0.02)
    pred, info = decoder(coarse, z, action)
    pred.pow(2).mean().backward()
    assert decoder.graph_projs[0].weight.grad is not None
    assert decoder.residual_logit.grad is not None
    assert "gene_graph_mpnn_residual_gamma" in info


def test_signed_gene_token_response_decoder_has_gradients() -> None:
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_weight = torch.ones(3)
    decoder = build_response_decoder(ResponseDecoderConfig(
        response_decoder="signed_gene_token",
        n_genes=9,
        latent_dim=6,
        action_dim=4,
        response_programs=7,
        use_sparse_programs=True,
        nonnegative_basis=True,
        graph_prior=GeneGraphPriorConfig(
            mode="both",
            edge_index=edge_index,
            edge_weight=edge_weight,
            basis_weight=0.05,
            output_weight=0.05,
        ),
    ))
    coarse = torch.randn(6, 9)
    z = torch.randn(6, 6)
    action = torch.randn(6, 4)
    torch.nn.init.normal_(decoder.query[-1].weight, std=0.02)
    pred, info = decoder(coarse, z, action)
    pred.pow(2).mean().backward()
    assert decoder.gene_emb.grad is not None
    assert decoder.down_gene_emb.grad is not None
    assert "signed_up_norm" in info


def test_signed_gene_mpnn_residual_response_decoder_has_gradients() -> None:
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_weight = torch.ones(4)
    decoder = build_response_decoder(ResponseDecoderConfig(
        response_decoder="signed_gene_mpnn_residual",
        n_genes=9,
        latent_dim=6,
        action_dim=4,
        response_programs=7,
        use_sparse_programs=True,
        nonnegative_basis=True,
        graph_prior=GeneGraphPriorConfig(
            mode="both",
            edge_index=edge_index,
            edge_weight=edge_weight,
            basis_weight=0.05,
            output_weight=0.05,
        ),
        graph_layers=2,
    ))
    coarse = torch.randn(6, 9)
    z = torch.randn(6, 6)
    action = torch.randn(6, 4)
    torch.nn.init.normal_(decoder.query[-1].weight, std=0.02)
    pred, info = decoder(coarse, z, action)
    pred.pow(2).mean().backward()
    assert decoder.graph_projs[0].weight.grad is not None
    assert decoder.residual_logit.grad is not None
    assert "gene_graph_mpnn_residual_gamma" in info
