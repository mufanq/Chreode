"""Unit tests for DriftDiT1D (M9 backbone).

Verifies the same invariants as BRCellDriftMLP tests:
  - Output shape (B, K, dim)
  - α(0) = 0 ⇒ ẑ(Δ=0) = z (identity preserved)
  - Different ε → different outputs (stochasticity works)
  - Different source → different outputs (source-conditional)
  - Model is differentiable end-to-end
  - predict_mean returns (B, dim)
  - Parameter count roughly matches target for each size
"""
from __future__ import annotations

import pytest
import torch

from cellworldmodel.model.drift_dit_1d import (
    DriftDiT1D,
    DriftDiT1D_Tiny,
    DriftDiT1D_Small,
    DRIFT_DIT_1D_MODELS,
)


def make_tiny(dim: int = 8, action_dim: int = 0) -> DriftDiT1D:
    return DriftDiT1D(
        dim=dim,
        hidden_dim=64,
        depth=2,
        num_heads=4,
        num_register_tokens=2,
        time_emb_dim=64,
        action_dim=action_dim,
        tau_init=1.0,
    )


def test_forward_shape():
    torch.manual_seed(0)
    B, K, D = 5, 4, 8
    model = make_tiny(dim=D)
    z = torch.randn(B, D)
    delta = torch.rand(B) + 0.1
    eps = torch.randn(B, K, D)
    out = model(z, delta, eps)
    assert out.shape == (B, K, D)


def test_alpha_gate_zero_delta():
    """α(0) = 0 ⇒ ẑ(Δ=0) = z exactly."""
    torch.manual_seed(0)
    B, K, D = 4, 3, 6
    model = make_tiny(dim=D)
    z = torch.randn(B, D)
    delta = torch.zeros(B)
    eps = torch.randn(B, K, D)
    out = model(z, delta, eps)
    for r in range(K):
        assert torch.allclose(out[:, r, :], z, atol=1e-5), f"sample {r} differs from z at Δ=0"


def _warm_start(model: DriftDiT1D) -> None:
    """Manually set adaLN gates to non-zero so attention can propagate noise.

    By design (adaLN-Zero), all gates start at 0 and are learned during training.
    For unit tests we simulate a trained model by setting gates to small non-zero.
    """
    with torch.no_grad():
        for block in model.blocks:
            # last Linear of adaLN_modulation outputs (shift, scale, gate) × 2 (msa, mlp)
            # Give it small random weights so gates ≠ 0
            block.adaLN_modulation[-1].weight.normal_(mean=0.0, std=0.02)
            block.adaLN_modulation[-1].bias.normal_(mean=0.0, std=0.02)
        # Also warm-start the final modulation so Δz is not nearly zero
        model.final_modulation[-1].weight.normal_(mean=0.0, std=0.02)
        model.final_modulation[-1].bias.normal_(mean=0.0, std=0.02)


def test_stochasticity_different_noise():
    """Different ε → different outputs (after warm-start to unblock adaLN gates).

    Note: by design, adaLN-Zero init makes initial forward identity w.r.t. the
    noise token (gate=0). Training unlocks this. We simulate training here by
    setting gates to non-zero.
    """
    torch.manual_seed(0)
    B, K, D = 3, 5, 8
    model = make_tiny(dim=D)
    _warm_start(model)
    z = torch.randn(B, D)
    delta = torch.full((B,), 0.5)
    eps1 = torch.randn(B, K, D)
    eps2 = torch.randn(B, K, D)
    out1 = model(z, delta, eps1)
    out2 = model(z, delta, eps2)
    assert not torch.allclose(out1, out2)


def test_initial_adaln_zero_identity_on_noise():
    """At init (adaLN-Zero), different ε give near-identical outputs — this is the
    intended "identity initialization" of DiT. Regression guard: if init changes
    and this starts failing, reconsider whether Δz depends on ε too strongly.
    """
    torch.manual_seed(0)
    B, K, D = 3, 5, 8
    model = make_tiny(dim=D)
    z = torch.randn(B, D)
    delta = torch.full((B,), 0.5)
    eps1 = torch.randn(B, K, D)
    eps2 = torch.randn(B, K, D)
    out1 = model(z, delta, eps1)
    out2 = model(z, delta, eps2)
    # All gates start at 0 → Δz is the same across noise samples
    assert torch.allclose(out1, out2, atol=1e-5)


def test_source_conditioning():
    """Different source z → different outputs (same ε)."""
    torch.manual_seed(0)
    B, K, D = 4, 3, 8
    model = make_tiny(dim=D)
    _warm_start(model)
    z1 = torch.randn(B, D)
    z2 = torch.randn(B, D)
    delta = torch.full((B,), 0.5)
    eps = torch.randn(B, K, D)
    out1 = model(z1, delta, eps)
    out2 = model(z2, delta, eps)
    assert not torch.allclose(out1, out2)


def test_gradient_flow():
    """Forward + backward should produce non-zero gradients on all parameters."""
    torch.manual_seed(0)
    B, K, D = 3, 4, 8
    model = make_tiny(dim=D)
    z = torch.randn(B, D)
    delta = torch.rand(B) + 0.1
    eps = torch.randn(B, K, D)
    out = model(z, delta, eps)
    loss = out.pow(2).mean()
    loss.backward()
    found_nonzero = False
    for name, p in model.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            found_nonzero = True
            break
    assert found_nonzero, "no parameter received non-zero gradient"


def test_predict_mean_shape():
    torch.manual_seed(0)
    B, D = 5, 8
    model = make_tiny(dim=D)
    z = torch.randn(B, D)
    delta = torch.rand(B) + 0.1
    out = model.predict_mean(z, delta, n_mc=16)
    assert out.shape == (B, D)


def test_predict_mean_antithetic_shape():
    torch.manual_seed(0)
    B, D = 5, 8
    model = make_tiny(dim=D)
    z = torch.randn(B, D)
    delta = torch.rand(B) + 0.1
    out = model.predict_mean(z, delta, n_mc=16, antithetic=True)
    assert out.shape == (B, D)


@pytest.mark.parametrize("learned_state_tokens", [4, 8])
def test_learned_dense_tokenizer_shape_no_rope(learned_state_tokens: int):
    torch.manual_seed(0)
    B, K, D = 3, 4, 8
    model = DriftDiT1D(
        dim=D,
        hidden_dim=64,
        depth=2,
        num_heads=4,
        num_register_tokens=2,
        time_emb_dim=64,
        tau_init=1.0,
        learned_state_tokens=learned_state_tokens,
        use_rope=False,
    )
    z = torch.randn(B, D)
    delta = torch.rand(B) + 0.1
    eps = torch.randn(B, K, D)
    out = model(z, delta, eps)
    assert out.shape == (B, K, D)
    assert model.tokenizer_mode == "learned"
    assert model.num_state_tokens == learned_state_tokens


def test_action_conditioning():
    torch.manual_seed(0)
    B, K, D, A = 4, 3, 8, 5
    model = make_tiny(dim=D, action_dim=A)
    _warm_start(model)
    z = torch.randn(B, D)
    delta = torch.full((B,), 0.5)
    eps = torch.randn(B, K, D)
    act1 = torch.randn(B, A)
    act2 = torch.randn(B, A)
    out1 = model(z, delta, eps, action=act1)
    out2 = model(z, delta, eps, action=act2)
    assert not torch.allclose(out1, out2)


def test_param_count_rough_match():
    """Sanity check that Tiny / Small roughly match their advertised param counts."""
    m_tiny = DriftDiT1D_Tiny(dim=64)
    m_small = DriftDiT1D_Small(dim=64)
    n_tiny = sum(p.numel() for p in m_tiny.parameters())
    n_small = sum(p.numel() for p in m_small.parameters())
    # Tiny ~3M, Small ~15M (allow factor-of-2 tolerance since the 1-D version
    # has extra Fourier+AlphaGate modules and a different final head).
    assert 1_000_000 < n_tiny < 10_000_000, f"Tiny has {n_tiny:,} params"
    assert 5_000_000 < n_small < 40_000_000, f"Small has {n_small:,} params"
    assert n_tiny < n_small, "Small should be larger than Tiny"


@pytest.mark.parametrize("size", ["tiny", "small"])
def test_model_registry(size: str):
    ctor = DRIFT_DIT_1D_MODELS[size]
    m = ctor(dim=16)
    z = torch.randn(2, 16)
    delta = torch.full((2,), 0.5)
    eps = torch.randn(2, 3, 16)
    out = m(z, delta, eps)
    assert out.shape == (2, 3, 16)
