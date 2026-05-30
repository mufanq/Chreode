"""Unit tests for BR-CellDrift-MLP.

Verifies:
  - Output shape is (B, K, dim)
  - α(0) = 0 ⇒ ẑ(Δ=0) = z (identity preserved)
  - K-sample mean equals deterministic predict_mean() (zero-mean centering)
  - Model is differentiable end-to-end
  - Different noise → different outputs (stochasticity works)
  - Same noise + different source → different outputs (conditional dependency)
  - Action conditioning works
"""
from __future__ import annotations

import pytest
import torch

from cellworldmodel.model.br_celldrift_bench import BRCellDriftMLP, AlphaGate, FourierTimeEmbedding


def make_model(dim=8, noise_dim=4, action_dim=0, hidden_dim=32, n_layers=2):
    return BRCellDriftMLP(
        dim=dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        noise_dim=noise_dim,
        time_emb_dim=16,
        action_dim=action_dim,
    )


def test_forward_shape():
    torch.manual_seed(0)
    B, K, D = 5, 4, 8
    model = make_model(dim=D)
    z = torch.randn(B, D)
    delta = torch.rand(B) + 0.1
    eps = torch.randn(B, K, D)
    out = model(z, delta, eps)
    assert out.shape == (B, K, D)


def test_alpha_gate_zero_delta():
    """α(0) = 0 ⇒ ẑ(Δ=0) = z exactly."""
    torch.manual_seed(0)
    B, K, D = 4, 3, 6
    model = make_model(dim=D)
    z = torch.randn(B, D)
    delta = torch.zeros(B)
    eps = torch.randn(B, K, D)
    out = model(z, delta, eps)
    # Each of K predictions should equal z (since α(0) = 0)
    for r in range(K):
        assert torch.allclose(out[:, r, :], z, atol=1e-5), f"sample {r} differs from z at Δ=0"


def test_zero_mean_centering():
    """K-sample mean == deterministic predict_mean (since h̃ is centered)."""
    torch.manual_seed(0)
    B, K, D = 6, 8, 10
    model = make_model(dim=D, noise_dim=6)
    z = torch.randn(B, D)
    delta = torch.rand(B) + 0.5
    eps = torch.randn(B, K, D)

    z_hat_stochastic = model(z, delta, eps)  # (B, K, D)
    z_hat_mean = z_hat_stochastic.mean(dim=1)  # (B, D)

    z_hat_deterministic = model.predict_mean(z, delta)  # (B, D)

    # They should match exactly (zero-mean centering guarantees this)
    assert torch.allclose(z_hat_mean, z_hat_deterministic, atol=1e-5), (
        f"max diff: {(z_hat_mean - z_hat_deterministic).abs().max().item()}"
    )


def test_stochasticity_different_noise():
    """Different ε → different outputs."""
    torch.manual_seed(0)
    B, K, D = 3, 5, 8
    model = make_model(dim=D)
    z = torch.randn(B, D)
    delta = torch.rand(B) + 0.5
    eps = torch.randn(B, K, D)

    out = model(z, delta, eps)  # (B, K, D)
    # K samples per source should differ
    for i in range(B):
        dists = ((out[i, :, None, :] - out[i, None, :, :]) ** 2).sum(-1)  # (K, K)
        off_diag = dists[~torch.eye(K, dtype=torch.bool)]
        assert off_diag.mean().item() > 1e-4, f"source {i} has identical predictions"


def test_differentiable():
    """End-to-end differentiable w.r.t. parameters."""
    torch.manual_seed(0)
    B, K, D = 4, 3, 6
    model = make_model(dim=D)
    z = torch.randn(B, D, requires_grad=True)
    delta = torch.rand(B) + 0.3
    eps = torch.randn(B, K, D)

    out = model(z, delta, eps)
    loss = out.pow(2).mean()
    loss.backward()

    # Check gradients exist for all parameters
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"

    # Check z gradient
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_action_conditioning():
    """Different actions should give different outputs."""
    torch.manual_seed(0)
    B, K, D, A = 4, 3, 6, 5
    model = make_model(dim=D, action_dim=A)
    z = torch.randn(B, D)
    delta = torch.rand(B) + 0.5
    eps = torch.randn(B, K, D)
    action1 = torch.randn(B, A)
    action2 = torch.randn(B, A) * 2

    out1 = model(z, delta, eps, action=action1)
    out2 = model(z, delta, eps, action=action2)

    assert not torch.allclose(out1, out2, atol=1e-3), "action conditioning has no effect"


def test_conditional_on_source():
    """Different source z → different outputs."""
    torch.manual_seed(0)
    B, K, D = 4, 3, 6
    model = make_model(dim=D)
    z1 = torch.randn(B, D)
    z2 = torch.randn(B, D) + 5.0  # shifted far
    delta = torch.rand(B) + 0.5
    eps = torch.randn(B, K, D)

    out1 = model(z1, delta, eps)
    out2 = model(z2, delta, eps)
    # Since ẑ = z + α·R, and z differs, outputs should clearly differ
    assert not torch.allclose(out1, out2, atol=0.1)


def test_alpha_gate_monotonic():
    """α(Δ) should be monotonically increasing in Δ."""
    gate = AlphaGate(tau_init=1.0)
    deltas = torch.linspace(0.0, 10.0, 50)
    alpha = gate(deltas)
    diffs = alpha[1:] - alpha[:-1]
    assert (diffs >= 0).all(), "α(Δ) not monotonic"
    assert alpha[0].item() == pytest.approx(0.0, abs=1e-6), "α(0) != 0"
    assert alpha[-1].item() > 0.99, "α(10τ) should be close to 1"


def test_time_embedding_shape():
    emb = FourierTimeEmbedding(out_dim=32, n_freqs=16)
    delta = torch.rand(7)
    out = emb(delta)
    assert out.shape == (7, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
