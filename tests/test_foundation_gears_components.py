from __future__ import annotations

import torch
import torch.nn as nn

from cellworldmodel.foundation.gears_losses import GearsLossComputer, GearsLossWeights
from cellworldmodel.foundation.perturbation_predictors import (
    available_perturbation_predictors,
    build_perturbation_predictor,
    transition_uses_action,
)


class DummyTransition(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(latent_dim))

    def predict_mean(self, z, delta, action=None, n_mc: int = 1):
        del delta, action, n_mc
        return z + self.bias


def test_perturbation_predictor_registry_builds_all_registered_predictors() -> None:
    names = available_perturbation_predictors()
    assert names == ["direct_action", "hybrid_kick_rollout", "kick_only", "kick_rollout"]
    assert transition_uses_action("direct_action")
    assert not transition_uses_action("kick_only")
    assert not transition_uses_action("kick_rollout")
    assert not transition_uses_action("hybrid_kick_rollout")

    z = torch.randn(3, 8)
    action = torch.randn(3, 4)
    for name in names:
        predictor = build_perturbation_predictor(
            model_type=name,
            transition_model=DummyTransition(8),
            latent_dim=8,
            action_dim=4,
            k_samples=2,
        )
        out = predictor(z, action)
        assert out.z.shape == z.shape
        assert "kick_norm" in out.aux
        assert "gate_mean" in out.aux


def test_gears_direction_loss_is_differentiable() -> None:
    pred_x = torch.zeros(4, 5, requires_grad=True)
    target_x = torch.tensor([
        [1.0, -1.0, 0.5, -0.5, 2.0],
        [1.0, -1.0, 0.5, -0.5, 2.0],
        [1.0, -1.0, 0.5, -0.5, 2.0],
        [1.0, -1.0, 0.5, -0.5, 2.0],
    ])
    pred_z = torch.zeros(4, 2)
    target_z = torch.zeros(4, 2)
    loss, info = GearsLossComputer(GearsLossWeights(
        latent=0.0,
        expression=0.0,
        de=0.0,
        delta=0.0,
        direction=1.0,
    ))(
        pred_z=pred_z,
        target_z=target_z,
        pred_x=pred_x,
        target_x=target_x,
        control_mean=torch.zeros(5),
        de_idx=torch.arange(5),
    )
    loss.backward()
    assert pred_x.grad is not None
    assert torch.count_nonzero(pred_x.grad).item() > 0
    assert info["direction_loss"] > 0.0
    assert "direction_opposite_rate" in info
