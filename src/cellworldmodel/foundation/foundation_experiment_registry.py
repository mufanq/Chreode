"""Foundation experiment recipes for perturbation-pretraining ablations."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FoundationExperimentRecipe:
    name: str
    description: str
    vae_checkpoint: str
    pretrain_protocol: str
    dynamics_init: str
    downstream_init: str


FOUNDATION_EXPERIMENTS: dict[str, FoundationExperimentRecipe] = {
    "vae2_only": FoundationExperimentRecipe(
        name="vae2_only",
        description="A0: two-epoch VAE only; downstream perturbation transition module starts random.",
        vae_checkpoint="epoch_2.pt",
        pretrain_protocol="none",
        dynamics_init="none",
        downstream_init="random",
    ),
    "vae2_staticdit2": FoundationExperimentRecipe(
        name="vae2_staticdit2",
        description="A1: two-epoch VAE + two epoch-equivalent static DiT reconstruction control.",
        vae_checkpoint="epoch_2.pt",
        pretrain_protocol="static_dit_reconstruction",
        dynamics_init="static_dit",
        downstream_init="static_dit",
    ),
    "vae2_dynamicsdit2": FoundationExperimentRecipe(
        name="vae2_dynamicsdit2",
        description="A2: two-epoch VAE + two epoch-equivalent temporal W-DiT dynamics pretraining.",
        vae_checkpoint="epoch_2.pt",
        pretrain_protocol="temporal_dynamics",
        dynamics_init="temporal_dynamics",
        downstream_init="temporal_dynamics",
    ),
}


def get_foundation_experiment(name: str) -> FoundationExperimentRecipe:
    try:
        return FOUNDATION_EXPERIMENTS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown foundation experiment={name!r}; expected one of {sorted(FOUNDATION_EXPERIMENTS)}"
        ) from exc


def foundation_experiment_names() -> list[str]:
    return sorted(FOUNDATION_EXPERIMENTS)
