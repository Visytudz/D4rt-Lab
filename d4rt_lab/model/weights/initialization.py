"""Select and apply one D4RT model initialization source."""

from __future__ import annotations

from ..d4rt import D4RTModel
from .artifact import load_model_artifact
from .config import InitializationConfig
from .pretrained import load_pretrained_encoder


def initialize_model(
    model: D4RTModel,
    initialization_cfg: InitializationConfig,
) -> None:
    """Initialize a model from one canonical source."""
    if initialization_cfg.source == "random":
        return
    elif initialization_cfg.source == "artifact":
        artifact_path = initialization_cfg.artifact_path
        if artifact_path is None:
            raise ValueError("Artifact initialization requires artifact_path.")
        artifact = load_model_artifact(artifact_path)
        model.load_state_dict(artifact.state_dict, strict=True)
    elif initialization_cfg.source == "encoder_pretrained":
        load_pretrained_encoder(model.encoder, initialization_cfg.encoder_pretrained)
    else:
        raise ValueError(
            f"Unsupported initialization source: {initialization_cfg.source!r}"
        )
