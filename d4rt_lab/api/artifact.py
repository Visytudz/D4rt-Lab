"""Canonical model-artifact construction for the public API."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from d4rt_lab.model import D4RTModel, D4RTModelConfig
from d4rt_lab.model.weights import load_model_artifact


def resolve_device(device: str | torch.device) -> torch.device:
    resolved = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if str(device).lower() == "auto"
        else torch.device(device)
    )
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return resolved


def model_from_artifact(
    path: str, device: str | torch.device
) -> tuple[D4RTModel, D4RTModelConfig]:
    """Construct a strictly configured model from one canonical artifact."""
    artifact = load_model_artifact(path)
    structured = OmegaConf.merge(
        OmegaConf.structured(D4RTModelConfig), artifact.model_config
    )
    model_config = OmegaConf.to_object(structured)
    model = D4RTModel(model_config)
    model.load_state_dict(artifact.state_dict, strict=True)
    return model.to(resolve_device(device)), model_config
