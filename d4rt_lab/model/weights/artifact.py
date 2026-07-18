"""Save and load complete D4RT model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


MODEL_ARTIFACT_FORMAT = "open-d4rt-model"


@dataclass(frozen=True)
class ModelArtifact:
    """Serialized model configuration, parameters, and metadata.

    Parameters
    ----------
    model_config : dict[str, object]
        Configuration required to reconstruct the D4RT model.
    state_dict : dict[str, torch.Tensor]
        Complete D4RT model parameters.
    metadata : dict[str, object]
        Optional information about the exported model.
    """

    model_config: dict[str, object]
    state_dict: dict[str, torch.Tensor]
    metadata: dict[str, object]


def save_model_artifact(
    path: str | Path,
    model_config: dict[str, object],
    state_dict: dict[str, torch.Tensor],
    metadata: dict[str, object] | None = None,
) -> None:
    """Save a complete, framework-independent D4RT model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": MODEL_ARTIFACT_FORMAT,
            "model_config": model_config,
            "state_dict": state_dict,
            "metadata": metadata or {},
        },
        path,
    )


def load_model_artifact(
    path: str | Path,
    map_location: str | torch.device = "cpu",
    mmap: bool = True,
) -> ModelArtifact:
    """Load and validate a complete D4RT model artifact."""
    payload = torch.load(
        Path(path), map_location=map_location, weights_only=True, mmap=mmap
    )
    if not isinstance(payload, dict):
        raise ValueError("Model artifact must be a dictionary.")
    if payload.get("format") != MODEL_ARTIFACT_FORMAT:
        raise ValueError(f"Expected model artifact format {MODEL_ARTIFACT_FORMAT!r}.")

    model_config = payload.get("model_config")
    state_dict = payload.get("state_dict")
    metadata = payload.get("metadata")
    if not isinstance(model_config, dict):
        raise ValueError("Model artifact requires model_config.")
    if not isinstance(state_dict, dict) or not all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in state_dict.items()
    ):
        raise ValueError("Model artifact requires a tensor state_dict.")
    if not isinstance(metadata, dict):
        raise ValueError("Model artifact requires metadata.")

    return ModelArtifact(model_config, state_dict, metadata)
