"""Structured configuration for D4RT model initialization."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EncoderPretrainedConfig:
    """External encoder checkpoint and failure policy."""

    type: str = "videomae_v2"
    path: str = ""
    must_succeed: bool = True


@dataclass
class InitializationConfig:
    """Select one source used to initialize D4RT model weights."""

    source: str = "random"
    artifact_path: str | None = None
    encoder_pretrained: EncoderPretrainedConfig = field(
        default_factory=EncoderPretrainedConfig
    )

    def __post_init__(self) -> None:
        """Validate fields required by the selected initialization source."""
        supported = {"random", "artifact", "encoder_pretrained"}
        if self.source not in supported:
            raise ValueError(
                f"Unsupported initialization source {self.source!r}; "
                f"expected one of {sorted(supported)}"
            )
        if self.source == "artifact" and not self.artifact_path:
            raise ValueError("Artifact initialization requires artifact_path.")
        if (
            self.source == "encoder_pretrained"
            and not self.encoder_pretrained.path
        ):
            raise ValueError(
                "Encoder-pretrained initialization requires encoder_pretrained.path."
            )
