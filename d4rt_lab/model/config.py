"""Strict structured architecture configuration for D4RT."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InputConfig:
    """Describe the video tensor expected by D4RT.

    Parameters
    ----------
    clip_frames : int
        Number of frames ``T`` in an input clip shaped ``[B, C, T, H, W]``.
    image_size : tuple[int, int]
        Preprocessed spatial size ``(H, W)``.
    channels : int
        Number of input channels ``C``.
    """

    clip_frames: int = 48
    image_size: tuple[int, int] = (256, 256)
    channels: int = 3


@dataclass
class EncoderConfig:
    """Configure :class:`src.model.encoder.D4RTEncoder`.

    Parameters
    ----------
    in_channels : int
        Input channel count ``C`` of video tensors ``[B, C, T, H, W]``.
    hidden_dim : int
        Encoder token dimension ``D_enc``.
    patch_size_t_h_w : tuple[int, int, int]
        Temporal and spatial patch size ``(P_t, P_h, P_w)``.
    num_layers : int
        Number of Transformer blocks.
    num_heads : int
        Number of attention heads per block.
    mlp_ratio : float
        Ratio between the MLP intermediate dimension and ``D_enc``.
    max_tokens : int
        Maximum number of video tokens retained by the encoder.
    attention_pattern : str
        Rule used to arrange spatial and global attention blocks.
    """

    in_channels: int = 3
    hidden_dim: int = 1408
    patch_size_t_h_w: tuple[int, int, int] = (2, 16, 16)
    num_layers: int = 40
    num_heads: int = 16
    mlp_ratio: float = 6144.0 / 1408.0
    max_tokens: int = 6144
    attention_pattern: str = "interleaved_spatial_and_global"


@dataclass
class AspectRatioTokenConfig:
    """Configure the optional aspect-ratio token.

    Parameters
    ----------
    enabled : bool
        Whether one aspect-ratio token is appended to the video tokens.
    hidden_dim : int
        Token dimension, equal to the encoder dimension ``D_enc``.
    """

    enabled: bool = True
    hidden_dim: int = 1408


@dataclass
class MemoryProjectionConfig:
    """Configure the encoder-to-decoder memory projection.

    Parameters
    ----------
    input_dim : int
        Encoder memory dimension ``D_enc`` in ``[B, N, D_enc]``.
    output_dim : int
        Decoder memory dimension ``D_dec`` in ``[B, N, D_dec]``.
    """

    input_dim: int = 1408
    output_dim: int = 1280


@dataclass
class QueryEmbeddingConfig:
    """Configure the embedding of spatiotemporal point queries.

    Parameters
    ----------
    input_channels : int
        Video channel count ``C`` used by local RGB sampling.
    hidden_dim : int
        Output query dimension ``D_dec``.
    clip_frames : int
        Number of timesteps ``T`` represented by temporal query coordinates.
    uv_num_bands : int
        Number of Fourier frequency bands for image coordinates ``(u, v)``.
    local_patch_enabled : bool
        Whether to sample a local RGB patch around each query point.
    local_patch_size : int
        Odd side length of the sampled RGB patch.
    """

    input_channels: int = 3
    hidden_dim: int = 1280
    clip_frames: int = 48
    uv_num_bands: int = 8
    local_patch_enabled: bool = True
    local_patch_size: int = 9


@dataclass
class DecoderConfig:
    """Configure :class:`src.model.decoder.D4RTDecoder`.

    Parameters
    ----------
    hidden_dim : int
        Query and memory token dimension ``D_dec``.
    num_layers : int
        Number of independent query-decoder blocks.
    num_heads : int
        Number of cross-attention heads per block.
    mlp_ratio : float
        Ratio between the MLP intermediate dimension and ``D_dec``.
    """

    hidden_dim: int = 1280
    num_layers: int = 8
    num_heads: int = 16
    mlp_ratio: float = 3.5


@dataclass
class HeadsConfig:
    """Configure the D4RT prediction heads.

    Parameters
    ----------
    hidden_dim : int
        Decoder feature dimension ``D_dec`` of tensors ``[B, Q, D_dec]``.
    """

    hidden_dim: int = 1280


@dataclass
class D4RTModelConfig:
    """Combine all independently configurable D4RT components.

    Parameters
    ----------
    input : InputConfig
        Shape contract for input video tensors ``[B, C, T, H, W]``.
    encoder : EncoderConfig
        Video encoder architecture.
    aspect_ratio_token : AspectRatioTokenConfig
        Optional aspect-ratio token configuration.
    memory_projection : MemoryProjectionConfig
        Projection from encoder memory ``[B, N, D_enc]`` to decoder memory
        ``[B, N, D_dec]``.
    query_embedding : QueryEmbeddingConfig
        Point-query embedding configuration.
    decoder : DecoderConfig
        Query decoder architecture.
    heads : HeadsConfig
        Prediction-head dimensions.
    """

    input: InputConfig = field(default_factory=InputConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    aspect_ratio_token: AspectRatioTokenConfig = field(default_factory=AspectRatioTokenConfig)
    memory_projection: MemoryProjectionConfig = field(default_factory=MemoryProjectionConfig)
    query_embedding: QueryEmbeddingConfig = field(default_factory=QueryEmbeddingConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    heads: HeadsConfig = field(default_factory=HeadsConfig)

    def __post_init__(self) -> None:
        """Validate shape contracts shared by independently configured modules.

        Every field remains explicit so a component config maps directly onto
        its module constructor. This validation prevents incompatible channel,
        frame, or hidden dimensions from reaching tensor operations.
        """
        equalities = {
            "input.channels == encoder.in_channels": (
                self.input.channels,
                self.encoder.in_channels,
            ),
            "input.channels == query_embedding.input_channels": (
                self.input.channels,
                self.query_embedding.input_channels,
            ),
            "input.clip_frames == query_embedding.clip_frames": (
                self.input.clip_frames,
                self.query_embedding.clip_frames,
            ),
            "encoder.hidden_dim == aspect_ratio_token.hidden_dim": (
                self.encoder.hidden_dim,
                self.aspect_ratio_token.hidden_dim,
            ),
            "encoder.hidden_dim == memory_projection.input_dim": (
                self.encoder.hidden_dim,
                self.memory_projection.input_dim,
            ),
            "memory_projection.output_dim == decoder.hidden_dim": (
                self.memory_projection.output_dim,
                self.decoder.hidden_dim,
            ),
            "query_embedding.hidden_dim == decoder.hidden_dim": (
                self.query_embedding.hidden_dim,
                self.decoder.hidden_dim,
            ),
            "heads.hidden_dim == decoder.hidden_dim": (
                self.heads.hidden_dim,
                self.decoder.hidden_dim,
            ),
        }
        mismatches = [
            f"{name} ({left} != {right})"
            for name, (left, right) in equalities.items()
            if left != right
        ]
        if mismatches:
            raise ValueError("Inconsistent D4RT model dimensions: " + "; ".join(mismatches))
