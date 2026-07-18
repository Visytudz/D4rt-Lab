"""Video encoder backbone for D4RT."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_attention_pattern(pattern: str | None) -> str:
    raw = (pattern or "global").strip().lower()
    aliases = {
        "interleaved_spatial_and_global": "interleaved_spatial_global",
        "interleaved_spatial_global": "interleaved_spatial_global",
        "global": "global",
        "full_global": "global",
    }
    return aliases.get(raw, raw)


def _sinusoidal_position_embedding(
    length: int,
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Construct a fixed sinusoidal position embedding.

    Parameters
    ----------
    length : int
        Number of token positions ``N``.
    dim : int
        Even embedding dimension ``C``.
    device : torch.device
        Destination device.

    Returns
    -------
    torch.Tensor, shape [N, C], dtype float32
        Fixed sine/cosine position features.
    """
    if dim % 2 != 0:
        raise ValueError("Sinusoidal position embedding requires an even dimension.")

    positions = torch.arange(
        length, device=device, dtype=torch.float32
    ).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )

    embedding = torch.zeros(length, dim, device=device, dtype=torch.float32)
    embedding[:, 0::2] = torch.sin(positions * frequencies)
    embedding[:, 1::2] = torch.cos(positions * frequencies)
    return embedding


class SelfAttentionBlock(nn.Module):
    """Pre-norm transformer block with self-attention + MLP."""

    def __init__(
        self, hidden_dim: int, num_heads: int, mlp_ratio: float, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        ff_dim = int(math.ceil(hidden_dim * mlp_ratio))
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        q = self.norm_attn(tokens)
        attn_out, _ = self.attn(q, q, q, need_weights=False)
        x = tokens + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x


class D4RTEncoder(nn.Module):
    """Encode video patches with interleaved spatial and global attention.

    Parameters
    ----------
    in_channels : int
        Number of input image channels.
    hidden_dim : int
        Transformer token dimension.
    patch_size_t_h_w : tuple[int, int, int]
        Temporal and spatial kernel/stride used by the 3D patch embedding.
    num_layers : int
        Number of transformer blocks.
    num_heads : int
        Number of self-attention heads.
    mlp_ratio : float
        Ratio between the feed-forward hidden size and token dimension.
    max_tokens : int
        Soft upper bound used to spatially pool non-standard inputs.
    attention_pattern : str or None
        Global attention or alternating per-time-slice spatial and global
        attention.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        patch_size_t_h_w: tuple[int, int, int],
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        max_tokens: int = 4096,
        attention_pattern: str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_tokens = max_tokens
        self.attention_pattern = _normalize_attention_pattern(attention_pattern)
        self.patch_embed = nn.Conv3d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=patch_size_t_h_w,
            stride=patch_size_t_h_w,
        )

        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=0.1,
                )
                for _ in range(num_layers)
            ]
        )
        self.block_modes = self._build_block_modes(num_layers)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def _build_block_modes(self, num_layers: int) -> list[Literal["spatial", "global"]]:
        if self.attention_pattern == "interleaved_spatial_global":
            return ["spatial" if (i % 2 == 0) else "global" for i in range(num_layers)]
        return ["global"] * num_layers

    def _token_cap(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C', T', H', W']
        b, c, tp, hp, wp = x.shape
        token_count = tp * hp * wp
        if token_count <= self.max_tokens:
            return x
        scale = math.sqrt(self.max_tokens / float(token_count))
        out_h = max(1, int(round(hp * scale)))
        out_w = max(1, int(round(wp * scale)))
        return F.adaptive_avg_pool3d(x, output_size=(tp, out_h, out_w))

    def forward(
        self, video: torch.Tensor, aspect_ratio_token: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode a batch of preprocessed video clips.

        Parameters
        ----------
        video : torch.Tensor, shape [B, T, C, H, W]
            Preprocessed floating-point video tensor.
        aspect_ratio_token : torch.Tensor or None, shape [B, 1, C']
            Learned token representing each video's original aspect
            ratio. It participates only in global-attention blocks.

        Returns
        -------
        torch.Tensor, shape [B, N(+1), C']
            Encoded video tokens, followed by the updated aspect-ratio token
            when ``aspect_ratio_token`` is provided.
        """
        # prepare video tokens
        if video.ndim != 5:
            raise ValueError(f"Expected video tensor with ndim=5, got {video.shape}")
        x = video.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        x = self.patch_embed(x)  # [B, C', T', H', W'], C' = hidden_dim
        x = self._token_cap(x)
        b, cp, tp, hp, wp = x.shape
        video_tokens = x.flatten(2).transpose(1, 2)  # [B, N, C']
        token_count = video_tokens.shape[1]
        pos = _sinusoidal_position_embedding(
            token_count, self.hidden_dim, video_tokens.device
        )  # [N, C']
        video_tokens = video_tokens + pos.unsqueeze(0)

        # validate aspect ratio token
        if aspect_ratio_token is not None:
            if aspect_ratio_token.ndim != 3:
                raise ValueError(
                    "Expected aspect_ratio_token [B, 1, C'], "
                    f"got {aspect_ratio_token.shape}"
                )
            if (
                aspect_ratio_token.shape[0] != b
                or aspect_ratio_token.shape[1] != 1
                or aspect_ratio_token.shape[2] != self.hidden_dim
            ):
                raise ValueError(
                    "aspect_ratio_token must have shape "
                    f"[B={b}, 1, C'={self.hidden_dim}], got {aspect_ratio_token.shape}"
                )

        # self-attention with interleaved spatial and global blocks
        spatial_tokens = hp * wp
        for mode, block in zip(self.block_modes, self.blocks):
            if mode == "spatial":
                spatial = video_tokens.reshape(b, tp, spatial_tokens, cp).reshape(
                    b * tp, spatial_tokens, cp
                )
                spatial = block(spatial)
                video_tokens = spatial.reshape(b, tp, spatial_tokens, cp).reshape(
                    b, tp * spatial_tokens, cp
                )
            elif mode == "global":
                if aspect_ratio_token is None:
                    video_tokens = block(video_tokens)
                else:
                    merged = torch.cat(
                        [video_tokens, aspect_ratio_token], dim=1
                    )  # [B, N+1, C']
                    merged = block(merged)
                    video_tokens = merged[:, :token_count]
                    aspect_ratio_token = merged[:, token_count:]
            else:
                raise ValueError(f"Unsupported attention block mode: {mode}")

        # final normalization and return
        if aspect_ratio_token is None:
            encoded = video_tokens  # [B, N, C']
        else:
            encoded = torch.cat(
                [video_tokens, aspect_ratio_token], dim=1
            )  # [B, N+1, C']
        return self.final_norm(encoded)
