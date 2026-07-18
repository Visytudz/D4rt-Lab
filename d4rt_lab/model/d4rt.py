"""Top-level D4RT model and query-oriented forward path."""

from __future__ import annotations

import torch
import torch.nn as nn

from d4rt_lab.core.types import ModelOutput, QueryBatch, TrainingBatch

from .config import D4RTModelConfig
from .decoder import D4RTDecoder
from .encoder import D4RTEncoder
from .heads import D4RTHeads
from .query import QueryEmbedder


class D4RTModel(nn.Module):
    """Compose the D4RT encoder, query decoder, and prediction heads.

    Parameters
    ----------
    model_cfg : D4RTModelConfig
        Strict structured architecture configuration.
    """

    def __init__(self, model_cfg: D4RTModelConfig) -> None:
        super().__init__()

        encoder_cfg = model_cfg.encoder
        aspect_cfg = model_cfg.aspect_ratio_token
        memory_cfg = model_cfg.memory_projection
        query_cfg = model_cfg.query_embedding
        decoder_cfg = model_cfg.decoder
        heads_cfg = model_cfg.heads

        # model components
        self.encoder = D4RTEncoder(
            in_channels=encoder_cfg.in_channels,
            hidden_dim=encoder_cfg.hidden_dim,
            patch_size_t_h_w=encoder_cfg.patch_size_t_h_w,
            num_layers=encoder_cfg.num_layers,
            num_heads=encoder_cfg.num_heads,
            mlp_ratio=encoder_cfg.mlp_ratio,
            max_tokens=encoder_cfg.max_tokens,
            attention_pattern=encoder_cfg.attention_pattern,
        )
        self.use_aspect_ratio_token = aspect_cfg.enabled
        self.aspect_ratio_proj = (
            nn.Sequential(
                nn.Linear(1, aspect_cfg.hidden_dim),
                nn.GELU(),
                nn.Linear(aspect_cfg.hidden_dim, aspect_cfg.hidden_dim),
            )
            if self.use_aspect_ratio_token
            else None
        )
        self.memory_proj = (
            nn.Identity()
            if memory_cfg.output_dim == memory_cfg.input_dim
            else nn.Linear(memory_cfg.input_dim, memory_cfg.output_dim)
        )
        self.query_embedder = QueryEmbedder(
            hidden_dim=query_cfg.hidden_dim,
            clip_frames=query_cfg.clip_frames,
            local_patch_enabled=query_cfg.local_patch_enabled,
            local_patch_size=query_cfg.local_patch_size,
            uv_num_bands=query_cfg.uv_num_bands,
        )
        self.decoder = D4RTDecoder(
            hidden_dim=decoder_cfg.hidden_dim,
            num_layers=decoder_cfg.num_layers,
            num_heads=decoder_cfg.num_heads,
            mlp_ratio=decoder_cfg.mlp_ratio,
        )
        self.heads = D4RTHeads(hidden_dim=heads_cfg.hidden_dim)

    def _project_aspect_ratio_token(
        self,
        video: torch.Tensor,
        aspect_ratio: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.aspect_ratio_proj is None:
            return None
        if aspect_ratio is None:
            aspect_ratio = torch.ones(
                (video.shape[0], 1), dtype=video.dtype, device=video.device
            )
        else:
            if not torch.is_tensor(aspect_ratio):
                aspect_ratio = torch.as_tensor(
                    aspect_ratio, dtype=video.dtype, device=video.device
                )
            aspect_ratio = aspect_ratio.to(device=video.device, dtype=video.dtype)
            if aspect_ratio.ndim == 1:
                aspect_ratio = aspect_ratio.unsqueeze(-1)
            if aspect_ratio.ndim != 2 or aspect_ratio.shape[0] != video.shape[0]:
                raise ValueError(
                    f"Expected aspect_ratio [B,1] or [B], got {tuple(aspect_ratio.shape)}"
                )
        return self.aspect_ratio_proj(aspect_ratio).unsqueeze(1)  # [B, 1, C']

    def encode_video(
        self,
        video: torch.Tensor,
        aspect_ratio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a batch of preprocessed video clips.

        Parameters
        ----------
        video : torch.Tensor, shape [B, T, C, H, W]
            Preprocessed floating-point video tensor.
        aspect_ratio : torch.Tensor or None, shape [B, 1]
            Original aspect ratio of each video clip. It is used to compute
            the learned aspect-ratio token when ``self.use_aspect_ratio_token`` is True.

        Returns
        -------
        torch.Tensor, shape [B, N, D]
            Encoded video tokens, where D is the decoder hidden dimension.
        """
        if video.ndim != 5:
            raise ValueError(f"Expected video [B,T,C,H,W], got {video.shape}")
        aspect_ratio_token = self._project_aspect_ratio_token(
            video=video, aspect_ratio=aspect_ratio
        )
        return self.memory_proj(
            self.encoder(video, aspect_ratio_token=aspect_ratio_token)
        )

    def decode_queries(
        self,
        video: torch.Tensor,
        query: QueryBatch | dict[str, torch.Tensor],
        memory: torch.Tensor,
    ) -> ModelOutput:
        """Decode spatio-temporal queries against cached video memory.

        Parameters
        ----------
        video : torch.Tensor, shape [B, T, C, H, W]
            Preprocessed floating-point video tensor.
        query : QueryBatch or dict[str, torch.Tensor]
            Query coordinates and frame-index tensors with shape ``[B, Q]``.
        memory : torch.Tensor, shape [B, N, D]
            Cached encoder tokens.

        Returns
        -------
        ModelOutput
            Typed predictions whose tensors have leading shape ``[B, Q]``.
        """
        if video.ndim != 5:
            raise ValueError(f"Expected video [B,T,C,H,W], got {video.shape}")

        if not isinstance(query, QueryBatch):
            query = QueryBatch.from_mapping(query)

        u = query.u.to(dtype=video.dtype)
        v = query.v.to(dtype=video.dtype)
        t_src = query.t_src.long()
        t_tgt = query.t_tgt.long()
        t_cam = query.t_cam.long()

        query_tokens = self.query_embedder(
            video=video,
            u=u,
            v=v,
            t_src=t_src,
            t_tgt=t_tgt,
            t_cam=t_cam,
        )
        decoded = self.decoder(query_tokens, memory)
        return ModelOutput.from_mapping(self.heads(decoded))

    def forward(self, batch: TrainingBatch) -> ModelOutput:
        """Predict all configured targets for a typed training batch."""
        video = batch.video
        if video.ndim != 5:
            raise ValueError(f"Expected video [B,T,C,H,W], got {video.shape}")
        memory = self.encode_video(video=video, aspect_ratio=batch.aspect_ratio)
        return self.decode_queries(video=video, query=batch.query, memory=memory)
