"""Query embeddings for D4RT decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatures(nn.Module):
    """Fourier feature mapping for normalized UV coordinates."""

    def __init__(
        self, input_dim: int = 2, num_bands: int = 8, include_input: bool = True
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_bands = num_bands
        self.include_input = include_input
        self.register_buffer(
            "frequencies",
            2.0 ** torch.arange(num_bands, dtype=torch.float32) * torch.pi,
            persistent=False,
        )

    @property
    def output_dim(self) -> int:
        base = self.input_dim if self.include_input else 0
        return base + self.input_dim * self.num_bands * 2

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        # uv: [B, Q, D], normalized to [0, 1], D=input_dim
        uv = uv.to(dtype=torch.float32)
        x = uv.unsqueeze(-1) * self.frequencies.view(
            1, 1, 1, -1
        )  # [B, Q, D, num_bands]
        sin = torch.sin(x)
        cos = torch.cos(x)
        out = [sin, cos]
        if self.include_input:
            out.insert(0, uv.unsqueeze(-1))  # [B, Q, D, 1]
        merged = torch.cat(out, dim=-1)  # [B, Q, D, (1+)num_bands*2]
        return merged.flatten(start_dim=2)  # [B, Q, D*((1+)num_bands*2)]


class QueryEmbedder(nn.Module):
    """Build query tokens from coordinates, time indices, and local RGB.

    Parameters
    ----------
    hidden_dim : int
        Output token dimension.
    clip_frames : int
        Maximum number of addressable video frames.
    local_patch_enabled : bool
        Whether to sample an RGB patch around each source pixel.
    local_patch_size : int
        Width and height of the sampled source-frame patch.
    uv_num_bands : int
        Number of Fourier frequency bands used for UV coordinates.
    """

    def __init__(
        self,
        hidden_dim: int,
        clip_frames: int,
        local_patch_enabled: bool,
        local_patch_size: int = 9,
        uv_num_bands: int = 8,
    ) -> None:
        super().__init__()
        # configuration
        self.hidden_dim = hidden_dim
        self.max_frames = clip_frames
        self.local_patch_enabled = local_patch_enabled
        self.local_patch_size = local_patch_size

        # components
        self.uv_encoder = FourierFeatures(
            input_dim=2, num_bands=uv_num_bands, include_input=True
        )
        self.uv_proj = nn.Linear(self.uv_encoder.output_dim, hidden_dim)
        self.t_src_embed = nn.Embedding(clip_frames, hidden_dim)
        self.t_tgt_embed = nn.Embedding(clip_frames, hidden_dim)
        self.t_cam_embed = nn.Embedding(clip_frames, hidden_dim)
        if local_patch_enabled:
            patch_dim = 3 * local_patch_size * local_patch_size
            self.patch_proj = nn.Sequential(
                nn.Linear(patch_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.patch_proj = None
        self.out_norm = nn.LayerNorm(hidden_dim)

    def _clamp_t(self, t: torch.Tensor) -> torch.Tensor:
        return t.clamp(min=0, max=self.max_frames - 1)

    def _sample_local_rgb_patches(
        self,
        video: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        t_src: torch.Tensor,
    ) -> torch.Tensor:
        """Sample a subpixel RGB patch around every source query.

        Parameters
        ----------
        video : torch.Tensor, shape [B, T, C, H, W], dtype floating
            Preprocessed video tensor.
        u, v : torch.Tensor, shape [B, Q], dtype floating
            Normalized query centers in ``[0, 1]``.
        t_src : torch.Tensor, shape [B, Q], dtype torch.long
            Source-frame index for every query.

        Returns
        -------
        torch.Tensor, shape [B, Q, C * P * P]
            Flattened bilinearly sampled RGB patches, where ``P`` is
            ``local_patch_size``.
        """
        batch_size, _, channels, height, width = video.shape
        _, num_queries = u.shape
        patch_size = self.local_patch_size

        # 1. Select the source frame associated with each query.
        batch_indices = torch.arange(batch_size, device=video.device)[:, None]  # [B, 1]
        batch_indices = batch_indices.expand(batch_size, num_queries)  # [B, Q]
        query_source_frames = video[batch_indices, t_src]  # [B, Q, C, H, W]
        query_source_frames = query_source_frames.reshape(
            batch_size * num_queries, channels, height, width
        )  # [B*Q, C, H, W]

        # 2. Express integer pixel offsets in grid_sample's [-1, 1] space.
        pixel_offsets = (
            torch.arange(patch_size, device=video.device, dtype=torch.float32)
            - (patch_size - 1) / 2.0
        )
        x_offsets = pixel_offsets * (2.0 / max(width - 1, 1))
        y_offsets = pixel_offsets * (2.0 / max(height - 1, 1))
        offset_y, offset_x = torch.meshgrid(y_offsets, x_offsets, indexing="ij") # [P, P]
        patch_offset_grid = torch.stack([offset_x, offset_y], dim=-1)  # [P, P, 2]

        # 3. Translate the relative patch grid to every subpixel query center.
        query_centers = torch.stack([u * 2.0 - 1.0, v * 2.0 - 1.0], dim=-1).reshape(
            batch_size * num_queries, 1, 1, 2
        )  # [B*Q, 1, 1, 2]
        sampling_grid = query_centers + patch_offset_grid.unsqueeze(0)
        # [B*Q, P, P, 2]

        # 4. Bilinear interpolation makes non-integer query centers valid.
        sampled_patches = F.grid_sample(
            query_source_frames,  # [B*Q, C, H, W]
            sampling_grid,  # [B*Q, P, P, 2]
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # [B*Q, C, P, P]
        return sampled_patches.reshape(
            batch_size, num_queries, channels * patch_size * patch_size
        )  # [B, Q, C * P * P]

    def forward(
        self,
        video: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        t_src: torch.Tensor,
        t_tgt: torch.Tensor,
        t_cam: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of spatio-temporal queries.

        Parameters
        ----------
        video : torch.Tensor, shape [B, T, C, H, W]
            Preprocessed floating-point video tensor.
        u, v : torch.Tensor, shape [B, Q], dtype floating
            Normalized pixel coordinates.
        t_src, t_tgt, t_cam : torch.Tensor, shape [B, Q], dtype torch.long
            Source, target, and camera frame indices.

        Returns
        -------
        torch.Tensor, shape [B, Q, D]
            Query-token representations.
        """
        # prepare uv token
        uv = torch.stack([u, v], dim=-1)  # [B, Q, 2]
        uv_token = self.uv_proj(self.uv_encoder(uv))  # [B, Q, D], D=hidden_dim

        # prepare time token
        t_src = self._clamp_t(t_src)  # [B, Q]
        t_tgt = self._clamp_t(t_tgt)  # [B, Q]
        t_cam = self._clamp_t(t_cam)  # [B, Q]
        time_token = (
            self.t_src_embed(t_src) + self.t_tgt_embed(t_tgt) + self.t_cam_embed(t_cam)
        )  # [B, Q, D]

        # combine tokens
        token = uv_token + time_token  # [B, Q, D]
        if self.patch_proj is not None:
            patches = self._sample_local_rgb_patches(video, u, v, t_src)
            token = token + self.patch_proj(patches)

        return self.out_norm(token)
