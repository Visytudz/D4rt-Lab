"""Public downstream interface for D4RT encoding and querying."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from d4rt_lab.core.types import ModelOutput, QueryBatch, TrainingBatch
from d4rt_lab.model import D4RTModel, D4RTModelConfig

from .artifact import model_from_artifact, resolve_device
from .query import IndexInput, UVInput, build_query, outputs_for_display
from .types import EncodedVideo
from .video import load_video_rgb, prepare_video_tensor


class D4RTAPI(nn.Module):
    """Expose D4RT model construction, video encoding, and point queries."""

    def __init__(self, model: D4RTModel, config: D4RTModelConfig) -> None:
        super().__init__()
        self.model = model
        self.config = config

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @classmethod
    def from_config(
        cls, config: D4RTModelConfig, device: str | torch.device = "cpu"
    ) -> "D4RTAPI":
        return cls(D4RTModel(config), config).to(resolve_device(device))

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cuda",
    ) -> "D4RTAPI":
        model, config = model_from_artifact(str(checkpoint_path), device)
        return cls(model, config)

    def forward(self, batch: TrainingBatch) -> ModelOutput:
        return self.model(batch)

    def encode_tensor(
        self, video: torch.Tensor, aspect_ratio: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode video ``[B, T, C, H, W]`` into memory ``[B, N, D]``."""
        return self.model.encode_video(video, aspect_ratio)

    def decode(
        self,
        video: torch.Tensor,
        query: QueryBatch | dict[str, torch.Tensor],
        memory: torch.Tensor,
    ) -> ModelOutput:
        """Decode point queries against cached video memory."""
        return self.model.decode_queries(video, query, memory)

    def encode(self, video_rgb: np.ndarray, fps: float = 10.0) -> EncodedVideo:
        """Preprocess and encode one RGB video ``[T, H, W, 3]``."""
        video_rgb = np.asarray(video_rgb)[: self.config.input.clip_frames]
        model_size = tuple(self.config.input.image_size)
        original_size = (int(video_rgb.shape[1]), int(video_rgb.shape[2]))
        video, aspect_ratio = prepare_video_tensor(video_rgb, model_size, self.device)
        memory = self.encode_tensor(video, aspect_ratio)
        return EncodedVideo(
            video, memory, aspect_ratio, original_size, model_size, float(fps)
        )

    def encode_path(self, path: str | Path) -> EncodedVideo:
        """Decode and encode one video file."""
        frames, fps = load_video_rgb(path, max_frames=self.config.input.clip_frames)
        return self.encode(frames, fps)

    def query(
        self,
        encoded: EncodedVideo,
        uv: UVInput,
        t_src: IndexInput,
        t_tgt: IndexInput,
        t_cam: IndexInput = 0,
        coordinates: str = "pixels",
    ) -> ModelOutput:
        """Query one encoded video without recomputing encoder memory."""
        query = build_query(
            uv,
            t_src,
            t_tgt,
            t_cam,
            encoded.original_size,
            encoded.num_frames,
            self.device,
            coordinates,
        )
        return self.decode(encoded.video, query, encoded.memory)

    @staticmethod
    def for_display(
        output: ModelOutput, original_size: tuple[int, int]
    ) -> dict[str, torch.Tensor]:
        return outputs_for_display(output, original_size)
