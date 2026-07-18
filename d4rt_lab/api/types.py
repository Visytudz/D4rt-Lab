"""Shared API data containers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EncodedVideo:
    """A preprocessed video and its cached encoder memory.

    Parameters
    ----------
    video : torch.Tensor, shape [1, T, C, H_model, W_model]
        Preprocessed model input.
    memory : torch.Tensor, shape [1, N, D]
        Cached encoder output. Its computation graph is preserved unless the
        caller used ``torch.inference_mode``.
    aspect_ratio : torch.Tensor, shape [1, 1]
        Original width-to-height ratio.
    original_size : tuple[int, int]
        Original ``(height, width)``.
    model_size : tuple[int, int]
        Model-space ``(height, width)``.
    fps : float
        Source video frame rate retained as metadata.
    """

    video: torch.Tensor
    memory: torch.Tensor
    aspect_ratio: torch.Tensor
    original_size: tuple[int, int]
    model_size: tuple[int, int]
    fps: float

    @property
    def num_frames(self) -> int:
        return int(self.video.shape[1])
