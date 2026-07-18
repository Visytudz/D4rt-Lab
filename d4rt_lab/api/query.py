"""Conveniences for constructing interactive point queries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Sequence

import numpy as np
import torch

IndexInput = int | Sequence[int] | np.ndarray | torch.Tensor
UVInput = Sequence[Sequence[float]] | np.ndarray | torch.Tensor


def build_query(
    uv: UVInput,
    t_src: IndexInput,
    t_tgt: IndexInput,
    t_cam: IndexInput,
    original_size: tuple[int, int],
    num_frames: int,
    device: torch.device,
    coordinates: str,
) -> dict[str, torch.Tensor]:
    """Build a single-video D4RT query dictionary.

    Parameters
    ----------
    uv : array-like, shape [Q, 2]
        Pixel or normalized query coordinates.
    t_src, t_tgt, t_cam : int or array-like, shape [Q]
        Source, target, and camera frame indices.
    original_size : tuple[int, int]
        Original video ``(height, width)``.
    num_frames : int
        Number of frames in the encoded clip.
    device : torch.device
        Destination query-tensor device.
    coordinates : {'pixels', 'normalized'}
        Coordinate system used by ``uv``.

    Returns
    -------
    dict[str, torch.Tensor]
        Query tensors with shape ``[1, Q]``.
    """
    uv_tensor = torch.as_tensor(uv, device=device, dtype=torch.float32)
    if uv_tensor.ndim == 1:
        uv_tensor = uv_tensor.unsqueeze(0)
    if uv_tensor.ndim != 2 or uv_tensor.shape[1] != 2:
        raise ValueError(f"Expected uv [Q,2], got {tuple(uv_tensor.shape)}")
    num_queries = int(uv_tensor.shape[0])
    if coordinates == "pixels":
        height, width = original_size
        uv_tensor = uv_tensor.clone()
        uv_tensor[:, 0] /= float(max(width - 1, 1))
        uv_tensor[:, 1] /= float(max(height - 1, 1))
    elif coordinates != "normalized":
        raise ValueError("coordinates must be 'pixels' or 'normalized'.")
    if torch.any((uv_tensor < 0) | (uv_tensor > 1)):
        raise ValueError("Query coordinates must lie inside the video frame.")

    def expand_indices(value: IndexInput, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=device, dtype=torch.long).reshape(-1)
        if tensor.numel() == 1:
            tensor = tensor.expand(num_queries)
        if tensor.numel() != num_queries:
            raise ValueError(f"{name} must contain 1 or {num_queries} values, got {tensor.numel()}")
        if torch.any((tensor < 0) | (tensor >= num_frames)):
            raise ValueError(f"{name} must be in [0, {num_frames - 1}].")
        return tensor.view(1, -1)

    return {
        "u": uv_tensor[:, 0].view(1, -1),
        "v": uv_tensor[:, 1].view(1, -1),
        "t_src": expand_indices(t_src, "t_src"),
        "t_tgt": expand_indices(t_tgt, "t_tgt"),
        "t_cam": expand_indices(t_cam, "t_cam"),
    }


def outputs_for_display(
    output: Mapping[str, torch.Tensor], original_size: tuple[int, int]
) -> dict[str, torch.Tensor]:
    """Convert raw model outputs to display-friendly CPU tensors.

    Parameters
    ----------
    output : Mapping[str, torch.Tensor]
        Model predictions with leading shape ``[1, Q]``.
    original_size : tuple[int, int]
        Original video ``(height, width)``.

    Returns
    -------
    dict[str, torch.Tensor]
        Squeezed tensors with leading shape ``[Q]`` or ``[Q, K]``, including
        ``uv_2d_pixels`` and ``visibility_probability``.
    """
    result = {name: value.squeeze(0).detach().cpu() for name, value in output.items()}
    uv_pixels = result["uv_2d"].clone()
    height, width = original_size
    uv_pixels[:, 0] *= float(max(width - 1, 1))
    uv_pixels[:, 1] *= float(max(height - 1, 1))
    result["uv_2d_pixels"] = uv_pixels
    result["visibility_probability"] = torch.sigmoid(result["visibility"])
    return result
