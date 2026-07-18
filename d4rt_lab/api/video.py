"""Optional video I/O and preprocessing conveniences."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def load_video_rgb(path: str | Path, max_frames: int | None = None) -> tuple[np.ndarray, float]:
    """Decode RGB frames from a video file.

    Parameters
    ----------
    path : str or pathlib.Path
        Input video path.
    max_frames : int or None
        Maximum number of leading frames to decode.

    Returns
    -------
    video_rgb : numpy.ndarray, shape [T, H, W, 3], dtype uint8
        Decoded RGB frames.
    fps : float
        Reported frame rate, or ``10.0`` when unavailable.
    """
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 10.0
    frames: list[np.ndarray] = []
    try:
        while max_frames is None or len(frames) < int(max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {path}")
    return np.stack(frames), fps


def prepare_video_tensor(
    video_rgb: np.ndarray,
    model_size: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize an RGB video and convert it to model tensors.

    Parameters
    ----------
    video_rgb : numpy.ndarray, shape [T, H, W, 3], dtype uint8
        RGB input frames.
    model_size : tuple[int, int]
        Target ``(height, width)``.
    device : torch.device
        Destination device.

    Returns
    -------
    video : torch.Tensor, shape [1, T, 3, H_model, W_model], dtype float32
        Video normalized to ``[0, 1]``.
    aspect_ratio : torch.Tensor, shape [1, 1], dtype float32
        Original width-to-height ratio.
    """
    video_rgb = np.asarray(video_rgb)
    if video_rgb.ndim != 4 or video_rgb.shape[-1] != 3:
        raise ValueError(f"Expected video [T,H,W,3], got {video_rgb.shape}")
    if video_rgb.shape[0] < 2:
        raise ValueError("D4RT requires at least two video frames.")
    model_h, model_w = model_size
    original_h, original_w = int(video_rgb.shape[1]), int(video_rgb.shape[2])
    resized = np.stack(
        [cv2.resize(frame, (model_w, model_h), interpolation=cv2.INTER_AREA) for frame in video_rgb]
    )
    video = (
        torch.from_numpy(np.ascontiguousarray(resized))
        .to(device=device, dtype=torch.float32)
        .permute(0, 3, 1, 2)
        .unsqueeze(0)
        / 255.0
    )
    aspect_ratio = torch.tensor(
        [[float(original_w) / float(max(original_h, 1))]], device=device, dtype=video.dtype
    )
    return video, aspect_ratio
