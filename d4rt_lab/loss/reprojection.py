"""Camera reprojection loss for D4RT."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .reduction import masked_mean
from d4rt_lab.core.types import TrainingBatch
from .config import ReprojectionLossConfig


def reprojection_loss(
    predicted_xyz: torch.Tensor,
    batch: TrainingBatch,
    cfg: ReprojectionLossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Project predicted 3D points into target cameras and measure UV error.

    Parameters
    ----------
    predicted_xyz : torch.Tensor, shape [B, Q, 3]
        Points expressed in the camera frame selected by ``query.t_cam``.
    batch : TrainingBatch
        Training batch containing video, query, camera, target, and mask data.
    cfg : ReprojectionLossConfig
        Reprojection-loss configuration.

    Returns
    -------
    torch.Tensor
        Scalar weighted reprojection loss.
    dict[str, torch.Tensor]
        Reprojection metrics.
    """
    zero = predicted_xyz.new_zeros(())
    if not cfg.enabled:
        return zero, {}

    camera = batch.camera
    query = batch.query
    video = batch.video
    required = (
        camera.K if camera else None,
        camera.T_wc if camera else None,
        camera.camera_valid if camera else None,
        query.t_cam, query.t_tgt, video,
    )
    if not all(torch.is_tensor(value) for value in required):
        return zero, {"loss_reprojection_uv_from_xyz": zero}

    k_seq, t_wc_seq, camera_valid, t_cam, t_tgt, video = required
    batch_size, _, _, image_h, image_w = video.shape
    batch_index = torch.arange(batch_size, device=predicted_xyz.device)[:, None]
    camera_index = t_cam.long().clamp_min(0)
    target_index = t_tgt.long().clamp_min(0)
    k_target = k_seq[batch_index, target_index]
    t_wc_camera = t_wc_seq[batch_index, camera_index]
    t_wc_target = t_wc_seq[batch_index, target_index]
    valid = camera_valid[batch_index, camera_index] & camera_valid[
        batch_index, target_index
    ]

    homogeneous = torch.cat(
        [predicted_xyz, torch.ones_like(predicted_xyz[..., :1])], dim=-1
    )
    world = torch.matmul(t_wc_camera, homogeneous.unsqueeze(-1)).squeeze(-1)
    try:
        target_xyz = torch.matmul(
            torch.linalg.inv(t_wc_target), world.unsqueeze(-1)
        ).squeeze(-1)[..., :3]
    except RuntimeError:
        return zero, {"loss_reprojection_uv_from_xyz": zero}

    depth = target_xyz[..., 2]
    projected = torch.matmul(k_target, target_xyz.unsqueeze(-1)).squeeze(-1)
    safe_depth = torch.where(depth.abs() > 1e-6, depth, torch.ones_like(depth))
    predicted_uv = torch.stack(
        [
            projected[..., 0] / safe_depth / float(max(image_w - 1, 1)),
            projected[..., 1] / safe_depth / float(max(image_h - 1, 1)),
        ],
        dim=-1,
    )
    valid &= torch.isfinite(target_xyz).all(-1)
    valid &= torch.isfinite(predicted_uv).all(-1) & (depth > 1e-6)
    valid &= batch.mask.xyz_3d & batch.mask.uv_2d

    if cfg.detach_xyz:
        predicted_uv = predicted_uv.detach()
    pixel_scale = predicted_uv.new_tensor([image_w - 1, image_h - 1])
    difference = (predicted_uv - batch.target.uv_2d) * pixel_scale
    difference = torch.where(valid.unsqueeze(-1), difference, torch.zeros_like(difference))

    robust = cfg.robust_loss.lower()
    if robust in {"huber", "smooth_l1"}:
        error = F.huber_loss(
            difference,
            torch.zeros_like(difference),
            reduction="none",
            delta=cfg.huber_delta_px,
        ).mean(-1)
    elif robust in {"l1", "mae"}:
        error = difference.abs().mean(-1)
    else:
        raise ValueError(f"Unsupported reprojection robust loss: {robust}")

    loss = masked_mean(error, valid) * cfg.weight
    metrics = {
        "loss_reprojection_uv_from_xyz": loss,
        "reprojection_uv_from_xyz_error_mean_px": masked_mean(
            difference.abs().mean(-1), valid
        ),
    }
    return loss, metrics
