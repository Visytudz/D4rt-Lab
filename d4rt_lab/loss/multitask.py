"""D4RT multi-task loss composition."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from d4rt_lab.loss.reduction import masked_mean, masked_mean_per_sample
from d4rt_lab.loss.reprojection import reprojection_loss
from d4rt_lab.core.types import ModelOutput, TargetBatch, TargetMask, TrainingBatch
from .config import D4RTLossConfig


class D4RTLoss(nn.Module):
    """Compute the weighted multi-task loss for D4RT predictions."""

    def __init__(self, loss_cfg: D4RTLossConfig) -> None:
        super().__init__()
        self.loss_cfg = loss_cfg

    def _xyz_preprocess(
        self,
        xyz: torch.Tensor,
        mask: torch.Tensor,
        normalize_depth: bool,
        transform_log: bool,
    ) -> torch.Tensor:
        out = xyz
        if normalize_depth:
            depth = out[..., 2].abs()
            scale = masked_mean_per_sample(depth, mask).clamp_min(1e-6).unsqueeze(-1)
            out = out / scale
        if transform_log:
            out = torch.sign(out) * torch.log1p(out.abs())
        return out

    def _compute_xyz_loss(
        self,
        outputs: ModelOutput,
        target: TargetBatch,
        mask: TargetMask,
        metrics: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        cfg = self.loss_cfg.xyz_3d
        if not cfg.enabled:
            return outputs.xyz_3d.new_zeros(())

        valid = mask.xyz_3d
        predicted_xyz = self._xyz_preprocess(
            outputs.xyz_3d,
            valid,
            cfg.normalize_by_mean_depth,
            cfg.use_signed_log,
        )
        target_xyz = self._xyz_preprocess(
            target.xyz_3d,
            valid,
            cfg.normalize_by_mean_depth,
            cfg.use_signed_log,
        )
        xyz_error = (predicted_xyz - target_xyz).abs().mean(dim=-1)
        xyz_error, confidence_loss = self._apply_confidence(
            xyz_error,
            outputs.confidence,
            valid,
            metrics,
        )

        xyz_loss = masked_mean(xyz_error, valid) * cfg.weight
        metrics["loss_xyz_3d"] = xyz_loss
        return xyz_loss + confidence_loss

    def _apply_confidence(
        self,
        xyz_error: torch.Tensor,
        confidence_logits: torch.Tensor,
        mask: torch.Tensor,
        metrics: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply confidence weighting and compute confidence supervision.

        Parameters
        ----------
        xyz_error : torch.Tensor, shape [B, Q]
            Per-query XYZ error before confidence weighting.
        confidence_logits : torch.Tensor, shape [B, Q]
            Raw confidence predictions.
        mask : torch.Tensor, shape [B, Q]
            Valid XYZ supervision mask.
        metrics : dict[str, torch.Tensor]
            Mutable loss metrics collected by :meth:`forward`.

        Returns
        -------
        weighted_xyz_error : torch.Tensor, shape [B, Q]
            XYZ error, optionally weighted by predicted confidence.
        confidence_loss : torch.Tensor, shape []
            Confidence penalty plus optional explicit-target supervision.
        """
        cfg = self.loss_cfg.confidence
        if not cfg.enabled:
            return xyz_error, xyz_error.new_zeros(())

        confidence = torch.sigmoid(confidence_logits).clamp(1e-4, 1.0 - 1e-4)
        weighted_xyz_error = (
            xyz_error * confidence if cfg.confidence_weights_xyz_error else xyz_error
        )
        confidence_loss = masked_mean(-torch.log(confidence), mask) * cfg.weight
        metrics["loss_confidence_penalty"] = confidence_loss
        metrics["confidence_mean"] = masked_mean(confidence, mask)

        target_cfg = cfg.confidence_target
        if target_cfg.enabled:
            target_confidence = torch.exp(-xyz_error.detach()).clamp(1e-4, 1.0 - 1e-4)
            target_error = (confidence - target_confidence).abs()
            target_loss = masked_mean(target_error, mask) * target_cfg.weight
            confidence_loss = confidence_loss + target_loss
            metrics["loss_confidence_target"] = target_loss
            metrics["confidence_target_mean"] = masked_mean(target_confidence, mask)

        return weighted_xyz_error, confidence_loss

    def _compute_uv_loss(
        self,
        outputs: ModelOutput,
        target: TargetBatch,
        mask: TargetMask,
        metrics: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        cfg = self.loss_cfg.uv_2d
        if not cfg.enabled:
            return outputs.uv_2d.new_zeros(())
        uv_err = (outputs.uv_2d - target.uv_2d).abs().mean(dim=-1)  # [B, Q]
        uv_loss = masked_mean(uv_err, mask.uv_2d) * cfg.weight
        metrics["loss_uv_2d"] = uv_loss
        return uv_loss

    def _compute_reprojection_uv_from_xyz_loss(
        self,
        outputs: ModelOutput,
        batch: TrainingBatch,
        metrics: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        loss, reprojection_metrics = reprojection_loss(
            predicted_xyz=outputs.xyz_3d,
            batch=batch,
            cfg=self.loss_cfg.reprojection_uv_from_xyz,
        )
        metrics.update(reprojection_metrics)
        return loss

    def _compute_vis_loss(
        self,
        outputs: ModelOutput,
        target: TargetBatch,
        mask: TargetMask,
        metrics: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        cfg = self.loss_cfg.visibility
        if not cfg.enabled:
            return outputs.visibility.new_zeros(())
        bce = F.binary_cross_entropy_with_logits(
            outputs.visibility, target.visibility, reduction="none"
        )  # [B, Q]
        vis_loss = masked_mean(bce, mask.visibility) * cfg.weight
        metrics["loss_visibility"] = vis_loss
        return vis_loss

    def _compute_disp_loss(
        self,
        outputs: ModelOutput,
        target: TargetBatch,
        mask: TargetMask,
        metrics: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        cfg = self.loss_cfg.displacement
        if not cfg.enabled:
            return outputs.xyz_3d.new_zeros(())
        disp_err = (outputs.displacement - target.displacement).abs().mean(dim=-1)
        disp_loss = masked_mean(disp_err, mask.displacement) * cfg.weight
        metrics["loss_displacement"] = disp_loss
        return disp_loss

    def _compute_normal_loss(
        self,
        outputs: ModelOutput,
        target: TargetBatch,
        mask: TargetMask,
        metrics: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        cfg = self.loss_cfg.normal
        if not cfg.enabled:
            return outputs.xyz_3d.new_zeros(())
        pred = F.normalize(outputs.normal, dim=-1)
        gt = F.normalize(target.normal, dim=-1)
        cos = F.cosine_similarity(pred, gt, dim=-1)
        norm_loss = masked_mean(1.0 - cos, mask.normal) * cfg.weight
        metrics["loss_normal"] = norm_loss
        return norm_loss

    def forward(
        self,
        outputs: ModelOutput,
        batch: TrainingBatch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target = batch.target
        mask = batch.mask
        metrics: dict[str, torch.Tensor] = {}

        total = outputs.xyz_3d.new_zeros(())
        total = total + self._compute_xyz_loss(outputs, target, mask, metrics)
        total = total + self._compute_uv_loss(outputs, target, mask, metrics)
        total = total + self._compute_reprojection_uv_from_xyz_loss(
            outputs, batch, metrics
        )
        total = total + self._compute_vis_loss(outputs, target, mask, metrics)
        total = total + self._compute_disp_loss(outputs, target, mask, metrics)
        total = total + self._compute_normal_loss(outputs, target, mask, metrics)

        metrics["loss_total"] = total
        return total, metrics
