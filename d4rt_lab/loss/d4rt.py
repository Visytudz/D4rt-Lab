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

    def _xyz_preprocess(self, xyz: torch.Tensor, mask: torch.Tensor, normalize_depth: bool, transform_log: bool) -> torch.Tensor:
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

        m = mask.xyz_3d
        pred = outputs.xyz_3d
        gt = target.xyz_3d
        use_norm = cfg.normalize_by_mean_depth
        use_log = cfg.value_transform == "sign_x_log1p_abs_x"
        pred = self._xyz_preprocess(pred, m, use_norm, use_log)
        gt = self._xyz_preprocess(gt, m, use_norm, use_log)

        xyz_err_raw = (pred - gt).abs().mean(dim=-1)
        xyz_err = xyz_err_raw

        conf_cfg = self.loss_cfg.confidence
        conf_total = pred.new_zeros(())
        if conf_cfg.enabled:
            c = torch.sigmoid(outputs.confidence).clamp(1e-4, 1.0 - 1e-4)
            if conf_cfg.confidence_weights_xyz_error:
                xyz_err = xyz_err * c

            conf_penalty = masked_mean(-torch.log(c), m) * conf_cfg.weight
            conf_total = conf_total + conf_penalty
            metrics["loss_confidence_penalty"] = conf_penalty
            metrics["confidence_mean"] = masked_mean(c, m)

            mode = conf_cfg.mode.strip().lower()
            lconf_cfg = conf_cfg.lconf_ablation
            lconf_enabled = lconf_cfg.enabled or mode == "appendix_lconf"
            if mode not in {"main_text", "appendix_lconf"}:
                raise ValueError(f"Unsupported loss.confidence.mode: {mode}")
            if lconf_enabled:
                target_conf = torch.exp(-xyz_err_raw.detach()).clamp(1e-4, 1.0 - 1e-4)
                lconf_raw = (c - target_conf).abs()
                lconf_weight = lconf_cfg.weight
                lconf_loss = masked_mean(lconf_raw, m) * lconf_weight
                conf_total = conf_total + lconf_loss
                metrics["loss_confidence_lconf"] = lconf_loss
                metrics["confidence_target_mean"] = masked_mean(target_conf, m)

        xyz_loss = masked_mean(xyz_err, m) * cfg.weight
        metrics["loss_xyz_3d"] = xyz_loss
        return xyz_loss + conf_total

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
        uv_err = (outputs.uv_2d - target.uv_2d).abs().mean(dim=-1)
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
        bce = F.binary_cross_entropy_with_logits(outputs.visibility, target.visibility, reduction="none")
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
        total = total + self._compute_reprojection_uv_from_xyz_loss(outputs, batch, metrics)
        total = total + self._compute_vis_loss(outputs, target, mask, metrics)
        total = total + self._compute_disp_loss(outputs, target, mask, metrics)
        total = total + self._compute_normal_loss(outputs, target, mask, metrics)

        metrics["loss_total"] = total
        return total, metrics
